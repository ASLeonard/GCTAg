/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * Streaming MLMA (--mlma-stream) – streams SNPs via Geno::loopDouble instead
 * of pre-loading the full genotype matrix into RAM, using a REML state that
 * is either loaded from disk or computed inline from a GRM.
 *
 * REML state source (exactly one is required):
 *   --load-reml <file>   load a previously saved REML state and run the
 *                         streaming association test.
 *   --grm <prefix>        run REML inline against the given GRM. Combine
 *                         with --save-reml to only fit REML and write the
 *                         state to "<out>.reml" (no association test is run).
 *   --reml-woodbury-reuse <file>
 *                         like --grm, but no GRM is read: the Woodbury basis
 *                         (Uk, dk, lambda_tail, tail_d_var) is taken from a Woodbury (TUNA)
 *                         .reml saved earlier, and only the variance components
 *                         and fixed effects are re-fitted for the current
 *                         --pheno/--covar. The .reml must come from the same
 *                         analysis samples in the same order: n and a hash of
 *                         the ordered sample IDs are checked from the first
 *                         28 bytes of the file, before anything else is read
 *                         (--load-reml does the same). tail_d_var is stored in
 *                         the file, so the fit matches a from-scratch run.
 *                         .reml files from older builds (format v1) are
 *                         rejected. Combine with --save-reml to write a new
 *                         .reml for this phenotype.
 *
 * --save-reml is a boolean flag (no argument) valid only alongside --grm or
 * --reml-woodbury-reuse.
 */


#include "MLMA_stream_common.hpp"
#include "Covar.h"
#include "main/StatFunc.h"
#include "mlma_woodbury.hpp"
#include "symmetric_eigendecomp.hpp"
#include "chunked_grm_matvec.hpp"
#include "RemlState.hpp"
#include "RemlCtx.hpp"
#include "RemlEngine.hpp"
#include "grm_binary_io.hpp"

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cctype>
#include <array>
#include <charconv>
#include <limits>
#include <cstdio>
#include <cstring>
#include <cerrno>

using std::map;
using std::string;
using std::to_string;
using std::vector;
using std::function;

// ---------------------------------------------------------------------------
// Static member definitions
// ---------------------------------------------------------------------------
map<string, string>  MLMA::options;
map<string, double>  MLMA::options_d;
map<string, vector<double>> MLMA::options_vd;
vector<string>       MLMA::processFunctions;

// ---------------------------------------------------------------------------
// Internal types (anonymous namespace)
// ---------------------------------------------------------------------------
namespace {

// Shared GRM I/O helpers from include/grm_binary_io.hpp.
using gcta_grm_io::read_grm_binary;
using gcta_grm_io::match_ids_to_grm;

template <typename T>
string to_text(T x)
{
    std::array<char, 128> buf{};
    auto [ptr, ec] = std::to_chars(buf.data(), buf.data() + buf.size(),
                                   x, std::chars_format::general,
                                   std::numeric_limits<T>::max_digits10);
    if (ec == std::errc())
        return string(buf.data(), static_cast<size_t>(ptr - buf.data()));
    return to_string(x);
}

// RemlState is now defined in include/RemlState.hpp (shared with MLMA_loco).
// The readRemlState() function below remains file-local (only MLMA_stream needs it).

constexpr size_t kMaxChunk = 1u << 30; // 1 GiB per syscall, safely under INT_MAX, which is a hard constraint on e.g. macOS.
static void read_exact(int fd, void* dst, size_t nbytes, const std::string& path)
{
    char* ptr = static_cast<char*>(dst);
    size_t total_read = 0;
    while (total_read < nbytes) {
        size_t chunk = std::min(kMaxChunk, nbytes - total_read);
        const ssize_t res = ::read(fd, ptr + total_read, chunk);
        if (res == 0) {
            close(fd);
            LOGGER.e(0, "Unexpected EOF while reading [" + path + "] (expected " +
                        to_string(nbytes) + " bytes, got " + to_string(total_read) + ").");
        }
        if (res < 0) {
            if (errno == EINTR) continue;
            close(fd);
            LOGGER.e(0, "I/O error while reading [" + path + "] (errno=" +
                        std::to_string(errno) + ").");
        }
        total_read += static_cast<size_t>(res);
    }
}

// ---------------------------------------------------------------------------
// .reml on-disk format (v2). Every file starts with a fixed 28-byte prefix
//     [Header: magic[4], n, x_c, num_varcmp, num_r_indx][uint64 id_hash]
// followed, depending on the magic, by
//   "TUNA" (Woodbury):
//     [int32 k][double lambda_tail][double tail_d_var]
//     [Uk: k x n float][dk: k float][b: x_c float, omitted if no_adj_covar]
//     [varcmp: num_varcmp float]
//   "GOBY" (dense):
//     [int32 factor_kind][packed upper-triangular Vi_L: n(n+1)/2 float]
//     [b: x_c float, omitted if no_adj_covar][varcmp: num_varcmp float]
// id_hash fingerprints the ordered analysis IDs ("FID\tIID"), i.e. the samples
// the n x n GRM -- and hence the Woodbury basis / V -- was built for, not the
// full .grm.id list. It sits in the prefix so a file made for other samples (or
// the same samples in another order) is refused after reading 28 bytes, before
// any bulk read. tail_d_var is stored so that a --reml-woodbury-reuse fit
// reproduces the from-scratch logL and HE start.
// ---------------------------------------------------------------------------
constexpr std::string_view kMagicWoodbury = "TUNA";
constexpr std::string_view kMagicDense    = "GOBY";
constexpr size_t kRemlPrefixBytes = sizeof(Header) + sizeof(uint64_t);

// FNV-1a (64-bit) over the ordered IDs, each terminated by '\n'. Deterministic
// across builds/platforms (unlike std::hash). Never returns 0.
uint64_t hash_sample_ids(const vector<string>& ids)
{
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t h = 1469598103934665603ULL;
    for (const string& s : ids) {
        for (const unsigned char c : s) { h ^= c; h *= kPrime; }
        h ^= static_cast<unsigned char>('\n');
        h *= kPrime;
    }
    return h != 0 ? h : 1ULL;
}

string hash_hex(uint64_t h)
{
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return string(buf);
}

struct RemlPrefix {
    Header   hdr{};
    uint64_t id_hash  = 0;
    bool     woodbury = false;   // true: TUNA, false: GOBY
};

// Read and validate the fixed prefix of a .reml from `fd` (positioned at 0).
// Closes fd on any error.
RemlPrefix read_reml_prefix(int fd, const std::string& filename)
{
    RemlPrefix p;
    read_exact(fd, &p.hdr, sizeof(Header), filename);
    read_exact(fd, &p.id_hash, sizeof(uint64_t), filename);

    const std::string_view magic(p.hdr.magic, 4);
    if (magic == kMagicWoodbury) {
        p.woodbury = true;
    } else if (magic != kMagicDense) {
        close(fd);
        if (magic != "TUNA" && magic != "GOBY")
            LOGGER.e(0, "[" + filename + "] was written by an older build (format v1: no sample-ID hash) "
                        "and can no longer be read; re-create it with --save-reml.");
        LOGGER.e(0, "[" + filename + "] is not a REML state file (unrecognised magic).");
    }
    return p;
}

// Refuse a .reml made for different samples, using only its prefix.
void check_reml_samples(int fd, const RemlPrefix& p, int expected_n, uint64_t expected_id_hash,
                        const std::string& filename)
{
    if (p.hdr.n != expected_n) {
        close(fd);
        LOGGER.e(0, "Sample size mismatch: [" + filename + "] has n=" + to_string(p.hdr.n) +
                    " vs dataset n=" + to_string(expected_n) +
                    ". Use the same filters (--keep/--remove/--pheno) as the run that saved it.");
    }
    if (p.id_hash != expected_id_hash) {
        close(fd);
        LOGGER.e(0, "Sample ID mismatch: the ordered analysis IDs (hash " + hash_hex(expected_id_hash) +
                    ") differ from those [" + filename + "] was saved with (hash " + hash_hex(p.id_hash) +
                    "). Use the same samples in the same order (--keep/--remove and the row order of "
                    "--pheno/--covar) as the run that saved it.");
    }
}

// Read the binary REML state written by writeRemlStateFromCtx().
// When !no_adj_covar the 'b' vector is loaded; otherwise it is skipped.
// The prefix is checked first (format, n, sample-ID hash) so a file made for
// other samples is refused before the bulk read.
// The file is read serially in one contiguous pass to avoid HPC/Lustre I/O
// storms from many threads touching scattered offsets concurrently; only the
// in-memory unpack/decode is parallelized afterwards.
RemlState readRemlState(const std::string& filename, bool no_adj_covar,
                        int expected_n, uint64_t expected_id_hash)
{
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) LOGGER.e(0, "Cannot open file [" + filename + "].");

    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        close(fd);
        LOGGER.e(0, "Failed to stat file [" + filename + "].");
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);
    if (file_size < kRemlPrefixBytes) {
        close(fd);
        LOGGER.e(0, "[" + filename + "] is empty or too short to be a REML state.");
    }

    const RemlPrefix prefix = read_reml_prefix(fd, filename);
    check_reml_samples(fd, prefix, expected_n, expected_id_hash, filename);

    std::vector<char> file_buf(file_size);
    std::memcpy(file_buf.data(), &prefix.hdr, sizeof(Header));
    std::memcpy(file_buf.data() + sizeof(Header), &prefix.id_hash, sizeof(uint64_t));
    read_exact(fd, file_buf.data() + kRemlPrefixBytes, file_size - kRemlPrefixBytes, filename);
    close(fd);

    const char* mapped = file_buf.data();
    size_t offset = kRemlPrefixBytes;
    auto read_bytes = [&](void* dst, size_t nbytes) {
        if (offset + nbytes > file_size) {
            LOGGER.e(0, "Unexpected EOF in [" + filename + "].");
        }
        std::memcpy(dst, mapped + offset, nbytes);
        offset += nbytes;
    };

    // --- Header (already read and validated above) ---
    const Header& hdr = prefix.hdr;

    if (hdr.n <= 0 || hdr.x_c < 0 || hdr.num_varcmp <= 0) {
        LOGGER.e(0, "[" + filename + "] has invalid header dimensions.");
    }

    const std::string_view magic(hdr.magic, 4);
    RemlState st;
    st.n   = hdr.n;
    st.x_c = hdr.x_c;

    // ------------------------------------------------------------------ GOBY
    if (magic == kMagicDense) {
        int32_t factor_kind = 0;
        read_bytes(&factor_kind, sizeof(int32_t));
        st.is_llt = (factor_kind == 0);

        st.Vi_L_f.resize(hdr.n, hdr.n);

        const size_t tri = static_cast<size_t>(hdr.n) * (hdr.n + 1) / 2;
        std::vector<float> packed_buf(tri);
        read_bytes(packed_buf.data(), tri * sizeof(float));

        // Pre-compute starting indices per column to avoid serial loop dependencies.
        // The file bytes are already resident in memory; the parallel work here is only
        // the unpack/decode step, not the disk-facing read path. Mirrors
        // writeRemlStateFromCtx's packing: column j holds head(j+1) (rows 0..j),
        // growing with j -- st.Vi_L_f is upper-triangle-valid on return, matching
        // what run_mlma_stream_association's STRSV/STRSM/STRMM calls expect
        // (CblasUpper / triangularView<Eigen::Upper>()).
        std::vector<size_t> col_offsets(hdr.n);
        size_t current_idx = 0;
        for (int32_t j = 0; j < hdr.n; ++j) {
            col_offsets[j] = current_idx;
            current_idx += static_cast<size_t>(j + 1);
        }

        #pragma omp parallel for schedule(static)
        for (int32_t j = 0; j < hdr.n; ++j) {
            const int32_t len = j + 1;
            std::memcpy(st.Vi_L_f.col(j).head(len).data(),
                        packed_buf.data() + col_offsets[j],
                        static_cast<size_t>(len) * sizeof(float));
        }

        if (!no_adj_covar) {
            st.b.resize(hdr.x_c);
            read_bytes(st.b.data(), static_cast<size_t>(hdr.x_c) * sizeof(float));
        }

        st.varcmp.resize(hdr.num_varcmp);
        read_bytes(st.varcmp.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float));

        return st;
    }

    // ------------------------------------------------------------------ TUNA
    if (magic == kMagicWoodbury) {
        st.is_woodbury = true;
        int32_t k = 0;
        read_bytes(&k, sizeof(int32_t));
        if (k <= 0) {
            LOGGER.e(0, "[" + filename + "] has invalid Woodbury rank k=" + to_string(k) + ".");
        }

        double lambda_tail = 0.0;
        read_bytes(&lambda_tail, sizeof(double));
        st.lambda_tail_f = static_cast<float>(lambda_tail);

        // tail_d_var is stored for --reml-woodbury-reuse; association does not use it.
        double tail_d_var = 0.0;
        read_bytes(&tail_d_var, sizeof(double));
        (void)tail_d_var;

        Eigen::MatrixXf Uk(k, hdr.n);
        {
            const size_t uk_elems = static_cast<size_t>(k) * hdr.n;
            std::vector<float> uk_buf(uk_elems);
            read_bytes(uk_buf.data(), uk_elems * sizeof(float));
            #pragma omp parallel for schedule(static)
            for (int32_t col = 0; col < hdr.n; ++col) {
                std::memcpy(Uk.data() + static_cast<size_t>(col) * k,
                            uk_buf.data() + static_cast<size_t>(col) * k,
                            static_cast<size_t>(k) * sizeof(float));
            }
        }

        Eigen::VectorXf dk(k);
        read_bytes(dk.data(), static_cast<size_t>(k) * sizeof(float));
        st.dk_f = dk;

        if (!no_adj_covar) {
            st.b.resize(hdr.x_c);
            read_bytes(st.b.data(), static_cast<size_t>(hdr.x_c) * sizeof(float));
        }

        Eigen::VectorXf vc(hdr.num_varcmp);
        read_bytes(vc.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float));
        st.varcmp = vc;

        const double sg2   = static_cast<double>(vc[0]);
        const double se2   = static_cast<double>(vc[hdr.num_varcmp - 1]);
        const double sig2e = sg2 * lambda_tail + se2;

        st.wb.Uk_f.swap(Uk);
        st.wb.ck_f.resize(k);
        for (int32_t j = 0; j < k; ++j) {
            const double delta = std::max(0.0, static_cast<double>(dk[j]) - lambda_tail);
            const double sd    = sg2 * delta;
            st.wb.ck_f[j] = static_cast<float>(sd / (sig2e + sd));
        }
        st.wb.sqrt_ck_f    = st.wb.ck_f.cwiseSqrt();
        st.wb.sigma2_eff_f = static_cast<float>(sig2e);

        return st;
    }

    LOGGER.e(0, "[" + filename + "] unsupported format.");
    return st;
}

// Writes the v2 format described above. id_hash = hash_sample_ids(analysis IDs).
void writeRemlStateFromCtx(const std::string& filename, RemlCtx& ctx, bool no_adj_covar,
                           uint64_t id_hash)
{
    // Use POSIX low-level I/O for direct control over OS write buffering
    int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
        LOGGER.e(0, "cannot open [" + filename + "] for writing.");

    auto write_bytes = [fd, &filename](const void* src, size_t nbytes) {
        const char* ptr = static_cast<const char*>(src);
        size_t total_written = 0;
        while (total_written < nbytes) {
            size_t chunk = std::min(kMaxChunk, nbytes - total_written);
            ssize_t res = write(fd, ptr + total_written, chunk);
            if (res < 0) {
                if (errno == EINTR) continue; // interrupted, retry
                close(fd);
                LOGGER.e(0, "write error on [" + filename + "]: " + std::string(strerror(errno)));
            }
            if (res == 0) {
                close(fd);
                LOGGER.e(0, "write error on [" + filename + "]: write() returned 0 unexpectedly.");
            }
            total_written += static_cast<size_t>(res);
        }
    };

    if (ctx.Vi_use_woodbury_basis) {
        // --- TUNA Branch Optimization ---
        struct Header {
            char    magic[4] = {'T', 'U', 'N', 'A'};
            int32_t n = 0, x_c = 0, num_varcmp = 0, num_r_indx = 0;
        } hdr;
        hdr.n = static_cast<int32_t>(ctx.n);
        hdr.x_c = static_cast<int32_t>(ctx.X_c);
        hdr.num_varcmp = static_cast<int32_t>(ctx.varcmp.size());
        hdr.num_r_indx = static_cast<int32_t>(ctx.varcmp.size());
        write_bytes(&hdr, sizeof(hdr));
        write_bytes(&id_hash, sizeof(uint64_t));   // completes the 28-byte prefix

        const int32_t k = static_cast<int32_t>(ctx.dk.size());
        write_bytes(&k, sizeof(int32_t));

        const double lambda_tail = ctx.lambda_tail;
        write_bytes(&lambda_tail, sizeof(double));

        const double tail_d_var = ctx.tail_d_var;
        write_bytes(&tail_d_var, sizeof(double));

        if (ctx.Uk.rows() != ctx.n || ctx.Uk.cols() != k)
            LOGGER.e(0, "invalid Woodbury REML context dimensions before save.");

        // Uk_f must be written k×n (see mlma_woodbury.hpp layout convention),
        // transposed relative to ctx.Uk's n×k storage -- readers place these
        // bytes directly into a k×n matrix with no further transpose.
        // Parallel manual transpose+cast (not ctx.Uk.transpose().cast<float>(),
        // which Eigen evaluates single-threaded) so large Uk blocks use all cores.
        Eigen::MatrixXf Uk_f(k, ctx.n);
        #pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < ctx.n; ++i) {
            for (int32_t j = 0; j < k; ++j)
                Uk_f(j, i) = static_cast<float>(ctx.Uk(i, j));
        }
        write_bytes(Uk_f.data(), static_cast<size_t>(ctx.n) * k * sizeof(float));

        Eigen::VectorXf dk_f = ctx.dk.cast<float>();
        write_bytes(dk_f.data(), static_cast<size_t>(k) * sizeof(float));

        if (!no_adj_covar) {
            Eigen::VectorXf b_f = ctx.b.cast<float>();
            write_bytes(b_f.data(), static_cast<size_t>(ctx.X_c) * sizeof(float));
        }

        Eigen::VectorXf varcmp_f = Eigen::Map<const Eigen::VectorXd>(ctx.varcmp.data(),
                                                                      static_cast<Eigen::Index>(ctx.varcmp.size())).cast<float>();
        write_bytes(varcmp_f.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float));

    } else {
        // --- GOBY Branch Optimization ---
        struct Header {
            char    magic[4] = {'G', 'O', 'B', 'Y'};
            int32_t n = 0, x_c = 0, num_varcmp = 0, num_r_indx = 0;
        } hdr;
        hdr.n = static_cast<int32_t>(ctx.n);
        hdr.x_c = static_cast<int32_t>(ctx.X_c);
        hdr.num_varcmp = static_cast<int32_t>(ctx.varcmp.size());
        hdr.num_r_indx = static_cast<int32_t>(ctx.varcmp.size());
        write_bytes(&hdr, sizeof(hdr));
        write_bytes(&id_hash, sizeof(uint64_t));   // completes the 28-byte prefix

        const int32_t factor_kind = ctx.Vi_use_llt ? 0 : 1;
        write_bytes(&factor_kind, sizeof(int32_t));

        // Prepare factorization matrix reference
        const Eigen::MatrixXd* mat_ptr = &ctx.Vi_L;
        Eigen::MatrixXd Vi_copy;

        if (!ctx.Vi_use_llt) {
            if (ctx.Vi.rows() != ctx.n || ctx.Vi.cols() != ctx.n)
                LOGGER.e(0, "invalid dense REML context dimensions before save.");
            // Copy to preserve original ctx.Vi state
            Vi_copy = ctx.Vi;
            gcta_blas_int blas_n = static_cast<gcta_blas_int>(ctx.n);
            if (gcta_dpotrf(blas_n, Vi_copy.data(), blas_n, 'U') != 0)
                LOGGER.e(0, "Vi is not positive definite when factorising for save.");
            mat_ptr = &Vi_copy;
        }

        const Eigen::MatrixXd& mat = *mat_ptr;
        const size_t n_val = static_cast<size_t>(ctx.n);
        const size_t tri_elements = n_val * (n_val + 1) / 2;

        // Allocate packed single-precision output buffer once
        std::vector<float> packed_buf(tri_elements);

        // Pre-compute offsets per column to allow multi-threaded parallel packing.
        // mat is upper-triangle-valid (a Cholesky factor from dpotrf('U'), either
        // ctx.Vi_L directly or the freshly-factorised Vi_copy above), so column j
        // packs its own head(j+1) (rows 0..j) -- growing with j, the mirror image
        // of the pre-upper-triangle-swap layout, which packed tail(n-j) instead.
        std::vector<size_t> col_offsets(n_val);
        size_t current_idx = 0;
        for (size_t j = 0; j < n_val; ++j) {
            col_offsets[j] = current_idx;
            current_idx += (j + 1);
        }

        // Parallel float conversion & packing: zero lock contention across CPU cores
        #pragma omp parallel for schedule(static)
        for (int32_t j = 0; j < ctx.n; ++j) {
            const int32_t len = j + 1;
            const double* col_src = mat.col(j).head(len).data();
            float* dst = packed_buf.data() + col_offsets[j];

            for (int32_t i = 0; i < len; ++i) {
                dst[i] = static_cast<float>(col_src[i]);
            }
        }

        // Single contiguous bulk write (utilizes full disk IO bus speed)
        write_bytes(packed_buf.data(), tri_elements * sizeof(float));

        if (!no_adj_covar) {
            Eigen::VectorXf b_f = ctx.b.cast<float>();
            write_bytes(b_f.data(), static_cast<size_t>(ctx.X_c) * sizeof(float));
        }

        Eigen::VectorXf varcmp_f = Eigen::Map<const Eigen::VectorXd>(ctx.varcmp.data(),
                                                                      static_cast<Eigen::Index>(ctx.varcmp.size())).cast<float>();
        write_bytes(varcmp_f.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float));
    }

    close(fd);
}

// --reml-woodbury-reuse: load the phenotype-independent part of a Woodbury
// (TUNA) .reml -- Uk, dk, lambda_tail, tail_d_var -- straight into ctx,
// replacing the GRM read and compute_woodbury_basis(). Only the leading bytes
// are read:
//   [Header][uint64 id_hash][int32 k][double lambda_tail][double tail_d_var]
//   [Uk: k x n float][dk: k float]
// A file made for other samples (n or ordered-ID hash) is refused after the
// first 28 bytes. The trailing b / varcmp belong to the phenotype the file was
// fitted on and are never touched, so whether the file was written with or
// without b does not matter here.
void load_woodbury_basis_into_ctx(const string& filename, RemlCtx& ctx, uint64_t expected_id_hash)
{
    const int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) LOGGER.e(0, "Cannot open file [" + filename + "].");

    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        close(fd);
        LOGGER.e(0, "Failed to stat file [" + filename + "].");
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);

    const RemlPrefix prefix = read_reml_prefix(fd, filename);
    if (!prefix.woodbury) {
        close(fd);
        LOGGER.e(0, "[" + filename + "] is a dense (GOBY) REML state; --reml-woodbury-reuse needs a "
                    "Woodbury (TUNA) state saved from a --reml-woodbury-basis fit.");
    }
    check_reml_samples(fd, prefix, ctx.n, expected_id_hash, filename);

    int32_t k           = 0;
    double  lambda_tail = 0.0;
    double  tail_d_var  = 0.0;
    read_exact(fd, &k, sizeof(int32_t), filename);
    read_exact(fd, &lambda_tail, sizeof(double), filename);
    read_exact(fd, &tail_d_var, sizeof(double), filename);
    if (k <= 0 || k >= ctx.n) {
        close(fd);
        LOGGER.e(0, "[" + filename + "] has invalid Woodbury rank k=" + to_string(k) + ".");
    }
    if (!std::isfinite(lambda_tail) || !std::isfinite(tail_d_var) || tail_d_var < 0.0) {
        close(fd);
        LOGGER.e(0, "[" + filename + "] has invalid Woodbury tail statistics.");
    }

    const size_t uk_bytes = static_cast<size_t>(k) * static_cast<size_t>(ctx.n) * sizeof(float);
    const size_t dk_bytes = static_cast<size_t>(k) * sizeof(float);
    const size_t need = kRemlPrefixBytes + sizeof(int32_t) + 2 * sizeof(double) + uk_bytes + dk_bytes;
    if (file_size < need) {
        close(fd);
        LOGGER.e(0, "[" + filename + "] is too short for n=" + to_string(ctx.n) +
                    ", k=" + to_string(k) + " (expected at least " + to_string(need) +
                    " bytes, found " + to_string(file_size) + ").");
    }

    // Uk is stored k x n (column i = sample i, see writeRemlStateFromCtx).
    Eigen::MatrixXf Uk_f(k, ctx.n);
    read_exact(fd, Uk_f.data(), uk_bytes, filename);
    Eigen::VectorXf dk_f(k);
    read_exact(fd, dk_f.data(), dk_bytes, filename);
    close(fd);

    // k x n float -> n x k double; parallel over samples (contiguous reads).
    ctx.Uk.resize(ctx.n, k);
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < ctx.n; ++i)
        for (int32_t j = 0; j < k; ++j)
            ctx.Uk(i, j) = static_cast<double>(Uk_f(j, i));
    Uk_f.resize(0, 0);

    ctx.dk                       = dk_f.cast<double>();
    ctx.lambda_tail              = lambda_tail;
    ctx.tail_d_var               = tail_d_var;
    ctx.woodbury_basis_rank_     = k;
    ctx.woodbury_basis_rank      = k;
    ctx.Vi_use_woodbury_basis    = true;
    ctx.woodbury_basis_preloaded = true;

    LOGGER.i(0, "Woodbury basis loaded from [" + filename + "]: n=" + to_string(ctx.n) +
                ", k=" + to_string(k) + ", lambda_tail=" + to_text(lambda_tail) +
                ", tail_d_var=" + to_text(tail_d_var) + ".");
}

void write_hsq_from_ctx(const string& out_prefix, const RemlCtx& ctx)
{
    const string hsq_file = out_prefix + ".hsq";
    std::ofstream o_hsq(hsq_file);
    if (!o_hsq) LOGGER.e(0, "cannot open [" + hsq_file + "] for writing.");

    o_hsq << "Source\tVariance\tSE\n";
    const int m = static_cast<int>(ctx.varcmp.size());
    for (int i = 0; i < m; ++i) {
        const string name = (i < static_cast<int>(ctx.var_name.size()))
            ? ctx.var_name[i]
            : (i + 1 == m ? "V(e)" : ("V(G" + to_string(i + 1) + ")"));
        const double se = (i < static_cast<int>(ctx.varcmp_se.size())) ? ctx.varcmp_se[i] : 0.0;
        o_hsq << name << "\t" << to_text(ctx.varcmp[i]) << "\t" << to_text(se) << "\n";
    }

    o_hsq << "Vp\t" << to_text(ctx.Vp) << "\t" << to_text(ctx.Vp_se) << "\n";

    const int ngen = static_cast<int>(ctx.hsq.size());
    for (int i = 0; i < ngen; ++i) {
        const string hname = (i < static_cast<int>(ctx.hsq_name.size()))
            ? ctx.hsq_name[i]
            : ("V(G" + to_string(i + 1) + ")/Vp");
        const double hse = (i < static_cast<int>(ctx.hsq_se.size())) ? ctx.hsq_se[i] : 0.0;
        o_hsq << hname << "\t" << to_text(ctx.hsq[i]) << "\t" << to_text(hse) << "\n";
    }

    if (ctx.has_logL)
        o_hsq << "logL\t" << to_text(ctx.logL) << "\n";

    if (ctx.reml_iterations > 0)
        o_hsq << "reml_iterations\t" << ctx.reml_iterations << "\n";

    o_hsq << "n\t" << ctx.n << "\n";
    o_hsq.close();
    LOGGER.i(0, "Summary REML results saved to [" + hsq_file + "].");
}

// Load a "<prefix>.eigenvec" written by --pca-approx (rSVD or Lanczos) and
// align its rows to `analysis_ids` (the same FID\tIID order used to build
// ctx.y / ctx.X / ctx.A[0]), for use as ctx.Uk before reml::compute() so
// compute_woodbury_basis_basis()'s existing has_warm path seeds its rSVD sketch
// from it instead of a cold Gaussian draw.
//
// A silently misaligned warm start would seed the sketch with the wrong
// individual's eigenvector — worse than no warm start at all — so any
// mismatch between the file's sample set and the current analysis sample
// set is a hard error, not a fallback.
Eigen::MatrixXd load_pca_warm_start(const string& eigenvec_prefix,
                                     const vector<string>& analysis_ids)
{
    const string path = eigenvec_prefix + ".eigenvec";
    std::ifstream in(path);
    if (!in) LOGGER.e(0, "--reml-woodbury-basis-warm-start: cannot open [" + path + "].");

    std::unordered_map<string, vector<double>> by_id;
    by_id.reserve(analysis_ids.size() * 2);

    string line, fid, iid;
    int k = -1;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        if (!(ss >> fid >> iid))
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: malformed line in [" + path + "].");
        vector<double> pcs;
        double v;
        while (ss >> v) pcs.push_back(v);
        if (k < 0) k = static_cast<int>(pcs.size());
        else if (static_cast<int>(pcs.size()) != k)
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: inconsistent eigenvector count in [" + path + "].");
        by_id.emplace(fid + "\t" + iid, std::move(pcs));
    }
    if (k <= 0)
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: no eigenvectors read from [" + path + "].");

    const int n = static_cast<int>(analysis_ids.size());
    Eigen::MatrixXd Uk(n, k);
    for (int i = 0; i < n; ++i) {
        const auto it = by_id.find(analysis_ids[i]);
        if (it == by_id.end())
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: individual [" + analysis_ids[i] +
                        "] not found in [" + path + "]. The warm-start file must come from "
                        "the same sample set (e.g. --pca-approx run on the same --grm).");
        for (int j = 0; j < k; ++j) Uk(i, j) = it->second[j];
    }

    LOGGER.i(0, "Woodbury warm-start: loaded " + to_string(k) + " eigenvector(s) for " +
                to_string(n) + " individuals from [" + path + "].");
    return Uk;
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// registerOption
// ---------------------------------------------------------------------------
int MLMA::registerOption(map<string, vector<string>>& options_in)
{
    // Always capture --out (shared flag)
    // Use the pre-processed "out" key (no dashes), which is always set by main.cpp
    if (!options_in["out"].empty())
        options["out"] = options_in["out"][0];

    const bool has_mlma_stream = options_in.find("--mlma-stream") != options_in.end();
    const bool has_load_reml   = options_in.find("--load-reml")   != options_in.end()
                                  && !options_in["--load-reml"].empty();
    // --save-reml is a boolean flag (no argument): the REML state is always
    // written to "<out>.reml". Any value accidentally supplied is ignored.
    const bool has_save_reml   = options_in.find("--save-reml")   != options_in.end();
    const bool has_reuse       = options_in.find("--reml-woodbury-reuse") != options_in.end();

    auto capture_common_reml_flags = [&]() {
        if (options_in.find("--reml-alg") != options_in.end()
                && !options_in["--reml-alg"].empty()) {
            options_d["reml_alg"] = std::stod(options_in["--reml-alg"][0]);
            options_in.erase("--reml-alg");
        }
        if (options_in.find("--reml-no-constrain") != options_in.end()) {
            options["no_constrain"] = "1";
            options_in.erase("--reml-no-constrain");
        }
        if (options_in.find("--reml-diagV-adj") != options_in.end()
                && !options_in["--reml-diagV-adj"].empty()) {
            options_d["reml_diagV_adj"] = std::stod(options_in["--reml-diagV-adj"][0]);
            options_in.erase("--reml-diagV-adj");
        }
        if (options_in.find("--seed") != options_in.end()
             && !options_in["--seed"].empty()) {
            options_d["seed"] = std::stod(options_in["--seed"][0]);
            options_in.erase("--seed");
        }
    };

    if (!has_mlma_stream)
        return 0;

    if (has_load_reml && has_save_reml)
        LOGGER.e(0, "--mlma-stream does not allow --load-reml with --save-reml.");
    if (has_reuse && has_load_reml)
        LOGGER.e(0, "--reml-woodbury-reuse re-fits REML for the current phenotype and cannot be "
                    "combined with --load-reml (which uses a fitted state as-is).");
    if (has_reuse && (options_in["--reml-woodbury-reuse"].empty()
                      || options_in["--reml-woodbury-reuse"][0].empty()))
        LOGGER.e(0, "--reml-woodbury-reuse requires a .reml file argument.");

    if (has_save_reml) {
        if (!options_in["--save-reml"].empty())
            LOGGER.w(0, "--save-reml takes no argument in --mlma-stream; the REML state "
                        "will be saved to [" + options.at("out") + ".reml" + "].");
        options["save_reml"] = "1";
        options_in.erase("--save-reml");
    }

    capture_common_reml_flags();

    if (has_load_reml) {
        // Pre-built REML state: load from file.
        options["load_reml"] = options_in["--load-reml"][0];
        options_in.erase("--load-reml");
        // REML tuning flags are irrelevant when a saved state is loaded; consume to suppress warnings.
        options_in.erase("--reml-woodbury-basis");
        options_in.erase("--svd-method");
        options_in.erase("--reml-trace-hutchpp");
        options_in.erase("--reml-trace-hutchpp-fixed-probes");
        options_in.erase("--reml-maxit");
        options_in.erase("--reml-priors");
        options_in.erase("--reml-priors-var");
        options_in.erase("--reml-no-HE-start");
        options_in.erase("--reml-woodbury-basis-EIG-mass");
        options_in.erase("--reml-woodbury-basis-VAR-tail");
        options_in.erase("--reml-woodbury-basis-range");
        options_in.erase("--grm-chunked-budget");
        options_in.erase("--reml-ai-robust");
        options_in.erase("--reml-ai-robust-tol");
        options_in.erase("--reml-ai-robust-risk");
        options_in.erase("--reml-force-dense-V");
        options_in.erase("--reml-print-trajectory");
    } else {
        // Inline REML path: --grm is required, unless the Woodbury basis comes
        // from a saved .reml (--reml-woodbury-reuse), in which case no GRM is read.
        const bool has_grm = options_in.find("--grm") != options_in.end()
                              && !options_in["--grm"].empty();
        if (has_reuse) {
            if (has_grm)
                LOGGER.e(0, "--reml-woodbury-reuse takes the basis from the .reml file and does not read a GRM; remove --grm.");
            if (options_in.find("--reml-woodbury-basis") != options_in.end())
                LOGGER.e(0, "--reml-woodbury-reuse cannot be combined with --reml-woodbury-basis (the rank comes from the .reml file).");
            if (options_in.find("--reml-woodbury-basis-posthoc-correction") != options_in.end())
                LOGGER.e(0, "--reml-woodbury-reuse is incompatible with --reml-woodbury-basis-posthoc-correction (no exact GRM is loaded).");
            options["woodbury_reuse"] = options_in["--reml-woodbury-reuse"][0];
            options_in.erase("--reml-woodbury-reuse");
        } else {
            if (!has_grm)
                LOGGER.e(0, "--mlma-stream requires either --load-reml <file>, --reml-woodbury-reuse <file> or --grm <prefix>.");
            options["grm"] = options_in["--grm"][0];
            options_in.erase("--grm");
        }

        // --reml-woodbury [k]  (k optional; -1 = MP-k, -2 = EIG, -3 = variance)
        if (options_in.find("--reml-woodbury-basis") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis"];
            if (!vals.empty() && !vals[0].empty()) {
                if (vals[0] == "EIG")
                    options_d["woodbury_basis_rank"] = -2.0;  // Eigenvalue mass k
                else if (vals[0] == "VAR")
                    options_d["woodbury_basis_rank"] = -3.0;  // Variance-k
                else if (vals[0] == "MP")
                    options_d["woodbury_basis_rank"] = -1.0;  // MP-k
                else
                    options_d["woodbury_basis_rank"] = std::stod(vals[0]);
            } else
                options_d["woodbury_basis_rank"] = -1.0;   // MP-k
            options_in.erase("--reml-woodbury-basis");
        }
        if (options_in.find("--reml-woodbury-basis-EIG-mass") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis-EIG-mass"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--reml-woodbury-basis-EIG-mass requires a fractional argument (e.g. 0.15).");
            options_d["woodbury_basis_eigen_mass"] = std::stof(vals[0]);
            options_in.erase("--reml-woodbury-basis-EIG-mass");
        }
        if (options_in.find("--reml-woodbury-basis-VAR-tail") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis-VAR-tail"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--reml-woodbury-basis-VAR-tail requires a relative Frobenius error argument (e.g. 0.001).");
            options_d["woodbury_basis_var_tail"] = std::stod(vals[0]);
            options_in.erase("--reml-woodbury-basis-VAR-tail");
        }
        // --svd-method <power|nystrom>: select the Woodbury basis sketch.
        // The default power path uses subspace iteration; Nystrom is single-pass.
        if (options_in.find("--svd-method") != options_in.end()) {
            const auto& vals = options_in["--svd-method"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--svd-method requires one argument: power or nystrom.");
            string method = vals[0];
            std::transform(method.begin(), method.end(), method.begin(),
                           [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (method == "nystrom") {
                options["svd_nystrom"] = "1";
            } else if (method != "power") {
                LOGGER.e(0, "Unsupported --svd-method value [" + vals[0] + "]. Allowed values: power, nystrom.");
            }
            if (vals.size() > 1)
                LOGGER.w(0, "--svd-method expects exactly one value; using the first one.");
            options_in.erase("--svd-method");
        }
        if (options_in.find("--svd-power-iter") != options_in.end()) {
            const auto& vals = options_in["--svd-power-iter"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--svd-power-iter requires one integer argument.");
            options["svd_power_iter"] = vals[0];
            if (vals.size() > 1)
                LOGGER.w(0, "--svd-power-iter expects exactly one value; using the first one.");
            options_in.erase("--svd-power-iter");
        }
        // --reml-woodbury-basis-warm-start <prefix>: seed the Woodbury rSVD sketch
        // with eigenvectors from a prior "<prefix>.eigenvec" (e.g. written by
        // --pca-approx rSVD on the same GRM/sample set), instead of a cold
        // Gaussian draw. Ignored when --svd-method nystrom is set, since
        // the Nystrom path doesn't use power iteration.
        if (options_in.find("--reml-woodbury-basis-warm-start") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis-warm-start"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--reml-woodbury-basis-warm-start requires a file prefix argument.");
            options["woodbury_basis_warm_start"] = vals[0];
            options_in.erase("--reml-woodbury-basis-warm-start");
        }
        if (options_in.find("--reml-no-HE-start") != options_in.end()) {
            if (!options_in["--reml-no-HE-start"].empty())
                LOGGER.w(0, "--reml-no-HE-start takes no argument; ignoring the supplied value.");
            options["no_HE_start"] = false;
            options_in.erase("--reml-no-HE-start");
        }
        if (options_in.find("--reml-force-dense-V") != options_in.end()) {
            if (!options_in["--reml-force-dense-V"].empty())
                LOGGER.w(0, "--reml-force-dense-V takes no argument; ignoring the supplied value.");
            options["force_dense_V"] = "1";
            options_in.erase("--reml-force-dense-V");
        }
        // --grm-chunked-budget: read K in lower-triangular tiles for the
        // Woodbury basis rSVD instead of holding a dense n x n K resident —
        // see chunked_grm_matvec.hpp. Requires the caller (below, once
        // grm_binary_io.hpp's reader is wired up) to populate
        // ctx.grm_tile_reader; RemlEngine enforces this at runtime.
        if (options_in.find("--grm-chunked-budget") != options_in.end()) {
            const auto& vals = options_in["--grm-chunked-budget"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--grm-chunked-budget requires a target memory in GB.");
            options_d["grm_chunked_budget"] = std::stod(vals[0]);
            options_in.erase("--grm-chunked-budget");
        }
        if (options_in.find("--reml-woodbury-basis-MP-margin") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis-MP-margin"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--reml-woodbury-basis-MP-margin requires a fractional argument (e.g. 0.15).");
            options_d["woodbury_basis_edge_margin"] = std::stod(vals[0]);
            options_in.erase("--reml-woodbury-basis-MP-margin");
        }
        if (options_in.find("--reml-woodbury-basis-MP-confirm") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis-MP-confirm"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--reml-woodbury-basis-MP-confirm requires an integer argument.");
            options_d["woodbury_basis_edge_confirm"] = std::stod(vals[0]);
            options_in.erase("--reml-woodbury-basis-MP-confirm");
        }
        if (options_in.find("--reml-woodbury-basis-EIG-k-buffer") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis-EIG-k-buffer"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--reml-woodbury-basis-EIG-k-buffer requires an integer argument.");
            options_d["woodbury_basis_EIG_k_buffer"] = std::stod(vals[0]);
            options_in.erase("--reml-woodbury-basis-EIG-k-buffer");
        }
        // --reml-woodbury-basis-mem-budget <GB>: hard cap on k_svd derived from an
        // approximate rSVD sketch memory budget, independent of n-1. See the
        // svd_mem_budget_gb comment in RemlCtx.hpp — chunking K does
        // nothing to bound this, since the sketch buffers scale with k_ext
        // regardless of whether K itself is chunked or dense.
        if (options_in.find("--reml-woodbury-basis-mem-budget") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis-mem-budget"];
            if (vals.empty() || vals[0].empty())
                LOGGER.e(0, "--reml-woodbury-basis-mem-budget requires a GB argument (e.g. 32).");
            options_d["woodbury_basis_mem_budget_gb"] = std::stod(vals[0]);
            options_in.erase("--reml-woodbury-basis-mem-budget");
        }
        if (options_in.find("--reml-woodbury-basis-range") != options_in.end()) {
            const auto& vals = options_in["--reml-woodbury-basis-range"];
            if (vals.size() < 2)
                LOGGER.e(0, "--reml-woodbury-basis-range requires two integer arguments.");
            options_d["woodbury_basis_range_init"] = std::stod(vals[0]);
            options_d["woodbury_basis_range_max"] = std::stod(vals[1]);
            options_in.erase("--reml-woodbury-basis-range");
        }
        if (options_in.find("--reml-woodbury-basis-posthoc-correction") != options_in.end()) {
            options["woodbury_basis_posthoc_correction"] = "1";
            options_in.erase("--reml-woodbury-basis-posthoc-correction");
            LOGGER.w(0, "--reml-woodbury-basis-posthoc-correction is an experimental feature and likely to worsen results.");
        }
        if (options_in.find("--reml-trace-hutchpp") != options_in.end()) {
            options["trace_hutchpp"] = "1";
            const auto& v = options_in["--reml-trace-hutchpp"];
            if (!v.empty() && !v[0].empty())
                options_d["trace_hutchpp_nprobes"] = std::stod(v[0]);
            options_in.erase("--reml-trace-hutchpp");
        }
        if (options_in.find("--reml-trace-hutchpp-fixed-probes") != options_in.end()) {
            options["trace_hutchpp_fixed_probes"] = "1";
            options_in.erase("--reml-trace-hutchpp-fixed-probes");
        }
        if (options_in.find("--reml-maxit") != options_in.end()
                && !options_in["--reml-maxit"].empty()) {
            options_d["reml_maxit"] = std::stod(options_in["--reml-maxit"][0]);
            options_in.erase("--reml-maxit");
        }
        if (options_in.find("--reml-priors-var") != options_in.end()) {
            options["reml_priors_var"] = "1";
            const auto& vals = options_in["--reml-priors-var"];
            for (const auto& s : vals) {
                if (!s.empty())
                    options_vd["reml_priors_var"].push_back(std::stod(s));
            }
            options_in.erase("--reml-priors-var");
        }
        if (options_in.find("--reml-priors") != options_in.end()) {
            const auto& vals = options_in["--reml-priors"];
            for (const auto& s : vals) {
                if (!s.empty())
                    options_vd["reml_priors"].push_back(std::stod(s));
            }
            options_in.erase("--reml-priors");
        }
        if (options_in.find("--reml-ai-robust") != options_in.end()) {
            options["reml_ai_robust"] = "1";
        }
        if (options_in.find("--reml-ai-robust-tol") != options_in.end()
                && !options_in["--reml-ai-robust-tol"].empty()) {
            options_d["reml_ai_robust_tol"] = std::stod(options_in["--reml-ai-robust-tol"][0]);
        }
        if (options_in.find("--reml-ai-robust-risk") != options_in.end()
                && !options_in["--reml-ai-robust-risk"].empty()) {
            options_d["reml_ai_robust_risk"] = std::stod(options_in["--reml-ai-robust-risk"][0]);
        }   
        if (options_in.find("--reml-print-trajectory") != options_in.end()) {
            options["reml_print_trajectory"] = "1";
            options_in.erase("--reml-print-trajectory");
        }
    }

    if (options_in.find("--mlma-no-preadj-covar") != options_in.end()) {
        options["no_adj_covar"] = "1";
        options_in.erase("--mlma-no-preadj-covar");
    }
    if (options_in.find("--mlma-stream") != options_in.end()) {
        const auto& vals = options_in["--mlma-stream"];
        if (!vals.empty()) {
            if (vals.size()==1)
                options_d["mlma_tile_budget_gb"] = {std::stod(vals[0])};
            else
                LOGGER.e(0, "--mlma-stream accepts at most one value: the memory budget in GB for tile-based streaming.");
        }
    }
    options_in.erase("--mlma-stream");
    if (options_in.find("--log-pval") != options_in.end()) {
        options["log_pval"] = "1";
        options_in.erase("--log-pval");
    }
    if (options_in.find("--model") != options_in.end()) {
        const auto& vals = options_in["--model"];
        if (vals.size() != 1 || vals[0].empty())
            LOGGER.e(0, "--model expects exactly one value: additive, nonadditive, or dominance.");

        string model = vals[0];
        std::transform(model.begin(), model.end(), model.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (model != "additive" && model != "nonadditive" && model != "dominance") {
            LOGGER.e(0, "Unsupported --model value [" + vals[0] + "]. Allowed values: additive, nonadditive, dominance.");
        }
        options["model"] = model;
        options_in.erase("--model");
    }

    processFunctions.push_back("MLMA");
    return 1;
}


// ---------------------------------------------------------------------------
// processMain
// ---------------------------------------------------------------------------
void MLMA::processMain()
{
    for (const auto& pf : processFunctions) {
        if (pf != "MLMA") continue;

        const string out_prefix     = options.at("out");
        const bool   no_adj_covar   = options.count("no_adj_covar") > 0;
        const bool   log_pval       = options.count("log_pval") > 0;
        // --save-reml only fits/saves the REML state and never streams SNPs,
        // so the bim (Marker) and genotype (Geno) data are never touched.
        const bool   reml_only      = options.count("save_reml") > 0;

        if (no_adj_covar)
            LOGGER.e(0, "--mlma-no-preadj-covar is not yet supported in --MLMA. "
                        "Re-run without this flag (pre-adjustment is the default).");

        // only if requested; otherwise leave it seeded from entropy (the default).
        if (options_d.count("seed")) {
            const auto seed = static_cast<std::mt19937::result_type>(options_d.at("seed"));
            gcta_eigh::shared_rng().seed(seed);
            LOGGER.i(0, "Using random seed: " + to_string(seed) + ".");
        }

        if (options.count("reml_ai_robust") && options.count("trace_hutchpp") > 0 &&options.count("trace_hutchpp_fixed_probes") == 0)
            LOGGER.w(0, "Using fresh Hutch++ probes without robust AI convergence can lead to unstable REML convergence. Consider using --reml-ai-robust or --reml-trace-hutchpp-fixed-probes.");

        // ---- Pheno / Marker / Geno (Geno re-reads pheno state in loopDouble) ----
        // Pheno alone reads the .fam (sample IDs); Marker/Geno additionally need
        // the .bim/.bed and are skipped entirely in --save-reml-only mode.
        Pheno*  pheno  = new Pheno();
        Marker* marker = nullptr;
        Geno*   geno   = nullptr;
        if (!reml_only) {
            marker = new Marker();
            geno   = new Geno(pheno, marker);
            const string model_name = options.count("model") ? options.at("model") : "additive";
            geno->setGenoCodingModel(model_name);
            LOGGER.i(0, "MLMA_stream genotype model: " + model_name + ".");
        }

        // ---- Phenotype ----
        const uint32_t n_init = pheno->count_keep();
        vector<string> ids;
        vector<double> phenos_vec;
        pheno->get_pheno(ids, phenos_vec);
        if (ids.size() != n_init)
            LOGGER.e(0, "--MLMA requires --pheno to be specified.");

        // ---- Covariates ----
        Covar covar;
        bool  has_covar = false;
        int   x_c       = 1;
        Eigen::MatrixXd X_design;

        {
            vector<uint32_t> remain_index, covar_index;
            has_covar = covar.getCommonSampleIndex(ids, remain_index, covar_index);

            if (has_covar) {
                LOGGER.i(0, to_string(remain_index.size()) +
                            " overlapping individuals with non-missing covariate data.");
                pheno->filter_keep_index(remain_index);
            } else if (covar.hasCovar()) {
                LOGGER.e(0, "no overlapping individuals with non-missing covariate data.");
            } else {
                // No covariate files specified – keep all phenotype samples
                remain_index.resize(ids.size());
                std::iota(remain_index.begin(), remain_index.end(), 0u);
            }

            // Compact phenotype values to the kept sample order.
            phenos_vec = compact_sample_vector(phenos_vec, remain_index);
        }

        const int n = static_cast<int>(pheno->count_keep());
        LOGGER.i(0, "After matching all files, " + to_string(n) +
                    " individuals to be included in --MLMA.");

        // Build design matrix X  (n × x_c, column-major)
        if (has_covar) {
            vector<string> final_ids = pheno->get_id(0, n - 1, "\t");
            vector<double> covar_X;
            vector<uint32_t> ck_dummy;
            covar.getCovarX(final_ids, covar_X, ck_dummy);
            const int covar_cols = (n > 0) ? static_cast<int>(covar_X.size()) / n : 0;
            x_c = 1 + covar_cols;
            X_design.resize(n, x_c);
            X_design.col(0).setOnes();
            if (covar_cols > 0)
                X_design.rightCols(covar_cols) =
                    Eigen::Map<const Eigen::MatrixXd>(covar_X.data(), n, covar_cols);
        } else {
            X_design.resize(n, 1);
            X_design.setOnes();
        }

        // ---- Obtain REML state (either from file or inline) ----
        RemlState state;
        bool use_inline_ctx = false;
        Eigen::VectorXf y_vec = Eigen::Map<const Eigen::VectorXd>(phenos_vec.data(), n).cast<float>();
        //for (int i = 0; i < n; ++i) y_vec[i] = static_cast<float>(phenos_vec[i]);

        // Streaming SNP-block width, sized from a memory budget rather than
        // the previous fixed BLOCK=10000 (which allocated the same
        // GenoBufItem::geno + X_block footprint regardless of n, reaching
        // multi-GB at large n). Opt-in, matching --GRM-tile-budget /
        // --grm-chunked-budget elsewhere: omitting the flag reproduces the
        // old fixed-10000 behavior exactly (see resolve_mlma_block_size).
        const double mlma_tile_budget_gb = options_d.count("mlma_tile_budget_gb")
            ? options_d.at("mlma_tile_budget_gb") : 0.0;

        if (options.count("load_reml")) {
            // --load-reml uses the serialized RemlState path: the state is read
            // from disk and then consumed by run_mlma_stream_association(RemlState&)
            // below. This is the saved-state code path and uses the float-first
            // Woodbury/L factor data in RemlState (see RemlState.hpp and the
            // RemlState overload in MLMA_stream_common.hpp).
            const string load_reml_file = options.at("load_reml");
            LOGGER.i(0, "Loading REML state from [" + load_reml_file + "]...");
            LOGGER.ts("load_reml");
            state = readRemlState(load_reml_file, no_adj_covar, n,
                                  hash_sample_ids(pheno->get_id(0, n - 1, "\t")));
            LOGGER.i(0, "REML state loaded in " + to_string(LOGGER.tp("load_reml")) + " seconds.");

            if (state.n != n)
                LOGGER.e(0, "Sample size mismatch: REML state n=" + to_string(state.n) +
                            " vs dataset n=" + to_string(n) +
                            ". Use the same filters (--keep/--remove/--pheno) in both runs.");
            if (state.x_c != x_c)
                LOGGER.e(0, "Covariate count mismatch: REML state x_c=" + to_string(state.x_c) +
                            " vs current x_c=" + to_string(x_c) +
                            ". Use the same --qcovar/--covar as during --save-reml.");

            // y_adj = y - X*b
            {
                const Eigen::MatrixXf Xf = X_design.cast<float>();
                y_vec -= Xf * state.b;
            }
        } else {
            // --grm takes the inline REML path: we build a RemlCtx from the GRM,
            // run reml::compute() to populate its V^{-1} / Woodbury state, and then
            // pass that RemlCtx through run_mlma_stream_association(RemlCtx&) below.
            // This is distinct from the --load-reml serialized RemlState path above.
            const bool   reuse_basis = options.count("woodbury_reuse") > 0;
            const string grm_pfx     = reuse_basis ? string() : options.at("grm");
            if (reuse_basis)
                LOGGER.i(0, "Running inline REML with the Woodbury basis reused from [" +
                            options.at("woodbury_reuse") + "] (no GRM is read) ...");
            else
                LOGGER.i(0, "Running inline REML using GRM [" + grm_pfx + "] ...");

            const vector<string> analysis_ids = pheno->get_id(0, n - 1, "\t");
            const uint64_t id_hash = hash_sample_ids(analysis_ids);
            bool grm_chunked = !reuse_basis && options_d.count("grm_chunked_budget") > 0.0;
            const double grm_chunked_budget = options_d.count("grm_chunked_budget")
                ? options_d.at("grm_chunked_budget") : 0.0;

            // REML tuning parameters. Parsed here (ahead of the GRM load below,
            // not after it as before) because woodbury_basis_mem_budget_gb feeds
            // the chunk-size solve immediately below, which needs to run before
            // committing to a chunked vs. dense load. Also lets the
            // reml_alg/svd_nystrom/etc. validation checks fail fast, ahead of
            // the (potentially expensive) GRM read, instead of after it.
            const int  woodbury_basis_rank  = options_d.count("woodbury_basis_rank")
                ? static_cast<int>(options_d.at("woodbury_basis_rank")) : 0;
            const bool trace_hutchpp   = options.count("trace_hutchpp") > 0;
            const int  trace_hutchpp_nprobes  = options_d.count("trace_hutchpp_nprobes")
                ? static_cast<int>(options_d.at("trace_hutchpp_nprobes")) : 100;
            const int  reml_maxit     = options_d.count("reml_maxit")
                ? static_cast<int>(options_d.at("reml_maxit")) : 100;
            const int  reml_alg       = options_d.count("reml_alg")
                ? static_cast<int>(options_d.at("reml_alg")) : 0;
            const int  reml_diagV_adj = options_d.count("reml_diagV_adj")
                ? static_cast<int>(options_d.at("reml_diagV_adj")) : 0;
            const bool no_constrain   = options.count("no_constrain") > 0;
            const bool svd_nystrom = options.count("svd_nystrom") > 0;
            const int svd_power_iter = options_d.count("svd_power_iter")
                ? static_cast<int>(options_d.at("svd_power_iter")) : 3;
            const float  woodbury_basis_eigen_mass  = options_d.count("woodbury_basis_eigen_mass")
                ? static_cast<float>(options_d.at("woodbury_basis_eigen_mass")) : 0.99f;
            const double woodbury_basis_edge_margin = options_d.count("woodbury_basis_edge_margin")
                ? options_d.at("woodbury_basis_edge_margin") : 0.15;
            const int    woodbury_basis_edge_confirm = options_d.count("woodbury_basis_edge_confirm")
                ? static_cast<int>(options_d.at("woodbury_basis_edge_confirm")) : 20;
            const int woodbury_basis_EIG_k_buffer = options_d.count("woodbury_basis_EIG_k_buffer")
                ? static_cast<int>(options_d.at("woodbury_basis_EIG_k_buffer")) : 0;
            const double woodbury_basis_mem_budget_gb = options_d.count("woodbury_basis_mem_budget_gb")
                ? options_d.at("woodbury_basis_mem_budget_gb") : 0.0;
            const bool no_HE_start = options.count("no_HE_start") > 0;
            const bool reml_ai_robust = options.count("reml_ai_robust") > 0;
            const double reml_ai_robust_tol = options_d.count("reml_ai_robust_tol")
                ? options_d.at("reml_ai_robust_tol") : 1e-4;
            const double reml_ai_robust_risk = options_d.count("reml_ai_robust_risk")
                ? options_d.at("reml_ai_robust_risk") : 0.01;

            if (reml_alg < 0 || reml_alg > 2)
                LOGGER.e(0, "--reml-alg should be 0, 1 or 2.");
            if (reml_diagV_adj < 0 || reml_diagV_adj > 2)
                LOGGER.e(0, "--reml-diagV-adj should be 0, 1, or 2.");
            if ((woodbury_basis_rank != 0 || reuse_basis) && reml_alg == 1)
                LOGGER.e(0, "--reml-woodbury is incompatible with Fisher-scoring REML (--reml-alg 1). Use AI-REML (default) or EM-REML (--reml-alg 2).");
            if (svd_nystrom && woodbury_basis_rank == 0)
                LOGGER.e(0, "--svd-method nystrom requires --reml-woodbury <k|MP|EIG|VAR>.");
            if ((woodbury_basis_rank != 0 || reuse_basis) && trace_hutchpp)
                LOGGER.w(0, "--reml-woodbury-basis/--reml-woodbury-reuse and --reml-trace-hutchpp both given; "
                            "the Woodbury basis provides an exact tr(PA) and takes precedence — "
                            "--reml-trace-hutchpp is ignored.");

            int grm_chunk_rows = 0;
            vector<string> grm_ids;
            Eigen::MatrixXd G_n;      // left empty when grm_chunked
            double m_all = options_d["woodbury_basis_rank"] == -1 ? 0.0 : -1.0;  // only populated when dense GRM and Woodbury-MP mode is selected
            gcta_grm_io::ChunkedGrmHandle chunked_grm;  // only populated when grm_chunked

            if (grm_chunked) {
                // Skip the dense O(n_grm^2) load entirely — the whole point
                // of --grm-chunked. make_chunked_grm_reader does its own ID
                // validation (same fail-loud contract as the dense path
                // below), reads m_snps from .grm.N.bin's diagonal without
                // touching .grm.bin, and now also solves grm_chunk_rows and
                // its own row-band-cache reservation together (see
                // grm_binary_io.hpp) instead of this file deriving them
                // separately beforehand.
                //
                // NOTE: this now opens the file and ID-matches unconditionally
                // whenever grm_chunked is true, even if the grm_chunk_rows >= n
                // check below ends up falling back to dense -- previously that
                // check ran first (via a standalone solve_chunk_rows call) and
                // skipped opening the file entirely in that case. The extra
                // work is cheap (ID matching + one N-mean read, no tile reads
                // yet) but it's a real behavior change worth knowing about.
                chunked_grm = gcta_grm_io::make_chunked_grm_reader(
                    grm_pfx, analysis_ids, grm_chunked_budget, /*k_ext=*/0, "--grm-chunked-budget");
                grm_chunk_rows = chunked_grm.chunk_rows;
                m_all = chunked_grm.m_snps;
                if (grm_chunk_rows >= n) {
                    LOGGER.w(0, "--grm-chunked-budget=" + to_string(grm_chunked_budget) +
                                "GB covers the full GRM (n=" + to_string(n) + ") in a single chunk. "
                                "Falling back to dense loading instead of paying chunking overhead "
                                "for no benefit.");
                    grm_chunked = false;
                }
                // make_chunked_grm_reader already logged the budget/chunk-size
                // split; nothing further to log here. Note the dense fallback
                // just above discards chunked_grm/its reader rather than using
                // it, since the decision can only be made after construction now.
            } else if (!reuse_basis) {
                // upper_only=true: G_n is valid on its upper triangle (row<=col)
                // only. This is safe here because ctx.A[0] (fed by G_n below) is
                // itself expected to be upper-triangle-only by RemlEngine -- see
                // grm_binary_io.hpp's upper_only doc comment.
                read_grm_binary(grm_pfx, grm_ids, G_n, m_all, /*upper_only=*/true);

                // Get post-filter analysis IDs (FID\tIID) and match to GRM
                const vector<int> kp = match_ids_to_grm(analysis_ids, grm_ids);
                for (int i = 0; i < n; ++i)
                    if (kp[i] < 0)
                        LOGGER.e(0, "Individual [" + analysis_ids[i] +
                                    "] not found in GRM [" + grm_pfx + "]. "
                                    "Re-build the GRM from the same sample set.");

                // Subset GRM to the n analysis individuals.
                // Fast path: if GRM sample == analysis sample (in order), no copy needed.
                const int  n_grm        = static_cast<int>(grm_ids.size());
                bool       is_identity  = (n_grm == n);
                for (int i = 0; i < n && is_identity; ++i)
                    if (kp[i] != i) is_identity = false;

                if (!is_identity) {
                    Eigen::MatrixXd G_sub(n, n);
                    // G_n is upper-triangle-only (row<=col valid). The old
                    // gather read G_n(kp[i], src_col) directly for every (i,j)
                    // pair, which is only correct when kp[i] <= src_col --
                    // otherwise it silently reads unfaulted/zero data now that
                    // the mirror pass is gone. Fixed two ways at once:
                    //   1. Only the upper triangle of G_sub is written
                    //      (i <= j) -- that's all ctx.A[0] needs downstream,
                    //      and it keeps G_sub's own lower-triangle pages
                    //      unfaulted too, carrying the memory saving through
                    //      this subsetting step.
                    //   2. Each read uses min/max on (kp[i], src_col) so it
                    //      always lands in G_n's valid upper triangle,
                    //      regardless of how the permutation reorders things.
                    #pragma omp parallel for schedule(dynamic, 64)
                    for (int j = 0; j < n; ++j) {
                        const int src_col = kp[j];
                        for (int i = 0; i <= j; ++i) {
                            const int r = kp[i], c = src_col;
                            G_sub(i, j) = (r <= c) ? G_n(r, c) : G_n(c, r);
                        }
                    }
                    G_n = std::move(G_sub);
                }
                // else: G_n already contains the right n×n block
            }


            const vector<double> priors =
                options_vd.count("reml_priors") ? options_vd.at("reml_priors") : vector<double>{};
            const vector<double> priors_var =
                options_vd.count("reml_priors_var") ? options_vd.at("reml_priors_var") : vector<double>{};

            // Build REML context
            RemlCtx ctx;
            ctx.n   = n;
            ctx.X_c = x_c;
            ctx.X   = X_design;
            {
                Eigen::VectorXd y_d(n);
                for (int i = 0; i < n; ++i) y_d[i] = phenos_vec[i];
                ctx.y     = y_d;
                ctx.y_Ssq = ctx.y.squaredNorm();
            }
            ctx.A.resize(2);
            if (grm_chunked) {
                // ctx.A[0] stays empty (default RemlMat()); RemlEngine reads
                // K through ctx.grm_tile_reader instead. See the guard in
                // compute_woodbury_basis_basis that errors out if this flag is set
                // without a reader.
                ctx.grm_chunk_rows_from_budget = chunked_grm.chunk_rows;
                ctx.grm_tile_reader = std::move(chunked_grm.reader);
            } else {
                ctx.A[0] = std::move(G_n);
            }
            ctx.grm_N.resize(1, 1);
            ctx.grm_N(0, 0) = m_all;
            // ctx.A[1] left default (size 0 == identity convention for residual)
            ctx.r_indx   = {0, 1};
            ctx.var_name = {"V(G)", "V(e)"};
            ctx.hsq_name = {"V(G)/Vp"};
            ctx.out      = out_prefix;

            ctx.reml_mtd                 = reml_alg;
            ctx.reml_max_iter            = reml_maxit;
            ctx.reml_inv_mtd             = 0;  // LLT
            ctx.reml_diagV_adj           = reml_diagV_adj;
            ctx.reml_force_dense_vi     = options.count("force_dense_V") > 0;
            ctx.woodbury_basis_rank            = woodbury_basis_rank;
            ctx.woodbury_basis_edge_margin        = woodbury_basis_edge_margin;
            ctx.woodbury_basis_edge_confirm       = woodbury_basis_edge_confirm;
            ctx.woodbury_basis_EIG_k_buffer  = woodbury_basis_EIG_k_buffer;
            ctx.woodbury_basis_var_thresh         = options_d.count("woodbury_basis_var_thresh")
                ? options_d.at("woodbury_basis_var_thresh") : 0.001;
            ctx.woodbury_basis_k_init = options_d.count("woodbury_basis_range_init")
                ? static_cast<int>(options_d.at("woodbury_basis_range_init")) : 2000;
            ctx.woodbury_basis_k_max  = options_d.count("woodbury_basis_range_max")
                ? static_cast<int>(options_d.at("woodbury_basis_range_max")) : 25000;
            ctx.woodbury_basis_posthoc_correction = options.count("woodbury_basis_posthoc_correction") > 0;
            ctx.svd_mem_budget_gb   = woodbury_basis_mem_budget_gb;
            ctx.svd_nystrom         = svd_nystrom;
            ctx.svd_power_iter       = svd_power_iter;
            ctx.grm_chunked_budget         = grm_chunked ? grm_chunked_budget : 0.0;
            ctx.reml_trace_hutchpp        = trace_hutchpp;
            ctx.reml_trace_hutchpp_nprobes = trace_hutchpp_nprobes;
            ctx.reml_hutchpp_fixed_probes = options.count("trace_hutchpp_fixed_probes") > 0;
            ctx.woodbury_basis_eigen_mass             = woodbury_basis_eigen_mass;
            ctx.reml_no_HE_start         = no_HE_start;
            ctx.reml_ai_robust      = reml_ai_robust;
            ctx.reml_ai_robust_tol  = reml_ai_robust_tol;
            ctx.reml_ai_robust_risk = reml_ai_robust_risk;
            ctx.reml_print_trajectory = options.count("reml_print_trajectory") > 0;

            if (options.count("woodbury_basis_warm_start")) {
                if (woodbury_basis_rank == 0)
                    LOGGER.w(0, "--reml-woodbury-basis-warm-start given without --reml-woodbury; ignoring.");
                else if (svd_nystrom)
                    LOGGER.w(0, "--reml-woodbury-basis-warm-start has no effect with --svd-method nystrom "
                                "(the Nystrom path doesn't use power iteration); ignoring.");
                else
                    ctx.Uk = load_pca_warm_start(options.at("woodbury_basis_warm_start"), analysis_ids);
            }

            // Fill ctx.Uk/dk/lambda_tail from the saved basis (after the warm-start
            // block above, which it supersedes); reml::compute() then skips the
            // basis construction (ctx.woodbury_basis_preloaded).
            if (reuse_basis)
                load_woodbury_basis_into_ctx(options.at("woodbury_reuse"), ctx, id_hash);

            reml::compute(ctx, priors, priors_var, no_constrain);

            LOGGER.i(0, "Inline REML complete. Variance components:");
            for (size_t ci = 0; ci < ctx.varcmp.size(); ++ci)
                LOGGER.i(0, "  " + ctx.var_name[ci] + " = " + to_string(ctx.varcmp[ci]));

            // Reuse the already-computed REML summary to write .hsq (same feature as --mlma).
            write_hsq_from_ctx(out_prefix, ctx);

            if (options.count("save_reml")) {
                const string save_reml_file = out_prefix + ".reml";
                LOGGER.i(0, "Saving REML state to [" + save_reml_file + "] ...");
                LOGGER.ts("save_reml");
                writeRemlStateFromCtx(save_reml_file, ctx, no_adj_covar, id_hash);
                LOGGER.i(0, "REML state saved in " + to_string(LOGGER.tp("save_reml")) + " seconds.");
                LOGGER.i(0, "REML estimation completed. Use --load-reml " + save_reml_file +
                            " to perform association tests.");
                delete geno;
                delete marker;
                delete pheno;
                continue;
            }

            // y_adj = y - X*b
            {
                const Eigen::MatrixXf Xf = X_design.cast<float>();
                y_vec -= Xf * ctx.b.cast<float>();
            }

            use_inline_ctx = true;

            // ---- Open output file ----
            const string out_file = out_prefix + ".mlma";
            std::ofstream ofile(out_file);
            if (!ofile) LOGGER.e(0, "cannot open [" + out_file + "] for writing.");
            ofile << "Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\t"
                  << (log_pval ? "log_p" : "p") << "\n";

            Eigen::VectorXf w_sqrt = pheno->get_sqrt_weight_keep();
            run_mlma_stream_association(ctx, y_vec, w_sqrt, geno, marker, n, log_pval, ofile,
                                        mlma_tile_budget_gb);
            LOGGER << "\nAssociation results saved to [" << out_file << "]." << std::endl;
            ofile.close();
        }

        if (use_inline_ctx) {
            delete geno;
            delete marker;
            delete pheno;
            continue;
        }

        // ---- Open output file ----
        const string out_file = out_prefix + ".mlma";
        std::ofstream ofile(out_file);
        if (!ofile) LOGGER.e(0, "cannot open [" + out_file + "] for writing.");
        ofile << "Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\t"
              << (log_pval ? "log_p" : "p") << "\n";

        // ---- Stream SNPs ----
        Eigen::VectorXf w_sqrt = pheno->get_sqrt_weight_keep();
        run_mlma_stream_association(state, y_vec, w_sqrt, geno, marker, n, log_pval, ofile,
                                    mlma_tile_budget_gb);
        LOGGER << "\nAssociation results saved to [" << out_file << "]." << std::endl;
        ofile.close();

        delete geno;
        delete marker;
        delete pheno;
    }
}