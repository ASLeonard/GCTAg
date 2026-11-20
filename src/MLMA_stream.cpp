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
#ifdef _OPENMP
  #include <omp.h>
#endif

using std::string;
using std::to_string;
using std::vector;
using std::function;

// ---------------------------------------------------------------------------
// Internal types (anonymous namespace)
// ---------------------------------------------------------------------------
namespace {

// Shared GRM I/O helpers from include/grm_binary_io.hpp.
using gcta_grm_io::read_exact;
using gcta_grm_io::write_exact;
using gcta_grm_io::read_grm_binary;
using gcta_grm_io::match_ids_to_grm;
using gcta_grm_io::load_and_subset_grm_upper;

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

// RemlState is defined in include/RemlState.hpp.
// The readRemlState() function below remains file-local (only MLMA_stream needs it).

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
    gcta_fadvise_sequential(fd);

    // --- Header (already read and validated above) ---
    const Header& hdr = prefix.hdr;

    if (hdr.n <= 0 || hdr.x_c < 0 || hdr.num_varcmp <= 0) {
        close(fd);
        LOGGER.e(0, "[" + filename + "] has invalid header dimensions.");
    }

    const std::string_view magic(hdr.magic, 4);
    RemlState st;
    st.n   = hdr.n;
    st.x_c = hdr.x_c;

    // ------------------------------------------------------------------ GOBY
    if (magic == kMagicDense) {
        int32_t factor_kind = 0;
        read_exact(fd, &factor_kind, sizeof(int32_t), filename);
        st.is_llt = (factor_kind == 0);

        st.Vi_L_f.resize(hdr.n, hdr.n);
        gcta_numa_interleave(st.Vi_L_f.data(),
                             static_cast<size_t>(hdr.n) * static_cast<size_t>(hdr.n) * sizeof(float));

        const size_t tri = static_cast<size_t>(hdr.n) * (hdr.n + 1) / 2;

        // Stream the packed Vi_L block in bounded windows instead of reading
        // the whole tri*4-byte block into RAM at once: read one batch of
        // whole columns (a single large sequential read_exact -- still one
        // contiguous forward pass through the file, just chunked at column
        // boundaries rather than in one shot), unpack that batch into Vi_L_f
        // in parallel, repeat. Bounds the extra buffer to ~kUnpackWindowBytes
        // regardless of n instead of the full tri bytes.
        constexpr size_t kUnpackWindowBytes = 1ULL << 29; // 512 MiB
        LOGGER.i(0, "Reading [" + filename + "]'s packed Vi_L block in " +
                    to_string(kUnpackWindowBytes >> 20) + " MiB windows.");
        AlignedRawBuffer<char, 64> chunk_buf(std::min(kUnpackWindowBytes, tri * sizeof(float)));

        int32_t j = 0;
        while (j < hdr.n) {
            int32_t j_end = j;
            size_t chunk_bytes = 0;
            while (j_end < hdr.n) {
                const size_t col_bytes = static_cast<size_t>(j_end + 1) * sizeof(float);
                if (chunk_bytes > 0 && chunk_bytes + col_bytes > kUnpackWindowBytes) break;
                chunk_bytes += col_bytes;
                ++j_end;
            }

            chunk_buf.resize(chunk_bytes);
            read_exact(fd, chunk_buf.data(), chunk_bytes, filename);

            const size_t chunk_start_elem = static_cast<size_t>(j) * (j + 1) / 2;
            #pragma omp parallel for schedule(static)
            for (int32_t jj = j; jj < j_end; ++jj) {
                const int32_t len = jj + 1;
                const size_t col_elem = static_cast<size_t>(jj) * (jj + 1) / 2;
                const size_t local_off = (col_elem - chunk_start_elem) * sizeof(float);
                std::memcpy(st.Vi_L_f.col(jj).head(len).data(),
                            chunk_buf.data() + local_off,
                            static_cast<size_t>(len) * sizeof(float));
            }

            j = j_end;
        }
        chunk_buf.reset();

        if (!no_adj_covar) {
            st.b.resize(hdr.x_c);
            read_exact(fd, st.b.data(), static_cast<size_t>(hdr.x_c) * sizeof(float), filename);
        }

        st.varcmp.resize(hdr.num_varcmp);
        read_exact(fd, st.varcmp.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float), filename);

        gcta_fadvise_dontneed(fd);
        close(fd);
        return st;
    }

    // ------------------------------------------------------------------ TUNA
    if (magic == kMagicWoodbury) {
        st.is_woodbury = true;
        int32_t k = 0;
        read_exact(fd, &k, sizeof(int32_t), filename);
        if (k <= 0) {
            close(fd);
            LOGGER.e(0, "[" + filename + "] has invalid Woodbury rank k=" + to_string(k) + ".");
        }

        double lambda_tail = 0.0;
        read_exact(fd, &lambda_tail, sizeof(double), filename);
        st.lambda_tail_f = static_cast<float>(lambda_tail);

        // tail_d_var is stored for --reml-woodbury-reuse; association does not use it.
        double tail_d_var = 0.0;
        read_exact(fd, &tail_d_var, sizeof(double), filename);
        (void)tail_d_var;

        // Uk is stored k x n, column-major -- byte-identical to Eigen::MatrixXf(k, n)'s
        // packed layout. Stage reads through a 64 MiB buffer and parallel-copy into
        // NUMA-interleaved Uk so multi-GB pages are faulted across all cores/nodes
        // instead of pinned to Core 0's NUMA domain by a single-threaded read().
        const size_t uk_bytes = static_cast<size_t>(k) * static_cast<size_t>(hdr.n) * sizeof(float);
        Eigen::MatrixXf Uk(k, hdr.n);
        gcta_numa_interleave(Uk.data(), uk_bytes);
        constexpr size_t kStageBytes = 64ULL << 20; // 64 MiB
        AlignedRawBuffer<char, 64> stage(std::min(kStageBytes, uk_bytes));
        char* const uk_dst = reinterpret_cast<char*>(Uk.data());
        size_t done = 0;
        while (done < uk_bytes) {
            const size_t chunk = std::min(stage.size(), uk_bytes - done);
            read_exact(fd, stage.data(), chunk, filename);
            #pragma omp parallel
            {
                const int tid =
                #ifdef _OPENMP
                    omp_get_thread_num();
                #else
                    0;
                #endif
                const int nt =
                #ifdef _OPENMP
                    omp_get_num_threads();
                #else
                    1;
                #endif
                const size_t beg = (chunk * static_cast<size_t>(tid)) / static_cast<size_t>(nt);
                const size_t end = (chunk * static_cast<size_t>(tid + 1)) / static_cast<size_t>(nt);
                if (end > beg) {
                    std::memcpy(uk_dst + done + beg, stage.data() + beg, end - beg);
                }
            }
            done += chunk;
        }
        stage.reset();

        Eigen::VectorXf dk(k);
        read_exact(fd, dk.data(), static_cast<size_t>(k) * sizeof(float), filename);
        st.dk_f = dk;

        if (!no_adj_covar) {
            st.b.resize(hdr.x_c);
            read_exact(fd, st.b.data(), static_cast<size_t>(hdr.x_c) * sizeof(float), filename);
        }

        Eigen::VectorXf vc(hdr.num_varcmp);
        read_exact(fd, vc.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float), filename);
        st.varcmp = vc;

        gcta_fadvise_dontneed(fd);
        close(fd);

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

    close(fd);
    LOGGER.e(0, "[" + filename + "] unsupported format.");
    return st;
}

// Writes the v2 format described above. id_hash = hash_sample_ids(analysis IDs).
// Note: writeRemlStateFromCtx is only invoked on the --save-reml path, which
// exits immediately after saving; it may therefore factorize ctx.Vi in-place.
void writeRemlStateFromCtx(const std::string& filename, RemlCtx& ctx, bool no_adj_covar,
                           uint64_t id_hash)
{
    // Use POSIX low-level I/O for direct control over OS write buffering
    int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
        LOGGER.e(0, "cannot open [" + filename + "] for writing.");

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
        write_exact(fd, &hdr, sizeof(hdr), filename);
        write_exact(fd, &id_hash, sizeof(uint64_t), filename);   // completes the 28-byte prefix

        const int32_t k = static_cast<int32_t>(ctx.dk.size());
        write_exact(fd, &k, sizeof(int32_t), filename);

        const double lambda_tail = ctx.lambda_tail;
        write_exact(fd, &lambda_tail, sizeof(double), filename);

        const double tail_d_var = ctx.tail_d_var;
        write_exact(fd, &tail_d_var, sizeof(double), filename);

        if (ctx.Uk.rows() != ctx.n || ctx.Uk.cols() != k)
            LOGGER.e(0, "invalid Woodbury REML context dimensions before save.");

        // Uk_f must be written k×n (see mlma_woodbury.hpp layout convention),
        // transposed relative to ctx.Uk's n×k storage -- readers place these
        // bytes directly into a k×n matrix with no further transpose.
        // Stream in bounded 512 MiB sample-column chunks using a 64x64
        // cache-blocked transpose (contiguous reads from ctx.Uk and contiguous
        // writes to uk_chunk) so save does not double peak RSS or thrash TLBs.
        constexpr size_t kUkWriteWindowBytes = 1ULL << 29; // 512 MiB
        constexpr int32_t kBlk = 64;
        const size_t col_bytes = static_cast<size_t>(k) * sizeof(float);
        const int32_t cols_per_chunk = static_cast<int32_t>(
            std::max<size_t>(1, kUkWriteWindowBytes / std::max<size_t>(1, col_bytes)));
        AlignedRawBuffer<float, 64> uk_chunk(
            static_cast<size_t>(k) * static_cast<size_t>(std::min(ctx.n, cols_per_chunk)));
        const float* const uk_src = ctx.Uk.data();
        const size_t n_stride = static_cast<size_t>(ctx.n);
        const size_t k_stride = static_cast<size_t>(k);
        const int32_t nj_tiles = (k + kBlk - 1) / kBlk;
        for (int32_t i0 = 0; i0 < ctx.n; i0 += cols_per_chunk) {
            const int32_t i_len = std::min(cols_per_chunk, ctx.n - i0);
            const int32_t ni_tiles = (i_len + kBlk - 1) / kBlk;
            const int64_t total_tiles = static_cast<int64_t>(ni_tiles) * static_cast<int64_t>(nj_tiles);
            #pragma omp parallel for schedule(static)
            for (int64_t t = 0; t < total_tiles; ++t) {
                const int32_t di0 = static_cast<int32_t>(t / nj_tiles) * kBlk;
                const int32_t j0  = static_cast<int32_t>(t % nj_tiles) * kBlk;
                const int32_t di_len = std::min(kBlk, i_len - di0);
                const int32_t j_len  = std::min(kBlk, k - j0);
                alignas(64) float tile[kBlk][kBlk];
                for (int32_t dj = 0; dj < j_len; ++dj) {
                    const float* src_col = uk_src + static_cast<size_t>(j0 + dj) * n_stride + (i0 + di0);
                    for (int32_t ddi = 0; ddi < di_len; ++ddi) {
                        tile[ddi][dj] = src_col[ddi];
                    }
                }
                for (int32_t ddi = 0; ddi < di_len; ++ddi) {
                    float* dst_col = uk_chunk.data() + static_cast<size_t>(di0 + ddi) * k_stride + j0;
                    std::memcpy(dst_col, tile[ddi], static_cast<size_t>(j_len) * sizeof(float));
                }
            }
            write_exact(fd, uk_chunk.data(), static_cast<size_t>(i_len) * col_bytes, filename);
        }
        uk_chunk.reset();
        ctx.Uk.resize(0, 0);

        Eigen::VectorXf dk_f = ctx.dk.cast<float>();
        write_exact(fd, dk_f.data(), static_cast<size_t>(k) * sizeof(float), filename);

        if (!no_adj_covar) {
            Eigen::VectorXf b_f = ctx.b.cast<float>();
            write_exact(fd, b_f.data(), static_cast<size_t>(ctx.X_c) * sizeof(float), filename);
        }

        Eigen::VectorXf varcmp_f = Eigen::Map<const Eigen::VectorXd>(ctx.varcmp.data(),
                                                                      static_cast<Eigen::Index>(ctx.varcmp.size())).cast<float>();
        write_exact(fd, varcmp_f.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float), filename);

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
        write_exact(fd, &hdr, sizeof(hdr), filename);
        write_exact(fd, &id_hash, sizeof(uint64_t), filename);   // completes the 28-byte prefix

        const int32_t factor_kind = ctx.Vi_use_llt ? 0 : 1;
        write_exact(fd, &factor_kind, sizeof(int32_t), filename);

        // Prepare factorization matrix reference. Since --save-reml exits immediately
        // after writing, factorize ctx.Vi in-place when !ctx.Vi_use_llt to avoid an
        // 8*n^2-byte copy.
        const Eigen::MatrixXd* mat_ptr = &ctx.Vi_L;

        if (!ctx.Vi_use_llt) {
            if (ctx.Vi.rows() != ctx.n || ctx.Vi.cols() != ctx.n)
                LOGGER.e(0, "invalid dense REML context dimensions before save.");
            gcta_blas_int blas_n = static_cast<gcta_blas_int>(ctx.n);
            if (gcta_dpotrf(blas_n, ctx.Vi.data(), blas_n, 'U') != 0)
                LOGGER.e(0, "Vi is not positive definite when factorising for save.");
            mat_ptr = &ctx.Vi;
        }

        const Eigen::MatrixXd& mat = *mat_ptr;
        const size_t n_val = static_cast<size_t>(ctx.n);
        const size_t tri_elements = n_val * (n_val + 1) / 2;

        // Stream the packed upper-triangular Vi_L block in bounded 512 MiB column
        // windows (matching readRemlState) instead of allocating a full
        // tri_elements * 4-byte buffer at once.
        constexpr size_t kPackWindowBytes = 1ULL << 29; // 512 MiB
        std::vector<float> chunk_buf;
        chunk_buf.reserve(std::min(kPackWindowBytes / sizeof(float), tri_elements));

        int32_t j = 0;
        while (j < ctx.n) {
            int32_t j_end = j;
            size_t chunk_elems = 0;
            const size_t max_window_elems = kPackWindowBytes / sizeof(float);
            while (j_end < ctx.n) {
                const size_t col_elems = static_cast<size_t>(j_end + 1);
                if (chunk_elems > 0 && chunk_elems + col_elems > max_window_elems) break;
                chunk_elems += col_elems;
                ++j_end;
            }

            chunk_buf.resize(chunk_elems);
            const size_t chunk_start_elem = static_cast<size_t>(j) * (j + 1) / 2;

            #pragma omp parallel for schedule(static)
            for (int32_t jj = j; jj < j_end; ++jj) {
                const int32_t len = jj + 1;
                const double* col_src = mat.col(jj).head(len).data();
                const size_t col_elem = static_cast<size_t>(jj) * (jj + 1) / 2;
                float* dst = chunk_buf.data() + (col_elem - chunk_start_elem);
                for (int32_t i = 0; i < len; ++i)
                    dst[i] = static_cast<float>(col_src[i]);
            }

            write_exact(fd, chunk_buf.data(), chunk_elems * sizeof(float), filename);
            j = j_end;
        }

        if (!no_adj_covar) {
            Eigen::VectorXf b_f = ctx.b.cast<float>();
            write_exact(fd, b_f.data(), static_cast<size_t>(ctx.X_c) * sizeof(float), filename);
        }

        Eigen::VectorXf varcmp_f = Eigen::Map<const Eigen::VectorXd>(ctx.varcmp.data(),
                                                                      static_cast<Eigen::Index>(ctx.varcmp.size())).cast<float>();
        write_exact(fd, varcmp_f.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float), filename);
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
    gcta_fadvise_sequential(fd);

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
    // Stream in bounded 512 MiB sample chunks and run a 64x64 cache-blocked
    // transpose directly into NUMA-interleaved ctx.Uk (n x k), halving peak RSS
    // and eliminating stride-4n TLB thrashing.
    ctx.Uk.resize(ctx.n, k);
    gcta_numa_interleave(ctx.Uk.data(), uk_bytes);
    {
        constexpr size_t kUkReadWindowBytes = 1ULL << 29; // 512 MiB
        constexpr int32_t kBlk = 64;
        const size_t col_bytes = static_cast<size_t>(k) * sizeof(float);
        const int32_t cols_per_chunk = static_cast<int32_t>(
            std::max<size_t>(1, kUkReadWindowBytes / std::max<size_t>(1, col_bytes)));
        AlignedRawBuffer<float, 64> uk_chunk(
            static_cast<size_t>(k) * static_cast<size_t>(std::min(ctx.n, cols_per_chunk)));
        float* const uk_dst = ctx.Uk.data();
        const size_t n_stride = static_cast<size_t>(ctx.n);
        const size_t k_stride = static_cast<size_t>(k);
        const int32_t nj_tiles = (k + kBlk - 1) / kBlk;

        for (int32_t i0 = 0; i0 < ctx.n; i0 += cols_per_chunk) {
            const int32_t i_len = std::min(cols_per_chunk, ctx.n - i0);
            read_exact(fd, uk_chunk.data(), static_cast<size_t>(i_len) * col_bytes, filename);

            const int32_t ni_tiles = (i_len + kBlk - 1) / kBlk;
            const int64_t total_tiles = static_cast<int64_t>(ni_tiles) * static_cast<int64_t>(nj_tiles);
            #pragma omp parallel for schedule(static)
            for (int64_t t = 0; t < total_tiles; ++t) {
                const int32_t j0  = static_cast<int32_t>(t / ni_tiles) * kBlk;
                const int32_t di0 = static_cast<int32_t>(t % ni_tiles) * kBlk;
                const int32_t di_len = std::min(kBlk, i_len - di0);
                const int32_t j_len  = std::min(kBlk, k - j0);
                alignas(64) float tile[kBlk][kBlk];
                for (int32_t ddi = 0; ddi < di_len; ++ddi) {
                    const float* src_col = uk_chunk.data() + static_cast<size_t>(di0 + ddi) * k_stride + j0;
                    for (int32_t dj = 0; dj < j_len; ++dj) {
                        tile[dj][ddi] = src_col[dj];
                    }
                }
                for (int32_t dj = 0; dj < j_len; ++dj) {
                    float* dst_col = uk_dst + static_cast<size_t>(j0 + dj) * n_stride + (i0 + di0);
                    std::memcpy(dst_col, tile[dj], static_cast<size_t>(di_len) * sizeof(float));
                }
            }
        }
        uk_chunk.reset();
    }

    Eigen::VectorXf dk_f(k);
    read_exact(fd, dk_f.data(), dk_bytes, filename);
    gcta_fadvise_dontneed(fd);
    close(fd);

    ctx.dk                       = dk_f.cast<double>();
    ctx.lambda_tail              = lambda_tail;
    ctx.tail_d_var               = tail_d_var;
    ctx.woodbury_basis           = WoodburyBasisSpec{WoodburyBasisSpec::Kind::Fixed, k};
    ctx.woodbury_basis_rank_     = k;
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
Eigen::MatrixXf load_pca_warm_start(const string& eigenvec_prefix,
                                     const vector<string>& analysis_ids)
{
    const string path = eigenvec_prefix + ".eigenvec";
    std::ifstream in(path);
    if (!in) LOGGER.e(0, "--reml-woodbury-basis-warm-start: cannot open [" + path + "].");

    const int n = static_cast<int>(analysis_ids.size());
    std::unordered_map<std::string_view, int> id_to_row;
    id_to_row.reserve(analysis_ids.size() * 2);
    for (int i = 0; i < n; ++i) {
        id_to_row.emplace(analysis_ids[static_cast<size_t>(i)], i);
    }

    auto is_ws = [](unsigned char ch) noexcept {
        return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '\v' || ch == '\f';
    };
    auto next_token = [&](const char*& p, const char* end, std::string_view& tok) noexcept -> bool {
        while (p < end && is_ws(static_cast<unsigned char>(*p))) ++p;
        if (p >= end) return false;
        const char* beg = p;
        while (p < end && !is_ws(static_cast<unsigned char>(*p))) ++p;
        tok = std::string_view(beg, static_cast<size_t>(p - beg));
        return true;
    };

    Eigen::MatrixXf Uk;
    std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
    std::vector<float> row_vals;
    string line;
    string key;
    int k = -1;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const char* p = line.data();
        const char* const end = p + line.size();

        std::string_view fid, iid;
        if (!next_token(p, end, fid) || !next_token(p, end, iid))
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: malformed line in [" + path + "].");

        row_vals.clear();
        std::string_view tok;
        while (next_token(p, end, tok)) {
            const char* tb = tok.data();
            const char* const te = tb + tok.size();
            if (tb < te && *tb == '+') ++tb;
            double vd = 0.0;
            auto [ptr, ec] = std::from_chars(tb, te, vd);
            if (ec == std::errc::result_out_of_range) {
                char num_tmp[128];
                const size_t tlen = static_cast<size_t>(te - tb);
                if (tlen < sizeof(num_tmp)) {
                    std::memcpy(num_tmp, tb, tlen);
                    num_tmp[tlen] = '\0';
                    char* endp = nullptr;
                    vd = std::strtod(num_tmp, &endp);
                    if (endp == num_tmp + tlen) {
                        ec = std::errc();
                        ptr = te;
                    }
                }
            }
            if (ec != std::errc() || ptr != te) {
                LOGGER.e(0, "--reml-woodbury-basis-warm-start: malformed eigenvector value in [" + path + "].");
            }
            row_vals.push_back(static_cast<float>(vd));
        }

        if (k < 0) {
            k = static_cast<int>(row_vals.size());
            if (k <= 0)
                LOGGER.e(0, "--reml-woodbury-basis-warm-start: no eigenvectors read from [" + path + "].");
            Uk.resize(n, k);
            gcta_numa_interleave(Uk.data(), static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(float));
            row_vals.reserve(static_cast<size_t>(k));
        } else if (static_cast<int>(row_vals.size()) != k) {
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: inconsistent eigenvector count in [" + path + "].");
        }

        key.clear();
        key.reserve(fid.size() + 1 + iid.size());
        key.append(fid);
        key.push_back('\t');
        key.append(iid);
        const auto it = id_to_row.find(std::string_view(key));
        if (it != id_to_row.end()) {
            const int row = it->second;
            if (!seen[static_cast<size_t>(row)]) {
                seen[static_cast<size_t>(row)] = 1;
                for (int j = 0; j < k; ++j) {
                    Uk(row, j) = row_vals[static_cast<size_t>(j)];
                }
            }
        }
    }
    if (k <= 0)
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: no eigenvectors read from [" + path + "].");

    for (int i = 0; i < n; ++i) {
        if (!seen[static_cast<size_t>(i)])
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: individual [" + analysis_ids[static_cast<size_t>(i)] +
                        "] not found in [" + path + "]. The warm-start file must come from "
                        "the same sample set (e.g. --pca-approx run on the same --grm).");
    }

    LOGGER.i(0, "Woodbury warm-start: loaded " + to_string(k) + " eigenvector(s) for " +
                to_string(n) + " individuals from [" + path + "].");
    return Uk;
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// MLMA::run
// ---------------------------------------------------------------------------
void MLMA::run(const gcta_cli::MlmaOptions& opts, const std::string& out_prefix)
{
    const auto& r = opts.reml;
    const bool   no_adj_covar   = opts.no_preadj_covar;
    const bool   log_pval       = opts.log_pval;
    const bool   weighted_pheno = opts.pheno_covar.weighted_phenotypes;
    // --save-reml only fits/saves the REML state and never streams SNPs,
    // so the bim (Marker) and genotype (Geno) data are never touched.
    const bool   reml_only      = opts.save_reml;
    const FastMlmaMode fast_mlma_mode = opts.fast_mlma.value_or(FastMlmaMode::Off);
    if (opts.fast_mlma.has_value() && reml_only) {
        LOGGER.w(0, "--fast-mlma applies to association scanning and has no effect with --save-reml.");
    }

    if (no_adj_covar)
        LOGGER.e(0, "--mlma-no-preadj-covar is not yet supported in --MLMA. "
                    "Re-run without this flag (pre-adjustment is the default).");

    // only if requested; otherwise leave it seeded from entropy (the default).
    if (r.seed.has_value()) {
        const auto seed = static_cast<std::mt19937::result_type>(*r.seed);
        gcta_eigh::shared_rng().seed(seed);
        LOGGER.i(0, "Using random seed: " + to_string(seed) + ".");
    }

    if (r.ai_robust && r.trace_hutchpp.has_value() && !r.trace_hutchpp_fixed_probes)
        LOGGER.w(0, "Using fresh Hutch++ probes without robust AI convergence can lead to unstable REML convergence. Consider using --reml-ai-robust or --reml-trace-hutchpp-fixed-probes.");
    if (r.woodbury_posthoc_correction)
        LOGGER.w(0, "--reml-woodbury-basis-posthoc-correction is an experimental feature and likely to worsen results.");

    // ---- Pheno / Marker / Geno (Geno re-reads pheno state in loopDouble) ----
    // Pheno alone reads the .fam (sample IDs); Marker/Geno additionally need
    // the .bim/.bed and are skipped entirely in --save-reml-only mode.
    Pheno pheno(opts.geno, opts.sample_filter, &opts.pheno_covar);
    std::unique_ptr<Marker> marker;
    std::unique_ptr<Geno>   geno;
    if (!reml_only) {
        marker = std::make_unique<Marker>(opts.geno, opts.sample_filter, opts.marker_filter);
        geno   = std::make_unique<Geno>(&pheno, marker.get(), opts.geno, opts.marker_filter, out_prefix);
        geno->setGenoCodingModel(opts.model);
        LOGGER.i(0, std::string("MLMA_stream genotype model: ") + gcta_cli::geno_model_name(opts.model) + ".");
    }

        // ---- Phenotype ----
        const uint32_t n_init = pheno.count_keep();
        vector<string> ids;
        vector<double> phenos_vec;
        pheno.get_pheno(ids, phenos_vec);
        if (ids.size() != n_init)
            LOGGER.e(0, "--MLMA requires --pheno to be specified.");

        // ---- Covariates ----
        Covar covar(opts.pheno_covar);
        bool  has_covar = false;
        int   x_c       = 1;
        Eigen::MatrixXd X_design;

        {
            vector<uint32_t> remain_index, covar_index;
            has_covar = covar.getCommonSampleIndex(ids, remain_index, covar_index);

            if (has_covar) {
                LOGGER.i(0, to_string(remain_index.size()) +
                            " overlapping individuals with non-missing covariate data.");
                pheno.filter_keep_index(remain_index);
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

        const int n = static_cast<int>(pheno.count_keep());
        LOGGER.i(0, "After matching all files, " + to_string(n) +
                    " individuals to be included in --MLMA.");

        // Build design matrix X  (n × x_c, column-major)
        if (has_covar) {
            vector<string> final_ids = pheno.get_id(0, n - 1, "\t");
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
        const double mlma_tile_budget_gb = opts.stream_budget_gb.value_or(0.0);

        if (!opts.load_reml.empty()) {
            // --load-reml uses the serialized RemlState path: the state is read
            // from disk and then consumed by run_mlma_stream_association(RemlState&)
            // below. This is the saved-state code path and uses the float-first
            // Woodbury/L factor data in RemlState (see RemlState.hpp and the
            // RemlState overload in MLMA_stream_common.hpp).
            const string& load_reml_file = opts.load_reml;
            LOGGER.i(0, "Loading REML state from [" + load_reml_file + "]...");
            LOGGER.ts("load_reml");
            state = readRemlState(load_reml_file, no_adj_covar, n,
                                  hash_sample_ids(pheno.get_id(0, n - 1, "\t")));
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
            const bool   reuse_basis = !r.woodbury_reuse.empty();
            const string grm_pfx     = reuse_basis ? string() : r.grm;
            if (reuse_basis)
                LOGGER.i(0, "Running inline REML with the Woodbury basis reused from [" +
                            r.woodbury_reuse + "] (no GRM is read) ...");
            else
                LOGGER.i(0, "Running inline REML using GRM [" + grm_pfx + "] ...");

            const vector<string> analysis_ids = pheno.get_id(0, n - 1, "\t");
            const uint64_t id_hash = hash_sample_ids(analysis_ids);
            const double grm_chunked_budget = reuse_basis ? 0.0 : r.grm_chunked_budget.value_or(0.0);
            bool grm_chunked = grm_chunked_budget > 0.0;
            const bool grm_fp16 = !reuse_basis && r.grm_fp16;
            if (grm_chunked && grm_fp16)
                LOGGER.e(0, "--grm-fp16 and --grm-chunked-budget are mutually exclusive: "
                            "--grm-fp16 holds the packed fp16 lower triangle in memory, whereas "
                            "--grm-chunked-budget streams float32 bands from disk.");

            // REML tuning parameters. Parsed here (ahead of the GRM load below,
            // not after it as before) because woodbury_basis_mem_budget_gb feeds
            // the chunk-size solve immediately below, which needs to run before
            // committing to a chunked vs. dense load. Also lets the
            // svd_nystrom/etc. validation checks fail fast, ahead of
            // the (potentially expensive) GRM read, instead of after it.
            const bool   have_wb                      = r.woodbury_basis.has_value();
            const bool   is_wb_mp                     = have_wb && r.woodbury_basis->kind == WoodburyBasisSpec::Kind::MP;
            const bool   trace_hutchpp                = r.trace_hutchpp.has_value();
            const int    trace_hutchpp_nprobes        = r.trace_hutchpp.value_or(100);
            const int    reml_maxit                   = r.maxit.value_or(100);
            const bool   no_constrain                 = r.no_constrain;
            const bool   svd_nystrom                  = (r.svd_method == gcta_cli::SvdMethod::Nystrom);
            const int    svd_power_iter               = r.svd_power_iter.value_or(3);
            const float  woodbury_basis_eigen_mass    = static_cast<float>(r.woodbury_eig_mass.value_or(0.99));
            const double woodbury_basis_edge_margin   = r.woodbury_mp_margin.value_or(0.15);
            const int    woodbury_basis_edge_confirm  = r.woodbury_mp_confirm.value_or(20);
            const int    woodbury_basis_EIG_k_buffer  = r.woodbury_eig_k_buffer.value_or(0);
            const double woodbury_basis_mem_budget_gb = r.woodbury_mem_budget_gb.value_or(0.0);
            const bool   no_HE_start                  = r.no_he_start;
            const bool   reml_ai_robust               = r.ai_robust;
            const double reml_ai_robust_tol           = r.ai_robust_tol.value_or(1e-4);
            const double reml_ai_robust_risk          = r.ai_robust_risk.value_or(0.01);

            if ((have_wb || reuse_basis) && r.alg == RemlAlg::Fisher)
                LOGGER.e(0, "--reml-woodbury is incompatible with Fisher-scoring REML (--reml-alg 1). Use AI-REML (default) or EM-REML (--reml-alg 2).");
            if (svd_nystrom && !have_wb)
                LOGGER.e(0, "--svd-method nystrom requires --reml-woodbury <k|MP|EIG|VAR>.");
            if ((have_wb || reuse_basis) && trace_hutchpp)
                LOGGER.w(0, "--reml-woodbury-basis/--reml-woodbury-reuse and --reml-trace-hutchpp both given; "
                            "the Woodbury basis provides an exact tr(PA) and takes precedence — "
                            "--reml-trace-hutchpp is ignored.");

            int grm_chunk_rows = 0;
            Eigen::MatrixXf G_n;      // left empty when grm_chunked or grm_fp16
            double m_all = is_wb_mp ? 0.0 : -1.0;  // only populated when Woodbury-MP mode is selected
            gcta_grm_io::ChunkedGrmHandle chunked_grm;       // only populated when grm_chunked
            gcta_grm_io::PackedGrmFp16Handle fp16_grm;       // only populated when grm_fp16

            if (grm_fp16) {
                fp16_grm = gcta_grm_io::load_packed_grm_fp16(
                    grm_pfx, analysis_ids, /*need_m_snps=*/is_wb_mp, "--grm-fp16");
                grm_chunk_rows = fp16_grm.chunk_rows;
                m_all = fp16_grm.m_snps;
            } else if (grm_chunked) {
                chunked_grm = gcta_grm_io::make_chunked_grm_reader(
                    grm_pfx, analysis_ids, grm_chunked_budget, /*k_ext=*/0, "--grm-chunked-budget",
                    /*need_m_snps=*/is_wb_mp);
                grm_chunk_rows = chunked_grm.chunk_rows;
                m_all = chunked_grm.m_snps;
                if (grm_chunk_rows >= n) {
                    LOGGER.w(0, "--grm-chunked-budget=" + to_string(grm_chunked_budget) +
                                "GB covers the full GRM (n=" + to_string(n) + ") in a single chunk. "
                                "Falling back to dense loading instead of paying chunking overhead "
                                "for no benefit.");
                    chunked_grm = {};
                    grm_chunked = false;
                }
            }
            if (!grm_chunked && !grm_fp16 && !reuse_basis) {
                double m_all_load = (m_all > 0.0) ? -1.0 : m_all;
                load_and_subset_grm_upper(grm_pfx, analysis_ids, G_n, m_all_load, "--mlma");
                if (m_all <= 0.0) m_all = m_all_load;
            }


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
            if (grm_fp16) {
                ctx.grm_fp16 = true;
                ctx.grm_fp16_trace_K = fp16_grm.trace_K;
                ctx.grm_fp16_trace_K2 = fp16_grm.trace_K2;
                ctx.grm_chunk_rows_from_budget = fp16_grm.chunk_rows;
                ctx.grm_tile_reader = std::move(fp16_grm.reader);
                ctx.grm_tile_reader_f32 = std::move(fp16_grm.reader_f32);
                ctx.grm_fp16_release = [file = std::move(fp16_grm.file), &ctx]() mutable {
                    file.reset();
                    ctx.grm_tile_reader = nullptr;
                    ctx.grm_tile_reader_f32 = nullptr;
                };
            } else if (grm_chunked) {
                // ctx.A[0] stays empty (default RemlMat()); RemlEngine reads
                // K through ctx.grm_tile_reader instead. See the guard in
                // compute_woodbury_basis_basis that errors out if this flag is set
                // without a reader.
                ctx.grm_chunk_rows_from_budget = chunked_grm.chunk_rows;
                ctx.grm_tile_reader = std::move(chunked_grm.reader);
                ctx.grm_tile_reader_f32 = std::move(chunked_grm.reader_f32);
                // Drop extra shared_ptr ref so ChunkedGrmReader frees as soon as
                // RemlEngine releases ctx.grm_tile_reader (avoids 10-60 GB leak
                // persisting through the downstream MLMA streaming scan).
                chunked_grm.file.reset();
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

            ctx.reml_mtd                 = r.alg;
            ctx.reml_max_iter            = reml_maxit;
            ctx.reml_inv_mtd             = 0;  // LLT
            ctx.reml_diagV_adj           = r.diagV_adj;
            ctx.reml_force_dense_vi      = r.force_dense_v;
            ctx.woodbury_basis           = r.woodbury_basis;
            ctx.woodbury_basis_edge_margin        = woodbury_basis_edge_margin;
            ctx.woodbury_basis_edge_confirm       = woodbury_basis_edge_confirm;
            ctx.woodbury_basis_EIG_k_buffer  = woodbury_basis_EIG_k_buffer;
            ctx.woodbury_basis_var_thresh         = r.woodbury_var_tail.value_or(0.001);
            ctx.woodbury_basis_k_init = r.woodbury_range ? r.woodbury_range->first : 2000;
            ctx.woodbury_basis_k_max  = r.woodbury_range ? r.woodbury_range->second : 25000;
            ctx.woodbury_basis_posthoc_correction = r.woodbury_posthoc_correction;
            ctx.svd_mem_budget_gb   = woodbury_basis_mem_budget_gb;
            ctx.svd_nystrom         = svd_nystrom;
            ctx.svd_power_iter       = svd_power_iter;
            ctx.grm_chunked_budget         = grm_chunked ? grm_chunked_budget : 0.0;
            ctx.reml_trace_hutchpp        = trace_hutchpp;
            ctx.reml_trace_hutchpp_nprobes = trace_hutchpp_nprobes;
            ctx.reml_hutchpp_fixed_probes = r.trace_hutchpp_fixed_probes;
            ctx.woodbury_basis_eigen_mass             = woodbury_basis_eigen_mass;
            ctx.reml_no_HE_start         = no_HE_start;
            ctx.reml_ai_robust      = reml_ai_robust;
            ctx.reml_ai_robust_tol  = reml_ai_robust_tol;
            ctx.reml_ai_robust_risk = reml_ai_robust_risk;
            ctx.reml_print_trajectory = r.print_trajectory;

            if (!r.woodbury_warm_start.empty()) {
                if (!have_wb)
                    LOGGER.w(0, "--reml-woodbury-basis-warm-start given without --reml-woodbury; ignoring.");
                else if (svd_nystrom)
                    LOGGER.w(0, "--reml-woodbury-basis-warm-start has no effect with --svd-method nystrom "
                                "(the Nystrom path doesn't use power iteration); ignoring.");
                else
                    ctx.Uk = load_pca_warm_start(r.woodbury_warm_start, analysis_ids);
            }

            // Fill ctx.Uk/dk/lambda_tail from the saved basis (after the warm-start
            // block above, which it supersedes); reml::compute() then skips the
            // basis construction (ctx.woodbury_basis_preloaded).
            if (reuse_basis)
                load_woodbury_basis_into_ctx(r.woodbury_reuse, ctx, id_hash);

            reml::compute(ctx, r.priors, r.priors_var, no_constrain);
            ctx.grm_tile_reader = nullptr;
            ctx.grm_tile_reader_f32 = nullptr;

            LOGGER.i(0, "Inline REML complete. Variance components:");
            for (size_t ci = 0; ci < ctx.varcmp.size(); ++ci)
                LOGGER.i(0, "  " + ctx.var_name[ci] + " = " + to_string(ctx.varcmp[ci]));

            // Reuse the already-computed REML summary to write .hsq (same feature as --mlma).
            write_hsq_from_ctx(out_prefix, ctx);

            if (reml_only) {
                const string save_reml_file = out_prefix + ".reml";
                LOGGER.i(0, "Saving REML state to [" + save_reml_file + "] ...");
                LOGGER.ts("save_reml");
                writeRemlStateFromCtx(save_reml_file, ctx, no_adj_covar, id_hash);
                LOGGER.i(0, "REML state saved in " + to_string(LOGGER.tp("save_reml")) + " seconds.");
                LOGGER.i(0, "REML estimation completed. Use --load-reml " + save_reml_file +
                            " to perform association tests.");
                return;
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

            Eigen::VectorXf w_sqrt = weighted_pheno
                ? pheno.get_sqrt_weight_keep()
                : Eigen::VectorXf::Ones(n);
            run_mlma_stream_association(ctx, y_vec, w_sqrt, geno.get(), marker.get(), n, log_pval, ofile,
                                        mlma_tile_budget_gb, fast_mlma_mode);
            LOGGER << "\nAssociation results saved to [" << out_file << "]." << std::endl;
            ofile.close();
        }

        if (use_inline_ctx) {
            return;
        }

        // ---- Open output file ----
        const string out_file = out_prefix + ".mlma";
        std::ofstream ofile(out_file);
        if (!ofile) LOGGER.e(0, "cannot open [" + out_file + "] for writing.");
        ofile << "Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\t"
              << (log_pval ? "log_p" : "p") << "\n";

        // ---- Stream SNPs ----
        Eigen::VectorXf w_sqrt = weighted_pheno
            ? pheno.get_sqrt_weight_keep()
            : Eigen::VectorXf::Ones(n);
        run_mlma_stream_association(state, y_vec, w_sqrt, geno.get(), marker.get(), n, log_pval, ofile,
                                    mlma_tile_budget_gb, fast_mlma_mode);
        LOGGER << "\nAssociation results saved to [" << out_file << "]." << std::endl;
        ofile.close();
}