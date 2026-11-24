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


#include "MLMA.h"
#include "MLMA_stream_common.hpp"
#include "Covar.h"
#include "main/StatFunc.h"
#include "mlma_woodbury.hpp"
#include "symmetric_eigendecomp.hpp"
#include "chunked_grm_matvec.hpp"
#include "RemlState.hpp"
#include "RemlCtx.hpp"
#include "RemlEngine.hpp"
#include "async_seq_io.hpp"
#include "grm_format.hpp"
#include "grm_binary_io.hpp"

#include <Eigen/Dense>
#include <filesystem>
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
#include <chrono>
#include <format>
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

// Shared GRM & sequential I/O helpers.
using gcta_grm_io::read_exact;
using gcta_grm_io::write_exact;
using gcta_grm_io::read_grm_binary;
using gcta_grm_io::match_ids_to_grm;
using gcta_grm_io::load_and_subset_grm_upper;
using gcta_io::hash_bytes;
using gcta_io::hash_sample_ids;
using gcta_io::hash_hex;

struct ScopedFd {
    int fd = -1;
    ScopedFd() = default;
    explicit ScopedFd(int f) noexcept : fd(f) {}
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ScopedFd(ScopedFd&& o) noexcept : fd(o.fd) { o.fd = -1; }
    ScopedFd& operator=(ScopedFd&& o) noexcept {
        if (this != &o) { reset(); fd = o.fd; o.fd = -1; }
        return *this;
    }
    ~ScopedFd() noexcept { reset(); }
    void reset() noexcept { if (fd >= 0) { ::close(fd); fd = -1; } }
    int release() noexcept { int f = fd; fd = -1; return f; }
    int get() const noexcept { return fd; }
    explicit operator bool() const noexcept { return fd >= 0; }
};

bool is_regular_file_path(const string& path) noexcept
{
    struct stat st{};
    return (::stat(path.c_str(), &st) == 0) && S_ISREG(st.st_mode);
}

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

// ---------------------------------------------------------------------------
// .reml on-disk formats (v3). Every file starts with a fixed 28-byte prefix:
//     [Header: magic[4], n, x_c, num_varcmp, num_r_indx][uint64 id_hash]
// followed, depending on magic, by:
//   "TUN3" (Compact 16-bit Woodbury, header-first sequential stream):
//     [uint32 flags][int32 k][double lambda_tail][double tail_d_var]
//     [dk: k float][b: x_c float (unless kTun3FlagNoAdjCovar)]
//     [varcmp: num_varcmp float][uk_row_scale: k float]
//     [uint64 y_adj_hash][Vi_y: n float (if kTun3FlagHasViY)]
//     [Uk_q: k x n uint16 (ColMajor, hybrid Row-INT16(qmax=32385) / IEEE FP16)]
//     [row_eps_sq: k float]
//   "TREF" (Thin multi-trait Woodbury reference state, ~2 MB):
//     [uint64 basis_hash][uint32 flags][uint32 path_len][char basis_path[path_len]]
//     [int32 k][double lambda_tail][double tail_d_var]
//     [dk: k float][b: x_c float (unless kTun3FlagNoAdjCovar)]
//     [varcmp: num_varcmp float][uint64 y_adj_hash][Vi_y: n float (if kTun3FlagHasViY)]
//   "GOBY" (Exact dense float32 Cholesky factor):
//     [int32 factor_kind][packed upper-triangular Vi_L: n(n+1)/2 float]
//     [b: x_c float, omitted if no_adj_covar][varcmp: num_varcmp float]
// Legacy "TUNA" v2 (float32 Woodbury) and v1 files are rejected with a clear message.
// ---------------------------------------------------------------------------
constexpr std::string_view kMagicTun3       = "TUN3";
constexpr std::string_view kMagicTref       = "TREF";
constexpr std::string_view kMagicDense      = "GOBY";
constexpr std::string_view kMagicLegacyTuna = "TUNA";
constexpr std::string_view kMagicEvec       = "EVEC";
constexpr size_t kRemlPrefixBytes = sizeof(Header) + sizeof(uint64_t);

constexpr uint32_t kTun3FlagHasViY     = 1u << 0;
constexpr uint32_t kTun3FlagNoAdjCovar = 1u << 1;

struct EvecHeader {
    char     magic[4]{'E', 'V', 'E', 'C'};
    uint32_t version  = 1u;
    uint32_t dtype    = 0u; // 0 = float32, 1 = row-scaled INT16
    int32_t  n        = 0;
    int32_t  k        = 0;
    uint32_t reserved = 0u;
    uint64_t id_hash  = 0;
    uint64_t id_bytes = 0;
};
static_assert(sizeof(EvecHeader) == 40, "EvecHeader must be 40 bytes");
static_assert(std::is_trivially_copyable_v<EvecHeader>, "EvecHeader must be trivially copyable");

enum class RemlKind : uint8_t { Tun3 = 0, Tref = 1, Goby = 2 };

struct RemlPrefix {
    Header   hdr{};
    uint64_t id_hash = 0;
    RemlKind kind    = RemlKind::Goby;
};

// Deterministic fixed-column-order y_adj = y_f - X * b_f (bitwise identical across --threads).
Eigen::VectorXf compute_y_adj_deterministic(const Eigen::VectorXf& y_f,
                                            const Eigen::MatrixXd& X_d,
                                            const Eigen::VectorXf& b_f)
{
    const int n  = static_cast<int>(y_f.size());
    const int xc = static_cast<int>(b_f.size());
    Eigen::VectorXf out = y_f;
    if (xc <= 0 || X_d.rows() != n || X_d.cols() != xc) return out;
    #pragma omp parallel for schedule(static) if (n >= 32768)
    for (int i = 0; i < n; ++i) {
        double xb = 0.0;
        for (int c = 0; c < xc; ++c) {
            xb += static_cast<double>(static_cast<float>(X_d(i, c)))
                * static_cast<double>(b_f[c]);
        }
        out[i] = y_f[i] - static_cast<float>(xb);
    }
    return out;
}

uint64_t compute_tun3_basis_hash(int32_t n, uint64_t id_hash, int32_t k,
                                 double lambda_tail, double tail_d_var,
                                 const float* dk, const float* uk_row_scale) noexcept
{
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](const void* ptr, size_t len) {
        const auto* p = static_cast<const unsigned char*>(ptr);
        for (size_t i = 0; i < len; ++i) { h ^= p[i]; h *= kPrime; }
    };
    mix(&n, sizeof(n));
    mix(&id_hash, sizeof(id_hash));
    mix(&k, sizeof(k));
    mix(&lambda_tail, sizeof(lambda_tail));
    mix(&tail_d_var, sizeof(tail_d_var));
    if (k > 0) {
        mix(dk, static_cast<size_t>(k) * sizeof(float));
        mix(uk_row_scale, static_cast<size_t>(k) * sizeof(float));
    }
    return h != 0 ? h : 1ULL;
}

void compute_woodbury_ck_from_saved(const Eigen::VectorXf& dk_f, double lambda_tail,
                                    const Eigen::VectorXf& vc_f,
                                    Eigen::VectorXf& ck_f, Eigen::VectorXf& sqrt_ck_f,
                                    float& sigma2_eff_f)
{
    const int32_t k = static_cast<int32_t>(dk_f.size());
    const double sg2   = static_cast<double>(vc_f[0]);
    const double se2   = static_cast<double>(vc_f[vc_f.size() - 1]);
    const double sig2e = sg2 * lambda_tail + se2;
    ck_f.resize(k);
    for (int32_t j = 0; j < k; ++j) {
        const double delta = std::max(0.0, static_cast<double>(dk_f[j]) - lambda_tail);
        const double sd    = sg2 * delta;
        ck_f[j] = static_cast<float>((sig2e + sd > 0.0) ? (sd / (sig2e + sd)) : 0.0);
    }
    sqrt_ck_f    = ck_f.cwiseSqrt();
    sigma2_eff_f = static_cast<float>(sig2e);
}

// Read and validate the fixed 28-byte prefix of a .reml from `fd` (at offset 0).
// Caller owns `fd` (ScopedFd).
RemlPrefix read_reml_prefix(int fd, const string& filename)
{
    RemlPrefix p;
    try {
        gcta_io::read_exact_throwing(fd, &p.hdr, sizeof(Header), filename);
        gcta_io::read_exact_throwing(fd, &p.id_hash, sizeof(uint64_t), filename);
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }

    const std::string_view magic(p.hdr.magic, 4);
    if (magic == kMagicTun3) {
        p.kind = RemlKind::Tun3;
    } else if (magic == kMagicTref) {
        p.kind = RemlKind::Tref;
    } else if (magic == kMagicDense) {
        p.kind = RemlKind::Goby;
    } else if (magic == kMagicLegacyTuna) {
        LOGGER.e(0, "[" + filename + "] uses the legacy float32 Woodbury .reml format ('TUNA' v2), "
                    "which has been replaced by the compact 16-bit 'TUN3' format. "
                    "Re-create it with 'GCTAg reml ...' (--save-reml).");
    } else {
        LOGGER.e(0, "[" + filename + "] is not a supported REML state file "
                    "(unrecognised or legacy v1 magic; re-create it with --save-reml).");
    }
    return p;
}

void check_reml_samples(const RemlPrefix& p, int expected_n, uint64_t expected_id_hash,
                        const string& filename)
{
    if (p.hdr.n != expected_n) {
        LOGGER.e(0, "Sample size mismatch: [" + filename + "] has n=" + to_string(p.hdr.n) +
                    " vs dataset n=" + to_string(expected_n) +
                    ". Use the same filters (--keep/--remove/--pheno) as the run that saved it.");
    }
    if (p.id_hash != expected_id_hash) {
        LOGGER.e(0, "Sample ID mismatch: the ordered analysis IDs (hash " + hash_hex(expected_id_hash) +
                    ") differ from those [" + filename + "] was saved with (hash " + hash_hex(p.id_hash) +
                    "). Use the same samples in the same order (--keep/--remove and the row order of "
                    "--pheno/--covar) as the run that saved it.");
    }
}

struct Tun3Metadata {
    Header hdr{};
    uint64_t id_hash     = 0;
    uint32_t flags       = 0;
    int32_t  k           = 0;
    double   lambda_tail = 0.0;
    double   tail_d_var  = 0.0;
    Eigen::VectorXf dk_f;
    Eigen::VectorXf b_f;
    Eigen::VectorXf varcmp_f;
    std::vector<float> uk_row_scale;
    uint64_t y_adj_hash  = 0;
    Eigen::VectorXf vi_y_f;
    off_t    body_offset = 0;
    uint64_t basis_hash  = 0;
};

struct TrefMetadata {
    Header hdr{};
    uint64_t id_hash     = 0;
    uint64_t basis_hash  = 0;
    uint32_t flags       = 0;
    string   basis_path;
    int32_t  k           = 0;
    double   lambda_tail = 0.0;
    double   tail_d_var  = 0.0;
    Eigen::VectorXf dk_f;
    Eigen::VectorXf b_f;
    Eigen::VectorXf varcmp_f;
    uint64_t y_adj_hash  = 0;
    Eigen::VectorXf vi_y_f;
};

// Read TUN3 header metadata sequentially from `fd` (positioned right after the 28-byte prefix).
Tun3Metadata read_tun3_metadata(int fd, const RemlPrefix& p, size_t file_size,
                                const string& filename, bool load_vi_y)
{
    Tun3Metadata m;
    m.hdr     = p.hdr;
    m.id_hash = p.id_hash;
    if (m.hdr.n <= 0 || m.hdr.x_c < 0 || m.hdr.num_varcmp <= 0) {
        LOGGER.e(0, "[" + filename + "] has invalid TUN3 header dimensions.");
    }

    size_t pos = kRemlPrefixBytes;
    auto rd = [&](void* dst, size_t bytes) {
        try {
            gcta_io::read_exact_throwing(fd, dst, bytes, filename);
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }
        pos += bytes;
    };

    rd(&m.flags, sizeof(uint32_t));
    rd(&m.k, sizeof(int32_t));
    rd(&m.lambda_tail, sizeof(double));
    rd(&m.tail_d_var, sizeof(double));

    if (m.k <= 0 || m.k > m.hdr.n) {
        LOGGER.e(0, "[" + filename + "] has invalid Woodbury rank k=" + to_string(m.k) + ".");
    }
    if (!std::isfinite(m.lambda_tail) || !std::isfinite(m.tail_d_var) || m.tail_d_var < 0.0) {
        LOGGER.e(0, "[" + filename + "] has invalid Woodbury tail statistics.");
    }

    m.dk_f.resize(m.k);
    rd(m.dk_f.data(), static_cast<size_t>(m.k) * sizeof(float));

    const bool has_b = ((m.flags & kTun3FlagNoAdjCovar) == 0u) && (m.hdr.x_c > 0);
    if (has_b) {
        m.b_f.resize(m.hdr.x_c);
        rd(m.b_f.data(), static_cast<size_t>(m.hdr.x_c) * sizeof(float));
    }

    m.varcmp_f.resize(m.hdr.num_varcmp);
    rd(m.varcmp_f.data(), static_cast<size_t>(m.hdr.num_varcmp) * sizeof(float));

    m.uk_row_scale.resize(static_cast<size_t>(m.k));
    rd(m.uk_row_scale.data(), static_cast<size_t>(m.k) * sizeof(float));

    rd(&m.y_adj_hash, sizeof(uint64_t));

    if ((m.flags & kTun3FlagHasViY) != 0u) {
        const size_t viy_bytes = static_cast<size_t>(m.hdr.n) * sizeof(float);
        if (load_vi_y) {
            m.vi_y_f.resize(m.hdr.n);
            rd(m.vi_y_f.data(), viy_bytes);
        } else {
            pos += viy_bytes;
            if (::lseek(fd, static_cast<off_t>(pos), SEEK_SET) == static_cast<off_t>(-1)) {
                LOGGER.e(0, "lseek() failed on [" + filename + "]: " + string(std::strerror(errno)));
            }
        }
    }

    m.body_offset = static_cast<off_t>(pos);
    const size_t uk_bytes      = static_cast<size_t>(m.k) * static_cast<size_t>(m.hdr.n) * sizeof(uint16_t);
    const size_t trailer_bytes = static_cast<size_t>(m.k) * sizeof(float);
    const size_t expected_min  = pos + uk_bytes + trailer_bytes;
    if (file_size < expected_min) {
        LOGGER.e(0, "[" + filename + "] is truncated for TUN3 n=" + to_string(m.hdr.n) +
                    ", k=" + to_string(m.k) + " (expected at least " + to_string(expected_min) +
                    " bytes, found " + to_string(file_size) + ").");
    }

    m.basis_hash = compute_tun3_basis_hash(m.hdr.n, m.id_hash, m.k, m.lambda_tail, m.tail_d_var,
                                           m.dk_f.data(), m.uk_row_scale.data());
    return m;
}

// Read TREF thin reference metadata sequentially from `fd` (right after 28-byte prefix).
TrefMetadata read_tref_metadata(int fd, const RemlPrefix& p, const string& filename, bool load_vi_y)
{
    TrefMetadata m;
    m.hdr     = p.hdr;
    m.id_hash = p.id_hash;
    if (m.hdr.n <= 0 || m.hdr.x_c < 0 || m.hdr.num_varcmp <= 0) {
        LOGGER.e(0, "[" + filename + "] has invalid TREF header dimensions.");
    }

    auto rd = [&](void* dst, size_t bytes) {
        try {
            gcta_io::read_exact_throwing(fd, dst, bytes, filename);
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }
    };

    rd(&m.basis_hash, sizeof(uint64_t));
    rd(&m.flags, sizeof(uint32_t));
    uint32_t path_len = 0;
    rd(&path_len, sizeof(uint32_t));
    if (path_len == 0 || path_len > (1u << 20)) {
        LOGGER.e(0, "[" + filename + "] has invalid TREF basis_path length=" + to_string(path_len) + ".");
    }
    m.basis_path.resize(path_len);
    rd(m.basis_path.data(), path_len);

    rd(&m.k, sizeof(int32_t));
    rd(&m.lambda_tail, sizeof(double));
    rd(&m.tail_d_var, sizeof(double));
    if (m.k <= 0 || m.k > m.hdr.n) {
        LOGGER.e(0, "[" + filename + "] has invalid TREF Woodbury rank k=" + to_string(m.k) + ".");
    }
    if (!std::isfinite(m.lambda_tail) || !std::isfinite(m.tail_d_var) || m.tail_d_var < 0.0) {
        LOGGER.e(0, "[" + filename + "] has invalid TREF Woodbury tail statistics.");
    }

    m.dk_f.resize(m.k);
    rd(m.dk_f.data(), static_cast<size_t>(m.k) * sizeof(float));

    const bool has_b = ((m.flags & kTun3FlagNoAdjCovar) == 0u) && (m.hdr.x_c > 0);
    if (has_b) {
        m.b_f.resize(m.hdr.x_c);
        rd(m.b_f.data(), static_cast<size_t>(m.hdr.x_c) * sizeof(float));
    }

    m.varcmp_f.resize(m.hdr.num_varcmp);
    rd(m.varcmp_f.data(), static_cast<size_t>(m.hdr.num_varcmp) * sizeof(float));

    rd(&m.y_adj_hash, sizeof(uint64_t));
    if ((m.flags & kTun3FlagHasViY) != 0u && load_vi_y) {
        m.vi_y_f.resize(m.hdr.n);
        rd(m.vi_y_f.data(), static_cast<size_t>(m.hdr.n) * sizeof(float));
    }
    return m;
}

string resolve_tref_basis_path(const string& tref_file, const string& stored_path,
                               const string& override_path)
{
    if (!override_path.empty()) return override_path;

    namespace fs = std::filesystem;
    const fs::path sp(stored_path);
    const fs::path tref_dir = fs::path(tref_file).parent_path();
    if (sp.is_relative()) {
        return (tref_dir / sp).lexically_normal().string();
    }
    if (is_regular_file_path(stored_path)) {
        return stored_path;
    }
    const fs::path sibling = (tref_dir / sp.filename()).lexically_normal();
    if (is_regular_file_path(sibling.string())) {
        return sibling.string();
    }
    return stored_path;
}

// Open the underlying TUN3 basis file referenced by a TREF state and verify basis_hash.
void open_and_verify_tref_basis(const string& tref_file, const TrefMetadata& tref,
                                const string& basis_file_override,
                                ScopedFd& out_fd, string& out_path, Tun3Metadata& out_meta)
{
    out_path = resolve_tref_basis_path(tref_file, tref.basis_path, basis_file_override);
    out_fd = ScopedFd(::open(out_path.c_str(), O_RDONLY));
    if (!out_fd) {
        LOGGER.e(0, "Thin TREF state [" + tref_file + "] references basis [" + out_path +
                    "], which cannot be opened. Pass --woodbury-basis-file <file> if the basis moved.");
    }
    struct stat sb{};
    if (::fstat(out_fd.get(), &sb) != 0 || sb.st_size < static_cast<off_t>(kRemlPrefixBytes)) {
        LOGGER.e(0, "Failed to stat referenced basis file [" + out_path + "].");
    }
    const RemlPrefix bp = read_reml_prefix(out_fd.get(), out_path);
    if (bp.kind != RemlKind::Tun3) {
        LOGGER.e(0, "Referenced basis file [" + out_path + "] for TREF [" + tref_file +
                    "] is not a compact TUN3 basis file.");
    }
    if (bp.hdr.n != tref.hdr.n || bp.id_hash != tref.id_hash) {
        LOGGER.e(0, "Basis sample mismatch between TREF [" + tref_file + "] and [" + out_path +
                    "] (n/id_hash differ).");
    }
    out_meta = read_tun3_metadata(out_fd.get(), bp, static_cast<size_t>(sb.st_size), out_path, /*load_vi_y=*/false);
    if (out_meta.k != tref.k || out_meta.basis_hash != tref.basis_hash) {
        LOGGER.e(0, "Basis hash mismatch between TREF [" + tref_file + "] (hash " + hash_hex(tref.basis_hash) +
                    ") and [" + out_path + "] (hash " + hash_hex(out_meta.basis_hash) +
                    "). Ensure --woodbury-basis-file points to the exact TUN3 basis used during --woodbury-reuse.");
    }
}

// Parallel dequantize a k x cols sample slab of TUN3 16-bit hybrid columns into float32 (k x cols).
void dequant_tun3_slab_to_f32(const uint16_t* raw_u16, int k, int cols,
                              const std::vector<float>& uk_row_scale,
                              const std::vector<int>& fp16_rows,
                              float* dst_f32,
                              int num_threads_override = 0)
{
    const float* const sc = uk_row_scale.data();
    const bool mostly_i16 = (fp16_rows.size() * 2 < static_cast<size_t>(k));
    const int use_nt = (num_threads_override > 0) ? num_threads_override : std::max(1, omp_get_max_threads());
    #pragma omp parallel for schedule(static) num_threads(use_nt)
    for (int c = 0; c < cols; ++c) {
        const uint16_t* const src = raw_u16 + static_cast<size_t>(c) * static_cast<size_t>(k);
        float* const dst          = dst_f32 + static_cast<size_t>(c) * static_cast<size_t>(k);
        if (mostly_i16) {
            #pragma omp simd
            for (int r = 0; r < k; ++r) {
                dst[r] = static_cast<float>(static_cast<int16_t>(src[r])) * sc[r];
            }
            for (int r : fp16_rows) {
                dst[r] = wb_fp16_bits_to_f32(src[r]);
            }
        } else {
            for (int r = 0; r < k; ++r) {
                const float s = sc[r];
                dst[r] = (s > 0.0f)
                    ? (static_cast<float>(static_cast<int16_t>(src[r])) * s)
                    : ((s < 0.0f) ? wb_fp16_bits_to_f32(src[r]) : 0.0f);
            }
        }
    }
}

// Build sample-aligned chunks (each <= target_bytes, whole columns of size col_bytes)
// starting at `body_offset`.
struct SampleChunk {
    off_t  offset = 0;
    size_t bytes  = 0;
    int    s_beg  = 0;
    int    s_end  = 0;
};

std::vector<SampleChunk> plan_sample_chunks(off_t body_offset, int n, size_t col_bytes,
                                            size_t target_bytes)
{
    const int cols_per = static_cast<int>(
        std::max<size_t>(1, target_bytes / std::max<size_t>(1, col_bytes)));
    std::vector<SampleChunk> out;
    out.reserve(static_cast<size_t>((n + cols_per - 1) / cols_per));
    off_t cur_off = body_offset;
    for (int s0 = 0; s0 < n; s0 += cols_per) {
        const int s1 = std::min(n, s0 + cols_per);
        const size_t bytes = static_cast<size_t>(s1 - s0) * col_bytes;
        out.push_back(SampleChunk{cur_off, bytes, s0, s1});
        cur_off += static_cast<off_t>(bytes);
    }
    return out;
}

// Read the binary REML state written by writeRemlStateFromCtx().
// Supports "TUN3", "TREF" (-> referenced "TUN3"), and "GOBY".
// When stream_fast_mode != Off on a Woodbury state, streams 16-bit Uk directly
// into QuantizedWoodburyUk in ONE pass via AsyncSeqReader without allocating Uk_f.
RemlState readRemlState(const string& filename, bool no_adj_covar,
                        int expected_n, int expected_xc, uint64_t expected_id_hash,
                        const string& basis_file_override,
                        FastMlmaMode stream_fast_mode,
                        const Eigen::VectorXf& y_raw_f,
                        const Eigen::MatrixXd& X_design,
                        const Eigen::VectorXf& w_sqrt,
                        size_t tun3_slot_bytes = (256ULL << 20),
                        int max_q_rows = 4096)
{
    ScopedFd fd(::open(filename.c_str(), O_RDONLY));
    if (!fd) LOGGER.e(0, "Cannot open file [" + filename + "].");

    struct stat sb{};
    if (::fstat(fd.get(), &sb) != 0) {
        LOGGER.e(0, "Failed to stat file [" + filename + "].");
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);
    if (file_size < kRemlPrefixBytes) {
        LOGGER.e(0, "[" + filename + "] is empty or too short to be a REML state.");
    }

    const RemlPrefix prefix = read_reml_prefix(fd.get(), filename);
    check_reml_samples(prefix, expected_n, expected_id_hash, filename);

    const Header& hdr = prefix.hdr;
    if (hdr.n <= 0 || hdr.x_c < 0 || hdr.num_varcmp <= 0) {
        LOGGER.e(0, "[" + filename + "] has invalid header dimensions.");
    }
    if (expected_xc >= 0 && hdr.x_c != expected_xc) {
        LOGGER.e(0, "Covariate count mismatch: REML state x_c=" + to_string(hdr.x_c) +
                    " vs current x_c=" + to_string(expected_xc) +
                    ". Use the same --qcovar/--covar as during --save-reml.");
    }

    RemlState st;
    st.n   = hdr.n;
    st.x_c = hdr.x_c;

    // ------------------------------------------------------------------ GOBY
    if (prefix.kind == RemlKind::Goby) {
        int32_t factor_kind = 0;
        try {
            gcta_io::read_exact_throwing(fd.get(), &factor_kind, sizeof(int32_t), filename);
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }
        st.is_llt = (factor_kind == 0);
        st.Vi_L_f.resize(hdr.n, hdr.n);

        const size_t tri = static_cast<size_t>(hdr.n) * (hdr.n + 1) / 2;
        constexpr size_t kGobySlotBytes = 64ULL << 20; // 64 MiB double-buffered
        const off_t body_off = static_cast<off_t>(kRemlPrefixBytes + sizeof(int32_t));

        struct ColChunk { int32_t j_beg, j_end; };
        std::vector<std::pair<off_t, size_t>> io_chunks;
        std::vector<ColChunk> col_chunks;
        size_t max_chunk_bytes = 0;
        off_t cur_off = body_off;
        for (int32_t j = 0; j < hdr.n; ) {
            int32_t j_end = j;
            size_t chunk_bytes = 0;
            while (j_end < hdr.n) {
                const size_t col_bytes = static_cast<size_t>(j_end + 1) * sizeof(float);
                if (chunk_bytes > 0 && chunk_bytes + col_bytes > kGobySlotBytes) break;
                chunk_bytes += col_bytes;
                ++j_end;
            }
            io_chunks.emplace_back(cur_off, chunk_bytes);
            col_chunks.push_back(ColChunk{j, j_end});
            max_chunk_bytes = std::max(max_chunk_bytes, chunk_bytes);
            cur_off += static_cast<off_t>(chunk_bytes);
            j = j_end;
        }

        LOGGER.i(0, "Reading [" + filename + "]'s packed Vi_L block via double-buffered 64 MiB async stream.");
        try {
            gcta_io::AsyncSeqReader reader(fd.get(), filename, std::move(io_chunks), max_chunk_bytes, false);
            const char* data = nullptr;
            size_t len = 0;
            size_t ci = 0;
            while (reader.next_slot(data, len)) {
                const int32_t j_beg = col_chunks[ci].j_beg;
                const int32_t j_end = col_chunks[ci].j_end;
                const size_t chunk_start_elem = static_cast<size_t>(j_beg) * (j_beg + 1) / 2;
                #pragma omp parallel for schedule(guided)
                for (int32_t jj = j_beg; jj < j_end; ++jj) {
                    const int32_t col_len = jj + 1;
                    const size_t col_elem = static_cast<size_t>(jj) * (jj + 1) / 2;
                    const size_t local_off = (col_elem - chunk_start_elem) * sizeof(float);
                    std::memcpy(st.Vi_L_f.col(jj).head(col_len).data(),
                                data + local_off,
                                static_cast<size_t>(col_len) * sizeof(float));
                }
                reader.release_slot();
                ++ci;
            }
            reader.stop();

            off_t tail_off = body_off + static_cast<off_t>(tri * sizeof(float));
            if (!no_adj_covar) {
                st.b.resize(hdr.x_c);
                gcta_io::read_exact_at_throwing(fd.get(), st.b.data(),
                    static_cast<size_t>(hdr.x_c) * sizeof(float), tail_off, filename);
                tail_off += static_cast<off_t>(static_cast<size_t>(hdr.x_c) * sizeof(float));
            }
            st.varcmp.resize(hdr.num_varcmp);
            gcta_io::read_exact_at_throwing(fd.get(), st.varcmp.data(),
                static_cast<size_t>(hdr.num_varcmp) * sizeof(float), tail_off, filename);
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }
        return st;
    }

    // ---------------------------------------------------------- TUN3 / TREF
    st.is_woodbury = true;
    ScopedFd tun3_fd_owner;
    int tun3_fd = -1;
    string tun3_path;
    Tun3Metadata basis_meta;

    uint32_t state_flags = 0;
    uint64_t stored_y_hash = 0;
    Eigen::VectorXf stored_vi_y;

    if (prefix.kind == RemlKind::Tref) {
        TrefMetadata tref = read_tref_metadata(fd.get(), prefix, filename, /*load_vi_y=*/true);
        st.dk_f          = std::move(tref.dk_f);
        st.lambda_tail_f = static_cast<float>(tref.lambda_tail);
        if (!no_adj_covar) st.b = std::move(tref.b_f);
        st.varcmp        = std::move(tref.varcmp_f);
        state_flags      = tref.flags;
        stored_y_hash    = tref.y_adj_hash;
        stored_vi_y      = std::move(tref.vi_y_f);

        open_and_verify_tref_basis(filename, tref, basis_file_override,
                                   tun3_fd_owner, tun3_path, basis_meta);
        tun3_fd = tun3_fd_owner.get();
        LOGGER.i(0, "Resolved thin TREF [" + filename + "] -> compact basis [" + tun3_path +
                    "] (hash " + hash_hex(basis_meta.basis_hash) + ").");
    } else {
        basis_meta       = read_tun3_metadata(fd.get(), prefix, file_size, filename, /*load_vi_y=*/true);
        st.dk_f          = basis_meta.dk_f;
        st.lambda_tail_f = static_cast<float>(basis_meta.lambda_tail);
        if (!no_adj_covar) st.b = basis_meta.b_f;
        st.varcmp        = basis_meta.varcmp_f;
        state_flags      = basis_meta.flags;
        stored_y_hash    = basis_meta.y_adj_hash;
        stored_vi_y      = std::move(basis_meta.vi_y_f);
        tun3_fd          = fd.get();
        tun3_path        = filename;
    }

    const int32_t k = basis_meta.k;
    const int32_t n = hdr.n;
    const double lambda_tail = basis_meta.lambda_tail;
    compute_woodbury_ck_from_saved(st.dk_f, lambda_tail, st.varcmp,
                                   st.wb.ck_f, st.wb.sqrt_ck_f, st.wb.sigma2_eff_f);

    const Eigen::VectorXf y_adj_f = (!no_adj_covar && st.b.size() == expected_xc)
        ? compute_y_adj_deterministic(y_raw_f, X_design, st.b)
        : y_raw_f;
    const uint64_t cur_y_hash = hash_bytes(y_adj_f.data(), static_cast<size_t>(n) * sizeof(float));
    const bool vi_y_hit = ((state_flags & kTun3FlagHasViY) != 0u) &&
                          (stored_vi_y.size() == n) &&
                          (cur_y_hash == stored_y_hash);
    if (vi_y_hit) {
        st.cached_Vi_y = std::move(stored_vi_y);
        st.has_cached_Vi_y = true;
    }

    const size_t kTun3SlotBytes = std::clamp<size_t>(tun3_slot_bytes, 1ULL << 20, 512ULL << 20);
    const size_t col_bytes = static_cast<size_t>(k) * sizeof(uint16_t);
    const std::vector<SampleChunk> plan = plan_sample_chunks(basis_meta.body_offset, n, col_bytes, kTun3SlotBytes);
    const int unpack_nt = std::max(1, omp_get_max_threads() - 1);

    // Path 1: 1-pass streaming quantization directly into QuantizedWoodburyUk (0 Uk_f RAM!).
    if (stream_fast_mode != FastMlmaMode::Off) {
        bool unit_weights = true;
        double S_w2 = static_cast<double>(n);
        for (int s = 0; s < n; ++s) {
            if (std::abs(w_sqrt[s] - 1.0f) > 1.0e-7f) {
                unit_weights = false;
                break;
            }
        }
        std::vector<float> exact_max_w;
        std::vector<double> pre_ck_uy;
        bool need_w_max_pass = false;
        float w_spread_ratio = 1.0f;
        if (!unit_weights) {
            S_w2 = w_sqrt.cast<double>().array().square().sum();
            if (stream_fast_mode == FastMlmaMode::Slice1 || stream_fast_mode == FastMlmaMode::Hybrid) {
                std::vector<float> abs_w(static_cast<size_t>(n));
                for (int s = 0; s < n; ++s) abs_w[static_cast<size_t>(s)] = std::fabs(w_sqrt[s]);
                const float w_max = *std::max_element(abs_w.begin(), abs_w.end());
                std::nth_element(abs_w.begin(), abs_w.begin() + n / 2, abs_w.end());
                const float w_med = abs_w[static_cast<size_t>(n / 2)];
                if (w_med > 0.0f && w_max > 2.0f * w_med) {
                    need_w_max_pass = true;
                    w_spread_ratio = w_max / w_med;
                }
            }
        }
        // When Vi_y cache misses under !unit_weights or 1slice/hybrid, a lightweight streaming
        // pre-pass computes exact u_y (and exact weighted max_abs) so Vi_y is reconstructed
        // directly from 15-bit TUN3 rather than 8-bit Q or 1/w_s division.
        const bool need_exact_vi_y_pass = (!vi_y_hit) &&
            (!unit_weights || stream_fast_mode != FastMlmaMode::Slice2);
        const bool need_prepass = need_w_max_pass || need_exact_vi_y_pass;

        if (need_prepass) {
            if (need_w_max_pass) {
                LOGGER.i(0, "Weighted phenotype has wide weight spread (max/median=" +
                            to_text(w_spread_ratio) + "); running streaming max_abs pre-pass over TUN3.");
            }
            if (!unit_weights) exact_max_w.assign(static_cast<size_t>(k), 0.0f);
            std::vector<std::pair<off_t, size_t>> p1_chunks;
            p1_chunks.reserve(plan.size());
            size_t max_b = 0;
            for (const auto& c : plan) {
                p1_chunks.emplace_back(c.offset, c.bytes);
                max_b = std::max(max_b, c.bytes);
            }
            const int max_t = unpack_nt;
            const size_t k_pad_f = (static_cast<size_t>(k) + 15) / 16 * 16;
            const size_t k_pad_d = (static_cast<size_t>(k) + 7) / 8 * 8;
            std::vector<float> thr_max;
            if (!unit_weights) thr_max.assign(static_cast<size_t>(max_t) * k_pad_f, 0.0f);
            std::vector<double> thr_uy;
            if (need_exact_vi_y_pass) thr_uy.assign(static_cast<size_t>(max_t) * k_pad_d, 0.0);
            try {
                gcta_io::AsyncSeqReader p1_reader(tun3_fd, tun3_path, std::move(p1_chunks), max_b, false);
                const char* data = nullptr;
                size_t len = 0;
                size_t ci = 0;
                while (p1_reader.next_slot(data, len)) {
                    const int s_beg = plan[ci].s_beg;
                    const int s_end = plan[ci].s_end;
                    const uint16_t* raw = reinterpret_cast<const uint16_t*>(data);
                    const float* rs = basis_meta.uk_row_scale.data();
                    const float* wp = w_sqrt.data();
                    const float* yp = y_adj_f.data();
                    #pragma omp parallel num_threads(max_t)
                    {
                        const int tid = omp_get_thread_num();
                        float* lmax = thr_max.empty() ? nullptr : (thr_max.data() + static_cast<size_t>(tid) * k_pad_f);
                        double* luy = thr_uy.empty() ? nullptr : (thr_uy.data() + static_cast<size_t>(tid) * k_pad_d);
                        #pragma omp for schedule(static)
                        for (int s = s_beg; s < s_end; ++s) {
                            const uint16_t* rc = raw + static_cast<size_t>(s - s_beg) * static_cast<size_t>(k);
                            const float ws = lmax ? std::fabs(wp[s]) : 1.0f;
                            const double ys = luy ? static_cast<double>(yp[s]) : 0.0;
                            for (int r = 0; r < k; ++r) {
                                const float sc = rs[r];
                                const float u = (sc > 0.0f)
                                    ? (static_cast<float>(static_cast<int16_t>(rc[r])) * sc)
                                    : ((sc < 0.0f) ? wb_fp16_bits_to_f32(rc[r]) : 0.0f);
                                if (lmax) {
                                    const float av = std::fabs(u) * ws;
                                    if (av > lmax[r]) lmax[r] = av;
                                }
                                if (luy) {
                                    luy[r] += static_cast<double>(u) * ys;
                                }
                            }
                        }
                    }
                    p1_reader.release_slot();
                    ++ci;
                }
                p1_reader.stop();
            } catch (const std::exception& e) {
                LOGGER.e(0, e.what());
            }
            if (!unit_weights) {
                for (int r = 0; r < k; ++r) {
                    float m = 0.0f;
                    for (int t = 0; t < max_t; ++t) {
                        m = std::max(m, thr_max[static_cast<size_t>(t) * k_pad_f + static_cast<size_t>(r)]);
                    }
                    exact_max_w[static_cast<size_t>(r)] = m;
                }
            }
            if (need_exact_vi_y_pass) {
                pre_ck_uy.assign(static_cast<size_t>(k), 0.0);
                for (int r = 0; r < k; ++r) {
                    double acc = 0.0;
                    for (int t = 0; t < max_t; ++t) {
                        acc += thr_uy[static_cast<size_t>(t) * k_pad_d + static_cast<size_t>(r)];
                    }
                    pre_ck_uy[static_cast<size_t>(r)] = static_cast<double>(st.wb.ck_f[r]) * acc;
                }
            }
        }

        const bool defer_uy_reconstruct = (!vi_y_hit && !need_exact_vi_y_pass);
        QuantizedWoodburyStreamBuilder builder(
            k, n, stream_fast_mode,
            st.wb.sqrt_ck_f.data(),
            basis_meta.uk_row_scale.data(),
            w_sqrt, unit_weights, S_w2,
            exact_max_w.empty() ? nullptr : exact_max_w.data(),
            defer_uy_reconstruct ? y_adj_f.data() : nullptr,
            max_q_rows);
        if (need_exact_vi_y_pass) {
            st.cached_Vi_y.resize(n);
            builder.set_exact_vi_y_target(pre_ck_uy.data(), y_adj_f.data(),
                                          st.wb.sigma2_eff_f, st.cached_Vi_y.data());
        }

        std::vector<std::pair<off_t, size_t>> io_chunks;
        io_chunks.reserve(plan.size());
        size_t max_b = 0;
        for (const auto& c : plan) {
            io_chunks.emplace_back(c.offset, c.bytes);
            max_b = std::max(max_b, c.bytes);
        }

        std::vector<float> row_eps_sq(static_cast<size_t>(k), 0.0f);
        try {
            gcta_io::AsyncSeqReader reader(tun3_fd, tun3_path, std::move(io_chunks), max_b, false);
            const char* data = nullptr;
            size_t len = 0;
            size_t ci = 0;
            while (reader.next_slot(data, len)) {
                builder.process_sample_slab(plan[ci].s_beg, plan[ci].s_end,
                                            reinterpret_cast<const uint16_t*>(data),
                                            unpack_nt);
                reader.release_slot();
                ++ci;
            }
            reader.stop();

            const off_t trailer_off = basis_meta.body_offset +
                static_cast<off_t>(static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(uint16_t));
            gcta_io::read_exact_at_throwing(tun3_fd, row_eps_sq.data(),
                static_cast<size_t>(k) * sizeof(float), trailer_off, tun3_path);
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }

        QuantizedWoodburyUk q_uk = builder.finalize(row_eps_sq.data());
        if (!vi_y_hit) {
            LOGGER.i(0, "Phenotype/covariate adjustment differs from saved .reml Vi_y cache; "
                        "reconstructing V^{-1} y_adj from streamed basis.");
            if (defer_uy_reconstruct) {
                st.cached_Vi_y = reconstruct_vi_y_from_q_uk(
                    q_uk, st.wb.sqrt_ck_f, st.wb.sigma2_eff_f,
                    y_adj_f, w_sqrt, unit_weights, builder.reduced_uy());
            }
            st.has_cached_Vi_y = true;
        }
        st.prebuilt_q_uk = std::move(q_uk);
        return st;
    }

    // Path 2: Exact (non-fast-mlma) load — stream 16-bit TUN3 and dequantize in parallel into Uk_f (k x n).
    std::vector<int> fp16_rows;
    for (int r = 0; r < k; ++r) {
        if (basis_meta.uk_row_scale[static_cast<size_t>(r)] < 0.0f) fp16_rows.push_back(r);
    }
    Eigen::MatrixXf Uk(k, n);
    std::vector<std::pair<off_t, size_t>> io_chunks;
    io_chunks.reserve(plan.size());
    size_t max_b = 0;
    for (const auto& c : plan) {
        io_chunks.emplace_back(c.offset, c.bytes);
        max_b = std::max(max_b, c.bytes);
    }
    try {
        gcta_io::AsyncSeqReader reader(tun3_fd, tun3_path, std::move(io_chunks), max_b, false);
        const char* data = nullptr;
        size_t len = 0;
        size_t ci = 0;
        while (reader.next_slot(data, len)) {
            const int s_beg = plan[ci].s_beg;
            const int cols  = plan[ci].s_end - s_beg;
            dequant_tun3_slab_to_f32(
                reinterpret_cast<const uint16_t*>(data), k, cols,
                basis_meta.uk_row_scale, fp16_rows,
                Uk.data() + static_cast<size_t>(s_beg) * static_cast<size_t>(k),
                unpack_nt);
            reader.release_slot();
            ++ci;
        }
        reader.stop();
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }
    st.wb.Uk_f.swap(Uk);
    return st;
}

struct StreamingReuseHandle {
    string           tun3_path;
    uint64_t         basis_hash = 0;
    Tun3Metadata     meta;
    std::vector<int> fp16_rows;
    size_t           io_slot_bytes = 256ULL << 20;
    bool             active = false;
};

// Pass 3 of zero-Uk streaming TREF reuse: stream 16-bit TUN3 slabs once through a
// reusable <= 256 MiB float32 slab to compute proj_y = Uk^T * uy_scaled_f and return
// vi_y_f = (y_adj_f - proj_y) / sig2e_f without ever allocating 4*k*n ctx.Uk.
Eigen::VectorXf compute_vi_y_streaming_tun3(const StreamingReuseHandle& h,
                                            int n,
                                            const Eigen::VectorXf& y_adj_f,
                                            const Eigen::VectorXf& uy_scaled_f,
                                            float sig2e_f)
{
    ScopedFd fd(::open(h.tun3_path.c_str(), O_RDONLY));
    if (!fd) {
        LOGGER.e(0, "Cannot re-open TUN3 basis [" + h.tun3_path + "] for TREF Vi_y pass.");
    }
    const int32_t k = h.meta.k;
    const size_t col_bytes = static_cast<size_t>(k) * sizeof(uint16_t);
    const size_t u16_slot_bytes = std::clamp<size_t>(h.io_slot_bytes, 1ULL << 20, 128ULL << 20);
    const std::vector<SampleChunk> plan =
        plan_sample_chunks(h.meta.body_offset, n, col_bytes, u16_slot_bytes);
    std::vector<std::pair<off_t, size_t>> io_chunks;
    io_chunks.reserve(plan.size());
    size_t max_b = 0;
    int max_cols = 1;
    for (const auto& c : plan) {
        io_chunks.emplace_back(c.offset, c.bytes);
        max_b = std::max(max_b, c.bytes);
        max_cols = std::max(max_cols, c.s_end - c.s_beg);
    }

    AlignedRawBuffer<float, 64> slab_f32(static_cast<size_t>(k) * static_cast<size_t>(max_cols));
    Eigen::VectorXf proj_y = Eigen::VectorXf::Zero(n);
    const int unpack_nt = std::max(1, omp_get_max_threads() - 1);

    try {
        gcta_io::AsyncSeqReader reader(fd.get(), h.tun3_path, std::move(io_chunks), max_b, false);
        const char* data = nullptr;
        size_t len = 0;
        size_t ci = 0;
        while (reader.next_slot(data, len)) {
            const int s_beg = plan[ci].s_beg;
            const int cols  = plan[ci].s_end - s_beg;
            dequant_tun3_slab_to_f32(
                reinterpret_cast<const uint16_t*>(data), k, cols,
                h.meta.uk_row_scale, h.fp16_rows, slab_f32.data(), unpack_nt);
            reader.release_slot();
            cblas_sgemv(CblasColMajor, CblasTrans,
                        static_cast<gcta_blas_int>(k), static_cast<gcta_blas_int>(cols),
                        1.0f, slab_f32.data(), static_cast<gcta_blas_int>(k),
                        uy_scaled_f.data(), 1,
                        0.0f, proj_y.data() + s_beg, 1);
            ++ci;
        }
        reader.stop();
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }
    return (y_adj_f - proj_y) / sig2e_f;
}

// Writes TUN3 (compact 16-bit Woodbury), TREF (thin reference on --woodbury-reuse
// unless reml_standalone=true), or GOBY (exact dense float32 Cholesky).
// Note: invoked only on --save-reml (which exits immediately), so it may release ctx.Uk
// or factorize ctx.Vi in place.
void writeRemlStateFromCtx(const string& filename, RemlCtx& ctx, bool no_adj_covar,
                           uint64_t id_hash,
                           const string& reuse_basis_tun3_path = {},
                           uint64_t reuse_basis_hash = 0,
                           bool reml_standalone = false,
                           const StreamingReuseHandle* streaming_reuse = nullptr,
                           size_t io_slot_bytes = (256ULL << 20))
{
    ScopedFd fd(::open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644));
    if (!fd) {
        LOGGER.e(0, "cannot open [" + filename + "] for writing: " + string(std::strerror(errno)));
    }

    auto wr = [&](const void* buf, size_t bytes) {
        try {
            gcta_io::write_exact_throwing(fd.get(), buf, bytes, filename);
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }
    };

    if (ctx.Vi_use_woodbury_basis) {
        const int32_t n = static_cast<int32_t>(ctx.n);
        const int32_t k = static_cast<int32_t>(ctx.dk.size());
        const bool is_streaming_reuse = (streaming_reuse != nullptr && streaming_reuse->active);
        const bool uk_ok = is_streaming_reuse
            ? (ctx.wb_subspace_ready && ctx.UkTX.rows() == k && static_cast<int32_t>(ctx.UkTy.size()) == k)
            : (ctx.Uk_is_transposed
                ? (ctx.Uk.rows() == k && ctx.Uk.cols() == n)
                : (ctx.Uk.rows() == n && ctx.Uk.cols() >= k));
        if (!uk_ok) {
            LOGGER.e(0, "invalid Woodbury REML context dimensions before save.");
        }

        const Eigen::VectorXf dk_f = ctx.dk.cast<float>();
        const Eigen::VectorXf b_f  = (!no_adj_covar && ctx.X_c > 0) ? ctx.b.cast<float>() : Eigen::VectorXf();
        const Eigen::VectorXf varcmp_f = Eigen::Map<const Eigen::VectorXd>(
            ctx.varcmp.data(), static_cast<Eigen::Index>(ctx.varcmp.size())).cast<float>();

        // Precompute exact float32 Vi_y + deterministic y_adj_hash from final saved parameters.
        const Eigen::VectorXf y_raw_f = ctx.y.cast<float>();
        const Eigen::VectorXf y_adj_f = (!no_adj_covar && b_f.size() == ctx.X_c)
            ? compute_y_adj_deterministic(y_raw_f, ctx.X, b_f)
            : y_raw_f;
        const uint64_t y_adj_hash = hash_bytes(y_adj_f.data(), static_cast<size_t>(n) * sizeof(float));

        Eigen::VectorXf ck_f, sqrt_ck_f;
        float sig2e_f = 0.0f;
        compute_woodbury_ck_from_saved(dk_f, ctx.lambda_tail, varcmp_f, ck_f, sqrt_ck_f, sig2e_f);

        std::vector<float> max_abs;
        bool have_max_abs = false;
        Eigen::VectorXf vi_y_f;
        if (is_streaming_reuse) {
            RemlVec uy_d = ctx.UkTy;
            if (!no_adj_covar && ctx.X_c > 0) {
                uy_d.noalias() -= ctx.UkTX * ctx.b;
            }
            const Eigen::VectorXf uy_scaled = uy_d.cast<float>().cwiseProduct(ck_f);
            vi_y_f = compute_vi_y_streaming_tun3(*streaming_reuse, n, y_adj_f, uy_scaled, sig2e_f);
        } else if (ctx.Uk_is_transposed) {
            Eigen::VectorXf uy = ctx.Uk * y_adj_f;
            uy.array() *= ck_f.array();
            vi_y_f = (y_adj_f - ctx.Uk.transpose() * uy) / sig2e_f;
        } else {
            // Fuse max_abs[r] and uy[r] = dot(Uk[:,r], y_adj_f) in a single pass over Uk.
            max_abs.assign(static_cast<size_t>(k), 0.0f);
            Eigen::VectorXf uy(k);
            const float* const uk_src = ctx.Uk.data();
            const float* const y_adj_ptr = y_adj_f.data();
            #pragma omp parallel for schedule(static)
            for (int32_t r = 0; r < k; ++r) {
                const float* const col = uk_src + static_cast<size_t>(r) * static_cast<size_t>(n);
                float m = 0.0f;
                double dot = 0.0;
                #pragma omp simd reduction(max:m) reduction(+:dot)
                for (int32_t s = 0; s < n; ++s) {
                    const float v = col[s];
                    const float av = std::fabs(v);
                    if (av > m) m = av;
                    dot += static_cast<double>(v) * static_cast<double>(y_adj_ptr[s]);
                }
                max_abs[static_cast<size_t>(r)] = m;
                uy[r] = static_cast<float>(dot);
            }
            have_max_abs = true;
            uy.array() *= ck_f.array();
            vi_y_f = (y_adj_f - ctx.Uk.leftCols(k) * uy) / sig2e_f;
        }

        const uint32_t flags = kTun3FlagHasViY | (no_adj_covar ? kTun3FlagNoAdjCovar : 0u);

        // --- Thin TREF Branch (--woodbury-reuse --save-reml without --reml-standalone) ---
        if (!reuse_basis_tun3_path.empty() && reuse_basis_hash != 0 && !reml_standalone) {
            Header hdr{};
            std::memcpy(hdr.magic, kMagicTref.data(), 4);
            hdr.n          = n;
            hdr.x_c        = static_cast<int32_t>(ctx.X_c);
            hdr.num_varcmp = static_cast<int32_t>(ctx.varcmp.size());
            hdr.num_r_indx = static_cast<int32_t>(ctx.varcmp.size());

            const string stored_path =
                std::filesystem::absolute(reuse_basis_tun3_path).lexically_normal().string();
            const uint32_t path_len = static_cast<uint32_t>(stored_path.size());

            wr(&hdr, sizeof(hdr));
            wr(&id_hash, sizeof(uint64_t));
            wr(&reuse_basis_hash, sizeof(uint64_t));
            wr(&flags, sizeof(uint32_t));
            wr(&path_len, sizeof(uint32_t));
            wr(stored_path.data(), path_len);
            wr(&k, sizeof(int32_t));
            wr(&ctx.lambda_tail, sizeof(double));
            wr(&ctx.tail_d_var, sizeof(double));
            wr(dk_f.data(), static_cast<size_t>(k) * sizeof(float));
            if (!no_adj_covar && ctx.X_c > 0) {
                wr(b_f.data(), static_cast<size_t>(ctx.X_c) * sizeof(float));
            }
            wr(varcmp_f.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float));
            wr(&y_adj_hash, sizeof(uint64_t));
            wr(vi_y_f.data(), static_cast<size_t>(n) * sizeof(float));
            ctx.Uk.resize(0, 0);

            const double mb = static_cast<double>(
                kRemlPrefixBytes + 16 + path_len + 20 +
                (static_cast<size_t>(k) + static_cast<size_t>(ctx.X_c) +
                 static_cast<size_t>(hdr.num_varcmp) + static_cast<size_t>(n)) * sizeof(float)) / (1024.0 * 1024.0);
            LOGGER.i(0, "Saved thin TREF REML state (" + std::format("{:.2f}", mb) +
                        " MiB) referencing basis [" + stored_path + "] (hash " +
                        hash_hex(reuse_basis_hash) + ").");
            return;
        }

        // --- Compact 16-bit TUN3 Branch ---
        const float* const uk_src = ctx.Uk.data();
        if (!have_max_abs) {
            max_abs.assign(static_cast<size_t>(k), 0.0f);
            if (!ctx.Uk_is_transposed) {
                #pragma omp parallel for schedule(static)
                for (int32_t r = 0; r < k; ++r) {
                    const float* const col = uk_src + static_cast<size_t>(r) * static_cast<size_t>(n);
                    float m = 0.0f;
                    #pragma omp simd reduction(max:m)
                    for (int32_t s = 0; s < n; ++s) {
                        const float av = std::fabs(col[s]);
                        if (av > m) m = av;
                    }
                    max_abs[static_cast<size_t>(r)] = m;
                }
            } else {
                const int max_t = std::max(1, omp_get_max_threads());
                const size_t k_pad = (static_cast<size_t>(k) + 15) / 16 * 16;
                std::vector<float> thr_max(static_cast<size_t>(max_t) * k_pad, 0.0f);
                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    float* const lmax = thr_max.data() + static_cast<size_t>(tid) * k_pad;
                    #pragma omp for schedule(static)
                    for (int32_t s = 0; s < n; ++s) {
                        const float* const col = uk_src + static_cast<size_t>(s) * static_cast<size_t>(k);
                        #pragma omp simd
                        for (int32_t r = 0; r < k; ++r) {
                            const float av = std::fabs(col[r]);
                            if (av > lmax[r]) lmax[r] = av;
                        }
                    }
                }
                for (int32_t r = 0; r < k; ++r) {
                    float m = 0.0f;
                    for (int t = 0; t < max_t; ++t) {
                        m = std::max(m, thr_max[static_cast<size_t>(t) * k_pad + static_cast<size_t>(r)]);
                    }
                    max_abs[static_cast<size_t>(r)] = m;
                }
            }
        }

        std::vector<float> uk_row_scale(static_cast<size_t>(k), 0.0f);
        std::vector<float> inv_row_scale(static_cast<size_t>(k), 0.0f);
        const double sqrt_n = std::sqrt(static_cast<double>(n));
        int fp16_row_cnt = 0;
        for (int32_t r = 0; r < k; ++r) {
            const float m = max_abs[static_cast<size_t>(r)];
            if (m <= 1.0e-30f) {
                uk_row_scale[static_cast<size_t>(r)] = 0.0f;
            } else if (sqrt_n * static_cast<double>(m) <= kTun3CrestFp16Thresh) {
                uk_row_scale[static_cast<size_t>(r)]  = m / kTun3Int16QMax;
                inv_row_scale[static_cast<size_t>(r)] = kTun3Int16QMax / m;
            } else {
                uk_row_scale[static_cast<size_t>(r)] = -m; // negative => IEEE FP16 bit-pattern
                ++fp16_row_cnt;
            }
        }

        Header hdr{};
        std::memcpy(hdr.magic, kMagicTun3.data(), 4);
        hdr.n          = n;
        hdr.x_c        = static_cast<int32_t>(ctx.X_c);
        hdr.num_varcmp = static_cast<int32_t>(ctx.varcmp.size());
        hdr.num_r_indx = static_cast<int32_t>(ctx.varcmp.size());

        // Write prefix + full metadata header sequentially BEFORE Uk.
        wr(&hdr, sizeof(hdr));
        wr(&id_hash, sizeof(uint64_t));
        wr(&flags, sizeof(uint32_t));
        wr(&k, sizeof(int32_t));
        wr(&ctx.lambda_tail, sizeof(double));
        wr(&ctx.tail_d_var, sizeof(double));
        wr(dk_f.data(), static_cast<size_t>(k) * sizeof(float));
        if (!no_adj_covar && ctx.X_c > 0) {
            wr(b_f.data(), static_cast<size_t>(ctx.X_c) * sizeof(float));
        }
        wr(varcmp_f.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float));
        wr(uk_row_scale.data(), static_cast<size_t>(k) * sizeof(float));
        wr(&y_adj_hash, sizeof(uint64_t));
        wr(vi_y_f.data(), static_cast<size_t>(n) * sizeof(float));

        // Stream k x n 16-bit Uk body via double-buffered AsyncSeqWriter (256 MiB default, T-1 workers).
        const size_t kTun3WriteSlotBytes = std::clamp<size_t>(io_slot_bytes, 1ULL << 20, 256ULL << 20);
        const size_t col_bytes = static_cast<size_t>(k) * sizeof(uint16_t);
        const int32_t cols_per_chunk = static_cast<int32_t>(
            std::max<size_t>(1, kTun3WriteSlotBytes / std::max<size_t>(1, col_bytes)));
        const size_t init_slot_bytes = std::max(kTun3WriteSlotBytes, col_bytes);

        const int io_nt = std::max(1, omp_get_max_threads() - 1);
        const size_t k_pad64 = (static_cast<size_t>(k) + 7) / 8 * 8;
        std::vector<double> thr_eps(static_cast<size_t>(io_nt) * k_pad64, 0.0);

        const float* const sc_ptr  = uk_row_scale.data();
        const float* const isc_ptr = inv_row_scale.data();
        const size_t n_stride = static_cast<size_t>(n);
        const size_t k_stride = static_cast<size_t>(k);

        try {
            gcta_io::AsyncSeqWriter writer(fd.get(), filename, init_slot_bytes, /*take_ownership=*/false);

            for (int32_t i0 = 0; i0 < n; i0 += cols_per_chunk) {
                const int32_t i_len = std::min(cols_per_chunk, n - i0);
                const size_t chunk_bytes = static_cast<size_t>(i_len) * col_bytes;
                AlignedRawBuffer<char, 64>& slot = writer.acquire_slot();
                if (slot.size() < chunk_bytes) slot.resize(chunk_bytes);
                uint16_t* const dst_u16 = reinterpret_cast<uint16_t*>(slot.data());

                if (ctx.Uk_is_transposed) {
                    // Already k x n: quantize columns directly with zero transpose!
                    #pragma omp parallel num_threads(io_nt)
                    {
                        const int tid = omp_get_thread_num();
                        double* const my_eps = thr_eps.data() + static_cast<size_t>(tid) * k_pad64;
                        #pragma omp for schedule(static)
                        for (int32_t di = 0; di < i_len; ++di) {
                            const float* const src_col = uk_src + static_cast<size_t>(i0 + di) * k_stride;
                            uint16_t* const dst_col    = dst_u16 + static_cast<size_t>(di) * k_stride;
                            for (int32_t r = 0; r < k; ++r) {
                                const float sc = sc_ptr[r];
                                const float v  = src_col[r];
                                if (sc > 0.0f) {
                                    const float qf = std::min(kTun3Int16QMax, std::max(-kTun3Int16QMax,
                                        std::nearbyintf(v * isc_ptr[r])));
                                    dst_col[r] = static_cast<uint16_t>(static_cast<int16_t>(qf));
                                    const double err = static_cast<double>(v) - static_cast<double>(qf) * static_cast<double>(sc);
                                    my_eps[r] += err * err;
                                } else if (sc < 0.0f) {
                                    const uint16_t h = wb_f32_to_fp16_bits(v);
                                    dst_col[r] = h;
                                    const double err = static_cast<double>(v) - static_cast<double>(wb_fp16_bits_to_f32(h));
                                    my_eps[r] += err * err;
                                } else {
                                    dst_col[r] = 0u;
                                }
                            }
                        }
                    }
                } else {
                    // Cache-blocked 64x64 transpose + 16-bit quantization inside L1D.
                    constexpr int32_t kBlk = 64;
                    const int32_t ni_tiles = (i_len + kBlk - 1) / kBlk;
                    const int32_t nj_tiles = (k + kBlk - 1) / kBlk;
                    const int64_t total_tiles = static_cast<int64_t>(ni_tiles) * static_cast<int64_t>(nj_tiles);

                    #pragma omp parallel num_threads(io_nt)
                    {
                        const int tid = omp_get_thread_num();
                        double* const my_eps = thr_eps.data() + static_cast<size_t>(tid) * k_pad64;
                        #pragma omp for schedule(static)
                        for (int64_t t = 0; t < total_tiles; ++t) {
                            const int32_t di0 = static_cast<int32_t>(t / nj_tiles) * kBlk;
                            const int32_t j0  = static_cast<int32_t>(t % nj_tiles) * kBlk;
                            const int32_t di_len = std::min(kBlk, i_len - di0);
                            const int32_t j_len  = std::min(kBlk, k - j0);
                            alignas(64) uint16_t qtile[kBlk][kBlk + 4];

                            for (int32_t dj = 0; dj < j_len; ++dj) {
                                const int32_t r = j0 + dj;
                                const float* const src_col = uk_src + static_cast<size_t>(r) * n_stride + (i0 + di0);
                                const float sc = sc_ptr[r];
                                double row_err = 0.0;
                                if (sc > 0.0f) {
                                    const float isc = isc_ptr[r];
                                    for (int32_t ddi = 0; ddi < di_len; ++ddi) {
                                        const float v  = src_col[ddi];
                                        const float qf = std::min(kTun3Int16QMax, std::max(-kTun3Int16QMax,
                                            std::nearbyintf(v * isc)));
                                        qtile[ddi][dj] = static_cast<uint16_t>(static_cast<int16_t>(qf));
                                        const double err = static_cast<double>(v) - static_cast<double>(qf) * static_cast<double>(sc);
                                        row_err += err * err;
                                    }
                                } else if (sc < 0.0f) {
                                    for (int32_t ddi = 0; ddi < di_len; ++ddi) {
                                        const float v    = src_col[ddi];
                                        const uint16_t h = wb_f32_to_fp16_bits(v);
                                        qtile[ddi][dj] = h;
                                        const double err = static_cast<double>(v) - static_cast<double>(wb_fp16_bits_to_f32(h));
                                        row_err += err * err;
                                    }
                                } else {
                                    for (int32_t ddi = 0; ddi < di_len; ++ddi) {
                                        qtile[ddi][dj] = 0u;
                                    }
                                }
                                my_eps[r] += row_err;
                            }
                            for (int32_t ddi = 0; ddi < di_len; ++ddi) {
                                uint16_t* const dst_col = dst_u16 + static_cast<size_t>(di0 + ddi) * k_stride + j0;
                                std::memcpy(dst_col, qtile[ddi], static_cast<size_t>(j_len) * sizeof(uint16_t));
                            }
                        }
                    }
                }
                writer.submit_slot(chunk_bytes);
            }

            // Write 4*k trailer row_eps_sq[k] sequentially after Uk body (contiguous SIMD sum).
            const size_t trailer_bytes = static_cast<size_t>(k) * sizeof(float);
            AlignedRawBuffer<char, 64>& tslot = writer.acquire_slot();
            if (tslot.size() < trailer_bytes) tslot.resize(trailer_bytes);
            float* const eps_dst = reinterpret_cast<float*>(tslot.data());
            std::vector<double> eps_sum(static_cast<size_t>(k), 0.0);
            for (int t = 0; t < io_nt; ++t) {
                const double* __restrict__ row_t = thr_eps.data() + static_cast<size_t>(t) * k_pad64;
                #pragma omp simd
                for (int32_t r = 0; r < k; ++r) {
                    eps_sum[static_cast<size_t>(r)] += row_t[r];
                }
            }
            #pragma omp simd
            for (int32_t r = 0; r < k; ++r) {
                eps_dst[r] = static_cast<float>(eps_sum[static_cast<size_t>(r)]);
            }
            writer.submit_slot(trailer_bytes);
            writer.finish();
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }

        ctx.Uk.resize(0, 0);
        const double uk_gb = static_cast<double>(static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(uint16_t)) / 1e9;
        LOGGER.i(0, "Saved compact TUN3 REML state: n=" + to_string(n) + ", k=" + to_string(k) +
                    " (" + std::format("{:.3f}", uk_gb) + " GB 16-bit Uk, " +
                    to_string(k - fp16_row_cnt) + " INT16 / " + to_string(fp16_row_cnt) + " FP16 rows).");
        return;
    }

    // --- GOBY Dense Exact Float32 Branch (double-buffered 64 MiB AsyncSeqWriter) ---
    Header hdr{};
    std::memcpy(hdr.magic, kMagicDense.data(), 4);
    hdr.n          = static_cast<int32_t>(ctx.n);
    hdr.x_c        = static_cast<int32_t>(ctx.X_c);
    hdr.num_varcmp = static_cast<int32_t>(ctx.varcmp.size());
    hdr.num_r_indx = static_cast<int32_t>(ctx.varcmp.size());
    wr(&hdr, sizeof(hdr));
    wr(&id_hash, sizeof(uint64_t));

    const int32_t factor_kind = ctx.Vi_use_llt ? 0 : 1;
    wr(&factor_kind, sizeof(int32_t));

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
    const size_t kGobyWriteSlotBytes = std::clamp<size_t>(io_slot_bytes, 1ULL << 20, 256ULL << 20);
    const size_t max_window_elems = kGobyWriteSlotBytes / sizeof(float);
    const int goby_io_nt = std::max(1, omp_get_max_threads() - 1);

    try {
        gcta_io::AsyncSeqWriter writer(fd.get(), filename, kGobyWriteSlotBytes, /*take_ownership=*/false);
        int32_t j = 0;
        while (j < ctx.n) {
            int32_t j_end = j;
            size_t chunk_elems = 0;
            while (j_end < ctx.n) {
                const size_t col_elems = static_cast<size_t>(j_end + 1);
                if (chunk_elems > 0 && chunk_elems + col_elems > max_window_elems) break;
                chunk_elems += col_elems;
                ++j_end;
            }
            const size_t chunk_bytes = chunk_elems * sizeof(float);
            AlignedRawBuffer<char, 64>& slot = writer.acquire_slot();
            if (slot.size() < chunk_bytes) slot.resize(chunk_bytes);
            float* const dst_base = reinterpret_cast<float*>(slot.data());
            const size_t chunk_start_elem = static_cast<size_t>(j) * (j + 1) / 2;

            #pragma omp parallel for schedule(guided) num_threads(goby_io_nt)
            for (int32_t jj = j; jj < j_end; ++jj) {
                const int32_t len = jj + 1;
                const double* col_src = mat.col(jj).head(len).data();
                const size_t col_elem = static_cast<size_t>(jj) * (jj + 1) / 2;
                float* dst = dst_base + (col_elem - chunk_start_elem);
                #pragma omp simd
                for (int32_t i = 0; i < len; ++i)
                    dst[i] = static_cast<float>(col_src[i]);
            }
            writer.submit_slot(chunk_bytes);
            j = j_end;
        }
        writer.drain();
        writer.finish();
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }

    if (!no_adj_covar) {
        Eigen::VectorXf b_f = ctx.b.cast<float>();
        wr(b_f.data(), static_cast<size_t>(ctx.X_c) * sizeof(float));
    }
    Eigen::VectorXf varcmp_f = Eigen::Map<const Eigen::VectorXd>(
        ctx.varcmp.data(), static_cast<Eigen::Index>(ctx.varcmp.size())).cast<float>();
    wr(varcmp_f.data(), static_cast<size_t>(hdr.num_varcmp) * sizeof(float));
}

struct LoadedBasisInfo {
    string   tun3_path;
    uint64_t basis_hash = 0;
};

// --reml-woodbury-reuse: load a TUN3 (or TREF -> TUN3) basis directly into
// ctx.Uk in k x n layout (ctx.Uk_is_transposed = true), eliminating both 40 GB transposes!
LoadedBasisInfo load_woodbury_basis_into_ctx(const string& filename, RemlCtx& ctx,
                                             uint64_t expected_id_hash,
                                             const string& basis_file_override,
                                             size_t io_slot_bytes = (256ULL << 20))
{
    ScopedFd fd(::open(filename.c_str(), O_RDONLY));
    if (!fd) LOGGER.e(0, "Cannot open file [" + filename + "].");

    struct stat sb{};
    if (::fstat(fd.get(), &sb) != 0) {
        LOGGER.e(0, "Failed to stat file [" + filename + "].");
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);
    if (file_size < kRemlPrefixBytes) {
        LOGGER.e(0, "[" + filename + "] is empty or too short to be a REML state.");
    }

    const RemlPrefix prefix = read_reml_prefix(fd.get(), filename);
    if (prefix.kind == RemlKind::Goby) {
        LOGGER.e(0, "[" + filename + "] is a dense (GOBY) REML state; --reml-woodbury-reuse needs a "
                    "Woodbury (TUN3/TREF) state saved from a --reml-woodbury-basis fit.");
    }
    check_reml_samples(prefix, ctx.n, expected_id_hash, filename);

    ScopedFd tun3_fd_owner;
    int tun3_fd = -1;
    string tun3_path;
    Tun3Metadata meta;

    if (prefix.kind == RemlKind::Tref) {
        TrefMetadata tref = read_tref_metadata(fd.get(), prefix, filename, /*load_vi_y=*/false);
        open_and_verify_tref_basis(filename, tref, basis_file_override,
                                   tun3_fd_owner, tun3_path, meta);
        tun3_fd = tun3_fd_owner.get();
    } else {
        meta      = read_tun3_metadata(fd.get(), prefix, file_size, filename, /*load_vi_y=*/false);
        tun3_fd   = fd.get();
        tun3_path = filename;
    }

    const int32_t k = meta.k;
    if (k <= 0 || k >= ctx.n) {
        LOGGER.e(0, "[" + tun3_path + "] has invalid Woodbury rank k=" + to_string(k) + ".");
    }

    std::vector<int> fp16_rows;
    for (int32_t r = 0; r < k; ++r) {
        if (meta.uk_row_scale[static_cast<size_t>(r)] < 0.0f) fp16_rows.push_back(r);
    }

    // Keep Uk in native k x n ColMajor layout (ZERO transpose!).
    ctx.Uk.resize(k, ctx.n);
    ctx.Uk_is_transposed = true;

    const size_t kReadSlotBytes = std::clamp<size_t>(io_slot_bytes, 1ULL << 20, 256ULL << 20);
    const size_t col_bytes = static_cast<size_t>(k) * sizeof(uint16_t);
    const std::vector<SampleChunk> plan = plan_sample_chunks(meta.body_offset, ctx.n, col_bytes, kReadSlotBytes);
    std::vector<std::pair<off_t, size_t>> io_chunks;
    io_chunks.reserve(plan.size());
    size_t max_b = 0;
    for (const auto& c : plan) {
        io_chunks.emplace_back(c.offset, c.bytes);
        max_b = std::max(max_b, c.bytes);
    }
    const int unpack_nt = std::max(1, omp_get_max_threads() - 1);

    try {
        gcta_io::AsyncSeqReader reader(tun3_fd, tun3_path, std::move(io_chunks), max_b, false);
        const char* data = nullptr;
        size_t len = 0;
        size_t ci = 0;
        while (reader.next_slot(data, len)) {
            const int s_beg = plan[ci].s_beg;
            const int cols  = plan[ci].s_end - s_beg;
            dequant_tun3_slab_to_f32(
                reinterpret_cast<const uint16_t*>(data), k, cols,
                meta.uk_row_scale, fp16_rows,
                ctx.Uk.data() + static_cast<size_t>(s_beg) * static_cast<size_t>(k),
                unpack_nt);
            reader.release_slot();
            ++ci;
        }
        reader.stop();
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }

    ctx.dk                       = meta.dk_f.cast<double>();
    ctx.lambda_tail              = meta.lambda_tail;
    ctx.tail_d_var               = meta.tail_d_var;
    ctx.woodbury_basis           = WoodburyBasisSpec{WoodburyBasisSpec::Kind::Fixed, k};
    ctx.woodbury_basis_rank_     = k;
    ctx.Vi_use_woodbury_basis    = true;
    ctx.woodbury_basis_preloaded = true;

    LOGGER.i(0, "Woodbury basis loaded directly in k x n layout from [" + tun3_path + "]: n=" +
                to_string(ctx.n) + ", k=" + to_string(k) +
                ", lambda_tail=" + to_text(meta.lambda_tail) +
                ", tail_d_var=" + to_text(meta.tail_d_var) + ".");
    return LoadedBasisInfo{tun3_path, meta.basis_hash};
}

// Zero-Uk streaming --reml-woodbury-reuse for REML-only thin TREF output:
// Never allocates 4*k*n ctx.Uk (12 GB at N=150k, k=20k); instead streams 16-bit TUN3
// slabs through a single reusable <= 256 MiB float32 slab across two passes to fill
// UkTX, UkTy, wb_XtX_perp, wb_Xty_perp, wb_yty_perp, cutting reuse RSS ~30x.
StreamingReuseHandle preload_woodbury_subspace_streaming(const string& filename,
                                                         RemlCtx& ctx,
                                                         uint64_t expected_id_hash,
                                                         const string& basis_file_override,
                                                         size_t io_slot_bytes = (256ULL << 20))
{
    ScopedFd fd(::open(filename.c_str(), O_RDONLY));
    if (!fd) LOGGER.e(0, "Cannot open file [" + filename + "].");

    struct stat sb{};
    if (::fstat(fd.get(), &sb) != 0) {
        LOGGER.e(0, "Failed to stat file [" + filename + "].");
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);
    if (file_size < kRemlPrefixBytes) {
        LOGGER.e(0, "[" + filename + "] is empty or too short to be a REML state.");
    }

    const RemlPrefix prefix = read_reml_prefix(fd.get(), filename);
    if (prefix.kind == RemlKind::Goby) {
        LOGGER.e(0, "[" + filename + "] is a dense (GOBY) REML state; --reml-woodbury-reuse needs a "
                    "Woodbury (TUN3/TREF) state saved from a --reml-woodbury-basis fit.");
    }
    check_reml_samples(prefix, ctx.n, expected_id_hash, filename);

    ScopedFd tun3_fd_owner;
    int tun3_fd = -1;
    StreamingReuseHandle out;
    out.io_slot_bytes = io_slot_bytes;

    if (prefix.kind == RemlKind::Tref) {
        TrefMetadata tref = read_tref_metadata(fd.get(), prefix, filename, /*load_vi_y=*/false);
        open_and_verify_tref_basis(filename, tref, basis_file_override,
                                   tun3_fd_owner, out.tun3_path, out.meta);
        tun3_fd = tun3_fd_owner.get();
    } else {
        out.meta      = read_tun3_metadata(fd.get(), prefix, file_size, filename, /*load_vi_y=*/false);
        tun3_fd       = fd.get();
        out.tun3_path = filename;
    }

    const int32_t k = out.meta.k;
    const int n = ctx.n;
    const int c = ctx.X_c;
    if (k <= 0 || k >= n) {
        LOGGER.e(0, "[" + out.tun3_path + "] has invalid Woodbury rank k=" + to_string(k) + ".");
    }
    for (int32_t r = 0; r < k; ++r) {
        if (out.meta.uk_row_scale[static_cast<size_t>(r)] < 0.0f) {
            out.fp16_rows.push_back(r);
        }
    }

    ctx.Uk.resize(0, 0);
    ctx.Uk_is_transposed         = false;
    ctx.dk                       = out.meta.dk_f.cast<double>();
    ctx.lambda_tail              = out.meta.lambda_tail;
    ctx.tail_d_var               = out.meta.tail_d_var;
    ctx.woodbury_basis           = WoodburyBasisSpec{WoodburyBasisSpec::Kind::Fixed, k};
    ctx.woodbury_basis_rank_     = k;
    ctx.Vi_use_woodbury_basis    = true;
    ctx.woodbury_basis_preloaded = true;
    out.basis_hash               = out.meta.basis_hash;
    out.active                   = true;

    Eigen::MatrixXf XY_f(n, c + 1);
    #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(c) >= 65536)
    for (int j = 0; j < c; ++j) {
        const double* __restrict__ src = ctx.X.col(j).data();
        float* __restrict__ dst = XY_f.col(j).data();
        #pragma omp simd
        for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    }
    {
        const double* __restrict__ src = ctx.y.data();
        float* __restrict__ dst = XY_f.col(c).data();
        #pragma omp parallel for schedule(static) if (n >= 65536)
        for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    }

    const size_t col_bytes = static_cast<size_t>(k) * sizeof(uint16_t);
    const size_t u16_slot_bytes = std::clamp<size_t>(io_slot_bytes, 1ULL << 20, 128ULL << 20);
    const std::vector<SampleChunk> plan = plan_sample_chunks(out.meta.body_offset, n, col_bytes, u16_slot_bytes);
    std::vector<std::pair<off_t, size_t>> io_chunks;
    io_chunks.reserve(plan.size());
    size_t max_b = 0;
    int max_cols = 1;
    for (const auto& ch : plan) {
        io_chunks.emplace_back(ch.offset, ch.bytes);
        max_b = std::max(max_b, ch.bytes);
        max_cols = std::max(max_cols, ch.s_end - ch.s_beg);
    }

    AlignedRawBuffer<float, 64> slab_f32(static_cast<size_t>(k) * static_cast<size_t>(max_cols));
    Eigen::MatrixXf UkTXY_f = Eigen::MatrixXf::Zero(k, c + 1);
    const int unpack_nt = std::max(1, omp_get_max_threads() - 1);

    try {
        // Pass 1: UkTXY_f = sum_slab slab_f32 * XY_f[s_beg:s_end]
        {
            gcta_io::AsyncSeqReader reader(tun3_fd, out.tun3_path, io_chunks, max_b, false);
            const char* data = nullptr;
            size_t len = 0;
            size_t ci = 0;
            while (reader.next_slot(data, len)) {
                const int s_beg = plan[ci].s_beg;
                const int cols  = plan[ci].s_end - s_beg;
                dequant_tun3_slab_to_f32(
                    reinterpret_cast<const uint16_t*>(data), k, cols,
                    out.meta.uk_row_scale, out.fp16_rows, slab_f32.data(), unpack_nt);
                reader.release_slot();
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<gcta_blas_int>(k), static_cast<gcta_blas_int>(c + 1),
                            static_cast<gcta_blas_int>(cols),
                            1.0f, slab_f32.data(), static_cast<gcta_blas_int>(k),
                            XY_f.data() + s_beg, static_cast<gcta_blas_int>(n),
                            1.0f, UkTXY_f.data(), static_cast<gcta_blas_int>(k));
                ++ci;
            }
            reader.stop();
        }
        XY_f.resize(0, 0);

        ctx.UkTX = UkTXY_f.leftCols(c).cast<double>();
        ctx.UkTy = UkTXY_f.col(c).cast<double>();

        // Pass 2: proj_f = Uk^T * UkTXY_f (computes exact orthogonalized residual subspace)
        Eigen::MatrixXf proj_f(n, c + 1);
        {
            gcta_io::AsyncSeqReader reader(tun3_fd, out.tun3_path, io_chunks, max_b, false);
            const char* data = nullptr;
            size_t len = 0;
            size_t ci = 0;
            while (reader.next_slot(data, len)) {
                const int s_beg = plan[ci].s_beg;
                const int cols  = plan[ci].s_end - s_beg;
                dequant_tun3_slab_to_f32(
                    reinterpret_cast<const uint16_t*>(data), k, cols,
                    out.meta.uk_row_scale, out.fp16_rows, slab_f32.data(), unpack_nt);
                reader.release_slot();
                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            static_cast<gcta_blas_int>(cols), static_cast<gcta_blas_int>(c + 1),
                            static_cast<gcta_blas_int>(k),
                            1.0f, slab_f32.data(), static_cast<gcta_blas_int>(k),
                            UkTXY_f.data(), static_cast<gcta_blas_int>(k),
                            0.0f, proj_f.data() + s_beg, static_cast<gcta_blas_int>(n));
                ++ci;
            }
            reader.stop();
        }
        slab_f32.clear();

        RemlMat X_perp(n, c);
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(c) >= 65536)
        for (int j = 0; j < c; ++j) {
            const double* __restrict__ xcol = ctx.X.col(j).data();
            const float*  __restrict__ pcol = proj_f.col(j).data();
            double*       __restrict__ dcol = X_perp.col(j).data();
            #pragma omp simd
            for (int i = 0; i < n; ++i) dcol[i] = xcol[i] - static_cast<double>(pcol[i]);
        }
        RemlVec y_perp(n);
        {
            const double* __restrict__ ycol = ctx.y.data();
            const float*  __restrict__ pcol = proj_f.col(c).data();
            double*       __restrict__ dcol = y_perp.data();
            #pragma omp parallel for schedule(static) if (n >= 65536)
            for (int i = 0; i < n; ++i) dcol[i] = ycol[i] - static_cast<double>(pcol[i]);
        }
        proj_f.resize(0, 0);

        ctx.wb_XtX_perp.resize(c, c);
        ctx.wb_XtX_perp.noalias() = X_perp.transpose() * X_perp;
        ctx.wb_Xty_perp.resize(c);
        ctx.wb_Xty_perp.noalias() = X_perp.transpose() * y_perp;
        ctx.wb_yty_perp = y_perp.squaredNorm();
        ctx.wb_subspace_ready = true;
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }

    LOGGER.i(0, "Woodbury subspace streamed (zero-Uk) from [" + out.tun3_path + "]: n=" +
                to_string(n) + ", k=" + to_string(k) +
                ", lambda_tail=" + to_text(out.meta.lambda_tail) +
                ", tail_d_var=" + to_text(out.meta.tail_d_var) + ".");
    return out;
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

// Load a binary EVEC file (<prefix>.eigenvec.bin) directly into n x k ColMajor Uk (ZERO transpose).
Eigen::MatrixXf load_evec_bin_warm_start(const string& path, const vector<string>& analysis_ids)
{
    ScopedFd fd(::open(path.c_str(), O_RDONLY));
    if (!fd) LOGGER.e(0, "--reml-woodbury-basis-warm-start: cannot open [" + path + "].");

    struct stat sb{};
    if (::fstat(fd.get(), &sb) != 0 || sb.st_size < static_cast<off_t>(sizeof(EvecHeader))) {
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: [" + path + "] is too short to be an EVEC file.");
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);

    EvecHeader hdr{};
    try {
        gcta_io::read_exact_throwing(fd.get(), &hdr, sizeof(hdr), path);
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }
    if (std::string_view(hdr.magic, 4) != kMagicEvec || hdr.version != 1u) {
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: [" + path + "] is not a supported EVEC v1 file.");
    }
    if (hdr.dtype > 1u || hdr.n <= 0 || hdr.k <= 0) {
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: invalid EVEC dimensions/dtype in [" + path + "].");
    }

    const int n_file = hdr.n;
    const int k      = hdr.k;
    const int n      = static_cast<int>(analysis_ids.size());
    const uint64_t expected_hash = hash_sample_ids(analysis_ids);

    string ids_blob(static_cast<size_t>(hdr.id_bytes), '\0');
    try {
        if (hdr.id_bytes > 0) {
            gcta_io::read_exact_throwing(fd.get(), ids_blob.data(), static_cast<size_t>(hdr.id_bytes), path);
        }
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }

    std::vector<float> scales;
    off_t body_off = static_cast<off_t>(sizeof(EvecHeader) + hdr.id_bytes);
    if (hdr.dtype == 1u) {
        scales.resize(static_cast<size_t>(k));
        try {
            gcta_io::read_exact_throwing(fd.get(), scales.data(), static_cast<size_t>(k) * sizeof(float), path);
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }
        body_off += static_cast<off_t>(static_cast<size_t>(k) * sizeof(float));
    }

    const size_t elem_bytes = (hdr.dtype == 0u) ? sizeof(float) : sizeof(int16_t);
    const size_t col_bytes  = static_cast<size_t>(n_file) * elem_bytes;
    const size_t body_bytes = static_cast<size_t>(k) * col_bytes;
    if (file_size < static_cast<size_t>(body_off) + body_bytes) {
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: truncated EVEC body in [" + path + "].");
    }

    const bool exact_match = (n_file == n && hdr.id_hash == expected_hash);
    std::vector<int> kp;
    std::vector<gcta_grm_io::KpRun> kp_runs;
    if (!exact_match) {
        vector<string> file_ids;
        file_ids.reserve(static_cast<size_t>(n_file));
        const char* p = ids_blob.data();
        const char* const end = p + ids_blob.size();
        while (p < end) {
            const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
            const char* le = nl ? nl : end;
            if (le > p && *(le - 1) == '\r') --le;
            file_ids.emplace_back(p, static_cast<size_t>(le - p));
            p = nl ? (nl + 1) : end;
        }
        if (static_cast<int>(file_ids.size()) != n_file) {
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: malformed ID table in [" + path + "].");
        }
        kp = match_ids_to_grm(analysis_ids, file_ids);
        bool monotonic = true;
        for (int i = 0; i < n; ++i) {
            if (kp[static_cast<size_t>(i)] < 0) {
                LOGGER.e(0, "--reml-woodbury-basis-warm-start: individual [" + analysis_ids[static_cast<size_t>(i)] +
                            "] not found in [" + path + "].");
            }
            if (i > 0 && kp[static_cast<size_t>(i)] <= kp[static_cast<size_t>(i - 1)]) monotonic = false;
        }
        if (monotonic && n > 0) {
            auto runs = gcta_grm_io::build_kp_runs(kp);
            if (runs.size() * 2 < static_cast<size_t>(n)) kp_runs = std::move(runs);
        }
    }

    Eigen::MatrixXf Uk(n, k);
    constexpr size_t kEvecSlotBytes = 32ULL << 20;
    const int cols_per = static_cast<int>(std::max<size_t>(1, kEvecSlotBytes / std::max<size_t>(1, col_bytes)));
    std::vector<std::pair<off_t, size_t>> io_chunks;
    std::vector<std::pair<int, int>> col_ranges;
    size_t max_b = 0;
    off_t cur = body_off;
    for (int j0 = 0; j0 < k; j0 += cols_per) {
        const int j1 = std::min(k, j0 + cols_per);
        const size_t bytes = static_cast<size_t>(j1 - j0) * col_bytes;
        io_chunks.emplace_back(cur, bytes);
        col_ranges.emplace_back(j0, j1);
        max_b = std::max(max_b, bytes);
        cur += static_cast<off_t>(bytes);
    }

    try {
        gcta_io::AsyncSeqReader reader(fd.get(), path, std::move(io_chunks), max_b, false);
        const char* data = nullptr;
        size_t len = 0;
        size_t ci = 0;
        while (reader.next_slot(data, len)) {
            const int j0 = col_ranges[ci].first;
            const int j1 = col_ranges[ci].second;
            const int jlen = j1 - j0;

            #pragma omp parallel for schedule(static)
            for (int dj = 0; dj < jlen; ++dj) {
                const int j = j0 + dj;
                float* const dst = Uk.col(j).data();
                if (hdr.dtype == 0u) {
                    const float* const src = reinterpret_cast<const float*>(data) +
                        static_cast<size_t>(dj) * static_cast<size_t>(n_file);
                    if (exact_match) {
                        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
                    } else if (!kp_runs.empty()) {
                        for (const auto& run : kp_runs) {
                            std::memcpy(dst + run.dst_start, src + run.src_start,
                                        static_cast<size_t>(run.len) * sizeof(float));
                        }
                    } else {
                        for (int i = 0; i < n; ++i) dst[i] = src[kp[static_cast<size_t>(i)]];
                    }
                } else {
                    const int16_t* const src = reinterpret_cast<const int16_t*>(data) +
                        static_cast<size_t>(dj) * static_cast<size_t>(n_file);
                    const float sc = scales[static_cast<size_t>(j)];
                    if (exact_match) {
                        #pragma omp simd
                        for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]) * sc;
                    } else if (!kp_runs.empty()) {
                        for (const auto& run : kp_runs) {
                            const int16_t* s_seg = src + run.src_start;
                            float* d_seg = dst + run.dst_start;
                            #pragma omp simd
                            for (int r = 0; r < run.len; ++r) d_seg[r] = static_cast<float>(s_seg[r]) * sc;
                        }
                    } else {
                        for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[kp[static_cast<size_t>(i)]]) * sc;
                    }
                }
            }
            reader.release_slot();
            ++ci;
        }
        reader.stop();
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }

    LOGGER.i(0, "Woodbury warm-start: loaded " + to_string(k) + " eigenvector(s) for " +
                to_string(n) + " individuals from binary EVEC [" + path + "] (dtype=" +
                (hdr.dtype == 1u ? "bin16" : "f32") + ").");
    return Uk;
}

// Load eigenvectors from a TUN3 or TREF .reml file as n x k ColMajor warm-start vectors.
Eigen::MatrixXf load_reml_as_warm_start(const string& path, const vector<string>& analysis_ids,
                                        const string& basis_file_override)
{
    ScopedFd fd(::open(path.c_str(), O_RDONLY));
    if (!fd) LOGGER.e(0, "--reml-woodbury-basis-warm-start: cannot open [" + path + "].");
    struct stat sb{};
    if (::fstat(fd.get(), &sb) != 0 || sb.st_size < static_cast<off_t>(kRemlPrefixBytes)) {
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: [" + path + "] is too short.");
    }
    const RemlPrefix prefix = read_reml_prefix(fd.get(), path);
    if (prefix.kind == RemlKind::Goby) {
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: [" + path + "] is a dense GOBY state (no eigenvectors).");
    }
    const int n = static_cast<int>(analysis_ids.size());
    check_reml_samples(prefix, n, hash_sample_ids(analysis_ids), path);

    ScopedFd tun3_fd_owner;
    int tun3_fd = -1;
    string tun3_path;
    Tun3Metadata meta;
    if (prefix.kind == RemlKind::Tref) {
        TrefMetadata tref = read_tref_metadata(fd.get(), prefix, path, false);
        open_and_verify_tref_basis(path, tref, basis_file_override, tun3_fd_owner, tun3_path, meta);
        tun3_fd = tun3_fd_owner.get();
    } else {
        meta = read_tun3_metadata(fd.get(), prefix, static_cast<size_t>(sb.st_size), path, false);
        tun3_fd = fd.get();
        tun3_path = path;
    }

    const int k = meta.k;
    std::vector<int> fp16_rows;
    for (int r = 0; r < k; ++r) {
        if (meta.uk_row_scale[static_cast<size_t>(r)] < 0.0f) fp16_rows.push_back(r);
    }

    Eigen::MatrixXf Uk(n, k);
    constexpr size_t kSlotBytes = 64ULL << 20;
    const size_t col_bytes = static_cast<size_t>(k) * sizeof(uint16_t);
    const std::vector<SampleChunk> plan = plan_sample_chunks(meta.body_offset, n, col_bytes, kSlotBytes);
    std::vector<std::pair<off_t, size_t>> io_chunks;
    io_chunks.reserve(plan.size());
    size_t max_b = 0;
    int max_cols = 0;
    for (const auto& c : plan) {
        io_chunks.emplace_back(c.offset, c.bytes);
        max_b = std::max(max_b, c.bytes);
        max_cols = std::max(max_cols, c.s_end - c.s_beg);
    }
    AlignedRawBuffer<float, 64> tmp_kn(static_cast<size_t>(k) * static_cast<size_t>(max_cols));
    const size_t n_stride = static_cast<size_t>(n);
    const size_t k_stride = static_cast<size_t>(k);

    try {
        gcta_io::AsyncSeqReader reader(tun3_fd, tun3_path, std::move(io_chunks), max_b, false);
        const char* data = nullptr;
        size_t len = 0;
        size_t ci = 0;
        while (reader.next_slot(data, len)) {
            const int s_beg = plan[ci].s_beg;
            const int slen  = plan[ci].s_end - s_beg;
            dequant_tun3_slab_to_f32(reinterpret_cast<const uint16_t*>(data), k, slen,
                                     meta.uk_row_scale, fp16_rows, tmp_kn.data());
            constexpr int kBlk = 64;
            const int ni_tiles = (slen + kBlk - 1) / kBlk;
            const int nj_tiles = (k + kBlk - 1) / kBlk;
            const int64_t total_tiles = static_cast<int64_t>(ni_tiles) * static_cast<int64_t>(nj_tiles);
            #pragma omp parallel for schedule(static)
            for (int64_t t = 0; t < total_tiles; ++t) {
                const int j0  = static_cast<int>(t / ni_tiles) * kBlk;
                const int di0 = static_cast<int>(t % ni_tiles) * kBlk;
                const int di_len = std::min(kBlk, slen - di0);
                const int j_len  = std::min(kBlk, k - j0);
                alignas(64) float tile[kBlk][kBlk + 4];
                for (int ddi = 0; ddi < di_len; ++ddi) {
                    const float* src = tmp_kn.data() + static_cast<size_t>(di0 + ddi) * k_stride + j0;
                    for (int dj = 0; dj < j_len; ++dj) tile[dj][ddi] = src[dj];
                }
                for (int dj = 0; dj < j_len; ++dj) {
                    std::memcpy(Uk.data() + static_cast<size_t>(j0 + dj) * n_stride + (s_beg + di0),
                                tile[dj], static_cast<size_t>(di_len) * sizeof(float));
                }
            }
            reader.release_slot();
            ++ci;
        }
        reader.stop();
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }

    LOGGER.i(0, "Woodbury warm-start: loaded " + to_string(k) + " eigenvector(s) for " +
                to_string(n) + " individuals from REML state [" + tun3_path + "].");
    return Uk;
}

// Fast text .eigenvec warm-start loader: 1 sequential AsyncSeqReader thread (2 x 32 MiB)
// + OpenMP parallel from_chars(..., float) + 64x64 cache-blocked transpose into Uk (n x k).
Eigen::MatrixXf load_text_eigenvec_warm_start(const string& path, const vector<string>& analysis_ids)
{
    ScopedFd fd(::open(path.c_str(), O_RDONLY));
    if (!fd) LOGGER.e(0, "--reml-woodbury-basis-warm-start: cannot open [" + path + "].");

    struct stat sb{};
    if (::fstat(fd.get(), &sb) != 0) {
        LOGGER.e(0, "--reml-woodbury-basis-warm-start: cannot stat [" + path + "].");
    }
    const uint64_t file_size = static_cast<uint64_t>(std::max<off_t>(0, sb.st_size));
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
    std::vector<float> slab;
    int k = -1;
    int seq_hint = 0;

    struct PendingLine {
        const char* val_beg = nullptr;
        const char* val_end = nullptr;
        int dst_row         = -1;
    };

    auto process_lines_block = [&](const char* buf, size_t len) {
        if (len == 0) return;
        std::vector<PendingLine> lines;
        string key;

        const char* p = buf;
        const char* const end = buf + len;
        while (p < end) {
            const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
            const char* le = nl ? nl : end;
            const char* next_p = nl ? (nl + 1) : end;
            if (le > p && *(le - 1) == '\r') --le;

            const char* q = p;
            while (q < le && is_ws(static_cast<unsigned char>(*q))) ++q;
            if (q == le) { p = next_p; continue; }

            std::string_view fid, iid;
            if (!next_token(q, le, fid) || !next_token(q, le, iid)) {
                LOGGER.e(0, "--reml-woodbury-basis-warm-start: malformed line in [" + path + "].");
            }

            if (k < 0) {
                // Determine k from the first non-empty line.
                const char* cq = q;
                std::string_view tok;
                int cnt = 0;
                while (next_token(cq, le, tok)) ++cnt;
                k = cnt;
                if (k <= 0) {
                    LOGGER.e(0, "--reml-woodbury-basis-warm-start: no eigenvectors read from [" + path + "].");
                }
                Uk.resize(n, k);
            }

            key.clear();
            key.reserve(fid.size() + 1 + iid.size());
            key.append(fid);
            key.push_back('\t');
            key.append(iid);

            int dst_row = -1;
            if (seq_hint < n && analysis_ids[static_cast<size_t>(seq_hint)] == key) {
                dst_row = seq_hint++;
            } else {
                const auto it = id_to_row.find(std::string_view(key));
                if (it != id_to_row.end()) {
                    dst_row = it->second;
                    seq_hint = dst_row + 1;
                }
            }
            if (dst_row >= 0 && !seen[static_cast<size_t>(dst_row)]) {
                seen[static_cast<size_t>(dst_row)] = 1;
                lines.push_back(PendingLine{q, le, dst_row});
            } else {
                // Still validate line token count & numeric format? Keep lines with dst_row=-1?
                // Keep exact contract: validate tokens for matched rows (and validate malformed counts).
                lines.push_back(PendingLine{q, le, -1});
            }
            p = next_p;
        }

        if (lines.empty() || k <= 0) return;

        // Compact kept lines for slab storage while validating all lines in parallel.
        std::vector<int> line_to_slab(lines.size(), -1);
        int kept_cnt = 0;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].dst_row >= 0) line_to_slab[i] = kept_cnt++;
        }
        slab.resize(static_cast<size_t>(kept_cnt) * static_cast<size_t>(k));

        int err_kind = 0; // 1 = malformed float, 2 = inconsistent count
        #pragma omp parallel for schedule(static)
        for (size_t li = 0; li < lines.size(); ++li) {
            if (__atomic_load_n(&err_kind, __ATOMIC_RELAXED) != 0) continue;
            const char* q = lines[li].val_beg;
            const char* const le = lines[li].val_end;
            const int sidx = line_to_slab[li];
            float* const dst_row_ptr = (sidx >= 0)
                ? (slab.data() + static_cast<size_t>(sidx) * static_cast<size_t>(k))
                : nullptr;
            int cnt = 0;
            std::string_view tok;
            while (next_token(q, le, tok)) {
                const char* tb = tok.data();
                const char* const te = tb + tok.size();
                if (tb < te && *tb == '+') ++tb;
                float vf = 0.0f;
                auto [ptr, ec] = std::from_chars(tb, te, vf);
                if (ec == std::errc::result_out_of_range) {
                    char num_tmp[128];
                    const size_t tlen = static_cast<size_t>(te - tb);
                    if (tlen < sizeof(num_tmp)) {
                        std::memcpy(num_tmp, tb, tlen);
                        num_tmp[tlen] = '\0';
                        char* endp = nullptr;
                        double vd = std::strtod(num_tmp, &endp);
                        if (endp == num_tmp + tlen) {
                            vf = static_cast<float>(vd);
                            ec = std::errc();
                            ptr = te;
                        }
                    }
                }
                if (ec != std::errc() || ptr != te) {
                    __atomic_store_n(&err_kind, 1, __ATOMIC_RELAXED);
                    break;
                }
                if (cnt < k && dst_row_ptr != nullptr) {
                    dst_row_ptr[cnt] = vf;
                }
                ++cnt;
            }
            if (__atomic_load_n(&err_kind, __ATOMIC_RELAXED) == 0 && cnt != k) {
                __atomic_store_n(&err_kind, 2, __ATOMIC_RELAXED);
            }
        }
        if (err_kind == 1) {
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: malformed eigenvector value in [" + path + "].");
        }
        if (err_kind == 2) {
            LOGGER.e(0, "--reml-woodbury-basis-warm-start: inconsistent eigenvector count in [" + path + "].");
        }
        if (kept_cnt == 0) return;

        // Gather dst_rows for kept lines and check contiguity for 64x64 cache-blocked transpose.
        std::vector<int> kept_dst;
        kept_dst.reserve(static_cast<size_t>(kept_cnt));
        for (const auto& ln : lines) {
            if (ln.dst_row >= 0) kept_dst.push_back(ln.dst_row);
        }
        bool contiguous = true;
        const int base_row = kept_dst[0];
        for (int r = 1; r < kept_cnt; ++r) {
            if (kept_dst[static_cast<size_t>(r)] != base_row + r) {
                contiguous = false;
                break;
            }
        }

        const size_t n_stride = static_cast<size_t>(n);
        const size_t k_stride = static_cast<size_t>(k);
        float* const uk_dst   = Uk.data();
        const float* const sb_ptr = slab.data();

        if (contiguous) {
            constexpr int kBlk = 64;
            const int nr_tiles = (kept_cnt + kBlk - 1) / kBlk;
            const int nj_tiles = (k + kBlk - 1) / kBlk;
            const int64_t total_tiles = static_cast<int64_t>(nr_tiles) * static_cast<int64_t>(nj_tiles);
            #pragma omp parallel for schedule(static)
            for (int64_t t = 0; t < total_tiles; ++t) {
                const int j0  = static_cast<int>(t / nr_tiles) * kBlk;
                const int r0  = static_cast<int>(t % nr_tiles) * kBlk;
                const int r_len = std::min(kBlk, kept_cnt - r0);
                const int j_len = std::min(kBlk, k - j0);
                alignas(64) float tile[kBlk][kBlk + 4];
                for (int dr = 0; dr < r_len; ++dr) {
                    const float* src = sb_ptr + static_cast<size_t>(r0 + dr) * k_stride + j0;
                    for (int dj = 0; dj < j_len; ++dj) tile[dj][dr] = src[dj];
                }
                for (int dj = 0; dj < j_len; ++dj) {
                    std::memcpy(uk_dst + static_cast<size_t>(j0 + dj) * n_stride + (base_row + r0),
                                tile[dj], static_cast<size_t>(r_len) * sizeof(float));
                }
            }
        } else {
            // Column-tiled parallel scatter: 64x fewer TLB misses than unblocked row scatter.
            constexpr int kColTile = 64;
            #pragma omp parallel for schedule(static)
            for (int j0 = 0; j0 < k; j0 += kColTile) {
                const int j1 = std::min(k, j0 + kColTile);
                for (int r = 0; r < kept_cnt; ++r) {
                    const int dst_r = kept_dst[static_cast<size_t>(r)];
                    const float* const src = sb_ptr + static_cast<size_t>(r) * k_stride;
                    for (int j = j0; j < j1; ++j) {
                        uk_dst[static_cast<size_t>(j) * n_stride + static_cast<size_t>(dst_r)] = src[j];
                    }
                }
            }
        }
    };

    string carry;
    try {
        constexpr size_t kTextSlotBytes = 32ULL << 20; // 32 MiB double-buffered
        gcta_io::AsyncSeqReader reader(fd.get(), path, 0, file_size, kTextSlotBytes, false);
        const char* data = nullptr;
        size_t len = 0;
        while (reader.next_slot(data, len)) {
            if (len == 0) { reader.release_slot(); continue; }
            const char* cur = data;
            size_t rem = len;

            if (!carry.empty()) {
                const char* first_nl = static_cast<const char*>(std::memchr(cur, '\n', rem));
                if (!first_nl) {
                    carry.append(cur, rem);
                    reader.release_slot();
                    continue;
                }
                const size_t piece = static_cast<size_t>((first_nl - cur) + 1);
                carry.append(cur, piece);
                process_lines_block(carry.data(), carry.size());
                carry.clear();
                cur += piece;
                rem -= piece;
            }

            if (rem > 0) {
                const char* last_nl = nullptr;
                for (size_t i = rem; i > 0; --i) {
                    if (cur[i - 1] == '\n') { last_nl = cur + (i - 1); break; }
                }
                if (last_nl != nullptr) {
                    const size_t full_len = static_cast<size_t>((last_nl - cur) + 1);
                    process_lines_block(cur, full_len);
                    const size_t tail_len = rem - full_len;
                    if (tail_len > 0) carry.assign(cur + full_len, tail_len);
                } else {
                    carry.assign(cur, rem);
                }
            }
            reader.release_slot();
        }
        reader.stop();
    } catch (const std::exception& e) {
        LOGGER.e(0, e.what());
    }

    if (!carry.empty()) {
        process_lines_block(carry.data(), carry.size());
        carry.clear();
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

// Auto-detect binary EVEC (<prefix>.eigenvec.bin or direct file), TUN3/TREF .reml,
// or text <prefix>.eigenvec for --woodbury-warm-start.
Eigen::MatrixXf load_pca_warm_start(const string& eigenvec_prefix,
                                    const vector<string>& analysis_ids,
                                    const string& basis_file_override)
{
    if (is_regular_file_path(eigenvec_prefix)) {
        ScopedFd f(::open(eigenvec_prefix.c_str(), O_RDONLY));
        if (f) {
            char magic[4]{};
            const ssize_t r = ::pread(f.get(), magic, 4, 0);
            f.reset();
            if (r == 4) {
                const std::string_view mv(magic, 4);
                if (mv == kMagicEvec) {
                    return load_evec_bin_warm_start(eigenvec_prefix, analysis_ids);
                }
                if (mv == kMagicTun3 || mv == kMagicTref) {
                    return load_reml_as_warm_start(eigenvec_prefix, analysis_ids, basis_file_override);
                }
                if (mv == kMagicLegacyTuna) {
                    LOGGER.e(0, "--reml-woodbury-basis-warm-start: [" + eigenvec_prefix +
                                "] uses legacy 'TUNA' v2 format; re-create with 'GCTAg reml'.");
                }
            }
        }
        return load_text_eigenvec_warm_start(eigenvec_prefix, analysis_ids);
    }

    const string bin_path = eigenvec_prefix + ".eigenvec.bin";
    if (is_regular_file_path(bin_path)) {
        return load_evec_bin_warm_start(bin_path, analysis_ids);
    }
    return load_text_eigenvec_warm_start(eigenvec_prefix + ".eigenvec", analysis_ids);
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// MLMA::run
// ---------------------------------------------------------------------------
void MLMA::run(const gcta_cli::MlmaOptions& opts, const std::string& out_prefix)
{
    const auto mlma_t0 = std::chrono::steady_clock::now();
    MlmaTuningOptions mlma_tuning;
    mlma_tuning.p_budget_mb   = opts.mlma_p_budget_mb;
    mlma_tuning.q_rows        = opts.mlma_q_rows;
    mlma_tuning.snp_tile      = opts.mlma_snp_tile;
    if (opts.mlma_reorder_q == gcta_cli::MlmaReorderQ::On) {
        mlma_tuning.reorder_q = true;
    } else if (opts.mlma_reorder_q == gcta_cli::MlmaReorderQ::Off) {
        mlma_tuning.reorder_q = false;
    } else if (const char* env_rq = std::getenv("GCTA_MLMA_REORDER_Q")) {
        mlma_tuning.reorder_q = (std::strcmp(env_rq, "1") == 0);
    } else {
        mlma_tuning.reorder_q = false;
    }
    mlma_tuning.block_size    = opts.mlma_block_size;
    mlma_tuning.sample_tile   = opts.mlma_sample_tile;
    mlma_tuning.ukx_budget_mb = opts.mlma_ukx_budget_mb;
    mlma_tuning.tun3_slot_mb  = opts.mlma_tun3_slot_mb;
    mlma_tuning.profile       = opts.mlma_profile;
    MlmaProfileStats mlma_prof;
    MlmaProfileStats* mlma_prof_ptr = mlma_tuning.profile ? &mlma_prof : nullptr;
    auto log_mlma_profile = [&]() {
        if (!mlma_tuning.profile) return;
        const double total_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mlma_t0).count();
        LOGGER.i(0, std::format(
            "[mlma-profile] reml_load_s={:.3f} q_build_s={:.3f} q_reorder_s={:.3f} "
            "decode_s={:.3f} xvx_gemm_s={:.3f} xvx_reduce_s={:.3f} format_write_s={:.3f} "
            "total_s={:.3f} blocks={} valid_snps={} total_snps={} q_tiles={} block_size={}",
            mlma_prof.reml_load_s, mlma_prof.q_build_s, mlma_prof.q_reorder_s,
            mlma_prof.decode_s, mlma_prof.xvx_gemm_s, mlma_prof.xvx_reduce_s,
            mlma_prof.format_write_s, total_s, mlma_prof.blocks,
            mlma_prof.valid_snps, mlma_prof.total_snps, mlma_prof.q_tiles,
            mlma_prof.block_size));
    };

    const auto& r = opts.reml;
    const bool   no_adj_covar   = opts.no_preadj_covar;
    const bool   log_pval       = opts.log_pval;
    const bool   weighted_pheno = opts.pheno_covar.weighted_phenotypes;
    // --save-reml only fits/saves the REML state and never streams SNPs,
    // so the bim (Marker) and genotype (Geno) data are never touched.
    const bool reml_only   = opts.save_reml;
    const bool fast_preset = opts.fast || r.fast;
    const string& basis_file_override = !opts.woodbury_basis_file.empty()
        ? opts.woodbury_basis_file
        : r.woodbury_basis_file;
    FastMlmaMode fast_mlma_mode = opts.fast_mlma.value_or(FastMlmaMode::Off);
    const bool fast_enabled_mlma = fast_preset && !reml_only && !opts.fast_mlma.has_value();
    if (fast_enabled_mlma) {
        fast_mlma_mode = FastMlmaMode::Slice2;
    }
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
        const vector<string> analysis_ids = pheno.get_id(0, n - 1, "\t");
        const uint64_t id_hash = hash_sample_ids(analysis_ids);

        // Build design matrix X  (n × x_c, column-major)
        if (has_covar) {
            vector<double> covar_X;
            vector<uint32_t> ck_dummy;
            covar.getCovarX(analysis_ids, covar_X, ck_dummy);
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
        Eigen::VectorXf w_sqrt = weighted_pheno
            ? pheno.get_sqrt_weight_keep()
            : Eigen::VectorXf::Ones(n);

        // Streaming SNP-block width, sized from a memory budget rather than
        // the previous fixed BLOCK=10000 (which allocated the same
        // GenoBufItem::geno + X_block footprint regardless of n, reaching
        // multi-GB at large n). Opt-in, matching --GRM-tile-budget /
        // --grm-chunked-budget elsewhere: omitting the flag reproduces the
        // old fixed-10000 behavior exactly (see resolve_mlma_block_size).
        const double mlma_tile_budget_gb = opts.stream_budget_gb.value_or(0.0);

        if (!opts.load_reml.empty()) {
            if (fast_enabled_mlma) {
                LOGGER.i(0, "[--fast] enabled: fast-mlma=2slice (1-pass streaming TUN3 load).");
            }
            const string& load_reml_file = opts.load_reml;
            const bool can_stream_fast = (fast_mlma_mode != FastMlmaMode::Off) &&
                                         geno->canUseInt8Mlma() &&
                                         (std::getenv("GCTA_DISABLE_INT8_MLMA") == nullptr);
            LOGGER.i(0, "Loading REML state from [" + load_reml_file + "]...");
            LOGGER.ts("load_reml");
            const auto t_reml0 = std::chrono::steady_clock::now();
            state = readRemlState(load_reml_file, no_adj_covar, n, x_c,
                                  id_hash,
                                  basis_file_override,
                                  can_stream_fast ? fast_mlma_mode : FastMlmaMode::Off,
                                  y_vec, X_design, w_sqrt,
                                  static_cast<size_t>(std::max(1, mlma_tuning.tun3_slot_mb)) << 20,
                                  mlma_tuning.q_rows);
            mlma_prof.reml_load_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_reml0).count();
            LOGGER.i(0, "REML state loaded in " + to_string(LOGGER.tp("load_reml")) + " seconds.");

            if (state.n != n)
                LOGGER.e(0, "Sample size mismatch: REML state n=" + to_string(state.n) +
                            " vs dataset n=" + to_string(n) +
                            ". Use the same filters (--keep/--remove/--pheno) in both runs.");
            if (state.x_c != x_c)
                LOGGER.e(0, "Covariate count mismatch: REML state x_c=" + to_string(state.x_c) +
                            " vs current x_c=" + to_string(x_c) +
                            ". Use the same --qcovar/--covar as during --save-reml.");

            // Deterministic y_adj = y - X*b
            y_vec = compute_y_adj_deterministic(y_vec, X_design, state.b);
        } else {
            const bool   reuse_basis = !r.woodbury_reuse.empty();
            const string grm_pfx     = reuse_basis ? string() : r.grm;
            if (reuse_basis)
                LOGGER.i(0, "Running inline REML with the Woodbury basis reused from [" +
                            r.woodbury_reuse + "] (no GRM is read) ...");
            else
                LOGGER.i(0, "Running inline REML using GRM [" + grm_pfx + "] ...");

            const double grm_chunked_budget = reuse_basis ? 0.0 : r.grm_chunked_budget.value_or(0.0);
            bool grm_chunked = grm_chunked_budget > 0.0;
            bool grm_fp16 = !reuse_basis && r.grm_fp16;
            std::optional<WoodburyBasisSpec> woodbury_basis_eff = r.woodbury_basis;

            if (fast_preset) {
                vector<string> fast_notes;
                if (!reuse_basis) {
                    if (!grm_chunked && !grm_fp16) {
                        grm_fp16 = true;
                        fast_notes.push_back("grm-fp16=on");
                    }
                    if (!woodbury_basis_eff.has_value()) {
                        woodbury_basis_eff = WoodburyBasisSpec{WoodburyBasisSpec::Kind::MP, 0};
                        fast_notes.push_back("woodbury-basis=MP");
                    }
                }
                if (fast_enabled_mlma) {
                    fast_notes.push_back("fast-mlma=2slice");
                }
                if (!fast_notes.empty()) {
                    string summary;
                    for (size_t i = 0; i < fast_notes.size(); ++i) {
                        if (i > 0) summary += ", ";
                        summary += fast_notes[i];
                    }
                    LOGGER.i(0, "[--fast] enabled: " + summary + ".");
                }
            }

            if (grm_chunked && grm_fp16)
                LOGGER.e(0, "--grm-fp16 and --grm-chunked-budget are mutually exclusive: "
                            "--grm-fp16 holds the packed fp16 lower triangle in memory, whereas "
                            "--grm-chunked-budget streams float32 bands from disk.");

            if (!reuse_basis) {
                const auto resolved_grm = gcta_grm_io::resolve_grm_input(grm_pfx, grm_fp16);
                if (resolved_grm.format == gcta_grm_io::GrmDiskFormat::F16 &&
                    !grm_chunked && !grm_fp16 && r.alg != RemlAlg::Fisher) {
                    grm_fp16 = true;
                    LOGGER.i(0, "[auto-detect] Found [" + resolved_grm.data_path + "]; enabling --grm-fp16.");
                }
            }

            // REML tuning parameters. Parsed here (ahead of the GRM load below,
            // not after it as before) because woodbury_basis_mem_budget_gb feeds
            // the chunk-size solve immediately below, which needs to run before
            // committing to a chunked vs. dense load. Also lets the
            // svd_nystrom/etc. validation checks fail fast, ahead of
            // the (potentially expensive) GRM read, instead of after it.
            const bool   have_wb                      = woodbury_basis_eff.has_value();
            const bool   is_wb_mp                     = have_wb && woodbury_basis_eff->kind == WoodburyBasisSpec::Kind::MP;
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
                ctx.grm_chunk_rows_from_budget = chunked_grm.chunk_rows;
                ctx.grm_tile_reader = std::move(chunked_grm.reader);
                ctx.grm_tile_reader_f32 = std::move(chunked_grm.reader_f32);
                chunked_grm.file.reset();
            } else {
                ctx.A[0] = std::move(G_n);
            }
            ctx.grm_N.resize(1, 1);
            ctx.grm_N(0, 0) = m_all;
            ctx.r_indx   = {0, 1};
            ctx.var_name = {"V(G)", "V(e)"};
            ctx.hsq_name = {"V(G)/Vp"};
            ctx.out      = out_prefix;

            ctx.reml_mtd                 = r.alg;
            ctx.reml_max_iter            = reml_maxit;
            ctx.reml_inv_mtd             = 0;  // LLT
            ctx.reml_diagV_adj           = r.diagV_adj;
            ctx.reml_force_dense_vi      = r.force_dense_v;
            ctx.woodbury_basis           = woodbury_basis_eff;
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
                else {
                    ctx.Uk = load_pca_warm_start(r.woodbury_warm_start, analysis_ids, basis_file_override);
                    ctx.Uk_is_transposed = false;
                }
            }

            LoadedBasisInfo reuse_info;
            StreamingReuseHandle streaming_reuse_info;
            if (reuse_basis) {
                const bool use_streaming_reuse = reml_only && !r.reml_standalone;
                if (use_streaming_reuse) {
                    streaming_reuse_info = preload_woodbury_subspace_streaming(
                        r.woodbury_reuse, ctx, id_hash, basis_file_override, 256ULL << 20);
                    reuse_info.tun3_path  = streaming_reuse_info.tun3_path;
                    reuse_info.basis_hash = streaming_reuse_info.basis_hash;
                } else {
                    reuse_info = load_woodbury_basis_into_ctx(
                        r.woodbury_reuse, ctx, id_hash, basis_file_override, 256ULL << 20);
                }
            }

            reml::compute(ctx, r.priors, r.priors_var, no_constrain);
            ctx.grm_tile_reader = nullptr;
            ctx.grm_tile_reader_f32 = nullptr;

            LOGGER.i(0, "Inline REML complete. Variance components:");
            for (size_t ci = 0; ci < ctx.varcmp.size(); ++ci)
                LOGGER.i(0, "  " + ctx.var_name[ci] + " = " + to_string(ctx.varcmp[ci]));

            write_hsq_from_ctx(out_prefix, ctx);

            if (reml_only) {
                const string save_reml_file = out_prefix + ".reml";
                LOGGER.i(0, "Saving REML state to [" + save_reml_file + "] ...");
                LOGGER.ts("save_reml");
                writeRemlStateFromCtx(save_reml_file, ctx, no_adj_covar, id_hash,
                                      reuse_info.tun3_path, reuse_info.basis_hash,
                                      r.reml_standalone,
                                      streaming_reuse_info.active ? &streaming_reuse_info : nullptr,
                                      256ULL << 20);
                LOGGER.i(0, "REML state saved in " + to_string(LOGGER.tp("save_reml")) + " seconds.");
                LOGGER.i(0, "REML estimation completed. Use --load-reml " + save_reml_file +
                            " to perform association tests.");
                return;
            }

            // Deterministic y_adj = y - X*b
            y_vec = compute_y_adj_deterministic(y_vec, X_design, ctx.b.cast<float>());
            use_inline_ctx = true;

            const string out_file = out_prefix + ".mlma";
            try {
                gcta_io::AsyncSeqWriter mlma_writer(out_file, 16ULL << 20); // 2 x 16 MiB
                run_mlma_stream_association(ctx, y_vec, w_sqrt, geno.get(), marker.get(), n, log_pval,
                                            mlma_writer, mlma_tile_budget_gb, fast_mlma_mode,
                                            mlma_tuning, mlma_prof_ptr);
                mlma_writer.finish();
            } catch (const std::exception& e) {
                LOGGER.e(0, e.what());
            }
            log_mlma_profile();
            LOGGER << "\nAssociation results saved to [" << out_file << "]." << std::endl;
        }

        if (use_inline_ctx) {
            return;
        }

        const string out_file = out_prefix + ".mlma";
        try {
            gcta_io::AsyncSeqWriter mlma_writer(out_file, 16ULL << 20); // 2 x 16 MiB
            run_mlma_stream_association(state, y_vec, w_sqrt, geno.get(), marker.get(), n, log_pval,
                                        mlma_writer, mlma_tile_budget_gb, fast_mlma_mode,
                                        mlma_tuning, mlma_prof_ptr);
            mlma_writer.finish();
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }
        log_mlma_profile();
        LOGGER << "\nAssociation results saved to [" << out_file << "]." << std::endl;
}