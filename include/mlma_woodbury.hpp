#pragma once
/*
 * Shared Woodbury-MLMA types and helpers used by both the classic gcta MLMA
 * path (main/mlm_assoc.cpp) and the streaming MLMA path (src/MLMA_stream.cpp).
 *
 * Layout convention: Uk_f is stored as k×n (rows=k, cols=n) — transposed
 * relative to the n×k eigenvector matrix — so that Uk_f * x  produces a k×1
 * product without an explicit .transpose() call, which is cache-friendlier for
 * the n×BLOCK batch products in the hot MLMA loop.
 *
 * All helper functions below assume this k×n layout.
 */

#include <Eigen/Dense>
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <unistd.h>
#endif
#include "Logger.h"
#include "ModeTypes.hpp"
#include "main/StatFunc.h"
#include "cpu.h"
#include "mem.hpp"

// ---------------------------------------------------------------------------
// WoodburyMLMACache
// Float-precision factors sufficient to apply V^{-1} via the Woodbury identity.
// Uk_f is k×n (transposed for GEMM efficiency).
// ---------------------------------------------------------------------------
struct WoodburyMLMACache {
    Eigen::MatrixXf Uk_f;          // k×n  (transposed eigenvectors)
    Eigen::VectorXf ck_f;          // k    (Woodbury coefficients)
    Eigen::VectorXf sqrt_ck_f;     // k    cwiseSqrt(ck_f); safe because ck >= 0
    float           sigma2_eff_f = 0.0f;
};

enum class MlmaReorderPolicy { Auto = 0, Off = 1, On = 2 };

struct MlmaTuningOptions {
    int p_budget_mb = 0;         // 0 = auto (576 MiB baseline + headroom-aware 1-pass)
    int q_rows = 8192;           // max q_rows per Q tile
    int snp_tile = 0;            // 0 = auto from p_budget_mb
    MlmaReorderPolicy reorder_q = MlmaReorderPolicy::Auto;
    int block_size = 0;          // 0 = auto from --stream-budget / resolve_mlma_block_size
    int sample_tile = 8192;      // std_i8 X_panel_f32 sample tile
    int ukx_budget_mb = 160;     // std_i8 UkX_scratch budget
    int tun3_slot_mb = 256;      // TUN3 AsyncSeqReader slot size
    bool profile = false;
};

struct MlmaProfileStats {
    double reml_load_s = 0.0;
    double q_build_s = 0.0;
    double q_reorder_s = 0.0;
    double decode_s = 0.0;
    double xvx_gemm_s = 0.0;
    double xvx_reduce_s = 0.0;
    double format_write_s = 0.0;
    uint64_t blocks = 0;
    uint64_t valid_snps = 0;
    uint64_t total_snps = 0;
    uint64_t refined_snps = 0;
    int q_tiles = 0;
    int block_size = 0;
};

// ---------------------------------------------------------------------------
// woodbury_apply_Vi_f
// Apply V^{-1} to a float vector:
//   V^{-1} v = (v − Uk^T (ck ⊙ (Uk v))) / σ²_eff
// Uk is k×n so Uk*v produces k×1 directly (no explicit transpose needed).
// ---------------------------------------------------------------------------
inline Eigen::VectorXf woodbury_apply_Vi_f(const WoodburyMLMACache& wb,
                                            const Eigen::VectorXf& v)
{
    Eigen::VectorXf Ukv = wb.Uk_f * v;                    // k×n * n×1 = k×1
    Ukv.array() *= wb.ck_f.array();
    return (v - wb.Uk_f.transpose() * Ukv) / wb.sigma2_eff_f;  // n×1
}

// ---------------------------------------------------------------------------
// woodbury_apply_Vi_mat_f
// Apply V^{-1} to each column of a float matrix.
// ---------------------------------------------------------------------------
inline Eigen::MatrixXf woodbury_apply_Vi_mat_f(const WoodburyMLMACache& wb,
                                                const Eigen::MatrixXf& M)
{
    Eigen::MatrixXf UkM = wb.Uk_f * M;                    // k×n * n×p = k×p
    UkM.array().colwise() *= wb.ck_f.array();
    return (M - wb.Uk_f.transpose() * UkM) / wb.sigma2_eff_f;  // n×p
}

// ---------------------------------------------------------------------------
// woodbury_xvx_diag_block
// Compute diag(X^T V^{-1} X) for the first bs columns of X_block using:
//   x^T V^{-1} x = (||x||² − ||sqrt_ck ⊙ (Uk x)||²) / σ²_eff
// Result is clamped to >= 0 to guard against floating-point cancellation.
// Uses pre-allocated UkX_scratch (k x BLOCK) and multi-threaded cblas_sgemm.
// ---------------------------------------------------------------------------
inline void woodbury_xvx_diag_block(const WoodburyMLMACache& wb,
                                     const Eigen::MatrixXf& X_block, Eigen::Index bs,
                                     Eigen::VectorXf& xvx_diag,
                                     Eigen::MatrixXf& UkX_scratch,
                                     const float* precomputed_x_norm = nullptr)
{
    const int k = static_cast<int>(wb.Uk_f.rows());
    const int n = static_cast<int>(wb.Uk_f.cols());
    const int bs_int = static_cast<int>(bs);
    if (bs_int <= 0) return;

    // cblas_sgemm: Uk_f (k×n) * X_block (n×bs) -> UkX_scratch (k×bs)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                k, bs_int, n,
                1.0f, wb.Uk_f.data(), k,
                X_block.data(), n,
                0.0f, UkX_scratch.data(), k);

    const float* const sqrt_ck_ptr = wb.sqrt_ck_f.data();
    const float* const ukx_base    = UkX_scratch.data();

    // Fuse sqrt_ck scaling directly into the parallel squared-norm pass so we
    // never execute a single-threaded pass or dirty UkX_scratch with dead stores.
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < bs_int; ++j) {
        const float x_norm = precomputed_x_norm
            ? precomputed_x_norm[j]
            : X_block.col(j).squaredNorm();
        const float* const ukx_col = ukx_base + static_cast<size_t>(j) * static_cast<size_t>(k);
        double ukx_norm_d = 0.0;
        #pragma omp simd reduction(+:ukx_norm_d)
        for (int r = 0; r < k; ++r) {
            const float scaled = ukx_col[r] * sqrt_ck_ptr[r];
            ukx_norm_d += static_cast<double>(scaled) * static_cast<double>(scaled);
        }
        const float ukx_norm = static_cast<float>(ukx_norm_d);
        const float val      = (x_norm - ukx_norm) / wb.sigma2_eff_f;
        xvx_diag[j] = std::max(0.0f, val);
    }
}

inline void woodbury_xvx_diag_block(const WoodburyMLMACache& wb,
                                     const Eigen::MatrixXf& X_block, Eigen::Index bs,
                                     Eigen::VectorXf& xvx_diag)
{
    const int k = static_cast<int>(wb.Uk_f.rows());
    Eigen::MatrixXf UkX_scratch(k, bs);
    woodbury_xvx_diag_block(wb, X_block, bs, xvx_diag, UkX_scratch);
}

// ---------------------------------------------------------------------------
// woodbury_xvx_diag_block_i8
// Compute diag(X^T V^{-1} X) directly from an uncentered int8_t genotype
// block G_block_i8 (n x bs, column-major, stride ldG >= n) with raw additive
// calls {0, 1, 2} (missing -> 0), avoiding the 4x larger float X_block:
//   Uk * x_j = Uk * diag(w_sqrt) * g_j - mu_j * u_sum
//              + mu_j * sum_{s in M_j} w_sqrt[s] * Uk(:, s)
// where u_sum = Uk * w_sqrt is precomputed once for all blocks.
// Cache-blocked as j0 outer (kSnpTile sized so k x kSnpTile UkX_scratch fits in
// <= ~160 MiB) -> s0 middle (kSampleTile = 8192): each G_block_i8 slice is
// converted into X_panel_f32 once and fed to a full-k cblas_sgemm.
// ---------------------------------------------------------------------------
inline void woodbury_xvx_diag_block_i8(
    const WoodburyMLMACache& wb,
    const Eigen::VectorXf& u_sum,
    const Eigen::VectorXf& w_sqrt,
    bool unit_weights,
    const int8_t* G_block_i8,
    int ldG,
    int bs,
    const std::vector<uint8_t>& valid_v,
    const std::vector<float>& mu_v,
    const std::vector<float>& x_norm_v,
    const std::vector<std::vector<int>>& snp_miss_samples,
    Eigen::VectorXf& xvx_diag,
    Eigen::MatrixXf& UkX_scratch,
    AlignedRawBuffer<float, 64>& X_panel_f32,
    bool block_has_miss = true,
    std::vector<double>* ukx_norm_sq_scratch = nullptr,
    int sample_tile_opt = 8192,
    int ukx_budget_mb_opt = 160,
    MlmaProfileStats* prof = nullptr)
{
    (void)ukx_norm_sq_scratch;
    using Clock = std::chrono::steady_clock;
    const int k = static_cast<int>(wb.Uk_f.rows());
    const int n = static_cast<int>(wb.Uk_f.cols());
    if (bs <= 0) return;

    if (k <= 0 || n <= 0) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < bs; ++j) {
            xvx_diag[j] = valid_v[j]
                ? std::max(0.0f, x_norm_v[j] / wb.sigma2_eff_f)
                : 0.0f;
        }
        return;
    }

    const int kSampleTile = std::max(256, sample_tile_opt);
    const size_t kUkXTileBudgetBytes = static_cast<size_t>(std::max(16, ukx_budget_mb_opt)) << 20;
    const size_t raw_cols = kUkXTileBudgetBytes / (static_cast<size_t>(k) * sizeof(float));
    const int rounded_cols = static_cast<int>((raw_cols / 64ULL) * 64ULL);
    const int kSnpTile = std::min(bs, std::clamp(rounded_cols, 1024, 4096));

    if (UkX_scratch.rows() < k || UkX_scratch.cols() < kSnpTile) {
        UkX_scratch.resize(k, kSnpTile);
    }
    const gcta_blas_int ldUkX = static_cast<gcta_blas_int>(UkX_scratch.rows());

    auto calc_ld_panel = [](int r) -> int {
        int ld = ((r + 15) / 16) * 16;
        if ((static_cast<size_t>(ld) * sizeof(float)) % 4096 == 0) ld += 16;
        return std::max(r, ld);
    };
    const size_t panel_need =
        static_cast<size_t>(calc_ld_panel(std::min(n, kSampleTile))) *
        static_cast<size_t>(kSnpTile);
    if (X_panel_f32.size() < panel_need) {
        X_panel_f32.resize(panel_need);
    }

    const float* const w_ptr       = w_sqrt.data();
    const float* const u_sum_ptr   = u_sum.data();
    const float* const sqrt_ck_ptr = wb.sqrt_ck_f.data();
    const float* const uk_base     = wb.Uk_f.data();
    float* const       ukx_buf     = UkX_scratch.data();

    for (int j0 = 0; j0 < bs; j0 += kSnpTile) {
        const int cols = std::min(kSnpTile, bs - j0);

        const auto t_gemm0 = prof ? Clock::now() : Clock::time_point{};
        for (int s0 = 0; s0 < n; s0 += kSampleTile) {
            const int rows = std::min(kSampleTile, n - s0);
            const int ldPanel = calc_ld_panel(rows);
            const float beta = (s0 == 0) ? 0.0f : 1.0f;
            const float* const uk_slab = uk_base + static_cast<size_t>(s0) * static_cast<size_t>(k);

            if (unit_weights) {
                #pragma omp parallel for schedule(static)
                for (int c = 0; c < cols; ++c) {
                    const int8_t* src = G_block_i8 +
                        static_cast<size_t>(j0 + c) * static_cast<size_t>(ldG) + s0;
                    float* dst = X_panel_f32.data() +
                        static_cast<size_t>(c) * static_cast<size_t>(ldPanel);
                    #pragma omp simd
                    for (int r = 0; r < rows; ++r) {
                        dst[r] = static_cast<float>(src[r]);
                    }
                }
            } else {
                const float* w_slice = w_ptr + s0;
                #pragma omp parallel for schedule(static)
                for (int c = 0; c < cols; ++c) {
                    const int8_t* src = G_block_i8 +
                        static_cast<size_t>(j0 + c) * static_cast<size_t>(ldG) + s0;
                    float* dst = X_panel_f32.data() +
                        static_cast<size_t>(c) * static_cast<size_t>(ldPanel);
                    #pragma omp simd
                    for (int r = 0; r < rows; ++r) {
                        dst[r] = static_cast<float>(src[r]) * w_slice[r];
                    }
                }
            }

            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<gcta_blas_int>(k),
                        static_cast<gcta_blas_int>(cols),
                        static_cast<gcta_blas_int>(rows),
                        1.0f,
                        uk_slab, static_cast<gcta_blas_int>(k),
                        X_panel_f32.data(), static_cast<gcta_blas_int>(ldPanel),
                        beta,
                        ukx_buf, ldUkX);
        }
        if (prof) {
            prof->xvx_gemm_s += std::chrono::duration<double>(Clock::now() - t_gemm0).count();
        }

        const auto t_red0 = prof ? Clock::now() : Clock::time_point{};
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < cols; ++c) {
            const int j = j0 + c;
            if (!valid_v[j]) {
                xvx_diag[j] = 0.0f;
                continue;
            }

            float* const ukx_col = ukx_buf + static_cast<size_t>(c) * static_cast<size_t>(ldUkX);
            const float mu_j = mu_v[j];

            if (block_has_miss) {
                const auto& miss_list = snp_miss_samples[j];
                if (!miss_list.empty()) {
                    if (unit_weights) {
                        for (int s : miss_list) {
                            const float* uk_s = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k);
                            #pragma omp simd
                            for (int r = 0; r < k; ++r) {
                                ukx_col[r] += mu_j * uk_s[r];
                            }
                        }
                    } else {
                        for (int s : miss_list) {
                            const float scale = mu_j * w_ptr[s];
                            const float* uk_s = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k);
                            #pragma omp simd
                            for (int r = 0; r < k; ++r) {
                                ukx_col[r] += scale * uk_s[r];
                            }
                        }
                    }
                }
            }

            double ukx_norm_d = 0.0;
            #pragma omp simd reduction(+:ukx_norm_d)
            for (int r = 0; r < k; ++r) {
                const float centered = ukx_col[r] - mu_j * u_sum_ptr[r];
                const float scaled   = centered * sqrt_ck_ptr[r];
                ukx_norm_d += static_cast<double>(scaled) * static_cast<double>(scaled);
            }
            const float ukx_norm = static_cast<float>(ukx_norm_d);
            const float val      = (x_norm_v[j] - ukx_norm) / wb.sigma2_eff_f;
            xvx_diag[j] = std::max(0.0f, val);
        }
        if (prof) {
            prof->xvx_reduce_s += std::chrono::duration<double>(Clock::now() - t_red0).count();
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizedWoodburyUk & woodbury_xvx_diag_block_fast_i8 (--fast-mlma)
// Pre-quantizes w_sqrt ⊙ sqrt(c_k) ⊙ Uk into K-tiled INT8 panels (q_rows x n,
// with q_rows <= 4096) so the N x BLOCK projection U_k^T G_block_i8 runs as
// a u8s8s32 GEMM (aocl_gemm_u8s8s32os32 on Zen 4 VNNI with pre-reordered B)
// without Uk_f DRAM thrashing or K x BLOCK UkX_scratch, matched quantized
// centering u_sum_q, and analytical quadratic quantization de-biasing.
//
// 2-slice tiles are stored split ([0..r_count-1] = qh, [r_count..2*r_count-1] = ql)
// for unit-stride SIMD in both quantization and P_i32 reduction.
// ---------------------------------------------------------------------------
struct QuantizedWoodburyTile {
    int r_begin = 0;
    int r_count = 0;
    int q_rows = 0;       // is_2slice ? 2 * r_count : r_count
    int q_ld = 0;         // 64B-padded, 4K-alias-free column stride of Q
    bool is_2slice = true;
    bool has_lo_refine = false;
    int q_lo_ld = 0;
    AlignedUniquePtr<int8_t, 64> Q;                       // q_ld x n, col-major (first q_rows active)
    size_t q_bytes = 0;
    AlignedUniquePtr<int8_t, 64> Q_lo_refine;             // q_lo_ld x n, single lo slice for Adaptive Stage 2
    size_t q_lo_bytes = 0;
    AlignedUniquePtr<int8_t, 64> Q_reordered;             // AOCL pre-reordered B buffer (null if unused)
    size_t q_reordered_bytes = 0;
    std::vector<double> alpha_hi;                         // r_count
    std::vector<double> alpha_lo;                         // (is_2slice || has_lo_refine) ? r_count : 0
    std::vector<double> u_sum_q;                          // r_count (Stage-1 centering sum)
    std::vector<double> u_sum_q_2s;                       // r_count (has_lo_refine Stage-2 centering sum)
};

struct QuantizedWoodburyUk {
    int k = 0;
    int n = 0;
    FastMlmaMode mode = FastMlmaMode::Off;
    double delta_quant = 0.0;
    double delta_quant_2s = 0.0;
    std::vector<QuantizedWoodburyTile> tiles;

    size_t total_q_bytes() const {
        size_t total = 0;
        for (const auto& t : tiles) total += t.q_bytes + t.q_lo_bytes;
        return total;
    }

    size_t total_q_reordered_bytes() const {
        size_t total = 0;
        for (const auto& t : tiles) total += t.q_reordered_bytes;
        return total;
    }
};

// Pre-reorder each tile.Q once via AOCL-DLP u8s8s32 reorder API (called after
// Uk_f has been freed so Q_reordered does not coexist with Uk_f in RAM).
// No-op when !GCTA_HAS_AOCL_S8_GEMM or when n exceeds the int32 K-tile cap.
inline void reorder_quantized_woodbury_uk(QuantizedWoodburyUk& q_uk)
{
    constexpr int kAoclMaxSampleTile = 8388608;
    if (q_uk.n <= 0 || q_uk.n > kAoclMaxSampleTile) return;
    for (auto& tile : q_uk.tiles) {
        if (tile.q_rows <= 0 || !tile.Q) continue;
        const size_t sz = gcta_u8s8s32_reorder_b_size(
            static_cast<gcta_blas_int>(tile.q_rows),
            static_cast<gcta_blas_int>(q_uk.n));
        if (sz == 0) continue;
        tile.Q_reordered = make_aligned_uninitialized<int8_t, 64>(sz);
        if (!tile.Q_reordered) {
            LOGGER.w(0, "Failed to allocate pre-reordered Q tile; falling back to unpacked Q.");
            continue;
        }
        tile.q_reordered_bytes = sz;
        gcta_u8s8s32_reorder_b(
            static_cast<gcta_blas_int>(tile.q_rows),
            static_cast<gcta_blas_int>(q_uk.n),
            tile.Q.get(),
            static_cast<gcta_blas_int>(tile.q_ld > 0 ? tile.q_ld : tile.q_rows),
            tile.Q_reordered.get());
    }
}

namespace mlma_woodbury_detail {

inline bool parse_uint64_trimmed(std::string_view sv, uint64_t& out) noexcept
{
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t' || sv.front() == '\r' || sv.front() == '\n')) {
        sv.remove_prefix(1);
    }
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' || sv.back() == '\r' || sv.back() == '\n')) {
        sv.remove_suffix(1);
    }
    if (sv.empty()) return false;
    const char* const beg = sv.data();
    const char* const end = beg + sv.size();
    uint64_t val = 0;
    const auto [ptr, ec] = std::from_chars(beg, end, val, 10);
    if (ec != std::errc{} || ptr != end) return false;
    out = val;
    return true;
}

inline bool read_uint64_file(const std::string& path, uint64_t& out)
{
    std::ifstream in(path);
    if (!in) return false;
    std::string content;
    if (!std::getline(in, content)) return false;
    return parse_uint64_trimmed(content, out);
}

inline bool read_proc_kb_line(const char* path, std::string_view key, uint64_t& bytes_out)
{
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() <= key.size() || std::string_view(line).substr(0, key.size()) != key) continue;
        std::string_view rest = std::string_view(line).substr(key.size());
        while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.remove_prefix(1);
        const char* const beg = rest.data();
        const char* const end = beg + rest.size();
        uint64_t kb = 0;
        const auto [ptr, ec] = std::from_chars(beg, end, kb, 10);
        if (ec == std::errc{} && ptr != beg) {
            if (kb > (std::numeric_limits<uint64_t>::max() >> 10)) return false;
            bytes_out = kb << 10;
            return true;
        }
        return false;
    }
    return false;
}

inline bool read_env_uint64(const char* name, uint64_t& val) noexcept
{
    const char* const raw = std::getenv(name);
    if (raw == nullptr) return false;
    return parse_uint64_trimmed(raw, val);
}

} // namespace mlma_woodbury_detail

// Return total AOCL pre-reorder buffer size in bytes across all tiles of q_uk,
// or 0 when pre-reordering is unavailable/unsupported for this kernel/size.
inline size_t estimate_quantized_woodbury_reorder_bytes(const QuantizedWoodburyUk& q_uk)
{
    constexpr int kAoclMaxSampleTile = 8388608;
    if (q_uk.n <= 0 || q_uk.n > kAoclMaxSampleTile) return 0;
    size_t total = 0;
    for (const auto& tile : q_uk.tiles) {
        if (tile.q_rows <= 0 || !tile.Q) continue;
        const size_t sz = gcta_u8s8s32_reorder_b_size(
            static_cast<gcta_blas_int>(tile.q_rows),
            static_cast<gcta_blas_int>(q_uk.n));
        if (sz == 0) return 0;
        total += sz;
    }
    return total;
}

// Conservative estimate of remaining usable memory headroom in bytes on Linux
// (minimum across /proc/meminfo MemAvailable, cgroup v2/v1 limits, and Slurm
// node/CPU limits minus current VmRSS). Returns 0 if unknown or non-Linux.
inline uint64_t detect_available_memory_bytes()
{
#if defined(__linux__)
    bool has_bound = false;
    uint64_t min_avail = std::numeric_limits<uint64_t>::max();
    auto tighten = [&](uint64_t headroom) noexcept {
        has_bound = true;
        min_avail = std::min(min_avail, headroom);
    };

    if (uint64_t mem_avail = 0;
        mlma_woodbury_detail::read_proc_kb_line("/proc/meminfo", "MemAvailable:", mem_avail)) {
        tighten(mem_avail);
    }

    bool found_cg_v2 = false;
    {
        std::ifstream cg_in("/proc/self/cgroup");
        std::string line;
        while (std::getline(cg_in, line)) {
            if (line.size() < 3 || line[0] != '0' || line[1] != ':' || line[2] != ':') continue;
            std::string rel = line.substr(3);
            while (!rel.empty() && (rel.back() == '\r' || rel.back() == '\n')) rel.pop_back();
            std::string dir = (rel.empty() || rel == "/")
                ? std::string("/sys/fs/cgroup")
                : (std::string("/sys/fs/cgroup") + rel);
            for (;;) {
                uint64_t max_b = 0;
                uint64_t cur_b = 0;
                if (mlma_woodbury_detail::read_uint64_file(dir + "/memory.max", max_b) &&
                    mlma_woodbury_detail::read_uint64_file(dir + "/memory.current", cur_b) &&
                    max_b > 0) {
                    found_cg_v2 = true;
                    tighten(max_b > cur_b ? (max_b - cur_b) : 0ULL);
                }
                if (dir == "/sys/fs/cgroup") break;
                const size_t slash = dir.rfind('/');
                if (slash == std::string::npos || slash < 13) break; // stop at /sys/fs/cgroup
                dir.resize(slash);
            }
            break;
        }
    }
    if (!found_cg_v2) {
        uint64_t lim_b = 0;
        uint64_t cur_b = 0;
        if (mlma_woodbury_detail::read_uint64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes", lim_b) &&
            mlma_woodbury_detail::read_uint64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes", cur_b) &&
            lim_b > 0 && lim_b < (1ULL << 60)) {
            tighten(lim_b > cur_b ? (lim_b - cur_b) : 0ULL);
        }
    }

    uint64_t slurm_mib = 0;
    if (!mlma_woodbury_detail::read_env_uint64("SLURM_MEM_PER_NODE", slurm_mib)) {
        uint64_t mem_per_cpu = 0;
        uint64_t cpus = 0;
        if (mlma_woodbury_detail::read_env_uint64("SLURM_MEM_PER_CPU", mem_per_cpu) &&
            (mlma_woodbury_detail::read_env_uint64("SLURM_CPUS_ON_NODE", cpus) ||
             mlma_woodbury_detail::read_env_uint64("SLURM_CPUS_PER_TASK", cpus)) &&
            mem_per_cpu > 0 && cpus > 0 &&
            mem_per_cpu <= (std::numeric_limits<uint64_t>::max() / cpus)) {
            slurm_mib = mem_per_cpu * cpus;
        }
    }
    if (slurm_mib > 0 && slurm_mib <= (std::numeric_limits<uint64_t>::max() >> 20)) {
        const uint64_t slurm_bytes = slurm_mib << 20;
        uint64_t rss_bytes = 0;
        mlma_woodbury_detail::read_proc_kb_line("/proc/self/status", "VmRSS:", rss_bytes);
        tighten(slurm_bytes > rss_bytes ? (slurm_bytes - rss_bytes) : 0ULL);
    }

    return has_bound ? min_avail : 0ULL;
#else
    return 0ULL;
#endif
}

// Auto-enable AOCL pre-reordering of q_uk when multiple SNP blocks amortize the
// one-time reorder cost and remaining headroom exceeds 2x reorder size + 2 GiB.
inline bool should_auto_reorder_woodbury_uk(
    const QuantizedWoodburyUk& q_uk,
    uint32_t total_m,
    int block_size)
{
    const uint64_t reord_bytes = static_cast<uint64_t>(estimate_quantized_woodbury_reorder_bytes(q_uk));
    if (reord_bytes == 0) return false;
    const uint64_t bs = static_cast<uint64_t>(std::max(1, block_size));
    const uint64_t n_blocks = (static_cast<uint64_t>(total_m) + bs - 1ULL) / bs;
    if (n_blocks < 4) return false;
    const uint64_t avail = detect_available_memory_bytes();
    constexpr uint64_t kHeadroomReserve = 2ULL << 30;
    if (avail <= kHeadroomReserve) return false;
    return reord_bytes <= (avail - kHeadroomReserve) / 2ULL;
}

// Allocate evenly-balanced, 64-row-aligned, 4K-alias-free Q tiles covering [0, k)
// for the chosen FastMlmaMode. Returns k_hi (the split boundary between 2-slice and 1-slice rows).
inline int populate_quantized_woodbury_tiles(
    QuantizedWoodburyUk& out,
    int k,
    int n,
    FastMlmaMode mode,
    int max_q_rows = 8192)
{
    int k_hi = k;
    if (mode == FastMlmaMode::Slice1) {
        k_hi = 0;
    } else if (mode == FastMlmaMode::Hybrid || mode == FastMlmaMode::Adaptive) {
        k_hi = std::min(k, std::max(std::min(k, 100), k / 10));
    }

    const int raw_q = (max_q_rows <= 0) ? 8192 : max_q_rows;
    const int capped_q_rows = std::clamp(((raw_q + 63) / 64) * 64, 256, 32768);

    auto push_tiles = [&](int beg, int end, bool is_2slice) {
        const int total = end - beg;
        if (total <= 0) return;
        const int rc_max = is_2slice ? std::max(1, capped_q_rows / 2) : capped_q_rows;
        const int num_t = (total + rc_max - 1) / rc_max;
        const int rc_unit = is_2slice ? 32 : 64;
        const int total_u = (total + rc_unit - 1) / rc_unit;
        int r_cur = beg;
        auto append_tile = [&](int rc) {
            QuantizedWoodburyTile t;
            t.r_begin = r_cur;
            t.r_count = rc;
            t.is_2slice = is_2slice;
            t.q_rows = is_2slice ? (2 * rc) : rc;
            int q_ld = ((t.q_rows + 63) / 64) * 64;
            if (q_ld % 4096 == 0) q_ld += 64;
            t.q_ld = q_ld;
            t.q_bytes = static_cast<size_t>(t.q_ld) * static_cast<size_t>(n);
            t.Q = make_aligned_uninitialized<int8_t, 64>(t.q_bytes);
            if (t.q_bytes > 0 && !t.Q) {
                LOGGER.e(0, "Failed to allocate quantized Woodbury Q tile.");
            }
            t.alpha_hi.assign(static_cast<size_t>(rc), 0.0);
            if (is_2slice) t.alpha_lo.assign(static_cast<size_t>(rc), 0.0);
            t.u_sum_q.assign(static_cast<size_t>(rc), 0.0);
            if (!is_2slice && mode == FastMlmaMode::Adaptive) {
                t.has_lo_refine = true;
                t.q_lo_ld = q_ld;
                t.q_lo_bytes = t.q_bytes;
                t.Q_lo_refine = make_aligned_uninitialized<int8_t, 64>(t.q_lo_bytes);
                if (t.q_lo_bytes > 0 && !t.Q_lo_refine) {
                    LOGGER.e(0, "Failed to allocate quantized Woodbury Q_lo_refine tile.");
                }
                t.alpha_lo.assign(static_cast<size_t>(rc), 0.0);
                t.u_sum_q_2s.assign(static_cast<size_t>(rc), 0.0);
            }
            out.tiles.push_back(std::move(t));
            r_cur += rc;
        };
        for (int ti = 0; ti < num_t && r_cur < end; ++ti) {
            const int u_this = (total_u / num_t) + (ti < (total_u % num_t) ? 1 : 0);
            const int rem_rows = end - r_cur;
            const int rc_want = (ti + 1 == num_t)
                ? rem_rows
                : std::min(rem_rows, std::max(rc_unit, u_this * rc_unit));
            append_tile(std::clamp(rc_want, 1, rc_max));
        }
        while (r_cur < end) {
            append_tile(std::min(rc_max, end - r_cur));
        }
    };
    if (k_hi > 0) push_tiles(0, k_hi, true);
    if (k_hi < k) push_tiles(k_hi, k, false);
    return k_hi;
}

inline QuantizedWoodburyUk build_quantized_woodbury_uk(
    const WoodburyMLMACache& wb,
    const Eigen::VectorXf& w_sqrt,
    bool unit_weights,
    double S_w2,
    FastMlmaMode mode,
    int max_q_rows = 8192)
{
    QuantizedWoodburyUk out;
    const int k = static_cast<int>(wb.Uk_f.rows());
    const int n = static_cast<int>(wb.Uk_f.cols());
    out.k = k;
    out.n = n;
    out.mode = mode;
    if (k <= 0 || n <= 0 || mode == FastMlmaMode::Off) return out;

    const int k_hi = populate_quantized_woodbury_tiles(out, k, n, mode, max_q_rows);
    const int k_lo_cap = (mode == FastMlmaMode::Adaptive) ? k : k_hi;

    const float* const uk_base     = wb.Uk_f.data();
    const float* const sqrt_ck_ptr = wb.sqrt_ck_f.data();
    const float* const w_ptr       = w_sqrt.data();

    const int max_threads =
    #ifdef _OPENMP
        std::max(1, omp_get_max_threads());
    #else
        1;
    #endif

    // Phase 1: single parallel float32 SIMD max_abs scan over |Uk_f * ws| across
    // all k rows (without faulting tile.Q pages yet), scaled by sqrt_ck[r] once after.
    const size_t k_pad_f = (static_cast<size_t>(k) + 15) / 16 * 16;
    std::vector<float> thr_max(static_cast<size_t>(max_threads) * k_pad_f, 0.0f);
    std::vector<float> max_abs_raw(static_cast<size_t>(k), 0.0f);
    int p1_threads = 1;

    #pragma omp parallel
    {
        const int tid =
        #ifdef _OPENMP
            omp_get_thread_num();
        #else
            0;
        #endif
        #pragma omp single
        {
            p1_threads =
            #ifdef _OPENMP
                omp_get_num_threads();
            #else
                1;
            #endif
        }
        float* const lmax = thr_max.data() + static_cast<size_t>(tid) * k_pad_f;
        if (unit_weights) {
            #pragma omp for schedule(static)
            for (int s = 0; s < n; ++s) {
                const float* const uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k);
                #pragma omp simd
                for (int r = 0; r < k; ++r) {
                    const float av = std::fabs(uk_col[r]);
                    if (av > lmax[r]) lmax[r] = av;
                }
            }
        } else {
            #pragma omp for schedule(static)
            for (int s = 0; s < n; ++s) {
                const float* const uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k);
                const float ws = std::fabs(w_ptr[s]);
                #pragma omp simd
                for (int r = 0; r < k; ++r) {
                    const float av = std::fabs(uk_col[r]) * ws;
                    if (av > lmax[r]) lmax[r] = av;
                }
            }
        }

        #pragma omp for schedule(static)
        for (int r = 0; r < k; ++r) {
            float m = 0.0f;
            for (int t = 0; t < p1_threads; ++t) {
                const float v = thr_max[static_cast<size_t>(t) * k_pad_f + static_cast<size_t>(r)];
                if (v > m) m = v;
            }
            max_abs_raw[static_cast<size_t>(r)] = m;
        }
    }
    thr_max.clear();
    thr_max.shrink_to_fit();

    // Flat per-row float32 scales for vectorized Phase 2 quantization.
    std::vector<float> alpha_hi_f(static_cast<size_t>(k), 0.0f);
    std::vector<float> inv_hi_f(static_cast<size_t>(k), 0.0f);
    std::vector<float> alpha_lo_f(static_cast<size_t>(k_lo_cap), 0.0f);
    std::vector<float> inv_lo_f(static_cast<size_t>(k_lo_cap), 0.0f);

    const size_t num_tiles = out.tiles.size();
    for (size_t ti = 0; ti < num_tiles; ++ti) {
        auto& tile = out.tiles[ti];
        const int r0 = tile.r_begin;
        const int rc = tile.r_count;
        const bool need_lo = tile.is_2slice || tile.has_lo_refine;
        for (int rl = 0; rl < rc; ++rl) {
            const int r = r0 + rl;
            const double max_v = static_cast<double>(sqrt_ck_ptr[r])
                               * static_cast<double>(max_abs_raw[static_cast<size_t>(r)]);
            if (max_v > 1.0e-30) {
                const double ahi = max_v / 127.0;
                tile.alpha_hi[static_cast<size_t>(rl)] = ahi;
                alpha_hi_f[static_cast<size_t>(r)]     = static_cast<float>(ahi);
                inv_hi_f[static_cast<size_t>(r)]       = static_cast<float>(1.0 / ahi);
                if (need_lo) {
                    const double alo = ahi / 254.0;
                    tile.alpha_lo[static_cast<size_t>(rl)] = alo;
                    alpha_lo_f[static_cast<size_t>(r)]     = static_cast<float>(alo);
                    inv_lo_f[static_cast<size_t>(r)]       = static_cast<float>(1.0 / alo);
                }
            }
        }
    }

    constexpr int kReleaseSlab = 32768;
    const size_t k_pad64    = (static_cast<size_t>(k) + 7) / 8 * 8;
    const size_t k_lo_pad64 = (static_cast<size_t>(k_lo_cap) + 7) / 8 * 8;
    std::vector<int64_t> thr_sum_hi(static_cast<size_t>(max_threads) * k_pad64, 0);
    std::vector<int64_t> thr_sum_lo(static_cast<size_t>(max_threads) * k_lo_pad64, 0);
    double total_eps_sq = 0.0;
    double total_eps_2s_sq = 0.0;
    int p2_threads = 1;

    #if defined(__linux__) || defined(__APPLE__)
    const long raw_page_sz = sysconf(_SC_PAGESIZE);
    const uintptr_t page_sz = (raw_page_sz > 0) ? static_cast<uintptr_t>(raw_page_sz) : 0;
    #endif

    const float* const ahi_base = alpha_hi_f.data();
    const float* const ihi_base = inv_hi_f.data();
    const float* const alo_base = alpha_lo_f.data();
    const float* const ilo_base = inv_lo_f.data();

    #pragma omp parallel reduction(+:total_eps_sq,total_eps_2s_sq)
    {
        const int tid =
        #ifdef _OPENMP
            omp_get_thread_num();
        #else
            0;
        #endif
        #pragma omp single
        {
            p2_threads =
            #ifdef _OPENMP
                omp_get_num_threads();
            #else
                1;
            #endif
        }

        std::vector<int32_t> local_hi32(static_cast<size_t>(k), 0);
        std::vector<int32_t> local_lo32(static_cast<size_t>(k_lo_cap), 0);
        int64_t* const my_sum_hi = thr_sum_hi.data() + static_cast<size_t>(tid) * k_pad64;
        int64_t* const my_sum_lo = (k_lo_cap > 0)
            ? (thr_sum_lo.data() + static_cast<size_t>(tid) * k_lo_pad64)
            : nullptr;

        for (int s0 = 0; s0 < n; s0 += kReleaseSlab) {
            const int s_end = std::min(n, s0 + kReleaseSlab);

            #pragma omp for schedule(static) nowait
            for (int s = s0; s < s_end; ++s) {
                const float* const uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k);
                const float ws_f = unit_weights ? 1.0f : w_ptr[s];
                float sample_eps = 0.0f;
                float sample_eps_2s = 0.0f;

                for (size_t ti = 0; ti < num_tiles; ++ti) {
                    const auto& tile = out.tiles[ti];
                    const int r0 = tile.r_begin;
                    const int rc = tile.r_count;
                    const int q_ld = tile.q_ld;
                    const float* const uk_t = uk_col + r0;
                    const float* const sck  = sqrt_ck_ptr + r0;
                    const float* const ihi  = ihi_base + r0;
                    const float* const ahi  = ahi_base + r0;
                    int32_t* const     lhi  = local_hi32.data() + r0;
                    int8_t* const      q_hi = tile.Q.get() + static_cast<size_t>(s) * static_cast<size_t>(q_ld);

                    if (tile.is_2slice) {
                        int8_t* const      q_lo = q_hi + rc;
                        int32_t* const     llo  = local_lo32.data() + r0;
                        const float* const ilo  = ilo_base + r0;
                        const float* const alo  = alo_base + r0;
                        if (unit_weights) {
                            #pragma omp simd reduction(+:sample_eps,sample_eps_2s)
                            for (int rl = 0; rl < rc; ++rl) {
                                const float v    = uk_t[rl] * sck[rl];
                                const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                                const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                                const int32_t qh = static_cast<int32_t>(qh_f);
                                const int32_t ql = static_cast<int32_t>(ql_f);
                                q_hi[rl] = static_cast<int8_t>(qh);
                                q_lo[rl] = static_cast<int8_t>(ql);
                                lhi[rl] += qh;
                                llo[rl] += ql;
                                const float eps = std::fmaf(-alo[rl], ql_f, rem);
                                const float eps2 = eps * eps;
                                sample_eps += eps2;
                                sample_eps_2s += eps2;
                            }
                        } else {
                            #pragma omp simd reduction(+:sample_eps,sample_eps_2s)
                            for (int rl = 0; rl < rc; ++rl) {
                                const float v    = (uk_t[rl] * ws_f) * sck[rl];
                                const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                                const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                                const int32_t qh = static_cast<int32_t>(qh_f);
                                const int32_t ql = static_cast<int32_t>(ql_f);
                                q_hi[rl] = static_cast<int8_t>(qh);
                                q_lo[rl] = static_cast<int8_t>(ql);
                                lhi[rl] += qh;
                                llo[rl] += ql;
                                const float eps = std::fmaf(-alo[rl], ql_f, rem);
                                const float eps2 = eps * eps;
                                sample_eps += eps2;
                                sample_eps_2s += eps2;
                            }
                        }
                    } else if (tile.has_lo_refine) {
                        int8_t* const      q_lo = tile.Q_lo_refine.get() + static_cast<size_t>(s) * static_cast<size_t>(tile.q_lo_ld);
                        int32_t* const     llo  = local_lo32.data() + r0;
                        const float* const ilo  = ilo_base + r0;
                        const float* const alo  = alo_base + r0;
                        if (unit_weights) {
                            #pragma omp simd reduction(+:sample_eps,sample_eps_2s)
                            for (int rl = 0; rl < rc; ++rl) {
                                const float v    = uk_t[rl] * sck[rl];
                                const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                                const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                                const int32_t qh = static_cast<int32_t>(qh_f);
                                const int32_t ql = static_cast<int32_t>(ql_f);
                                q_hi[rl] = static_cast<int8_t>(qh);
                                q_lo[rl] = static_cast<int8_t>(ql);
                                lhi[rl] += qh;
                                llo[rl] += ql;
                                const float eps2 = std::fmaf(-alo[rl], ql_f, rem);
                                sample_eps += rem * rem;
                                sample_eps_2s += eps2 * eps2;
                            }
                        } else {
                            #pragma omp simd reduction(+:sample_eps,sample_eps_2s)
                            for (int rl = 0; rl < rc; ++rl) {
                                const float v    = (uk_t[rl] * ws_f) * sck[rl];
                                const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                                const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                                const int32_t qh = static_cast<int32_t>(qh_f);
                                const int32_t ql = static_cast<int32_t>(ql_f);
                                q_hi[rl] = static_cast<int8_t>(qh);
                                q_lo[rl] = static_cast<int8_t>(ql);
                                lhi[rl] += qh;
                                llo[rl] += ql;
                                const float eps2 = std::fmaf(-alo[rl], ql_f, rem);
                                sample_eps += rem * rem;
                                sample_eps_2s += eps2 * eps2;
                            }
                        }
                    } else {
                        if (unit_weights) {
                            #pragma omp simd reduction(+:sample_eps)
                            for (int rl = 0; rl < rc; ++rl) {
                                const float v    = uk_t[rl] * sck[rl];
                                const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                const int32_t qh = static_cast<int32_t>(qh_f);
                                q_hi[rl] = static_cast<int8_t>(qh);
                                lhi[rl] += qh;
                                const float eps = std::fmaf(-ahi[rl], qh_f, v);
                                sample_eps += eps * eps;
                            }
                        } else {
                            #pragma omp simd reduction(+:sample_eps)
                            for (int rl = 0; rl < rc; ++rl) {
                                const float v    = (uk_t[rl] * ws_f) * sck[rl];
                                const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                const int32_t qh = static_cast<int32_t>(qh_f);
                                q_hi[rl] = static_cast<int8_t>(qh);
                                lhi[rl] += qh;
                                const float eps = std::fmaf(-ahi[rl], qh_f, v);
                                sample_eps += eps * eps;
                            }
                        }
                    }
                }
                total_eps_sq += static_cast<double>(sample_eps);
                total_eps_2s_sq += static_cast<double>(sample_eps_2s);
            }

            // Fold slab int32 sums into thread-owned int64 accumulators before syncing.
            for (int r = 0; r < k; ++r) {
                my_sum_hi[r] += static_cast<int64_t>(local_hi32[static_cast<size_t>(r)]);
                local_hi32[static_cast<size_t>(r)] = 0;
            }
            for (int r = 0; r < k_lo_cap; ++r) {
                my_sum_lo[r] += static_cast<int64_t>(local_lo32[static_cast<size_t>(r)]);
                local_lo32[static_cast<size_t>(r)] = 0;
            }

            #pragma omp barrier
            #pragma omp single
            {
                #if defined(__linux__) || defined(__APPLE__)
                if (page_sz > 0) {
                    const uintptr_t col_start_addr = reinterpret_cast<uintptr_t>(
                        uk_base + static_cast<size_t>(s0) * static_cast<size_t>(k));
                    const uintptr_t col_end_addr   = reinterpret_cast<uintptr_t>(
                        uk_base + static_cast<size_t>(s_end) * static_cast<size_t>(k));
                    const uintptr_t page_beg = (col_start_addr + page_sz - 1) & ~(page_sz - 1);
                    const uintptr_t page_end = col_end_addr & ~(page_sz - 1);
                    if (page_end > page_beg) {
                    #if defined(__linux__)
                        ::madvise(reinterpret_cast<void*>(page_beg), page_end - page_beg, MADV_DONTNEED);
                    #elif defined(__APPLE__)
                        ::madvise(reinterpret_cast<void*>(page_beg), page_end - page_beg, MADV_FREE);
                    #endif
                    }
                }
                #endif
            }
        }
    }

    // Reduce per-thread int64 sums once at the end and populate tile.u_sum_q.
    for (size_t ti = 0; ti < num_tiles; ++ti) {
        auto& tile = out.tiles[ti];
        const int r0 = tile.r_begin;
        const int rc = tile.r_count;
        if (tile.is_2slice) {
            #pragma omp parallel for schedule(static) if (rc >= 512)
            for (int rl = 0; rl < rc; ++rl) {
                int64_t sh = 0;
                int64_t sl = 0;
                const size_t r = static_cast<size_t>(r0 + rl);
                for (int t = 0; t < p2_threads; ++t) {
                    sh += thr_sum_hi[static_cast<size_t>(t) * k_pad64 + r];
                    sl += thr_sum_lo[static_cast<size_t>(t) * k_lo_pad64 + r];
                }
                tile.u_sum_q[static_cast<size_t>(rl)] =
                    tile.alpha_hi[static_cast<size_t>(rl)] * static_cast<double>(sh)
                  + tile.alpha_lo[static_cast<size_t>(rl)] * static_cast<double>(sl);
            }
        } else if (tile.has_lo_refine) {
            #pragma omp parallel for schedule(static) if (rc >= 512)
            for (int rl = 0; rl < rc; ++rl) {
                int64_t sh = 0;
                int64_t sl = 0;
                const size_t r = static_cast<size_t>(r0 + rl);
                for (int t = 0; t < p2_threads; ++t) {
                    sh += thr_sum_hi[static_cast<size_t>(t) * k_pad64 + r];
                    sl += thr_sum_lo[static_cast<size_t>(t) * k_lo_pad64 + r];
                }
                const double ahi = tile.alpha_hi[static_cast<size_t>(rl)];
                const double alo = tile.alpha_lo[static_cast<size_t>(rl)];
                tile.u_sum_q[static_cast<size_t>(rl)]    = ahi * static_cast<double>(sh);
                tile.u_sum_q_2s[static_cast<size_t>(rl)] = ahi * static_cast<double>(sh) + alo * static_cast<double>(sl);
            }
        } else {
            #pragma omp parallel for schedule(static) if (rc >= 512)
            for (int rl = 0; rl < rc; ++rl) {
                int64_t sh = 0;
                const size_t r = static_cast<size_t>(r0 + rl);
                for (int t = 0; t < p2_threads; ++t) {
                    sh += thr_sum_hi[static_cast<size_t>(t) * k_pad64 + r];
                }
                tile.u_sum_q[static_cast<size_t>(rl)] =
                    tile.alpha_hi[static_cast<size_t>(rl)] * static_cast<double>(sh);
            }
        }
    }

    const double norm_denom = (n > 1) ? (S_w2 * static_cast<double>(n - 1) / static_cast<double>(n)) : S_w2;
    out.delta_quant    = (norm_denom > 0.0) ? (total_eps_sq / norm_denom) : 0.0;
    out.delta_quant_2s = (mode == FastMlmaMode::Adaptive && norm_denom > 0.0)
        ? (total_eps_2s_sq / norm_denom)
        : out.delta_quant;
    return out;
}

// ---------------------------------------------------------------------------
// Compact TUN3 16-bit hybrid encoding helpers (Row-Scaled INT16 qmax=32385
// when crest <= 24.0, IEEE FP16 bit-pattern when crest > 24.0)
// ---------------------------------------------------------------------------
inline constexpr float  kTun3Int16QMax       = 32385.0f; // 127 * 254 + 127 = 127 * 255
inline constexpr double kTun3CrestFp16Thresh = 24.0;

inline uint16_t wb_f32_to_fp16_bits(float x) noexcept {
#if defined(__FLT16_MANT_DIG__)
    return std::bit_cast<uint16_t>(static_cast<_Float16>(x));
#else
    const uint32_t u = std::bit_cast<uint32_t>(x);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t  exp  = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t       mant = u & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000u;
        const int shift = 14 - exp;
        const uint32_t round_bit = 1u << (shift - 1);
        const uint32_t sticky    = (mant & (round_bit - 1u)) != 0u;
        const uint32_t lsb       = (mant >> shift) & 1u;
        return static_cast<uint16_t>(sign | ((mant + round_bit - 1u + (sticky | lsb)) >> shift));
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    const uint32_t round_bit = 0x00001000u;
    const uint32_t sticky    = (mant & 0x00000FFFu) != 0u;
    const uint32_t lsb       = (mant >> 13) & 1u;
    uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    h += (mant & round_bit) && (sticky || lsb);
    return static_cast<uint16_t>(h);
#endif
}

inline float wb_fp16_bits_to_f32(uint16_t h) noexcept {
#if defined(__FLT16_MANT_DIG__)
    return static_cast<float>(std::bit_cast<_Float16>(h));
#else
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    uint32_t       mant = static_cast<uint32_t>(h) & 0x03FFu;
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            uint32_t e = 127 - 14;
            while ((mant & 0x0400u) == 0u) { mant <<= 1; --e; }
            mant &= 0x03FFu;
            u = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(u);
#endif
}

// ---------------------------------------------------------------------------
// QuantizedWoodburyStreamBuilder
// 1-pass streaming builder that consumes k x chunk_n uint16_t TUN3 sample
// slabs directly from AsyncSeqReader and populates QuantizedWoodburyUk::tiles
// WITHOUT ever allocating the 4*k*n float32 Uk_f matrix.
//
// Bulk INT16 rows (uk_row_scale[r] > 0) under 2slice + unit_weights decompose
// LOSSLESSLY into (qh, ql) in [-127, 127] via q16 = 254 * qh + ql with pure
// integer arithmetic.
// ---------------------------------------------------------------------------
class QuantizedWoodburyStreamBuilder {
public:
    QuantizedWoodburyStreamBuilder(
        int k,
        int n,
        FastMlmaMode mode,
        const float* sqrt_ck_ptr,
        const float* uk_row_scale,
        const Eigen::VectorXf& w_sqrt,
        bool unit_weights,
        double S_w2,
        const float* exact_max_abs_override = nullptr,
        const float* y_adj_for_uy = nullptr,
        int max_q_rows = 8192)
        : k_(k),
          n_(n),
          unit_weights_(unit_weights),
          has_exact_max_override_(exact_max_abs_override != nullptr),
          S_w2_(S_w2),
          sqrt_ck_(sqrt_ck_ptr, sqrt_ck_ptr + std::max(0, k)),
          uk_row_scale_(uk_row_scale, uk_row_scale + std::max(0, k)),
          w_ptr_(w_sqrt.data()),
          y_adj_ptr_(y_adj_for_uy)
    {
        out_.k = k;
        out_.n = n;
        out_.mode = mode;
        if (k <= 0 || n <= 0 || mode == FastMlmaMode::Off) return;

        k_hi_ = populate_quantized_woodbury_tiles(out_, k, n, mode, max_q_rows);
        k_lo_ = (mode == FastMlmaMode::Adaptive) ? k : k_hi_;

        const double w_max = unit_weights_
            ? 1.0
            : static_cast<double>(w_sqrt.cwiseAbs().maxCoeff());

        alpha_hi_f_.assign(static_cast<size_t>(k), 0.0f);
        inv_hi_f_.assign(static_cast<size_t>(k), 0.0f);
        alpha_lo_f_.assign(static_cast<size_t>(k_lo_), 0.0f);
        inv_lo_f_.assign(static_cast<size_t>(k_lo_), 0.0f);
        row_unit_f_.assign(static_cast<size_t>(k), 0.0f);

        const size_t num_tiles = out_.tiles.size();
        tile_all_i16_.assign(num_tiles, true);
        tile_fp16_rows_.resize(num_tiles);
        int max_rc = 0;

        for (size_t ti = 0; ti < num_tiles; ++ti) {
            auto& tile = out_.tiles[ti];
            const int r0 = tile.r_begin;
            const int rc = tile.r_count;
            max_rc = std::max(max_rc, rc);
            const bool need_lo = tile.is_2slice || tile.has_lo_refine;
            for (int rl = 0; rl < rc; ++rl) {
                const int r = r0 + rl;
                const float rs = uk_row_scale_[static_cast<size_t>(r)];
                const bool is_fp16 = (rs < 0.0f);
                if (is_fp16) {
                    tile_all_i16_[ti] = false;
                    tile_fp16_rows_[ti].push_back(rl);
                }

                const double raw_max_abs = is_fp16
                    ? static_cast<double>(-rs)
                    : (static_cast<double>(rs) * static_cast<double>(kTun3Int16QMax));

                if (!is_fp16 && unit_weights_ && !has_exact_max_override_) {
                    // Exact lossless alignment with on-disk qmax=32385 = 127*254 + 127 = 127*255.
                    const double unit_v = static_cast<double>(sqrt_ck_[static_cast<size_t>(r)])
                                        * static_cast<double>(rs);
                    row_unit_f_[static_cast<size_t>(r)] = static_cast<float>(unit_v);
                    if (unit_v > 1.0e-35) {
                        if (need_lo) {
                            const double alo = unit_v;
                            const double ahi = 254.0 * unit_v;
                            tile.alpha_hi[static_cast<size_t>(rl)] = ahi;
                            tile.alpha_lo[static_cast<size_t>(rl)] = alo;
                            alpha_hi_f_[static_cast<size_t>(r)]    = static_cast<float>(ahi);
                            inv_hi_f_[static_cast<size_t>(r)]      = static_cast<float>(1.0 / ahi);
                            alpha_lo_f_[static_cast<size_t>(r)]    = static_cast<float>(alo);
                            inv_lo_f_[static_cast<size_t>(r)]      = static_cast<float>(1.0 / alo);
                        } else {
                            const double ahi = 255.0 * unit_v;
                            tile.alpha_hi[static_cast<size_t>(rl)] = ahi;
                            alpha_hi_f_[static_cast<size_t>(r)]    = static_cast<float>(ahi);
                            inv_hi_f_[static_cast<size_t>(r)]      = static_cast<float>(1.0 / ahi);
                        }
                    }
                } else {
                    const double eff_max = has_exact_max_override_
                        ? static_cast<double>(exact_max_abs_override[r])
                        : (raw_max_abs * w_max);
                    const double max_v = static_cast<double>(sqrt_ck_[static_cast<size_t>(r)]) * eff_max;
                    if (max_v > 1.0e-30) {
                        const double ahi = max_v / 127.0;
                        tile.alpha_hi[static_cast<size_t>(rl)] = ahi;
                        alpha_hi_f_[static_cast<size_t>(r)]    = static_cast<float>(ahi);
                        inv_hi_f_[static_cast<size_t>(r)]      = static_cast<float>(1.0 / ahi);
                        if (need_lo) {
                            const double alo = ahi / 254.0;
                            tile.alpha_lo[static_cast<size_t>(rl)] = alo;
                            alpha_lo_f_[static_cast<size_t>(r)]    = static_cast<float>(alo);
                            inv_lo_f_[static_cast<size_t>(r)]      = static_cast<float>(1.0 / alo);
                        }
                    }
                }
            }
        }

        max_threads_ =
        #ifdef _OPENMP
            std::max(1, omp_get_max_threads());
        #else
            1;
        #endif
        k_pad64_    = (static_cast<size_t>(k) + 7) / 8 * 8;
        k_lo_pad64_ = (static_cast<size_t>(k_lo_) + 7) / 8 * 8;
        max_tile_rc_pad_ = (static_cast<size_t>(max_rc) + 15) / 16 * 16;
        thr_sum_hi_.assign(static_cast<size_t>(max_threads_) * k_pad64_, 0);
        if (k_lo_ > 0) {
            thr_sum_lo_.assign(static_cast<size_t>(max_threads_) * k_lo_pad64_, 0);
            thr_local_lo32_.resize(static_cast<size_t>(max_threads_) * max_tile_rc_pad_);
        }
        thr_local_hi32_.resize(static_cast<size_t>(max_threads_) * max_tile_rc_pad_);
        if (y_adj_ptr_ != nullptr) {
            thr_uy_.assign(static_cast<size_t>(max_threads_) * k_pad64_, 0.0);
        }
    }

    // Optional: compute exact 15-bit Vi_y[s] = (y_adj[s] - sum_r ck_uy[r]*U[r,s]) / sigma2_eff
    // directly from TUN3 slabs during process_sample_slab (used when prepass computed u_y).
    void set_exact_vi_y_target(const double* ck_uy_k,
                               const float* y_adj_n,
                               float sigma2_eff_f,
                               float* out_vi_y_n)
    {
        if (k_ <= 0 || n_ <= 0 || !ck_uy_k || !y_adj_n || !out_vi_y_n) return;
        pre_ck_uy_.assign(ck_uy_k, ck_uy_k + k_);
        vi_y_in_ptr_  = y_adj_n;
        vi_y_out_ptr_ = out_vi_y_n;
        vi_y_inv_s2e_ = 1.0 / static_cast<double>(sigma2_eff_f);
    }

    // Consume a contiguous sample slab [s_begin, s_end) where raw_u16 has
    // layout k x (s_end - s_begin) ColMajor (each column = 1 sample).
    // Loops tile-outer so thread-local INT32 accumulators stay L1D-resident with zero
    // per-slab heap allocations.
    void process_sample_slab(int s_begin, int s_end, const uint16_t* raw_u16, int num_threads_override = 0)
    {
        if (k_ <= 0 || s_end <= s_begin) return;
        const size_t num_tiles = out_.tiles.size();
        const float* const rs_base    = uk_row_scale_.data();
        const float* const sck_base   = sqrt_ck_.data();
        const float* const ahi_base   = alpha_hi_f_.data();
        const float* const ihi_base   = inv_hi_f_.data();
        const float* const alo_base   = alpha_lo_f_.data();
        const float* const ilo_base   = inv_lo_f_.data();
        const float* const runit_base = row_unit_f_.data();
        const bool fast_unit = unit_weights_ && !has_exact_max_override_;
        const int use_nt = (num_threads_override > 0) ? std::min(num_threads_override, max_threads_) : max_threads_;

        double slab_eps_sq = 0.0;
        double slab_eps_2s_sq = 0.0;

        #pragma omp parallel num_threads(use_nt) reduction(+:slab_eps_sq,slab_eps_2s_sq)
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
            const int slab_n = s_end - s_begin;
            const int s_lo = s_begin + static_cast<int>((static_cast<int64_t>(slab_n) * tid) / nt);
            const int s_hi = s_begin + static_cast<int>((static_cast<int64_t>(slab_n) * (tid + 1)) / nt);

            int32_t* const lhi = thr_local_hi32_.data() + static_cast<size_t>(tid) * max_tile_rc_pad_;
            int32_t* const llo = (k_lo_ > 0)
                ? (thr_local_lo32_.data() + static_cast<size_t>(tid) * max_tile_rc_pad_)
                : nullptr;
            int64_t* const my_sum_hi = thr_sum_hi_.data() + static_cast<size_t>(tid) * k_pad64_;
            int64_t* const my_sum_lo = (k_lo_ > 0)
                ? (thr_sum_lo_.data() + static_cast<size_t>(tid) * k_lo_pad64_)
                : nullptr;
            double* const my_uy = (y_adj_ptr_ != nullptr)
                ? (thr_uy_.data() + static_cast<size_t>(tid) * k_pad64_)
                : nullptr;

            if (my_uy != nullptr || vi_y_out_ptr_ != nullptr) {
                const double* const cku = pre_ck_uy_.data();
                for (int s = s_lo; s < s_hi; ++s) {
                    const uint16_t* const raw_col = raw_u16 +
                        static_cast<size_t>(s - s_begin) * static_cast<size_t>(k_);
                    if (my_uy != nullptr) {
                        const double ys = static_cast<double>(y_adj_ptr_[s]);
                        for (int r = 0; r < k_; ++r) {
                            const float sc = rs_base[r];
                            const float u = (sc > 0.0f)
                                ? (static_cast<float>(static_cast<int16_t>(raw_col[r])) * sc)
                                : ((sc < 0.0f) ? wb_fp16_bits_to_f32(raw_col[r]) : 0.0f);
                            my_uy[r] += static_cast<double>(u) * ys;
                        }
                    }
                    if (vi_y_out_ptr_ != nullptr) {
                        double proj = 0.0;
                        for (int r = 0; r < k_; ++r) {
                            const float sc = rs_base[r];
                            const float u = (sc > 0.0f)
                                ? (static_cast<float>(static_cast<int16_t>(raw_col[r])) * sc)
                                : ((sc < 0.0f) ? wb_fp16_bits_to_f32(raw_col[r]) : 0.0f);
                            proj += cku[r] * static_cast<double>(u);
                        }
                        vi_y_out_ptr_[s] = static_cast<float>(
                            (static_cast<double>(vi_y_in_ptr_[s]) - proj) * vi_y_inv_s2e_);
                    }
                }
            }

            constexpr int kSubSlab = 32768;
            for (size_t ti = 0; ti < num_tiles; ++ti) {
                const auto& tile = out_.tiles[ti];
                const int r0 = tile.r_begin;
                const int rc = tile.r_count;
                const int q_ld = tile.q_ld;
                const bool need_lo = tile.is_2slice || tile.has_lo_refine;
                int8_t* const      tile_q_base = tile.Q.get();
                int8_t* const      tile_qlo_base = tile.has_lo_refine ? tile.Q_lo_refine.get() : nullptr;
                const float* const rs_t  = rs_base + r0;
                const float* const sck   = sck_base + r0;
                const float* const ihi   = ihi_base + r0;
                const float* const ahi   = ahi_base + r0;
                const float* const ilo   = need_lo ? (ilo_base + r0) : nullptr;
                const float* const alo   = need_lo ? (alo_base + r0) : nullptr;
                const float* const runit = runit_base + r0;
                const bool use_fast_i16_tile = fast_unit &&
                    (tile_all_i16_[ti] || tile_fp16_rows_[ti].size() * 2 < static_cast<size_t>(rc));
                const bool all_i16 = tile_all_i16_[ti];

                for (int sb = s_lo; sb < s_hi; sb += kSubSlab) {
                    const int se = std::min(s_hi, sb + kSubSlab);
                    std::memset(lhi, 0, static_cast<size_t>(rc) * sizeof(int32_t));
                    if (need_lo) {
                        std::memset(llo, 0, static_cast<size_t>(rc) * sizeof(int32_t));
                    }

                    for (int s = sb; s < se; ++s) {
                        const uint16_t* const raw_t = raw_u16 +
                            static_cast<size_t>(s - s_begin) * static_cast<size_t>(k_) + r0;
                        const float ws_f = unit_weights_ ? 1.0f : w_ptr_[s];
                        int8_t* const q_hi = tile_q_base + static_cast<size_t>(s) * static_cast<size_t>(q_ld);
                        float tile_eps = 0.0f;
                        float tile_eps_2s = 0.0f;

                        if (tile.is_2slice) {
                            int8_t* const q_lo = q_hi + rc;
                            if (use_fast_i16_tile) {
                                #pragma omp simd
                                for (int rl = 0; rl < rc; ++rl) {
                                    const int32_t q = static_cast<int32_t>(static_cast<int16_t>(raw_t[rl]));
                                    const int32_t bias = (q >= 0) ? 127 : -127;
                                    int32_t qh = (q + bias) / 254;
                                    if (qh > 127) qh = 127;
                                    if (qh < -127) qh = -127;
                                    const int32_t ql = q - 254 * qh;
                                    q_hi[rl] = static_cast<int8_t>(qh);
                                    q_lo[rl] = static_cast<int8_t>(ql);
                                    lhi[rl] += qh;
                                    llo[rl] += ql;
                                }
                                if (!all_i16) {
                                    for (int rl : tile_fp16_rows_[ti]) {
                                        lhi[rl] -= static_cast<int32_t>(q_hi[rl]);
                                        llo[rl] -= static_cast<int32_t>(q_lo[rl]);
                                        const float u    = wb_fp16_bits_to_f32(raw_t[rl]);
                                        const float v    = u * sck[rl];
                                        const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                        const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                                        const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                                        const int32_t qh = static_cast<int32_t>(qh_f);
                                        const int32_t ql = static_cast<int32_t>(ql_f);
                                        q_hi[rl] = static_cast<int8_t>(qh);
                                        q_lo[rl] = static_cast<int8_t>(ql);
                                        lhi[rl] += qh;
                                        llo[rl] += ql;
                                        const float eps = std::fmaf(-alo[rl], ql_f, rem);
                                        const float eps2 = eps * eps;
                                        tile_eps += eps2;
                                        tile_eps_2s += eps2;
                                    }
                                }
                            } else {
                                for (int rl = 0; rl < rc; ++rl) {
                                    const float sc = rs_t[rl];
                                    const float u = (sc > 0.0f)
                                        ? (static_cast<float>(static_cast<int16_t>(raw_t[rl])) * sc)
                                        : ((sc < 0.0f) ? wb_fp16_bits_to_f32(raw_t[rl]) : 0.0f);
                                    const float v    = (u * ws_f) * sck[rl];
                                    const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                    const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                                    const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                                    const int32_t qh = static_cast<int32_t>(qh_f);
                                    const int32_t ql = static_cast<int32_t>(ql_f);
                                    q_hi[rl] = static_cast<int8_t>(qh);
                                    q_lo[rl] = static_cast<int8_t>(ql);
                                    lhi[rl] += qh;
                                    llo[rl] += ql;
                                    const float eps = std::fmaf(-alo[rl], ql_f, rem);
                                    const float eps2 = eps * eps;
                                    tile_eps += eps2;
                                    tile_eps_2s += eps2;
                                }
                            }
                        } else if (tile.has_lo_refine) {
                            int8_t* const q_lo = tile_qlo_base + static_cast<size_t>(s) * static_cast<size_t>(tile.q_lo_ld);
                            if (use_fast_i16_tile) {
                                #pragma omp simd reduction(+:tile_eps)
                                for (int rl = 0; rl < rc; ++rl) {
                                    const int32_t q = static_cast<int32_t>(static_cast<int16_t>(raw_t[rl]));
                                    const int32_t bias = (q >= 0) ? 127 : -127;
                                    int32_t qh = (q + bias) / 254;
                                    if (qh > 127) qh = 127;
                                    if (qh < -127) qh = -127;
                                    const int32_t ql = q - 254 * qh;
                                    q_hi[rl] = static_cast<int8_t>(qh);
                                    q_lo[rl] = static_cast<int8_t>(ql);
                                    lhi[rl] += qh;
                                    llo[rl] += ql;
                                    const float rem1 = static_cast<float>(ql) * runit[rl];
                                    tile_eps += rem1 * rem1;
                                }
                                if (!all_i16) {
                                    for (int rl : tile_fp16_rows_[ti]) {
                                        const int32_t prev_ql = static_cast<int32_t>(q_lo[rl]);
                                        lhi[rl] -= static_cast<int32_t>(q_hi[rl]);
                                        llo[rl] -= prev_ql;
                                        const float prev_rem1 = static_cast<float>(prev_ql) * runit[rl];
                                        tile_eps -= prev_rem1 * prev_rem1;
                                        const float u    = wb_fp16_bits_to_f32(raw_t[rl]);
                                        const float v    = u * sck[rl];
                                        const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                        const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                                        const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                                        const int32_t qh = static_cast<int32_t>(qh_f);
                                        const int32_t ql = static_cast<int32_t>(ql_f);
                                        q_hi[rl] = static_cast<int8_t>(qh);
                                        q_lo[rl] = static_cast<int8_t>(ql);
                                        lhi[rl] += qh;
                                        llo[rl] += ql;
                                        const float eps2 = std::fmaf(-alo[rl], ql_f, rem);
                                        tile_eps += rem * rem;
                                        tile_eps_2s += eps2 * eps2;
                                    }
                                }
                            } else {
                                for (int rl = 0; rl < rc; ++rl) {
                                    const float sc = rs_t[rl];
                                    const float u = (sc > 0.0f)
                                        ? (static_cast<float>(static_cast<int16_t>(raw_t[rl])) * sc)
                                        : ((sc < 0.0f) ? wb_fp16_bits_to_f32(raw_t[rl]) : 0.0f);
                                    const float v    = (u * ws_f) * sck[rl];
                                    const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                    const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                                    const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                                    const int32_t qh = static_cast<int32_t>(qh_f);
                                    const int32_t ql = static_cast<int32_t>(ql_f);
                                    q_hi[rl] = static_cast<int8_t>(qh);
                                    q_lo[rl] = static_cast<int8_t>(ql);
                                    lhi[rl] += qh;
                                    llo[rl] += ql;
                                    const float eps2 = std::fmaf(-alo[rl], ql_f, rem);
                                    tile_eps += rem * rem;
                                    tile_eps_2s += eps2 * eps2;
                                }
                            }
                        } else {
                            if (fast_unit && all_i16) {
                                constexpr float kInv255 = 1.0f / 255.0f;
                                #pragma omp simd reduction(+:tile_eps)
                                for (int rl = 0; rl < rc; ++rl) {
                                    const float qf   = static_cast<float>(static_cast<int16_t>(raw_t[rl]));
                                    const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(qf * kInv255)));
                                    const int32_t qh = static_cast<int32_t>(qh_f);
                                    q_hi[rl] = static_cast<int8_t>(qh);
                                    lhi[rl] += qh;
                                    const float eps = std::fmaf(-255.0f, qh_f, qf) * runit[rl];
                                    tile_eps += eps * eps;
                                }
                            } else {
                                for (int rl = 0; rl < rc; ++rl) {
                                    const float sc = rs_t[rl];
                                    const float u = (sc > 0.0f)
                                        ? (static_cast<float>(static_cast<int16_t>(raw_t[rl])) * sc)
                                        : ((sc < 0.0f) ? wb_fp16_bits_to_f32(raw_t[rl]) : 0.0f);
                                    const float v    = (u * ws_f) * sck[rl];
                                    const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                                    const int32_t qh = static_cast<int32_t>(qh_f);
                                    q_hi[rl] = static_cast<int8_t>(qh);
                                    lhi[rl] += qh;
                                    const float eps = std::fmaf(-ahi[rl], qh_f, v);
                                    tile_eps += eps * eps;
                                }
                            }
                        }
                        slab_eps_sq += static_cast<double>(tile_eps);
                        slab_eps_2s_sq += static_cast<double>(tile_eps_2s);
                    }

                    for (int rl = 0; rl < rc; ++rl) {
                        my_sum_hi[r0 + rl] += static_cast<int64_t>(lhi[rl]);
                    }
                    if (need_lo) {
                        for (int rl = 0; rl < rc; ++rl) {
                            my_sum_lo[r0 + rl] += static_cast<int64_t>(llo[rl]);
                        }
                    }
                }
            }
        }
        extra_eps_sq_ += slab_eps_sq;
        extra_eps_2s_sq_ += slab_eps_2s_sq;
    }

    // Reduce accumulated u_y = Uk * y_adj (valid only when y_adj_for_uy was provided).
    Eigen::VectorXd reduced_uy() const
    {
        Eigen::VectorXd uy = Eigen::VectorXd::Zero(k_);
        if (thr_uy_.empty()) return uy;
        #pragma omp parallel for schedule(static) if (k_ >= 1024)
        for (int r = 0; r < k_; ++r) {
            double acc = 0.0;
            for (int t = 0; t < max_threads_; ++t) {
                acc += thr_uy_[static_cast<size_t>(t) * k_pad64_ + static_cast<size_t>(r)];
            }
            uy[r] = acc;
        }
        return uy;
    }

    // Finalize u_sum_q and exact delta_quant using the TUN3 trailer row_eps_sq[k].
    QuantizedWoodburyUk finalize(const float* row_eps_sq_trailer)
    {
        const size_t num_tiles = out_.tiles.size();
        for (size_t ti = 0; ti < num_tiles; ++ti) {
            auto& tile = out_.tiles[ti];
            const int r0 = tile.r_begin;
            const int rc = tile.r_count;
            if (tile.is_2slice) {
                #pragma omp parallel for schedule(static) if (rc >= 512)
                for (int rl = 0; rl < rc; ++rl) {
                    int64_t sh = 0;
                    int64_t sl = 0;
                    const size_t r = static_cast<size_t>(r0 + rl);
                    for (int t = 0; t < max_threads_; ++t) {
                        sh += thr_sum_hi_[static_cast<size_t>(t) * k_pad64_ + r];
                        sl += thr_sum_lo_[static_cast<size_t>(t) * k_lo_pad64_ + r];
                    }
                    tile.u_sum_q[static_cast<size_t>(rl)] =
                        tile.alpha_hi[static_cast<size_t>(rl)] * static_cast<double>(sh)
                      + tile.alpha_lo[static_cast<size_t>(rl)] * static_cast<double>(sl);
                }
            } else if (tile.has_lo_refine) {
                #pragma omp parallel for schedule(static) if (rc >= 512)
                for (int rl = 0; rl < rc; ++rl) {
                    int64_t sh = 0;
                    int64_t sl = 0;
                    const size_t r = static_cast<size_t>(r0 + rl);
                    for (int t = 0; t < max_threads_; ++t) {
                        sh += thr_sum_hi_[static_cast<size_t>(t) * k_pad64_ + r];
                        sl += thr_sum_lo_[static_cast<size_t>(t) * k_lo_pad64_ + r];
                    }
                    const double ahi = tile.alpha_hi[static_cast<size_t>(rl)];
                    const double alo = tile.alpha_lo[static_cast<size_t>(rl)];
                    tile.u_sum_q[static_cast<size_t>(rl)]    = ahi * static_cast<double>(sh);
                    tile.u_sum_q_2s[static_cast<size_t>(rl)] = ahi * static_cast<double>(sh) + alo * static_cast<double>(sl);
                }
            } else {
                #pragma omp parallel for schedule(static) if (rc >= 512)
                for (int rl = 0; rl < rc; ++rl) {
                    int64_t sh = 0;
                    const size_t r = static_cast<size_t>(r0 + rl);
                    for (int t = 0; t < max_threads_; ++t) {
                        sh += thr_sum_hi_[static_cast<size_t>(t) * k_pad64_ + r];
                    }
                    tile.u_sum_q[static_cast<size_t>(rl)] =
                        tile.alpha_hi[static_cast<size_t>(rl)] * static_cast<double>(sh);
                }
            }
        }

        double disk_eps_sq = 0.0;
        if (row_eps_sq_trailer != nullptr && n_ > 0) {
            const double w_scale = unit_weights_ ? 1.0 : (S_w2_ / static_cast<double>(n_));
            for (int r = 0; r < k_; ++r) {
                const double sck = static_cast<double>(sqrt_ck_[static_cast<size_t>(r)]);
                disk_eps_sq += (sck * sck) * static_cast<double>(row_eps_sq_trailer[r]);
            }
            disk_eps_sq *= w_scale;
        }
        const double norm_denom = (n_ > 1)
            ? (S_w2_ * static_cast<double>(n_ - 1) / static_cast<double>(n_))
            : S_w2_;
        out_.delta_quant = (norm_denom > 0.0) ? ((disk_eps_sq + extra_eps_sq_) / norm_denom) : 0.0;
        out_.delta_quant_2s = (out_.mode == FastMlmaMode::Adaptive && norm_denom > 0.0)
            ? ((disk_eps_sq + extra_eps_2s_sq_) / norm_denom)
            : out_.delta_quant;
        return std::move(out_);
    }

private:
    int k_ = 0;
    int n_ = 0;
    int k_hi_ = 0;
    int k_lo_ = 0;
    int max_threads_ = 1;
    size_t k_pad64_ = 0;
    size_t k_lo_pad64_ = 0;
    size_t max_tile_rc_pad_ = 0;
    bool unit_weights_ = true;
    bool has_exact_max_override_ = false;
    double S_w2_ = 0.0;
    double extra_eps_sq_ = 0.0;
    double extra_eps_2s_sq_ = 0.0;

    std::vector<float> sqrt_ck_;
    std::vector<float> uk_row_scale_;
    const float* w_ptr_ = nullptr;
    const float* y_adj_ptr_ = nullptr;
    std::vector<double> pre_ck_uy_;
    const float* vi_y_in_ptr_ = nullptr;
    float* vi_y_out_ptr_ = nullptr;
    double vi_y_inv_s2e_ = 0.0;

    std::vector<float> alpha_hi_f_;
    std::vector<float> inv_hi_f_;
    std::vector<float> alpha_lo_f_;
    std::vector<float> inv_lo_f_;
    std::vector<float> row_unit_f_;
    std::vector<bool> tile_all_i16_;
    std::vector<std::vector<int>> tile_fp16_rows_;
    std::vector<int64_t> thr_sum_hi_;
    std::vector<int64_t> thr_sum_lo_;
    AlignedRawBuffer<int32_t, 64> thr_local_hi32_;
    AlignedRawBuffer<int32_t, 64> thr_local_lo32_;
    std::vector<double> thr_uy_;
    QuantizedWoodburyUk out_;
};

// Reconstruct V^{-1} y_adj directly from QuantizedWoodburyUk + exact u_y = Uk * y_adj
// (fallback used only when TUN3/TREF y_adj_hash does not match the current phenotype).
inline Eigen::VectorXf reconstruct_vi_y_from_q_uk(
    const QuantizedWoodburyUk& q_uk,
    const Eigen::VectorXf& sqrt_ck_f,
    float sigma2_eff_f,
    const Eigen::VectorXf& y_adj,
    const Eigen::VectorXf& w_sqrt,
    bool unit_weights,
    const Eigen::VectorXd& u_y)
{
    const int n = q_uk.n;
    const int k = q_uk.k;
    Eigen::VectorXf vi_y(n);
    if (n <= 0) return vi_y;
    const double inv_s2e = 1.0 / static_cast<double>(sigma2_eff_f);
    std::vector<double> coeff(static_cast<size_t>(k), 0.0);
    for (int r = 0; r < k; ++r) {
        coeff[static_cast<size_t>(r)] = static_cast<double>(sqrt_ck_f[r]) * u_y[r];
    }

    const size_t num_tiles = q_uk.tiles.size();
    const float* const y_ptr = y_adj.data();
    const float* const w_ptr = w_sqrt.data();
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < n; ++s) {
        double proj = 0.0;
        for (size_t ti = 0; ti < num_tiles; ++ti) {
            const auto& tile = q_uk.tiles[ti];
            const int r0 = tile.r_begin;
            const int rc = tile.r_count;
            const int q_ld = tile.q_ld;
            const int8_t* const q_hi = tile.Q.get() + static_cast<size_t>(s) * static_cast<size_t>(q_ld);
            const double* const ahi  = tile.alpha_hi.data();
            const double* const cf   = coeff.data() + r0;
            if (tile.is_2slice) {
                const int8_t* const q_lo = q_hi + rc;
                const double* const alo  = tile.alpha_lo.data();
                #pragma omp simd reduction(+:proj)
                for (int rl = 0; rl < rc; ++rl) {
                    const double qv = ahi[rl] * static_cast<double>(q_hi[rl])
                                    + alo[rl] * static_cast<double>(q_lo[rl]);
                    proj += cf[rl] * qv;
                }
            } else {
                #pragma omp simd reduction(+:proj)
                for (int rl = 0; rl < rc; ++rl) {
                    proj += cf[rl] * (ahi[rl] * static_cast<double>(q_hi[rl]));
                }
            }
        }
        if (!unit_weights) {
            const double ws = static_cast<double>(w_ptr[s]);
            proj = (std::fabs(ws) > 1.0e-12) ? (proj / ws) : 0.0;
        }
        vi_y[s] = static_cast<float>((static_cast<double>(y_ptr[s]) - proj) * inv_s2e);
    }
    return vi_y;
}

// Direct inline quantization from n x k ColMajor ctx.Uk (never allocates 4*k*n Uk_f).
// Loops tile-outer using a <= 64 MiB transpose scratch slab so source Uk pages can be
// progressively released via madvise as Q tiles are faulted in.
inline QuantizedWoodburyUk build_quantized_woodbury_uk_from_nxk(
    const Eigen::Ref<const Eigen::MatrixXf>& Uk_nxk,
    const Eigen::VectorXf& sqrt_ck_f,
    const Eigen::VectorXf& w_sqrt,
    bool unit_weights,
    double S_w2,
    FastMlmaMode mode,
    int max_q_rows = 8192,
    bool release_src_pages = false)
{
    QuantizedWoodburyUk out;
    const int n = static_cast<int>(Uk_nxk.rows());
    const int k = static_cast<int>(Uk_nxk.cols());
    out.k = k;
    out.n = n;
    out.mode = mode;
    if (k <= 0 || n <= 0 || mode == FastMlmaMode::Off) return out;

    const int k_hi = populate_quantized_woodbury_tiles(out, k, n, mode, max_q_rows);
    const int k_lo_cap = (mode == FastMlmaMode::Adaptive) ? k : k_hi;

    const float* const uk_src      = Uk_nxk.data();
    const size_t ld_uk             = static_cast<size_t>(Uk_nxk.outerStride());
    const float* const sqrt_ck_ptr = sqrt_ck_f.data();
    const float* const w_ptr       = w_sqrt.data();

    // Phase 1: unit-stride SIMD max_abs over each contiguous n-length column of Uk_nxk.
    std::vector<float> max_abs_raw(static_cast<size_t>(k), 0.0f);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < k; ++r) {
        const float* const col = uk_src + static_cast<size_t>(r) * ld_uk;
        float m = 0.0f;
        if (unit_weights) {
            #pragma omp simd reduction(max:m)
            for (int s = 0; s < n; ++s) {
                const float av = std::fabs(col[s]);
                if (av > m) m = av;
            }
        } else {
            #pragma omp simd reduction(max:m)
            for (int s = 0; s < n; ++s) {
                const float av = std::fabs(col[s] * w_ptr[s]);
                if (av > m) m = av;
            }
        }
        max_abs_raw[static_cast<size_t>(r)] = m;
    }

    std::vector<float> alpha_hi_f(static_cast<size_t>(k), 0.0f);
    std::vector<float> inv_hi_f(static_cast<size_t>(k), 0.0f);
    std::vector<float> alpha_lo_f(static_cast<size_t>(k_lo_cap), 0.0f);
    std::vector<float> inv_lo_f(static_cast<size_t>(k_lo_cap), 0.0f);

    int max_rc = 1;
    const size_t num_tiles = out.tiles.size();
    for (size_t ti = 0; ti < num_tiles; ++ti) {
        auto& tile = out.tiles[ti];
        const int r0 = tile.r_begin;
        const int rc = tile.r_count;
        const bool need_lo = tile.is_2slice || tile.has_lo_refine;
        max_rc = std::max(max_rc, rc);
        for (int rl = 0; rl < rc; ++rl) {
            const int r = r0 + rl;
            const double max_v = static_cast<double>(sqrt_ck_ptr[r])
                               * static_cast<double>(max_abs_raw[static_cast<size_t>(r)]);
            if (max_v > 1.0e-30) {
                const double ahi = max_v / 127.0;
                tile.alpha_hi[static_cast<size_t>(rl)] = ahi;
                alpha_hi_f[static_cast<size_t>(r)]     = static_cast<float>(ahi);
                inv_hi_f[static_cast<size_t>(r)]       = static_cast<float>(1.0 / ahi);
                if (need_lo) {
                    const double alo = ahi / 254.0;
                    tile.alpha_lo[static_cast<size_t>(rl)] = alo;
                    alpha_lo_f[static_cast<size_t>(r)]     = static_cast<float>(alo);
                    inv_lo_f[static_cast<size_t>(r)]       = static_cast<float>(1.0 / alo);
                }
            }
        }
    }

    const int max_threads =
    #ifdef _OPENMP
        std::max(1, omp_get_max_threads());
    #else
        1;
    #endif
    const size_t rc_pad64 = (static_cast<size_t>(max_rc) + 7) / 8 * 8;
    std::vector<int64_t> thr_sum_hi(static_cast<size_t>(max_threads) * rc_pad64, 0);
    std::vector<int64_t> thr_sum_lo(static_cast<size_t>(max_threads) * rc_pad64, 0);
    double total_eps_sq = 0.0;
    double total_eps_2s_sq = 0.0;

    // Phase 2: tile-outer quantization using a bounded <= 64 MiB rc x slen float32 slab.
    // Releasing source column-block pages per tile avoids coexisting n x k + Q RSS peaks.
    constexpr size_t kSlabBudgetBytes = 64ULL << 20;
    const size_t max_slab_elems = std::max<size_t>(
        static_cast<size_t>(max_rc) * 64ULL,
        std::min<size_t>(kSlabBudgetBytes / sizeof(float),
                         static_cast<size_t>(max_rc) * static_cast<size_t>(n)));
    AlignedRawBuffer<float, 64> slab_rc_s(max_slab_elems);

    #if defined(__linux__) || defined(__APPLE__)
    const long raw_page_sz = release_src_pages ? sysconf(_SC_PAGESIZE) : 0;
    const uintptr_t page_sz = (raw_page_sz > 0) ? static_cast<uintptr_t>(raw_page_sz) : 0;
    auto madvise_drop_range = [&](const float* ptr, size_t elems) {
        if (page_sz == 0 || elems == 0) return;
        const uintptr_t beg_addr = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t end_addr = reinterpret_cast<uintptr_t>(ptr + elems);
        const uintptr_t page_beg = (beg_addr + page_sz - 1) & ~(page_sz - 1);
        const uintptr_t page_end = end_addr & ~(page_sz - 1);
        if (page_end > page_beg) {
        #if defined(__linux__)
            ::madvise(reinterpret_cast<void*>(page_beg), page_end - page_beg, MADV_DONTNEED);
        #elif defined(__APPLE__)
            ::madvise(reinterpret_cast<void*>(page_beg), page_end - page_beg, MADV_FREE);
        #endif
        }
    };
    #endif

    const float* const ahi_base = alpha_hi_f.data();
    const float* const ihi_base = inv_hi_f.data();
    const float* const alo_base = alpha_lo_f.data();
    const float* const ilo_base = inv_lo_f.data();
    const size_t n_stride = ld_uk;

    for (size_t ti = 0; ti < num_tiles; ++ti) {
        auto& tile = out.tiles[ti];
        const int r0 = tile.r_begin;
        const int rc = tile.r_count;
        const int q_ld = tile.q_ld;
        const bool is_2s = tile.is_2slice;
        const bool has_lo = tile.has_lo_refine;
        const bool need_lo = is_2s || has_lo;
        const size_t rc_stride = static_cast<size_t>(rc);
        const int slab_max = std::min(
            n,
            std::clamp<int>(static_cast<int>(max_slab_elems / rc_stride), 64, 32768));

        std::fill(thr_sum_hi.begin(), thr_sum_hi.end(), 0);
        if (need_lo) std::fill(thr_sum_lo.begin(), thr_sum_lo.end(), 0);

        const float* const sck = sqrt_ck_ptr + r0;
        const float* const ihi = ihi_base + r0;
        const float* const ahi = ahi_base + r0;
        const float* const ilo = need_lo ? (ilo_base + r0) : nullptr;
        const float* const alo = need_lo ? (alo_base + r0) : nullptr;
        int8_t* const tile_q_base = tile.Q.get();
        int8_t* const tile_qlo_base = has_lo ? tile.Q_lo_refine.get() : nullptr;

        for (int s0 = 0; s0 < n; s0 += slab_max) {
            const int slen = std::min(slab_max, n - s0);
            constexpr int kBlk = 64;
            const int ni_tiles = (slen + kBlk - 1) / kBlk;
            const int nj_tiles = (rc + kBlk - 1) / kBlk;
            const int64_t total_tiles = static_cast<int64_t>(ni_tiles) * static_cast<int64_t>(nj_tiles);

            #pragma omp parallel for schedule(static)
            for (int64_t t = 0; t < total_tiles; ++t) {
                const int di0 = static_cast<int>(t / nj_tiles) * kBlk;
                const int j0  = static_cast<int>(t % nj_tiles) * kBlk;
                const int di_len = std::min(kBlk, slen - di0);
                const int j_len  = std::min(kBlk, rc - j0);
                alignas(64) float tbuf[kBlk][kBlk + 4];
                for (int dj = 0; dj < j_len; ++dj) {
                    const float* src_col = uk_src + static_cast<size_t>(r0 + j0 + dj) * n_stride + (s0 + di0);
                    for (int ddi = 0; ddi < di_len; ++ddi) {
                        tbuf[ddi][dj] = src_col[ddi];
                    }
                }
                for (int ddi = 0; ddi < di_len; ++ddi) {
                    float* dst_col = slab_rc_s.data() + static_cast<size_t>(di0 + ddi) * rc_stride + j0;
                    std::memcpy(dst_col, tbuf[ddi], static_cast<size_t>(j_len) * sizeof(float));
                }
            }

            #pragma omp parallel reduction(+:total_eps_sq,total_eps_2s_sq)
            {
                const int tid =
                #ifdef _OPENMP
                    omp_get_thread_num();
                #else
                    0;
                #endif
                std::vector<int32_t> local_hi32(rc_stride, 0);
                std::vector<int32_t> local_lo32(need_lo ? rc_stride : 0, 0);
                int64_t* const my_sum_hi = thr_sum_hi.data() + static_cast<size_t>(tid) * rc_pad64;
                int64_t* const my_sum_lo = need_lo ? (thr_sum_lo.data() + static_cast<size_t>(tid) * rc_pad64) : nullptr;

                #pragma omp for schedule(static) nowait
                for (int ds = 0; ds < slen; ++ds) {
                    const int s = s0 + ds;
                    const float* const uk_t = slab_rc_s.data() + static_cast<size_t>(ds) * rc_stride;
                    const float ws_f = unit_weights ? 1.0f : w_ptr[s];
                    float sample_eps = 0.0f;
                    float sample_eps_2s = 0.0f;
                    int8_t* const q_hi = tile_q_base + static_cast<size_t>(s) * static_cast<size_t>(q_ld);

                    if (is_2s) {
                        int8_t* const q_lo = q_hi + rc;
                        #pragma omp simd reduction(+:sample_eps,sample_eps_2s)
                        for (int rl = 0; rl < rc; ++rl) {
                            const float v    = (uk_t[rl] * ws_f) * sck[rl];
                            const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                            const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                            const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                            const int32_t qh = static_cast<int32_t>(qh_f);
                            const int32_t ql = static_cast<int32_t>(ql_f);
                            q_hi[rl] = static_cast<int8_t>(qh);
                            q_lo[rl] = static_cast<int8_t>(ql);
                            local_hi32[rl] += qh;
                            local_lo32[rl] += ql;
                            const float eps = std::fmaf(-alo[rl], ql_f, rem);
                            const float eps2 = eps * eps;
                            sample_eps += eps2;
                            sample_eps_2s += eps2;
                        }
                    } else if (has_lo) {
                        int8_t* const q_lo = tile_qlo_base + static_cast<size_t>(s) * static_cast<size_t>(tile.q_lo_ld);
                        #pragma omp simd reduction(+:sample_eps,sample_eps_2s)
                        for (int rl = 0; rl < rc; ++rl) {
                            const float v    = (uk_t[rl] * ws_f) * sck[rl];
                            const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                            const float rem  = std::fmaf(-ahi[rl], qh_f, v);
                            const float ql_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(rem * ilo[rl])));
                            const int32_t qh = static_cast<int32_t>(qh_f);
                            const int32_t ql = static_cast<int32_t>(ql_f);
                            q_hi[rl] = static_cast<int8_t>(qh);
                            q_lo[rl] = static_cast<int8_t>(ql);
                            local_hi32[rl] += qh;
                            local_lo32[rl] += ql;
                            const float eps2 = std::fmaf(-alo[rl], ql_f, rem);
                            sample_eps += rem * rem;
                            sample_eps_2s += eps2 * eps2;
                        }
                    } else {
                        #pragma omp simd reduction(+:sample_eps)
                        for (int rl = 0; rl < rc; ++rl) {
                            const float v    = (uk_t[rl] * ws_f) * sck[rl];
                            const float qh_f = std::min(127.0f, std::max(-127.0f, std::nearbyintf(v * ihi[rl])));
                            const int32_t qh = static_cast<int32_t>(qh_f);
                            q_hi[rl] = static_cast<int8_t>(qh);
                            local_hi32[rl] += qh;
                            const float eps = std::fmaf(-ahi[rl], qh_f, v);
                            sample_eps += eps * eps;
                        }
                    }
                    total_eps_sq += static_cast<double>(sample_eps);
                    total_eps_2s_sq += static_cast<double>(sample_eps_2s);
                }

                for (int rl = 0; rl < rc; ++rl) {
                    my_sum_hi[rl] += static_cast<int64_t>(local_hi32[static_cast<size_t>(rl)]);
                }
                if (need_lo) {
                    for (int rl = 0; rl < rc; ++rl) {
                        my_sum_lo[rl] += static_cast<int64_t>(local_lo32[static_cast<size_t>(rl)]);
                    }
                }
            }
        }

        if (is_2s) {
            #pragma omp parallel for schedule(static) if (rc >= 512)
            for (int rl = 0; rl < rc; ++rl) {
                int64_t sh = 0;
                int64_t sl = 0;
                for (int t = 0; t < max_threads; ++t) {
                    sh += thr_sum_hi[static_cast<size_t>(t) * rc_pad64 + static_cast<size_t>(rl)];
                    sl += thr_sum_lo[static_cast<size_t>(t) * rc_pad64 + static_cast<size_t>(rl)];
                }
                tile.u_sum_q[static_cast<size_t>(rl)] =
                    tile.alpha_hi[static_cast<size_t>(rl)] * static_cast<double>(sh)
                  + tile.alpha_lo[static_cast<size_t>(rl)] * static_cast<double>(sl);
            }
        } else if (has_lo) {
            #pragma omp parallel for schedule(static) if (rc >= 512)
            for (int rl = 0; rl < rc; ++rl) {
                int64_t sh = 0;
                int64_t sl = 0;
                for (int t = 0; t < max_threads; ++t) {
                    sh += thr_sum_hi[static_cast<size_t>(t) * rc_pad64 + static_cast<size_t>(rl)];
                    sl += thr_sum_lo[static_cast<size_t>(t) * rc_pad64 + static_cast<size_t>(rl)];
                }
                const double ahi = tile.alpha_hi[static_cast<size_t>(rl)];
                const double alo = tile.alpha_lo[static_cast<size_t>(rl)];
                tile.u_sum_q[static_cast<size_t>(rl)]    = ahi * static_cast<double>(sh);
                tile.u_sum_q_2s[static_cast<size_t>(rl)] = ahi * static_cast<double>(sh) + alo * static_cast<double>(sl);
            }
        } else {
            #pragma omp parallel for schedule(static) if (rc >= 512)
            for (int rl = 0; rl < rc; ++rl) {
                int64_t sh = 0;
                for (int t = 0; t < max_threads; ++t) {
                    sh += thr_sum_hi[static_cast<size_t>(t) * rc_pad64 + static_cast<size_t>(rl)];
                }
                tile.u_sum_q[static_cast<size_t>(rl)] =
                    tile.alpha_hi[static_cast<size_t>(rl)] * static_cast<double>(sh);
            }
        }

        #if defined(__linux__) || defined(__APPLE__)
        if (release_src_pages) {
            if (ld_uk == static_cast<size_t>(n)) {
                madvise_drop_range(uk_src + static_cast<size_t>(r0) * ld_uk,
                                   static_cast<size_t>(rc) * ld_uk);
            } else {
                for (int rl = 0; rl < rc; ++rl) {
                    madvise_drop_range(uk_src + static_cast<size_t>(r0 + rl) * ld_uk,
                                       static_cast<size_t>(n));
                }
            }
        }
        #endif
    }

    const double norm_denom = (n > 1) ? (S_w2 * static_cast<double>(n - 1) / static_cast<double>(n)) : S_w2;
    out.delta_quant = (norm_denom > 0.0) ? (total_eps_sq / norm_denom) : 0.0;
    out.delta_quant_2s = (mode == FastMlmaMode::Adaptive && norm_denom > 0.0)
        ? (total_eps_2s_sq / norm_denom)
        : out.delta_quant;
    return out;
}

// Partition `bs` into `ceil(bs / aligned_max)` balanced 64-column-aligned sub-tiles
// (guaranteed <= aligned_max when bs > aligned_max, e.g. 16384 -> 8192 + 8192 instead of 12224 + 4160).
[[nodiscard]] inline int balanced_snp_tile_cols(int bs, int max_cols) noexcept
{
    if (bs <= 0) return 0;
    const int aligned_max = std::max(64, (max_cols / 64) * 64);
    if (bs <= aligned_max) return bs;
    const int n_sub = (bs + aligned_max - 1) / aligned_max;
    const int total_u = (bs + 63) / 64;
    const int sub_u = (total_u + n_sub - 1) / n_sub;
    return std::min(bs, sub_u * 64);
}

inline void woodbury_xvx_diag_block_fast_i8(
    const WoodburyMLMACache& wb,
    const QuantizedWoodburyUk& q_uk,
    const int8_t* G_block_i8,
    int ldG,
    int bs,
    const std::vector<uint8_t>& valid_v,
    const std::vector<float>& mu_v,
    const std::vector<float>& x_norm_v,
    const std::vector<std::vector<int>>& snp_miss_samples,
    Eigen::VectorXf& xvx_diag,
    AlignedRawBuffer<int32_t, 64>& P_i32,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32,
    bool block_has_miss = true,
    std::vector<double>* ukx_norm_sq_scratch = nullptr,
    AlignedRawBuffer<int32_t, 64>* thr_miss_acc_buf = nullptr,
    int p_budget_mb = 0,
    int snp_tile_override = 0,
    MlmaProfileStats* prof = nullptr,
    std::vector<double>* ukx_norm_sq_hi_out = nullptr)
{
    using Clock = std::chrono::steady_clock;
    if (bs <= 0) return;
    const int n = q_uk.n;
    const double debias_scale = 1.0 + q_uk.delta_quant;
    const double sig2e = static_cast<double>(wb.sigma2_eff_f);

    if (ukx_norm_sq_hi_out != nullptr) {
        if (static_cast<int>(ukx_norm_sq_hi_out->size()) < bs) {
            ukx_norm_sq_hi_out->resize(static_cast<size_t>(bs));
        }
        std::fill_n(ukx_norm_sq_hi_out->data(), bs, 0.0);
    }

    const size_t num_tiles = q_uk.tiles.size();
    if (num_tiles == 0) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < bs; ++j) {
            if (!valid_v[j]) {
                xvx_diag[j] = 0.0f;
            } else {
                const double val = (debias_scale * static_cast<double>(x_norm_v[j])) / sig2e;
                xvx_diag[j] = std::max(0.0f, static_cast<float>(val));
            }
        }
        return;
    }

#if GCTA_HAS_AOCL_S8_GEMM
    const int kSampleTile = std::min(n, 8388608);
#else
    constexpr int kSampleTile = 8192;
#endif

    std::vector<double> local_norm_sq;
    std::vector<double>& ukx_norm_sq = ukx_norm_sq_scratch ? *ukx_norm_sq_scratch : local_norm_sq;
    if (static_cast<int>(ukx_norm_sq.size()) < bs) {
        ukx_norm_sq.resize(static_cast<size_t>(bs));
    }
    std::fill_n(ukx_norm_sq.data(), bs, 0.0);

    AlignedRawBuffer<int32_t, 64> local_miss_buf;
    AlignedRawBuffer<int32_t, 64>& miss_buf = thr_miss_acc_buf ? *thr_miss_acc_buf : local_miss_buf;
    const int max_threads =
    #ifdef _OPENMP
        std::max(1, omp_get_max_threads());
    #else
        1;
    #endif

    size_t max_ldP_block = 64;
    for (size_t ti = 0; ti < num_tiles; ++ti) {
        const int qr_t = q_uk.tiles[ti].q_rows;
        int ldp_t = ((qr_t + 15) / 16) * 16;
        if ((static_cast<size_t>(ldp_t) * sizeof(int32_t)) % 4096 == 0) ldp_t += 16;
        max_ldP_block = std::max(max_ldP_block, static_cast<size_t>(std::max(qr_t, ldp_t)));
    }
    size_t eff_p_budget_bytes = 576ULL << 20; // covers ldP=8208 (qr=8192 + 4K pad) x bs=16384 (513 MiB) in 1 pass
    if (p_budget_mb > 0) {
        eff_p_budget_bytes = static_cast<size_t>(std::max(16, p_budget_mb)) << 20;
    } else {
        const size_t one_pass_bytes = max_ldP_block * static_cast<size_t>(bs) * sizeof(int32_t);
        if (one_pass_bytes > eff_p_budget_bytes && one_pass_bytes <= (2048ULL << 20)) {
            const uint64_t avail = detect_available_memory_bytes();
            constexpr uint64_t kReserve = 4ULL << 30;
            if (avail > kReserve + 2ULL * static_cast<uint64_t>(one_pass_bytes)) {
                eff_p_budget_bytes = one_pass_bytes;
            }
        }
    }

    for (size_t ti = 0; ti < num_tiles; ++ti) {
        const auto& tile = q_uk.tiles[ti];
        const bool is_last_tile = (ti + 1 == num_tiles);
        const int qr = tile.q_rows;
        const int rc = tile.r_count;
        const int q_ld = (tile.q_ld > 0) ? tile.q_ld : qr;
        const int8_t* const q_base = tile.Q.get();

        // Pad ldP to multiple of 16 int32s (64B), skipping 4096-byte (1024-elem) aliasing strides.
        int ldP = ((qr + 15) / 16) * 16;
        if ((static_cast<size_t>(ldP) * sizeof(int32_t)) % 4096 == 0) ldP += 16;
        ldP = std::max(qr, ldP);

        int kSnpTile = 0;
        if (snp_tile_override > 0) {
            kSnpTile = std::min(bs, snp_tile_override);
        } else {
#if GCTA_HAS_AOCL_S8_GEMM
            const size_t budget_cols = eff_p_budget_bytes / (static_cast<size_t>(ldP) * sizeof(int32_t));
            const size_t aligned_cols = (budget_cols / 64ULL) * 64ULL;
            const int max_cols = std::clamp(
                static_cast<int>(std::min<size_t>(aligned_cols, 32768ULL)), 256, 32768);
            kSnpTile = balanced_snp_tile_cols(bs, max_cols);
#else
            kSnpTile = balanced_snp_tile_cols(bs, 1024);
#endif
        }
        const size_t need_p = static_cast<size_t>(ldP) * static_cast<size_t>(kSnpTile);
        if (P_i32.size() < need_p) P_i32.resize(need_p);

        const size_t miss_stride = ((static_cast<size_t>(qr) + 15) / 16) * 16;
        if (block_has_miss) {
            const size_t need_miss = static_cast<size_t>(max_threads) * miss_stride;
            if (miss_buf.size() < need_miss) {
                miss_buf.resize(need_miss);
            }
        }

        const double* const ahi = tile.alpha_hi.data();
        const double* const alo = tile.is_2slice ? tile.alpha_lo.data() : nullptr;
        const double* const usq = tile.u_sum_q.data();

        for (int j0 = 0; j0 < bs; j0 += kSnpTile) {
            const int cols = std::min(kSnpTile, bs - j0);
            const auto t_gemm0 = prof ? Clock::now() : Clock::time_point{};
            for (int s0 = 0; s0 < n; s0 += kSampleTile) {
                const int rows = std::min(kSampleTile, n - s0);
                const int32_t beta = (s0 == 0) ? 0 : 1;
                const int8_t* const q_reordered = (s0 == 0 && rows == n) ? tile.Q_reordered.get() : nullptr;
                gcta_gemm_u8s8s32_colmajor_nn_reordered(
                    static_cast<gcta_blas_int>(qr),
                    static_cast<gcta_blas_int>(cols),
                    static_cast<gcta_blas_int>(rows),
                    q_reordered,
                    q_base + static_cast<size_t>(s0) * static_cast<size_t>(q_ld),
                    static_cast<gcta_blas_int>(q_ld),
                    G_block_i8 + static_cast<size_t>(j0) * static_cast<size_t>(ldG) + s0,
                    static_cast<gcta_blas_int>(ldG),
                    P_i32.data(),
                    static_cast<gcta_blas_int>(ldP),
                    beta,
                    scratch_A_f32,
                    scratch_B_f32,
                    scratch_C_f32);
            }
            if (prof) {
                prof->xvx_gemm_s += std::chrono::duration<double>(Clock::now() - t_gemm0).count();
            }

            const auto t_red0 = prof ? Clock::now() : Clock::time_point{};
            if (!block_has_miss) {
                // Zero-missing fast path (post-Beagle): unswitched unit-stride SIMD reduction,
                // fusing final xvx_diag normalization on the last tile.
                if (tile.is_2slice) {
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        const int j = j0 + c;
                        if (!valid_v[j]) {
                            if (is_last_tile) xvx_diag[j] = 0.0f;
                            continue;
                        }
                        const int32_t* const p_hi = P_i32.data() + static_cast<size_t>(c) * static_cast<size_t>(ldP);
                        const int32_t* const p_lo = p_hi + rc;
                        const double mu_j = static_cast<double>(mu_v[j]);
                        double tile_norm_sq = 0.0;
                        #pragma omp simd reduction(+:tile_norm_sq)
                        for (int rl = 0; rl < rc; ++rl) {
                            const double val = ahi[rl] * static_cast<double>(p_hi[rl])
                                             + alo[rl] * static_cast<double>(p_lo[rl])
                                             - mu_j * usq[rl];
                            tile_norm_sq += val * val;
                        }
                        const double acc_sq = ukx_norm_sq[j] + tile_norm_sq;
                        ukx_norm_sq[j] = acc_sq;
                        if (tile.is_2slice && ukx_norm_sq_hi_out != nullptr) {
                            (*ukx_norm_sq_hi_out)[j] = acc_sq;
                        }
                        if (is_last_tile) {
                            const double val = (debias_scale * static_cast<double>(x_norm_v[j]) - acc_sq) / sig2e;
                            xvx_diag[j] = std::max(0.0f, static_cast<float>(val));
                        }
                    }
                } else {
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        const int j = j0 + c;
                        if (!valid_v[j]) {
                            if (is_last_tile) xvx_diag[j] = 0.0f;
                            continue;
                        }
                        const int32_t* const p_hi = P_i32.data() + static_cast<size_t>(c) * static_cast<size_t>(ldP);
                        const double mu_j = static_cast<double>(mu_v[j]);
                        double tile_norm_sq = 0.0;
                        #pragma omp simd reduction(+:tile_norm_sq)
                        for (int rl = 0; rl < rc; ++rl) {
                            const double val = ahi[rl] * static_cast<double>(p_hi[rl])
                                             - mu_j * usq[rl];
                            tile_norm_sq += val * val;
                        }
                        const double acc_sq = ukx_norm_sq[j] + tile_norm_sq;
                        ukx_norm_sq[j] = acc_sq;
                        if (is_last_tile) {
                            const double val = (debias_scale * static_cast<double>(x_norm_v[j]) - acc_sq) / sig2e;
                            xvx_diag[j] = std::max(0.0f, static_cast<float>(val));
                        }
                    }
                }
            } else {
                #pragma omp parallel
                {
                    const int tid =
                    #ifdef _OPENMP
                        omp_get_thread_num();
                    #else
                        0;
                    #endif
                    int32_t* const ma = miss_buf.data() + static_cast<size_t>(tid) * miss_stride;
                    #pragma omp for schedule(dynamic, 16)
                    for (int c = 0; c < cols; ++c) {
                        const int j = j0 + c;
                        if (!valid_v[j]) {
                            if (is_last_tile) xvx_diag[j] = 0.0f;
                            continue;
                        }

                        const int32_t* const p_hi = P_i32.data() + static_cast<size_t>(c) * static_cast<size_t>(ldP);
                        const double mu_j = static_cast<double>(mu_v[j]);
                        const auto& miss_list = snp_miss_samples[j];
                        const bool has_miss = !miss_list.empty();

                        double tile_norm_sq = 0.0;
                        if (!has_miss) {
                            if (tile.is_2slice) {
                                const int32_t* const p_lo = p_hi + rc;
                                #pragma omp simd reduction(+:tile_norm_sq)
                                for (int rl = 0; rl < rc; ++rl) {
                                    const double val = ahi[rl] * static_cast<double>(p_hi[rl])
                                                     + alo[rl] * static_cast<double>(p_lo[rl])
                                                     - mu_j * usq[rl];
                                    tile_norm_sq += val * val;
                                }
                            } else {
                                #pragma omp simd reduction(+:tile_norm_sq)
                                for (int rl = 0; rl < rc; ++rl) {
                                    const double val = ahi[rl] * static_cast<double>(p_hi[rl])
                                                     - mu_j * usq[rl];
                                    tile_norm_sq += val * val;
                                }
                            }
                        } else {
                            std::memset(ma, 0, static_cast<size_t>(qr) * sizeof(int32_t));
                            for (int s : miss_list) {
                                const int8_t* const q_col = q_base + static_cast<size_t>(s) * static_cast<size_t>(q_ld);
                                #pragma omp simd
                                for (int q = 0; q < qr; ++q) {
                                    ma[q] += static_cast<int32_t>(q_col[q]);
                                }
                            }
                            if (tile.is_2slice) {
                                const int32_t* const p_lo = p_hi + rc;
                                const int32_t* const m_lo = ma + rc;
                                #pragma omp simd reduction(+:tile_norm_sq)
                                for (int rl = 0; rl < rc; ++rl) {
                                    const double val = ahi[rl] * (static_cast<double>(p_hi[rl]) + mu_j * static_cast<double>(ma[rl]))
                                                     + alo[rl] * (static_cast<double>(p_lo[rl]) + mu_j * static_cast<double>(m_lo[rl]))
                                                     - mu_j * usq[rl];
                                    tile_norm_sq += val * val;
                                }
                            } else {
                                #pragma omp simd reduction(+:tile_norm_sq)
                                for (int rl = 0; rl < rc; ++rl) {
                                    const double val = ahi[rl] * (static_cast<double>(p_hi[rl]) + mu_j * static_cast<double>(ma[rl]))
                                                     - mu_j * usq[rl];
                                    tile_norm_sq += val * val;
                                }
                            }
                        }
                        const double acc_sq = ukx_norm_sq[j] + tile_norm_sq;
                        ukx_norm_sq[j] = acc_sq;
                        if (tile.is_2slice && ukx_norm_sq_hi_out != nullptr) {
                            (*ukx_norm_sq_hi_out)[j] = acc_sq;
                        }
                        if (is_last_tile) {
                            const double val = (debias_scale * static_cast<double>(x_norm_v[j]) - acc_sq) / sig2e;
                            xvx_diag[j] = std::max(0.0f, static_cast<float>(val));
                        }
                    }
                }
            }
            if (prof) {
                prof->xvx_reduce_s += std::chrono::duration<double>(Clock::now() - t_red0).count();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage-2 2-slice refinement for Adaptive fast-mlma candidates (chi2 >= 4.5,
// corresponding to p_tent < ~0.034). Recomputes tail tiles [k_hi, k) using
// both Q (hi) and Q_lo_refine (lo) on the small gathered M_cand columns.
// ---------------------------------------------------------------------------
inline void refine_adaptive_candidates_fast_i8(
    const WoodburyMLMACache& wb,
    const QuantizedWoodburyUk& q_uk,
    const int8_t* G_block_i8,
    int ldG,
    int bs,
    const std::vector<uint8_t>& valid_v,
    const std::vector<float>& mu_v,
    const std::vector<float>& x_norm_v,
    const Eigen::VectorXf& xt_vi_y,
    const std::vector<std::vector<int>>& snp_miss_samples,
    const std::vector<double>& ukx_norm_sq_hi,
    Eigen::VectorXf& xvx_diag,
    bool block_has_miss,
    AlignedRawBuffer<int8_t, 64>& cand_G_buf,
    AlignedRawBuffer<int32_t, 64>& P_hi_cand,
    AlignedRawBuffer<int32_t, 64>& P_lo_cand,
    AlignedRawBuffer<int32_t, 64>& miss_acc_buf,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32,
    MlmaProfileStats* prof = nullptr)
{
    using Clock = std::chrono::steady_clock;
    if (bs <= 0) return;

    bool has_refine_tile = false;
    for (const auto& t : q_uk.tiles) {
        if (t.has_lo_refine) {
            has_refine_tile = true;
            break;
        }
    }
    if (!has_refine_tile) return;

    std::vector<int> cand_idx;
    cand_idx.reserve(static_cast<size_t>(bs) / 8 + 16);
    for (int j = 0; j < bs; ++j) {
        if (!valid_v[j]) continue;
        const double num = static_cast<double>(xt_vi_y[j]);
        const double den = static_cast<double>(xvx_diag[j]);
        if ((den <= 0.0 && num != 0.0) || (den > 0.0 && (num * num / den) >= 4.5)) {
            cand_idx.push_back(j);
        }
    }
    if (cand_idx.empty()) return;

    const auto t_ref0 = prof ? Clock::now() : Clock::time_point{};
    const int m_cand = static_cast<int>(cand_idx.size());
    const int n = q_uk.n;
    const size_t need_g = static_cast<size_t>(ldG) * static_cast<size_t>(m_cand);
    if (cand_G_buf.size() < need_g) cand_G_buf.resize(need_g);

    #pragma omp parallel for schedule(static) if (m_cand >= 4)
    for (int c = 0; c < m_cand; ++c) {
        const int j = cand_idx[static_cast<size_t>(c)];
        std::memcpy(cand_G_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(ldG),
                    G_block_i8 + static_cast<size_t>(j) * static_cast<size_t>(ldG),
                    static_cast<size_t>(n));
    }

#if GCTA_HAS_AOCL_S8_GEMM
    const int kSampleTile = std::min(n, 8388608);
#else
    constexpr int kSampleTile = 8192;
#endif
    const int max_threads =
    #ifdef _OPENMP
        std::max(1, omp_get_max_threads());
    #else
        1;
    #endif

    std::vector<double> tail_norm_sq(static_cast<size_t>(m_cand), 0.0);

    for (size_t ti = 0; ti < q_uk.tiles.size(); ++ti) {
        const auto& tile = q_uk.tiles[ti];
        if (!tile.has_lo_refine) continue;
        const int rc = tile.r_count;
        int ldP = ((rc + 15) / 16) * 16;
        if ((static_cast<size_t>(ldP) * sizeof(int32_t)) % 4096 == 0) ldP += 16;
        ldP = std::max(rc, ldP);

        const size_t need_p = static_cast<size_t>(ldP) * static_cast<size_t>(m_cand);
        if (P_hi_cand.size() < need_p) P_hi_cand.resize(need_p);
        if (P_lo_cand.size() < need_p) P_lo_cand.resize(need_p);

        const size_t miss_stride = ((static_cast<size_t>(2 * rc) + 15) / 16) * 16;
        if (block_has_miss) {
            const size_t need_miss = static_cast<size_t>(max_threads) * miss_stride;
            if (miss_acc_buf.size() < need_miss) miss_acc_buf.resize(need_miss);
        }

        const auto t_g0 = prof ? Clock::now() : Clock::time_point{};
        for (int s0 = 0; s0 < n; s0 += kSampleTile) {
            const int rows = std::min(kSampleTile, n - s0);
            const int32_t beta = (s0 == 0) ? 0 : 1;
            const int8_t* const q_reord = (s0 == 0 && rows == n) ? tile.Q_reordered.get() : nullptr;
            gcta_gemm_u8s8s32_colmajor_nn_reordered(
                static_cast<gcta_blas_int>(rc),
                static_cast<gcta_blas_int>(m_cand),
                static_cast<gcta_blas_int>(rows),
                q_reord,
                tile.Q.get() + static_cast<size_t>(s0) * static_cast<size_t>(tile.q_ld),
                static_cast<gcta_blas_int>(tile.q_ld),
                cand_G_buf.data() + s0,
                static_cast<gcta_blas_int>(ldG),
                P_hi_cand.data(),
                static_cast<gcta_blas_int>(ldP),
                beta,
                scratch_A_f32, scratch_B_f32, scratch_C_f32);
            gcta_gemm_u8s8s32_colmajor_nn_reordered(
                static_cast<gcta_blas_int>(rc),
                static_cast<gcta_blas_int>(m_cand),
                static_cast<gcta_blas_int>(rows),
                nullptr,
                tile.Q_lo_refine.get() + static_cast<size_t>(s0) * static_cast<size_t>(tile.q_lo_ld),
                static_cast<gcta_blas_int>(tile.q_lo_ld),
                cand_G_buf.data() + s0,
                static_cast<gcta_blas_int>(ldG),
                P_lo_cand.data(),
                static_cast<gcta_blas_int>(ldP),
                beta,
                scratch_A_f32, scratch_B_f32, scratch_C_f32);
        }
        if (prof) prof->xvx_gemm_s += std::chrono::duration<double>(Clock::now() - t_g0).count();

        const double* const ahi   = tile.alpha_hi.data();
        const double* const alo   = tile.alpha_lo.data();
        const double* const usq2s = tile.u_sum_q_2s.data();
        const int8_t* const q_hi_base = tile.Q.get();
        const int8_t* const q_lo_base = tile.Q_lo_refine.get();
        const int q_hi_ld = tile.q_ld;
        const int q_lo_ld = tile.q_lo_ld;

        #pragma omp parallel
        {
            const int tid =
            #ifdef _OPENMP
                omp_get_thread_num();
            #else
                0;
            #endif
            int32_t* const ma_hi = miss_acc_buf.data() + static_cast<size_t>(tid) * miss_stride;
            int32_t* const ma_lo = ma_hi + rc;

            #pragma omp for schedule(static)
            for (int c = 0; c < m_cand; ++c) {
                const int j = cand_idx[static_cast<size_t>(c)];
                const int32_t* const p_hi = P_hi_cand.data() + static_cast<size_t>(c) * static_cast<size_t>(ldP);
                const int32_t* const p_lo = P_lo_cand.data() + static_cast<size_t>(c) * static_cast<size_t>(ldP);
                const double mu_j = static_cast<double>(mu_v[j]);
                double tile_sq = 0.0;

                if (!block_has_miss || snp_miss_samples[j].empty()) {
                    #pragma omp simd reduction(+:tile_sq)
                    for (int rl = 0; rl < rc; ++rl) {
                        const double val = ahi[rl] * static_cast<double>(p_hi[rl])
                                         + alo[rl] * static_cast<double>(p_lo[rl])
                                         - mu_j * usq2s[rl];
                        tile_sq += val * val;
                    }
                } else {
                    std::memset(ma_hi, 0, static_cast<size_t>(2 * rc) * sizeof(int32_t));
                    for (int s : snp_miss_samples[j]) {
                        const int8_t* const qh_s = q_hi_base + static_cast<size_t>(s) * static_cast<size_t>(q_hi_ld);
                        const int8_t* const ql_s = q_lo_base + static_cast<size_t>(s) * static_cast<size_t>(q_lo_ld);
                        #pragma omp simd
                        for (int rl = 0; rl < rc; ++rl) {
                            ma_hi[rl] += static_cast<int32_t>(qh_s[rl]);
                            ma_lo[rl] += static_cast<int32_t>(ql_s[rl]);
                        }
                    }
                    #pragma omp simd reduction(+:tile_sq)
                    for (int rl = 0; rl < rc; ++rl) {
                        const double val = ahi[rl] * (static_cast<double>(p_hi[rl]) + mu_j * static_cast<double>(ma_hi[rl]))
                                         + alo[rl] * (static_cast<double>(p_lo[rl]) + mu_j * static_cast<double>(ma_lo[rl]))
                                         - mu_j * usq2s[rl];
                        tile_sq += val * val;
                    }
                }
                tail_norm_sq[static_cast<size_t>(c)] += tile_sq;
            }
        }
    }

    const double debias_2s = 1.0 + q_uk.delta_quant_2s;
    const double sig2e     = static_cast<double>(wb.sigma2_eff_f);
    #pragma omp parallel for schedule(static) if (m_cand >= 64)
    for (int c = 0; c < m_cand; ++c) {
        const int j = cand_idx[static_cast<size_t>(c)];
        const double tot_sq = ukx_norm_sq_hi[j] + tail_norm_sq[static_cast<size_t>(c)];
        const double val = (debias_2s * static_cast<double>(x_norm_v[j]) - tot_sq) / sig2e;
        xvx_diag[j] = std::max(0.0f, static_cast<float>(val));
    }

    if (prof) {
        prof->refined_snps += static_cast<uint64_t>(m_cand);
        prof->xvx_reduce_s += std::chrono::duration<double>(Clock::now() - t_ref0).count();
    }
}

// ---------------------------------------------------------------------------
// mlma_pchisq_df1_from_abs_z
// Exact upper-tail df=1 chi-squared probability from |z|:
//   P(chi2_1 >= z^2) = erfc(|z| / sqrt(2))
// With log_pval=true, switches to an 8-term log-space asymptotic tail for
// |z| > 26 (where u = 1/z^2 < 1/676 gives < 1 ULP error in double without
// underflow or Boost gamma_q overhead).
// ---------------------------------------------------------------------------
[[nodiscard]] inline double mlma_pchisq_df1_from_abs_z(double abs_z, bool log_pval) noexcept
{
    constexpr double kInvSqrt2      = 0.707106781186547524400844362104849039;
    constexpr double kHalfLogHalfPi = 0.225791352644727432363097614947441008; // 0.5 * ln(pi / 2)
    if (!std::isfinite(abs_z)) {
        if (std::isnan(abs_z)) return std::numeric_limits<double>::quiet_NaN();
        return log_pval ? -std::numeric_limits<double>::infinity() : 0.0;
    }
    if (!log_pval) {
        return std::erfc(abs_z * kInvSqrt2);
    }
    if (abs_z <= 26.0) {
        return std::log(std::erfc(abs_z * kInvSqrt2));
    }
    const double z2 = abs_z * abs_z;
    const double u  = 1.0 / z2;
    const double s_m1 = u * (-1.0 + u * (3.0 + u * (-15.0 + u * (105.0
                        + u * (-945.0 + u * (10395.0 + u * (-135135.0 + u * 2027025.0)))))));
    return -0.5 * z2 - std::log(abs_z) - kHalfLogHalfPi + std::log1p(s_m1);
}

// ---------------------------------------------------------------------------
// mlma_snp_stat
// Compute beta, se, and p-value for a single SNP.
//
//   numerator   = x^T Vi y  (or x^T Vi y − covariate correction in the
//                             covariate-adjusted path)
//   denominator = x^T Vi x  (or its Schur complement in the covariate path)
//
// Returns false if denominator <= 1e-30 (degenerate/monomorphic SNP);
// beta, se, and pval are unchanged in that case.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool mlma_snp_stat(float numerator, float denominator,
                                         bool log_pval,
                                         float& beta, float& se, double& pval)
{
    if (!(denominator > 1.0e-30f)) return false;
    const float inv_d    = 1.0f / denominator;
    const float sqrt_inv = std::sqrt(inv_d);
    beta = numerator * inv_d;
    se   = sqrt_inv;
    // |z| = |beta / se| = |numerator / sqrt(denom)|
    const double abs_z = std::fabs(static_cast<double>(numerator) * static_cast<double>(sqrt_inv));
    pval = mlma_pchisq_df1_from_abs_z(abs_z, log_pval);
    return true;
}
