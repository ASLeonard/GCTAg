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
#include <cmath>
#include <cstdint>
#include <vector>
#include "main/StatFunc.h"
#include "cpu.h"
#include "mem.hpp"

enum class FastMlmaMode : int {
    Off = 0,
    Slice1 = 1,
    Slice2 = 2,
    Hybrid = 3
};

inline const char* fast_mlma_mode_name(FastMlmaMode mode) {
    switch (mode) {
        case FastMlmaMode::Slice1: return "1slice";
        case FastMlmaMode::Slice2: return "2slice";
        case FastMlmaMode::Hybrid: return "hybrid";
        case FastMlmaMode::Off:
        default:                   return "off";
    }
}

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
                                     Eigen::MatrixXf& UkX_scratch)
{
    const int k = static_cast<int>(wb.Uk_f.rows());
    const int n = static_cast<int>(wb.Uk_f.cols());
    const int bs_int = static_cast<int>(bs);

    // cblas_sgemm: Uk_f (k×n) * X_block (n×bs) -> UkX_scratch (k×bs)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                k, bs_int, n,
                1.0f, wb.Uk_f.data(), k,
                X_block.data(), n,
                0.0f, UkX_scratch.data(), k);

    auto UkX = UkX_scratch.leftCols(bs);
    UkX.array().colwise() *= wb.sqrt_ck_f.array();

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < bs_int; ++j) {
        const float x_norm   = X_block.col(j).squaredNorm();
        const float ukx_norm = UkX.col(j).squaredNorm();
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
// Uses a small L3-resident float panel X_panel_f32 (at most 8192 x 1024 = 32 MB)
// to stream int8 -> float conversion into multi-threaded cblas_sgemm.
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
    std::vector<float>& X_panel_f32)
{
    const int k = static_cast<int>(wb.Uk_f.rows());
    const int n = static_cast<int>(wb.Uk_f.cols());
    if (bs <= 0) return;

    constexpr int kSampleTile = 8192;
    constexpr int kSnpTile    = 1024;
    const size_t panel_need =
        static_cast<size_t>(std::min(n, kSampleTile)) *
        static_cast<size_t>(std::min(bs, kSnpTile));
    if (X_panel_f32.size() < panel_need) {
        X_panel_f32.resize(panel_need);
    }

    const float* w_ptr = w_sqrt.data();
    for (int j0 = 0; j0 < bs; j0 += kSnpTile) {
        const int cols = std::min(kSnpTile, bs - j0);
        for (int s0 = 0; s0 < n; s0 += kSampleTile) {
            const int rows = std::min(kSampleTile, n - s0);
            if (unit_weights) {
                #pragma omp parallel for schedule(static)
                for (int c = 0; c < cols; ++c) {
                    const int8_t* src = G_block_i8 +
                        static_cast<size_t>(j0 + c) * static_cast<size_t>(ldG) + s0;
                    float* dst = X_panel_f32.data() +
                        static_cast<size_t>(c) * static_cast<size_t>(rows);
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
                        static_cast<size_t>(c) * static_cast<size_t>(rows);
                    for (int r = 0; r < rows; ++r) {
                        dst[r] = static_cast<float>(src[r]) * w_slice[r];
                    }
                }
            }

            const float beta = (s0 == 0) ? 0.0f : 1.0f;
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        k, cols, rows,
                        1.0f,
                        wb.Uk_f.data() + static_cast<size_t>(s0) * static_cast<size_t>(k), k,
                        X_panel_f32.data(), rows,
                        beta,
                        UkX_scratch.data() + static_cast<size_t>(j0) * static_cast<size_t>(k), k);
        }
    }

    const float* u_sum_ptr   = u_sum.data();
    const float* sqrt_ck_ptr = wb.sqrt_ck_f.data();
    const float* uk_base     = wb.Uk_f.data();

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < bs; ++j) {
        if (!valid_v[j]) {
            xvx_diag[j] = 0.0f;
            continue;
        }
        float* ukx_col = UkX_scratch.data() + static_cast<size_t>(j) * static_cast<size_t>(k);
        const float mu_j = mu_v[j];

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

        double ukx_norm_d = 0.0;
        #pragma omp simd reduction(+:ukx_norm_d)
        for (int r = 0; r < k; ++r) {
            const float centered = ukx_col[r] - mu_j * u_sum_ptr[r];
            const float scaled   = centered * sqrt_ck_ptr[r];
            ukx_col[r] = scaled;
            ukx_norm_d += static_cast<double>(scaled) * static_cast<double>(scaled);
        }
        const float ukx_norm = static_cast<float>(ukx_norm_d);
        const float val      = (x_norm_v[j] - ukx_norm) / wb.sigma2_eff_f;
        xvx_diag[j] = std::max(0.0f, val);
    }
}

// ---------------------------------------------------------------------------
// QuantizedWoodburyUk & woodbury_xvx_diag_block_fast_i8 (--fast-mlma)
// Pre-quantizes w_sqrt ⊙ sqrt(c_k) ⊙ Uk into K-tiled INT8 panels (q_rows x n,
// with q_rows <= 4096) so the N x BLOCK projection U_k^T G_block_i8 runs as
// an s8s8s32 GEMM (aocl_gemm_s8s8s32os32 on Zen 4 VNNI) with no Uk_f DRAM
// thrashing, no K x BLOCK UkX_scratch, matched quantized centering u_sum_q,
// and analytical quadratic quantization de-biasing delta_quant.
// ---------------------------------------------------------------------------
struct QuantizedWoodburyTile {
    int r_begin = 0;
    int r_count = 0;
    int q_rows = 0;       // is_2slice ? 2 * r_count : r_count
    bool is_2slice = true;
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> Q;  // q_rows x n, col-major
    std::vector<double> alpha_hi;                         // r_count
    std::vector<double> alpha_lo;                         // is_2slice ? r_count : 0
    std::vector<double> u_sum_q;                          // r_count
};

struct QuantizedWoodburyUk {
    int k = 0;
    int n = 0;
    FastMlmaMode mode = FastMlmaMode::Off;
    double delta_quant = 0.0;
    std::vector<QuantizedWoodburyTile> tiles;

    size_t total_q_bytes() const {
        size_t total = 0;
        for (const auto& t : tiles) total += t.Q.size();
        return total;
    }
};

inline QuantizedWoodburyUk build_quantized_woodbury_uk(
    const WoodburyMLMACache& wb,
    const Eigen::VectorXf& w_sqrt,
    bool unit_weights,
    double S_w2,
    FastMlmaMode mode)
{
    QuantizedWoodburyUk out;
    const int k = static_cast<int>(wb.Uk_f.rows());
    const int n = static_cast<int>(wb.Uk_f.cols());
    out.k = k;
    out.n = n;
    out.mode = mode;
    if (k <= 0 || n <= 0 || mode == FastMlmaMode::Off) return out;

    int k_hi = k;
    if (mode == FastMlmaMode::Slice1) {
        k_hi = 0;
    } else if (mode == FastMlmaMode::Hybrid) {
        k_hi = std::min(k, std::max(std::min(k, 100), k / 10));
    }

    constexpr int kTileMax = 2048;
    auto push_tiles = [&](int beg, int end, bool is_2slice) {
        for (int r0 = beg; r0 < end; r0 += kTileMax) {
            QuantizedWoodburyTile t;
            t.r_begin = r0;
            t.r_count = std::min(kTileMax, end - r0);
            t.is_2slice = is_2slice;
            t.q_rows = is_2slice ? 2 * t.r_count : t.r_count;
            t.Q.resize(static_cast<size_t>(t.q_rows) * static_cast<size_t>(n), 0);
            t.alpha_hi.assign(t.r_count, 0.0);
            if (is_2slice) t.alpha_lo.assign(t.r_count, 0.0);
            t.u_sum_q.assign(t.r_count, 0.0);
            out.tiles.push_back(std::move(t));
        }
    };
    if (k_hi > 0) push_tiles(0, k_hi, true);
    if (k_hi < k) push_tiles(k_hi, k, false);

    const float* uk_base = wb.Uk_f.data();
    const float* sqrt_ck_ptr = wb.sqrt_ck_f.data();
    const float* w_ptr = w_sqrt.data();
    double total_eps_sq = 0.0;

    auto clamp_i8 = [](double x) -> int8_t {
        const long long v = std::llround(x);
        return static_cast<int8_t>(std::clamp<long long>(v, -127, 127));
    };

    for (auto& tile : out.tiles) {
        const int r0 = tile.r_begin;
        const int rc = tile.r_count;
        const int qr = tile.q_rows;

        std::vector<double> max_abs_hi(rc, 0.0);
        #pragma omp parallel
        {
            std::vector<double> local_max(rc, 0.0);
            #pragma omp for schedule(static)
            for (int s = 0; s < n; ++s) {
                const float* uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k) + r0;
                const double ws = unit_weights ? 1.0 : static_cast<double>(w_ptr[s]);
                for (int rl = 0; rl < rc; ++rl) {
                    const double v = static_cast<double>(sqrt_ck_ptr[r0 + rl])
                                   * static_cast<double>(uk_col[rl]) * ws;
                    const double av = std::fabs(v);
                    if (av > local_max[rl]) local_max[rl] = av;
                }
            }
            #pragma omp critical
            {
                for (int rl = 0; rl < rc; ++rl) {
                    if (local_max[rl] > max_abs_hi[rl]) max_abs_hi[rl] = local_max[rl];
                }
            }
        }

        std::vector<double> inv_hi(rc, 0.0);
        for (int rl = 0; rl < rc; ++rl) {
            if (max_abs_hi[rl] > 0.0) {
                tile.alpha_hi[rl] = max_abs_hi[rl] / 127.0;
                inv_hi[rl] = 1.0 / tile.alpha_hi[rl];
            }
        }

        if (!tile.is_2slice) {
            std::vector<int64_t> sum_q_hi(rc, 0);
            double tile_eps_sq = 0.0;
            #pragma omp parallel
            {
                std::vector<int64_t> local_sum(rc, 0);
                double local_eps = 0.0;
                #pragma omp for schedule(static)
                for (int s = 0; s < n; ++s) {
                    const float* uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k) + r0;
                    int8_t* q_col = tile.Q.data() + static_cast<size_t>(s) * static_cast<size_t>(qr);
                    const double ws = unit_weights ? 1.0 : static_cast<double>(w_ptr[s]);
                    for (int rl = 0; rl < rc; ++rl) {
                        const double v = static_cast<double>(sqrt_ck_ptr[r0 + rl])
                                       * static_cast<double>(uk_col[rl]) * ws;
                        const int8_t qh = (inv_hi[rl] > 0.0) ? clamp_i8(v * inv_hi[rl]) : 0;
                        q_col[rl] = qh;
                        local_sum[rl] += static_cast<int64_t>(qh);
                        const double eps = v - tile.alpha_hi[rl] * static_cast<double>(qh);
                        local_eps += eps * eps;
                    }
                }
                #pragma omp critical
                {
                    for (int rl = 0; rl < rc; ++rl) sum_q_hi[rl] += local_sum[rl];
                    tile_eps_sq += local_eps;
                }
            }
            for (int rl = 0; rl < rc; ++rl) {
                tile.u_sum_q[rl] = tile.alpha_hi[rl] * static_cast<double>(sum_q_hi[rl]);
            }
            total_eps_sq += tile_eps_sq;
        } else {
            std::vector<double> max_abs_lo(rc, 0.0);
            #pragma omp parallel
            {
                std::vector<double> local_max(rc, 0.0);
                #pragma omp for schedule(static)
                for (int s = 0; s < n; ++s) {
                    const float* uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k) + r0;
                    int8_t* q_col = tile.Q.data() + static_cast<size_t>(s) * static_cast<size_t>(qr);
                    const double ws = unit_weights ? 1.0 : static_cast<double>(w_ptr[s]);
                    for (int rl = 0; rl < rc; ++rl) {
                        const double v = static_cast<double>(sqrt_ck_ptr[r0 + rl])
                                       * static_cast<double>(uk_col[rl]) * ws;
                        const int8_t qh = (inv_hi[rl] > 0.0) ? clamp_i8(v * inv_hi[rl]) : 0;
                        q_col[2 * rl] = qh;
                        const double rem = v - tile.alpha_hi[rl] * static_cast<double>(qh);
                        const double ar = std::fabs(rem);
                        if (ar > local_max[rl]) local_max[rl] = ar;
                    }
                }
                #pragma omp critical
                {
                    for (int rl = 0; rl < rc; ++rl) {
                        if (local_max[rl] > max_abs_lo[rl]) max_abs_lo[rl] = local_max[rl];
                    }
                }
            }

            std::vector<double> inv_lo(rc, 0.0);
            for (int rl = 0; rl < rc; ++rl) {
                if (max_abs_lo[rl] > 0.0) {
                    tile.alpha_lo[rl] = max_abs_lo[rl] / 127.0;
                    inv_lo[rl] = 1.0 / tile.alpha_lo[rl];
                }
            }

            std::vector<int64_t> sum_q_hi(rc, 0), sum_q_lo(rc, 0);
            double tile_eps_sq = 0.0;
            #pragma omp parallel
            {
                std::vector<int64_t> local_hi(rc, 0), local_lo(rc, 0);
                double local_eps = 0.0;
                #pragma omp for schedule(static)
                for (int s = 0; s < n; ++s) {
                    const float* uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k) + r0;
                    int8_t* q_col = tile.Q.data() + static_cast<size_t>(s) * static_cast<size_t>(qr);
                    const double ws = unit_weights ? 1.0 : static_cast<double>(w_ptr[s]);
                    for (int rl = 0; rl < rc; ++rl) {
                        const double v = static_cast<double>(sqrt_ck_ptr[r0 + rl])
                                       * static_cast<double>(uk_col[rl]) * ws;
                        const int8_t qh = q_col[2 * rl];
                        const double rem = v - tile.alpha_hi[rl] * static_cast<double>(qh);
                        const int8_t ql = (inv_lo[rl] > 0.0) ? clamp_i8(rem * inv_lo[rl]) : 0;
                        q_col[2 * rl + 1] = ql;
                        local_hi[rl] += static_cast<int64_t>(qh);
                        local_lo[rl] += static_cast<int64_t>(ql);
                        const double eps = rem - tile.alpha_lo[rl] * static_cast<double>(ql);
                        local_eps += eps * eps;
                    }
                }
                #pragma omp critical
                {
                    for (int rl = 0; rl < rc; ++rl) {
                        sum_q_hi[rl] += local_hi[rl];
                        sum_q_lo[rl] += local_lo[rl];
                    }
                    tile_eps_sq += local_eps;
                }
            }
            for (int rl = 0; rl < rc; ++rl) {
                tile.u_sum_q[rl] = tile.alpha_hi[rl] * static_cast<double>(sum_q_hi[rl])
                                 + tile.alpha_lo[rl] * static_cast<double>(sum_q_lo[rl]);
            }
            total_eps_sq += tile_eps_sq;
        }
    }

    const double norm_denom = (n > 1) ? (S_w2 * static_cast<double>(n - 1) / static_cast<double>(n)) : S_w2;
    out.delta_quant = (norm_denom > 0.0) ? (total_eps_sq / norm_denom) : 0.0;
    return out;
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
    std::vector<int32_t>& P_i32,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32)
{
    if (bs <= 0) return;
    const int n = q_uk.n;
    constexpr int kSampleTile = 8192;
    constexpr int kSnpTile = 1024;

    std::vector<double> ukx_norm_sq(bs, 0.0);

    for (const auto& tile : q_uk.tiles) {
        const int qr = tile.q_rows;
        const int rc = tile.r_count;
        const int max_cols = std::min(bs, kSnpTile);
        const size_t need_p = static_cast<size_t>(qr) * static_cast<size_t>(max_cols);
        if (P_i32.size() < need_p) P_i32.resize(need_p);

        for (int j0 = 0; j0 < bs; j0 += kSnpTile) {
            const int cols = std::min(kSnpTile, bs - j0);
            for (int s0 = 0; s0 < n; s0 += kSampleTile) {
                const int rows = std::min(kSampleTile, n - s0);
                const int32_t beta = (s0 == 0) ? 0 : 1;
                gcta_gemm_s8s8s32_colmajor_nn(
                    static_cast<gcta_blas_int>(qr),
                    static_cast<gcta_blas_int>(cols),
                    static_cast<gcta_blas_int>(rows),
                    tile.Q.data() + static_cast<size_t>(s0) * static_cast<size_t>(qr),
                    static_cast<gcta_blas_int>(qr),
                    G_block_i8 + static_cast<size_t>(j0) * static_cast<size_t>(ldG) + s0,
                    static_cast<gcta_blas_int>(ldG),
                    P_i32.data(),
                    static_cast<gcta_blas_int>(qr),
                    beta,
                    scratch_A_f32,
                    scratch_B_f32,
                    scratch_C_f32);
            }

            #pragma omp parallel
            {
                std::vector<int64_t> miss_acc(qr, 0);
                #pragma omp for schedule(static)
                for (int c = 0; c < cols; ++c) {
                    const int j = j0 + c;
                    if (!valid_v[j]) continue;

                    const int32_t* p_col = P_i32.data() + static_cast<size_t>(c) * static_cast<size_t>(qr);
                    const double mu_j = static_cast<double>(mu_v[j]);
                    const auto& miss_list = snp_miss_samples[j];
                    const bool has_miss = !miss_list.empty();
                    if (has_miss) {
                        std::fill(miss_acc.begin(), miss_acc.end(), 0);
                        for (int s : miss_list) {
                            const int8_t* q_col = tile.Q.data() + static_cast<size_t>(s) * static_cast<size_t>(qr);
                            for (int q = 0; q < qr; ++q) {
                                miss_acc[q] += static_cast<int64_t>(q_col[q]);
                            }
                        }
                    }

                    double tile_norm_sq = 0.0;
                    if (tile.is_2slice) {
                        for (int rl = 0; rl < rc; ++rl) {
                            double val = tile.alpha_hi[rl] * static_cast<double>(p_col[2 * rl])
                                       + tile.alpha_lo[rl] * static_cast<double>(p_col[2 * rl + 1]);
                            if (has_miss) {
                                val += mu_j * (tile.alpha_hi[rl] * static_cast<double>(miss_acc[2 * rl])
                                             + tile.alpha_lo[rl] * static_cast<double>(miss_acc[2 * rl + 1]));
                            }
                            val -= mu_j * tile.u_sum_q[rl];
                            tile_norm_sq += val * val;
                        }
                    } else {
                        for (int rl = 0; rl < rc; ++rl) {
                            double val = tile.alpha_hi[rl] * static_cast<double>(p_col[rl]);
                            if (has_miss) {
                                val += mu_j * (tile.alpha_hi[rl] * static_cast<double>(miss_acc[rl]));
                            }
                            val -= mu_j * tile.u_sum_q[rl];
                            tile_norm_sq += val * val;
                        }
                    }
                    ukx_norm_sq[j] += tile_norm_sq;
                }
            }
        }
    }

    const double debias_scale = 1.0 + q_uk.delta_quant;
    const double sig2e = static_cast<double>(wb.sigma2_eff_f);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < bs; ++j) {
        if (!valid_v[j]) {
            xvx_diag[j] = 0.0f;
            continue;
        }
        const double val = (debias_scale * static_cast<double>(x_norm_v[j]) - ukx_norm_sq[j]) / sig2e;
        xvx_diag[j] = std::max(0.0f, static_cast<float>(val));
    }
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
    if (denominator <= 1.0e-30f) return false;
    const float inv_d    = 1.0f / denominator;
    const float sqrt_inv = std::sqrt(inv_d);
    beta = numerator * inv_d;
    se   = sqrt_inv;
    // chisq = beta/se = (numerator/denom) / (1/sqrt(denom)) = numerator/sqrt(denom)
    const float chisq = numerator * sqrt_inv;
    pval = StatFunc::pchisq(static_cast<double>(chisq * chisq), 1, log_pval);
    return true;
}
