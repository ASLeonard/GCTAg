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
