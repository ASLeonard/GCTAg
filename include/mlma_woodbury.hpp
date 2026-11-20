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
#include <limits>
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
// Cache-blocked as kTile(2048) outer -> kSampleTile(8192) middle ->
// kSnpTile(2048) inner so the Uk_f slab (64 MiB), X_panel_f32 (64 MiB), and
// UkX_tile (<= 80 MiB) remain L3-resident and Uk_f is streamed once per block.
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
    std::vector<float>& X_panel_f32,
    bool block_has_miss = true,
    std::vector<double>* ukx_norm_sq_scratch = nullptr)
{
    const int k = static_cast<int>(wb.Uk_f.rows());
    const int n = static_cast<int>(wb.Uk_f.cols());
    if (bs <= 0) return;

    constexpr int kTile       = 2048;
    constexpr int kSampleTile = 8192;
    constexpr int kSnpTile    = 2048;

    const int max_kb = std::min(k, kTile);
    if (UkX_scratch.rows() < max_kb || UkX_scratch.cols() < bs) {
        UkX_scratch.resize(max_kb, bs);
    }

    const size_t panel_need =
        static_cast<size_t>(std::min(n, kSampleTile)) *
        static_cast<size_t>(std::min(bs, kSnpTile));
    if (X_panel_f32.size() < panel_need) {
        X_panel_f32.resize(panel_need);
    }

    std::vector<double> local_norm_sq;
    std::vector<double>& ukx_norm_sq = ukx_norm_sq_scratch ? *ukx_norm_sq_scratch : local_norm_sq;
    if (static_cast<int>(ukx_norm_sq.size()) < bs) {
        ukx_norm_sq.resize(static_cast<size_t>(bs));
    }
    std::fill_n(ukx_norm_sq.data(), bs, 0.0);

    const float* const w_ptr       = w_sqrt.data();
    const float* const u_sum_ptr   = u_sum.data();
    const float* const sqrt_ck_ptr = wb.sqrt_ck_f.data();
    const float* const uk_base     = wb.Uk_f.data();
    float* const       ukx_buf     = UkX_scratch.data();

    for (int k0 = 0; k0 < k; k0 += kTile) {
        const int kb = std::min(kTile, k - k0);

        for (int s0 = 0; s0 < n; s0 += kSampleTile) {
            const int rows = std::min(kSampleTile, n - s0);
            const float beta = (s0 == 0) ? 0.0f : 1.0f;
            const float* const uk_slab = uk_base + k0 + static_cast<size_t>(s0) * static_cast<size_t>(k);

            for (int j0 = 0; j0 < bs; j0 += kSnpTile) {
                const int cols = std::min(kSnpTile, bs - j0);
                if (unit_weights) {
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        const int8_t* src = G_block_i8 +
                            static_cast<size_t>(j0 + c) * static_cast<size_t>(ldG) + s0;
                        float* dst = X_panel_f32.data() +
                            static_cast<size_t>(c) * static_cast<size_t>(rows);
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
                            static_cast<size_t>(c) * static_cast<size_t>(rows);
                        #pragma omp simd
                        for (int r = 0; r < rows; ++r) {
                            dst[r] = static_cast<float>(src[r]) * w_slice[r];
                        }
                    }
                }

                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            kb, cols, rows,
                            1.0f,
                            uk_slab, k,
                            X_panel_f32.data(), rows,
                            beta,
                            ukx_buf + static_cast<size_t>(j0) * static_cast<size_t>(kb), kb);
            }
        }

        const float* const u_sum_tile   = u_sum_ptr + k0;
        const float* const sqrt_ck_tile = sqrt_ck_ptr + k0;

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < bs; ++j) {
            if (!valid_v[j]) continue;

            float* const ukx_col = ukx_buf + static_cast<size_t>(j) * static_cast<size_t>(kb);
            const float mu_j = mu_v[j];

            if (block_has_miss) {
                const auto& miss_list = snp_miss_samples[j];
                if (!miss_list.empty()) {
                    if (unit_weights) {
                        for (int s : miss_list) {
                            const float* uk_s = uk_base + k0 + static_cast<size_t>(s) * static_cast<size_t>(k);
                            #pragma omp simd
                            for (int dr = 0; dr < kb; ++dr) {
                                ukx_col[dr] += mu_j * uk_s[dr];
                            }
                        }
                    } else {
                        for (int s : miss_list) {
                            const float scale = mu_j * w_ptr[s];
                            const float* uk_s = uk_base + k0 + static_cast<size_t>(s) * static_cast<size_t>(k);
                            #pragma omp simd
                            for (int dr = 0; dr < kb; ++dr) {
                                ukx_col[dr] += scale * uk_s[dr];
                            }
                        }
                    }
                }
            }

            double tile_norm_d = 0.0;
            #pragma omp simd reduction(+:tile_norm_d)
            for (int dr = 0; dr < kb; ++dr) {
                const float centered = ukx_col[dr] - mu_j * u_sum_tile[dr];
                const float scaled   = centered * sqrt_ck_tile[dr];
                tile_norm_d += static_cast<double>(scaled) * static_cast<double>(scaled);
            }
            ukx_norm_sq[j] += tile_norm_d;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < bs; ++j) {
        if (!valid_v[j]) {
            xvx_diag[j] = 0.0f;
            continue;
        }
        const float ukx_norm = static_cast<float>(ukx_norm_sq[j]);
        const float val      = (x_norm_v[j] - ukx_norm) / wb.sigma2_eff_f;
        xvx_diag[j] = std::max(0.0f, val);
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
    int q_rows = 0;       // is_2slice ? 2 * r_count : r_count (always <= 4096)
    bool is_2slice = true;
    AlignedUniquePtr<int8_t, 64> Q;                       // q_rows x n, col-major (NUMA-interleaved)
    size_t q_bytes = 0;
    AlignedUniquePtr<int8_t, 64> Q_reordered;             // AOCL pre-reordered B buffer (null if unused)
    size_t q_reordered_bytes = 0;
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
        for (const auto& t : tiles) total += t.q_bytes;
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
        tile.Q_reordered = make_numa_interleaved_uninitialized<int8_t, 64>(sz);
        if (!tile.Q_reordered) {
            LOGGER.w(0, "Failed to allocate pre-reordered Q tile; falling back to unpacked Q.");
            continue;
        }
        tile.q_reordered_bytes = sz;
        gcta_u8s8s32_reorder_b(
            static_cast<gcta_blas_int>(tile.q_rows),
            static_cast<gcta_blas_int>(q_uk.n),
            tile.Q.get(),
            static_cast<gcta_blas_int>(tile.q_rows),
            tile.Q_reordered.get());
    }
}

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

    // Cap every tile at q_rows <= 4096 (r_count <= 2048 for 2-slice, <= 4096 for 1-slice).
    constexpr int kMaxQRows = 4096;
    auto push_tiles = [&](int beg, int end, bool is_2slice) {
        const int rc_max = is_2slice ? (kMaxQRows / 2) : kMaxQRows;
        for (int r0 = beg; r0 < end; r0 += rc_max) {
            QuantizedWoodburyTile t;
            t.r_begin = r0;
            t.r_count = std::min(rc_max, end - r0);
            t.is_2slice = is_2slice;
            t.q_rows = is_2slice ? 2 * t.r_count : t.r_count;
            t.q_bytes = static_cast<size_t>(t.q_rows) * static_cast<size_t>(n);
            t.Q = make_numa_interleaved_uninitialized<int8_t, 64>(t.q_bytes);
            if (t.q_bytes > 0 && !t.Q) {
                LOGGER.e(0, "Failed to allocate quantized Woodbury Q tile.");
            }
            t.alpha_hi.assign(static_cast<size_t>(t.r_count), 0.0);
            if (is_2slice) t.alpha_lo.assign(static_cast<size_t>(t.r_count), 0.0);
            t.u_sum_q.assign(static_cast<size_t>(t.r_count), 0.0);
            out.tiles.push_back(std::move(t));
        }
    };
    if (k_hi > 0) push_tiles(0, k_hi, true);
    if (k_hi < k) push_tiles(k_hi, k, false);

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
    // For 2-slice rows, |rem| <= 0.5 * alpha_hi analytically, so
    // alpha_lo = alpha_hi / 254.0 maps rem / alpha_lo strictly into [-127, 127]
    // with zero clipping and eliminates an entire second scan over Uk_f.
    std::vector<float> alpha_hi_f(static_cast<size_t>(k), 0.0f);
    std::vector<float> inv_hi_f(static_cast<size_t>(k), 0.0f);
    std::vector<float> alpha_lo_f(static_cast<size_t>(k_hi), 0.0f);
    std::vector<float> inv_lo_f(static_cast<size_t>(k_hi), 0.0f);

    const size_t num_tiles = out.tiles.size();
    for (size_t ti = 0; ti < num_tiles; ++ti) {
        auto& tile = out.tiles[ti];
        const int r0 = tile.r_begin;
        const int rc = tile.r_count;
        for (int rl = 0; rl < rc; ++rl) {
            const int r = r0 + rl;
            const double max_v = static_cast<double>(sqrt_ck_ptr[r])
                               * static_cast<double>(max_abs_raw[static_cast<size_t>(r)]);
            if (max_v > 1.0e-30) {
                const double ahi = max_v / 127.0;
                tile.alpha_hi[static_cast<size_t>(rl)] = ahi;
                alpha_hi_f[static_cast<size_t>(r)]     = static_cast<float>(ahi);
                inv_hi_f[static_cast<size_t>(r)]       = static_cast<float>(1.0 / ahi);
                if (tile.is_2slice) {
                    const double alo = ahi / 254.0;
                    tile.alpha_lo[static_cast<size_t>(rl)] = alo;
                    alpha_lo_f[static_cast<size_t>(r)]     = static_cast<float>(alo);
                    inv_lo_f[static_cast<size_t>(r)]       = static_cast<float>(1.0 / alo);
                }
            }
        }
    }

    // Phase 2: single OpenMP parallel region across all 32768-sample slabs.
    // Each slab reads each Uk_f column once across all tiles, stores split
    // [qh | ql] INT8 slices with vectorized nearbyintf/fmaf, accumulates into
    // flat per-thread sum arrays (no critical sections, no per-slab mallocs),
    // and progressively returns completed Uk_f OS pages after a barrier.
    constexpr int kReleaseSlab = 32768;
    const size_t k_pad64    = (static_cast<size_t>(k) + 7) / 8 * 8;
    const size_t k_hi_pad64 = (static_cast<size_t>(k_hi) + 7) / 8 * 8;
    std::vector<int64_t> thr_sum_hi(static_cast<size_t>(max_threads) * k_pad64, 0);
    std::vector<int64_t> thr_sum_lo(static_cast<size_t>(max_threads) * k_hi_pad64, 0);
    double total_eps_sq = 0.0;
    int p2_threads = 1;

    #if defined(__linux__) || defined(__APPLE__)
    const long raw_page_sz = sysconf(_SC_PAGESIZE);
    const uintptr_t page_sz = (raw_page_sz > 0) ? static_cast<uintptr_t>(raw_page_sz) : 0;
    #endif

    const float* const ahi_base = alpha_hi_f.data();
    const float* const ihi_base = inv_hi_f.data();
    const float* const alo_base = alpha_lo_f.data();
    const float* const ilo_base = inv_lo_f.data();

    #pragma omp parallel reduction(+:total_eps_sq)
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

        // Per-slab int32 accumulators stay strictly within [-4.16e6, +4.16e6]
        // (kReleaseSlab * 127 << INT32_MAX), allowing 32-bit SIMD in the hot loop.
        std::vector<int32_t> local_hi32(static_cast<size_t>(k), 0);
        std::vector<int32_t> local_lo32(static_cast<size_t>(k_hi), 0);
        int64_t* const my_sum_hi = thr_sum_hi.data() + static_cast<size_t>(tid) * k_pad64;
        int64_t* const my_sum_lo = (k_hi > 0)
            ? (thr_sum_lo.data() + static_cast<size_t>(tid) * k_hi_pad64)
            : nullptr;

        for (int s0 = 0; s0 < n; s0 += kReleaseSlab) {
            const int s_end = std::min(n, s0 + kReleaseSlab);

            #pragma omp for schedule(static) nowait
            for (int s = s0; s < s_end; ++s) {
                const float* const uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k);
                const float ws_f = unit_weights ? 1.0f : w_ptr[s];
                float sample_eps = 0.0f;

                for (size_t ti = 0; ti < num_tiles; ++ti) {
                    const auto& tile = out.tiles[ti];
                    const int r0 = tile.r_begin;
                    const int rc = tile.r_count;
                    const int qr = tile.q_rows;
                    const float* const uk_t = uk_col + r0;
                    const float* const sck  = sqrt_ck_ptr + r0;
                    const float* const ihi  = ihi_base + r0;
                    const float* const ahi  = ahi_base + r0;
                    int32_t* const     lhi  = local_hi32.data() + r0;
                    int8_t* const      q_hi = tile.Q.get() + static_cast<size_t>(s) * static_cast<size_t>(qr);

                    if (!tile.is_2slice) {
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
                    } else {
                        int8_t* const      q_lo = q_hi + rc;
                        int32_t* const     llo  = local_lo32.data() + r0;
                        const float* const ilo  = ilo_base + r0;
                        const float* const alo  = alo_base + r0;
                        if (unit_weights) {
                            #pragma omp simd reduction(+:sample_eps)
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
                                sample_eps += eps * eps;
                            }
                        } else {
                            #pragma omp simd reduction(+:sample_eps)
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
                                sample_eps += eps * eps;
                            }
                        }
                    }
                }
                total_eps_sq += static_cast<double>(sample_eps);
            }

            // Fold slab int32 sums into thread-owned int64 accumulators before syncing.
            for (int r = 0; r < k; ++r) {
                my_sum_hi[r] += static_cast<int64_t>(local_hi32[static_cast<size_t>(r)]);
                local_hi32[static_cast<size_t>(r)] = 0;
            }
            for (int r = 0; r < k_hi; ++r) {
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
        if (!tile.is_2slice) {
            for (int rl = 0; rl < rc; ++rl) {
                int64_t sh = 0;
                const size_t r = static_cast<size_t>(r0 + rl);
                for (int t = 0; t < p2_threads; ++t) {
                    sh += thr_sum_hi[static_cast<size_t>(t) * k_pad64 + r];
                }
                tile.u_sum_q[static_cast<size_t>(rl)] =
                    tile.alpha_hi[static_cast<size_t>(rl)] * static_cast<double>(sh);
            }
        } else {
            for (int rl = 0; rl < rc; ++rl) {
                int64_t sh = 0;
                int64_t sl = 0;
                const size_t r = static_cast<size_t>(r0 + rl);
                for (int t = 0; t < p2_threads; ++t) {
                    sh += thr_sum_hi[static_cast<size_t>(t) * k_pad64 + r];
                    sl += thr_sum_lo[static_cast<size_t>(t) * k_hi_pad64 + r];
                }
                tile.u_sum_q[static_cast<size_t>(rl)] =
                    tile.alpha_hi[static_cast<size_t>(rl)] * static_cast<double>(sh)
                  + tile.alpha_lo[static_cast<size_t>(rl)] * static_cast<double>(sl);
            }
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
    std::vector<float>& scratch_C_f32,
    bool block_has_miss = true,
    std::vector<double>* ukx_norm_sq_scratch = nullptr)
{
    if (bs <= 0) return;
    const int n = q_uk.n;
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

    for (const auto& tile : q_uk.tiles) {
        const int qr = tile.q_rows;
        const int rc = tile.r_count;
        const int8_t* const q_base = tile.Q.get();
#if GCTA_HAS_AOCL_S8_GEMM
        const size_t budget_cols = (256ULL << 20) / (static_cast<size_t>(qr) * sizeof(int32_t));
        const int kSnpTile = std::min(
            bs,
            std::max(2048, static_cast<int>(std::min<size_t>(static_cast<size_t>(bs), budget_cols))));
#else
        const int kSnpTile = std::min(bs, 1024);
#endif
        const size_t need_p = static_cast<size_t>(qr) * static_cast<size_t>(kSnpTile);
        if (P_i32.size() < need_p) P_i32.resize(need_p);

        const double* const ahi = tile.alpha_hi.data();
        const double* const alo = tile.is_2slice ? tile.alpha_lo.data() : nullptr;
        const double* const usq = tile.u_sum_q.data();

        for (int j0 = 0; j0 < bs; j0 += kSnpTile) {
            const int cols = std::min(kSnpTile, bs - j0);
            for (int s0 = 0; s0 < n; s0 += kSampleTile) {
                const int rows = std::min(kSampleTile, n - s0);
                const int32_t beta = (s0 == 0) ? 0 : 1;
                const int8_t* const q_reordered = (s0 == 0 && rows == n) ? tile.Q_reordered.get() : nullptr;
                gcta_gemm_u8s8s32_colmajor_nn_reordered(
                    static_cast<gcta_blas_int>(qr),
                    static_cast<gcta_blas_int>(cols),
                    static_cast<gcta_blas_int>(rows),
                    q_reordered,
                    q_base + static_cast<size_t>(s0) * static_cast<size_t>(qr),
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

            if (!block_has_miss) {
                // Zero-missing fast path (post-Beagle): no miss_acc alloc/zero,
                // unswitched unit-stride SIMD reduction over split [qh | ql].
                if (tile.is_2slice) {
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        const int j = j0 + c;
                        if (!valid_v[j]) continue;
                        const int32_t* const p_hi = P_i32.data() + static_cast<size_t>(c) * static_cast<size_t>(qr);
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
                        ukx_norm_sq[j] += tile_norm_sq;
                    }
                } else {
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        const int j = j0 + c;
                        if (!valid_v[j]) continue;
                        const int32_t* const p_hi = P_i32.data() + static_cast<size_t>(c) * static_cast<size_t>(qr);
                        const double mu_j = static_cast<double>(mu_v[j]);
                        double tile_norm_sq = 0.0;
                        #pragma omp simd reduction(+:tile_norm_sq)
                        for (int rl = 0; rl < rc; ++rl) {
                            const double val = ahi[rl] * static_cast<double>(p_hi[rl])
                                             - mu_j * usq[rl];
                            tile_norm_sq += val * val;
                        }
                        ukx_norm_sq[j] += tile_norm_sq;
                    }
                }
            } else {
                #pragma omp parallel
                {
                    std::vector<int64_t> miss_acc;
                    #pragma omp for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        const int j = j0 + c;
                        if (!valid_v[j]) continue;

                        const int32_t* const p_hi = P_i32.data() + static_cast<size_t>(c) * static_cast<size_t>(qr);
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
                            if (static_cast<int>(miss_acc.size()) < qr) {
                                miss_acc.resize(static_cast<size_t>(qr));
                            }
                            std::fill_n(miss_acc.data(), qr, int64_t{0});
                            int64_t* const ma = miss_acc.data();
                            for (int s : miss_list) {
                                const int8_t* const q_col = q_base + static_cast<size_t>(s) * static_cast<size_t>(qr);
                                #pragma omp simd
                                for (int q = 0; q < qr; ++q) {
                                    ma[q] += static_cast<int64_t>(q_col[q]);
                                }
                            }
                            if (tile.is_2slice) {
                                const int32_t* const p_lo = p_hi + rc;
                                const int64_t* const m_lo = ma + rc;
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
                        ukx_norm_sq[j] += tile_norm_sq;
                    }
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
