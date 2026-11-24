/*
 * chunked_grm_matvec.hpp
 *
 * K @ X for a symmetric n x n matrix K, computed from lower-triangular tiles
 * read on demand, without ever materializing a dense n x n K in memory.
 * Companion to symmetric_eigendecomp.hpp: this produces the same `apply`
 * signature (X -> K@X) that build_randomized_sketch / power_iterate_and_project
 * / SpectraMatVecOp already consume.
 *
 * Also provides upper-triangular tiled dense symmetric matvec (`dense_upper_symm_f32`)
 * with optional AOCL-DLP BF16 GEMM acceleration for rSVD sketch/power passes while
 * keeping Rayleigh-Ritz passes in exact FP32.
 *
 * Layout & Tiling:
 *   - Row bands [rs, re) are decoupled from column tiles [cs, ce) (width <= col_tile)
 *     so disk-backed ChunkedGrmReader bounds its row-band cache strictly within its
 *     byte budget even at the bottom of an N = 500k-1M GRM.
 *   - Preferred fast-path tile layout is transposed column-major `tile_T` of shape
 *     (ce-cs) x (re-rs), so each GRM row is unpacked contiguously (eliminating
 *     16 KiB power-of-2 L1D/STLB stride scatter). On diagonal tiles (rs == cs),
 *     `tile_T` has its UPPER triangle populated.
 *   - For each row band [rs, re):
 *       * Off-diagonal column tiles cs = 0, col_tile, ... with ce = min(cs + col_tile, rs)
 *         lie strictly below the diagonal; one tile updates both Y[rs:re] and Y[cs:ce].
 *       * The square diagonal tile [rs, re) x [rs, re) covers all (r, c) with rs <= c <= r < re.
 */
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <omp.h>

#include "cpu.h"
#include "mem.hpp"

namespace gcta_chunked {

inline constexpr int kDefaultColTile = 4096;

// Whether rSVD sketch/power passes should use BF16 GEMM by default.
// Enabled automatically when AOCL-DLP / BLIS native BF16 GEMM is available (Zen 4).
inline constexpr bool default_bf16_sketch() noexcept {
#if GCTA_HAS_AOCL_BF16_GEMM
    return true;
#else
    return false;
#endif
}

// Optional RAII band-prefetch handle for double-buffered streaming sweeps.
class TileSweep {
public:
    virtual ~TileSweep() = default;
    virtual void acquire(int band_idx) = 0;
    virtual void release(int band_idx) = 0;
};

// Backward-compatible callable tile-reader descriptor.
// Supports legacy invocation `read_tile(rs, re, cs, ce)` (lower-triangle ColMajor),
// nullptr assignment, boolean test, and implicit construction from any matching
// callable, while also carrying fast-path hooks for contiguous transposed tiles
// (`read_T`), direct diagonal pread (`read_diagonal_fn`), bounded-byte row-band
// partitioning (`row_bands_fn`), double-buffered async band prefetch (`begin_sweep_fn`),
// and explicit cache release (`release_fn`).
template <typename MatrixT>
struct TileReaderBase {
    using LegacyFn = std::function<Eigen::Ref<const MatrixT>(int rs, int re, int cs, int ce)>;

    LegacyFn read_fn;
    LegacyFn read_T_fn;
    std::function<Eigen::VectorXd()> read_diagonal_fn;
    std::function<std::vector<int>(int max_rows_per_band)> row_bands_fn;
    std::function<std::unique_ptr<TileSweep>(const std::vector<int>& bands)> begin_sweep_fn;
    std::function<void()> release_fn;
    std::function<void(const Eigen::Ref<const MatrixT>& X,
                       MatrixT& Y,
                       double* trace_K_out,
                       double* trace_K2_out)> fused_symm_fn;
    int small_k_block_size = 0;
    int fused_max_k = 128;

    TileReaderBase() = default;
    TileReaderBase(std::nullptr_t) noexcept {}
    TileReaderBase(const TileReaderBase&) = default;
    TileReaderBase(TileReaderBase&&) noexcept = default;
    TileReaderBase& operator=(const TileReaderBase&) = default;
    TileReaderBase& operator=(TileReaderBase&&) noexcept = default;

    TileReaderBase& operator=(std::nullptr_t) noexcept {
        read_fn = nullptr;
        read_T_fn = nullptr;
        read_diagonal_fn = nullptr;
        row_bands_fn = nullptr;
        begin_sweep_fn = nullptr;
        release_fn = nullptr;
        fused_symm_fn = nullptr;
        small_k_block_size = 0;
        fused_max_k = 128;
        return *this;
    }

    template <typename Fn,
              typename Decay = std::decay_t<Fn>,
              std::enable_if_t<!std::is_same_v<Decay, TileReaderBase> &&
                               !std::is_same_v<Decay, std::nullptr_t> &&
                               std::is_invocable_r_v<Eigen::Ref<const MatrixT>, Fn&, int, int, int, int>,
                               int> = 0>
    TileReaderBase(Fn&& fn) : read_fn(std::forward<Fn>(fn)) {}

    template <typename Fn,
              typename Decay = std::decay_t<Fn>,
              std::enable_if_t<!std::is_same_v<Decay, TileReaderBase> &&
                               !std::is_same_v<Decay, std::nullptr_t> &&
                               std::is_invocable_r_v<Eigen::Ref<const MatrixT>, Fn&, int, int, int, int>,
                               int> = 0>
    TileReaderBase& operator=(Fn&& fn) {
        *this = nullptr;
        read_fn = std::forward<Fn>(fn);
        return *this;
    }

    explicit operator bool() const noexcept {
        return static_cast<bool>(read_fn) || static_cast<bool>(read_T_fn);
    }

    Eigen::Ref<const MatrixT> operator()(int rs, int re, int cs, int ce) const {
        return read_fn(rs, re, cs, ce);
    }

    bool has_transposed() const noexcept {
        return static_cast<bool>(read_T_fn);
    }

    Eigen::Ref<const MatrixT> read_T(int rs, int re, int cs, int ce) const {
        return read_T_fn(rs, re, cs, ce);
    }

    std::vector<int> compute_row_bands(int n, int max_rows_per_band) const {
        const int step = std::max(1, max_rows_per_band);
        if (row_bands_fn) {
            return row_bands_fn(step);
        }
        std::vector<int> bands;
        bands.reserve(static_cast<size_t>((n + step - 1) / step) + 1);
        for (int s = 0; s < n; s += step) bands.push_back(s);
        bands.push_back(n);
        return bands;
    }

    std::unique_ptr<TileSweep> begin_sweep(const std::vector<int>& bands) const {
        if (begin_sweep_fn && bands.size() > 2) {
            return begin_sweep_fn(bands);
        }
        return nullptr;
    }

    void release() const {
        if (release_fn) release_fn();
    }
};

using TileReader    = TileReaderBase<Eigen::MatrixXd>;
using TileReaderF32 = TileReaderBase<Eigen::MatrixXf>;

struct BlockPartition {
    std::vector<int> starts;  // starts[i]..starts[i+1) is block i; starts.back() == n

    explicit BlockPartition(int n, int block_size) {
        const int step = std::max(1, block_size);
        for (int s = 0; s < n; s += step) starts.push_back(s);
        starts.push_back(n);
    }
    int num_blocks() const { return static_cast<int>(starts.size()) - 1; }
    int block_start(int i) const { return starts[i]; }
    int block_end(int i) const { return starts[i + 1]; }
};

// Solve for the column-tile / max-block width that fits a given memory budget.
// Preserves the dense-fallback trigger (returning `n` when the entire n x n matrix
// fits in budget_gb - reserved_gb - Y), while capping chunked tile width at
// kDefaultColTile (4096) so tile scratch stays L3-resident (~32-128 MB) and
// disk-backed row bands remain bounded by ChunkedGrmReader's byte budget.
inline int solve_chunk_rows(int n, double budget_gb, int k_ext, double reserved_gb = 0.0) {
    const double y_bytes = 8.0 * static_cast<double>(n) * static_cast<double>(k_ext);
    const double budget_bytes = std::max(0.0, budget_gb * 1e9 - reserved_gb * 1e9 - y_bytes);
    const double block_size_d = std::sqrt(budget_bytes / (2.0 * 8.0));
    const int raw_chunk = static_cast<int>(block_size_d);
    if (raw_chunk >= n) return n;
    return std::min({std::max(raw_chunk, 0), n, kDefaultColTile});
}

namespace detail {

template <typename Scalar>
inline void parallel_zero_matrix(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M) noexcept {
    const Eigen::Index rows = M.rows();
    const Eigen::Index cols = M.cols();
    if (rows <= 0 || cols <= 0) return;
    const size_t total_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t col_bytes = static_cast<size_t>(rows) * sizeof(Scalar);
    #pragma omp parallel for schedule(static) if (total_elems >= 65536)
    for (Eigen::Index c = 0; c < cols; ++c) {
        std::memset(M.col(c).data(), 0, col_bytes);
    }
}

// Cache-blocked 64x64 mirror of upper triangle (r <= c) into strict lower triangle (r > c)
// of a column-major dim x dim BF16 tile (L3-resident, <= 32-128 MB).
inline void mirror_upper_to_lower_bf16(uint16_t* T_bf16, int dim) noexcept {
    constexpr int kMir = 64;
    #pragma omp parallel for schedule(guided) if (dim >= 256)
    for (int c0 = 0; c0 < dim; c0 += kMir) {
        const int c1 = std::min(dim, c0 + kMir);
        for (int r0 = c0; r0 < dim; r0 += kMir) {
            const int r1 = std::min(dim, r0 + kMir);
            for (int c = c0; c < c1; ++c) {
                const int r_start = std::max(r0, c + 1);
                uint16_t* dst_col = T_bf16 + static_cast<size_t>(c) * static_cast<size_t>(dim);
                for (int r = r_start; r < r1; ++r) {
                    dst_col[r] = T_bf16[c + static_cast<size_t>(r) * static_cast<size_t>(dim)];
                }
            }
        }
    }
}

inline void convert_colmajor_f32_to_bf16(
    const float* src, gcta_blas_int ld_src,
    uint16_t* dst, gcta_blas_int rows, gcta_blas_int cols) noexcept
{
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    #pragma omp parallel for schedule(static) if (total >= 32768)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        gcta_f32_to_bf16_rne(
            src + static_cast<size_t>(c) * static_cast<size_t>(ld_src),
            dst + static_cast<size_t>(c) * static_cast<size_t>(rows),
            static_cast<size_t>(rows));
    }
}

} // namespace detail

// Upper-triangular tiled dense symmetric matvec: Y = K * X
// where K is n x n ColMajor float32 with ONLY its upper triangle (r <= c) valid.
// - When use_bf16 == false (exact FP32): replaces monolithic cblas_ssymm with
//   zero-copy 4096x4096 upper-triangular cblas_sgemm (>96% of FLOPs) + diagonal cblas_ssymm.
// - When use_bf16 == true: converts X once to BF16 and tiles K in 4096x4096 L3-resident
//   BF16 blocks (mirroring diagonal blocks in scratch), executing via
//   gcta_gemm_bf16bf16f32_colmajor (Zen 4 AVX-512 BF16 vdpbf16ps under AOCL-DLP).
inline Eigen::MatrixXf dense_upper_symm_f32(
    const Eigen::MatrixXf& K,
    const Eigen::Ref<const Eigen::MatrixXf>& X,
    bool use_bf16 = false)
{
    const int n = static_cast<int>(K.rows());
    const int cols = static_cast<int>(X.cols());
    use_bf16 = use_bf16 && (cols >= 128);
    Eigen::MatrixXf Y(n, cols);
    if (n == 0 || cols == 0) return Y;

    const float* K_ptr = K.data();
    const float* X_ptr = X.data();
    float* Y_ptr = Y.data();
    const gcta_blas_int ldK = static_cast<gcta_blas_int>(K.outerStride());
    const gcta_blas_int ldX = static_cast<gcta_blas_int>(X.outerStride());
    const gcta_blas_int ldY = static_cast<gcta_blas_int>(n);

    if (!use_bf16) {
        if (cols == 1) {
            cblas_ssymv(CblasColMajor, CblasUpper,
                        static_cast<gcta_blas_int>(n), 1.0f,
                        K_ptr, ldK, X_ptr, 1, 0.0f, Y_ptr, 1);
            return Y;
        }
        constexpr int kTile = kDefaultColTile;
        if (n <= kTile || cols < 128) {
            cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
                        static_cast<gcta_blas_int>(n), static_cast<gcta_blas_int>(cols),
                        1.0f, K_ptr, ldK, X_ptr, ldX, 0.0f, Y_ptr, ldY);
            return Y;
        }

        detail::parallel_zero_matrix(Y);
        const int m = (n + kTile - 1) / kTile;
        for (int j = 0; j < m; ++j) {
            const int cs = j * kTile;
            const int ce = std::min(n, cs + kTile);
            const gcta_blas_int cols_j = static_cast<gcta_blas_int>(ce - cs);

            for (int i = 0; i < j; ++i) {
                const int rs = i * kTile;
                const gcta_blas_int rows_i = static_cast<gcta_blas_int>(kTile);
                const float* K_ij = K_ptr + rs + static_cast<size_t>(cs) * static_cast<size_t>(ldK);

                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            rows_i, static_cast<gcta_blas_int>(cols), cols_j,
                            1.0f, K_ij, ldK,
                            X_ptr + cs, ldX,
                            1.0f, Y_ptr + rs, ldY);
                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            cols_j, static_cast<gcta_blas_int>(cols), rows_i,
                            1.0f, K_ij, ldK,
                            X_ptr + rs, ldX,
                            1.0f, Y_ptr + cs, ldY);
            }

            const float* K_jj = K_ptr + cs + static_cast<size_t>(cs) * static_cast<size_t>(ldK);
            cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
                        cols_j, static_cast<gcta_blas_int>(cols),
                        1.0f, K_jj, ldK,
                        X_ptr + cs, ldX,
                        1.0f, Y_ptr + cs, ldY);
        }
        return Y;
    }

    // BF16 tiled path
    detail::parallel_zero_matrix(Y);

    AlignedRawBuffer<uint16_t, 64> X_bf16;
    X_bf16.resize(static_cast<size_t>(n) * static_cast<size_t>(cols));
    detail::convert_colmajor_f32_to_bf16(
        X_ptr, ldX, X_bf16.data(),
        static_cast<gcta_blas_int>(n), static_cast<gcta_blas_int>(cols));

    constexpr int kTile = kDefaultColTile;
    const int tile_cap = std::min(n, kTile);
    AlignedRawBuffer<uint16_t, 64> T_bf16(static_cast<size_t>(tile_cap) * static_cast<size_t>(tile_cap));

    const int m = (n + kTile - 1) / kTile;
    for (int j = 0; j < m; ++j) {
        const int cs = j * kTile;
        const int ce = std::min(n, cs + kTile);
        const int cols_j = ce - cs;

        for (int i = 0; i < j; ++i) {
            const int rs = i * kTile;
            const int rows_i = kTile;
            const float* K_ij = K_ptr + rs + static_cast<size_t>(cs) * static_cast<size_t>(ldK);

            detail::convert_colmajor_f32_to_bf16(
                K_ij, ldK, T_bf16.data(),
                static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols_j));

            gcta_gemm_bf16bf16f32_colmajor(
                'N', 'N',
                static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                static_cast<gcta_blas_int>(cols_j),
                1.0f, T_bf16.data(), static_cast<gcta_blas_int>(rows_i),
                X_bf16.data() + cs, static_cast<gcta_blas_int>(n),
                1.0f, Y_ptr + rs, ldY);
            gcta_gemm_bf16bf16f32_colmajor(
                'T', 'N',
                static_cast<gcta_blas_int>(cols_j), static_cast<gcta_blas_int>(cols),
                static_cast<gcta_blas_int>(rows_i),
                1.0f, T_bf16.data(), static_cast<gcta_blas_int>(rows_i),
                X_bf16.data() + rs, static_cast<gcta_blas_int>(n),
                1.0f, Y_ptr + cs, ldY);
        }

        // Diagonal tile (j, j): convert upper triangle and mirror in scratch
        const float* K_jj = K_ptr + cs + static_cast<size_t>(cs) * static_cast<size_t>(ldK);
        #pragma omp parallel for schedule(guided) if (cols_j >= 256)
        for (int c = 0; c < cols_j; ++c) {
            gcta_f32_to_bf16_rne(
                K_jj + static_cast<size_t>(c) * static_cast<size_t>(ldK),
                T_bf16.data() + static_cast<size_t>(c) * static_cast<size_t>(cols_j),
                static_cast<size_t>(c + 1));
        }
        detail::mirror_upper_to_lower_bf16(T_bf16.data(), cols_j);
        gcta_gemm_bf16bf16f32_colmajor(
            'N', 'N',
            static_cast<gcta_blas_int>(cols_j), static_cast<gcta_blas_int>(cols),
            static_cast<gcta_blas_int>(cols_j),
            1.0f, T_bf16.data(), static_cast<gcta_blas_int>(cols_j),
            X_bf16.data() + cs, static_cast<gcta_blas_int>(n),
            1.0f, Y_ptr + cs, ldY);
    }
    return Y;
}

// K @ X without ever holding a dense n x n K. Uses decoupled bounded-byte row bands
// and optional double-buffered async band prefetch + contiguous transposed tiles.
inline Eigen::MatrixXd chunked_symmetric_matvec(
    const TileReader& read_tile,
    int n,
    int block_size,
    const Eigen::MatrixXd& X)
{
    const int cols = static_cast<int>(X.cols());
    Eigen::MatrixXd Y(n, cols);
    if (n == 0 || cols == 0) return Y;
    detail::parallel_zero_matrix(Y);

    const int eff_block = std::min(n, std::max(1, block_size));
    const std::vector<int> bands = read_tile.compute_row_bands(n, eff_block);
    const int num_bands = static_cast<int>(bands.size()) - 1;
    auto sweep = read_tile.begin_sweep(bands);

    const bool use_T = read_tile.has_transposed();
    const double* X_ptr = X.data();
    double* Y_ptr = Y.data();
    const gcta_blas_int ldX = static_cast<gcta_blas_int>(X.outerStride());
    const gcta_blas_int ldY = static_cast<gcta_blas_int>(n);

    for (int b = 0; b < num_bands; ++b) {
        if (sweep) sweep->acquire(b);
        const int rs = bands[b];
        const int re = bands[b + 1];
        const int rows_i = re - rs;

        // Strictly off-diagonal column tiles [cs, ce) with ce <= rs
        for (int cs = 0; cs < rs; cs += eff_block) {
            const int ce = std::min(cs + eff_block, rs);
            const int cols_j = ce - cs;
            if (use_T) {
                auto tile_T = read_tile.read_T(rs, re, cs, ce);  // cols_j x rows_i
                const gcta_blas_int ldT = static_cast<gcta_blas_int>(tile_T.outerStride());
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                            static_cast<gcta_blas_int>(cols_j),
                            1.0, tile_T.data(), ldT,
                            X_ptr + cs, ldX,
                            1.0, Y_ptr + rs, ldY);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<gcta_blas_int>(cols_j), static_cast<gcta_blas_int>(cols),
                            static_cast<gcta_blas_int>(rows_i),
                            1.0, tile_T.data(), ldT,
                            X_ptr + rs, ldX,
                            1.0, Y_ptr + cs, ldY);
            } else {
                auto tile = read_tile(rs, re, cs, ce);
                Y.middleRows(rs, rows_i).noalias() += tile * X.middleRows(cs, cols_j);
                Y.middleRows(cs, cols_j).noalias() += tile.transpose() * X.middleRows(rs, rows_i);
            }
        }

        // Square diagonal tile [rs, re) x [rs, re)
        if (use_T) {
            auto tile_T = read_tile.read_T(rs, re, rs, re);  // rows_i x rows_i, UPPER valid
            const gcta_blas_int ldT = static_cast<gcta_blas_int>(tile_T.outerStride());
            cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper,
                        static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                        1.0, tile_T.data(), ldT,
                        X_ptr + rs, ldX,
                        1.0, Y_ptr + rs, ldY);
        } else {
            auto tile = read_tile(rs, re, rs, re);
            Y.middleRows(rs, rows_i).noalias() +=
                tile.selfadjointView<Eigen::Lower>() * X.middleRows(rs, rows_i);
        }

        if (sweep) sweep->release(b);
    }
    return Y;
}

// Single-precision counterpart to chunked_symmetric_matvec: keeps tiles and output
// accumulator Y in native float32, with optional BF16 GEMM acceleration for rSVD
// sketch/power passes and optional fused trace(K)/trace(K^2) accumulation.
// For in-memory PackedGrmFp16 with small X.cols() (< 256, e.g. PCA), automatically
// upgrades tile size to small_k_block_size (8192) to avoid small-k tile starvation.
inline Eigen::MatrixXf chunked_symmetric_matvec_f32(
    const TileReaderF32& read_tile,
    int n,
    int block_size,
    const Eigen::Ref<const Eigen::MatrixXf>& X,
    double* trace_K_out = nullptr,
    double* trace_K2_out = nullptr,
    bool use_bf16 = false)
{
    const int cols = static_cast<int>(X.cols());
    Eigen::MatrixXf Y(n, cols);
    if (n == 0 || cols == 0) {
        if (trace_K_out) *trace_K_out = 0.0;
        if (trace_K2_out) *trace_K2_out = 0.0;
        return Y;
    }
    if (read_tile.fused_symm_fn && read_tile.fused_max_k > 0 && cols <= read_tile.fused_max_k) {
        read_tile.fused_symm_fn(X, Y, trace_K_out, trace_K2_out);
        return Y;
    }
    detail::parallel_zero_matrix(Y);

    int eff_block = std::min(n, std::max(1, block_size));
    if (cols < 256 && read_tile.small_k_block_size > eff_block) {
        eff_block = std::min(n, read_tile.small_k_block_size);
    }

    const std::vector<int> bands = read_tile.compute_row_bands(n, eff_block);
    const int num_bands = static_cast<int>(bands.size()) - 1;
    auto sweep = read_tile.begin_sweep(bands);

    const bool use_T    = read_tile.has_transposed();
    const bool want_tr  = (trace_K_out != nullptr);
    const bool want_tr2 = (trace_K2_out != nullptr);
    double tr_K  = 0.0;
    double tr_K2 = 0.0;

    const float* X_ptr = X.data();
    float* Y_ptr = Y.data();
    const gcta_blas_int ldX = static_cast<gcta_blas_int>(X.outerStride());
    const gcta_blas_int ldY = static_cast<gcta_blas_int>(n);

    AlignedRawBuffer<uint16_t, 64> X_bf16;
    AlignedRawBuffer<uint16_t, 64> T_bf16;
    if (use_bf16) {
        X_bf16.resize(static_cast<size_t>(n) * static_cast<size_t>(cols));
        detail::convert_colmajor_f32_to_bf16(
            X_ptr, ldX, X_bf16.data(),
            static_cast<gcta_blas_int>(n), static_cast<gcta_blas_int>(cols));
        int max_band_rows = 0;
        for (int b = 0; b < num_bands; ++b) {
            max_band_rows = std::max(max_band_rows, bands[b + 1] - bands[b]);
        }
        T_bf16.resize(static_cast<size_t>(eff_block) * static_cast<size_t>(std::max(max_band_rows, 1)));
    }

    for (int b = 0; b < num_bands; ++b) {
        if (sweep) sweep->acquire(b);
        const int rs = bands[b];
        const int re = bands[b + 1];
        const int rows_i = re - rs;

        // Strictly off-diagonal column tiles [cs, ce) with ce <= rs
        for (int cs = 0; cs < rs; cs += eff_block) {
            const int ce = std::min(cs + eff_block, rs);
            const int cols_j = ce - cs;

            if (use_T) {
                auto tile_T = read_tile.read_T(rs, re, cs, ce);  // cols_j x rows_i
                const gcta_blas_int ldT = static_cast<gcta_blas_int>(tile_T.outerStride());
                if (want_tr2) {
                    double off_sq = 0.0;
                    #pragma omp parallel for reduction(+:off_sq) schedule(static)
                    for (int c = 0; c < rows_i; ++c)
                        off_sq += tile_T.col(c).cast<double>().squaredNorm();
                    tr_K2 += 2.0 * off_sq;
                }
                if (use_bf16) {
                    detail::convert_colmajor_f32_to_bf16(
                        tile_T.data(), ldT, T_bf16.data(),
                        static_cast<gcta_blas_int>(cols_j), static_cast<gcta_blas_int>(rows_i));
                    gcta_gemm_bf16bf16f32_colmajor(
                        'T', 'N',
                        static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                        static_cast<gcta_blas_int>(cols_j),
                        1.0f, T_bf16.data(), static_cast<gcta_blas_int>(cols_j),
                        X_bf16.data() + cs, static_cast<gcta_blas_int>(n),
                        1.0f, Y_ptr + rs, ldY);
                    gcta_gemm_bf16bf16f32_colmajor(
                        'N', 'N',
                        static_cast<gcta_blas_int>(cols_j), static_cast<gcta_blas_int>(cols),
                        static_cast<gcta_blas_int>(rows_i),
                        1.0f, T_bf16.data(), static_cast<gcta_blas_int>(cols_j),
                        X_bf16.data() + rs, static_cast<gcta_blas_int>(n),
                        1.0f, Y_ptr + cs, ldY);
                } else {
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                                static_cast<gcta_blas_int>(cols_j),
                                1.0f, tile_T.data(), ldT,
                                X_ptr + cs, ldX,
                                1.0f, Y_ptr + rs, ldY);
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                static_cast<gcta_blas_int>(cols_j), static_cast<gcta_blas_int>(cols),
                                static_cast<gcta_blas_int>(rows_i),
                                1.0f, tile_T.data(), ldT,
                                X_ptr + rs, ldX,
                                1.0f, Y_ptr + cs, ldY);
                }
            } else {
                auto tile = read_tile(rs, re, cs, ce);  // rows_i x cols_j
                const gcta_blas_int ldT = static_cast<gcta_blas_int>(tile.outerStride());
                if (want_tr2) {
                    double off_sq = 0.0;
                    #pragma omp parallel for reduction(+:off_sq) schedule(static)
                    for (int c = 0; c < cols_j; ++c)
                        off_sq += tile.col(c).cast<double>().squaredNorm();
                    tr_K2 += 2.0 * off_sq;
                }
                if (use_bf16) {
                    detail::convert_colmajor_f32_to_bf16(
                        tile.data(), ldT, T_bf16.data(),
                        static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols_j));
                    gcta_gemm_bf16bf16f32_colmajor(
                        'N', 'N',
                        static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                        static_cast<gcta_blas_int>(cols_j),
                        1.0f, T_bf16.data(), static_cast<gcta_blas_int>(rows_i),
                        X_bf16.data() + cs, static_cast<gcta_blas_int>(n),
                        1.0f, Y_ptr + rs, ldY);
                    gcta_gemm_bf16bf16f32_colmajor(
                        'T', 'N',
                        static_cast<gcta_blas_int>(cols_j), static_cast<gcta_blas_int>(cols),
                        static_cast<gcta_blas_int>(rows_i),
                        1.0f, T_bf16.data(), static_cast<gcta_blas_int>(rows_i),
                        X_bf16.data() + rs, static_cast<gcta_blas_int>(n),
                        1.0f, Y_ptr + cs, ldY);
                } else {
                    Y.middleRows(rs, rows_i).noalias() += tile * X.middleRows(cs, cols_j);
                    Y.middleRows(cs, cols_j).noalias() += tile.transpose() * X.middleRows(rs, rows_i);
                }
            }
        }

        // Square diagonal tile [rs, re) x [rs, re)
        if (use_T) {
            auto tile_T = read_tile.read_T(rs, re, rs, re);  // rows_i x rows_i, UPPER valid
            const gcta_blas_int ldT = static_cast<gcta_blas_int>(tile_T.outerStride());
            if (want_tr) {
                tr_K += tile_T.diagonal().cast<double>().sum();
            }
            if (want_tr2) {
                double upper_incl_diag_sq = 0.0;
                #pragma omp parallel for reduction(+:upper_incl_diag_sq) schedule(guided)
                for (int c = 0; c < rows_i; ++c)
                    upper_incl_diag_sq += tile_T.col(c).head(c + 1).cast<double>().squaredNorm();
                const double diag_sq = tile_T.diagonal().cast<double>().squaredNorm();
                tr_K2 += 2.0 * upper_incl_diag_sq - diag_sq;
            }
            if (use_bf16) {
                #pragma omp parallel for schedule(guided) if (rows_i >= 256)
                for (int c = 0; c < rows_i; ++c) {
                    gcta_f32_to_bf16_rne(
                        tile_T.data() + static_cast<size_t>(c) * static_cast<size_t>(ldT),
                        T_bf16.data() + static_cast<size_t>(c) * static_cast<size_t>(rows_i),
                        static_cast<size_t>(c + 1));
                }
                detail::mirror_upper_to_lower_bf16(T_bf16.data(), rows_i);
                gcta_gemm_bf16bf16f32_colmajor(
                    'N', 'N',
                    static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                    static_cast<gcta_blas_int>(rows_i),
                    1.0f, T_bf16.data(), static_cast<gcta_blas_int>(rows_i),
                    X_bf16.data() + rs, static_cast<gcta_blas_int>(n),
                    1.0f, Y_ptr + rs, ldY);
            } else {
                cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
                            static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                            1.0f, tile_T.data(), ldT,
                            X_ptr + rs, ldX,
                            1.0f, Y_ptr + rs, ldY);
            }
        } else {
            auto tile = read_tile(rs, re, rs, re);  // rows_i x rows_i, LOWER valid
            if (want_tr) {
                tr_K += tile.diagonal().cast<double>().sum();
            }
            if (want_tr2) {
                double lower_incl_diag_sq = 0.0;
                #pragma omp parallel for reduction(+:lower_incl_diag_sq) schedule(guided)
                for (int c = 0; c < rows_i; ++c)
                    lower_incl_diag_sq += tile.col(c).tail(rows_i - c).cast<double>().squaredNorm();
                const double diag_sq = tile.diagonal().cast<double>().squaredNorm();
                tr_K2 += 2.0 * lower_incl_diag_sq - diag_sq;
            }
            if (use_bf16) {
                const gcta_blas_int ldT = static_cast<gcta_blas_int>(tile.outerStride());
                // Convert lower triangle of tile into upper triangle of T_bf16 (transpose), then mirror
                #pragma omp parallel for schedule(guided) if (rows_i >= 256)
                for (int c = 0; c < rows_i; ++c) {
                    const float* src_col = tile.data() + static_cast<size_t>(c) * static_cast<size_t>(ldT);
                    for (int r = c; r < rows_i; ++r) {
                        uint16_t h = 0;
                        gcta_f32_to_bf16_rne(src_col + r, &h, 1);
                        T_bf16[c + static_cast<size_t>(r) * static_cast<size_t>(rows_i)] = h;
                    }
                }
                detail::mirror_upper_to_lower_bf16(T_bf16.data(), rows_i);
                gcta_gemm_bf16bf16f32_colmajor(
                    'N', 'N',
                    static_cast<gcta_blas_int>(rows_i), static_cast<gcta_blas_int>(cols),
                    static_cast<gcta_blas_int>(rows_i),
                    1.0f, T_bf16.data(), static_cast<gcta_blas_int>(rows_i),
                    X_bf16.data() + rs, static_cast<gcta_blas_int>(n),
                    1.0f, Y_ptr + rs, ldY);
            } else {
                Y.middleRows(rs, rows_i).noalias() +=
                    tile.selfadjointView<Eigen::Lower>() * X.middleRows(rs, rows_i);
            }
        }

        if (sweep) sweep->release(b);
    }

    if (trace_K_out)  *trace_K_out  = tr_K;
    if (trace_K2_out) *trace_K2_out = tr_K2;
    return Y;
}

// Diagonal of K without reading any off-diagonal data: uses direct diagonal pread
// (on ChunkedGrmReader) or exact cached diagonal (on PackedGrmFp16) whenever available.
inline Eigen::VectorXd chunked_diagonal(const TileReader& read_tile, int n, int block_size) {
    if (read_tile.read_diagonal_fn) {
        return read_tile.read_diagonal_fn();
    }
    const int eff_block = std::min(n, std::max(1, block_size));
    const std::vector<int> bands = read_tile.compute_row_bands(n, eff_block);
    const int num_bands = static_cast<int>(bands.size()) - 1;
    const bool use_T = read_tile.has_transposed();
    Eigen::VectorXd d(n);
    for (int b = 0; b < num_bands; ++b) {
        const int rs = bands[b], re = bands[b + 1];
        if (use_T) {
            const auto tile_T = read_tile.read_T(rs, re, rs, re);
            d.segment(rs, re - rs) = tile_T.diagonal();
        } else {
            const auto tile = read_tile(rs, re, rs, re);
            d.segment(rs, re - rs) = tile.diagonal();
        }
    }
    return d;
}

// trace(K^2) = sum of squares of every entry of K.
inline double chunked_trace_K_squared(const TileReader& read_tile, int n, int block_size) {
    const int eff_block = std::min(n, std::max(1, block_size));
    const std::vector<int> bands = read_tile.compute_row_bands(n, eff_block);
    const int num_bands = static_cast<int>(bands.size()) - 1;
    auto sweep = read_tile.begin_sweep(bands);
    const bool use_T = read_tile.has_transposed();

    double total = 0.0;
    for (int b = 0; b < num_bands; ++b) {
        if (sweep) sweep->acquire(b);
        const int rs = bands[b], re = bands[b + 1];
        const int rows_i = re - rs;

        for (int cs = 0; cs < rs; cs += eff_block) {
            const int ce = std::min(cs + eff_block, rs);
            if (use_T) {
                auto tile_T = read_tile.read_T(rs, re, cs, ce);
                double off_sq = 0.0;
                #pragma omp parallel for reduction(+:off_sq) schedule(static)
                for (int c = 0; c < rows_i; ++c)
                    off_sq += tile_T.col(c).squaredNorm();
                total += 2.0 * off_sq;
            } else {
                auto tile = read_tile(rs, re, cs, ce);
                double off_sq = 0.0;
                #pragma omp parallel for reduction(+:off_sq) schedule(static)
                for (int c = 0; c < tile.cols(); ++c)
                    off_sq += tile.col(c).squaredNorm();
                total += 2.0 * off_sq;
            }
        }

        if (use_T) {
            auto tile_T = read_tile.read_T(rs, re, rs, re);
            double upper_incl_diag_sq = 0.0;
            #pragma omp parallel for reduction(+:upper_incl_diag_sq) schedule(guided)
            for (int c = 0; c < rows_i; ++c)
                upper_incl_diag_sq += tile_T.col(c).head(c + 1).squaredNorm();
            const double diag_sq = tile_T.diagonal().squaredNorm();
            total += 2.0 * upper_incl_diag_sq - diag_sq;
        } else {
            auto tile = read_tile(rs, re, rs, re);
            double lower_incl_diag_sq = 0.0;
            #pragma omp parallel for reduction(+:lower_incl_diag_sq) schedule(guided)
            for (int c = 0; c < rows_i; ++c)
                lower_incl_diag_sq += tile.col(c).tail(rows_i - c).squaredNorm();
            const double diag_sq = tile.diagonal().squaredNorm();
            total += 2.0 * lower_incl_diag_sq - diag_sq;
        }

        if (sweep) sweep->release(b);
    }
    return total;
}

} // namespace gcta_chunked
