/*
 * symmetric_eigendecomp.hpp
 *
 * Approximate top-k eigendecomposition methods for a symmetric (typically
 * PSD) matrix A, shared between grm.cpp (gcta::pca), pca_stream.cpp, and
 * RemlEngine.cpp (compute_woodbury_basis).
 *
 * Every entry point takes a matvec functor `apply`:
 *   - Legacy 1-arg: `apply(X)` returns A * X.
 *   - Precision-aware 2-arg: `apply(X, bool exact_fp32)` allows low-precision
 *     BF16 GEMM on sketch/power passes (`exact_fp32 = false`) while keeping
 *     final Rayleigh-Ritz projections and single-pass Nystrom in exact FP32
 *     (`exact_fp32 = true`).
 *
 * Methods:
 *   - randomized_symmetric_eigh  : Halko/Martinsson/Tropp randomized range
 *                                  finder + power iteration + Rayleigh-Ritz.
 *   - expand_symmetric_eigh      : Locked Rayleigh-Ritz basis expansion.
 *   - lanczos_symmetric_eigh     : Spectra Lanczos (implicitly restarted Arnoldi).
 *   - nystrom_symmetric_eigh     : Signed single-pass Nystrom approximation.
 */
#pragma once

#include <Eigen/Dense>
#include <Spectra/SymEigsSolver.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <omp.h>

#include "cpu.h"
#include "mem.hpp"

namespace gcta_eigh {

inline Eigen::VectorXd threaded_dense_matvec(
    const Eigen::MatrixXd& A,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    int col_block = 64)
{
    (void)col_block;
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n || x.size() != n)
        throw std::invalid_argument("threaded_dense_matvec: dimension mismatch.");
    return A * x;
}

template <typename Scalar = double>
struct EighResultT {
    Eigen::VectorXd eigenvalues;   // descending, size k_target
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;  // n x k_target, columns match eigenvalues
};

using EighResult  = EighResultT<double>;
using EighResultF = EighResultT<float>;

inline std::mt19937& shared_rng() {
    static std::mt19937 rng([] {
        std::random_device entropy;
        return entropy();
    }());
    return rng;
}

// Parallel standard-normal fill: each thread gets its own mt19937 seeded
// directly from shared_rng() serially up front.
template <typename Scalar>
inline void fill_standard_normal_impl(Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> matrix) {
    const Eigen::Index rows = matrix.rows();
    const Eigen::Index cols = matrix.cols();
    if (rows == 0 || cols == 0) return;

    const int num_threads = static_cast<int>(std::min<Eigen::Index>(omp_get_max_threads(), cols));

    std::vector<std::uint32_t> thread_seeds(num_threads);
    for (int t = 0; t < num_threads; ++t)
        thread_seeds[t] = static_cast<std::uint32_t>(shared_rng()());

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        std::mt19937 local_rng(thread_seeds[tid]);
        std::normal_distribution<double> normal(0.0, 1.0);

        const Eigen::Index cols_per_thread = (cols + num_threads - 1) / num_threads;
        const Eigen::Index col_begin = std::min(cols, static_cast<Eigen::Index>(tid) * cols_per_thread);
        const Eigen::Index col_end   = std::min(cols, col_begin + cols_per_thread);

        for (Eigen::Index col = col_begin; col < col_end; ++col)
            for (Eigen::Index row = 0; row < rows; ++row)
                matrix(row, col) = static_cast<Scalar>(normal(local_rng));
    }
}

inline void fill_standard_normal(Eigen::Ref<Eigen::MatrixXd> matrix) {
    fill_standard_normal_impl<double>(matrix);
}

inline void fill_standard_normal(Eigen::Ref<Eigen::MatrixXf> matrix) {
    fill_standard_normal_impl<float>(matrix);
}

inline int recommended_oversample(int k_target, int floor = 8, int cap = 128) {
    if (k_target <= 0) return 0;
    const int soft_target = std::max(8, k_target);
    return std::clamp(k_target / 8, std::min(floor, soft_target), cap);
}

namespace detail {

template <typename MatVecApply, typename MatArg>
auto invoke_apply(MatVecApply&& apply, const MatArg& X, bool exact_fp32) {
    if constexpr (requires { apply(X, exact_fp32); }) {
        return apply(X, exact_fp32);
    } else {
        (void)exact_fp32;
        return apply(X);
    }
}

template <typename MatVecApply, typename MatArg>
constexpr bool supports_precision_flag() {
    return requires(MatVecApply&& a, const MatArg& x) { a(x, false); };
}

// Compute B (k x k, double, UPPER triangle valid) = A^T * C where A and C are n x k.
// Only converts/touches the upper triangle when Scalar == float, in parallel.
template <typename Scalar>
inline void form_upper_gram_double(
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& A,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& C,
    Eigen::MatrixXd& B)
{
    const Eigen::Index k = A.cols();
    B.resize(k, k);
    if constexpr (std::is_same_v<Scalar, double>) {
        B.noalias() = A.transpose() * C;
    } else {
        Eigen::MatrixXf B_f(k, k);
        B_f.noalias() = A.transpose() * C;
        #pragma omp parallel for schedule(guided) if (k >= 128)
        for (Eigen::Index j = 0; j < k; ++j) {
            const float* __restrict__ src = B_f.col(j).data();
            double* __restrict__ dst = B.col(j).data();
            #pragma omp simd
            for (Eigen::Index i = 0; i <= j; ++i) {
                dst[i] = static_cast<double>(src[i]);
            }
        }
    }
}

// Parallel cast of a rectangular matrix (rows x cols) from Scalar to double into a destination block.
template <typename Scalar, typename DstExpr>
inline void assign_rect_to_double_parallel(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& src,
    DstExpr dst)
{
    const Eigen::Index rows = src.rows();
    const Eigen::Index cols = src.cols();
    if constexpr (std::is_same_v<Scalar, double>) {
        dst.noalias() = src;
    } else {
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(rows) * static_cast<size_t>(cols) >= 32768)
        for (Eigen::Index j = 0; j < cols; ++j) {
            const Scalar* __restrict__ s = src.col(j).data();
            double* __restrict__ d = dst.col(j).data();
            #pragma omp simd
            for (Eigen::Index i = 0; i < rows; ++i) {
                d[i] = static_cast<double>(s[i]);
            }
        }
    }
}

// Parallel cast of a double block into a Scalar matrix.
template <typename Scalar, typename SrcExpr>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
cast_double_block_parallel(const SrcExpr& src)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const Eigen::Index rows = src.rows();
    const Eigen::Index cols = src.cols();
    Mat out(rows, cols);
    if constexpr (std::is_same_v<Scalar, double>) {
        out.noalias() = src;
    } else {
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(rows) * static_cast<size_t>(cols) >= 32768)
        for (Eigen::Index j = 0; j < cols; ++j) {
            Scalar* __restrict__ d = out.col(j).data();
            for (Eigen::Index i = 0; i < rows; ++i) {
                d[i] = static_cast<Scalar>(src(i, j));
            }
        }
    }
    return out;
}

// In-place basis rotation: Q <- Q * Z.leftCols(k_target) in 4096-row slabs.
// Frees double Z immediately after casting its leading k_target columns to Scalar,
// eliminating the 2x peak RSS spike (saves ~40 GB at N=500k, k=20k).
template <typename Scalar>
inline void rotate_basis_inplace(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
    Eigen::MatrixXd& Z,
    int k_target)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int n = static_cast<int>(Q.rows());
    const int k_ext = static_cast<int>(Q.cols());
    if (n == 0 || k_target == 0) {
        Q.resize(n, k_target);
        Z.resize(0, 0);
        return;
    }

    Mat Z_s = cast_double_block_parallel<Scalar>(Z.leftCols(k_target));
    Z.resize(0, 0);

    constexpr int kSlabRows = 4096;
    const int max_rb = std::min(n, kSlabRows);
    AlignedRawBuffer<Scalar, 64> slab(static_cast<size_t>(max_rb) * static_cast<size_t>(k_target));
    Scalar* Q_ptr = Q.data();
    const Scalar* Z_ptr = Z_s.data();
    const gcta_blas_int ldQ = static_cast<gcta_blas_int>(n);
    const gcta_blas_int ldZ = static_cast<gcta_blas_int>(k_ext);

    for (int r0 = 0; r0 < n; r0 += kSlabRows) {
        const int rb = std::min(kSlabRows, n - r0);
        if constexpr (std::is_same_v<Scalar, float>) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<gcta_blas_int>(rb),
                        static_cast<gcta_blas_int>(k_target),
                        static_cast<gcta_blas_int>(k_ext),
                        1.0f, Q_ptr + r0, ldQ,
                        Z_ptr, ldZ,
                        0.0f, slab.data(), static_cast<gcta_blas_int>(rb));
        } else {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<gcta_blas_int>(rb),
                        static_cast<gcta_blas_int>(k_target),
                        static_cast<gcta_blas_int>(k_ext),
                        1.0, Q_ptr + r0, ldQ,
                        Z_ptr, ldZ,
                        0.0, slab.data(), static_cast<gcta_blas_int>(rb));
        }
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(rb) * static_cast<size_t>(k_target) >= 32768)
        for (int j = 0; j < k_target; ++j) {
            std::memcpy(Q_ptr + r0 + static_cast<size_t>(j) * static_cast<size_t>(n),
                        slab.data() + static_cast<size_t>(j) * static_cast<size_t>(rb),
                        static_cast<size_t>(rb) * sizeof(Scalar));
        }
    }

    if (k_target < k_ext) {
        Q.conservativeResize(Eigen::NoChange, k_target);
    }
}

} // namespace detail

// Full eigendecomposition of the symmetric k x k matrix whose UPPER triangle is
// stored in B. B is destroyed. On return w (k) and Z (k x k) hold eigenpairs in
// DESCENDING order.
//
// Prefers gcta_dsyevd (divide-and-conquer, Level-3 BLAS parallelized on AOCL
// libflame where dsyevr's dstemr is single-threaded) when k < 32766, swapping
// B into Z zero-copy on success; falls back to gcta_dsyevr when k >= 32766 or
// if gcta_dsyevd returns its untouched -1 workspace-overflow guard.
inline void eigh_upper_desc(Eigen::MatrixXd& B, Eigen::VectorXd& w, Eigen::MatrixXd& Z)
{
    const int k = static_cast<int>(B.rows());
    if (B.cols() != k || k < 1)
        throw std::invalid_argument("eigh_upper_desc: B must be square and non-empty.");
    w.resize(k);

    bool solved = false;
    if (k < 32766) {
        Z.resize(0, 0);  // release any prior Z before dsyevd allocates workspace
        const int info_d = gcta_dsyevd(static_cast<gcta_blas_int>(k), B.data(),
                                       static_cast<gcta_blas_int>(k), w.data());
        if (info_d == 0) {
            Z.swap(B);
            solved = true;
        } else if (info_d != -1) {
            throw std::runtime_error("eigh_upper_desc: dsyevd failed (info=" + std::to_string(info_d) + ").");
        }
    }

    if (!solved) {
        Z.resize(k, k);
        std::vector<gcta_blas_int> isuppz(2 * static_cast<size_t>(k));
        gcta_blas_int m_found = 0;
        const int info = gcta_dsyevr(static_cast<gcta_blas_int>(k), B.data(), static_cast<gcta_blas_int>(k),
                                     static_cast<gcta_blas_int>(1), static_cast<gcta_blas_int>(k), &m_found,
                                     w.data(), Z.data(), static_cast<gcta_blas_int>(k), isuppz.data());
        if (info != 0 || m_found != k)
            throw std::runtime_error("eigh_upper_desc: dsyevr failed (info=" + std::to_string(info) + ").");
    }

    // Reverse ascending -> descending in place, parallelized across column pairs.
    const int half = k / 2;
    #pragma omp parallel for schedule(static) if (k >= 128)
    for (int i = 0; i < half; ++i) {
        double* __restrict__ c1 = Z.col(i).data();
        double* __restrict__ c2 = Z.col(k - 1 - i).data();
        for (int r = 0; r < k; ++r) {
            std::swap(c1[r], c2[r]);
        }
        std::swap(w[i], w[k - 1 - i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Randomized range finder + power iteration + Rayleigh-Ritz (rSVD)
// ─────────────────────────────────────────────────────────────────────────

// Build the initial randomized sketch Y = A * omega (n x k_ext).
// If `warm_start` is supplied, its leading columns seed omega.
template <typename Scalar = double, typename MatVecApply>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
build_randomized_sketch(
    MatVecApply&& apply,
    int n,
    int k_ext,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* warm_start = nullptr,
    bool exact_fp32 = false)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Mat omega(n, k_ext);
    gcta_numa_interleave(omega.data(), static_cast<size_t>(n) * static_cast<size_t>(k_ext) * sizeof(Scalar));
    if (warm_start && warm_start->rows() == n && warm_start->cols() > 0) {
        const int k_copy = std::min(static_cast<int>(warm_start->cols()), k_ext);
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(k_copy) >= 65536)
        for (int j = 0; j < k_copy; ++j) {
            std::memcpy(omega.col(j).data(), warm_start->col(j).data(),
                        static_cast<size_t>(n) * sizeof(Scalar));
        }
        if (k_copy < k_ext)
            fill_standard_normal(omega.rightCols(k_ext - k_copy));
    } else {
        fill_standard_normal(omega);
    }
    Mat Y = detail::invoke_apply(apply, omega, exact_fp32);
    return {std::move(omega), std::move(Y)};
}

// Consuming variant: copies warm_start into omega and immediately frees *warm_start
// BEFORE calling apply(omega) so warm_start never coexists with both omega and Y.
template <typename Scalar = double, typename MatVecApply>
std::pair<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
build_randomized_sketch_consume(
    MatVecApply&& apply,
    int n,
    int k_ext,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* warm_start,
    bool exact_fp32 = false)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Mat omega(n, k_ext);
    gcta_numa_interleave(omega.data(), static_cast<size_t>(n) * static_cast<size_t>(k_ext) * sizeof(Scalar));
    if (warm_start && warm_start->rows() == n && warm_start->cols() > 0) {
        const int k_copy = std::min(static_cast<int>(warm_start->cols()), k_ext);
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(k_copy) >= 65536)
        for (int j = 0; j < k_copy; ++j) {
            std::memcpy(omega.col(j).data(), warm_start->col(j).data(),
                        static_cast<size_t>(n) * sizeof(Scalar));
        }
        warm_start->resize(0, 0);  // release warm-start memory before apply(omega) allocates Y
        if (k_copy < k_ext)
            fill_standard_normal(omega.rightCols(k_ext - k_copy));
    } else {
        if (warm_start) warm_start->resize(0, 0);
        fill_standard_normal(omega);
    }
    Mat Y = detail::invoke_apply(apply, omega, exact_fp32);
    return {std::move(omega), std::move(Y)};
}

// Core logic: given initial sketch Y = A * omega (n x k_ext), run `power_iter`
// power passes (low-precision BF16 allowed when supported), final CholeskyQR,
// and an exact FP32 Rayleigh-Ritz projection with in-place Q * Z rotation.
template <typename Scalar = double, typename MatVecApply>
EighResultT<Scalar> power_iterate_and_project(
    MatVecApply&& apply,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y,
    int k_target,
    int power_iter = 3)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int n     = static_cast<int>(Y.rows());
    const int k_ext = static_cast<int>(Y.cols());
    if (k_target > k_ext)
        throw std::invalid_argument("power_iterate_and_project: k_target exceeds sketch width k_ext.");

    for (int pi = 0; pi < power_iter; ++pi) {
        int info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                           static_cast<gcta_blas_int>(k_ext),
                                           Y.data(), static_cast<gcta_blas_int>(n));
        if (info != 0)
            throw std::runtime_error("power_iterate_and_project: orthogonalization failed (info=" +
                                      std::to_string(info) + ").");
        Y = detail::invoke_apply(apply, Y, /*exact_fp32=*/false);
    }

    {
        int info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                           static_cast<gcta_blas_int>(k_ext),
                                           Y.data(), static_cast<gcta_blas_int>(n));
        if (info != 0)
            throw std::runtime_error("power_iterate_and_project: orthogonalization failed (info=" +
                                      std::to_string(info) + ").");
    }
    Mat Q = std::move(Y);

    // Final Rayleigh-Ritz pass MUST run in exact FP32/FP64.
    Mat AQ = detail::invoke_apply(apply, Q, /*exact_fp32=*/true);
    Eigen::MatrixXd B;
    detail::form_upper_gram_double<Scalar>(Q, AQ, B);
    AQ.resize(0, 0);

    Eigen::VectorXd w;
    Eigen::MatrixXd Z;
    eigh_upper_desc(B, w, Z);
    B.resize(0, 0);

    detail::rotate_basis_inplace<Scalar>(Q, Z, k_target);

    EighResultT<Scalar> result;
    result.eigenvalues  = w.head(k_target);
    result.eigenvectors = std::move(Q);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Locked expansion: grow a converged Ritz basis without recomputing it
// ─────────────────────────────────────────────────────────────────────────
template <typename Scalar = double, typename MatVecApply>
EighResultT<Scalar> expand_symmetric_eigh(
    MatVecApply&& apply,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V0,
    const Eigen::VectorXd& theta0,
    int k_ext,
    int k_target,
    int power_iter = 3)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    const int n  = static_cast<int>(V0.rows());
    const int k0 = static_cast<int>(V0.cols());
    const int m  = k_ext - k0;
    if (theta0.size() != k0)
        throw std::invalid_argument("expand_symmetric_eigh: theta0 size does not match V0 columns.");
    if (m <= 0)
        throw std::invalid_argument("expand_symmetric_eigh: k_ext must exceed the locked basis width.");
    if (k_target > k_ext)
        throw std::invalid_argument("expand_symmetric_eigh: k_target exceeds k_ext.");
    if (k_ext > n)
        throw std::invalid_argument("expand_symmetric_eigh: k_ext exceeds n.");

    auto deflate = [&](Eigen::Ref<Mat> Ym, int reps) {
        for (int rep = 0; rep < reps; ++rep) {
            const Mat C = V0.transpose() * Ym;   // k0 x m
            Ym.noalias() -= V0 * C;
        }
    };
    auto orthonormalize = [&](Eigen::Ref<Mat> Ym) {
        const int info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                                 static_cast<gcta_blas_int>(m),
                                                 Ym.data(),
                                                 static_cast<gcta_blas_int>(Ym.outerStride()));
        if (info != 0)
            throw std::runtime_error("expand_symmetric_eigh: orthogonalization failed (info=" +
                                      std::to_string(info) + ").");
    };

    // First product: m random columns plus one probe column z = V0 g.
    Mat Y(n, m + 1);
    gcta_numa_interleave(Y.data(), static_cast<size_t>(n) * static_cast<size_t>(m + 1) * sizeof(Scalar));
    fill_standard_normal(Y.leftCols(m));
    Eigen::VectorXd g(k0);
    {
        Eigen::MatrixXd gm(k0, 1);
        fill_standard_normal(gm);
        g = gm.col(0);
    }
    Y.col(m).noalias() = V0 * g.template cast<Scalar>();
    Y = detail::invoke_apply(apply, Y, /*exact_fp32=*/false);
    const Vec Kz = Y.col(m);
    {
        constexpr bool bf16_sketch =
            std::is_same_v<Scalar, float> &&
            detail::supports_precision_flag<MatVecApply, Mat>() &&
            (GCTA_HAS_AOCL_BF16_GEMM != 0);
        constexpr double ritz_tol =
            std::is_same_v<Scalar, float> ? (bf16_sketch ? 2e-2 : 5e-3) : 1e-8;
        const Eigen::VectorXd expect = theta0.cwiseProduct(g);
        const double scale = expect.norm();
        const Eigen::VectorXd got  = (V0.transpose() * Kz).template cast<double>();
        if (scale > 0.0 && (got - expect).norm() > ritz_tol * scale)
            throw std::runtime_error("expand_symmetric_eigh: locked basis fails the Ritz invariance "
                                      "check (V0' A V0 != diag(theta0)).");
    }

    // Operate on Y.leftCols(m) without calling conservativeResize(NoChange, m)
    // (avoiding Eigen's handmade_aligned_realloc 2x copy spike); the first
    // power-iteration apply() naturally replaces Y with an n x m matrix.
    for (int pi = 0; pi < power_iter; ++pi) {
        Eigen::Ref<Mat> Ym = Y.leftCols(m);
        deflate(Ym, 1);
        orthonormalize(Ym);
        Y = detail::invoke_apply(apply, Eigen::Ref<const Mat>(Y.leftCols(m)), /*exact_fp32=*/false);
    }
    {
        Eigen::Ref<Mat> Ym = Y.leftCols(m);
        deflate(Ym, 2);
        orthonormalize(Ym);
    }
    const Eigen::Ref<const Mat> Q1 = Y.leftCols(m);
    {
        const double ortho_tol = std::is_same_v<Scalar, float>
            ? 5e-3 * std::sqrt(static_cast<double>(std::max(1, m)))
            : 1e-8;
        if (static_cast<double>((V0.transpose() * Q1).norm()) > ortho_tol)
            throw std::runtime_error("expand_symmetric_eigh: fresh block is not orthogonal to the locked basis.");
    }

    // Final Rayleigh-Ritz pass in exact FP32/FP64.
    Mat KQ1 = detail::invoke_apply(apply, Q1, /*exact_fp32=*/true);  // n x m

    Eigen::MatrixXd B(k_ext, k_ext);
    #pragma omp parallel for schedule(guided) if (k0 >= 128)
    for (int j = 0; j < k0; ++j) {
        double* col = B.col(j).data();
        if (j > 0) std::memset(col, 0, static_cast<size_t>(j) * sizeof(double));
        col[j] = theta0[j];
    }
    {
        const Mat V0tKQ1 = V0.transpose() * KQ1;
        detail::assign_rect_to_double_parallel<Scalar>(V0tKQ1, B.topRightCorner(k0, m));
    }
    {
        Eigen::MatrixXd B_br;
        detail::form_upper_gram_double<Scalar>(Q1, KQ1, B_br);
        #pragma omp parallel for schedule(guided) if (m >= 128)
        for (int j = 0; j < m; ++j) {
            std::memcpy(B.col(k0 + j).data() + k0, B_br.col(j).data(),
                        static_cast<size_t>(j + 1) * sizeof(double));
        }
    }
    KQ1.resize(0, 0);

    Eigen::VectorXd w;
    Eigen::MatrixXd Z;
    eigh_upper_desc(B, w, Z);
    B.resize(0, 0);

    const Mat Z_top = detail::cast_double_block_parallel<Scalar>(Z.topLeftCorner(k0, k_target));
    const Mat Z_bot = detail::cast_double_block_parallel<Scalar>(Z.bottomLeftCorner(m, k_target));
    Z.resize(0, 0);

    EighResultT<Scalar> result;
    result.eigenvalues = w.head(k_target);
    result.eigenvectors.resize(n, k_target);
    gcta_numa_interleave(result.eigenvectors.data(),
                         static_cast<size_t>(n) * static_cast<size_t>(k_target) * sizeof(Scalar));
    result.eigenvectors.noalias()  = V0 * Z_top;
    result.eigenvectors.noalias() += Q1 * Z_bot;
    return result;
}

// Convenience one-shot entry point: sketch + power-iterate + project.
template <typename Scalar = double, typename MatVecApply>
EighResultT<Scalar> randomized_symmetric_eigh(
    MatVecApply&& apply,
    int n,
    int k_target,
    int oversample = 20,
    int power_iter = 3,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* warm_start = nullptr)
{
    const int k_ext = std::min(k_target + oversample, n - 1);
    auto [omega, Y] = build_randomized_sketch<Scalar>(apply, n, k_ext, warm_start, /*exact_fp32=*/false);
    omega.resize(0, 0);  // release omega before power iterations
    return power_iterate_and_project<Scalar>(std::forward<MatVecApply>(apply), std::move(Y), k_target, power_iter);
}

// Refine an orthonormal trial basis against the original operator (exact FP32/FP64).
template <typename Scalar = double, typename MatVecApply>
EighResultT<Scalar> rayleigh_ritz_refine(
    MatVecApply&& apply,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> basis,
    int k_target)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int k_ext = static_cast<int>(basis.cols());
    if (k_target > k_ext)
        throw std::invalid_argument("rayleigh_ritz_refine: k_target exceeds basis width.");

    Mat A_basis = detail::invoke_apply(apply, basis, /*exact_fp32=*/true);
    Eigen::MatrixXd B;
    detail::form_upper_gram_double<Scalar>(basis, A_basis, B);
    A_basis.resize(0, 0);

    Eigen::VectorXd w;
    Eigen::MatrixXd Z;
    eigh_upper_desc(B, w, Z);
    B.resize(0, 0);

    detail::rotate_basis_inplace<Scalar>(basis, Z, k_target);

    EighResultT<Scalar> result;
    result.eigenvalues  = w.head(k_target);
    result.eigenvectors = std::move(basis);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Lanczos (Spectra), for well-separated top spectra with few matvecs
// ─────────────────────────────────────────────────────────────────────────

template <typename MatVecApply>
struct SpectraMatVecOp {
    using Scalar = double;
    MatVecApply apply;
    int n;
    int rows() const { return n; }
    int cols() const { return n; }
    void perform_op(const double* x_in, double* y_out) const {
        Eigen::Map<const Eigen::VectorXd> x(x_in, n);
        Eigen::Map<Eigen::VectorXd>       y(y_out, n);
        y.noalias() = apply(x);
    }
};

template <typename MatVecApply>
EighResult lanczos_symmetric_eigh(
    MatVecApply&& apply,
    int n,
    int k_target,
    int ncv = -1)
{
    if (ncv <= 0) ncv = std::min(n, std::max(3 * k_target + 1, 30));

    SpectraMatVecOp<std::decay_t<MatVecApply>> op{std::forward<MatVecApply>(apply), n};
    Spectra::SymEigsSolver<SpectraMatVecOp<std::decay_t<MatVecApply>>> eigs(op, k_target, ncv);
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestAlge);
    if (eigs.info() != Spectra::CompInfo::Successful)
        throw std::runtime_error("lanczos_symmetric_eigh: Spectra eigensolver failed.");

    EighResult result;
    result.eigenvalues = eigs.eigenvalues();
    result.eigenvectors = eigs.eigenvectors();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Nystrom single-pass approximation (always exact FP32/FP64)
// ─────────────────────────────────────────────────────────────────────────
template <typename Scalar = double, typename MatVecApply>
EighResultT<Scalar> nystrom_symmetric_eigh(
    MatVecApply&& apply,
    int n,
    int k_target,
    int oversample = 20,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* warm_start = nullptr)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int k_ext = std::min(k_target + oversample, n - 1);

    // One-pass sketch in exact FP32/FP64: omega ~ Gaussian (or warm-started), Y = K * omega
    auto [omega, Y] = build_randomized_sketch<Scalar>(apply, n, k_ext, warm_start, /*exact_fp32=*/true);

    Eigen::MatrixXd C;
    detail::form_upper_gram_double<Scalar>(omega, Y, C);
    omega.resize(0, 0);

    Eigen::VectorXd lam_C;
    Eigen::MatrixXd V_C;
    eigh_upper_desc(C, lam_C, V_C);
    C.resize(0, 0);

    const double eps = static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    const double max_abs_lam = (lam_C.size() > 0) ? lam_C.cwiseAbs().maxCoeff() : 0.0;
    const double eps_C = std::sqrt(static_cast<double>(n)) * eps * max_abs_lam;
    const Eigen::VectorXd lam_sqrt_abs_inv = lam_C.unaryExpr(
        [eps_C](double lam) { return (std::abs(lam) > eps_C) ? 1.0 / std::sqrt(std::abs(lam)) : 0.0; });
    const Eigen::VectorXd signs = lam_C.unaryExpr(
        [eps_C](double lam) { return (std::abs(lam) > eps_C) ? ((lam > 0.0) ? 1.0 : -1.0) : 0.0; });

    Mat V_scaled(k_ext, k_ext);
    #pragma omp parallel for schedule(static) if (static_cast<size_t>(k_ext) * static_cast<size_t>(k_ext) >= 32768)
    for (int j = 0; j < k_ext; ++j) {
        const double s = lam_sqrt_abs_inv[j];
        const double* __restrict__ src = V_C.col(j).data();
        Scalar* __restrict__ dst = V_scaled.col(j).data();
        #pragma omp simd
        for (int i = 0; i < k_ext; ++i) {
            dst[i] = static_cast<Scalar>(src[i] * s);
        }
    }
    V_C.resize(0, 0);

    Mat Z = Y * V_scaled;
    Y.resize(0, 0);
    V_scaled.resize(0, 0);

    Eigen::MatrixXd R(k_ext, k_ext);
    const int info_qr = gcta_cholesky_qr_thin_QR(static_cast<gcta_blas_int>(n),
                                                  static_cast<gcta_blas_int>(k_ext),
                                                  Z.data(), static_cast<gcta_blas_int>(n),
                                                  R.data(), static_cast<gcta_blas_int>(k_ext));
    if (info_qr != 0)
        throw std::runtime_error("nystrom_symmetric_eigh: orthogonalization failed (info=" + std::to_string(info_qr) + ").");

    // T = R * diag(signs) * R^T (only upper triangle is read by eigh_upper_desc).
    // When PSD (all signs >= 0), use cblas_dsyrk (2x fewer FLOPs than dgemm).
    Eigen::MatrixXd T(k_ext, k_ext);
    if (signs.size() == 0 || signs.minCoeff() >= 0.0) {
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(k_ext) * static_cast<size_t>(k_ext) >= 32768)
        for (int j = 0; j < k_ext; ++j) {
            if (signs[j] == 0.0) {
                std::memset(R.col(j).data(), 0, static_cast<size_t>(k_ext) * sizeof(double));
            }
        }
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                    static_cast<gcta_blas_int>(k_ext), static_cast<gcta_blas_int>(k_ext),
                    1.0, R.data(), static_cast<gcta_blas_int>(k_ext),
                    0.0, T.data(), static_cast<gcta_blas_int>(k_ext));
    } else {
        T.noalias() = R * signs.asDiagonal() * R.transpose();
    }
    R.resize(0, 0);

    Eigen::VectorXd w_T;
    Eigen::MatrixXd U_T;
    eigh_upper_desc(T, w_T, U_T);
    T.resize(0, 0);

    detail::rotate_basis_inplace<Scalar>(Z, U_T, k_target);

    EighResultT<Scalar> result;
    result.eigenvalues  = w_T.head(k_target);
    result.eigenvectors = std::move(Z);
    return result;
}

} // namespace gcta_eigh
