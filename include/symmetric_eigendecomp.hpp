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
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
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

// Parallel standard-normal fill directly into column-major BF16 storage (uint16_t)
// using a tiny L1-resident float32 chunk buffer, avoiding a full float32 omega matrix.
inline void fill_standard_normal_bf16(uint16_t* dst, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || dst == nullptr) return;
    const int num_threads = std::min(omp_get_max_threads(), cols);
    std::vector<std::uint32_t> thread_seeds(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_seeds[t] = static_cast<std::uint32_t>(shared_rng()());
    }
    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        std::mt19937 local_rng(thread_seeds[tid]);
        std::normal_distribution<double> normal(0.0, 1.0);
        const int cols_per_thread = (cols + num_threads - 1) / num_threads;
        const int col_begin = std::min(cols, tid * cols_per_thread);
        const int col_end = std::min(cols, col_begin + cols_per_thread);
        constexpr int kBuf = 256;
        alignas(64) float fbuf[kBuf];
        for (int col = col_begin; col < col_end; ++col) {
            uint16_t* __restrict__ col_dst = dst + static_cast<size_t>(col) * static_cast<size_t>(rows);
            for (int r0 = 0; r0 < rows; r0 += kBuf) {
                const int cnt = std::min(kBuf, rows - r0);
                for (int i = 0; i < cnt; ++i) {
                    fbuf[i] = static_cast<float>(normal(local_rng));
                }
                gcta_f32_to_bf16_rne(fbuf, col_dst + r0, static_cast<size_t>(cnt));
            }
        }
    }
}

struct RsvdOptions {
    int qr_power_passes = 0;       // 0 = auto (1 when BF16 sketch active, else 2)
    bool eigh_fp32 = false;        // true = ssyevd on float32 upper Gram instead of dsyevd
    int eigh_threads = 0;          // 0 = all threads; >0 caps LAPACK/BLAS threads during eigh
    bool keep_trailing_cols = false; // true = skip Q.conservativeResize(NoChange, k_target)
};

struct RsvdProfileStats {
    double sketch_fill_s = 0.0;
    double sketch_matvec_s = 0.0;
    double power_qr_s = 0.0;
    double power_matvec_s = 0.0;
    double final_qr_s = 0.0;
    double rr_matvec_s = 0.0;
    double gram_s = 0.0;
    double eigh_s = 0.0;
    double rotate_s = 0.0;
    double matvec_unpack_s = 0.0;
    double matvec_gemm_s = 0.0;
    uint64_t matvec_tiles = 0;
    uint64_t matvec_gemm_calls = 0;
    int qr_calls = 0;
    int qr_passes = 0;
    int qr_shifted = 0;
    int qr_householder = 0;
};

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
// When Scalar == float and k >= 512, executes upper-triangular blocked cblas_sgemm
// (~56% of full FLOPs) directly into L3-resident 25 MiB scratch, eliminating the
// full 1.62 GB B_f allocation and lower-triangle compute.
template <typename Scalar>
inline void form_upper_gram_double(
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& A,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& C,
    Eigen::MatrixXd& B)
{
    const Eigen::Index n = A.rows();
    const Eigen::Index k = A.cols();
    B.resize(k, k);
    if constexpr (std::is_same_v<Scalar, double>) {
        B.noalias() = A.transpose() * C;
    } else if (k < 512) {
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
    } else {
        constexpr Eigen::Index kBlk = 2560;
        const Eigen::Index max_b = std::min(k, kBlk);
        AlignedRawBuffer<float, 64> tile_f(static_cast<size_t>(max_b) * static_cast<size_t>(max_b));
        const gcta_blas_int ldA = static_cast<gcta_blas_int>(A.outerStride());
        const gcta_blas_int ldC = static_cast<gcta_blas_int>(C.outerStride());
        const gcta_blas_int blas_n = static_cast<gcta_blas_int>(n);
        const float* const A_ptr = A.data();
        const float* const C_ptr = C.data();
        const Eigen::Index nb = (k + kBlk - 1) / kBlk;
        for (Eigen::Index jb = 0; jb < nb; ++jb) {
            const Eigen::Index j0 = jb * kBlk;
            const Eigen::Index jlen = std::min(kBlk, k - j0);
            for (Eigen::Index ib = 0; ib <= jb; ++ib) {
                const Eigen::Index i0 = ib * kBlk;
                const Eigen::Index ilen = std::min(kBlk, k - i0);
                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            static_cast<gcta_blas_int>(ilen),
                            static_cast<gcta_blas_int>(jlen),
                            blas_n,
                            1.0f, A_ptr + static_cast<size_t>(i0) * static_cast<size_t>(ldA), ldA,
                            C_ptr + static_cast<size_t>(j0) * static_cast<size_t>(ldC), ldC,
                            0.0f, tile_f.data(), static_cast<gcta_blas_int>(ilen));
                if (ib < jb) {
                    #pragma omp parallel for schedule(static) if (static_cast<size_t>(ilen) * static_cast<size_t>(jlen) >= 16384)
                    for (Eigen::Index dc = 0; dc < jlen; ++dc) {
                        const float* __restrict__ src = tile_f.data() + static_cast<size_t>(dc) * static_cast<size_t>(ilen);
                        double* __restrict__ dst = B.col(j0 + dc).data() + i0;
                        #pragma omp simd
                        for (Eigen::Index dr = 0; dr < ilen; ++dr) {
                            dst[dr] = static_cast<double>(src[dr]);
                        }
                    }
                } else {
                    #pragma omp parallel for schedule(guided) if (jlen >= 128)
                    for (Eigen::Index dc = 0; dc < jlen; ++dc) {
                        const float* __restrict__ src = tile_f.data() + static_cast<size_t>(dc) * static_cast<size_t>(ilen);
                        double* __restrict__ dst = B.col(j0 + dc).data() + i0;
                        #pragma omp simd
                        for (Eigen::Index dr = 0; dr <= dc; ++dr) {
                            dst[dr] = static_cast<double>(src[dr]);
                        }
                    }
                }
            }
        }
    }
}

// Upper-triangular blocked float32 Gram: B_f (k x k, UPPER triangle valid) = A^T * C.
inline void form_upper_gram_float(
    const Eigen::Ref<const Eigen::MatrixXf>& A,
    const Eigen::Ref<const Eigen::MatrixXf>& C,
    Eigen::MatrixXf& B_f)
{
    const Eigen::Index n = A.rows();
    const Eigen::Index k = A.cols();
    B_f.resize(k, k);
    if (k < 512) {
        B_f.noalias() = A.transpose() * C;
        return;
    }
    constexpr Eigen::Index kBlk = 2560;
    const gcta_blas_int ldA = static_cast<gcta_blas_int>(A.outerStride());
    const gcta_blas_int ldC = static_cast<gcta_blas_int>(C.outerStride());
    const gcta_blas_int ldB = static_cast<gcta_blas_int>(k);
    const gcta_blas_int blas_n = static_cast<gcta_blas_int>(n);
    const float* const A_ptr = A.data();
    const float* const C_ptr = C.data();
    float* const B_ptr = B_f.data();
    const Eigen::Index nb = (k + kBlk - 1) / kBlk;
    for (Eigen::Index jb = 0; jb < nb; ++jb) {
        const Eigen::Index j0 = jb * kBlk;
        const Eigen::Index jlen = std::min(kBlk, k - j0);
        for (Eigen::Index ib = 0; ib <= jb; ++ib) {
            const Eigen::Index i0 = ib * kBlk;
            const Eigen::Index ilen = std::min(kBlk, k - i0);
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        static_cast<gcta_blas_int>(ilen),
                        static_cast<gcta_blas_int>(jlen),
                        blas_n,
                        1.0f, A_ptr + static_cast<size_t>(i0) * static_cast<size_t>(ldA), ldA,
                        C_ptr + static_cast<size_t>(j0) * static_cast<size_t>(ldC), ldC,
                        0.0f, B_ptr + i0 + static_cast<size_t>(j0) * static_cast<size_t>(ldB), ldB);
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

// Parallel cast/copy (and optional column reversal for ascending eigh output)
// of a dense eigenvector block into Scalar storage.
template <typename Scalar, typename ZMat>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
cast_or_reverse_block_parallel(const ZMat& Z, int k_cols, bool reverse_cols)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int rows = static_cast<int>(Z.rows());
    const int total_cols = static_cast<int>(Z.cols());
    Mat out(rows, k_cols);
    using ZScalar = typename ZMat::Scalar;
    if constexpr (std::is_same_v<Scalar, ZScalar>) {
        if (!reverse_cols) {
            out.noalias() = Z.leftCols(k_cols);
            return out;
        }
    }
    #pragma omp parallel for schedule(static) if (static_cast<size_t>(rows) * static_cast<size_t>(k_cols) >= 32768)
    for (int j = 0; j < k_cols; ++j) {
        const int src_j = reverse_cols ? (total_cols - 1 - j) : j;
        const ZScalar* __restrict__ s = Z.col(src_j).data();
        Scalar* __restrict__ d = out.col(j).data();
        #pragma omp simd
        for (int i = 0; i < rows; ++i) {
            d[i] = static_cast<Scalar>(s[i]);
        }
    }
    return out;
}

// In-place basis rotation: Q <- Q * (selected/reversed columns of Z) in 4096-row slabs.
// Frees Z immediately after casting its leading k_target columns to Scalar,
// and optionally skips conservativeResize when callers slice .leftCols(k_target).
template <typename Scalar, typename ZMat>
inline void rotate_basis_inplace(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
    ZMat& Z,
    int k_target,
    bool reverse_Z_cols = false,
    bool keep_trailing_cols = false)
{
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int n = static_cast<int>(Q.rows());
    const int k_ext = static_cast<int>(Q.cols());
    if (n == 0 || k_target == 0) {
        Q.resize(n, k_target);
        Z.resize(0, 0);
        return;
    }

    Mat Z_s = cast_or_reverse_block_parallel<Scalar>(Z, k_target, reverse_Z_cols);
    Z.resize(0, 0);

    if (static_cast<size_t>(n) * static_cast<size_t>(k_target) * sizeof(Scalar) <= (64ULL << 20)) {
        Mat Q_new(n, k_target);
        const gcta_blas_int ldQ = static_cast<gcta_blas_int>(n);
        const gcta_blas_int ldZ = static_cast<gcta_blas_int>(k_ext);
        if constexpr (std::is_same_v<Scalar, float>) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<gcta_blas_int>(n),
                        static_cast<gcta_blas_int>(k_target),
                        static_cast<gcta_blas_int>(k_ext),
                        1.0f, Q.data(), ldQ,
                        Z_s.data(), ldZ,
                        0.0f, Q_new.data(), ldQ);
        } else {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<gcta_blas_int>(n),
                        static_cast<gcta_blas_int>(k_target),
                        static_cast<gcta_blas_int>(k_ext),
                        1.0, Q.data(), ldQ,
                        Z_s.data(), ldZ,
                        0.0, Q_new.data(), ldQ);
        }
        Q.swap(Q_new);
        return;
    }

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

    if (!keep_trailing_cols && k_target < k_ext) {
        Q.conservativeResize(Eigen::NoChange, k_target);
    }
}

inline void eigh_upper_asc_d(Eigen::MatrixXd& B, Eigen::VectorXd& w, Eigen::MatrixXd& Z,
                             int eigh_threads = 0)
{
    const int k = static_cast<int>(B.rows());
    if (B.cols() != k || k < 1)
        throw std::invalid_argument("eigh_upper_asc_d: B must be square and non-empty.");
    w.resize(k);
    ScopedBlasThreadCap cap(eigh_threads);

    bool solved = false;
    if (k < 32766) {
        Z.resize(0, 0);
        const int info_d = gcta_dsyevd(static_cast<gcta_blas_int>(k), B.data(),
                                       static_cast<gcta_blas_int>(k), w.data());
        if (info_d == 0) {
            Z.swap(B);
            solved = true;
        } else if (info_d != -1) {
            throw std::runtime_error("eigh_upper_asc_d: dsyevd failed (info=" + std::to_string(info_d) + ").");
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
            throw std::runtime_error("eigh_upper_asc_d: dsyevr failed (info=" + std::to_string(info) + ").");
    }
}

inline void eigh_upper_asc_f(Eigen::MatrixXf& B_f, Eigen::VectorXd& w, Eigen::MatrixXf& Z_f,
                             int eigh_threads = 0)
{
    const int k = static_cast<int>(B_f.rows());
    if (B_f.cols() != k || k < 1)
        throw std::invalid_argument("eigh_upper_asc_f: B_f must be square and non-empty.");
    Eigen::VectorXf w_f(k);
    ScopedBlasThreadCap cap(eigh_threads);

    bool solved = false;
    if (k < 32766) {
        Z_f.resize(0, 0);
        const int info_d = gcta_ssyevd(static_cast<gcta_blas_int>(k), B_f.data(),
                                       static_cast<gcta_blas_int>(k), w_f.data());
        if (info_d == 0) {
            Z_f.swap(B_f);
            solved = true;
        } else if (info_d != -1) {
            throw std::runtime_error("eigh_upper_asc_f: ssyevd failed (info=" + std::to_string(info_d) + ").");
        }
    }

    if (!solved) {
        Z_f.resize(k, k);
        std::vector<gcta_blas_int> isuppz(2 * static_cast<size_t>(k));
        gcta_blas_int m_found = 0;
        const int info = gcta_ssyevr(static_cast<gcta_blas_int>(k), B_f.data(), static_cast<gcta_blas_int>(k),
                                     static_cast<gcta_blas_int>(1), static_cast<gcta_blas_int>(k), &m_found,
                                     w_f.data(), Z_f.data(), static_cast<gcta_blas_int>(k), isuppz.data());
        if (info != 0 || m_found != k)
            throw std::runtime_error("eigh_upper_asc_f: ssyevr failed (info=" + std::to_string(info) + ").");
    }
    w = w_f.cast<double>();
}

} // namespace detail

// Full eigendecomposition of the symmetric k x k matrix whose UPPER triangle is
// stored in B. B is destroyed. On return w (k) and Z (k x k) hold eigenpairs in
// DESCENDING order.
inline void eigh_upper_desc(Eigen::MatrixXd& B, Eigen::VectorXd& w, Eigen::MatrixXd& Z,
                            int eigh_threads = 0)
{
    detail::eigh_upper_asc_d(B, w, Z, eigh_threads);
    const int k = static_cast<int>(w.size());

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

// Extended power iteration + Rayleigh-Ritz with persistent QR workspaces,
// configurable power-iteration QR passes, ssyevd/thread-cap eigensolve,
// fused eigenvector reverse, optional zero-copy Q retention, and post-RR callback.
template <typename Scalar = double, typename MatVecApply>
EighResultT<Scalar> power_iterate_and_project_ex(
    MatVecApply&& apply,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y,
    int k_target,
    int power_iter = 3,
    const RsvdOptions& opts = {},
    RsvdProfileStats* prof = nullptr,
    const std::function<void()>& on_after_rr_matvec = {})
{
    using Clock = std::chrono::steady_clock;
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int n     = static_cast<int>(Y.rows());
    const int k_ext = static_cast<int>(Y.cols());
    if (k_target > k_ext)
        throw std::invalid_argument("power_iterate_and_project: k_target exceeds sketch width k_ext.");

    Eigen::MatrixXd qr_G;
    Eigen::MatrixXf qr_Gf;
    CholeskyQrStats qr_stats;
    constexpr bool has_prec = detail::supports_precision_flag<MatVecApply, Mat>();
    constexpr bool bf16_default =
        std::is_same_v<Scalar, float> && has_prec && (GCTA_HAS_AOCL_BF16_GEMM != 0);
    const int power_qr_passes = (opts.qr_power_passes > 0)
        ? opts.qr_power_passes
        : (bf16_default ? 1 : 2);

    for (int pi = 0; pi < power_iter; ++pi) {
        const auto t_qr0 = prof ? Clock::now() : Clock::time_point{};
        int info = 0;
        if constexpr (std::is_same_v<Scalar, float>) {
            info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                           static_cast<gcta_blas_int>(k_ext),
                                           Y.data(), static_cast<gcta_blas_int>(n),
                                           &qr_G, &qr_Gf, power_qr_passes, &qr_stats);
        } else {
            info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                           static_cast<gcta_blas_int>(k_ext),
                                           Y.data(), static_cast<gcta_blas_int>(n),
                                           &qr_G);
        }
        if (prof) {
            prof->power_qr_s += std::chrono::duration<double>(Clock::now() - t_qr0).count();
        }
        if (info != 0)
            throw std::runtime_error("power_iterate_and_project: orthogonalization failed (info=" +
                                      std::to_string(info) + ").");
        const auto t_mv0 = prof ? Clock::now() : Clock::time_point{};
        Y = detail::invoke_apply(apply, Y, /*exact_fp32=*/false);
        if (prof) {
            prof->power_matvec_s += std::chrono::duration<double>(Clock::now() - t_mv0).count();
        }
    }

    {
        const auto t_qr0 = prof ? Clock::now() : Clock::time_point{};
        int info = 0;
        if constexpr (std::is_same_v<Scalar, float>) {
            info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                           static_cast<gcta_blas_int>(k_ext),
                                           Y.data(), static_cast<gcta_blas_int>(n),
                                           &qr_G, &qr_Gf, 2, &qr_stats);
        } else {
            info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                           static_cast<gcta_blas_int>(k_ext),
                                           Y.data(), static_cast<gcta_blas_int>(n),
                                           &qr_G);
        }
        if (prof) {
            prof->final_qr_s += std::chrono::duration<double>(Clock::now() - t_qr0).count();
        }
        if (info != 0)
            throw std::runtime_error("power_iterate_and_project: orthogonalization failed (info=" +
                                      std::to_string(info) + ").");
    }
    qr_G.resize(0, 0);
    qr_Gf.resize(0, 0);
    Mat Q = std::move(Y);

    // Final Rayleigh-Ritz pass MUST run in exact FP32/FP64.
    const auto t_rr0 = prof ? Clock::now() : Clock::time_point{};
    Mat AQ = detail::invoke_apply(apply, Q, /*exact_fp32=*/true);
    if (prof) {
        prof->rr_matvec_s += std::chrono::duration<double>(Clock::now() - t_rr0).count();
    }
    if (on_after_rr_matvec) {
        on_after_rr_matvec();
    }

    Eigen::VectorXd w;
    const auto t_g0 = prof ? Clock::now() : Clock::time_point{};
    if constexpr (std::is_same_v<Scalar, float>) {
        if (opts.eigh_fp32) {
            Eigen::MatrixXf B_f;
            detail::form_upper_gram_float(Q, AQ, B_f);
            AQ.resize(0, 0);
            if (prof) {
                prof->gram_s += std::chrono::duration<double>(Clock::now() - t_g0).count();
            }
            const auto t_e0 = prof ? Clock::now() : Clock::time_point{};
            Eigen::MatrixXf Z_f;
            detail::eigh_upper_asc_f(B_f, w, Z_f, opts.eigh_threads);
            B_f.resize(0, 0);
            std::reverse(w.data(), w.data() + w.size());
            if (prof) {
                prof->eigh_s += std::chrono::duration<double>(Clock::now() - t_e0).count();
            }
            const auto t_rot0 = prof ? Clock::now() : Clock::time_point{};
            detail::rotate_basis_inplace<Scalar>(
                Q, Z_f, k_target, /*reverse_Z_cols=*/true, opts.keep_trailing_cols);
            if (prof) {
                prof->rotate_s += std::chrono::duration<double>(Clock::now() - t_rot0).count();
            }
        } else {
            Eigen::MatrixXd B;
            detail::form_upper_gram_double<Scalar>(Q, AQ, B);
            AQ.resize(0, 0);
            if (prof) {
                prof->gram_s += std::chrono::duration<double>(Clock::now() - t_g0).count();
            }
            const auto t_e0 = prof ? Clock::now() : Clock::time_point{};
            Eigen::MatrixXd Z;
            detail::eigh_upper_asc_d(B, w, Z, opts.eigh_threads);
            B.resize(0, 0);
            std::reverse(w.data(), w.data() + w.size());
            if (prof) {
                prof->eigh_s += std::chrono::duration<double>(Clock::now() - t_e0).count();
            }
            const auto t_rot0 = prof ? Clock::now() : Clock::time_point{};
            detail::rotate_basis_inplace<Scalar>(
                Q, Z, k_target, /*reverse_Z_cols=*/true, opts.keep_trailing_cols);
            if (prof) {
                prof->rotate_s += std::chrono::duration<double>(Clock::now() - t_rot0).count();
            }
        }
    } else {
        Eigen::MatrixXd B;
        detail::form_upper_gram_double<Scalar>(Q, AQ, B);
        AQ.resize(0, 0);
        if (prof) {
            prof->gram_s += std::chrono::duration<double>(Clock::now() - t_g0).count();
        }
        const auto t_e0 = prof ? Clock::now() : Clock::time_point{};
        Eigen::MatrixXd Z;
        detail::eigh_upper_asc_d(B, w, Z, opts.eigh_threads);
        B.resize(0, 0);
        std::reverse(w.data(), w.data() + w.size());
        if (prof) {
            prof->eigh_s += std::chrono::duration<double>(Clock::now() - t_e0).count();
        }
        const auto t_rot0 = prof ? Clock::now() : Clock::time_point{};
        detail::rotate_basis_inplace<Scalar>(
            Q, Z, k_target, /*reverse_Z_cols=*/true, opts.keep_trailing_cols);
        if (prof) {
            prof->rotate_s += std::chrono::duration<double>(Clock::now() - t_rot0).count();
        }
    }

    if (prof) {
        prof->qr_calls += qr_stats.calls;
        prof->qr_passes += qr_stats.passes;
        prof->qr_shifted += qr_stats.shifted;
        prof->qr_householder += qr_stats.householder;
    }

    EighResultT<Scalar> result;
    result.eigenvalues  = w.head(k_target);
    result.eigenvectors = std::move(Q);
    return result;
}

template <typename Scalar = double, typename MatVecApply>
EighResultT<Scalar> power_iterate_and_project(
    MatVecApply&& apply,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y,
    int k_target,
    int power_iter = 3)
{
    return power_iterate_and_project_ex<Scalar>(
        std::forward<MatVecApply>(apply), std::move(Y), k_target, power_iter);
}

// Persistent-buffer in-place BF16 rSVD for PackedGrmFp16:
// Allocates X_bf16 (2*n*k_ext) and Y (4*n*k_ext) ONCE across sketch and all
// power iterations (eliminating 12 GB omega + 60 GB Y/X_bf16 page-fault churn).
template <typename ApplyBf16Inplace, typename ApplyFp32>
EighResultF randomized_symmetric_eigh_bf16_inplace(
    ApplyBf16Inplace&& apply_bf16_inplace,
    ApplyFp32&& apply_fp32,
    int n,
    int k_target,
    int oversample,
    int power_iter,
    Eigen::MatrixXf* warm_start,
    const RsvdOptions& opts = {},
    RsvdProfileStats* prof = nullptr,
    const std::function<void()>& on_after_rr_matvec = {})
{
    using Clock = std::chrono::steady_clock;
    const int k_ext = std::min(k_target + oversample, n - 1);

    AlignedRawBuffer<uint16_t, 64> X_bf16(static_cast<size_t>(n) * static_cast<size_t>(k_ext));
    const auto t_fill0 = prof ? Clock::now() : Clock::time_point{};
    if (warm_start && warm_start->rows() == n && warm_start->cols() > 0) {
        const int k_copy = std::min(static_cast<int>(warm_start->cols()), k_ext);
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(k_copy) >= 32768)
        for (int j = 0; j < k_copy; ++j) {
            gcta_f32_to_bf16_rne(
                warm_start->col(j).data(),
                X_bf16.data() + static_cast<size_t>(j) * static_cast<size_t>(n),
                static_cast<size_t>(n));
        }
        warm_start->resize(0, 0);
        if (k_copy < k_ext) {
            fill_standard_normal_bf16(
                X_bf16.data() + static_cast<size_t>(k_copy) * static_cast<size_t>(n),
                n, k_ext - k_copy);
        }
    } else {
        if (warm_start) warm_start->resize(0, 0);
        fill_standard_normal_bf16(X_bf16.data(), n, k_ext);
    }
    if (prof) {
        prof->sketch_fill_s += std::chrono::duration<double>(Clock::now() - t_fill0).count();
    }

    Eigen::MatrixXf Y(n, k_ext);
    const auto t_sm0 = prof ? Clock::now() : Clock::time_point{};
    apply_bf16_inplace(X_bf16.data(), Y);
    if (prof) {
        prof->sketch_matvec_s += std::chrono::duration<double>(Clock::now() - t_sm0).count();
    }

    Eigen::MatrixXd qr_G;
    Eigen::MatrixXf qr_Gf;
    CholeskyQrStats qr_stats;
    const int power_qr_passes = (opts.qr_power_passes > 0) ? opts.qr_power_passes : 1;

    for (int pi = 0; pi < power_iter; ++pi) {
        const auto t_qr0 = prof ? Clock::now() : Clock::time_point{};
        const int info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                                 static_cast<gcta_blas_int>(k_ext),
                                                 Y.data(), static_cast<gcta_blas_int>(n),
                                                 &qr_G, &qr_Gf, power_qr_passes, &qr_stats);
        if (info != 0)
            throw std::runtime_error("randomized_symmetric_eigh_bf16_inplace: QR failed (info=" +
                                      std::to_string(info) + ").");
        // Parallel contiguous F32 -> BF16 conversion into persistent X_bf16.
        {
            const size_t total = static_cast<size_t>(n) * static_cast<size_t>(k_ext);
            const float* const src = Y.data();
            uint16_t* const dst = X_bf16.data();
            #pragma omp parallel if (total >= 32768)
            {
                const int tid = omp_get_thread_num();
                const int nt = omp_get_num_threads();
                const size_t lo = (total * static_cast<size_t>(tid)) / static_cast<size_t>(nt);
                const size_t hi = (total * static_cast<size_t>(tid + 1)) / static_cast<size_t>(nt);
                if (hi > lo) gcta_f32_to_bf16_rne(src + lo, dst + lo, hi - lo);
            }
        }
        if (prof) {
            prof->power_qr_s += std::chrono::duration<double>(Clock::now() - t_qr0).count();
        }
        const auto t_pm0 = prof ? Clock::now() : Clock::time_point{};
        apply_bf16_inplace(X_bf16.data(), Y);
        if (prof) {
            prof->power_matvec_s += std::chrono::duration<double>(Clock::now() - t_pm0).count();
        }
    }

    X_bf16.clear();

    {
        const auto t_fqr0 = prof ? Clock::now() : Clock::time_point{};
        const int info = gcta_cholesky_qr_thin_Q(static_cast<gcta_blas_int>(n),
                                                 static_cast<gcta_blas_int>(k_ext),
                                                 Y.data(), static_cast<gcta_blas_int>(n),
                                                 &qr_G, &qr_Gf, 2, &qr_stats);
        if (prof) {
            prof->final_qr_s += std::chrono::duration<double>(Clock::now() - t_fqr0).count();
        }
        if (info != 0)
            throw std::runtime_error("randomized_symmetric_eigh_bf16_inplace: final QR failed (info=" +
                                      std::to_string(info) + ").");
    }
    qr_G.resize(0, 0);
    qr_Gf.resize(0, 0);
    Eigen::MatrixXf Q = std::move(Y);

    const auto t_rr0 = prof ? Clock::now() : Clock::time_point{};
    Eigen::MatrixXf AQ = apply_fp32(Q);
    if (prof) {
        prof->rr_matvec_s += std::chrono::duration<double>(Clock::now() - t_rr0).count();
    }
    if (on_after_rr_matvec) {
        on_after_rr_matvec();
    }

    Eigen::VectorXd w;
    const auto t_g0 = prof ? Clock::now() : Clock::time_point{};
    if (opts.eigh_fp32) {
        Eigen::MatrixXf B_f;
        detail::form_upper_gram_float(Q, AQ, B_f);
        AQ.resize(0, 0);
        if (prof) {
            prof->gram_s += std::chrono::duration<double>(Clock::now() - t_g0).count();
        }
        const auto t_e0 = prof ? Clock::now() : Clock::time_point{};
        Eigen::MatrixXf Z_f;
        detail::eigh_upper_asc_f(B_f, w, Z_f, opts.eigh_threads);
        B_f.resize(0, 0);
        std::reverse(w.data(), w.data() + w.size());
        if (prof) {
            prof->eigh_s += std::chrono::duration<double>(Clock::now() - t_e0).count();
        }
        const auto t_rot0 = prof ? Clock::now() : Clock::time_point{};
        detail::rotate_basis_inplace<float>(
            Q, Z_f, k_target, /*reverse_Z_cols=*/true, opts.keep_trailing_cols);
        if (prof) {
            prof->rotate_s += std::chrono::duration<double>(Clock::now() - t_rot0).count();
        }
    } else {
        Eigen::MatrixXd B;
        detail::form_upper_gram_double<float>(Q, AQ, B);
        AQ.resize(0, 0);
        if (prof) {
            prof->gram_s += std::chrono::duration<double>(Clock::now() - t_g0).count();
        }
        const auto t_e0 = prof ? Clock::now() : Clock::time_point{};
        Eigen::MatrixXd Z;
        detail::eigh_upper_asc_d(B, w, Z, opts.eigh_threads);
        B.resize(0, 0);
        std::reverse(w.data(), w.data() + w.size());
        if (prof) {
            prof->eigh_s += std::chrono::duration<double>(Clock::now() - t_e0).count();
        }
        const auto t_rot0 = prof ? Clock::now() : Clock::time_point{};
        detail::rotate_basis_inplace<float>(
            Q, Z, k_target, /*reverse_Z_cols=*/true, opts.keep_trailing_cols);
        if (prof) {
            prof->rotate_s += std::chrono::duration<double>(Clock::now() - t_rot0).count();
        }
    }

    if (prof) {
        prof->qr_calls += qr_stats.calls;
        prof->qr_passes += qr_stats.passes;
        prof->qr_shifted += qr_stats.shifted;
        prof->qr_householder += qr_stats.householder;
    }

    EighResultF result;
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
        if (static_cast<size_t>(k0) * static_cast<size_t>(m) > 262144) {
            constexpr int kOrthoProbes = 16;
            Mat G_probe(m, kOrthoProbes);
            fill_standard_normal(G_probe);
            const Mat Zp = Q1 * G_probe;
            const Mat R = V0.transpose() * Zp;
            const double est_frob =
                static_cast<double>(R.norm()) / std::sqrt(static_cast<double>(kOrthoProbes));
            if (est_frob > ortho_tol * 2.0)
                throw std::runtime_error("expand_symmetric_eigh: fresh block is not orthogonal to the locked basis.");
        } else if (static_cast<double>((V0.transpose() * Q1).norm()) > ortho_tol) {
            throw std::runtime_error("expand_symmetric_eigh: fresh block is not orthogonal to the locked basis.");
        }
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
