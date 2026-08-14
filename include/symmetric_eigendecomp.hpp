/*
 * symmetric_eigendecomp.hpp
 *
 * Approximate top-k eigendecomposition methods for a symmetric (typically
 * PSD) matrix A, shared between grm.cpp (gcta::pca) and RemlEngine.cpp
 * (compute_woodbury_basis).
 *
 * Every entry point takes a matvec functor rather than a concrete matrix
 * type: `apply(X)` must return A * X for X either an n-vector or an n x m
 * block. This lets callers pass a plain MatrixXd, a selfadjointView<Upper>
 * or selfadjointView<Lower> (grm.cpp's GRM is fully populated; RemlEngine's
 * ctx.A GRM only has the lower triangle valid), or in future a tiled/
 * streaming matvec, without this header caring.
 *
 * Methods:
 *   - randomized_symmetric_eigh  : Halko/Martinsson/Tropp randomized range
 *                                  finder + power iteration + Rayleigh-Ritz
 *                                  projection. Good default; k+oversample
 *                                  dense matvecs dominate cost.
 *   - lanczos_symmetric_eigh     : Spectra Lanczos (implicitly restarted
 *                                  Arnoldi on a symmetric operator), useful
 *                                  when the spectrum is well separated near
 *                                  the top and few matvecs are wanted.
 *   - tall_skinny_thin_svd       : QR-then-small-SVD reduction, used to
 *                                  extract the thin SVD of an n x k sketch
 *                                  (e.g. the Nystrom sketch) without paying
 *                                  for a bidiagonal divide-and-conquer SVD
 *                                  at n scale.
 */
#pragma once

#include <Eigen/Dense>
#include <Spectra/SymEigsSolver.h>
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include "cpu.h"  // gcta_dsyevd

namespace gcta_eigh {

inline Eigen::VectorXd threaded_dense_matvec(
    const Eigen::MatrixXd& A,
    const Eigen::Ref<const Eigen::VectorXd>& x, 
    int col_block = 64)
{
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n || x.size() != n)
        throw std::invalid_argument("threaded_dense_matvec: dimension mismatch.");
    return A * x;
}

struct EighResult {
    Eigen::VectorXd eigenvalues;   // descending, size k_target
    Eigen::MatrixXd eigenvectors;  // n x k_target, columns match eigenvalues
};

inline std::mt19937& shared_rng() {
    static std::mt19937 rng([] {
        std::random_device entropy;
        return entropy();
    }());
    return rng;
}

inline void fill_standard_normal(Eigen::Ref<Eigen::MatrixXd> matrix) {
    std::normal_distribution<double> normal(0.0, 1.0);
    for (Eigen::Index col = 0; col < matrix.cols(); ++col)
        for (Eigen::Index row = 0; row < matrix.rows(); ++row)
            matrix(row, col) = normal(shared_rng());
}

// Oversampling doesn't need to grow with k_target for generic top-k subspace
// accuracy (HMT's error bound doesn't require it). But callers that use
// eigenvalues near the *tail* of a large k_target block to locate a spectral
// threshold (e.g. Woodbury's auto-k Marchenko-Pastur edge, or its EIG99 mass
// target) are relying on exactly the estimates that are least accurate —
// Rayleigh-Ritz quality degrades toward the edge of the requested block, and
// GRM eigenvalues cluster tightly there by construction (that's the point of
// the MP bulk edge). A fixed p=20 is a 3x buffer at k=6 (PCA; top eigenvalues
// are well-separated population-structure signal, not the bottleneck) but a
// 0.4% buffer at k=5000. Scale with k_target instead; the extra sketch
// columns are cheap once k_ext is already in the thousands.
inline int recommended_oversample(int k_target, int floor = 20, int cap = 200) {
    return std::clamp(k_target / 10, floor, cap);
}

// ─────────────────────────────────────────────────────────────────────────
// Randomized range finder + power iteration + Rayleigh-Ritz (rSVD)
// ─────────────────────────────────────────────────────────────────────────

// Build the initial randomized sketch Y = A * omega (n x k_ext).
// If `warm_start` is supplied (e.g. a previous Uk basis, or a PCA .eigenvec
// basis for the same GRM), its leading columns seed omega instead of a
// fresh Gaussian draw, which typically lets power_iterate_and_project()
// converge in fewer iterations.
template <typename MatVecApply>
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_randomized_sketch(
    MatVecApply&& apply,
    int n,
    int k_ext,
    const Eigen::MatrixXd* warm_start = nullptr)
{
    Eigen::MatrixXd omega;
    if (warm_start && warm_start->rows() == n && warm_start->cols() > 0) {
        omega.resize(n, k_ext);
        const int k_copy = std::min(static_cast<int>(warm_start->cols()), k_ext);
        omega.leftCols(k_copy) = warm_start->leftCols(k_copy);
        if (k_copy < k_ext)
            fill_standard_normal(omega.rightCols(k_ext - k_copy));
    } else {
        omega.resize(n, k_ext);
        fill_standard_normal(omega);
    }
    Eigen::MatrixXd Y = apply(omega);
    return {std::move(omega), std::move(Y)};
}

// Core duplicated logic: given an initial sketch Y = A * omega (n x k_ext),
// re-orthogonalize/re-multiply for `power_iter` passes, then do a final QR
// and a Rayleigh-Ritz projection onto the k_ext x k_ext subspace to recover
// the k_target dominant eigenpairs of A.
//
// `apply` is called power_iter + 1 more times. Y is consumed/overwritten.
template <typename MatVecApply>
EighResult power_iterate_and_project(
    MatVecApply&& apply,
    Eigen::MatrixXd Y,
    int k_target,
    int power_iter = 3)
{
    const int n     = static_cast<int>(Y.rows());
    const int k_ext = static_cast<int>(Y.cols());
    if (k_target > k_ext)
        throw std::invalid_argument("power_iterate_and_project: k_target exceeds sketch width k_ext.");

    // Pre-allocated thin-Q scratch, reused across every QR in this call so
    // each pass avoids a fresh n x k_ext zero-init allocation (this matters:
    // at n=500k, k_ext~1200, each Identity(n,k_ext) is ~4.9 GB).
    Eigen::MatrixXd qr_scratch = Eigen::MatrixXd::Identity(n, k_ext);

    for (int pi = 0; pi < power_iter; ++pi) {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
        Y.resize(0, 0);  // qr(Y)'s constructor already copied what it needs internally
        qr_scratch.setIdentity();
        Y = apply(qr.householderQ() * qr_scratch);
    }

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
    Y.resize(0, 0);  // same: redundant with qr's internal copy from here on
    qr_scratch.setIdentity();
    Eigen::MatrixXd Q = qr.householderQ() * qr_scratch;

    Eigen::MatrixXd AQ = apply(Q);
    Eigen::MatrixXd B  = Q.transpose() * AQ;   // k_ext x k_ext, symmetric
    AQ.resize(0, 0);

    Eigen::VectorXd w(k_ext);
    const int info = gcta_dsyevd((gcta_blas_int)k_ext, B.data(), (gcta_blas_int)k_ext, w.data());
    if (info != 0)
        throw std::runtime_error("power_iterate_and_project: dsyevd failed (info=" +
                                  std::to_string(info) + "). For k_ext > 32766, this is likely why.");

    EighResult result;
    // w is ascending; tail(k_target).reverse() -> descending top-k.
    result.eigenvalues = w.tail(k_target).reverse();
    // Materialise the reversed block into a plain contiguous matrix before the
    // n x k_target DGEMM — rowwise().reverse() on a block expression forces
    // column-by-column scatter during DGEMM, defeating tiling.
    const Eigen::MatrixXd evecs_sorted =
        B.rightCols(k_target).rowwise().reverse().eval();
    result.eigenvectors = Q * evecs_sorted;
    return result;
}

// Convenience one-shot entry point: sketch + power-iterate + project.
template <typename MatVecApply>
EighResult randomized_symmetric_eigh(
    MatVecApply&& apply,
    int n,
    int k_target,
    int oversample = 20,
    int power_iter = 3,
    const Eigen::MatrixXd* warm_start = nullptr)
{
    const int k_ext = std::min(k_target + oversample, n - 1);
    auto [omega, Y] = build_randomized_sketch(apply, n, k_ext, warm_start);
    (void)omega;
    return power_iterate_and_project(std::forward<MatVecApply>(apply), std::move(Y), k_target, power_iter);
}

// Refine an orthonormal trial basis against the original operator. This costs
// one additional block matvec and returns Rayleigh-Ritz eigenpairs, so callers
// can use the resulting eigenvalues for spectral-mass decisions.
template <typename MatVecApply>
EighResult rayleigh_ritz_refine(
    MatVecApply&& apply,
    Eigen::MatrixXd basis,
    int k_target)
{
    const int k_ext = static_cast<int>(basis.cols());
    if (k_target > k_ext)
        throw std::invalid_argument("rayleigh_ritz_refine: k_target exceeds basis width.");

    Eigen::MatrixXd A_basis = apply(basis);
    Eigen::MatrixXd B = basis.transpose() * A_basis;
    A_basis.resize(0, 0);
    B = 0.5 * (B + B.transpose());

    Eigen::VectorXd w(k_ext);
    const int info = gcta_dsyevd((gcta_blas_int)k_ext, B.data(), (gcta_blas_int)k_ext, w.data());
    if (info != 0)
        throw std::runtime_error("rayleigh_ritz_refine: dsyevd failed (info=" +
                                  std::to_string(info) + "). For k_ext > 32766, this is likely why.");

    EighResult result;
    result.eigenvalues = w.tail(k_target).reverse();
    const Eigen::MatrixXd evecs_sorted =
        B.rightCols(k_target).rowwise().reverse().eval();
    result.eigenvectors = basis * evecs_sorted;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Lanczos (Spectra), for well-separated top spectra with few matvecs
// ─────────────────────────────────────────────────────────────────────────

// Spectra requires single-vector perform_op(x_in, y_out); this wraps a
// block-capable `apply` functor (the same one rSVD uses) so call sites
// don't need to maintain two different matvec adapters for the same matrix.
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
// Tall-skinny thin SVD via QR reduction
// ─────────────────────────────────────────────────────────────────────────

struct ThinSVDResult {
    Eigen::MatrixXd U;                // n x k, orthonormal columns
    Eigen::VectorXd singular_values;  // descending, size k
};

// Thin SVD of an already-materialised n x k matrix Z (k << n), e.g. the
// Nystrom sketch. Eigen's BDCSVD has no LAPACKE binding (unlike
// HouseholderQR and SelfAdjointEigenSolver, which do when EIGEN_USE_LAPACKE
// is set), so calling it directly on an n x k matrix leaves it doing
// step to the k x k factor R, where its cost is negligible regardless of
// how it's implemented, while the n-scale work (QR, and the final Q * U_R)
// goes through LAPACKE_dgeqrf/dorgqr and GEMM respectively.
inline ThinSVDResult tall_skinny_thin_svd(const Eigen::MatrixXd& Z) {
    const int n = static_cast<int>(Z.rows());
    const int k = static_cast<int>(Z.cols());

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Z);
    const Eigen::MatrixXd R = qr.matrixQR().topRows(k).triangularView<Eigen::Upper>();
    const Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, k);

    Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU> svd_r(R);

    ThinSVDResult result;
    result.singular_values = svd_r.singularValues();
    result.U = Q * svd_r.matrixU();   // n x k GEMM
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Nystrom single-pass approximation
// ─────────────────────────────────────────────────────────────────────────
//
// Uses 1 matvec pass (vs power_iter+1 for rSVD), so it is 4x cheaper at
// the default power_iter=3. The trade-off: no power-iteration sharpening
// of the subspace estimate near the k-th eigenvalue. Accuracy degrades
// faster as k/n grows and eigenvalues become densely packed near the
// spectral boundary.
//
// GRM eigenvalue tails are often negative (missing genotypes introduce
// small negative eigenvalues). C = Omega^T K Omega inherits this: it is
// only PSD when K itself is. Use a signed pseudoinverse so the one-pass
// approximation represents those directions instead of treating every
// nonpositive eigenvalue as numerical noise. Only values at the backward
// error scale of forming C are treated as its numerical nullspace.
template <typename MatVecApply>
EighResult nystrom_symmetric_eigh(
    MatVecApply&& apply,
    int n,
    int k_target,
    int oversample = 20,
    const Eigen::MatrixXd* warm_start = nullptr)
{
    const int k_ext = std::min(k_target + oversample, n - 1);

    // One-pass sketch: omega ~ Gaussian (or warm-started), Y = K * omega
    auto [omega, Y] = build_randomized_sketch(apply, n, k_ext, warm_start);

    // C = omega^T * Y  (k_ext × k_ext)
    // Symmetric in exact arithmetic for PSD K; indefinite when K has
    // negative eigenvalues (e.g. GRM with missing genotypes).
    Eigen::MatrixXd C = omega.transpose() * Y;
    C = 0.5 * (C + C.transpose());

    // Bound the accumulated dot-product error in C = Omega^T Y. This is a
    // numerical-rank criterion, not a spectral regularizer: nonzero
    // negative eigenvalues remain part of the signed Nyström approximation.
    const double eps = std::numeric_limits<double>::epsilon();
    const double n_eps = static_cast<double>(n) * eps;
    const double gamma_n = n_eps / (1.0 - n_eps);
    const double eps_C = gamma_n * omega.norm() * Y.norm();
    omega.resize(0, 0);

    Eigen::VectorXd w_C(k_ext);
    const int info_C = gcta_dsyevd((gcta_blas_int)k_ext, C.data(), (gcta_blas_int)k_ext, w_C.data());
    if (info_C != 0)
        throw std::runtime_error("nystrom_symmetric_eigh: dsyevd on sketch C failed (info=" +
                                  std::to_string(info_C) + "). For k_ext > 32766, this is likely why.");

    const Eigen::VectorXd& lam_C = w_C;
    const Eigen::VectorXd lam_sqrt_abs_inv = lam_C.unaryExpr(
        [eps_C](double lam) { return (std::abs(lam) > eps_C) ? 1.0 / std::sqrt(std::abs(lam)) : 0.0; });
    const Eigen::VectorXd signs = lam_C.unaryExpr(
        [eps_C](double lam) { return (std::abs(lam) > eps_C) ? ((lam > 0.0) ? 1.0 : -1.0) : 0.0; });

    // K_nys = Z sign(Lambda) Z^T, where
    // Z = Y V |Lambda|^{-1/2}. This retains the signed spectrum of an
    // indefinite GRM without an additional K matvec.
    Eigen::MatrixXd Z = Y * (C * lam_sqrt_abs_inv.asDiagonal());
    Y.resize(0, 0);
    C.resize(0, 0);

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Z);
    const Eigen::MatrixXd R = qr.matrixQR().topRows(k_ext).triangularView<Eigen::Upper>();
    const Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, k_ext);
    Z.resize(0, 0);

    Eigen::MatrixXd T = R * signs.asDiagonal() * R.transpose();
    T = 0.5 * (T + T.transpose());
    Eigen::VectorXd w_T(k_ext);
    const int info_T = gcta_dsyevd((gcta_blas_int)k_ext, T.data(), (gcta_blas_int)k_ext, w_T.data());
    if (info_T != 0)
        throw std::runtime_error("nystrom_symmetric_eigh: dsyevd on signed core T failed (info=" +
                                  std::to_string(info_T) + "). For k_ext > 32766, this is likely why.");

    EighResult result;
    result.eigenvalues = w_T.tail(k_target).reverse();
    const Eigen::MatrixXd evecs_sorted =
        T.rightCols(k_target).rowwise().reverse().eval();
    result.eigenvectors = Q * evecs_sorted;
    return result;
}

} // namespace gcta_eigh