/*
 * symmetric_eigendecomp.hpp
 *
 * Approximate top-k eigendecomposition methods for a symmetric (typically
 * PSD) matrix A, shared between grm.cpp (gcta::pca) and RemlEngine.cpp
 * (compute_woodbury_basis).
 *
 * Every entry point takes a matvec functor rather than a concrete matrix
 * type: `apply(X)` must return A * X for X either an n-vector or an n x m
 * block. This lets callers pass a plain MatrixXd, a selfadjointView<Upper>,
 * or in future a tiled/streaming matvec, without this header caring.
 * Both current callers (pca_stream.cpp's PCA path and RemlEngine.cpp's
 * compute_woodbury_basis) now store their GRM upper-triangle-only and pass
 * apply = [](X){ return G.selfadjointView<Eigen::Upper>() * X; } — see
 * grm_binary_io.hpp's upper_only load option for why.
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
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "cpu.h"  // gcta_dsyevr

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

/*
inline void fill_standard_normal(Eigen::Ref<Eigen::MatrixXd> matrix) {
    std::normal_distribution<double> normal(0.0, 1.0);
    for (Eigen::Index col = 0; col < matrix.cols(); ++col)
        for (Eigen::Index row = 0; row < matrix.rows(); ++row)
            matrix(row, col) = normal(shared_rng());
}
*/
// Parallel fill: each thread gets its own mt19937, seeded from a draw off
// the shared generator rather than from seed_seq{base, tid} -- a handful of
// scalar differences fed to seed_seq can leave early generator states
// weakly correlated across threads; drawing each thread's seed directly
// off shared_rng() avoids that, at the cost of a handful of extra serial
// draws done up front.
//
// NOTE: output now depends on thread count/column partitioning -- no
// longer bit-reproducible across different thread-count runs. Omega is an
// ephemeral random sketch, so this shouldn't matter for correctness, but
// flag it if any test depends on exact reproducibility of the sketch.
inline void fill_standard_normal(Eigen::Ref<Eigen::MatrixXd> matrix) {
    const Eigen::Index rows = matrix.rows();
    const Eigen::Index cols = matrix.cols();
    if (rows == 0 || cols == 0) return;

    const int num_threads = static_cast<int>(std::min<Eigen::Index>(omp_get_max_threads(), cols));

    // Serial, cheap (num_threads draws) -- keeps shared_rng() as the sole
    // entropy source and avoids any parallel access to it.
    std::vector<std::uint32_t> thread_seeds(num_threads);
    for (int t = 0; t < num_threads; ++t)
        thread_seeds[t] = static_cast<std::uint32_t>(shared_rng()());

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        std::mt19937 local_rng(thread_seeds[tid]);
        std::normal_distribution<double> normal(0.0, 1.0);

        // Column-major storage: each thread's columns are contiguous,
        // disjoint memory -- no false sharing beyond block boundaries.
        const Eigen::Index cols_per_thread = (cols + num_threads - 1) / num_threads;
        const Eigen::Index col_begin = std::min(cols, static_cast<Eigen::Index>(tid) * cols_per_thread);
        const Eigen::Index col_end   = std::min(cols, col_begin + cols_per_thread);

        for (Eigen::Index col = col_begin; col < col_end; ++col)
            for (Eigen::Index row = 0; row < rows; ++row)
                matrix(row, col) = normal(local_rng);
    }
}

// Keep the sketch width proportional to the target rank, but avoid the old
// fixed 20-column floor on small k_target. The randomized range finder does
// not need a large oversample when the target rank is only a handful of PCs.
inline int recommended_oversample(int k_target, int floor = 8, int cap = 128) {
    if (k_target <= 0) return 0;
    const int soft_target = std::max(8, k_target);
    return std::clamp(k_target / 8, std::min(floor, soft_target), cap);
}

// Full eigendecomposition of the symmetric k x k matrix whose UPPER triangle is
// stored in B (via gcta_dsyevr, il=1, iu=k: LAPACK's all-eigenpairs MRRR path).
// B is destroyed. On return w (k) and Z (k x k) hold the eigenpairs in DESCENDING
// order; the leading columns of Z are contiguous, so Z.leftCols(k_top) is a plain
// column block for the caller's GEMM (no reversed copy).
//
// Only the upper triangle of B is read, so a raw product Q' (A Q) needs no B + B'
// symmetrisation, and every Rayleigh-Ritz call site sees the same triangle.
//
// Why dsyevr and not dsyevd: dsyevd (jobz='V') overwrites all of B with eigenvectors
// and needs a further ~2*k^2 workspace, so the never-written lower triangle would be
// faulted in anyway (peak ~3*k^2). dsyevr references only the upper triangle, returns
// vectors in a separate Z and needs O(k) workspace: peak ~ k^2/2 (touched upper
// triangle) + k^2 (Z). It also avoids gcta_dsyevd's k >= 32766 workspace guard.
inline void eigh_upper_desc(Eigen::MatrixXd& B, Eigen::VectorXd& w, Eigen::MatrixXd& Z)
{
    const int k = static_cast<int>(B.rows());
    if (B.cols() != k || k < 1)
        throw std::invalid_argument("eigh_upper_desc: B must be square and non-empty.");
    w.resize(k);
    Z.resize(k, k);
    std::vector<gcta_blas_int> isuppz(2 * static_cast<size_t>(k));
    gcta_blas_int m_found = 0;
    const int info = gcta_dsyevr((gcta_blas_int)k, B.data(), (gcta_blas_int)k,
                                 (gcta_blas_int)1, (gcta_blas_int)k, &m_found,
                                 w.data(), Z.data(), (gcta_blas_int)k, isuppz.data());
    if (info != 0 || m_found != k)
        throw std::runtime_error("eigh_upper_desc: dsyevr failed (info=" + std::to_string(info) + ").");
    for (int i = 0; i < k / 2; ++i) {     // ascending -> descending, in place (no copy)
        Z.col(i).swap(Z.col(k - 1 - i));
        std::swap(w[i], w[k - 1 - i]);
    }
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

    for (int pi = 0; pi < power_iter; ++pi) {
        // Cholesky QR2 (DSYRK+DPOTRF+DTRSM) rather than LAPACK Householder QR:
        // all Level-3 BLAS, so it keeps scaling with thread count at large
        // k_ext where dgeqrf/dorgqr's panel factorization stops parallelizing.
        // Deliberately not hoisting the k_ext x k_ext Gram scratch across
        // iterations: at k_ext in the thousands its malloc/free cost is
        // negligible next to the O(n*k_ext^2) SYRK/TRSM work, while keeping
        // it alive for this whole function's scope would hold it resident
        // alongside the much larger n x k_ext buffers below, raising peak RSS
        // (and, empirically, wall time too) for no measured benefit here.
        int info = gcta_cholesky_qr_thin_Q((gcta_blas_int)n, (gcta_blas_int)k_ext, Y.data(), (gcta_blas_int)n);
        if (info != 0)
            throw std::runtime_error("power_iterate_and_project: orthogonalization failed (info=" +
                                      std::to_string(info) + ").");
        Y = apply(Y);
    }

    {
        int info = gcta_cholesky_qr_thin_Q((gcta_blas_int)n, (gcta_blas_int)k_ext, Y.data(), (gcta_blas_int)n);
        if (info != 0)
            throw std::runtime_error("power_iterate_and_project: orthogonalization failed (info=" +
                                      std::to_string(info) + ").");
    }
    Eigen::MatrixXd Q = std::move(Y);

    Eigen::MatrixXd AQ = apply(Q);
    Eigen::MatrixXd B  = Q.transpose() * AQ;   // k_ext x k_ext, symmetric in exact arithmetic
    AQ.resize(0, 0);
    // eigh_upper_desc reads only the upper triangle, so no B + B' symmetrisation is
    // needed; rayleigh_ritz_refine does the same, so the two Rayleigh-Ritz call sites
    // still agree on which triangle's rounding error is used. Matters here more than
    // most places: Woodbury's EIG99/MP-edge logic reads tail eigenvalues straight out
    // of this call.
    Eigen::VectorXd w;
    Eigen::MatrixXd Z;
    eigh_upper_desc(B, w, Z);
    B.resize(0, 0);

    EighResult result;
    result.eigenvalues  = w.head(k_target);
    result.eigenvectors = Q * Z.leftCols(k_target);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Locked expansion: grow a converged Ritz basis without recomputing it
// ─────────────────────────────────────────────────────────────────────────
//
// (V0, theta0) are the top-k0 Ritz pairs of a previous Rayleigh-Ritz step on
// `apply`: V0 orthonormal and V0' A V0 = diag(theta0) (an identity of RR, exact up
// to rounding). Instead of a fresh k_ext-wide sketch, only m = k_ext - k0 NEW
// directions are sketched and power-iterated in the orthogonal complement of V0
// (iterating (I-P0) A (I-P0), P0 = V0 V0'), and ONE Rayleigh-Ritz is done on the
// orthonormal union W = [V0, Q1]:
//
//     W' A W = [ diag(theta0)   V0' A Q1 ]     A symmetric => Q1' A V0 = (V0' A Q1)',
//              [ Q1' A V0       Q1' A Q1 ]     and V0' A Q1 = V0' * (A Q1): A*V0 is never needed.
//
// What is guaranteed (exact arithmetic, A symmetric, not necessarily PSD):
//   * Exactness: this is the exact Rayleigh-Ritz of A on span(W), not an approximation.
//   * Monotone mass: span(W) contains span(V0), so by Cauchy interlacing every Ritz
//     value is >= its counterpart from the previous round; the captured trace
//     sum_{i<=j} theta_i can only grow at any fixed rank j <= k0. Ritz values stay
//     lower bounds of the true eigenvalues, so the mass lag is >= 0.
// What is not: how small the lag is. That depends on the spectral decay and on
// power_iter (fresh-block convergence factor ~ (lambda_{k_ext+1}/lambda_j)^(power_iter+1)).
//
// k_target is the number of Ritz pairs returned (descending). Callers that keep expanding
// should pass k_target = k_ext and lock ALL pairs: the oversample vectors are then
// refined by the next round instead of being discarded and recomputed.
//
// Cost: (power_iter + 2) block products on m columns (the first also carries a probe
// column), plus (power_iter + 2) deflation GEMM pairs costing ~2*k0/n of a block
// product each. Across a doubling schedule each column is touched once.
//
// Safety (any failure throws; callers should fall back to a cold sketch):
//   * A probe z = V0 g rides the first product for free; V0'(A z) must equal
//     theta0 .* g. This validates the operator, V0's orthonormality and theta0.
//   * After orthonormalisation, ||V0' Q1||_F must be ~0. gcta_cholesky_qr_thin_Q
//     silently falls back to Householder QR when a Gram matrix is not positive
//     definite; on a numerically rank-deficient block dorgqr completes Q with
//     directions that need not be orthogonal to V0, which would invalidate the
//     block form above.
template <typename MatVecApply>
EighResult expand_symmetric_eigh(
    MatVecApply&& apply,
    const Eigen::MatrixXd& V0,
    const Eigen::VectorXd& theta0,
    int k_ext,
    int k_target,
    int power_iter = 3)
{
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

    // Y <- (I - V0 V0') Y, `reps` times. One rep between power passes only has to
    // keep the next Cholesky-QR Gram matrix well conditioned; two reps ("twice is
    // enough") before the final orthogonalization give full orthogonality to V0.
    auto deflate = [&](Eigen::MatrixXd& Y, int reps) {
        for (int rep = 0; rep < reps; ++rep) {
            const Eigen::MatrixXd C = V0.transpose() * Y;   // k0 x m
            Y.noalias() -= V0 * C;
        }
    };
    auto orthonormalize = [&](Eigen::MatrixXd& Y) {
        const int info = gcta_cholesky_qr_thin_Q((gcta_blas_int)n, (gcta_blas_int)m, Y.data(), (gcta_blas_int)n);
        if (info != 0)
            throw std::runtime_error("expand_symmetric_eigh: orthogonalization failed (info=" +
                                      std::to_string(info) + ").");
    };

    // First product: m random columns plus one probe column z = V0 g.
    Eigen::MatrixXd Y(n, m + 1);
    fill_standard_normal(Y.leftCols(m));
    Eigen::VectorXd g(k0);
    {
        Eigen::MatrixXd gm(k0, 1);
        fill_standard_normal(gm);
        g = gm.col(0);
    }
    Y.col(m).noalias() = V0 * g;
    Y = apply(Y);
    const Eigen::VectorXd Kz = Y.col(m);
    Y.conservativeResize(Eigen::NoChange, m);
    {
        const Eigen::VectorXd expect = theta0.cwiseProduct(g);
        const double scale = expect.norm();
        const Eigen::VectorXd got  = V0.transpose() * Kz;
        if (scale > 0.0 && (got - expect).norm() > 1e-8 * scale)
            throw std::runtime_error("expand_symmetric_eigh: locked basis fails the Ritz invariance "
                                      "check (V0' A V0 != diag(theta0)).");
    }

    for (int pi = 0; pi < power_iter; ++pi) {
        deflate(Y, 1);
        orthonormalize(Y);
        Y = apply(Y);
    }
    deflate(Y, 2);
    orthonormalize(Y);                       // Y is now Q1
    if ((V0.transpose() * Y).norm() > 1e-8)
        throw std::runtime_error("expand_symmetric_eigh: fresh block is not orthogonal to the locked basis.");

    Eigen::MatrixXd KQ1 = apply(Y);          // n x m

    // eigh_upper_desc reads only the upper triangle: the strict lower triangle of the
    // leading k0 x k0 block and the whole lower-left block are never written or read,
    // so those pages stay unfaulted.
    Eigen::MatrixXd B(k_ext, k_ext);
    B.topLeftCorner(k0, k0).triangularView<Eigen::StrictlyUpper>().setZero();
    B.topLeftCorner(k0, k0).diagonal() = theta0;
    B.topRightCorner(k0, m).noalias() = V0.transpose() * KQ1;        // V0' A Q1
    B.bottomRightCorner(m, m).noalias() = Y.transpose() * KQ1;       // Q1' A Q1
    KQ1.resize(0, 0);

    Eigen::VectorXd w;
    Eigen::MatrixXd Z;                       // k_ext x k_ext, descending
    eigh_upper_desc(B, w, Z);
    B.resize(0, 0);

    EighResult result;
    result.eigenvalues = w.head(k_target);
    result.eigenvectors.resize(n, k_target);
    result.eigenvectors.noalias()  = V0 * Z.topLeftCorner(k0, k_target);
    result.eigenvectors.noalias() += Y * Z.bottomLeftCorner(m, k_target);
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

    Eigen::VectorXd w;
    Eigen::MatrixXd Z;
    eigh_upper_desc(B, w, Z);   // reads only the upper triangle; see power_iterate_and_project
    B.resize(0, 0);

    EighResult result;
    result.eigenvalues  = w.head(k_target);
    result.eigenvectors = basis * Z.leftCols(k_target);
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

    // Z is const& (caller retains ownership), so one n x k copy is
    // unavoidable here — same as HouseholderQR's internal copy today, no
    // regression. CholeskyQR2 (DSYRK+DPOTRF+DTRSM) replaces dgeqrf/dorgqr:
    // pure Level-3 BLAS, so it keeps scaling with thread count at large k
    // where dgeqrf/dorgqr's panel factorization stops parallelizing. R is
    // recovered as Q^T*Z (BLAS3 GEMM) rather than threaded through the
    // factorization, since the copy already makes the original Z available.
    Eigen::MatrixXd Q = Z;
    const int info = gcta_cholesky_qr_thin_Q((gcta_blas_int)n, (gcta_blas_int)k, Q.data(), (gcta_blas_int)n);
    if (info != 0)
        throw std::runtime_error("tall_skinny_thin_svd: orthogonalization failed (info=" + std::to_string(info) + ").");

    const Eigen::MatrixXd R = Q.transpose() * Z;

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
    // Only the upper triangle is read by eigh_upper_desc, so no symmetrisation.
    Eigen::MatrixXd C = omega.transpose() * Y;

    // Bound the accumulated dot-product error in C = Omega^T Y. This is a
    // numerical-rank criterion, not a spectral regularizer: nonzero
    // negative eigenvalues remain part of the signed Nyström approximation.
    const double eps = std::numeric_limits<double>::epsilon();
    const double n_eps = static_cast<double>(n) * eps;
    const double gamma_n = n_eps / (1.0 - n_eps);
    const double eps_C = gamma_n * omega.norm() * Y.norm();
    omega.resize(0, 0);

    Eigen::VectorXd lam_C;
    Eigen::MatrixXd V_C;                 // eigenvectors of C (order matches lam_C)
    eigh_upper_desc(C, lam_C, V_C);
    C.resize(0, 0);
    const Eigen::VectorXd lam_sqrt_abs_inv = lam_C.unaryExpr(
        [eps_C](double lam) { return (std::abs(lam) > eps_C) ? 1.0 / std::sqrt(std::abs(lam)) : 0.0; });
    const Eigen::VectorXd signs = lam_C.unaryExpr(
        [eps_C](double lam) { return (std::abs(lam) > eps_C) ? ((lam > 0.0) ? 1.0 : -1.0) : 0.0; });

    // K_nys = Z sign(Lambda) Z^T, where
    // Z = Y V |Lambda|^{-1/2}. This retains the signed spectrum of an
    // indefinite GRM without an additional K matvec.
    Eigen::MatrixXd Z = Y * (V_C * lam_sqrt_abs_inv.asDiagonal());
    Y.resize(0, 0);
    V_C.resize(0, 0);

    // Z is owned and mutable here (unlike tall_skinny_thin_svd's const&), so
    // this needs zero extra n x k_ext buffers: gcta_cholesky_qr_thin_QR
    // (DSYRK+DPOTRF+DTRSM, pure Level-3 BLAS) factors Z into its own
    // explicit Q in place and returns the tiny k_ext x k_ext R separately,
    // without an extra n-scale copy. Do NOT resize Z away below; it doubles
    // as Q for the rest of the function.
    Eigen::MatrixXd R(k_ext, k_ext);
    const int info_qr = gcta_cholesky_qr_thin_QR((gcta_blas_int)n, (gcta_blas_int)k_ext, Z.data(), (gcta_blas_int)n,
                                                  R.data(), (gcta_blas_int)k_ext);
    if (info_qr != 0)
        throw std::runtime_error("nystrom_symmetric_eigh: orthogonalization failed (info=" + std::to_string(info_qr) + ").");
    const Eigen::MatrixXd& Q = Z;   // Z now holds explicit Q; alias, not a copy

    Eigen::MatrixXd T = R * signs.asDiagonal() * R.transpose();   // upper triangle is read below
    Eigen::VectorXd w_T;
    Eigen::MatrixXd U_T;
    eigh_upper_desc(T, w_T, U_T);
    T.resize(0, 0);

    EighResult result;
    result.eigenvalues  = w_T.head(k_target);
    result.eigenvectors = Q * U_T.leftCols(k_target);
    return result;
}

} // namespace gcta_eigh