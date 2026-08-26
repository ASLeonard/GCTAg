#ifndef GCTA_CPU_H
#define GCTA_CPU_H

#include <cmath>
#include <limits>
#include <vector>
#include "Logger.h"

#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC)) || defined(__amd64)
  #define GCTA_CPU_x86 1
#else
  #define GCTA_CPU_x86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
  #define GCTA_CPU_ARM 1
#else
  #define GCTA_CPU_ARM 0
#endif

//Can replace this header entirely with a CMake file(GENERATE) approach:

#if defined(GCTA_USE_MKL)
  #include <mkl.h>
  typedef int gcta_blas_int;
#elif defined(GCTA_USE_OPENBLAS)
  #include <cblas.h>
  #include <lapacke.h>
  typedef lapack_int gcta_blas_int;

#elif defined(GCTA_USE_AOCL)
  #include <cblas.h>
  #include <lapacke.h>
  typedef lapack_int gcta_blas_int;

#elif defined(GCTA_USE_ACCELERATE)
  #include <veclib/cblas_new.h>
  #include <veclib/lapack.h>
  typedef __LAPACK_int gcta_blas_int;
#endif

// Portable dpotrf: Cholesky factorization of a symmetric positive-definite matrix
// (lower triangle, column-major).  Returns 0 on success, non-zero on failure.
inline int gcta_dpotrf(gcta_blas_int n, double* a, gcta_blas_int lda) {
#if defined(GCTA_USE_ACCELERATE)
    char uplo = 'L';
    gcta_blas_int info = 0;
    dpotrf_(&uplo, &n, a, &lda, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda));
#endif
}

// Portable dpotri: in-place inversion of a Cholesky-factored symmetric positive-definite
// matrix (lower triangle).  Returns 0 on success, non-zero on failure.
inline int gcta_dpotri(gcta_blas_int n, double* a, gcta_blas_int lda) {
#if defined(GCTA_USE_ACCELERATE)
    // Accelerate exposes Fortran-ABI dpotri_ with pointer arguments.
    char uplo = 'L';
    gcta_blas_int info = 0;
    dpotri_(&uplo, &n, a, &lda, &info);
    return static_cast<int>(info);
#else
    // MKL, OpenBLAS, and AOCL all provide the LAPACKE C interface.
    return static_cast<int>(LAPACKE_dpotri(LAPACK_COL_MAJOR, 'L', n, a, lda));
#endif
}

// Portable dgetrf: LU factorization with partial pivoting.
// ipiv must be pre-allocated to length n.  Returns 0 on success.
inline int gcta_dgetrf(gcta_blas_int n, double* a, gcta_blas_int lda, gcta_blas_int* ipiv) {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int info = 0;
    dgetrf_(&n, &n, a, &lda, ipiv, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ipiv));
#endif
}

// Portable dgetri: in-place inversion from an LU factorization produced by gcta_dgetrf.
// ipiv must be the pivot array returned by gcta_dgetrf.  Returns 0 on success.
inline int gcta_dgetri(gcta_blas_int n, double* a, gcta_blas_int lda, gcta_blas_int* ipiv) {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int info = 0;
    // Workspace query.
    gcta_blas_int lwork = -1;
    double work_query = 0.0;
    dgetri_(&n, a, &lda, ipiv, &work_query, &lwork, &info);
    lwork = static_cast<gcta_blas_int>(work_query);
    std::vector<double> work(lwork);
    dgetri_(&n, a, &lda, ipiv, work.data(), &lwork, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dgetri(LAPACK_COL_MAJOR, n, a, lda, ipiv));
#endif
}

// ---- gcta_dsyevd -------------------------------------------------------
// Full symmetric eigensolver (divide-and-conquer), column-major, upper triangle.
// a[n×n] is overwritten with eigenvectors; w receives eigenvalues ascending.
// Returns 0 on success, non-zero LAPACK info otherwise.
inline int gcta_dsyevd(gcta_blas_int n, double* a, gcta_blas_int lda, double* w)
{
#if defined(GCTA_USE_ACCELERATE)
    const char    jobz = 'V', uplo = 'U';
    gcta_blas_int info = 0;
    // Workspace query: pass lwork = liwork = -1 to obtain optimal sizes.
    gcta_blas_int lwork_q = -1, liwork_q = -1;
    double        work_sz = 0.0;
    gcta_blas_int iwork_sz = 0;
    dsyevd_(&jobz, &uplo, &n, a, &lda, w,
            &work_sz, &lwork_q, &iwork_sz, &liwork_q, &info);
    if (info != 0) return static_cast<int>(info);
    gcta_blas_int lwork  = static_cast<gcta_blas_int>(work_sz);
    gcta_blas_int liwork = iwork_sz;
    std::vector<double>        work(lwork);
    std::vector<gcta_blas_int> iwork(liwork);
    dsyevd_(&jobz, &uplo, &n, a, &lda, w,
            work.data(), &lwork, iwork.data(), &liwork, &info);
    return static_cast<int>(info);
#else
    // AOCL/OpenBLAS's LAPACKE_dsyevd workspace-size calculation (~2n^2)
    // overflows a 32-bit int above n ~= 32766 and crashes rather than
    // returning a nonzero info -- so callers checking info afterward can't
    // catch it. MKL promotes internally and doesn't hit this, so it's
    // exempted. -1 mirrors LAPACK's own "argument 1 (n) invalid" convention;
    // safe to reuse here since a genuine n<0 from LAPACK can't otherwise occur.
#if !defined(GCTA_USE_MKL)
    if (n >= 32766) return -1;
#endif
    return static_cast<int>(
        LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', n, a, lda, w));
#endif
}

// ---- gcta_dsyevr -------------------------------------------------------
// Partial symmetric eigensolver (MRRR), index range [il, iu] (1-based, ascending).
// a[n×n]: input matrix (upper triangle, column-major); overwritten on return.
// il, iu: indices of smallest/largest desired eigenvalue (1 ≤ il ≤ iu ≤ n).
// m_found: receives count actually returned.
// w:      eigenvalues (size ≥ iu-il+1), ascending.
// z:      eigenvectors (n × (iu-il+1), column-major); ldz ≥ n.
// isuppz: support array (size ≥ 2*(iu-il+1)).
// Returns 0 on success, non-zero LAPACK info otherwise.
inline int gcta_dsyevr(gcta_blas_int n, double* a, gcta_blas_int lda,
                       gcta_blas_int il, gcta_blas_int iu,
                       gcta_blas_int* m_found,
                       double* w, double* z, gcta_blas_int ldz,
                       gcta_blas_int* isuppz)
{
#if defined(GCTA_USE_ACCELERATE)
    const char   jobz = 'V', range = 'I', uplo = 'U';
    const double vl = 0.0, vu = 0.0, abstol = 0.0;
    gcta_blas_int info = 0;
    // Workspace query.
    gcta_blas_int lwork_q = -1, liwork_q = -1;
    double        work_sz = 0.0;
    gcta_blas_int iwork_sz = 0;
    dsyevr_(&jobz, &range, &uplo, &n, a, &lda,
            &vl, &vu, &il, &iu, &abstol,
            m_found, w, z, &ldz, isuppz,
            &work_sz, &lwork_q, &iwork_sz, &liwork_q, &info);
    if (info != 0) return static_cast<int>(info);
    gcta_blas_int lwork  = static_cast<gcta_blas_int>(work_sz);
    gcta_blas_int liwork = iwork_sz;
    std::vector<double>        work(lwork);
    std::vector<gcta_blas_int> iwork(liwork);
    dsyevr_(&jobz, &range, &uplo, &n, a, &lda,
            &vl, &vu, &il, &iu, &abstol,
            m_found, w, z, &ldz, isuppz,
            work.data(), &lwork, iwork.data(), &liwork, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(
        LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', n, a, lda,
                       0.0, 0.0, il, iu, 0.0,
                       m_found, w, z, ldz, isuppz));
#endif
}

// Raw Fortran ABI declarations are only needed on Accelerate, where veclib does
// not expose the LAPACKE C wrappers. On OpenBLAS/MKL/AOCL we use the LAPACKE
// entry points directly, matching the pattern used by gcta_dsyevr.
#if defined(GCTA_USE_ACCELERATE)
extern "C" {
void sgeqrf_(const gcta_blas_int* m, const gcta_blas_int* n, float* A,
             const gcta_blas_int* lda, float* tau, float* work,
             const gcta_blas_int* lwork, gcta_blas_int* info);
void sorgqr_(const gcta_blas_int* m, const gcta_blas_int* n, const gcta_blas_int* k,
             float* A, const gcta_blas_int* lda, const float* tau,
             float* work, const gcta_blas_int* lwork, gcta_blas_int* info);
void dgeqrf_(const gcta_blas_int* m, const gcta_blas_int* n, double* A,
             const gcta_blas_int* lda, double* tau, double* work,
             const gcta_blas_int* lwork, gcta_blas_int* info);
void dorgqr_(const gcta_blas_int* m, const gcta_blas_int* n, const gcta_blas_int* k,
             double* A, const gcta_blas_int* lda, const double* tau,
             double* work, const gcta_blas_int* lwork, gcta_blas_int* info);
double dlansy_(const char* norm, const char* uplo, const gcta_blas_int* n, const double* a,
               const gcta_blas_int* lda, double* work);
void dpocon_(const char* uplo, const gcta_blas_int* n, const double* a, const gcta_blas_int* lda,
             const double* anorm, double* rcond, double* work, gcta_blas_int* iwork, gcta_blas_int* info);
}
#endif

// Portable one-norm of a symmetric matrix (only the lower triangle is
// referenced), feeding gcta_dpocon's pre-factorization anorm argument.
inline double gcta_dlansy_one(gcta_blas_int n, const double* a, gcta_blas_int lda) {
#if defined(GCTA_USE_ACCELERATE)
    char norm = '1', uplo = 'L';
    std::vector<double> work(n);
    return dlansy_(&norm, &uplo, &n, a, &lda, work.data());
#else
    return LAPACKE_dlansy(LAPACK_COL_MAJOR, '1', 'L', n, a, lda);
#endif
}

// Reciprocal condition number estimate (DPOCON) of an SPD matrix already
// Cholesky-factored in place (lower triangle), given its pre-factorization
// one-norm `anorm` from gcta_dlansy_one. Returns 0 on success (rcond
// written), non-zero LAPACK info otherwise.
inline int gcta_dpocon(gcta_blas_int n, const double* chol_lower, gcta_blas_int lda, double anorm, double* rcond) {
#if defined(GCTA_USE_ACCELERATE)
    char uplo = 'L';
    gcta_blas_int info = 0;
    std::vector<double> work(3 * n);
    std::vector<gcta_blas_int> iwork(n);
    dpocon_(&uplo, &n, chol_lower, &lda, &anorm, rcond, work.data(), iwork.data(), &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dpocon(LAPACK_COL_MAJOR, 'L', n, chol_lower, lda, anorm, rcond));
#endif
}


inline int gcta_sgeqrf(gcta_blas_int m, gcta_blas_int k, float* A, gcta_blas_int lda, float* tau) {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int lwork = -1, info = 0;
    float wkopt = 0.0f;
    sgeqrf_(&m, &k, A, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) return static_cast<int>(info);
    lwork = static_cast<gcta_blas_int>(wkopt);
    Eigen::VectorXf work(lwork);
    sgeqrf_(&m, &k, A, &lda, tau, work.data(), &lwork, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, k, A, lda, tau));
#endif
}

inline int gcta_sorgqr(gcta_blas_int m, gcta_blas_int k, float* A, gcta_blas_int lda, const float* tau) {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int lwork = -1, info = 0;
    float wkopt = 0.0f;
    sorgqr_(&m, &k, &k, A, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) return static_cast<int>(info);
    lwork = static_cast<gcta_blas_int>(wkopt);
    Eigen::VectorXf work(lwork);
    sorgqr_(&m, &k, &k, A, &lda, tau, work.data(), &lwork, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_sorgqr(LAPACK_COL_MAJOR, m, k, k, A, lda, tau));
#endif
}

// Factor A (m x k, column-major, m >= k) in place: on return, the upper
// k x k triangle of A holds R, the rest holds Householder reflectors, and
// tau (size k) holds their scalars. This matches the portable pattern used by
// gcta_dsyevr: raw Fortran ABI on Accelerate, LAPACKE elsewhere.
inline int gcta_dgeqrf(gcta_blas_int m, gcta_blas_int k, double* A, gcta_blas_int lda, double* tau) {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int lwork = -1, info = 0;
    double wkopt = 0.0;
    dgeqrf_(&m, &k, A, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) return static_cast<int>(info);
    lwork = static_cast<gcta_blas_int>(wkopt);
    Eigen::VectorXd work(lwork);
    dgeqrf_(&m, &k, A, &lda, tau, work.data(), &lwork, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, k, A, lda, tau));
#endif
}

// Overwrite A (post-dgeqrf) with the explicit m x k orthonormal Q factor,
// in place, from the reflectors produced by gcta_dgeqrf. Destroys R — call
// after extracting R if the caller needs it.
inline int gcta_dorgqr(gcta_blas_int m, gcta_blas_int k, double* A, gcta_blas_int lda, const double* tau) {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int lwork = -1, info = 0;
    double wkopt = 0.0;
    dorgqr_(&m, &k, &k, A, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) return static_cast<int>(info);
    lwork = static_cast<gcta_blas_int>(wkopt);
    Eigen::VectorXd work(lwork);
    dorgqr_(&m, &k, &k, A, &lda, tau, work.data(), &lwork, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, k, k, A, lda, tau));
#endif
}

// Convenience fusion for callers that only need Q (R discarded) — the
// power-iteration QR passes in symmetric_eigendecomp.hpp never use R.
inline int gcta_qr_thin_Q(gcta_blas_int m, gcta_blas_int k, double* A, gcta_blas_int lda) {
    Eigen::VectorXd tau(k);
    int info = gcta_dgeqrf(m, k, A, lda, tau.data());
    if (info != 0) return info;
    return gcta_dorgqr(m, k, A, lda, tau.data());
}

// CholeskyQR2: forms the m x k orthonormal factor Q of Y in place using two
// rounds of DSYRK + DPOTRF + DTRSM. Unlike gcta_qr_thin_Q (dgeqrf/dorgqr),
// whose Householder panel factorization is Level-2-BLAS-bound and stops
// scaling with thread count once k gets large (e.g. k in the thousands for
// rSVD on n ~ 1e5), every FLOP here goes through a Level-3 BLAS kernel that
// OpenBLAS/MKL/AOCL/Accelerate all parallelize across the full thread pool.
// Two Cholesky-QR passes restore orthogonality to working precision as long
// as Y isn't extremely ill-conditioned. A pass-0 DPOTRF failure falls back to
// the LAPACK QR path; a pass-1 failure just keeps pass-0's already-valid Q
// (falling through to gcta_qr_thin_Q there would run the expensive Householder
// path on data that's already orthonormal to working precision).
// `workspace`, if supplied, is reused for the k x k Gram matrix across
// repeated calls (e.g. power_iterate_and_project's loop) instead of
// reallocating it every call; resize() is a no-op once it's the right size.
inline int gcta_cholesky_qr_thin_Q(gcta_blas_int m, gcta_blas_int k, double* Y, gcta_blas_int ldY,
                                    Eigen::MatrixXd* workspace = nullptr) {
    Eigen::MatrixXd local_G;
    Eigen::MatrixXd& G = workspace ? *workspace : local_G;
    G.resize(k, k);
    for (int pass = 0; pass < 2; ++pass) {
        cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, k, m, 1.0, Y, ldY, 0.0, G.data(), k);
        //const double anorm = (pass == 0) ? gcta_dlansy_one(k, G.data(), k) : 0.0;
        const int info = gcta_dpotrf(k, G.data(), k);
        if (info != 0) {
            if (pass != 0) break;  // pass 0's Q already orthonormal; skip the refinement.
            return gcta_qr_thin_Q(m, k, Y, ldY);
        }
        /*
        if (pass == 0) {
            double rcond = 0.0;
            gcta_dpocon(k, G.data(), k, anorm, &rcond);
            const double kappa_Y = (rcond > 0.0) ? std::sqrt(1.0 / rcond) : std::numeric_limits<double>::infinity();
            LOGGER.i(0, "gcta_cholesky_qr_thin_Q: k=" + std::to_string(k) + " kappa(Y)=" + std::to_string(kappa_Y));
        }
        */
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    m, k, 1.0, G.data(), k, Y, ldY);
    }
    return 0;
}

// Same as gcta_cholesky_qr_thin_Q, but also returns the composed upper
// triangular factor R (k x k, column-major, leading dimension ldR) such that
// the original Y equals Q*R. Used by callers (Nystrom) that need R for a
// follow-on small SVD but, unlike tall_skinny_thin_svd, can't afford an extra
// n x k copy of Y to recover it afterwards as Q^T*Y_original. Falls back to
// gcta_qr_thin_Q's dgeqrf/dorgqr (R read off its upper triangle) if the first
// DPOTRF fails; a rare second-pass failure just keeps the first pass's
// already-valid Q/R and skips the refinement.
inline int gcta_cholesky_qr_thin_QR(gcta_blas_int m, gcta_blas_int k, double* Y, gcta_blas_int ldY,
                                     double* R, gcta_blas_int ldR, Eigen::MatrixXd* workspace = nullptr) {
    Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> Rmat(R, k, k, Eigen::OuterStride<>(ldR));
    Eigen::MatrixXd local_G;
    Eigen::MatrixXd& G = workspace ? *workspace : local_G;
    G.resize(k, k);
    for (int pass = 0; pass < 2; ++pass) {
        cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, k, m, 1.0, Y, ldY, 0.0, G.data(), k);
        //const double anorm = (pass == 0) ? gcta_dlansy_one(k, G.data(), k) : 0.0;
        const int info = gcta_dpotrf(k, G.data(), k);
        if (info != 0) {
            if (pass != 0) break;  // pass 0's Q/R already valid; skip the refinement.
            Eigen::VectorXd tau(k);
            const int info_qr = gcta_dgeqrf(m, k, Y, ldY, tau.data());
            if (info_qr != 0) return info_qr;
            Rmat = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>(Y, k, k, Eigen::OuterStride<>(ldY))
                       .triangularView<Eigen::Upper>();
            return gcta_dorgqr(m, k, Y, ldY, tau.data());
        }
        /*
        if (pass == 0) {
            double rcond = 0.0;
            gcta_dpocon(k, G.data(), k, anorm, &rcond);
            const double kappa_Y = (rcond > 0.0) ? std::sqrt(1.0 / rcond) : std::numeric_limits<double>::infinity();
            LOGGER.i(0, "gcta_cholesky_qr_thin_QR: k=" + std::to_string(k) + " kappa(Y)=" + std::to_string(kappa_Y));
        }
        */
        const Eigen::MatrixXd R_pass = Eigen::MatrixXd(G.triangularView<Eigen::Lower>()).transpose();
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    m, k, 1.0, G.data(), k, Y, ldY);
        Rmat = (pass == 0) ? R_pass : Eigen::MatrixXd(R_pass * Rmat);
    }
    return 0;
}

// Scalar-generic QR factor/form-Q, dispatched at compile time so
// symmetric_eigendecomp.hpp doesn't need an #ifdef per call site — the
// SVDScalar typedef there picks the specialization.
template <typename Scalar>
inline int gcta_geqrf(gcta_blas_int m, gcta_blas_int k, Scalar* A, gcta_blas_int lda, Scalar* tau);

template <>
inline int gcta_geqrf<double>(gcta_blas_int m, gcta_blas_int k, double* A, gcta_blas_int lda, double* tau) {
    return gcta_dgeqrf(m, k, A, lda, tau);
}

template <>
inline int gcta_geqrf<float>(gcta_blas_int m, gcta_blas_int k, float* A, gcta_blas_int lda, float* tau) {
    return gcta_sgeqrf(m, k, A, lda, tau);
}

template <typename Scalar>
inline int gcta_orgqr(gcta_blas_int m, gcta_blas_int k, Scalar* A, gcta_blas_int lda, const Scalar* tau);

template <>
inline int gcta_orgqr<double>(gcta_blas_int m, gcta_blas_int k, double* A, gcta_blas_int lda, const double* tau) {
    return gcta_dorgqr(m, k, A, lda, tau);
}

template <>
inline int gcta_orgqr<float>(gcta_blas_int m, gcta_blas_int k, float* A, gcta_blas_int lda, const float* tau) {
    return gcta_sorgqr(m, k, A, lda, tau);
}

template <typename Scalar>
inline int gcta_qr_thin_Q(gcta_blas_int m, gcta_blas_int k, Scalar* A, gcta_blas_int lda) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tau(k);
    int info = gcta_geqrf<Scalar>(m, k, A, lda, tau.data());
    if (info != 0) return info;
    return gcta_orgqr<Scalar>(m, k, A, lda, tau.data());
}


#endif  //END GCTA_CPU_H
