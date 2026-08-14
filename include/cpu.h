#ifndef GCTA_CPU_H
#define GCTA_CPU_H

#include <vector>

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

#endif  //END GCTA_CPU_H
