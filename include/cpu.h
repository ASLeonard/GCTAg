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
  #if defined(GCTA_USE_AOCL_DLP)
    #if __has_include(<aocl_dlp.h>)
      #include <aocl_dlp.h>
    #elif __has_include(<classic/aocl_gemm_interface_apis.h>)
      #include <classic/aocl_gemm_interface_apis.h>
    #endif
  #elif __has_include(<blis.h>)
    #include <blis.h>
  #endif
  typedef lapack_int gcta_blas_int;

#elif defined(GCTA_USE_ACCELERATE)
  // Include Accelerate's modern CBLAS and LAPACK headers directly rather than
  // the <Accelerate/Accelerate.h> umbrella, which pulls in <vImage/vImage.h> ->
  // <CarbonCore/AIFF.h> (defining global `struct Marker`, colliding with
  // include/Marker.h) and <vecLib/blas_new.h> (conflicting with Eigen's blas.h).
    #include <vecLib/cblas_new.h>
  #include <vecLib/lapack.h>
  typedef __LAPACK_int gcta_blas_int;
#endif
#include <Eigen/Core>

// Portable dpotrf: Cholesky factorization of a symmetric positive-definite matrix
// (defaults to lower triangle, column-major). Returns 0 on success, non-zero on failure.
inline int gcta_dpotrf(gcta_blas_int n, double* a, gcta_blas_int lda, char uplo = 'L') {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int info = 0;
    dpotrf_(&uplo, &n, a, &lda, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, n, a, lda));
#endif
}

// Portable dpotri: in-place inversion of a Cholesky-factored symmetric positive-definite
// matrix. Returns 0 on success, non-zero on failure.
inline int gcta_dpotri(gcta_blas_int n, double* a, gcta_blas_int lda, char uplo = 'L') {
#if defined(GCTA_USE_ACCELERATE)
    // Accelerate exposes Fortran-ABI dpotri_ with pointer arguments.
    gcta_blas_int info = 0;
    dpotri_(&uplo, &n, a, &lda, &info);
    return static_cast<int>(info);
#else
    // MKL, OpenBLAS, and AOCL all provide the LAPACKE C interface.
    return static_cast<int>(LAPACKE_dpotri(LAPACK_COL_MAJOR, uplo, n, a, lda));
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
    // catch it. MKL (either as the primary backend or paired via
    // GCTA_LAPACKE_LIBRARY) promotes internally and doesn't hit this, so it's
    // exempted. -1 mirrors LAPACK's own "argument 1 (n) invalid" convention;
    // safe to reuse here since a genuine n<0 from LAPACK can't otherwise occur.
#if !defined(GCTA_USE_MKL) && !defined(GCTA_LAPACKE_IS_MKL)
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

// Convenience fusion for callers that only need Q (R discarded) — the
// power-iteration QR passes in symmetric_eigendecomp.hpp never use R.
inline int gcta_qr_thin_Q(gcta_blas_int m, gcta_blas_int k, double* A, gcta_blas_int lda) {
    Eigen::VectorXd tau(k);
    int info = gcta_dgeqrf(m, k, A, lda, tau.data());
    if (info != 0) return info;
    return gcta_dorgqr(m, k, A, lda, tau.data());
}

inline int gcta_qr_thin_Q(gcta_blas_int m, gcta_blas_int k, float* A, gcta_blas_int lda) {
    Eigen::VectorXf tau(k);
    int info = gcta_sgeqrf(m, k, A, lda, tau.data());
    if (info != 0) return info;
    return gcta_sorgqr(m, k, A, lda, tau.data());
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
        const int info = gcta_dpotrf(k, G.data(), k);
        if (info != 0) {
            if (pass != 0) break;  // pass 0's Q already orthonormal; skip the refinement.
            return gcta_qr_thin_Q(m, k, Y, ldY);
        }
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    m, k, 1.0, G.data(), k, Y, ldY);
    }
    return 0;
}

// Mixed-precision CholeskyQR2 for float32 m x k subspace matrices:
// O(m * k^2) SYRK and TRSM run in float32 (cblas_ssyrk / cblas_strsm), while
// the tiny k x k Gram Cholesky factorization runs in double (gcta_dpotrf) to
// avoid premature single-precision breakdown on squared condition numbers.
inline int gcta_cholesky_qr_thin_Q(gcta_blas_int m, gcta_blas_int k, float* Y, gcta_blas_int ldY,
                                    Eigen::MatrixXd* workspace = nullptr) {
    Eigen::MatrixXd local_G;
    Eigen::MatrixXd& G = workspace ? *workspace : local_G;
    G.resize(k, k);
    Eigen::MatrixXf G_f(k, k);
    for (int pass = 0; pass < 2; ++pass) {
        cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans, k, m, 1.0f, Y, ldY, 0.0f, G_f.data(), k);
        G.triangularView<Eigen::Lower>() = G_f.cast<double>();
        const int info = gcta_dpotrf(k, G.data(), k);
        if (info != 0) {
            if (pass != 0) break;  // pass 0's Q already orthonormal; skip the refinement.
            return gcta_qr_thin_Q(m, k, Y, ldY);
        }
        G_f.triangularView<Eigen::Lower>() = G.cast<float>();
        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    m, k, 1.0f, G_f.data(), k, Y, ldY);
    }
    return 0;
}

// Same as gcta_cholesky_qr_thin_Q, but also returns the composed upper
// triangular factor R (k x k, column-major, leading dimension ldR) such that
// the original Y equals Q*R. Used by callers (Nystrom) that need R for a
// follow-on small eigendecomposition and can't afford an extra
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
        const Eigen::MatrixXd R_pass = Eigen::MatrixXd(G.triangularView<Eigen::Lower>()).transpose();
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    m, k, 1.0, G.data(), k, Y, ldY);
        Rmat = (pass == 0) ? R_pass : Eigen::MatrixXd(R_pass * Rmat);
    }
    return 0;
}

inline int gcta_cholesky_qr_thin_QR(gcta_blas_int m, gcta_blas_int k, float* Y, gcta_blas_int ldY,
                                     double* R, gcta_blas_int ldR, Eigen::MatrixXd* workspace = nullptr) {
    Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> Rmat(R, k, k, Eigen::OuterStride<>(ldR));
    Eigen::MatrixXd local_G;
    Eigen::MatrixXd& G = workspace ? *workspace : local_G;
    G.resize(k, k);
    Eigen::MatrixXf G_f(k, k);
    for (int pass = 0; pass < 2; ++pass) {
        cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans, k, m, 1.0f, Y, ldY, 0.0f, G_f.data(), k);
        G.triangularView<Eigen::Lower>() = G_f.cast<double>();
        const int info = gcta_dpotrf(k, G.data(), k);
        if (info != 0) {
            if (pass != 0) break;  // pass 0's Q/R already valid; skip the refinement.
            Eigen::VectorXf tau(k);
            const int info_qr = gcta_sgeqrf(m, k, Y, ldY, tau.data());
            if (info_qr != 0) return info_qr;
            Rmat = Eigen::Map<Eigen::MatrixXf, 0, Eigen::OuterStride<>>(Y, k, k, Eigen::OuterStride<>(ldY))
                       .triangularView<Eigen::Upper>()
                       .toDenseMatrix()
                       .cast<double>();
            return gcta_sorgqr(m, k, Y, ldY, tau.data());
        }
        const Eigen::MatrixXd R_pass = Eigen::MatrixXd(G.triangularView<Eigen::Lower>()).transpose();
        G_f.triangularView<Eigen::Lower>() = G.cast<float>();
        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    m, k, 1.0f, G_f.data(), k, Y, ldY);
        Rmat = (pass == 0) ? R_pass : Eigen::MatrixXd(R_pass * Rmat);
    }
    return 0;
}

#if defined(GCTA_USE_AOCL_DLP) || (defined(GCTA_USE_AOCL) && defined(BLIS_ENABLE_ADDON_AOCL_GEMM))
  #define GCTA_HAS_AOCL_S8_GEMM 1
#else
  #define GCTA_HAS_AOCL_S8_GEMM 0
#endif

// Compute C(rows x cols, column-major, leading dim ldc) = A(rows x K, lda) * B(cols x K, ldb)^T
// where A and B are column-major int8_t matrices in {0, 1, 2} and C is a column-major int32_t matrix.
// On AOCL with BLIS_ENABLE_ADDON_AOCL_GEMM, dispatches directly to Zen 4 AVX-512 VNNI
// aocl_gemm_s8s8s32os32 via row-major transpose equivalence:
//   C_colmajor(rows x cols, ldc) == C_rowmajor(cols x rows, ldc) = B_rowmajor^T(cols x K) * A_rowmajor(K x rows).
// On other BLAS backends (where K <= 1024 and A, B in {0, 1, 2} so every dot product is <= 4096,
// well below float32's exact integer limit 2^24 = 16,777,216), converts the tile panels A and B
// to float32 scratch buffers and runs cblas_sgemm, yielding the exact same int32_t integers.
inline void gcta_gemm_s8s8s32_colmajor_nt(
    gcta_blas_int rows, gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A, gcta_blas_int lda,
    const int8_t* B, gcta_blas_int ldb,
    int32_t* C, gcta_blas_int ldc,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32)
{
    if (rows <= 0 || cols <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    (void)scratch_A_f32;
    (void)scratch_B_f32;
    (void)scratch_C_f32;
    aocl_gemm_s8s8s32os32('r', 't', 'n',
                          cols, rows, K,
                          1,
                          B, ldb, 'n',
                          A, lda, 'n',
                          0,
                          C, ldc,
                          nullptr);
#else
    const size_t size_A = static_cast<size_t>(rows) * static_cast<size_t>(K);
    if (scratch_A_f32.size() < size_A) scratch_A_f32.resize(size_A);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int k = 0; k < K; ++k) {
        const int8_t* src = A + static_cast<size_t>(k) * lda;
        float* dst = scratch_A_f32.data() + static_cast<size_t>(k) * rows;
        for (gcta_blas_int r = 0; r < rows; ++r) {
            dst[r] = static_cast<float>(src[r]);
        }
    }
    const size_t size_B = static_cast<size_t>(cols) * static_cast<size_t>(K);
    if (scratch_B_f32.size() < size_B) scratch_B_f32.resize(size_B);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int k = 0; k < K; ++k) {
        const int8_t* src = B + static_cast<size_t>(k) * ldb;
        float* dst = scratch_B_f32.data() + static_cast<size_t>(k) * cols;
        for (gcta_blas_int c = 0; c < cols; ++c) {
            dst[c] = static_cast<float>(src[c]);
        }
    }
    const size_t size_C = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (scratch_C_f32.size() < size_C) scratch_C_f32.resize(size_C);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                rows, cols, K,
                1.0f,
                scratch_A_f32.data(), rows,
                scratch_B_f32.data(), cols,
                0.0f,
                scratch_C_f32.data(), rows);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        const float* src = scratch_C_f32.data() + static_cast<size_t>(c) * rows;
        int32_t* dst = C + static_cast<size_t>(c) * ldc;
        for (gcta_blas_int r = 0; r < rows; ++r) {
            dst[r] = static_cast<int32_t>(src[r]);
        }
    }
#endif
}

// Compute C(rows x cols, column-major, leading dim ldc) =
//   beta * C + A(rows x K, lda) * B(K x cols, ldb)
// where A has signed int8_t values in [-127, 127], B has int8_t values in {0, 1, 2},
// and C is int32_t.
// On AOCL with BLIS_ENABLE_ADDON_AOCL_GEMM, dispatches directly to Zen 4 AVX-512 VNNI
// aocl_gemm_s8s8s32os32 via row-major transpose equivalence:
//   C_colmajor(rows x cols) == C_rowmajor(cols x rows) = B_col_as_row(cols x K) * A_col_as_row(K x rows).
// On other BLAS backends (for K <= 8192 where |dot| <= 2 * 127 * 8192 = 2,080,768 < 2^24),
// converts A and B to float32 scratch panels and executes cblas_sgemm, recovering exact integers.
inline void gcta_gemm_s8s8s32_colmajor_nn(
    gcta_blas_int rows, gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A, gcta_blas_int lda,
    const int8_t* B, gcta_blas_int ldb,
    int32_t* C, gcta_blas_int ldc,
    int32_t beta,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32)
{
    if (rows <= 0 || cols <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    (void)scratch_A_f32;
    (void)scratch_B_f32;
    (void)scratch_C_f32;
    aocl_gemm_s8s8s32os32('r', 'n', 'n',
                          cols, rows, K,
                          1,
                          B, ldb, 'n',
                          A, lda, 'n',
                          beta,
                          C, ldc,
                          nullptr);
#else
    const size_t size_A = static_cast<size_t>(rows) * static_cast<size_t>(K);
    if (scratch_A_f32.size() < size_A) scratch_A_f32.resize(size_A);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int k = 0; k < K; ++k) {
        const int8_t* src = A + static_cast<size_t>(k) * lda;
        float* dst = scratch_A_f32.data() + static_cast<size_t>(k) * rows;
        for (gcta_blas_int r = 0; r < rows; ++r) {
            dst[r] = static_cast<float>(src[r]);
        }
    }
    const size_t size_B = static_cast<size_t>(K) * static_cast<size_t>(cols);
    if (scratch_B_f32.size() < size_B) scratch_B_f32.resize(size_B);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        const int8_t* src = B + static_cast<size_t>(c) * ldb;
        float* dst = scratch_B_f32.data() + static_cast<size_t>(c) * K;
        for (gcta_blas_int k = 0; k < K; ++k) {
            dst[k] = static_cast<float>(src[k]);
        }
    }
    const size_t size_C = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (scratch_C_f32.size() < size_C) scratch_C_f32.resize(size_C);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                rows, cols, K,
                1.0f,
                scratch_A_f32.data(), rows,
                scratch_B_f32.data(), K,
                0.0f,
                scratch_C_f32.data(), rows);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        const float* src = scratch_C_f32.data() + static_cast<size_t>(c) * rows;
        int32_t* dst = C + static_cast<size_t>(c) * ldc;
        if (beta == 0) {
            for (gcta_blas_int r = 0; r < rows; ++r) {
                dst[r] = static_cast<int32_t>(std::lround(src[r]));
            }
        } else if (beta == 1) {
            for (gcta_blas_int r = 0; r < rows; ++r) {
                dst[r] += static_cast<int32_t>(std::lround(src[r]));
            }
        } else {
            for (gcta_blas_int r = 0; r < rows; ++r) {
                dst[r] = beta * dst[r] + static_cast<int32_t>(std::lround(src[r]));
            }
        }
    }
#endif
}

// Compute the lower triangle of C(cols x cols, column-major, leading dim ldc) = A(cols x K, lda) * A(cols x K, lda)^T.
inline void gcta_syrk_s8s8s32_lower_colmajor_nt(
    gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A, gcta_blas_int lda,
    int32_t* C, gcta_blas_int ldc,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_C_f32)
{
    if (cols <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    (void)scratch_A_f32;
    (void)scratch_C_f32;
    aocl_gemm_s8s8s32os32('r', 't', 'n',
                          cols, cols, K,
                          1,
                          A, lda, 'n',
                          A, lda, 'n',
                          0,
                          C, ldc,
                          nullptr);
#else
    const size_t size_A = static_cast<size_t>(cols) * static_cast<size_t>(K);
    if (scratch_A_f32.size() < size_A) scratch_A_f32.resize(size_A);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int k = 0; k < K; ++k) {
        const int8_t* src = A + static_cast<size_t>(k) * lda;
        float* dst = scratch_A_f32.data() + static_cast<size_t>(k) * cols;
        for (gcta_blas_int r = 0; r < cols; ++r) {
            dst[r] = static_cast<float>(src[r]);
        }
    }
    const size_t size_C = static_cast<size_t>(cols) * static_cast<size_t>(cols);
    if (scratch_C_f32.size() < size_C) scratch_C_f32.resize(size_C);
    cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans,
                cols, K,
                1.0f,
                scratch_A_f32.data(), cols,
                0.0f,
                scratch_C_f32.data(), cols);
    #pragma omp parallel for schedule(guided)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        const float* src = scratch_C_f32.data() + c + static_cast<size_t>(c) * cols;
        int32_t* dst = C + c + static_cast<size_t>(c) * ldc;
        const gcta_blas_int len = cols - c;
        for (gcta_blas_int r = 0; r < len; ++r) {
            dst[r] = static_cast<int32_t>(src[r]);
        }
    }
#endif
}

// Compute C(rows x cols, column-major, leading dim ldc) = A_kmajor(rows x K, ldK) * B_kmajor(cols x K, ldK)^T
// where A_kmajor and B_kmajor store K markers contiguously per sample (row-major with stride ldK >= K).
// Under AOCL VNNI (GCTA_HAS_AOCL_S8_GEMM):
//   C_colmajor(rows x cols, ldc) == C_rowmajor(cols x rows, ldc) = B_kmajor(cols x K) * A_kmajor(rows x K)^T.
// Eliminates large-sample-stride TLB/gather overhead during BLIS panel packing across tiles.
inline void gcta_gemm_s8s8s32_kmajor_nt(
    gcta_blas_int rows, gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    const int8_t* B_kmajor,
    int32_t* C, gcta_blas_int ldc,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32)
{
    if (rows <= 0 || cols <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    (void)scratch_A_f32;
    (void)scratch_B_f32;
    (void)scratch_C_f32;
    aocl_gemm_s8s8s32os32('r', 'n', 't',
                          cols, rows, K,
                          1,
                          B_kmajor, ldK, 'n',
                          A_kmajor, ldK, 'n',
                          0,
                          C, ldc,
                          nullptr);
#else
    const size_t size_A = static_cast<size_t>(rows) * static_cast<size_t>(K);
    if (scratch_A_f32.size() < size_A) scratch_A_f32.resize(size_A);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int r = 0; r < rows; ++r) {
        const int8_t* src = A_kmajor + static_cast<size_t>(r) * ldK;
        float* dst = scratch_A_f32.data() + static_cast<size_t>(r) * K;
        for (gcta_blas_int k = 0; k < K; ++k) {
            dst[k] = static_cast<float>(src[k]);
        }
    }
    const size_t size_B = static_cast<size_t>(cols) * static_cast<size_t>(K);
    if (scratch_B_f32.size() < size_B) scratch_B_f32.resize(size_B);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        const int8_t* src = B_kmajor + static_cast<size_t>(c) * ldK;
        float* dst = scratch_B_f32.data() + static_cast<size_t>(c) * K;
        for (gcta_blas_int k = 0; k < K; ++k) {
            dst[k] = static_cast<float>(src[k]);
        }
    }
    const size_t size_C = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (scratch_C_f32.size() < size_C) scratch_C_f32.resize(size_C);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                rows, cols, K,
                1.0f,
                scratch_A_f32.data(), K,
                scratch_B_f32.data(), K,
                0.0f,
                scratch_C_f32.data(), rows);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        const float* src = scratch_C_f32.data() + static_cast<size_t>(c) * rows;
        int32_t* dst = C + static_cast<size_t>(c) * ldc;
        for (gcta_blas_int r = 0; r < rows; ++r) {
            dst[r] = static_cast<int32_t>(src[r]);
        }
    }
#endif
}

// Compute the lower triangle of C(cols x cols, column-major, leading dim ldc) =
//   A_kmajor(cols x K, ldK) * A_kmajor(cols x K, ldK)^T.
inline void gcta_syrk_s8s8s32_lower_kmajor_nt(
    gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    int32_t* C, gcta_blas_int ldc,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_C_f32)
{
    if (cols <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    (void)scratch_A_f32;
    (void)scratch_C_f32;
    aocl_gemm_s8s8s32os32('r', 'n', 't',
                          cols, cols, K,
                          1,
                          A_kmajor, ldK, 'n',
                          A_kmajor, ldK, 'n',
                          0,
                          C, ldc,
                          nullptr);
#else
    const size_t size_A = static_cast<size_t>(cols) * static_cast<size_t>(K);
    if (scratch_A_f32.size() < size_A) scratch_A_f32.resize(size_A);
    #pragma omp parallel for schedule(static)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        const int8_t* src = A_kmajor + static_cast<size_t>(c) * ldK;
        float* dst = scratch_A_f32.data() + static_cast<size_t>(c) * K;
        for (gcta_blas_int k = 0; k < K; ++k) {
            dst[k] = static_cast<float>(src[k]);
        }
    }
    const size_t size_C = static_cast<size_t>(cols) * static_cast<size_t>(cols);
    if (scratch_C_f32.size() < size_C) scratch_C_f32.resize(size_C);
    cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans,
                cols, K,
                1.0f,
                scratch_A_f32.data(), K,
                0.0f,
                scratch_C_f32.data(), cols);
    #pragma omp parallel for schedule(guided)
    for (gcta_blas_int c = 0; c < cols; ++c) {
        const float* src = scratch_C_f32.data() + c + static_cast<size_t>(c) * cols;
        int32_t* dst = C + c + static_cast<size_t>(c) * ldc;
        const gcta_blas_int len = cols - c;
        for (gcta_blas_int r = 0; r < len; ++r) {
            dst[r] = static_cast<int32_t>(src[r]);
        }
    }
#endif
}

#endif  //END GCTA_CPU_H
