#ifndef GCTA_CPU_H
#define GCTA_CPU_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
  #include <dlfcn.h>
#endif
#ifdef _OPENMP
  #include <omp.h>
#endif
#include "Logger.h"

#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC)) || defined(__amd64)
  #define GCTA_CPU_x86 1
  #include <immintrin.h>
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

// Portable dtrtri: in-place inversion of a triangular matrix
// (defaults to upper triangle, non-unit diagonal, column-major).
// Returns 0 on success, non-zero on failure.
inline int gcta_dtrtri(gcta_blas_int n, double* a, gcta_blas_int lda, char uplo = 'U', char diag = 'N') {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int info = 0;
    dtrtri_(&uplo, &diag, &n, a, &lda, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dtrtri(LAPACK_COL_MAJOR, uplo, diag, n, a, lda));
#endif
}

// Portable dpstrf: Cholesky factorization with complete pivoting of a symmetric
// positive semidefinite matrix (P^T A P = U^T U when uplo='U', piv is 1-based).
// Returns 0 if full rank, >0 if stopped at rank < n (valid rank-`rank` factor), <0 on bad arg.
inline int gcta_dpstrf(gcta_blas_int n, double* a, gcta_blas_int lda,
                       gcta_blas_int* piv, gcta_blas_int* rank,
                       double tol = -1.0, char uplo = 'U') {
#if defined(GCTA_USE_ACCELERATE)
    gcta_blas_int info = 0;
    std::vector<double> work(static_cast<size_t>(std::max<gcta_blas_int>(1, 2 * n)));
    dpstrf_(&uplo, &n, a, &lda, piv, rank, &tol, work.data(), &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(LAPACKE_dpstrf(LAPACK_COL_MAJOR, uplo, n, a, lda, piv, rank, tol));
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

// ---- gcta_ssyevd -------------------------------------------------------
// Single-precision symmetric eigensolver (divide-and-conquer), col-major, upper.
inline int gcta_ssyevd(gcta_blas_int n, float* a, gcta_blas_int lda, float* w)
{
#if defined(GCTA_USE_ACCELERATE)
    const char    jobz = 'V', uplo = 'U';
    gcta_blas_int info = 0;
    gcta_blas_int lwork_q = -1, liwork_q = -1;
    float         work_sz = 0.0f;
    gcta_blas_int iwork_sz = 0;
    ssyevd_(&jobz, &uplo, &n, a, &lda, w,
            &work_sz, &lwork_q, &iwork_sz, &liwork_q, &info);
    if (info != 0) return static_cast<int>(info);
    gcta_blas_int lwork  = static_cast<gcta_blas_int>(work_sz);
    gcta_blas_int liwork = iwork_sz;
    std::vector<float>         work(lwork);
    std::vector<gcta_blas_int> iwork(liwork);
    ssyevd_(&jobz, &uplo, &n, a, &lda, w,
            work.data(), &lwork, iwork.data(), &liwork, &info);
    return static_cast<int>(info);
#else
#if !defined(GCTA_USE_MKL) && !defined(GCTA_LAPACKE_IS_MKL)
    if (n >= 32766) return -1;
#endif
    return static_cast<int>(
        LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'V', 'U', n, a, lda, w));
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

inline int gcta_ssyevr(gcta_blas_int n, float* a, gcta_blas_int lda,
                       gcta_blas_int il, gcta_blas_int iu,
                       gcta_blas_int* m_found,
                       float* w, float* z, gcta_blas_int ldz,
                       gcta_blas_int* isuppz)
{
#if defined(GCTA_USE_ACCELERATE)
    const char   jobz = 'V', range = 'I', uplo = 'U';
    const float  vl = 0.0f, vu = 0.0f, abstol = 0.0f;
    gcta_blas_int info = 0;
    gcta_blas_int lwork_q = -1, liwork_q = -1;
    float         work_sz = 0.0f;
    gcta_blas_int iwork_sz = 0;
    ssyevr_(&jobz, &range, &uplo, &n, a, &lda,
            &vl, &vu, &il, &iu, &abstol,
            m_found, w, z, &ldz, isuppz,
            &work_sz, &lwork_q, &iwork_sz, &liwork_q, &info);
    if (info != 0) return static_cast<int>(info);
    gcta_blas_int lwork  = static_cast<gcta_blas_int>(work_sz);
    gcta_blas_int liwork = iwork_sz;
    std::vector<float>         work(lwork);
    std::vector<gcta_blas_int> iwork(liwork);
    ssyevr_(&jobz, &range, &uplo, &n, a, &lda,
            &vl, &vu, &il, &iu, &abstol,
            m_found, w, z, &ldz, isuppz,
            work.data(), &lwork, iwork.data(), &liwork, &info);
    return static_cast<int>(info);
#else
    return static_cast<int>(
        LAPACKE_ssyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', n, a, lda,
                       0.0f, 0.0f, il, iu, 0.0f,
                       m_found, w, z, ldz, isuppz));
#endif
}

// Temporarily caps OpenMP + BLAS/LAPACK worker threads for a scoped section
// (e.g. dsyevd(20128) on 2-socket EPYC to prevent cross-socket dsymv regression)
// and restores the exact prior settings on destruction.
struct ScopedBlasThreadCap {
    int prev_omp_ = 0;
    int64_t prev_blis_ = 0;
    int prev_mkl_ = 0;
    int prev_ob_ = 0;
    using GetI64Fn = int64_t(*)();
    using SetI64Fn = void(*)(int64_t);
    using MklLocalFn = int(*)(int);
    using GetIntFn = int(*)();
    using SetIntFn = void(*)(int);
    SetI64Fn set_blis_ = nullptr;
    MklLocalFn set_mkl_ = nullptr;
    SetIntFn set_ob_ = nullptr;
    bool active_ = false;

    explicit ScopedBlasThreadCap(int cap) noexcept {
#ifdef _OPENMP
        if (cap <= 0) return;
        prev_omp_ = omp_get_max_threads();
        if (cap >= prev_omp_) return;
        active_ = true;
        omp_set_num_threads(cap);
#if defined(__unix__) || defined(__APPLE__)
        if (void* self = dlopen(nullptr, RTLD_LAZY)) {
            auto get_blis = reinterpret_cast<GetI64Fn>(dlsym(self, "bli_thread_get_num_threads"));
            set_blis_ = reinterpret_cast<SetI64Fn>(dlsym(self, "bli_thread_set_num_threads"));
            if (set_blis_) {
                prev_blis_ = get_blis ? get_blis() : 0;
                set_blis_(static_cast<int64_t>(cap));
            }
            // In libmkl_rt.so, PascalCase MKL_Set_Num_Threads_Local(int) is the C
            // pass-by-value ABI, whereas lowercase mkl_set_num_threads_local is the
            // Fortran pass-by-reference wrapper (passing an int by value segfaults).
            set_mkl_ = reinterpret_cast<MklLocalFn>(dlsym(self, "MKL_Set_Num_Threads_Local"));
            if (set_mkl_) {
                prev_mkl_ = set_mkl_(cap);
            }
            auto get_ob = reinterpret_cast<GetIntFn>(dlsym(self, "openblas_get_num_threads"));
            set_ob_ = reinterpret_cast<SetIntFn>(dlsym(self, "openblas_set_num_threads"));
            if (set_ob_) {
                prev_ob_ = get_ob ? get_ob() : 0;
                set_ob_(cap);
            }
            dlclose(self);
        }
#endif
#else
        (void)cap;
#endif
    }

    ~ScopedBlasThreadCap() noexcept {
#ifdef _OPENMP
        if (!active_) return;
        if (set_blis_) {
            set_blis_(prev_blis_ > 0 ? prev_blis_ : static_cast<int64_t>(prev_omp_));
        }
        if (set_mkl_) {
            set_mkl_(prev_mkl_);
        }
        if (set_ob_) {
            set_ob_(prev_ob_ > 0 ? prev_ob_ : prev_omp_);
        }
        omp_set_num_threads(prev_omp_);
#endif
    }

    ScopedBlasThreadCap(const ScopedBlasThreadCap&) = delete;
    ScopedBlasThreadCap& operator=(const ScopedBlasThreadCap&) = delete;
};

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

namespace gcta_cpu_detail {

// Parallelized lower-triangle cast float -> double (k x k, col-major, ld = k).
inline void copy_lower_f2d(gcta_blas_int k, const float* src, double* dst) noexcept {
    #pragma omp parallel for schedule(guided) if (k >= 256)
    for (gcta_blas_int j = 0; j < k; ++j) {
        const float* s = src + static_cast<size_t>(j) * k + j;
        double* d = dst + static_cast<size_t>(j) * k + j;
        const gcta_blas_int len = k - j;
        for (gcta_blas_int i = 0; i < len; ++i) {
            d[i] = static_cast<double>(s[i]);
        }
    }
}

// Parallelized lower-triangle cast double -> float (k x k, col-major, ld = k).
inline void copy_lower_d2f(gcta_blas_int k, const double* src, float* dst) noexcept {
    #pragma omp parallel for schedule(guided) if (k >= 256)
    for (gcta_blas_int j = 0; j < k; ++j) {
        const double* s = src + static_cast<size_t>(j) * k + j;
        float* d = dst + static_cast<size_t>(j) * k + j;
        const gcta_blas_int len = k - j;
        for (gcta_blas_int i = 0; i < len; ++i) {
            d[i] = static_cast<float>(s[i]);
        }
    }
}

// Write upper-triangular R (k x k, col-major, ldR) = G_lower^T and zero strict lower triangle.
inline void copy_lower_transpose_to_upper(
    gcta_blas_int k, const double* G_lower, double* R, gcta_blas_int ldR) noexcept
{
    #pragma omp parallel for schedule(static) if (k >= 256)
    for (gcta_blas_int j = 0; j < k; ++j) {
        double* dst_col = R + static_cast<size_t>(j) * ldR;
        for (gcta_blas_int i = 0; i <= j; ++i) {
            dst_col[i] = G_lower[j + static_cast<size_t>(i) * k];
        }
        for (gcta_blas_int i = j + 1; i < k; ++i) {
            dst_col[i] = 0.0;
        }
    }
}

// Extract upper triangle of float A (post-sgeqrf, col-major, ldA) into double R (ldR)
// with strict lower triangle zeroed.
inline void extract_upper_f2d(
    gcta_blas_int k, const float* A, gcta_blas_int ldA, double* R, gcta_blas_int ldR) noexcept
{
    #pragma omp parallel for schedule(static) if (k >= 256)
    for (gcta_blas_int j = 0; j < k; ++j) {
        const float* src_col = A + static_cast<size_t>(j) * ldA;
        double* dst_col = R + static_cast<size_t>(j) * ldR;
        for (gcta_blas_int i = 0; i <= j; ++i) {
            dst_col[i] = static_cast<double>(src_col[i]);
        }
        for (gcta_blas_int i = j + 1; i < k; ++i) {
            dst_col[i] = 0.0;
        }
    }
}

// Shifted Cholesky recovery (Fukaya et al. 2020): attempts dpotrf(G_f + s*I)
// with up to 3 escalating diagonal shifts s before giving up to Householder QR.
inline int try_shifted_dpotrf(
    gcta_blas_int m, gcta_blas_int k, const float* G_f, double* G) noexcept
{
    double trace_G = 0.0;
    double max_diag = 0.0;
    for (gcta_blas_int j = 0; j < k; ++j) {
        const double d = static_cast<double>(G_f[j + static_cast<size_t>(j) * k]);
        if (!std::isfinite(d)) return -1;
        if (d > 0.0) trace_G += d;
        if (d > max_diag) max_diag = d;
    }
    if (trace_G <= 0.0 || max_diag <= 0.0 || !std::isfinite(trace_G)) {
        return -1;
    }

    const double eps_f32 = static_cast<double>(std::numeric_limits<float>::epsilon());
    double s = std::max(11.0 * eps_f32 * trace_G,
                        11.0 * std::sqrt(static_cast<double>(m)) * eps_f32 * max_diag);
    const double s_cap = 0.5 * trace_G;

    for (int attempt = 0; attempt < 3; ++attempt) {
        const double cur_s = std::min(s, s_cap);
        copy_lower_f2d(k, G_f, G);
        for (gcta_blas_int j = 0; j < k; ++j) {
            G[j + static_cast<size_t>(j) * k] += cur_s;
        }
        if (gcta_dpotrf(k, G, k) == 0) {
            return 0;
        }
        if (cur_s >= s_cap) {
            break;
        }
        s *= 16.0;
    }
    return -1;
}

}  // namespace gcta_cpu_detail

struct CholeskyQrStats {
    int calls = 0;
    int passes = 0;
    int shifted = 0;
    int householder = 0;
};

enum class CholeskyQrMode : uint8_t { Auto = 0, Sketched = 1, Standard = 2 };

// Parallel row-slabbed in-place triangular right multiply:
//   Y <- Y * T   (Y is m x k ColMajor, ldY >= m; T is k x k ColMajor, ldT >= k)
// Strict off-triangle entries of T MUST already be zeroed:
//   T_is_upper == true  => T(i, j) == 0 for i > j (reverse column-block sweep)
//   T_is_upper == false => T(i, j) == 0 for i < j (forward column-block sweep)
// Uses 4096-row x 2048-col cblas_sgemm blocks so AOCL-BLIS parallelizes both
// JC and IC loops across all 192 cores instead of serializing cblas_strmm/strsm(Right).
inline void gcta_right_tri_sgemm_inplace(
    gcta_blas_int m, gcta_blas_int k,
    float* Y, gcta_blas_int ldY,
    const float* T, gcta_blas_int ldT,
    bool T_is_upper)
{
    if (m <= 0 || k <= 0) return;
    if (k < 512) {
        cblas_strmm(CblasColMajor, CblasRight,
                    T_is_upper ? CblasUpper : CblasLower,
                    CblasNoTrans, CblasNonUnit,
                    m, k, 1.0f, T, ldT, Y, ldY);
        return;
    }

    constexpr gcta_blas_int kSlabM   = 4096;
    constexpr gcta_blas_int kTriBlk  = 2048;
    const gcta_blas_int max_rb = std::min(m, kSlabM);
    const gcta_blas_int max_cb = std::min(k, kTriBlk);
    std::vector<float> tile_out(static_cast<size_t>(max_rb) * static_cast<size_t>(max_cb));
    const gcta_blas_int nb = (k + kTriBlk - 1) / kTriBlk;

    for (gcta_blas_int r0 = 0; r0 < m; r0 += kSlabM) {
        const gcta_blas_int rb = std::min(kSlabM, m - r0);
        if (T_is_upper) {
            for (gcta_blas_int cb = nb - 1; cb >= 0; --cb) {
                const gcta_blas_int c0   = cb * kTriBlk;
                const gcta_blas_int clen = std::min(kTriBlk, k - c0);
                const gcta_blas_int k_len = c0 + clen;
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            rb, clen, k_len,
                            1.0f,
                            Y + r0, ldY,
                            T + static_cast<size_t>(c0) * static_cast<size_t>(ldT), ldT,
                            0.0f, tile_out.data(), rb);
                #pragma omp parallel for schedule(static) if (static_cast<size_t>(rb) * static_cast<size_t>(clen) >= 32768)
                for (gcta_blas_int j = 0; j < clen; ++j) {
                    std::memcpy(Y + r0 + static_cast<size_t>(c0 + j) * static_cast<size_t>(ldY),
                                tile_out.data() + static_cast<size_t>(j) * static_cast<size_t>(rb),
                                static_cast<size_t>(rb) * sizeof(float));
                }
            }
        } else {
            for (gcta_blas_int cb = 0; cb < nb; ++cb) {
                const gcta_blas_int c0   = cb * kTriBlk;
                const gcta_blas_int clen = std::min(kTriBlk, k - c0);
                const gcta_blas_int k_len = k - c0;
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            rb, clen, k_len,
                            1.0f,
                            Y + r0 + static_cast<size_t>(c0) * static_cast<size_t>(ldY), ldY,
                            T + c0 + static_cast<size_t>(c0) * static_cast<size_t>(ldT), ldT,
                            0.0f, tile_out.data(), rb);
                #pragma omp parallel for schedule(static) if (static_cast<size_t>(rb) * static_cast<size_t>(clen) >= 32768)
                for (gcta_blas_int j = 0; j < clen; ++j) {
                    std::memcpy(Y + r0 + static_cast<size_t>(c0 + j) * static_cast<size_t>(ldY),
                                tile_out.data() + static_cast<size_t>(j) * static_cast<size_t>(rb),
                                static_cast<size_t>(rb) * sizeof(float));
                }
            }
        }
    }
}

namespace gcta_cpu_detail {

// Apply Y <- Y * L^{-T} for lower Cholesky factor G_lower (k x k, col-major, ld=k).
// Truncates G_lower to float32 and solves directly via Level-3 cblas_strsm(Right).
inline bool apply_lower_chol_inv_right(
    gcta_blas_int m, gcta_blas_int k,
    double* G_lower, float* G_f,
    float* Y, gcta_blas_int ldY)
{
    copy_lower_d2f(k, G_lower, G_f);
    cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                m, k, 1.0f, G_f, k, Y, ldY);
    return true;
}

inline uint64_t osnap_splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Column-parallel partitioned OSNAP sparse sign sketch W = S * Y (s x k, d = 4 nonzeros/row).
inline void build_osnap_sketch(
    gcta_blas_int m, gcta_blas_int k, gcta_blas_int s,
    const float* Y, gcta_blas_int ldY,
    float* W, gcta_blas_int ldW)
{
    const uint32_t part_sz = static_cast<uint32_t>(s / 4);
    std::vector<uint32_t> tbl(static_cast<size_t>(m) * 4ULL);
    #pragma omp parallel for schedule(static) if (m >= 8192)
    for (gcta_blas_int i = 0; i < m; ++i) {
        const uint64_t r0 = osnap_splitmix64(static_cast<uint64_t>(i) * 2ULL);
        const uint64_t r1 = osnap_splitmix64(static_cast<uint64_t>(i) * 2ULL + 1ULL);
        uint32_t* dst = tbl.data() + static_cast<size_t>(i) * 4ULL;
        const uint32_t i0 = 0U * part_sz + static_cast<uint32_t>(static_cast<uint32_t>(r0) % part_sz);
        const uint32_t i1 = 1U * part_sz + static_cast<uint32_t>(static_cast<uint32_t>(r0 >> 32) % part_sz);
        const uint32_t i2 = 2U * part_sz + static_cast<uint32_t>(static_cast<uint32_t>(r1) % part_sz);
        const uint32_t i3 = 3U * part_sz + static_cast<uint32_t>(static_cast<uint32_t>(r1 >> 32) % part_sz);
        dst[0] = i0 | (static_cast<uint32_t>((r0 >> 30) & 1ULL) << 31);
        dst[1] = i1 | (static_cast<uint32_t>((r0 >> 62) & 1ULL) << 31);
        dst[2] = i2 | (static_cast<uint32_t>((r1 >> 30) & 1ULL) << 31);
        dst[3] = i3 | (static_cast<uint32_t>((r1 >> 62) & 1ULL) << 31);
    }

    const uint32_t* const tptr = tbl.data();
    #pragma omp parallel for schedule(static) if (k >= 4)
    for (gcta_blas_int j = 0; j < k; ++j) {
        float* __restrict__ w_col = W + static_cast<size_t>(j) * static_cast<size_t>(ldW);
        std::memset(w_col, 0, static_cast<size_t>(s) * sizeof(float));
        const float* __restrict__ y_col = Y + static_cast<size_t>(j) * static_cast<size_t>(ldY);
        for (gcta_blas_int i = 0; i < m; ++i) {
            const float yv = 0.5f * y_col[i];
            const float nv = -yv;
            const uint32_t* const tp = tptr + static_cast<size_t>(i) * 4ULL;
            const uint32_t p0 = tp[0];
            const uint32_t p1 = tp[1];
            const uint32_t p2 = tp[2];
            const uint32_t p3 = tp[3];
            w_col[p0 & 0x7FFFFFFFu] += (p0 & 0x80000000u) ? nv : yv;
            w_col[p1 & 0x7FFFFFFFu] += (p1 & 0x80000000u) ? nv : yv;
            w_col[p2 & 0x7FFFFFFFu] += (p2 & 0x80000000u) ? nv : yv;
            w_col[p3 & 0x7FFFFFFFu] += (p3 & 0x80000000u) ? nv : yv;
        }
    }
}

}  // namespace gcta_cpu_detail

// Mixed-precision Shifted / Sketched CholeskyQR for float32 m x k subspace matrices:
//   - Column L2 pre-normalization balances wildly scaled power-iteration columns.
//   - Auto defaults to Standard Shifted CholeskyQR (direct cblas_ssyrk + dpotrf + strsm).
//     When qr_mode == Sketched (opt-in --reml-qr-mode sketched), Pass 0 forms
//     G_0 = (S Y)^T (S Y) from an OSNAP sparse sign sketch of height s = 2k + 256,
//     and min_passes >= 2 adds 1 exact full-m polish pass.
//   - LAPACK dpotrf on k >= 512 is capped at 48 threads to avoid barrier contention.
inline int gcta_cholesky_qr_thin_Q(gcta_blas_int m, gcta_blas_int k, float* Y, gcta_blas_int ldY,
                                    Eigen::MatrixXd* workspace = nullptr,
                                    Eigen::MatrixXf* workspace_f = nullptr,
                                    int min_passes = 2,
                                    CholeskyQrStats* stats = nullptr,
                                    CholeskyQrMode qr_mode = CholeskyQrMode::Auto) {
    if (m <= 0 || k <= 0) return 0;
    if (stats) ++stats->calls;

    // Column L2 pre-normalization (preserves col(Y), reduces artificial kappa(Y)).
    if (k <= 64 && m >= 16384) {
        double norm2_sum[64]{};
        float inv_norm[64]{};
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nt  = omp_get_num_threads();
            const gcta_blas_int r0 = static_cast<gcta_blas_int>((static_cast<int64_t>(m) * tid) / nt);
            const gcta_blas_int r1 = static_cast<gcta_blas_int>((static_cast<int64_t>(m) * (tid + 1)) / nt);
            double loc[64]{};
            for (gcta_blas_int j = 0; j < k; ++j) {
                const float* __restrict__ col = Y + static_cast<size_t>(j) * ldY;
                double s = 0.0;
                #pragma omp simd reduction(+:s)
                for (gcta_blas_int i = r0; i < r1; ++i) {
                    const double v = static_cast<double>(col[i]);
                    s += v * v;
                }
                loc[j] = s;
            }
            #pragma omp critical
            {
                for (gcta_blas_int j = 0; j < k; ++j) {
                    norm2_sum[j] += loc[j];
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                for (gcta_blas_int j = 0; j < k; ++j) {
                    const double n2 = norm2_sum[j];
                    inv_norm[j] = (n2 > 0.0 && std::isfinite(n2)) ? static_cast<float>(1.0 / std::sqrt(n2)) : 0.0f;
                }
            }
            for (gcta_blas_int j = 0; j < k; ++j) {
                const float inv = inv_norm[j];
                if (inv > 0.0f) {
                    float* __restrict__ col = Y + static_cast<size_t>(j) * ldY;
                    #pragma omp simd
                    for (gcta_blas_int i = r0; i < r1; ++i) {
                        col[i] *= inv;
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(m) * static_cast<size_t>(k) >= 65536)
        for (gcta_blas_int j = 0; j < k; ++j) {
            float* __restrict__ col = Y + static_cast<size_t>(j) * ldY;
            double norm2 = 0.0;
            #pragma omp simd reduction(+:norm2)
            for (gcta_blas_int i = 0; i < m; ++i) {
                const double v = static_cast<double>(col[i]);
                norm2 += v * v;
            }
            if (norm2 > 0.0 && std::isfinite(norm2)) {
                const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm2));
                #pragma omp simd
                for (gcta_blas_int i = 0; i < m; ++i) {
                    col[i] *= inv_norm;
                }
            }
        }
    }

    Eigen::MatrixXd local_G;
    Eigen::MatrixXd& G = workspace ? *workspace : local_G;
    G.resize(k, k);
    Eigen::MatrixXf local_Gf;
    Eigen::MatrixXf& G_f = workspace_f ? *workspace_f : local_Gf;
    G_f.resize(k, k);

    const int base_passes = std::clamp(min_passes, 1, 2);
    const int lapack_cap = (k >= 512) ? std::min(omp_get_max_threads(), 48) : 0;

    // Sketched CholeskyQR (sCholeskyQR): strictly opt-in via qr_mode == Sketched.
    const bool try_sketched = (qr_mode == CholeskyQrMode::Sketched) && (k >= 64 && m >= 3 * k);
    if (try_sketched) {
        gcta_blas_int s = std::min<gcta_blas_int>(
            m, std::max<gcta_blas_int>(2 * k + 256, 4096));
        s = (s / 4) * 4;
        if (s >= 2 * k && s < m) {
            Eigen::MatrixXf W(s, k);
            gcta_cpu_detail::build_osnap_sketch(m, k, s, Y, ldY, W.data(), s);
            cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans, k, s, 1.0f, W.data(), s, 0.0f, G_f.data(), k);
            W.resize(0, 0);
            gcta_cpu_detail::copy_lower_f2d(k, G_f.data(), G.data());

            int sk_info = 0;
            {
                ScopedBlasThreadCap cap(lapack_cap);
                sk_info = gcta_dpotrf(k, G.data(), k);
            }
            if (sk_info == 0 &&
                gcta_cpu_detail::apply_lower_chol_inv_right(m, k, G.data(), G_f.data(), Y, ldY)) {
                if (stats) ++stats->passes;
                if (base_passes == 1) {
                    return 0;
                }
                // Pass 1: 1 exact full-m unshifted polish pass.
                if (stats) ++stats->passes;
                cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans, k, m, 1.0f, Y, ldY, 0.0f, G_f.data(), k);
                gcta_cpu_detail::copy_lower_f2d(k, G_f.data(), G.data());
                int p1_info = 0;
                {
                    ScopedBlasThreadCap cap(lapack_cap);
                    p1_info = gcta_dpotrf(k, G.data(), k);
                }
                if (p1_info == 0 &&
                    gcta_cpu_detail::apply_lower_chol_inv_right(m, k, G.data(), G_f.data(), Y, ldY)) {
                    return 0;
                }
                // If exact polish failed on extreme ill-conditioning, fall through cleanly below
                // (or keep Pass-0 Q if it was simply near-singular roundoff).
                if (p1_info == 0) return 0;
            }
        }
    }

    bool used_shift = false;
    int max_passes = base_passes;
    for (int pass = 0; pass < max_passes; ++pass) {
        if (stats) ++stats->passes;
        cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans, k, m, 1.0f, Y, ldY, 0.0f, G_f.data(), k);
        gcta_cpu_detail::copy_lower_f2d(k, G_f.data(), G.data());
        int info = 0;
        {
            ScopedBlasThreadCap cap(lapack_cap);
            info = gcta_dpotrf(k, G.data(), k);
        }
        if (info != 0) {
            if (pass == 0) {
                int sh_info = -1;
                {
                    ScopedBlasThreadCap cap(lapack_cap);
                    sh_info = gcta_cpu_detail::try_shifted_dpotrf(m, k, G_f.data(), G.data());
                }
                if (sh_info == 0) {
                    used_shift = true;
                    if (stats) ++stats->shifted;
                    max_passes = base_passes + 1;
                } else {
                    if (stats) ++stats->householder;
                    return gcta_qr_thin_Q(m, k, Y, ldY);
                }
            } else if (used_shift && pass == 1) {
                // Shifted pass-0 Y_1 is not orthonormal yet; unshifted pass-1 failure
                // indicates genuine rank deficiency or extreme ill-conditioning.
                if (stats) ++stats->householder;
                return gcta_qr_thin_Q(m, k, Y, ldY);
            } else {
                // Previous unshifted pass already produced an orthonormal Q; skip refinement.
                break;
            }
        }
        if (!gcta_cpu_detail::apply_lower_chol_inv_right(m, k, G.data(), G_f.data(), Y, ldY)) {
            if (stats) ++stats->householder;
            return gcta_qr_thin_Q(m, k, Y, ldY);
        }
    }
    return 0;
}

// Same as gcta_cholesky_qr_thin_Q, but also returns the composed upper
// triangular factor R (k x k, column-major, leading dimension ldR) such that
// the original Y equals Q*R. Used by callers (Nystrom) that need R for a
// follow-on small eigendecomposition and can't afford an extra
// n x k copy of Y to recover it afterwards as Q^T*Y_original. Composes
// R <- L_pass^T * R in place via cblas_dtrmm (Level-3 BLAS, zero extra copy).
inline int gcta_cholesky_qr_thin_QR(gcta_blas_int m, gcta_blas_int k, double* Y, gcta_blas_int ldY,
                                     double* R, gcta_blas_int ldR, Eigen::MatrixXd* workspace = nullptr) {
    if (m <= 0 || k <= 0) return 0;
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
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    m, k, 1.0, G.data(), k, Y, ldY);
        if (pass == 0) {
            gcta_cpu_detail::copy_lower_transpose_to_upper(k, G.data(), R, ldR);
        } else {
            cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                        k, k, 1.0, G.data(), k, R, ldR);
        }
    }
    return 0;
}

inline int gcta_cholesky_qr_thin_QR(gcta_blas_int m, gcta_blas_int k, float* Y, gcta_blas_int ldY,
                                     double* R, gcta_blas_int ldR, Eigen::MatrixXd* workspace = nullptr) {
    if (m <= 0 || k <= 0) return 0;
    Eigen::MatrixXd local_G;
    Eigen::MatrixXd& G = workspace ? *workspace : local_G;
    G.resize(k, k);
    Eigen::MatrixXf G_f(k, k);

    bool used_shift = false;
    int max_passes = 2;
    for (int pass = 0; pass < max_passes; ++pass) {
        cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans, k, m, 1.0f, Y, ldY, 0.0f, G_f.data(), k);
        gcta_cpu_detail::copy_lower_f2d(k, G_f.data(), G.data());
        const int info = gcta_dpotrf(k, G.data(), k);
        if (info != 0) {
            if (pass == 0) {
                if (gcta_cpu_detail::try_shifted_dpotrf(m, k, G_f.data(), G.data()) == 0) {
                    used_shift = true;
                    max_passes = 3;
                } else {
                    Eigen::VectorXf tau(k);
                    const int info_qr = gcta_sgeqrf(m, k, Y, ldY, tau.data());
                    if (info_qr != 0) return info_qr;
                    gcta_cpu_detail::extract_upper_f2d(k, Y, ldY, R, ldR);
                    return gcta_sorgqr(m, k, Y, ldY, tau.data());
                }
            } else if (used_shift && pass == 1) {
                // Y currently holds Y_1 = Y_orig * R_0^{-1} and R holds R_0.
                // Factor Y_1 = Q * R_hh via Householder and compose R = R_hh * R_0.
                Eigen::VectorXf tau(k);
                const int info_qr = gcta_sgeqrf(m, k, Y, ldY, tau.data());
                if (info_qr != 0) return info_qr;
                gcta_cpu_detail::extract_upper_f2d(k, Y, ldY, G.data(), k);
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                            k, k, 1.0, G.data(), k, R, ldR);
                return gcta_sorgqr(m, k, Y, ldY, tau.data());
            } else {
                // Previous unshifted pass's Q/R already valid; skip refinement.
                break;
            }
        }
        gcta_cpu_detail::copy_lower_d2f(k, G.data(), G_f.data());
        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    m, k, 1.0f, G_f.data(), k, Y, ldY);
        if (pass == 0) {
            gcta_cpu_detail::copy_lower_transpose_to_upper(k, G.data(), R, ldR);
        } else {
            cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                        k, k, 1.0, G.data(), k, R, ldR);
        }
    }
    return 0;
}

#if defined(GCTA_USE_AOCL_DLP) || (defined(GCTA_USE_AOCL) && defined(BLIS_ENABLE_ADDON_AOCL_GEMM))
  #define GCTA_HAS_AOCL_S8_GEMM 1
  #define GCTA_HAS_AOCL_BF16_GEMM 1
#else
  #define GCTA_HAS_AOCL_S8_GEMM 0
  #define GCTA_HAS_AOCL_BF16_GEMM 0
#endif

namespace gcta_cpu_detail {

#if GCTA_HAS_AOCL_S8_GEMM
// Compile-time arity dispatch for AOCL-DLP 6.0 / AOCL-BLIS 5.1+ (trailing metadata/post_op
// pointer) vs AOCL-BLIS <= 5.0 (no trailing pointer) without macro fragility.
template <typename Fn>
inline size_t call_aocl_get_reorder_buf_size(
    Fn fn, char order, char trans, char mat_type, gcta_blas_int k, gcta_blas_int n)
{
    if constexpr (requires { fn(order, trans, mat_type, k, n, nullptr); }) {
        return static_cast<size_t>(fn(order, trans, mat_type, k, n, nullptr));
    } else {
        return static_cast<size_t>(fn(order, trans, mat_type, k, n));
    }
}

template <typename Fn, typename SrcPtr, typename DstPtr>
inline void call_aocl_reorder(
    Fn fn, char order, char trans, char mat_type,
    SrcPtr src, DstPtr dst, gcta_blas_int k, gcta_blas_int n, gcta_blas_int ldb)
{
    if constexpr (requires { fn(order, trans, mat_type, src, dst, k, n, ldb, nullptr); }) {
        fn(order, trans, mat_type, src, dst, k, n, ldb, nullptr);
    } else {
        fn(order, trans, mat_type, src, dst, k, n, ldb);
    }
}
#endif

#if GCTA_HAS_AOCL_BF16_GEMM
// Extract exact parameter pointer types from the installed header's declaration of
// aocl_gemm_bf16bf16f32of32 (handles int16_t / uint16_t / bfloat16 / aocl_bf16 typedefs).
template <typename Fn>
struct AoclGemm17Traits;

template <typename R,
          typename OrderT, typename TransAT, typename TransBT,
          typename MT, typename NT, typename KT,
          typename AlphaT,
          typename APtrT, typename LdaT, typename MemAT,
          typename BPtrT, typename LdbT, typename MemBT,
          typename BetaT,
          typename CPtrT, typename LdcT,
          typename MetaT>
struct AoclGemm17Traits<R(*)(OrderT, TransAT, TransBT,
                             MT, NT, KT,
                             AlphaT,
                             APtrT, LdaT, MemAT,
                             BPtrT, LdbT, MemBT,
                             BetaT,
                             CPtrT, LdcT,
                             MetaT)> {
    using a_ptr_t = APtrT;
    using b_ptr_t = BPtrT;
};
#endif

inline void f32_to_bf16_rne_scalar(const float* src, uint16_t* dst, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        uint32_t u = 0;
        std::memcpy(&u, src + i, sizeof(u));
        const bool is_nan = (u & 0x7FFFFFFFu) > 0x7F800000u;
        const uint32_t rne = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
        const uint32_t qnan = (u >> 16) | 0x0040u;
        dst[i] = static_cast<uint16_t>(is_nan ? qnan : rne);
    }
}

#if GCTA_CPU_x86 && (defined(__GNUC__) || defined(__clang__))
#if defined(__AVX512BF16__) && defined(__AVX512F__)
inline void f32_to_bf16_rne_avx512bf16(const float* src, uint16_t* dst, size_t n) noexcept {
#else
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16"), noinline))
inline void f32_to_bf16_rne_avx512bf16(const float* src, uint16_t* dst, size_t n) noexcept {
#endif
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m512 lo = _mm512_loadu_ps(src + i);
        const __m512 hi = _mm512_loadu_ps(src + i + 16);
        // Intel SDM VCVTNE2PS2BF16(hi, lo): lo -> dst[0..15], hi -> dst[16..31].
        const __m512bh bh = _mm512_cvtne2ps_pbh(hi, lo);
        std::memcpy(dst + i, &bh, 64);
    }
    if (i + 16 <= n) {
        const __m512 v = _mm512_loadu_ps(src + i);
        const __m256bh bh = _mm512_cvtneps_pbh(v);
        std::memcpy(dst + i, &bh, 32);
        i += 16;
    }
    for (; i < n; ++i) {
        uint32_t u = 0;
        std::memcpy(&u, src + i, sizeof(u));
        const bool is_nan = (u & 0x7FFFFFFFu) > 0x7F800000u;
        const uint32_t rne = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
        const uint32_t qnan = (u >> 16) | 0x0040u;
        dst[i] = static_cast<uint16_t>(is_nan ? qnan : rne);
    }
}
#endif

}  // namespace gcta_cpu_detail

// Compute C(rows x cols, column-major, leading dim ldc) = A(rows x K, lda) * B(cols x K, ldb)^T
// where A and B are column-major int8_t matrices in {0, 1, 2} and C is a column-major int32_t matrix.
// On AOCL (AOCL-DLP 6.0 or BLIS addon), dispatches directly to Zen 4 AVX-512 VNNI
// aocl_gemm_u8s8s32os32 (bit-exact for {0,1,2} inputs, avoiding s8s8s32's +128 bias/colsum overhead)
// via row-major transpose equivalence:
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
    aocl_gemm_u8s8s32os32('r', 't', 'n',
                          cols, rows, K,
                          1,
                          reinterpret_cast<const uint8_t*>(B), ldb, 'n',
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
// On AOCL (AOCL-DLP 6.0 or BLIS addon), dispatches directly to Zen 4 AVX-512 VNNI
// aocl_gemm_u8s8s32os32 via row-major transpose equivalence:
//   C_colmajor(rows x cols) == C_rowmajor(cols x rows) = B_col_as_row(cols x K, u8) * A_col_as_row(K x rows, s8).
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
    aocl_gemm_u8s8s32os32('r', 'n', 'n',
                          cols, rows, K,
                          1,
                          reinterpret_cast<const uint8_t*>(B), ldb, 'n',
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

// Query byte size required to pre-reorder A_col(rows x K, col-major signed int8_t)
// as the static Matrix-B operand for gcta_gemm_u8s8s32_colmajor_nn_reordered.
// Under row-major transpose equivalence, A_col(rows x K) is AOCL-DLP's row-major
// Matrix B of shape (K x rows), so AOCL receives (k=K, n=rows).
// Returns 0 when !GCTA_HAS_AOCL_S8_GEMM.
inline size_t gcta_u8s8s32_reorder_b_size(gcta_blas_int rows, gcta_blas_int K) {
#if GCTA_HAS_AOCL_S8_GEMM
    if (rows <= 0 || K <= 0) return 0;
    return gcta_cpu_detail::call_aocl_get_reorder_buf_size(
        &aocl_get_reorder_buf_size_u8s8s32os32, 'r', 'n', 'b', K, rows);
#else
    (void)rows;
    (void)K;
    return 0;
#endif
}

// Pack A_col(rows x K, col-major signed int8_t, lda >= rows) into dst_reordered
// (allocated to >= gcta_u8s8s32_reorder_b_size(rows, K) bytes) for reuse across
// repeated gcta_gemm_u8s8s32_colmajor_nn_reordered calls. No-op when !GCTA_HAS_AOCL_S8_GEMM.
inline void gcta_u8s8s32_reorder_b(
    gcta_blas_int rows, gcta_blas_int K,
    const int8_t* A_col, gcta_blas_int lda,
    int8_t* dst_reordered)
{
#if GCTA_HAS_AOCL_S8_GEMM
    if (rows <= 0 || K <= 0 || A_col == nullptr || dst_reordered == nullptr) return;
    gcta_cpu_detail::call_aocl_reorder(
        &aocl_reorder_u8s8s32os32, 'r', 'n', 'b',
        A_col, dst_reordered, K, rows, lda);
#else
    (void)rows;
    (void)K;
    (void)A_col;
    (void)lda;
    (void)dst_reordered;
#endif
}

// Compute C(rows x cols, col-major, ldc) = beta * C + A_col(rows x K) * B_col(K x cols, ldb)
// using pre-reordered A_reordered (from gcta_u8s8s32_reorder_b) when GCTA_HAS_AOCL_S8_GEMM
// and A_reordered != nullptr, otherwise safely falling back to gcta_gemm_s8s8s32_colmajor_nn
// with A_fallback_col. B_col must have values in {0, 1, 2}.
inline void gcta_gemm_u8s8s32_colmajor_nn_reordered(
    gcta_blas_int rows, gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_reordered, const int8_t* A_fallback_col, gcta_blas_int lda,
    const int8_t* B, gcta_blas_int ldb,
    int32_t* C, gcta_blas_int ldc,
    int32_t beta,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32)
{
    if (rows <= 0 || cols <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    if (A_reordered != nullptr) {
        (void)A_fallback_col;
        (void)scratch_A_f32;
        (void)scratch_B_f32;
        (void)scratch_C_f32;
        aocl_gemm_u8s8s32os32('r', 'n', 'n',
                              cols, rows, K,
                              1,
                              reinterpret_cast<const uint8_t*>(B), ldb, 'n',
                              A_reordered, lda, 'r',
                              beta,
                              C, ldc,
                              nullptr);
        return;
    }
#else
    (void)A_reordered;
#endif
    gcta_gemm_s8s8s32_colmajor_nn(
        rows, cols, K,
        A_fallback_col, lda,
        B, ldb,
        C, ldc,
        beta,
        scratch_A_f32, scratch_B_f32, scratch_C_f32);
}

// Compute the lower triangle of C(cols x cols, column-major, leading dim ldc) = A(cols x K, lda) * A(cols x K, lda)^T
// where A has values in {0, 1, 2}. Uses b=1024 diagonal SYRK sub-tiling under AOCL-DLP.
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
    constexpr gcta_blas_int kDiagBlock = 1024;
    for (gcta_blas_int j = 0; j < cols; j += kDiagBlock) {
        const gcta_blas_int jb = std::min(kDiagBlock, cols - j);
        const int8_t* A_j = A + j;
        aocl_gemm_u8s8s32os32('r', 't', 'n',
                              jb, jb, K,
                              1,
                              reinterpret_cast<const uint8_t*>(A_j), lda, 'n',
                              A_j, lda, 'n',
                              0,
                              C + j + static_cast<size_t>(j) * ldc, ldc,
                              nullptr);
        const gcta_blas_int rem = cols - (j + jb);
        if (rem > 0) {
            const int8_t* A_below = A + (j + jb);
            aocl_gemm_u8s8s32os32('r', 't', 'n',
                                  jb, rem, K,
                                  1,
                                  reinterpret_cast<const uint8_t*>(A_j), lda, 'n',
                                  A_below, lda, 'n',
                                  0,
                                  C + (j + jb) + static_cast<size_t>(j) * ldc, ldc,
                                  nullptr);
        }
    }
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

// Compute C(rows x cols, column-major, leading dim ldc) =
//   beta * C + A_kmajor(rows x K, ldK) * B_kmajor(cols x K, ldK)^T
// where A_kmajor and B_kmajor store K markers contiguously per sample in {0, 1, 2}
// (row-major with stride ldK >= K).
// Under AOCL VNNI (GCTA_HAS_AOCL_S8_GEMM), dispatches directly to aocl_gemm_u8s8s32os32:
//   C_colmajor(rows x cols, ldc) == C_rowmajor(cols x rows, ldc) = B_kmajor(cols x K, u8) * A_kmajor(rows x K, s8)^T.
// Supports arbitrary beta (0 for overwrite, 1 for in-place int32_t GRM accumulation).
inline void gcta_gemm_u8s8s32_kmajor_nt_beta(
    gcta_blas_int rows, gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    const int8_t* B_kmajor,
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
    aocl_gemm_u8s8s32os32('r', 'n', 't',
                          cols, rows, K,
                          1,
                          reinterpret_cast<const uint8_t*>(B_kmajor), ldK, 'n',
                          A_kmajor, ldK, 'n',
                          beta,
                          C, ldc,
                          nullptr);
#else
    constexpr gcta_blas_int kMaxTile = 4096;
    for (gcta_blas_int c0 = 0; c0 < cols; c0 += kMaxTile) {
        const gcta_blas_int cb = std::min(kMaxTile, cols - c0);
        const size_t size_B = static_cast<size_t>(cb) * static_cast<size_t>(K);
        if (scratch_B_f32.size() < size_B) scratch_B_f32.resize(size_B);
        #pragma omp parallel for schedule(static)
        for (gcta_blas_int c = 0; c < cb; ++c) {
            const int8_t* src = B_kmajor + static_cast<size_t>(c0 + c) * ldK;
            float* dst = scratch_B_f32.data() + static_cast<size_t>(c) * K;
            for (gcta_blas_int k = 0; k < K; ++k) {
                dst[k] = static_cast<float>(src[k]);
            }
        }
        for (gcta_blas_int r0 = 0; r0 < rows; r0 += kMaxTile) {
            const gcta_blas_int rb = std::min(kMaxTile, rows - r0);
            const size_t size_A = static_cast<size_t>(rb) * static_cast<size_t>(K);
            if (scratch_A_f32.size() < size_A) scratch_A_f32.resize(size_A);
            #pragma omp parallel for schedule(static)
            for (gcta_blas_int r = 0; r < rb; ++r) {
                const int8_t* src = A_kmajor + static_cast<size_t>(r0 + r) * ldK;
                float* dst = scratch_A_f32.data() + static_cast<size_t>(r) * K;
                for (gcta_blas_int k = 0; k < K; ++k) {
                    dst[k] = static_cast<float>(src[k]);
                }
            }
            const size_t size_C = static_cast<size_t>(rb) * static_cast<size_t>(cb);
            if (scratch_C_f32.size() < size_C) scratch_C_f32.resize(size_C);
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        rb, cb, K,
                        1.0f,
                        scratch_A_f32.data(), K,
                        scratch_B_f32.data(), K,
                        0.0f,
                        scratch_C_f32.data(), rb);
            #pragma omp parallel for schedule(static)
            for (gcta_blas_int c = 0; c < cb; ++c) {
                const float* src = scratch_C_f32.data() + static_cast<size_t>(c) * rb;
                int32_t* dst = C + r0 + static_cast<size_t>(c0 + c) * ldc;
                if (beta == 0) {
                    for (gcta_blas_int r = 0; r < rb; ++r) {
                        dst[r] = static_cast<int32_t>(src[r]);
                    }
                } else if (beta == 1) {
                    for (gcta_blas_int r = 0; r < rb; ++r) {
                        dst[r] += static_cast<int32_t>(src[r]);
                    }
                } else {
                    for (gcta_blas_int r = 0; r < rb; ++r) {
                        dst[r] = beta * dst[r] + static_cast<int32_t>(src[r]);
                    }
                }
            }
        }
    }
#endif
}

inline void gcta_gemm_s8s8s32_kmajor_nt(
    gcta_blas_int rows, gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    const int8_t* B_kmajor,
    int32_t* C, gcta_blas_int ldc,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32)
{
    gcta_gemm_u8s8s32_kmajor_nt_beta(
        rows, cols, K, A_kmajor, ldK, B_kmajor, C, ldc, 0,
        scratch_A_f32, scratch_B_f32, scratch_C_f32);
}

// Query byte size required to pre-reorder transposed row-major A_kmajor(rows x K, ldK)
// as the static Matrix-B operand (trans='t', shape K x rows after transpose) for
// gcta_gemm_u8s8s32_kmajor_nt_beta_reordered (e.g., GRM r0-outer strip pre-packing).
// Returns 0 when !GCTA_HAS_AOCL_S8_GEMM.
inline size_t gcta_u8s8s32_reorder_bt_size(gcta_blas_int rows, gcta_blas_int K) {
#if GCTA_HAS_AOCL_S8_GEMM
    if (rows <= 0 || K <= 0) return 0;
    return gcta_cpu_detail::call_aocl_get_reorder_buf_size(
        &aocl_get_reorder_buf_size_u8s8s32os32, 'r', 't', 'b', K, rows);
#else
    (void)rows;
    (void)K;
    return 0;
#endif
}

// Pack transposed row-major A_kmajor(rows x K, stride ldK >= K) into dst_reordered
// (allocated to >= gcta_u8s8s32_reorder_bt_size(rows, K) bytes) for repeated
// gcta_gemm_u8s8s32_kmajor_nt_beta_reordered calls. No-op when !GCTA_HAS_AOCL_S8_GEMM.
inline void gcta_u8s8s32_reorder_bt(
    gcta_blas_int rows, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    int8_t* dst_reordered)
{
#if GCTA_HAS_AOCL_S8_GEMM
    if (rows <= 0 || K <= 0 || A_kmajor == nullptr || dst_reordered == nullptr) return;
    gcta_cpu_detail::call_aocl_reorder(
        &aocl_reorder_u8s8s32os32, 'r', 't', 'b',
        A_kmajor, dst_reordered, K, rows, ldK);
#else
    (void)rows;
    (void)K;
    (void)A_kmajor;
    (void)ldK;
    (void)dst_reordered;
#endif
}

// Compute C(rows x cols, col-major, ldc) =
//   beta * C + A_kmajor(rows x K, ldK) * B_kmajor(cols x K, ldK)^T
// using pre-reordered A_reordered (from gcta_u8s8s32_reorder_bt) when GCTA_HAS_AOCL_S8_GEMM
// and A_reordered != nullptr; otherwise falls back to gcta_gemm_u8s8s32_kmajor_nt_beta
// with A_fallback_kmajor. Note: AOCL-DLP validates ldb against transb before inspecting
// mem_format_b='r', so transb='t' and ldb=ldK match the reorder call ('r','t','b',K,rows,ldK).
inline void gcta_gemm_u8s8s32_kmajor_nt_beta_reordered(
    gcta_blas_int rows, gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_reordered, const int8_t* A_fallback_kmajor, gcta_blas_int ldK,
    const int8_t* B_kmajor,
    int32_t* C, gcta_blas_int ldc,
    int32_t beta,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32)
{
    if (rows <= 0 || cols <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    if (A_reordered != nullptr) {
        (void)A_fallback_kmajor;
        (void)scratch_A_f32;
        (void)scratch_B_f32;
        (void)scratch_C_f32;
        aocl_gemm_u8s8s32os32('r', 'n', 't',
                              cols, rows, K,
                              1,
                              reinterpret_cast<const uint8_t*>(B_kmajor), ldK, 'n',
                              A_reordered, ldK, 'r',
                              beta,
                              C, ldc,
                              nullptr);
        return;
    }
#else
    (void)A_reordered;
#endif
    gcta_gemm_u8s8s32_kmajor_nt_beta(
        rows, cols, K,
        A_fallback_kmajor, ldK,
        B_kmajor,
        C, ldc,
        beta,
        scratch_A_f32, scratch_B_f32, scratch_C_f32);
}

// Compute the lower triangle of C(cols x cols, column-major, leading dim ldc) =
//   beta * C + A_kmajor(cols x K, ldK) * A_kmajor(cols x K, ldK)^T
// where A_kmajor stores K markers contiguously per sample in {0, 1, 2}.
// Under AOCL VNNI (GCTA_HAS_AOCL_S8_GEMM), sub-tiles the lower triangle with b=1024
// diagonal blocks + below-diagonal rectangular panels via aocl_gemm_u8s8s32os32,
// cutting diagonal-tile FLOPs nearly in half (~1.78x on 8192x8192) and keeping
// upper-triangle pages of C untouched outside the narrow 1024-wide diagonal band.
inline void gcta_syrk_u8s8s32_lower_kmajor_nt_beta(
    gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    int32_t* C, gcta_blas_int ldc,
    int32_t beta,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_C_f32,
    gcta_blas_int diag_block = 1024)
{
    if (cols <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    (void)scratch_A_f32;
    (void)scratch_C_f32;
    const gcta_blas_int kDiagBlock = std::max<gcta_blas_int>(64, diag_block);
    for (gcta_blas_int j = 0; j < cols; j += kDiagBlock) {
        const gcta_blas_int jb = std::min(kDiagBlock, cols - j);
        const int8_t* A_j = A_kmajor + static_cast<size_t>(j) * ldK;
        // Diagonal jb x jb sub-tile
        aocl_gemm_u8s8s32os32('r', 'n', 't',
                              jb, jb, K,
                              1,
                              reinterpret_cast<const uint8_t*>(A_j), ldK, 'n',
                              A_j, ldK, 'n',
                              beta,
                              C + j + static_cast<size_t>(j) * ldc, ldc,
                              nullptr);
        // Below-diagonal (cols - (j + jb)) x jb rectangular panel (strictly lower triangle)
        const gcta_blas_int rem = cols - (j + jb);
        if (rem > 0) {
            const int8_t* A_below = A_kmajor + static_cast<size_t>(j + jb) * ldK;
            aocl_gemm_u8s8s32os32('r', 'n', 't',
                                  jb, rem, K,
                                  1,
                                  reinterpret_cast<const uint8_t*>(A_j), ldK, 'n',
                                  A_below, ldK, 'n',
                                  beta,
                                  C + (j + jb) + static_cast<size_t>(j) * ldc, ldc,
                                  nullptr);
        }
    }
#else
    constexpr gcta_blas_int kMaxTile = 4096;
    std::vector<float> scratch_B_f32;
    for (gcta_blas_int j0 = 0; j0 < cols; j0 += kMaxTile) {
        const gcta_blas_int jb = std::min(kMaxTile, cols - j0);
        const size_t size_Aj = static_cast<size_t>(jb) * static_cast<size_t>(K);
        if (scratch_A_f32.size() < size_Aj) scratch_A_f32.resize(size_Aj);
        #pragma omp parallel for schedule(static)
        for (gcta_blas_int c = 0; c < jb; ++c) {
            const int8_t* src = A_kmajor + static_cast<size_t>(j0 + c) * ldK;
            float* dst = scratch_A_f32.data() + static_cast<size_t>(c) * K;
            for (gcta_blas_int k = 0; k < K; ++k) {
                dst[k] = static_cast<float>(src[k]);
            }
        }
        const size_t size_Cdiag = static_cast<size_t>(jb) * static_cast<size_t>(jb);
        if (scratch_C_f32.size() < size_Cdiag) scratch_C_f32.resize(size_Cdiag);
        cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans,
                    jb, K,
                    1.0f,
                    scratch_A_f32.data(), K,
                    0.0f,
                    scratch_C_f32.data(), jb);
        #pragma omp parallel for schedule(guided)
        for (gcta_blas_int c = 0; c < jb; ++c) {
            const float* src = scratch_C_f32.data() + c + static_cast<size_t>(c) * jb;
            int32_t* dst = C + (j0 + c) + static_cast<size_t>(j0 + c) * ldc;
            const gcta_blas_int len = jb - c;
            if (beta == 0) {
                for (gcta_blas_int r = 0; r < len; ++r) {
                    dst[r] = static_cast<int32_t>(src[r]);
                }
            } else if (beta == 1) {
                for (gcta_blas_int r = 0; r < len; ++r) {
                    dst[r] += static_cast<int32_t>(src[r]);
                }
            } else {
                for (gcta_blas_int r = 0; r < len; ++r) {
                    dst[r] = beta * dst[r] + static_cast<int32_t>(src[r]);
                }
            }
        }

        // Below-diagonal rectangular panels (when cols > kMaxTile)
        for (gcta_blas_int i0 = j0 + jb; i0 < cols; i0 += kMaxTile) {
            const gcta_blas_int ib = std::min(kMaxTile, cols - i0);
            const size_t size_Ai = static_cast<size_t>(ib) * static_cast<size_t>(K);
            if (scratch_B_f32.size() < size_Ai) scratch_B_f32.resize(size_Ai);
            #pragma omp parallel for schedule(static)
            for (gcta_blas_int r = 0; r < ib; ++r) {
                const int8_t* src = A_kmajor + static_cast<size_t>(i0 + r) * ldK;
                float* dst = scratch_B_f32.data() + static_cast<size_t>(r) * K;
                for (gcta_blas_int k = 0; k < K; ++k) {
                    dst[k] = static_cast<float>(src[k]);
                }
            }
            const size_t size_Crect = static_cast<size_t>(ib) * static_cast<size_t>(jb);
            if (scratch_C_f32.size() < size_Crect) scratch_C_f32.resize(size_Crect);
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        ib, jb, K,
                        1.0f,
                        scratch_B_f32.data(), K,
                        scratch_A_f32.data(), K,
                        0.0f,
                        scratch_C_f32.data(), ib);
            #pragma omp parallel for schedule(static)
            for (gcta_blas_int c = 0; c < jb; ++c) {
                const float* src = scratch_C_f32.data() + static_cast<size_t>(c) * ib;
                int32_t* dst = C + i0 + static_cast<size_t>(j0 + c) * ldc;
                if (beta == 0) {
                    for (gcta_blas_int r = 0; r < ib; ++r) {
                        dst[r] = static_cast<int32_t>(src[r]);
                    }
                } else if (beta == 1) {
                    for (gcta_blas_int r = 0; r < ib; ++r) {
                        dst[r] += static_cast<int32_t>(src[r]);
                    }
                } else {
                    for (gcta_blas_int r = 0; r < ib; ++r) {
                        dst[r] = beta * dst[r] + static_cast<int32_t>(src[r]);
                    }
                }
            }
        }
    }
#endif
#if !GCTA_HAS_AOCL_S8_GEMM
    (void)diag_block;
#endif
}

// Compute C(rows x cols, row-major, leading dim ldc >= cols) =
//   beta * C + A_kmajor(rows x K, ldK) * B_kmajor(cols x K, ldK)^T
// where A_kmajor and B_kmajor store K markers contiguously per sample in {0, 1, 2}
// (row-major with stride ldK >= K).
// Under AOCL VNNI (GCTA_HAS_AOCL_S8_GEMM), dispatches directly with M=rows and
// N=cols so N sits in BLIS/DLP's outermost JC loop and M x NC C-slabs stay L3-resident.
inline void gcta_gemm_u8s8s32_kmajor_nt_rowmajor_beta(
    gcta_blas_int rows, gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    const int8_t* B_kmajor,
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
    aocl_gemm_u8s8s32os32('r', 'n', 't',
                          rows, cols, K,
                          1,
                          reinterpret_cast<const uint8_t*>(A_kmajor), ldK, 'n',
                          B_kmajor, ldK, 'n',
                          beta,
                          C, ldc,
                          nullptr);
#else
    constexpr gcta_blas_int kMaxTile = 4096;
    for (gcta_blas_int r0 = 0; r0 < rows; r0 += kMaxTile) {
        const gcta_blas_int rb = std::min(kMaxTile, rows - r0);
        const size_t size_A = static_cast<size_t>(rb) * static_cast<size_t>(K);
        if (scratch_A_f32.size() < size_A) scratch_A_f32.resize(size_A);
        #pragma omp parallel for schedule(static)
        for (gcta_blas_int r = 0; r < rb; ++r) {
            const int8_t* src = A_kmajor + static_cast<size_t>(r0 + r) * ldK;
            float* dst = scratch_A_f32.data() + static_cast<size_t>(r) * K;
            for (gcta_blas_int k = 0; k < K; ++k) {
                dst[k] = static_cast<float>(src[k]);
            }
        }
        for (gcta_blas_int c0 = 0; c0 < cols; c0 += kMaxTile) {
            const gcta_blas_int cb = std::min(kMaxTile, cols - c0);
            const size_t size_B = static_cast<size_t>(cb) * static_cast<size_t>(K);
            if (scratch_B_f32.size() < size_B) scratch_B_f32.resize(size_B);
            #pragma omp parallel for schedule(static)
            for (gcta_blas_int c = 0; c < cb; ++c) {
                const int8_t* src = B_kmajor + static_cast<size_t>(c0 + c) * ldK;
                float* dst = scratch_B_f32.data() + static_cast<size_t>(c) * K;
                for (gcta_blas_int k = 0; k < K; ++k) {
                    dst[k] = static_cast<float>(src[k]);
                }
            }
            const size_t size_C = static_cast<size_t>(rb) * static_cast<size_t>(cb);
            if (scratch_C_f32.size() < size_C) scratch_C_f32.resize(size_C);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        rb, cb, K,
                        1.0f,
                        scratch_A_f32.data(), K,
                        scratch_B_f32.data(), K,
                        0.0f,
                        scratch_C_f32.data(), cb);
            #pragma omp parallel for schedule(static)
            for (gcta_blas_int r = 0; r < rb; ++r) {
                const float* src = scratch_C_f32.data() + static_cast<size_t>(r) * cb;
                int32_t* dst = C + static_cast<size_t>(r0 + r) * ldc + c0;
                if (beta == 0) {
                    for (gcta_blas_int c = 0; c < cb; ++c) {
                        dst[c] = static_cast<int32_t>(src[c]);
                    }
                } else if (beta == 1) {
                    for (gcta_blas_int c = 0; c < cb; ++c) {
                        dst[c] += static_cast<int32_t>(src[c]);
                    }
                } else {
                    for (gcta_blas_int c = 0; c < cb; ++c) {
                        dst[c] = beta * dst[c] + static_cast<int32_t>(src[c]);
                    }
                }
            }
        }
    }
#endif
}

// Compute the lower triangle of C(dim x dim, row-major, leading dim ldc >= dim) =
//   beta * C + A_kmajor(dim x K, ldK) * A_kmajor(dim x K, ldK)^T
// where A_kmajor stores K markers contiguously per sample in {0, 1, 2}.
inline void gcta_syrk_u8s8s32_lower_kmajor_nt_rowmajor_beta(
    gcta_blas_int dim, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    int32_t* C, gcta_blas_int ldc,
    int32_t beta,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_B_f32,
    std::vector<float>& scratch_C_f32,
    gcta_blas_int diag_block = 1024)
{
    if (dim <= 0 || K <= 0) return;
#if GCTA_HAS_AOCL_S8_GEMM
    (void)scratch_A_f32;
    (void)scratch_B_f32;
    (void)scratch_C_f32;
    const gcta_blas_int kDiagBlock = std::max<gcta_blas_int>(64, diag_block);
    for (gcta_blas_int j = 0; j < dim; j += kDiagBlock) {
        const gcta_blas_int jb = std::min(kDiagBlock, dim - j);
        const int8_t* A_j = A_kmajor + static_cast<size_t>(j) * ldK;
        aocl_gemm_u8s8s32os32('r', 'n', 't',
                              jb, jb, K,
                              1,
                              reinterpret_cast<const uint8_t*>(A_j), ldK, 'n',
                              A_j, ldK, 'n',
                              beta,
                              C + static_cast<size_t>(j) * ldc + j, ldc,
                              nullptr);
        const gcta_blas_int rem = dim - (j + jb);
        if (rem > 0) {
            const int8_t* A_below = A_kmajor + static_cast<size_t>(j + jb) * ldK;
            aocl_gemm_u8s8s32os32('r', 'n', 't',
                                  rem, jb, K,
                                  1,
                                  reinterpret_cast<const uint8_t*>(A_below), ldK, 'n',
                                  A_j, ldK, 'n',
                                  beta,
                                  C + static_cast<size_t>(j + jb) * ldc + j, ldc,
                                  nullptr);
        }
    }
#else
    (void)diag_block;
    constexpr gcta_blas_int kMaxTile = 4096;
    for (gcta_blas_int j0 = 0; j0 < dim; j0 += kMaxTile) {
        const gcta_blas_int jb = std::min(kMaxTile, dim - j0);
        const size_t size_Aj = static_cast<size_t>(jb) * static_cast<size_t>(K);
        if (scratch_A_f32.size() < size_Aj) scratch_A_f32.resize(size_Aj);
        #pragma omp parallel for schedule(static)
        for (gcta_blas_int r = 0; r < jb; ++r) {
            const int8_t* src = A_kmajor + static_cast<size_t>(j0 + r) * ldK;
            float* dst = scratch_A_f32.data() + static_cast<size_t>(r) * K;
            for (gcta_blas_int k = 0; k < K; ++k) {
                dst[k] = static_cast<float>(src[k]);
            }
        }
        const size_t size_Cdiag = static_cast<size_t>(jb) * static_cast<size_t>(jb);
        if (scratch_C_f32.size() < size_Cdiag) scratch_C_f32.resize(size_Cdiag);
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                    jb, K,
                    1.0f,
                    scratch_A_f32.data(), K,
                    0.0f,
                    scratch_C_f32.data(), jb);
        #pragma omp parallel for schedule(guided)
        for (gcta_blas_int r = 0; r < jb; ++r) {
            const float* src = scratch_C_f32.data() + static_cast<size_t>(r) * jb;
            int32_t* dst = C + static_cast<size_t>(j0 + r) * ldc + j0;
            const gcta_blas_int len = r + 1;
            if (beta == 0) {
                for (gcta_blas_int c = 0; c < len; ++c) {
                    dst[c] = static_cast<int32_t>(src[c]);
                }
            } else if (beta == 1) {
                for (gcta_blas_int c = 0; c < len; ++c) {
                    dst[c] += static_cast<int32_t>(src[c]);
                }
            } else {
                for (gcta_blas_int c = 0; c < len; ++c) {
                    dst[c] = beta * dst[c] + static_cast<int32_t>(src[c]);
                }
            }
        }
        const gcta_blas_int rem = dim - (j0 + jb);
        if (rem > 0) {
            gcta_gemm_u8s8s32_kmajor_nt_rowmajor_beta(
                rem, jb, K,
                A_kmajor + static_cast<size_t>(j0 + jb) * ldK, ldK,
                A_kmajor + static_cast<size_t>(j0) * ldK,
                C + static_cast<size_t>(j0 + jb) * ldc + j0, ldc,
                beta,
                scratch_A_f32, scratch_B_f32, scratch_C_f32);
        }
    }
#endif
}

inline void gcta_syrk_s8s8s32_lower_kmajor_nt(
    gcta_blas_int cols, gcta_blas_int K,
    const int8_t* A_kmajor, gcta_blas_int ldK,
    int32_t* C, gcta_blas_int ldc,
    std::vector<float>& scratch_A_f32,
    std::vector<float>& scratch_C_f32)
{
    gcta_syrk_u8s8s32_lower_kmajor_nt_beta(
        cols, K, A_kmajor, ldK, C, ldc, 0, scratch_A_f32, scratch_C_f32);
}

// ---------------------------------------------------------------------------
// BFloat16 conversion & GEMM helpers (Zen 4 AVX-512 BF16 / AOCL-DLP + portable fallback)
// ---------------------------------------------------------------------------

// Convert n float32 values to IEEE bfloat16 (stored as uint16_t bit patterns)
// with Round-to-Nearest-Even (RNE) and quiet-NaN preservation.
// Uses AVX-512 vcvtne2ps2bf16 directly under -march=znver4, or via runtime
// CPUID dispatch on x86_64 builds targeting x86-64-v3.
inline void gcta_f32_to_bf16_rne(const float* src, uint16_t* dst, size_t n) noexcept {
    if (n == 0) return;
#if GCTA_CPU_x86 && (defined(__GNUC__) || defined(__clang__))
  #if defined(__AVX512BF16__) && defined(__AVX512F__)
    gcta_cpu_detail::f32_to_bf16_rne_avx512bf16(src, dst, n);
  #else
    static const bool has_avx512bf16 =
        __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512bf16");
    if (has_avx512bf16) {
        gcta_cpu_detail::f32_to_bf16_rne_avx512bf16(src, dst, n);
        return;
    }
    gcta_cpu_detail::f32_to_bf16_rne_scalar(src, dst, n);
  #endif
#else
    gcta_cpu_detail::f32_to_bf16_rne_scalar(src, dst, n);
#endif
}

// Convert n bfloat16 values (uint16_t bit patterns) to float32 (exact zero-extension).
inline void gcta_bf16_to_f32(const uint16_t* src, float* dst, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t u = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(dst + i, &u, sizeof(u));
    }
}

// Compute column-major C_f32(m x n, ldc) = alpha * opA(A_bf16) * opB(B_bf16) + beta * C_f32
// where transA, transB are 'N'/'n' or 'T'/'t'.
// Under GCTA_HAS_AOCL_BF16_GEMM, dispatches directly to Zen 4 AVX-512 BF16
// aocl_gemm_bf16bf16f32of32 via row-major transpose equivalence:
//   C_row(n x m, ldc) = alpha * opB(B_row) * opA(A_row) + beta * C_row.
// Otherwise upcasts bounded tiles to float32 and calls cblas_sgemm.
inline void gcta_gemm_bf16bf16f32_colmajor(
    char transA, char transB,
    gcta_blas_int m, gcta_blas_int n, gcta_blas_int k,
    float alpha,
    const uint16_t* A_bf16, gcta_blas_int lda,
    const uint16_t* B_bf16, gcta_blas_int ldb,
    float beta,
    float* C_f32, gcta_blas_int ldc)
{
    if (m <= 0 || n <= 0) return;
    if (k <= 0) {
        if (beta == 0.0f) {
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(m) * n >= 65536)
            for (gcta_blas_int j = 0; j < n; ++j) {
                std::memset(C_f32 + static_cast<size_t>(j) * ldc, 0, static_cast<size_t>(m) * sizeof(float));
            }
        } else if (beta != 1.0f) {
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(m) * n >= 65536)
            for (gcta_blas_int j = 0; j < n; ++j) {
                float* col = C_f32 + static_cast<size_t>(j) * ldc;
                for (gcta_blas_int i = 0; i < m; ++i) {
                    col[i] *= beta;
                }
            }
        }
        return;
    }

    const bool tA = (transA == 'T' || transA == 't');
    const bool tB = (transB == 'T' || transB == 't');

#if GCTA_HAS_AOCL_BF16_GEMM
    using Traits = gcta_cpu_detail::AoclGemm17Traits<decltype(&aocl_gemm_bf16bf16f32of32)>;
    using a_ptr_t = typename Traits::a_ptr_t;
    using b_ptr_t = typename Traits::b_ptr_t;
    static_assert(sizeof(std::remove_cv_t<std::remove_pointer_t<a_ptr_t>>) == sizeof(uint16_t),
                  "aocl_gemm_bf16bf16f32of32 A element size must be 2 bytes");
    static_assert(sizeof(std::remove_cv_t<std::remove_pointer_t<b_ptr_t>>) == sizeof(uint16_t),
                  "aocl_gemm_bf16bf16f32of32 B element size must be 2 bytes");
    aocl_gemm_bf16bf16f32of32('r',
                              tB ? 't' : 'n',
                              tA ? 't' : 'n',
                              n, m, k,
                              alpha,
                              reinterpret_cast<a_ptr_t>(B_bf16), ldb, 'n',
                              reinterpret_cast<b_ptr_t>(A_bf16), lda, 'n',
                              beta,
                              C_f32, ldc,
                              nullptr);
#else
    constexpr gcta_blas_int kBlock = 2048;
    constexpr gcta_blas_int mBlock = 4096;
    constexpr gcta_blas_int nBlock = 4096;
    const gcta_blas_int max_mb = std::min(m, mBlock);
    const gcta_blas_int max_nb = std::min(n, nBlock);
    const gcta_blas_int max_kb = std::min(k, kBlock);
    std::vector<float> buf_A(static_cast<size_t>(max_mb) * static_cast<size_t>(max_kb));
    std::vector<float> buf_B(static_cast<size_t>(max_nb) * static_cast<size_t>(max_kb));

    for (gcta_blas_int j0 = 0; j0 < n; j0 += nBlock) {
        const gcta_blas_int nb = std::min(nBlock, n - j0);
        for (gcta_blas_int i0 = 0; i0 < m; i0 += mBlock) {
            const gcta_blas_int mb = std::min(mBlock, m - i0);
            for (gcta_blas_int k0 = 0; k0 < k; k0 += kBlock) {
                const gcta_blas_int kb = std::min(kBlock, k - k0);
                gcta_blas_int ldA_sub = mb;
                if (!tA) {
                    // A is col-major (m x k, lda): copy kb columns of length mb
                    #pragma omp parallel for schedule(static) if (static_cast<size_t>(mb) * kb >= 65536)
                    for (gcta_blas_int kk = 0; kk < kb; ++kk) {
                        gcta_bf16_to_f32(
                            A_bf16 + i0 + static_cast<size_t>(k0 + kk) * lda,
                            buf_A.data() + static_cast<size_t>(kk) * mb,
                            static_cast<size_t>(mb));
                    }
                    ldA_sub = mb;
                } else {
                    // A is col-major (k x m, lda): copy mb columns of length kb
                    #pragma omp parallel for schedule(static) if (static_cast<size_t>(mb) * kb >= 65536)
                    for (gcta_blas_int ii = 0; ii < mb; ++ii) {
                        gcta_bf16_to_f32(
                            A_bf16 + k0 + static_cast<size_t>(i0 + ii) * lda,
                            buf_A.data() + static_cast<size_t>(ii) * kb,
                            static_cast<size_t>(kb));
                    }
                    ldA_sub = kb;
                }

                gcta_blas_int ldB_sub = kb;
                if (!tB) {
                    // B is col-major (k x n, ldb): copy nb columns of length kb
                    #pragma omp parallel for schedule(static) if (static_cast<size_t>(nb) * kb >= 65536)
                    for (gcta_blas_int jj = 0; jj < nb; ++jj) {
                        gcta_bf16_to_f32(
                            B_bf16 + k0 + static_cast<size_t>(j0 + jj) * ldb,
                            buf_B.data() + static_cast<size_t>(jj) * kb,
                            static_cast<size_t>(kb));
                    }
                    ldB_sub = kb;
                } else {
                    // B is col-major (n x k, ldb): copy kb columns of length nb
                    #pragma omp parallel for schedule(static) if (static_cast<size_t>(nb) * kb >= 65536)
                    for (gcta_blas_int kk = 0; kk < kb; ++kk) {
                        gcta_bf16_to_f32(
                            B_bf16 + j0 + static_cast<size_t>(k0 + kk) * ldb,
                            buf_B.data() + static_cast<size_t>(kk) * nb,
                            static_cast<size_t>(nb));
                    }
                    ldB_sub = nb;
                }

                const float cur_beta = (k0 == 0) ? beta : 1.0f;
                cblas_sgemm(CblasColMajor,
                            tA ? CblasTrans : CblasNoTrans,
                            tB ? CblasTrans : CblasNoTrans,
                            mb, nb, kb,
                            alpha,
                            buf_A.data(), ldA_sub,
                            buf_B.data(), ldB_sub,
                            cur_beta,
                            C_f32 + i0 + static_cast<size_t>(j0) * ldc, ldc);
            }
        }
    }
#endif
}

inline void gcta_gemm_bf16bf16f32_colmajor(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    gcta_blas_int m, gcta_blas_int n, gcta_blas_int k,
    float alpha,
    const uint16_t* A_bf16, gcta_blas_int lda,
    const uint16_t* B_bf16, gcta_blas_int ldb,
    float beta,
    float* C_f32, gcta_blas_int ldc)
{
    gcta_gemm_bf16bf16f32_colmajor(
        transA == CblasNoTrans ? 'N' : 'T',
        transB == CblasNoTrans ? 'N' : 'T',
        m, n, k, alpha, A_bf16, lda, B_bf16, ldb, beta, C_f32, ldc);
}

#endif  //END GCTA_CPU_H
