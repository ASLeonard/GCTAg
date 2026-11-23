/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * RemlEngine — free-function REML engine for the v2 MLMALoco path.
 * Single-GRM, univariate, non-bivariate, non-within-family REML only.
 *
 * Extracted from main/est_hsq.cpp with the following substitutions:
 *   this->_foo       → ctx.foo
 *   eigenMatrix      → RemlMat  (= Eigen::MatrixXd)
 *   eigenVector      → RemlVec  (= Eigen::VectorXd)
 *   _bivar_reml      → false (branches removed)
 *   _within_family   → false (branches removed)
 */

#include "RemlEngine.hpp"
#include "RemlCtx.hpp"
#include "RemlState.hpp"
#include "Matrix.hpp"
#include "Logger.h"
#include "cpu.h"
#include "mem.hpp"
#include "mlma_woodbury.hpp"
#include "symmetric_eigendecomp.hpp"
#include "chunked_grm_matvec.hpp"

#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>


// ──────────────────────────────────────────────────────────────────────────────
// Local type aliases (double-precision throughout)
// ──────────────────────────────────────────────────────────────────────────────
using RemlMat = Eigen::MatrixXd;
using RemlVec = Eigen::VectorXd;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (file-local, not exposed via RemlEngine.hpp)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Multiply upper-triangle-only float32 symmetric matrix K (n x n) by a double
// vector or matrix, dispatching to upper-triangular tiled float32 SGEMM/SSYMV
// without widening K.
RemlVec symm_upper_f32_vec(const Eigen::MatrixXf& K, const Eigen::Ref<const RemlVec>& x) {
    const Eigen::VectorXf xf = x.cast<float>();
    const Eigen::MatrixXf yf = gcta_chunked::dense_upper_symm_f32(K, xf, /*use_bf16=*/false);
    return yf.col(0).cast<double>();
}

void symm_upper_f32_mat_into(const Eigen::MatrixXf& K,
                             const Eigen::Ref<const RemlMat>& X,
                             RemlMat& out) {
    const Eigen::Index n = K.rows();
    const Eigen::Index cols = X.cols();
    out.resize(n, cols);
    if (n == 0 || cols == 0) return;

    constexpr Eigen::Index kPanel = 256;
    Eigen::MatrixXf Xf_panel;
    for (Eigen::Index c0 = 0; c0 < cols; c0 += kPanel) {
        const Eigen::Index bw = std::min(kPanel, cols - c0);
        Xf_panel.resize(n, bw);
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(bw) >= 32768)
        for (Eigen::Index j = 0; j < bw; ++j) {
            const double* __restrict__ src = X.col(c0 + j).data();
            float* __restrict__ dst = Xf_panel.col(j).data();
            #pragma omp simd
            for (Eigen::Index i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
        }
        const Eigen::MatrixXf Yf_panel = gcta_chunked::dense_upper_symm_f32(K, Xf_panel, /*use_bf16=*/false);
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(bw) >= 32768)
        for (Eigen::Index j = 0; j < bw; ++j) {
            const float* __restrict__ src = Yf_panel.col(j).data();
            double* __restrict__ dst = out.col(c0 + j).data();
            #pragma omp simd
            for (Eigen::Index i = 0; i < n; ++i) dst[i] = static_cast<double>(src[i]);
        }
    }
}

RemlMat symm_upper_f32_mat(const Eigen::MatrixXf& K, const Eigen::Ref<const RemlMat>& X) {
    RemlMat out;
    symm_upper_f32_mat_into(K, X, out);
    return out;
}

// Chunked float32-preferred matvec helper matching the dense float32 GRM precision
// while halving tile bandwidth/memory and running SGEMM instead of DGEMM.
void chunked_grm_matvec_into(const RemlCtx& ctx,
                             const Eigen::Ref<const RemlMat>& Z,
                             RemlMat& out) {
    const Eigen::Index n = ctx.n;
    const Eigen::Index cols = Z.cols();
    out.resize(n, cols);
    if (n == 0 || cols == 0) return;

    if (ctx.grm_tile_reader_f32) {
        Eigen::MatrixXf Zf(n, cols);
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(cols) >= 32768)
        for (Eigen::Index j = 0; j < cols; ++j) {
            const double* __restrict__ src = Z.col(j).data();
            float* __restrict__ dst = Zf.col(j).data();
            #pragma omp simd
            for (Eigen::Index i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
        }
        const Eigen::MatrixXf Yf = gcta_chunked::chunked_symmetric_matvec_f32(
            ctx.grm_tile_reader_f32, ctx.n, ctx.grm_chunk_rows, Zf);
        #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(cols) >= 32768)
        for (Eigen::Index j = 0; j < cols; ++j) {
            const float* __restrict__ src = Yf.col(j).data();
            double* __restrict__ dst = out.col(j).data();
            #pragma omp simd
            for (Eigen::Index i = 0; i < n; ++i) dst[i] = static_cast<double>(src[i]);
        }
    } else {
        out = gcta_chunked::chunked_symmetric_matvec(
            ctx.grm_tile_reader, ctx.n, ctx.grm_chunk_rows, RemlMat(Z));
    }
}

RemlVec chunked_grm_matvec_vec(const RemlCtx& ctx, const Eigen::Ref<const RemlVec>& v) {
    if (ctx.grm_tile_reader_f32) {
        const Eigen::VectorXf vf = v.cast<float>();
        const Eigen::MatrixXf yf = gcta_chunked::chunked_symmetric_matvec_f32(
            ctx.grm_tile_reader_f32, ctx.n, ctx.grm_chunk_rows, vf);
        return yf.col(0).cast<double>();
    }
    return gcta_chunked::chunked_symmetric_matvec(
        ctx.grm_tile_reader, ctx.n, ctx.grm_chunk_rows, v);
}

// ──────────────────────────────────────────────────────────────────────────────
// Spectral-subspace Woodbury REML representation
//
// Any vector generated by {K_hat, V^{-1}, P} from {X, y} lies in
//   span(Uk) (+) span(X_perp, y_perp)
// where Uk^T Uk = I_k, X_perp = X - Uk*(Uk^T X), y_perp = y - Uk*(Uk^T y):
//   v = Uk * par + X_perp * alpha + beta * y_perp.
//
// Precomputing (UkTX, UkTy, XtX_perp, Xty_perp, yty_perp) ONCE before iter 0
// allows HE warm-start, calcu_P_impl, calcu_tr_PA_woodbury, ai_reml, em_reml,
// post-hoc tail correction, and final fixed effects to execute purely in
// O(k * c^2) (<0.5 ms/iter, 0 Uk scans after iter 0).
// ──────────────────────────────────────────────────────────────────────────────
struct WoodburySubspaceVec {
    RemlVec par;    // k
    RemlVec alpha;  // X_c
    double  beta = 0.0;
};

void ensure_woodbury_subspace_cache(RemlCtx& ctx) {
    const int k = ctx.woodbury_basis_rank_;
    const int c = ctx.X_c;
    const int n = ctx.n;
    if (ctx.wb_subspace_ready
        && ctx.UkTX.rows() == k && ctx.UkTX.cols() == c
        && static_cast<int>(ctx.UkTy.size()) == k
        && ctx.wb_XtX_perp.rows() == c && ctx.wb_XtX_perp.cols() == c
        && static_cast<int>(ctx.wb_Xty_perp.size()) == c) {
        return;
    }

    // Fused [X, y] projection: 2 GEMMs over Uk once per REML run.
    Eigen::MatrixXf XY_f(n, c + 1);
    #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(c) >= 65536)
    for (int j = 0; j < c; ++j) {
        const double* __restrict__ src = ctx.X.col(j).data();
        float* __restrict__ dst = XY_f.col(j).data();
        #pragma omp simd
        for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    }
    {
        const double* __restrict__ src = ctx.y.data();
        float* __restrict__ dst = XY_f.col(c).data();
        #pragma omp parallel for schedule(static) if (n >= 65536)
        for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    }

    const Eigen::MatrixXf UkTXY_f = ctx.Uk.transpose() * XY_f;  // k x (c + 1)
    ctx.UkTX = UkTXY_f.leftCols(c).cast<double>();
    ctx.UkTy = UkTXY_f.col(c).cast<double>();

    // Explicit perpendicular residuals ensure wb_XtX_perp is PSD even when
    // covariates (e.g. PCs) lie inside span(Uk).
    const Eigen::MatrixXf proj_f = ctx.Uk * UkTXY_f;            // n x (c + 1)
    RemlMat X_perp(n, c);
    #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(c) >= 65536)
    for (int j = 0; j < c; ++j) {
        const double* __restrict__ xcol = ctx.X.col(j).data();
        const float*  __restrict__ pcol = proj_f.col(j).data();
        double*       __restrict__ dcol = X_perp.col(j).data();
        #pragma omp simd
        for (int i = 0; i < n; ++i) dcol[i] = xcol[i] - static_cast<double>(pcol[i]);
    }
    RemlVec y_perp(n);
    {
        const double* __restrict__ ycol = ctx.y.data();
        const float*  __restrict__ pcol = proj_f.col(c).data();
        double*       __restrict__ dcol = y_perp.data();
        #pragma omp parallel for schedule(static) if (n >= 65536)
        for (int i = 0; i < n; ++i) dcol[i] = ycol[i] - static_cast<double>(pcol[i]);
    }

    ctx.wb_XtX_perp.resize(c, c);
    ctx.wb_XtX_perp.noalias() = X_perp.transpose() * X_perp;
    ctx.wb_Xty_perp.resize(c);
    ctx.wb_Xty_perp.noalias() = X_perp.transpose() * y_perp;
    ctx.wb_yty_perp = y_perp.squaredNorm();
    ctx.wb_subspace_ready = true;
}

double wb_inner(const RemlCtx& ctx, const WoodburySubspaceVec& u, const WoodburySubspaceVec& v) {
    const double par_dot = u.par.dot(v.par);
    const double perp_dot =
        u.alpha.dot(ctx.wb_XtX_perp * v.alpha)
        + u.beta * ctx.wb_Xty_perp.dot(v.alpha)
        + v.beta * ctx.wb_Xty_perp.dot(u.alpha)
        + u.beta * v.beta * ctx.wb_yty_perp;
    return par_dot + perp_dot;
}

WoodburySubspaceVec wb_apply_K(const RemlCtx& ctx, const WoodburySubspaceVec& v) {
    WoodburySubspaceVec out;
    out.par   = ctx.dk.cwiseProduct(v.par);
    out.alpha = ctx.lambda_tail * v.alpha;
    out.beta  = ctx.lambda_tail * v.beta;
    return out;
}

WoodburySubspaceVec wb_apply_P(const RemlCtx& ctx, const WoodburySubspaceVec& v) {
    const double inv_s2 = 1.0 / ctx.sigma2_eff;
    const RemlVec w_par   = ((1.0 - ctx.ck.array()) * inv_s2).matrix().cwiseProduct(v.par);
    const RemlVec w_alpha = inv_s2 * v.alpha;
    const double  w_beta  = inv_s2 * v.beta;

    RemlVec a(ctx.X_c);
    a.noalias()  = ctx.UkTX.transpose() * w_par;
    a.noalias() += ctx.wb_XtX_perp * w_alpha;
    a.noalias() += w_beta * ctx.wb_Xty_perp;

    const RemlVec b_v = ctx.Xt_Vi_X_i.selfadjointView<Eigen::Lower>() * a;

    WoodburySubspaceVec out;
    out.par = w_par;
    out.par.noalias() -= ctx.Uk_Vi_X * b_v;
    out.alpha = w_alpha - inv_s2 * b_v;
    out.beta  = w_beta;
    return out;
}

WoodburySubspaceVec wb_y_vec(const RemlCtx& ctx) {
    WoodburySubspaceVec yv;
    yv.par = ctx.UkTy;
    yv.alpha = RemlVec::Zero(ctx.X_c);
    yv.beta = 1.0;
    return yv;
}

// Woodbury n-vector helpers (retained for any direct non-subspace path)
RemlMat woodbury_basis_KZ(const RemlCtx& ctx, const RemlMat& Z) {
    RemlMat UkZ = (ctx.Uk.transpose() * Z.cast<float>()).cast<double>();
    UkZ.array().colwise() *= (ctx.dk.array() - ctx.lambda_tail);
    return ctx.lambda_tail * Z + (ctx.Uk * UkZ.cast<float>()).cast<double>();
}

RemlVec woodbury_basis_Viv(const RemlCtx& ctx, const RemlVec& v) {
    RemlVec Ukv = (ctx.Uk.transpose() * v.cast<float>()).cast<double>();
    Ukv.array() *= ctx.ck.array();
    return (v - (ctx.Uk * Ukv.cast<float>()).cast<double>()) / ctx.sigma2_eff;
}

RemlMat woodbury_basis_ViZ(const RemlCtx& ctx, const RemlMat& Z) {
    RemlMat UkZ = (ctx.Uk.transpose() * Z.cast<float>()).cast<double>();
    UkZ.array().colwise() *= ctx.ck.array();
    return (Z - (ctx.Uk * UkZ.cast<float>()).cast<double>()) / ctx.sigma2_eff;
}

// Implicit P*v without materialising n×n P
RemlVec applyP_vec(const RemlCtx& ctx, const RemlVec& v) {
    RemlVec w;
    if (ctx.Vi_use_woodbury_basis) {
        w = woodbury_basis_Viv(ctx, v);
    } else if (ctx.Vi_use_llt) {
        w = v;
        ctx.Vi_L.triangularView<Eigen::Upper>().adjoint().solveInPlace(w); // U^T z = v
        ctx.Vi_L.triangularView<Eigen::Upper>().solveInPlace(w);          // U w = z
    } else {
        w = RemlVec(ctx.Vi.selfadjointView<Eigen::Upper>() * v);
    }
    RemlVec a = ctx.Vi_X.transpose() * v;
    RemlVec b = ctx.Xt_Vi_X_i.selfadjointView<Eigen::Lower>() * a;
    w.noalias() -= ctx.Vi_X * b;
    return w;
}

// Implicit P*Z written directly into W (supports &W == &Z when ctx.Vi_use_llt;
// requires &W != &Z when using dense ctx.Vi). Computes Vi_X^T * Z BEFORE
// modifying W so in-place LLT solves remain exact with zero extra n x cols temp.
void applyP_mat_into(const RemlCtx& ctx, const RemlMat& Z, RemlMat& W) {
    const RemlMat A = ctx.Vi_X.transpose() * Z;
    if (ctx.Vi_use_woodbury_basis) {
        W = woodbury_basis_ViZ(ctx, Z);
    } else if (ctx.Vi_use_llt) {
        if (&W != &Z) {
            W.resize(Z.rows(), Z.cols());
            const size_t col_bytes = static_cast<size_t>(Z.rows()) * sizeof(double);
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(Z.rows()) * static_cast<size_t>(Z.cols()) >= 32768)
            for (Eigen::Index j = 0; j < Z.cols(); ++j) {
                std::memcpy(W.col(j).data(), Z.col(j).data(), col_bytes);
            }
        }
        ctx.Vi_L.triangularView<Eigen::Upper>().adjoint().solveInPlace(W); // U^T z = Z
        ctx.Vi_L.triangularView<Eigen::Upper>().solveInPlace(W);          // U w = z
    } else {
        W.resize(Z.rows(), Z.cols());
        W.noalias() = ctx.Vi.selfadjointView<Eigen::Upper>() * Z;
    }
    W.noalias() -= ctx.Vi_X * (ctx.Xt_Vi_X_i.selfadjointView<Eigen::Lower>() * A);
}

RemlMat applyP_mat(const RemlCtx& ctx, const RemlMat& Z) {
    RemlMat W;
    applyP_mat_into(ctx, Z, W);
    return W;
}

bool inverse_H(RemlCtx& ctx, RemlMat& H) {
    double d_buf = 0.0;
    INVmethod method = (ctx.reml_inv_mtd == 0) ? INV_LLT : static_cast<INVmethod>(ctx.reml_inv_mtd);
    int rank = 0;
    return SquareMatrixInverse(H, d_buf, rank, method);
}

void bend_V(RemlCtx& ctx, RemlMat& Vi) {
    const int bv_n = static_cast<int>(Vi.rows());
    if (bv_n > 5000)
        LOGGER.w(0, "bend_V: O(n³) full eigendecomposition on n=" + std::to_string(bv_n) +
                    " matrix — this may take several minutes.");
    Eigen::SelfAdjointEigenSolver<RemlMat> eigensolver(Vi);
    RemlVec eval = eigensolver.eigenvalues();
    // bending_eigenval inline:
    double eval_m = eval.mean();
    if (eval.minCoeff() > 1e-6) {
        // no bending needed — just reconstruct
    } else {
        double S = 0.0, P = 0.0;
        for (int j = 0; j < eval.size(); j++) {
            if (eval[j] >= 0) continue;
            S += eval[j];
            P = -eval[j];
        }
        double W = S * S / ctx.reml_diag_mul + 1.0;
        for (int j = 0; j < eval.size(); j++) {
            if (eval[j] >= 0) continue;
            eval[j] = P * (S - eval[j]) * (S - eval[j]) / W;
        }
        eval *= eval_m / eval.mean();
    }
    eval.array() = 1.0 / eval.array();
    Vi = eigensolver.eigenvectors() * eval.asDiagonal() * eigensolver.eigenvectors().transpose();
}

int constrain_varcmp(const RemlCtx& ctx, RemlVec& varcmp) {
    const int m = static_cast<int>(ctx.r_indx.size());
    double delta = 0.0;
    constexpr double constr_scale = 1e-6;
    int num = 0;
    std::vector<int> constrain(m, 0);
    for (int i = 0; i < m; i++) {
        if (varcmp[i] < 0) {
            delta += ctx.y_Ssq * constr_scale - varcmp[i];
            varcmp[i] = ctx.y_Ssq * constr_scale;
            constrain[i] = 1;
            num++;
        }
    }
    if (num < m) {
        delta /= (m - num);
        for (int i = 0; i < m; i++)
            if (constrain[i] < 1 && varcmp[i] > delta) varcmp[i] -= delta;
    }
    return num;
}

// Returns true if variance components successfully warm-started via HE, false if not (e.g. single-GRM but priors specified)
bool init_varcomp(RemlCtx& ctx,
                  const std::vector<double>& priors_var,
                  const std::vector<double>& priors,
                  RemlVec& varcmp) {
    const int m = static_cast<int>(ctx.r_indx.size());
    varcmp = RemlVec::Zero(m);

    if (!priors_var.empty()) {
        for (int i = 0; i < m - 1; i++) varcmp[i] = priors_var[i];
        if ((int)priors_var.size() < m)
            varcmp[m - 1] = ctx.y_Ssq - varcmp.sum();
        else
            varcmp[m - 1] = priors_var[m - 1];
    } else if (!priors.empty()) {
        double d_buf = 0.0;
        for (int i = 0; i < m - 1; i++) {
            varcmp[i] = priors[i] * ctx.y_Ssq;
            d_buf += priors[i];
        }
        if (d_buf > 1.0) LOGGER.e(0, "\n  --reml-priors. The sum of all prior values should not exceed 1.0.");
        varcmp[m - 1] = (1.0 - d_buf) * ctx.y_Ssq;
    } else {
        varcmp.setConstant(ctx.y_Ssq / m);
    }

    // Single-GRM HE warm-start (applies to both Woodbury and Exact REML when no priors specified)
    if (!ctx.reml_no_HE_start && (int)ctx.r_indx.size() == 2 && priors.empty() && priors_var.empty()) {
        const int n = ctx.n;
        const int dof = n - ctx.X_c;
        if (dof <= 0) return false;

        double trK = 0.0, trK2 = 0.0;
        RemlMat XtKX, XtK2X;
        double yKy = 0.0, yy = 0.0;
        Eigen::LDLT<RemlMat> XtX_ldlt;

        if (ctx.Vi_use_woodbury_basis) {
            // Pure spectral-subspace HE warm-start: O(k * c^2), zero Uk scans after cache build.
            ensure_woodbury_subspace_cache(ctx);
            const int k = ctx.woodbury_basis_rank_;
            trK  = ctx.dk.sum() + static_cast<double>(n - k) * ctx.lambda_tail;
            trK2 = ctx.dk.squaredNorm()
                 + static_cast<double>(n - k) * (ctx.tail_d_var + ctx.lambda_tail * ctx.lambda_tail);

            const RemlMat XtX = ctx.UkTX.transpose() * ctx.UkTX + ctx.wb_XtX_perp;
            XtX_ldlt.compute(XtX);
            if (XtX_ldlt.info() != Eigen::Success) return false;

            const RemlVec Xty = ctx.UkTX.transpose() * ctx.UkTy + ctx.wb_Xty_perp;
            const RemlVec beta_ols = XtX_ldlt.solve(Xty);

            const RemlMat dk_UkTX = ctx.dk.asDiagonal() * ctx.UkTX;
            XtKX.noalias()  = ctx.UkTX.transpose() * dk_UkTX + ctx.lambda_tail * ctx.wb_XtX_perp;
            XtK2X.noalias() = dk_UkTX.transpose() * dk_UkTX
                            + (ctx.lambda_tail * ctx.lambda_tail) * ctx.wb_XtX_perp;

            WoodburySubspaceVec yr_sub;
            yr_sub.par   = ctx.UkTy - ctx.UkTX * beta_ols;
            yr_sub.alpha = -beta_ols;
            yr_sub.beta  = 1.0;
            const WoodburySubspaceVec Kyr_sub = wb_apply_K(ctx, yr_sub);
            yKy = wb_inner(ctx, yr_sub, Kyr_sub);
            yy  = wb_inner(ctx, yr_sub, yr_sub);
        } else {
            const RemlMat XtX = ctx.X.transpose() * ctx.X;
            XtX_ldlt.compute(XtX);
            if (XtX_ldlt.info() != Eigen::Success) return false;
            const RemlVec y_r = ctx.y - ctx.X * XtX_ldlt.solve(ctx.X.transpose() * ctx.y);

            RemlMat KX;
            RemlVec Ky(n);
            if (ctx.grm_chunk_rows > 0) {
                // Fused single-pass float32 matvec on [X, y_r] (+ trace_K, trace_K2 when !fp16),
                // replacing 4 full-file double passes (2 TB at N=500k) with 1 pass in f32!
                if (ctx.grm_tile_reader_f32) {
                    Eigen::MatrixXf XY_f(n, ctx.X_c + 1);
                    XY_f.leftCols(ctx.X_c) = ctx.X.cast<float>();
                    XY_f.col(ctx.X_c)      = y_r.cast<float>();
                    double* tr_ptr  = ctx.grm_fp16 ? nullptr : &trK;
                    double* tr2_ptr = ctx.grm_fp16 ? nullptr : &trK2;
                    if (ctx.grm_fp16) {
                        trK  = ctx.grm_fp16_trace_K;
                        trK2 = ctx.grm_fp16_trace_K2;
                    }
                    const Eigen::MatrixXf KXY_f = gcta_chunked::chunked_symmetric_matvec_f32(
                        ctx.grm_tile_reader_f32, n, ctx.grm_chunk_rows, XY_f, tr_ptr, tr2_ptr);
                    KX = KXY_f.leftCols(ctx.X_c).cast<double>();
                    Ky = KXY_f.col(ctx.X_c).cast<double>();
                } else {
                    if (ctx.grm_fp16) {
                        trK  = ctx.grm_fp16_trace_K;
                        trK2 = ctx.grm_fp16_trace_K2;
                    } else {
                        trK  = gcta_chunked::chunked_diagonal(ctx.grm_tile_reader, n, ctx.grm_chunk_rows).sum();
                        trK2 = gcta_chunked::chunked_trace_K_squared(ctx.grm_tile_reader, n, ctx.grm_chunk_rows);
                    }
                    RemlMat XY(n, ctx.X_c + 1);
                    XY.leftCols(ctx.X_c) = ctx.X;
                    XY.col(ctx.X_c)      = y_r;
                    const RemlMat KXY = gcta_chunked::chunked_symmetric_matvec(
                        ctx.grm_tile_reader, n, ctx.grm_chunk_rows, XY);
                    KX = KXY.leftCols(ctx.X_c);
                    Ky = KXY.col(ctx.X_c);
                }
            } else if (!ctx.A.empty() && ctx.A[ctx.r_indx[0]].size() > 0) {
                const auto& K = ctx.A[ctx.r_indx[0]];
                trK = K.diagonal().cast<double>().sum();
                const double diag_sq = K.diagonal().cast<double>().squaredNorm();
                double off_sq = 0.0;
                #pragma omp parallel for reduction(+:off_sq) schedule(guided)
                for (int j = 0; j < n; ++j)
                    off_sq += K.col(j).head(j).cast<double>().squaredNorm();
                trK2 = diag_sq + 2.0 * off_sq;
                // Fused [X, y_r] matvec: one single sweep over dense K instead of two.
                RemlMat XY(n, ctx.X_c + 1);
                XY.leftCols(ctx.X_c) = ctx.X;
                XY.col(ctx.X_c)      = y_r;
                const RemlMat KXY = symm_upper_f32_mat(K, XY);
                KX = KXY.leftCols(ctx.X_c);
                Ky = KXY.col(ctx.X_c);
            }
            if (trK2 > 0.0) {
                XtKX.noalias()  = ctx.X.transpose() * KX;
                XtK2X.noalias() = KX.transpose() * KX;
                yKy = y_r.dot(Ky);
                yy  = y_r.squaredNorm();
            }
        }

        if (trK2 > 0.0) {
            const RemlMat B_inv_XtKX = XtX_ldlt.solve(XtKX);
            const double trQK = trK - B_inv_XtKX.trace();
            const double trQKQK = trK2 - 2.0 * XtX_ldlt.solve(XtK2X).trace()
                + (B_inv_XtKX * B_inv_XtKX).trace();
            const double denom = static_cast<double>(dof) * trQKQK - trQK * trQK;
            const double scale = yy / static_cast<double>(dof);
            if (denom > 1e-10 && std::isfinite(scale) && scale > 0.0) {
                const double sg_he = (static_cast<double>(dof) * yKy - trQK * yy) / denom;
                const double se_he = (trQKQK * yy - trQK * yKy) / denom;
                if (std::isfinite(sg_he) && std::isfinite(se_he)
                    && sg_he > 0.0 && se_he > 0.0
                    && sg_he <= 100.0 * scale && se_he <= 100.0 * scale) {
                    varcmp(0) = std::max(sg_he, 0.01 * scale);
                    varcmp(1) = std::max(se_he, 0.01 * scale);
                    LOGGER << "REML: used single-GRM HE warm-start for variance components = " << varcmp.transpose() << std::endl;
                    return true;
                } else {
                    LOGGER.w(0, "single-GRM HE warm-start produced implausible variance component(s) (sg=" + std::to_string(sg_he) + ", se=" + std::to_string(se_he) + ") -- ignoring.");
                }
            }
        }
    }
    return false;
}


// Fill ctx.Vi (upper triangle + diagonal) with sum_ci varcmp[ci] * A[ci].
// Directly assigns entries in parallel (eliminating the single-threaded setZero()
// sweep and triangular load imbalance).
void assemble_V_upper(RemlCtx& ctx, const RemlVec& varcmp) {
    const int num_comp = static_cast<int>(ctx.r_indx.size());
    double id_diag_sum = 0.0;
    for (int ci = 0; ci < num_comp; ++ci) {
        if (ctx.grm_chunk_rows > 0 && ci == 0) continue;
        if (ctx.A[ctx.r_indx[ci]].size() == 0) id_diag_sum += varcmp[ci];
    }

    if (ctx.grm_chunk_rows > 0) {
        const double sg2 = varcmp[0];
        const int eff_block = std::min(ctx.n, std::max(1, ctx.grm_chunk_rows));
        if (ctx.grm_tile_reader_f32 && ctx.grm_tile_reader_f32.has_transposed()) {
            const std::vector<int> bands = ctx.grm_tile_reader_f32.compute_row_bands(ctx.n, eff_block);
            const int num_bands = static_cast<int>(bands.size()) - 1;
            auto sweep = ctx.grm_tile_reader_f32.begin_sweep(bands);
            for (int b = 0; b < num_bands; ++b) {
                if (sweep) sweep->acquire(b);
                const int rs = bands[b], re = bands[b + 1];
                const int rows_i = re - rs;
                for (int cs = 0; cs < rs; cs += eff_block) {
                    const int ce = std::min(cs + eff_block, rs);
                    const int cols_j = ce - cs;
                    const auto tile_T = ctx.grm_tile_reader_f32.read_T(rs, re, cs, ce);  // cols_j x rows_i
                    #pragma omp parallel for schedule(static)
                    for (int col = 0; col < rows_i; ++col) {
                        const float* __restrict__ src = tile_T.col(col).data();
                        double* __restrict__ dst = ctx.Vi.col(rs + col).data() + cs;
                        #pragma omp simd
                        for (int r = 0; r < cols_j; ++r) dst[r] = sg2 * static_cast<double>(src[r]);
                    }
                }
                {
                    const auto tile_T = ctx.grm_tile_reader_f32.read_T(rs, re, rs, re);  // rows_i x rows_i, UPPER valid
                    #pragma omp parallel for schedule(guided)
                    for (int col = 0; col < rows_i; ++col) {
                        const float* __restrict__ src = tile_T.col(col).data();
                        double* __restrict__ dst = ctx.Vi.col(rs + col).data() + rs;
                        #pragma omp simd
                        for (int r = 0; r <= col; ++r) dst[r] = sg2 * static_cast<double>(src[r]);
                        dst[col] += id_diag_sum;
                    }
                }
                if (sweep) sweep->release(b);
            }
        } else {
            const std::vector<int> bands = ctx.grm_tile_reader.compute_row_bands(ctx.n, eff_block);
            const int num_bands = static_cast<int>(bands.size()) - 1;
            auto sweep = ctx.grm_tile_reader.begin_sweep(bands);
            const bool use_T = ctx.grm_tile_reader.has_transposed();
            for (int b = 0; b < num_bands; ++b) {
                if (sweep) sweep->acquire(b);
                const int rs = bands[b], re = bands[b + 1];
                const int rows_i = re - rs;
                for (int cs = 0; cs < rs; cs += eff_block) {
                    const int ce = std::min(cs + eff_block, rs);
                    const int cols_j = ce - cs;
                    if (use_T) {
                        const auto tile_T = ctx.grm_tile_reader.read_T(rs, re, cs, ce);
                        #pragma omp parallel for schedule(static)
                        for (int col = 0; col < rows_i; ++col) {
                            const double* __restrict__ src = tile_T.col(col).data();
                            double* __restrict__ dst = ctx.Vi.col(rs + col).data() + cs;
                            #pragma omp simd
                            for (int r = 0; r < cols_j; ++r) dst[r] = sg2 * src[r];
                        }
                    } else {
                        const auto tile = ctx.grm_tile_reader(rs, re, cs, ce);
                        #pragma omp parallel for schedule(static)
                        for (int col = 0; col < rows_i; ++col) {
                            ctx.Vi.col(rs + col).segment(cs, cols_j).noalias() =
                                sg2 * tile.row(col).transpose();
                        }
                    }
                }
                {
                    if (use_T) {
                        const auto tile_T = ctx.grm_tile_reader.read_T(rs, re, rs, re);
                        #pragma omp parallel for schedule(guided)
                        for (int col = 0; col < rows_i; ++col) {
                            const double* __restrict__ src = tile_T.col(col).data();
                            double* __restrict__ dst = ctx.Vi.col(rs + col).data() + rs;
                            #pragma omp simd
                            for (int r = 0; r <= col; ++r) dst[r] = sg2 * src[r];
                            dst[col] += id_diag_sum;
                        }
                    } else {
                        const auto tile = ctx.grm_tile_reader(rs, re, rs, re);
                        #pragma omp parallel for schedule(guided)
                        for (int col = 0; col < rows_i; ++col) {
                            ctx.Vi.col(rs + col).segment(rs, col + 1).noalias() =
                                sg2 * tile.row(col).head(col + 1).transpose();
                            ctx.Vi(rs + col, rs + col) += id_diag_sum;
                        }
                    }
                }
                if (sweep) sweep->release(b);
            }
        }
        return;
    }

    // Dense path: assign first non-empty matrix component directly (no serial setZero)
    // and accumulate remaining components + identity diagonal inside one guided pass.
    std::vector<int> dense_comps;
    dense_comps.reserve(static_cast<size_t>(num_comp));
    for (int ci = 0; ci < num_comp; ++ci) {
        if (ctx.A[ctx.r_indx[ci]].size() > 0) dense_comps.push_back(ci);
    }

    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < ctx.n; ++j) {
        double* __restrict__ dst = ctx.Vi.col(j).data();
        if (dense_comps.empty()) {
            std::memset(dst, 0, static_cast<size_t>(j + 1) * sizeof(double));
        } else {
            const int ci0 = dense_comps[0];
            const double w0 = varcmp[ci0];
            const float* __restrict__ s0 = ctx.A[ctx.r_indx[ci0]].col(j).data();
            #pragma omp simd
            for (int r = 0; r <= j; ++r) dst[r] = w0 * static_cast<double>(s0[r]);
            for (size_t k = 1; k < dense_comps.size(); ++k) {
                const int ci = dense_comps[k];
                const double wk = varcmp[ci];
                const float* __restrict__ sk = ctx.A[ctx.r_indx[ci]].col(j).data();
                #pragma omp simd
                for (int r = 0; r <= j; ++r) dst[r] += wk * static_cast<double>(sk[r]);
            }
        }
        dst[j] += id_diag_sum;
    }
}

// Returns false if V is not positive-definite and inversion failed.
bool calcu_Vi(RemlCtx& ctx, RemlVec& prev_varcmp, double& logdet, int& iter, bool factorize_only) {
    (void)iter;
    ctx.Vi_use_llt = false;

    if (ctx.Vi_use_woodbury_basis) {
        if (!factorize_only)
            LOGGER.e(0, "Woodbury REML is incompatible with explicit V^{-1} materialisation.");
        ctx.Vi.resize(0, 0);
        const double sg2 = prev_varcmp[0];
        const double se2 = prev_varcmp[ctx.r_indx.size() - 1];
        ctx.sigma2_eff = sg2 * ctx.lambda_tail + se2;
        if (ctx.sigma2_eff <= 0.0) return false;
        ctx.sg2 = sg2;
        const int k = ctx.woodbury_basis_rank_;
        ctx.ck.resize(k);
        logdet = static_cast<double>(ctx.n - k) * std::log(ctx.sigma2_eff);
        for (int j = 0; j < k; ++j) {
            const double delta     = std::max(0.0, ctx.dk[j] - ctx.lambda_tail);
            const double sig_delta = sg2 * delta;
            ctx.ck[j] = sig_delta / (ctx.sigma2_eff + sig_delta);
            logdet += std::log(ctx.sigma2_eff + sig_delta);
        }
        if (ctx.tail_d_var > 0.0) {
            const double r = sg2 / ctx.sigma2_eff;
            logdet -= 0.5 * r * r * static_cast<double>(ctx.n - k) * ctx.tail_d_var;
        }
        return true;
    }

    // Dense path: assemble V (upper triangle)
    if (factorize_only && static_cast<int>(ctx.r_indx.size()) > 1)
        ctx.Vi.swap(ctx.Vi_L);
    ctx.Vi.resize(ctx.n, ctx.n);

    if (ctx.r_indx.size() == 1) {
        #pragma omp parallel for schedule(guided)
        for (int j = 0; j < ctx.n; ++j) {
            if (j > 0) std::memset(ctx.Vi.col(j).data(), 0, static_cast<size_t>(j) * sizeof(double));
            ctx.Vi(j, j) = 1.0 / prev_varcmp[0];
        }
        logdet = ctx.n * std::log(prev_varcmp[0]);
    } else {
        assemble_V_upper(ctx, prev_varcmp);

        // LLT-only path (factorize_only, no diagV_adj)
        if (factorize_only && ctx.reml_diagV_adj == DiagVAdj::None && !ctx.reml_force_dense_vi) {
            gcta_blas_int blas_n = static_cast<gcta_blas_int>(ctx.n);
            if (gcta_dpotrf(blas_n, ctx.Vi.data(), blas_n, 'U') == 0) {
                logdet = 2.0 * ctx.Vi.diagonal().array().log().sum();
                ctx.Vi_L.swap(ctx.Vi);
                ctx.Vi.resize(0, 0);
                ctx.Vi_use_llt = true;
                return true;
            }
            ctx.Vi.resize(ctx.n, ctx.n);
            assemble_V_upper(ctx, prev_varcmp);
            LOGGER.w(0, "REML: final LLT factorization of V failed at convergence; falling back to dense inverse.");
        }

        INVmethod method_try = ctx.reml_inv_mtd ? static_cast<INVmethod>(ctx.reml_inv_mtd) : INV_LLT;
        int rank = 0;
        bool ret = true;

        if (method_try == INV_LLT && (!factorize_only || ctx.reml_force_dense_vi)) {
            gcta_blas_int blas_n_f = static_cast<gcta_blas_int>(ctx.n);
            bool llt_ok = (gcta_dpotrf(blas_n_f, ctx.Vi.data(), blas_n_f, 'U') == 0);
            if (llt_ok) {
                logdet = ctx.Vi.diagonal().array().square().log().sum();
                llt_ok = (gcta_dpotri(blas_n_f, ctx.Vi.data(), blas_n_f, 'U') == 0);
            }
            if (llt_ok) {
                return true;
            }
            ctx.Vi.resize(ctx.n, ctx.n);
            assemble_V_upper(ctx, prev_varcmp);
            ctx.Vi.triangularView<Eigen::Lower>() =
                ctx.Vi.triangularView<Eigen::Upper>().transpose();
            method_try = INV_LU;
        }

        if (!SquareMatrixInverse(ctx.Vi, logdet, rank, method_try)) {
            LOGGER << "Warning: V matrix is not invertible." << std::endl;
            if (ctx.reml_diagV_adj == DiagVAdj::Jitter) {
                LOGGER << "A small positive value is added to the diagonals. Results might not be reliable!" << std::endl;
                double d_buf = ctx.Vi.diagonal().mean() * ctx.reml_diag_mul;
                for (int j = 0; j < ctx.n; j++) ctx.Vi(j, j) += d_buf;
                if (!SquareMatrixInverse(ctx.Vi, logdet, rank, method_try)) {
                    LOGGER << "Still can't be inverted." << std::endl;
                    ret = false;
                }
            } else if (ctx.reml_diagV_adj == DiagVAdj::Bending) {
                LOGGER << "Switching to the bending approach." << std::endl;
                bend_V(ctx, ctx.Vi);
            } else {
                LOGGER.e(0, "the variance-covariance matrix V is not invertible.");
                ret = false;
            }
        }
        return ret;
    }
    return true;
}

double calcu_P_impl(RemlCtx& ctx, RemlMat* P) {
    if (ctx.Vi_use_woodbury_basis) {
        ensure_woodbury_subspace_cache(ctx);
        const double inv_s2 = 1.0 / ctx.sigma2_eff;
        ctx.Uk_Vi_X = ((1.0 - ctx.ck.array()) * inv_s2).matrix().asDiagonal() * ctx.UkTX;
        ctx.Xt_Vi_X_i.noalias()  = ctx.UkTX.transpose() * ctx.Uk_Vi_X;
        ctx.Xt_Vi_X_i.noalias() += inv_s2 * ctx.wb_XtX_perp;
        if (P) {
            const RemlMat ck_UkTX = ctx.ck.asDiagonal() * ctx.UkTX;
            ctx.Vi_X = (ctx.X - (ctx.Uk * ck_UkTX.cast<float>()).cast<double>()) * inv_s2;
        }
    } else if (ctx.Vi_use_llt) {
        ctx.Vi_X = ctx.X;
        ctx.Vi_L.triangularView<Eigen::Upper>().adjoint().solveInPlace(ctx.Vi_X); // U^T z = X
        ctx.Vi_L.triangularView<Eigen::Upper>().solveInPlace(ctx.Vi_X);          // U w = z
        ctx.Xt_Vi_X_i.noalias() = ctx.X.transpose() * ctx.Vi_X;
    } else {
        ctx.Vi_X.noalias() = ctx.Vi.selfadjointView<Eigen::Upper>() * ctx.X;
        ctx.Xt_Vi_X_i.noalias() = ctx.X.transpose() * ctx.Vi_X;
    }

    double logdet_Xt_Vi_X = 0.0;
    int rank = 0;
    INVmethod method = (ctx.reml_inv_mtd == 0) ? INV_LLT : static_cast<INVmethod>(ctx.reml_inv_mtd);
    if (!SquareMatrixInverse(ctx.Xt_Vi_X_i, logdet_Xt_Vi_X, rank, method))
        LOGGER.e(0, "the X'V^{-1}X matrix is not invertible. Please check covariates.");

    if (!P) return logdet_Xt_Vi_X;

    // Materialise V^{-1} if we need the full P matrix
    if (ctx.Vi_use_woodbury_basis) {
        ctx.Vi.resize(ctx.n, ctx.n);
        ctx.Vi.setIdentity();
        ctx.Vi /= ctx.sigma2_eff;
        RemlMat Uk_scaled = ctx.Uk.cast<double>();
        for (int j = 0; j < ctx.woodbury_basis_rank_; ++j)
            Uk_scaled.col(j) *= std::sqrt(ctx.ck[j] / ctx.sigma2_eff);
        ctx.Vi.selfadjointView<Eigen::Upper>().rankUpdate(Uk_scaled, -1.0);
    } else if (ctx.Vi_use_llt) {
        ctx.Vi.swap(ctx.Vi_L);
        gcta_blas_int blas_n_p = static_cast<gcta_blas_int>(ctx.n);
        if (gcta_dpotri(blas_n_p, ctx.Vi.data(), blas_n_p, 'U') != 0)
            LOGGER.e(0, "dpotri failed when materialising V^{-1} for P.");
        ctx.Vi_use_llt = false;
    }

    Eigen::LLT<RemlMat> llt(ctx.Xt_Vi_X_i);
    if (llt.info() == Eigen::Success) {
        RemlMat Z;
        Z.noalias() = ctx.Vi_X * llt.matrixL();
        P->swap(ctx.Vi);
        P->selfadjointView<Eigen::Upper>().rankUpdate(Z, -1.0);
        if (ctx.reml_mtd == RemlAlg::Fisher)
            P->triangularView<Eigen::Lower>() = P->transpose();
    } else {
        ctx.Vi.triangularView<Eigen::Lower>() = ctx.Vi.transpose();
        RemlMat W;
        W.noalias() = ctx.Vi_X * ctx.Xt_Vi_X_i;
        P->swap(ctx.Vi);
        P->noalias() -= W * ctx.Vi_X.transpose();
    }
    return logdet_Xt_Vi_X;
}

void calcu_tr_PA_woodbury(const RemlCtx& ctx, RemlVec& tr_PA, RemlVec* tr_PA_corrected = nullptr) {
    const int ncomp = static_cast<int>(ctx.r_indx.size());
    tr_PA.resize(ncomp);

    const double sigma2_eff = ctx.sigma2_eff;
    const double inv_s2     = 1.0 / sigma2_eff;
    const double sg2        = ctx.sg2;
    const double se2        = sigma2_eff - sg2 * ctx.lambda_tail;
    const double lambda_t   = ctx.lambda_tail;
    const int    n          = ctx.n;
    const int    k          = ctx.woodbury_basis_rank_;
    const double sum_ck     = ctx.ck.sum();
    const double tr_Vinv    = (n - sum_ck) * inv_s2;
    const double tr_Vinv_K  = (sg2 > 1e-15)
        ? (n * lambda_t * inv_s2 + (se2 / (sg2 * sigma2_eff)) * sum_ck)
        : tr_Vinv * lambda_t;

    // Spectral-subspace Gram: Vi_X^T Vi_X = Uk_Vi_X^T Uk_Vi_X + inv_s2^2 * XtX_perp (O(k * c^2)).
    const int c = ctx.X_c;
    RemlMat ViXTViX(c, c);
    ViXTViX.noalias()  = ctx.Uk_Vi_X.transpose() * ctx.Uk_Vi_X;
    ViXTViX.noalias() += (inv_s2 * inv_s2) * ctx.wb_XtX_perp;
    const double tr_corr_I = ctx.Xt_Vi_X_i.cwiseProduct(ViXTViX.transpose()).sum();

    tr_PA(ncomp - 1) = tr_Vinv - tr_corr_I;

    RemlVec delta = ctx.dk.array() - ctx.lambda_tail;
    RemlMat delta_UkViX = delta.asDiagonal() * ctx.Uk_Vi_X;
    RemlMat ViX_K_ViX = lambda_t * ViXTViX;
    ViX_K_ViX.noalias() += ctx.Uk_Vi_X.transpose() * delta_UkViX;
    const double tr_corr_K = ctx.Xt_Vi_X_i.cwiseProduct(ViX_K_ViX.transpose()).sum();

    tr_PA(0) = tr_Vinv_K - tr_corr_K;

    if (tr_PA_corrected) {
        tr_PA_corrected->resize(ncomp);
        double tr_Vinv_c   = tr_Vinv;
        double tr_Vinv_K_c = tr_Vinv_K;
        if (ctx.tail_d_var > 0.0 && n > k) {
            const double r = sg2 * inv_s2;
            tr_Vinv_c   += static_cast<double>(n - k) * r * r * ctx.tail_d_var * inv_s2;
            tr_Vinv_K_c -= static_cast<double>(n - k) * sg2 * se2 * ctx.tail_d_var
                          * (inv_s2 * inv_s2 * inv_s2);
        }
        (*tr_PA_corrected)(ncomp - 1) = tr_Vinv_c - tr_corr_I;
        (*tr_PA_corrected)(0)         = tr_Vinv_K_c - tr_corr_K;
    }
}

// Validates ctx.grm_tile_reader and derives a chunk-row count from the
// memory budget. Called once, at setup, by whichever feature needs the GRM
// stream (compute_woodbury_basis, or reml::compute() for chunked hutch++) —
// never from inside the AI-REML/EM-REML loop. feature_flag is only used to
// prefix the log/error message so the source is clear when both features
// share this budget knob.
int setup_chunked_grm_stream(const RemlCtx& ctx, const char* feature_flag) {
    if (!ctx.grm_tile_reader)
        LOGGER.e(0, std::string(feature_flag) + ": --grm-chunked-budget is set but "
                    "ctx.grm_tile_reader is empty — the GRM component wasn't actually "
                    "loaded either way.");
    // The row-chunk size and its row-band-cache reservation are solved once,
    // when grm_tile_reader was built (gcta_grm_io::make_chunked_grm_reader),
    // and cached on ctx as grm_chunk_rows_from_budget -- previously this
    // function re-derived them independently per call site (Woodbury-basis,
    // hutch++, plain exact), which is how a stale reservation assumption
    // went unnoticed here even after being fixed at the reader's own
    // construction site. A <1 value below should already have aborted the
    // run when the reader was constructed, so this is a sanity check, not
    // the primary validation.
    if (ctx.grm_chunk_rows_from_budget < 1)
        LOGGER.e(0, std::string(feature_flag) + ": --grm-chunked-budget=" +
                    std::to_string(ctx.grm_chunked_budget) + "GB produced no usable row-chunk "
                    "size (grm_chunk_rows_from_budget=" + std::to_string(ctx.grm_chunk_rows_from_budget) +
                    ") -- this should have been caught when the reader was constructed; "
                    "check the caller that built ctx.grm_tile_reader.");
    return ctx.grm_chunk_rows_from_budget;
}

// tr_PA_var receives, per component, the sampling variance of the Hutch++
// residual-term mean estimator (i.e. Var(tr_PA(ci))). It is a free byproduct
// of the existing colwise reduction below — used by ai_reml to build the
// Newton-decrement noise floor. Callers that don't need it (em_reml) can
// pass a throwaway vector.
void calcu_tr_PA_hutchpp(RemlCtx& ctx, RemlVec& tr_PA, RemlVec& tr_PA_var, int m_probes) {
    const int ncomp = static_cast<int>(ctx.r_indx.size());
    tr_PA.resize(ncomp);
    tr_PA_var.resize(ncomp);
    const int k = std::max(m_probes / 3, 3);
    const int n = ctx.n;

    const bool need_resize = (ctx.hutchpp_S.rows() != n || ctx.hutchpp_S.cols() != k);
    if (need_resize) {
        ctx.hutchpp_S.resize(n, k);
        ctx.hutchpp_G.resize(n, k);
        ctx.hutchpp_qr_scratch.resize(n, k);
        ctx.hutchpp_K.resize(n, k);
        ctx.hutchpp_Q.resize(n, k);
        ctx.hutchpp_QG.resize(n, 2 * k);
        ctx.hutchpp_MQG.resize(n, 2 * k);
        ctx.hutchpp_QtG.resize(k, k);
        ctx.hutchpp_R.resize(n, k);
        ctx.hutchpp_MR.resize(n, k);
    }
    if (need_resize || !ctx.reml_hutchpp_fixed_probes) {
        // Parallel Rademacher probe generation: seed per-thread PRNGs serially
        // from shared_rng(), then extract 32 sign bits per RNG draw in parallel.
        const int nt = std::min(omp_get_max_threads(), k);
        std::vector<uint32_t> seeds(static_cast<size_t>(nt));
        for (int t = 0; t < nt; ++t) seeds[static_cast<size_t>(t)] = static_cast<uint32_t>(gcta_eigh::shared_rng()());
        #pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num();
            std::mt19937 local_rng(seeds[static_cast<size_t>(tid)]);
            const int cols_per = (k + nt - 1) / nt;
            const int c_lo = std::min(k, tid * cols_per);
            const int c_hi = std::min(k, c_lo + cols_per);
            for (int j = c_lo; j < c_hi; ++j) {
                double* __restrict__ s_col = ctx.hutchpp_S.col(j).data();
                double* __restrict__ g_col = ctx.hutchpp_G.col(j).data();
                int r = 0;
                while (r < n) {
                    uint32_t bs = local_rng();
                    uint32_t bg = local_rng();
                    const int r_end = std::min(n, r + 32);
                    for (; r < r_end; ++r, bs >>= 1, bg >>= 1) {
                        s_col[r] = (bs & 1u) ? 1.0 : -1.0;
                        g_col[r] = (bg & 1u) ? 1.0 : -1.0;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd chol_qr_ws;
    for (int ci = 0; ci < ncomp; ci++) {
        const bool is_I = (ctx.A[ctx.r_indx[ci]].size() == 0) && !(ctx.grm_chunk_rows > 0 && ci == 0)
                          && !(ctx.Vi_use_woodbury_basis && ci == 0);

        auto applyA_into = [&](const RemlMat& Z, RemlMat& AZ) {
            if (ctx.grm_chunk_rows > 0 && ci == 0) {
                chunked_grm_matvec_into(ctx, Z, AZ);
            } else if (ctx.Vi_use_woodbury_basis && ci == 0) {
                AZ = woodbury_basis_KZ(ctx, Z);
            } else {
                symm_upper_f32_mat_into(ctx.A[ctx.r_indx[ci]], Z, AZ);
            }
        };

        // K = (P * A) * S using persistent scratch (zero per-iter heap allocations).
        if (is_I) {
            applyP_mat_into(ctx, ctx.hutchpp_S, ctx.hutchpp_K);
        } else {
            applyA_into(ctx.hutchpp_S, ctx.hutchpp_Q);
            applyP_mat_into(ctx, ctx.hutchpp_Q, ctx.hutchpp_K);
        }

        // Power iterations + orthonormalization via multi-threaded Level-3 BLAS
        // CholeskyQR2 (with automatic LAPACK dgeqrf/dorgqr fallback).
        for (int pw = 0; pw < ctx.reml_trace_power_iter; pw++) {
            const int info = gcta_cholesky_qr_thin_Q(
                static_cast<gcta_blas_int>(n), static_cast<gcta_blas_int>(k),
                ctx.hutchpp_K.data(), static_cast<gcta_blas_int>(n), &chol_qr_ws);
            if (info != 0)
                LOGGER.e(0, "calcu_tr_PA_hutchpp: orthogonalization failed (info=" + std::to_string(info) + ").");
            if (is_I) {
                applyP_mat_into(ctx, ctx.hutchpp_K, ctx.hutchpp_qr_scratch);
                ctx.hutchpp_K.swap(ctx.hutchpp_qr_scratch);
            } else {
                applyA_into(ctx.hutchpp_K, ctx.hutchpp_Q);
                applyP_mat_into(ctx, ctx.hutchpp_Q, ctx.hutchpp_K);
            }
        }
        {
            const int info = gcta_cholesky_qr_thin_Q(
                static_cast<gcta_blas_int>(n), static_cast<gcta_blas_int>(k),
                ctx.hutchpp_K.data(), static_cast<gcta_blas_int>(n), &chol_qr_ws);
            if (info != 0)
                LOGGER.e(0, "calcu_tr_PA_hutchpp: orthogonalization failed (info=" + std::to_string(info) + ").");
            ctx.hutchpp_Q.swap(ctx.hutchpp_K);
        }

        // Fused MQ/MG evaluation: applyPA_mat([Q G]) into persistent MQG without
        // transient n x 2k heap allocations.
        {
            const size_t col_bytes = static_cast<size_t>(n) * sizeof(double);
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(k) >= 32768)
            for (int j = 0; j < k; ++j) {
                std::memcpy(ctx.hutchpp_QG.col(j).data(), ctx.hutchpp_Q.col(j).data(), col_bytes);
                std::memcpy(ctx.hutchpp_QG.col(k + j).data(), ctx.hutchpp_G.col(j).data(), col_bytes);
            }
        }
        if (is_I) {
            applyP_mat_into(ctx, ctx.hutchpp_QG, ctx.hutchpp_MQG);
        } else {
            applyA_into(ctx.hutchpp_QG, ctx.hutchpp_MQG);
            if (ctx.Vi_use_llt) {
                applyP_mat_into(ctx, ctx.hutchpp_MQG, ctx.hutchpp_MQG);
            } else {
                // QG is no longer needed once A*QG is formed; reuse QG as output buffer and swap.
                applyP_mat_into(ctx, ctx.hutchpp_MQG, ctx.hutchpp_QG);
                ctx.hutchpp_MQG.swap(ctx.hutchpp_QG);
            }
        }
        const auto MQ = ctx.hutchpp_MQG.leftCols(k);
        const auto MG = ctx.hutchpp_MQG.rightCols(k);

        double t_lr = 0.0;
        #pragma omp parallel for reduction(+:t_lr) schedule(static)
        for (int j = 0; j < k; ++j) {
            t_lr += ctx.hutchpp_Q.col(j).dot(MQ.col(j));
        }

        ctx.hutchpp_QtG.noalias() = ctx.hutchpp_Q.transpose() * ctx.hutchpp_G;
        {
            const size_t col_bytes = static_cast<size_t>(n) * sizeof(double);
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(k) >= 32768)
            for (int j = 0; j < k; ++j) {
                std::memcpy(ctx.hutchpp_R.col(j).data(), ctx.hutchpp_G.col(j).data(), col_bytes);
                std::memcpy(ctx.hutchpp_MR.col(j).data(), MG.col(j).data(), col_bytes);
            }
        }
        ctx.hutchpp_R.noalias()  -= ctx.hutchpp_Q * ctx.hutchpp_QtG;
        ctx.hutchpp_MR.noalias() -= MQ * ctx.hutchpp_QtG;

        RemlVec per_col(k);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < k; ++j) {
            per_col[j] = ctx.hutchpp_R.col(j).dot(ctx.hutchpp_MR.col(j));
        }
        const double mean_r = per_col.mean();
        tr_PA(ci) = t_lr + mean_r;
        tr_PA_var(ci) = !ctx.reml_hutchpp_fixed_probes
            ? (per_col.array() - mean_r).square().sum() / (static_cast<double>(k) * (k - 1))
            : 0.0;
    }
}

void calcu_tr_PA(const RemlCtx& ctx, const RemlMat& P, RemlVec& tr_PA) {
    const int m = static_cast<int>(ctx.r_indx.size());
    tr_PA.resize(m);
    for (int i = 0; i < m; i++) {
        if (ctx.grm_chunk_rows > 0 && i == 0) {
            const int eff_block = std::min(ctx.n, std::max(1, ctx.grm_chunk_rows));
            double s = 0.0;
            if (ctx.grm_tile_reader_f32 && ctx.grm_tile_reader_f32.has_transposed()) {
                const std::vector<int> bands = ctx.grm_tile_reader_f32.compute_row_bands(ctx.n, eff_block);
                const int nb = static_cast<int>(bands.size()) - 1;
                auto sweep = ctx.grm_tile_reader_f32.begin_sweep(bands);
                for (int b = 0; b < nb; ++b) {
                    if (sweep) sweep->acquire(b);
                    const int rs = bands[b], re = bands[b + 1];
                    const int rows_i = re - rs;
                    for (int cs = 0; cs < rs; cs += eff_block) {
                        const int ce = std::min(cs + eff_block, rs);
                        const int cols_j = ce - cs;
                        const auto tile_T = ctx.grm_tile_reader_f32.read_T(rs, re, cs, ce);
                        double tile_s = 0.0;
                        #pragma omp parallel for reduction(+:tile_s) schedule(static)
                        for (int col = 0; col < rows_i; ++col) {
                            tile_s += P.col(rs + col).segment(cs, cols_j)
                                          .dot(tile_T.col(col).cast<double>());
                        }
                        s += 2.0 * tile_s;
                    }
                    {
                        const auto tile_T = ctx.grm_tile_reader_f32.read_T(rs, re, rs, re);
                        double tile_s = 0.0;
                        #pragma omp parallel for reduction(+:tile_s) schedule(guided)
                        for (int col = 0; col < rows_i; ++col) {
                            tile_s += P(rs + col, rs + col) * static_cast<double>(tile_T(col, col));
                            if (col > 0) {
                                tile_s += 2.0 * P.col(rs + col).segment(rs, col)
                                                    .dot(tile_T.col(col).head(col).cast<double>());
                            }
                        }
                        s += tile_s;
                    }
                    if (sweep) sweep->release(b);
                }
            } else {
                const std::vector<int> bands = ctx.grm_tile_reader.compute_row_bands(ctx.n, eff_block);
                const int nb = static_cast<int>(bands.size()) - 1;
                auto sweep = ctx.grm_tile_reader.begin_sweep(bands);
                const bool use_T = ctx.grm_tile_reader.has_transposed();
                for (int b = 0; b < nb; ++b) {
                    if (sweep) sweep->acquire(b);
                    const int rs = bands[b], re = bands[b + 1];
                    const int rows_i = re - rs;
                    for (int cs = 0; cs < rs; cs += eff_block) {
                        const int ce = std::min(cs + eff_block, rs);
                        const int cols_j = ce - cs;
                        double tile_s = 0.0;
                        if (use_T) {
                            const auto tile_T = ctx.grm_tile_reader.read_T(rs, re, cs, ce);
                            #pragma omp parallel for reduction(+:tile_s) schedule(static)
                            for (int col = 0; col < rows_i; ++col) {
                                tile_s += P.col(rs + col).segment(cs, cols_j).dot(tile_T.col(col));
                            }
                        } else {
                            const auto tile = ctx.grm_tile_reader(rs, re, cs, ce);
                            #pragma omp parallel for reduction(+:tile_s) schedule(static)
                            for (int col = 0; col < rows_i; ++col) {
                                tile_s += P.col(rs + col).segment(cs, cols_j).dot(tile.row(col).transpose());
                            }
                        }
                        s += 2.0 * tile_s;
                    }
                    {
                        double tile_s = 0.0;
                        if (use_T) {
                            const auto tile_T = ctx.grm_tile_reader.read_T(rs, re, rs, re);
                            #pragma omp parallel for reduction(+:tile_s) schedule(guided)
                            for (int col = 0; col < rows_i; ++col) {
                                tile_s += P(rs + col, rs + col) * tile_T(col, col);
                                if (col > 0)
                                    tile_s += 2.0 * P.col(rs + col).segment(rs, col)
                                                        .dot(tile_T.col(col).head(col));
                            }
                        } else {
                            const auto tile = ctx.grm_tile_reader(rs, re, rs, re);
                            #pragma omp parallel for reduction(+:tile_s) schedule(guided)
                            for (int col = 0; col < rows_i; ++col) {
                                tile_s += P(rs + col, rs + col) * tile(col, col);
                                if (col > 0)
                                    tile_s += 2.0 * P.col(rs + col).segment(rs, col)
                                                        .dot(tile.row(col).head(col).transpose());
                            }
                        }
                        s += tile_s;
                    }
                    if (sweep) sweep->release(b);
                }
            }
            tr_PA(i) = s;
        } else if (ctx.A[ctx.r_indx[i]].size() == 0) {
            tr_PA(i) = P.diagonal().sum();
        } else {
            const auto& Ai = ctx.A[ctx.r_indx[i]];
            double s = 0.0;
            #pragma omp parallel for reduction(+:s) schedule(guided)
            for (int col = 0; col < ctx.n; col++) {
                s += P(col, col) * static_cast<double>(Ai(col, col));
                if (col > 0)
                    s += 2.0 * P.col(col).head(col).dot(Ai.col(col).head(col).cast<double>());
            }
            tr_PA(i) = s;
        }
    }
}

void calcu_Hi(RemlCtx& ctx, RemlMat& P, RemlMat& Hi) {
    // Cache-blocked 64x64 parallel upper->lower mirror of P.
    const int n = ctx.n;
    {
        constexpr int kMir = 64;
        double* p_data = P.data();
        #pragma omp parallel for schedule(guided) if (n >= 256)
        for (int c0 = 0; c0 < n; c0 += kMir) {
            const int c1 = std::min(n, c0 + kMir);
            for (int r0 = c0; r0 < n; r0 += kMir) {
                const int r1 = std::min(n, r0 + kMir);
                for (int c = c0; c < c1; ++c) {
                    const int r_lo = std::max(r0, c + 1);
                    double* dst_col = p_data + static_cast<size_t>(c) * static_cast<size_t>(n);
                    for (int r = r_lo; r < r1; ++r) {
                        dst_col[r] = p_data[c + static_cast<size_t>(r) * static_cast<size_t>(n)];
                    }
                }
            }
        }
    }
    const int m = static_cast<int>(ctx.r_indx.size());

    std::vector<RemlMat> PA(m);
    for (int i = 0; i < m; i++) {
        if (ctx.Vi_use_woodbury_basis && i == 0) {
            const RemlMat Uk_d = ctx.Uk.cast<double>();
            RemlMat PUk(n, ctx.woodbury_basis_rank_);
            PUk.noalias() = P * Uk_d;
            RemlVec delta_v = ctx.dk.array() - ctx.lambda_tail;
            PA[i] = ctx.lambda_tail * P;
            PA[i].noalias() += PUk * delta_v.asDiagonal() * Uk_d.transpose();
        } else if (ctx.grm_chunk_rows > 0 && i == 0) {
            chunked_grm_matvec_into(ctx, P, PA[i]);
        } else if (ctx.A[ctx.r_indx[i]].size() == 0) {
            PA[i].resize(0, 0);
        } else {
            PA[i] = symm_upper_f32_mat(ctx.A[ctx.r_indx[i]], P);
        }
    }

    for (int i = 0; i < m; i++) {
        const bool i_id = (PA[i].size() == 0);
        for (int j = 0; j <= i; j++) {
            const bool j_id = (PA[j].size() == 0);
            double val = 0.0;
            if (i_id && j_id) {
                #pragma omp parallel for reduction(+:val) schedule(static) if (n >= 256)
                for (int c = 0; c < n; ++c) val += P.col(c).squaredNorm();
            } else if (i_id) {
                #pragma omp parallel for reduction(+:val) schedule(static) if (n >= 256)
                for (int c = 0; c < n; ++c) val += PA[j].col(c).dot(P.col(c));
            } else if (j_id) {
                #pragma omp parallel for reduction(+:val) schedule(static) if (n >= 256)
                for (int c = 0; c < n; ++c) val += PA[i].col(c).dot(P.col(c));
            } else {
                // tr((P A_i)(P A_j)) = sum_{r,c} PA_i(r,c) * PA_j(c,r)
                #pragma omp parallel for reduction(+:val) schedule(static) if (n >= 256)
                for (int c = 0; c < n; ++c) val += PA[i].col(c).dot(PA[j].row(c).transpose());
            }
            Hi(i, j) = Hi(j, i) = val;
        }
    }

    if (!inverse_H(ctx, Hi)) {
        if (ctx.reml_force_converge) {
            LOGGER.w(0, "the information matrix is not invertible.");
            ctx.reml_AI_not_invertible = true;
        } else {
            LOGGER.e(0, "the information matrix is not invertible.");
        }
    }
}

// Matrix-free inverse Average Information matrix Hi = (0.5 * APy^T P APy)^{-1}
// for Woodbury (O(k * c^2) in spectral subspace) and Hutch++ (O(N) via applyP_mat)
// modes, replacing the O(N^2) P_tmp materialisation on EM-REML / --reml-force-converge
// / non-PD rollback paths (prevents 2 TB dense N x N OOM at N=500k).
// Note: AI already includes the 0.5 factor (E[AI] = 0.5 * Fisher), so AI^{-1}
// already matches the SE scaling of 2 * Fisher^{-1} with NO extra factor of 2.
void compute_ai_Hi_matrix_free(RemlCtx& ctx, const RemlVec& Py, RemlMat& Hi) {
    const int m = static_cast<int>(ctx.r_indx.size());
    Hi.resize(m, m);

    if (ctx.Vi_use_woodbury_basis) {
        ensure_woodbury_subspace_cache(ctx);
        const WoodburySubspaceVec Py_sub = wb_apply_P(ctx, wb_y_vec(ctx));
        const WoodburySubspaceVec APy_0  = wb_apply_K(ctx, Py_sub);
        const WoodburySubspaceVec& APy_1 = Py_sub;
        const WoodburySubspaceVec PAPy_0 = wb_apply_P(ctx, APy_0);
        const WoodburySubspaceVec PAPy_1 = wb_apply_P(ctx, APy_1);
        Hi(0, 0) = 0.5 * wb_inner(ctx, APy_0, PAPy_0);
        Hi(0, 1) = Hi(1, 0) = 0.5 * wb_inner(ctx, APy_0, PAPy_1);
        Hi(1, 1) = 0.5 * wb_inner(ctx, APy_1, PAPy_1);
    } else {
        RemlMat APy(ctx.n, m);
        for (int i = 0; i < m; ++i) {
            if (ctx.grm_chunk_rows > 0 && i == 0)
                APy.col(i) = chunked_grm_matvec_vec(ctx, Py);
            else if (ctx.A[ctx.r_indx[i]].size() == 0)
                APy.col(i) = Py;
            else
                APy.col(i) = symm_upper_f32_vec(ctx.A[ctx.r_indx[i]], Py);
        }
        const RemlMat PAPy = applyP_mat(ctx, APy);
        Hi.noalias() = 0.5 * (APy.transpose() * PAPy);
    }

    if (!inverse_H(ctx, Hi)) {
        if (ctx.reml_force_converge) {
            LOGGER.w(0, "the information matrix is not invertible.");
            ctx.reml_AI_not_invertible = true;
        } else {
            LOGGER.e(0, "the information matrix is not invertible.");
        }
    }
}

// Fisher-scoring REML update (legacy reml_equation equivalent).
// Requires explicit P materialisation; reml_iteration enforces that for mtd==1.
void reml_equation(RemlCtx& ctx, RemlMat& P, RemlMat& Hi, RemlVec& Py, RemlVec& varcmp, double& yPy_out) {
    calcu_Hi(ctx, P, Hi);
    if (ctx.reml_AI_not_invertible) {
        yPy_out = 0.0;
        return;
    }

    Py.noalias() = P.selfadjointView<Eigen::Lower>() * ctx.y;
    yPy_out = ctx.y.dot(Py);

    const int m = static_cast<int>(ctx.r_indx.size());
    RemlVec R(m);
    if (ctx.reml_tmp_n.size() != ctx.n) ctx.reml_tmp_n.resize(ctx.n);
    for (int i = 0; i < m; i++) {
        if (ctx.A[ctx.r_indx[i]].size() == 0) {
            R(i) = Py.squaredNorm();
        } else {
            ctx.reml_tmp_n = symm_upper_f32_vec(ctx.A[ctx.r_indx[i]], Py);
            R(i) = Py.dot(ctx.reml_tmp_n);
        }
    }

    varcmp = Hi * R;
    Hi = 2 * Hi;  // keep same SE convention as legacy path
}

double newton_decrement_null_quantile(const RemlMat& Hi, const RemlVec& var_U, double alpha) {
    const RemlVec sqrt_var = var_U.cwiseMax(0.0).cwiseSqrt();
    if (sqrt_var.squaredNorm() < 1e-300) return 0.0;
    RemlMat B = sqrt_var.asDiagonal() * Hi * sqrt_var.asDiagonal();
    B = 0.5 * (B + B.transpose());
    const double trB  = B.trace();
    const double trB2 = B.squaredNorm();
    if (trB2 < 1e-300) return 0.0;
    const double g = trB2 / trB;
    const double h = trB * trB / trB2;
    boost::math::chi_squared_distribution<double> chi2(h);
    return g * boost::math::quantile(chi2, 1.0 - alpha);
}

void ai_reml(RemlCtx& ctx, RemlMat& P, RemlMat& Hi, RemlVec& Py,
             RemlVec& prev_varcmp, RemlVec& varcmp, double dlogL,
             double& lambda_sq, RemlVec& var_U,
             bool last_step_trustworthy, double& step_scale_out,
             double& yPy_out) {
    const bool use_approx            = ctx.reml_trace_hutchpp;
    const bool woodbury_basis_active = ctx.Vi_use_woodbury_basis;
    const int m = static_cast<int>(ctx.r_indx.size());

    RemlVec R(m);
    if (woodbury_basis_active) {
        // Spectral-subspace AI-REML: O(k * c^2), zero Uk scans!
        const WoodburySubspaceVec y_sub  = wb_y_vec(ctx);
        const WoodburySubspaceVec Py_sub = wb_apply_P(ctx, y_sub);
        yPy_out = wb_inner(ctx, y_sub, Py_sub);

        const WoodburySubspaceVec APy_0  = wb_apply_K(ctx, Py_sub);
        const WoodburySubspaceVec& APy_1 = Py_sub;
        R(0) = wb_inner(ctx, APy_0, Py_sub);
        R(1) = wb_inner(ctx, APy_1, Py_sub);

        const WoodburySubspaceVec PAPy_0 = wb_apply_P(ctx, APy_0);
        const WoodburySubspaceVec PAPy_1 = wb_apply_P(ctx, APy_1);
        Hi.resize(m, m);
        Hi(0, 0) = 0.5 * wb_inner(ctx, APy_0, PAPy_0);
        Hi(0, 1) = Hi(1, 0) = 0.5 * wb_inner(ctx, APy_0, PAPy_1);
        Hi(1, 1) = 0.5 * wb_inner(ctx, APy_1, PAPy_1);
    } else {
        if (use_approx)
            Py = applyP_vec(ctx, ctx.y);
        else
            Py.noalias() = P.selfadjointView<Eigen::Upper>() * ctx.y;
        yPy_out = ctx.y.dot(Py);

        RemlMat APy(ctx.n, m);
        for (int i = 0; i < m; i++) {
            if (ctx.grm_chunk_rows > 0 && i == 0)
                APy.col(i) = chunked_grm_matvec_vec(ctx, Py);
            else if (ctx.A[ctx.r_indx[i]].size() == 0)
                APy.col(i) = Py;
            else
                APy.col(i) = symm_upper_f32_vec(ctx.A[ctx.r_indx[i]], Py);
        }

        if (use_approx) {
            R.noalias() = APy.transpose() * Py;
            const RemlMat PAPy = applyP_mat(ctx, APy);
            Hi.noalias() = APy.transpose() * PAPy;
        } else {
            R.noalias() = APy.transpose() * Py;
            RemlMat PAPy(ctx.n, m);
            PAPy.noalias() = P.selfadjointView<Eigen::Upper>() * APy;
            Hi.noalias() = APy.transpose() * PAPy;
        }
        Hi = 0.5 * Hi;
    }

    RemlVec tr_PA;
    RemlVec tr_PA_var = RemlVec::Zero(m);
    if (woodbury_basis_active) {
        calcu_tr_PA_woodbury(ctx, tr_PA);
    } else if (use_approx) {
        const int eff_nprobes = (std::fabs(dlogL) < 1.0)
            ? ctx.reml_trace_hutchpp_nprobes
            : std::max(ctx.reml_trace_hutchpp_nprobes / 3, 30);
        calcu_tr_PA_hutchpp(ctx, tr_PA, tr_PA_var, eff_nprobes);
    } else {
        calcu_tr_PA(ctx, P, tr_PA);
    }
    R = -0.5 * (tr_PA - R);   // R is now the score vector U

    if (!inverse_H(ctx, Hi)) {
        if (ctx.reml_force_converge) {
            LOGGER.w(0, "the information matrix is not invertible.");
            ctx.reml_AI_not_invertible = true;
            return;
        } else {
            LOGGER.e(0, "the information matrix is not invertible.");
        }
    }
    RemlVec delta = Hi * R;
    lambda_sq = R.dot(delta);

    double fraction_to_boundary_cap = 1.0;
    {
        constexpr double tau = 0.995;
        for (int i = 0; i < delta.size(); ++i) {
            if (delta[i] < 0.0 && prev_varcmp[i] > 0.0) {
                const double boundary_scale = -tau * prev_varcmp[i] / delta[i];
                fraction_to_boundary_cap = std::min(fraction_to_boundary_cap, boundary_scale);
            }
        }
    }

    double step_scale = 1.0;
    if (dlogL > 1.0) step_scale = 0.316;
    if (ctx.reml_mtd == RemlAlg::AI && ctx.reml_ai_robust) {
        constexpr double target_decrement = 0.5;
        step_scale = last_step_trustworthy
            ? 1.0
            : std::min(1.0, std::sqrt(target_decrement / std::max(lambda_sq, 1e-12)));
    }
    step_scale = std::min(step_scale, fraction_to_boundary_cap);
    step_scale_out = step_scale;

    var_U = 0.25 * tr_PA_var;
    varcmp = prev_varcmp + step_scale * delta;
    if (ctx.reml_print_trajectory)
        LOGGER << "REML iteration: lambda_sq = " << std::to_string(lambda_sq) << ", step scale = " << std::to_string(step_scale) << ", delta = " << delta.transpose() << std::endl;
}

// tr_PA-only-corrected post-hoc step in spectral subspace (O(k * c), zero Uk scans).
RemlVec compute_woodbury_posthoc_delta_simple(RemlCtx& ctx, const RemlMat& Hi) {
    const int m = static_cast<int>(ctx.r_indx.size());

    RemlVec tr_PA(m), tr_PA_corrected(m);
    calcu_tr_PA_woodbury(ctx, tr_PA, &tr_PA_corrected);

    const WoodburySubspaceVec Py_sub = wb_apply_P(ctx, wb_y_vec(ctx));
    const WoodburySubspaceVec APy_0  = wb_apply_K(ctx, Py_sub);
    RemlVec R_flat(m);
    R_flat(0) = wb_inner(ctx, APy_0, Py_sub);
    R_flat(1) = wb_inner(ctx, Py_sub, Py_sub);

    const RemlVec U_partial = -0.5 * (tr_PA_corrected - R_flat);
    return Hi * U_partial;
}

void em_reml(RemlCtx& ctx, RemlMat& P, RemlVec& Py,
             RemlVec& prev_varcmp, RemlVec& varcmp, double dlogL,
             double& yPy_out) {
    const bool woodbury_basis_active = ctx.Vi_use_woodbury_basis;
    const bool use_approx            = ctx.reml_trace_hutchpp;
    const int m = static_cast<int>(ctx.r_indx.size());

    RemlVec tr_PA;
    RemlVec R(m);
    if (woodbury_basis_active) {
        calcu_tr_PA_woodbury(ctx, tr_PA);
        const WoodburySubspaceVec y_sub  = wb_y_vec(ctx);
        const WoodburySubspaceVec Py_sub = wb_apply_P(ctx, y_sub);
        yPy_out = wb_inner(ctx, y_sub, Py_sub);
        const WoodburySubspaceVec APy_0  = wb_apply_K(ctx, Py_sub);
        R(0) = wb_inner(ctx, APy_0, Py_sub);
        R(1) = wb_inner(ctx, Py_sub, Py_sub);
    } else {
        if (use_approx) {
            const int eff_nprobes = (std::fabs(dlogL) < 1.0)
                ? ctx.reml_trace_hutchpp_nprobes
                : std::max(ctx.reml_trace_hutchpp_nprobes / 3, 30);
            RemlVec tr_PA_var_unused;
            calcu_tr_PA_hutchpp(ctx, tr_PA, tr_PA_var_unused, eff_nprobes);
            Py = applyP_vec(ctx, ctx.y);
        } else {
            calcu_tr_PA(ctx, P, tr_PA);
            Py.noalias() = P.selfadjointView<Eigen::Upper>() * ctx.y;
        }
        yPy_out = ctx.y.dot(Py);

        if (ctx.reml_tmp_n.size() != ctx.n) ctx.reml_tmp_n.resize(ctx.n);
        for (int i = 0; i < m; i++) {
            if (ctx.grm_chunk_rows > 0 && i == 0) {
                ctx.reml_tmp_n = chunked_grm_matvec_vec(ctx, Py);
                R(i) = Py.dot(ctx.reml_tmp_n);
            } else if (ctx.A[ctx.r_indx[i]].size() == 0) {
                R(i) = Py.squaredNorm();
            } else {
                ctx.reml_tmp_n = symm_upper_f32_vec(ctx.A[ctx.r_indx[i]], Py);
                R(i) = Py.dot(ctx.reml_tmp_n);
            }
        }
    }

    for (int i = 0; i < m; i++) {
        varcmp(i) = prev_varcmp(i) - prev_varcmp(i) * prev_varcmp(i) * (tr_PA(i) - R(i)) / ctx.n;
    }
    if (ctx.reml_print_trajectory)
        LOGGER << "REML iteration: step scale = 1.000000, delta = " << (varcmp - prev_varcmp).transpose() << std::endl;
}

double reml_iteration(RemlCtx& ctx,
                      RemlMat& Vi_X_out, RemlMat& Xt_Vi_X_i_out, RemlMat& Hi,
                      RemlVec& Py, RemlVec& varcmp,
                      bool prior_var_flag, bool no_constrain) {
    const int m = static_cast<int>(ctx.r_indx.size());
    int constrain_num = 0, iter = 0;
    const RemlAlg reml_mtd_tmp = ctx.reml_mtd;
    double logdet = 0.0, logdet_Xt_Vi_X = 0.0;
    double prev_lgL = -1e20, lgL = -1e20, dlogL = 1000.0;
    double lambda_sq = 1e300;
    RemlVec var_U;
    RemlVec prev_prev_varcmp(varcmp), prev_varcmp(varcmp), varcomp_init(varcmp);

    constexpr double rho_threshold = 0.75;
    bool last_step_trustworthy = true;
    double step_scale_taken = 1.0;
    bool have_predicted = false;
    double predicted_dlogL = 0.0;

    const double ai_robust_alpha_per_iter =
        ctx.reml_ai_robust_risk / std::max(1, ctx.reml_max_iter);

    if (ctx.reml_trace_hutchpp && !ctx.Vi_use_woodbury_basis) {
        LOGGER << "Using Hutch++ stochastic trace estimator with "
               << ctx.reml_trace_hutchpp_nprobes << " " << (ctx.reml_hutchpp_fixed_probes ? "fixed" : "fresh") << " probes." << std::endl;
    }
    if (ctx.reml_mtd == RemlAlg::AI && ctx.reml_ai_robust) {
        LOGGER << "Using Newton-decrement convergence criterion (--reml-ai-robust, "
               << "logL tol=" << ctx.reml_ai_robust_tol
               << ", run-level false-stop risk=" << ctx.reml_ai_robust_risk
               << " => alpha/iter=" << ai_robust_alpha_per_iter << ")." << std::endl;
    }

    bool converged_flag = false;
    int small_delta_streak = 0;
    const int M = 1;
    double best_lgL = 0;
    RemlVec best_varcmp;
    for (iter = 0; iter < ctx.reml_max_iter; iter++) {
        if (iter == 0) {
            prev_varcmp = varcomp_init;
            if (ctx.reml_fixed_var) {
                LOGGER << "Variance components fixed at: ";
            } else {
                LOGGER << "Variance components initialised at: ";
            }
            LOGGER << varcmp.transpose() << std::endl;

            if (!prior_var_flag) {
                LOGGER << "Round 0 iteration using ";
                if (ctx.reml_no_HE_start || !ctx.he_warm_start_applied) {
                    ctx.reml_mtd = RemlAlg::EM;
                    LOGGER << "EM-REML ..." << std::endl;
                } else {
                    LOGGER << "AI-REML ..." << std::endl;
                }
            }
        }
        if (iter == 1) {
            ctx.reml_mtd = reml_mtd_tmp;
            LOGGER << "Running " << reml_alg_name(ctx.reml_mtd) << " algorithm ...\nIter.\tlogL\t";
            for (int i = 0; i < m; i++) LOGGER << ctx.var_name[ctx.r_indx[i]] << "\t";
            LOGGER << std::endl;
        }

        const bool skip_P_this_iter = (ctx.reml_trace_hutchpp || ctx.Vi_use_woodbury_basis) && ctx.reml_mtd != RemlAlg::Fisher;
        if (!calcu_Vi(ctx, prev_varcmp, logdet, iter, skip_P_this_iter)) {
            LOGGER << "Warning: V matrix is not positive-definite." << std::endl;
            varcmp = prev_prev_varcmp;
            if (skip_P_this_iter) {
                if (!calcu_Vi(ctx, varcmp, logdet, iter, true))
                    LOGGER.e(0, "V matrix is not positive-definite.");
                calcu_P_impl(ctx, nullptr);
                if (!ctx.Vi_use_woodbury_basis) Py = applyP_vec(ctx, ctx.y);
                compute_ai_Hi_matrix_free(ctx, Py, Hi);
            } else {
                if (!calcu_Vi(ctx, varcmp, logdet, iter, false))
                    LOGGER.e(0, "V matrix is not positive-definite.");
                RemlMat P_tmp(ctx.n, ctx.n);
                calcu_P_impl(ctx, &P_tmp);
                calcu_Hi(ctx, P_tmp, Hi);
                Hi = 2 * Hi;
            }
            break;
        }

        double logdet_Xt_Vi_X2;
        if (skip_P_this_iter) {
            logdet_Xt_Vi_X2 = calcu_P_impl(ctx, nullptr);
            ctx.P.resize(0, 0);
        } else {
            logdet_Xt_Vi_X2 = calcu_P_impl(ctx, &ctx.P);
            ctx.Vi.resize(0, 0);
        }
        logdet_Xt_Vi_X = logdet_Xt_Vi_X2;

        double yPy = 0.0;
        if (ctx.reml_mtd == RemlAlg::AI)
            ai_reml(ctx, ctx.P, Hi, Py, prev_varcmp, varcmp, dlogL, lambda_sq, var_U,
                    last_step_trustworthy, step_scale_taken, yPy);
        else if (ctx.reml_mtd == RemlAlg::Fisher)
            reml_equation(ctx, ctx.P, Hi, Py, varcmp, yPy);
        else if (ctx.reml_mtd == RemlAlg::EM)
            em_reml(ctx, ctx.P, Py, prev_varcmp, varcmp, dlogL, yPy);

        lgL = -0.5 * (logdet_Xt_Vi_X + logdet + yPy);

        if (ctx.reml_force_converge && ctx.reml_AI_not_invertible) break;

        if (!no_constrain) constrain_num = constrain_varcmp(ctx, varcmp);

        if (iter > 0) {
            LOGGER << iter << "\t" << std::fixed << LOGGER.setprecision(4) << lgL << "\t";
            for (int i = 0; i < m; i++) LOGGER << LOGGER.setprecision(5) << varcmp[i] << "\t";
            if (constrain_num > 0) LOGGER << "(" << constrain_num << " component(s) constrained)" << std::endl;
            else LOGGER << std::endl;
        } else {
            if (!prior_var_flag) LOGGER << "Updated prior values: " << varcmp.transpose() << std::endl;
            LOGGER << "logL: " << std::fixed << LOGGER.setprecision(4) << lgL << std::endl;
        }

        if (ctx.reml_fixed_var) { varcmp = prev_varcmp; break; }

        if (constrain_num * 2 > m) {
            if (ctx.reml_allow_constrain_run) {
                LOGGER.w(0, "more than half of the variance components are constrained.");
            } else {
                LOGGER.e(0, "analysis stopped because more than half of the variance components are constrained.");
            }
        }

        if ((ctx.reml_force_converge || ctx.reml_no_converge) && prev_lgL > lgL) {
            varcmp = prev_varcmp;
            if (skip_P_this_iter) {
                // Matrix-free 2x2 AI matrix inverse: avoids 2 TB dense N x N P_tmp OOM.
                compute_ai_Hi_matrix_free(ctx, Py, Hi);
            } else {
                calcu_Hi(ctx, ctx.P, Hi);
                Hi = 2 * Hi;
            }
            break;
        }

        dlogL = lgL - prev_lgL;

        const bool ai_robust_active_step = (ctx.reml_mtd == RemlAlg::AI && ctx.reml_ai_robust);
        if (ai_robust_active_step && iter > 0 && have_predicted && predicted_dlogL > 1e-12) {
            const double rho = dlogL / predicted_dlogL;
            last_step_trustworthy = (rho > rho_threshold);
            if (ctx.reml_print_trajectory) {
                LOGGER << "  [ai-robust: predicted dlogL = " << std::fixed << LOGGER.setprecision(4)
                    << predicted_dlogL << ", actual dlogL = " << dlogL
                    << ", rho = " << LOGGER.setprecision(3) << rho
                    << " (" << (last_step_trustworthy ? "trustworthy" : "not trustworthy") << ")]"
                    << std::endl;
            }
        }
        if (ai_robust_active_step) {
            predicted_dlogL = lambda_sq * step_scale_taken * (1.0 - 0.5 * step_scale_taken);
            have_predicted = true;
        }

        bool like_small;
        const bool ai_robust_active = (ctx.reml_mtd == RemlAlg::AI && ctx.reml_ai_robust);
        if (ai_robust_active) {
            const double lambda_sq_floor = 2.0 * ctx.reml_ai_robust_tol;
            const double null_q = newton_decrement_null_quantile(Hi, var_U, ai_robust_alpha_per_iter);
            const double thresh = std::max(lambda_sq_floor, null_q);
            like_small = (lambda_sq <= thresh);
        } else {
            const double lgL_scale = std::max(1.0, std::fabs(lgL));
            const double dlogL_rel = std::fabs(dlogL) / lgL_scale;
            like_small = (std::fabs(dlogL) < 1e-4) || (dlogL_rel < 1e-6);
        }

        if (lgL > best_lgL) {
            best_lgL = lgL;
            best_varcmp = varcmp;
        }

        if (ai_robust_active) {
            converged_flag = like_small;
        } else {
            small_delta_streak = like_small ? small_delta_streak + 1 : 0;
            converged_flag = (small_delta_streak >= M);
        }

        if (converged_flag) {
            if (ctx.Vi_use_woodbury_basis && ctx.reml_mtd == RemlAlg::AI && ctx.woodbury_basis_posthoc_correction) {
                const RemlVec delta_simple = compute_woodbury_posthoc_delta_simple(ctx, Hi);
                LOGGER << "REML: Woodbury post-hoc tail correction: delta = " << delta_simple.transpose() << std::endl;
            }
            if (ctx.reml_mtd == RemlAlg::EM) {
                if (skip_P_this_iter) {
                    // Matrix-free 2x2 AI matrix inverse: avoids 2 TB dense N x N P_tmp OOM.
                    compute_ai_Hi_matrix_free(ctx, Py, Hi);
                } else {
                    calcu_Hi(ctx, ctx.P, Hi);
                    Hi = 2 * Hi;
                }
            }
            break;
        }

        ctx.Vi.swap(ctx.P);
        prev_prev_varcmp = prev_varcmp;
        prev_varcmp = varcmp;
        prev_lgL = lgL;
    }

    if (ctx.reml_fixed_var)
        LOGGER.w(0, "model evaluated at fixed variance components; log-likelihood may not be maximised.");
    else {
        if (converged_flag) LOGGER << "Log-likelihood ratio converged." << std::endl;
        else if (ctx.reml_force_converge || ctx.reml_no_converge)
            LOGGER.w(0, "Log-likelihood not converged. Results are not reliable.");
        else if (iter == ctx.reml_max_iter) {
            std::ostringstream errmsg;
            errmsg << "Log-likelihood not converged (stop after " << ctx.reml_max_iter << " iterations). "
                   << "Use --reml-maxit for more iterations.";
            if (ctx.reml_max_iter > 1) LOGGER.e(0, errmsg.str());
        }
    }
    ctx.reml_iterations = std::max(0, iter + 1);
    ctx.P.resize(0, 0);

    if (!ctx.Vi_use_woodbury_basis) {
        Vi_X_out = ctx.Vi_X;
    }
    Xt_Vi_X_i_out = ctx.Xt_Vi_X_i;

    return lgL;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API implementations
// ─────────────────────────────────────────────────────────────────────────────

namespace reml {

namespace {

static int woodbury_rank_cap(const RemlCtx& ctx, int k_budget_ceiling) {
    const int n = ctx.n;
    const int user_cap = (ctx.woodbury_basis_k_max > 0) ? ctx.woodbury_basis_k_max : n - 1;
    return std::min({n - 1, k_budget_ceiling, user_cap});
}

static int woodbury_initial_k_svd(const RemlCtx& ctx, WoodburyBasisSpec::Kind mode, int k_budget_ceiling) {
    const int n = ctx.n;
    switch (mode) {
        case WoodburyBasisSpec::Kind::MP:
        case WoodburyBasisSpec::Kind::EIG:
        case WoodburyBasisSpec::Kind::VAR:
            // Locked expansion (expand_symmetric_eigh) reuses every converged column
            // across rounds with zero wasted work, so starting EIG/VAR at k_init
            // avoids jumping straight to n/20 (k_max=25k at N>=500k) on low-rank GRMs.
            return std::min({n - 1, k_budget_ceiling, ctx.woodbury_basis_k_init});
        case WoodburyBasisSpec::Kind::Fixed:
        default:
            return ctx.woodbury_basis ? ctx.woodbury_basis->fixed_rank : 0;
    }
}

struct RankEvalResult {
    bool satisfied = false;
    int k_target   = 0;
    int k_extra    = 0;    // MP: k_signal, count of eigenvalues above lambda+
    int run_len    = 0;    // MP: consecutive sub-threshold eigenvalues confirmed so far this round
    double metric  = 0.0;  // The actual tested convergence quantity for this mode:
                            //   EIG: captured mass (absolute, i.e. rho * trace_K_full)
                            //   VAR: relative Frobenius tail error (last value computed)
                            //   MP:  unused — MP's test is the discrete edge-band state (k_extra/run_len), not a scalar
};

static RankEvalResult evaluate_rank_criterion(
    WoodburyBasisSpec::Kind mode,
    const Eigen::VectorXd& eval_full,
    int k_svd,
    double lambda_plus,
    double target_mass,
    double trace_K_full,
    double trace_K2,
    const RemlCtx& ctx)
{
    RankEvalResult res;
    const int n = ctx.n;

    switch (mode) {
        case WoodburyBasisSpec::Kind::MP: {
            int k_signal = 0;
            while (k_signal < static_cast<int>(eval_full.size()) && eval_full[k_signal] > lambda_plus)
                ++k_signal;
            const double confirm_thresh = (1.0 - ctx.woodbury_basis_edge_margin) * lambda_plus;
            int run_len = 0, run_start = -1;
            for (int i = k_signal; i < static_cast<int>(eval_full.size()); ++i) {
                if (eval_full[i] < confirm_thresh) {
                    if (run_len == 0) run_start = i;
                    if (++run_len >= ctx.woodbury_basis_edge_confirm) break;
                } else {
                    run_len = 0;
                    run_start = -1;
                }
            }
            res.satisfied = (run_len >= ctx.woodbury_basis_edge_confirm);
            res.k_target  = res.satisfied ? run_start : k_svd;
            res.k_extra   = k_signal;
            res.run_len   = run_len;
            break;
        }
        case WoodburyBasisSpec::Kind::EIG: {
            double cumulative = 0.0;
            bool crossed_target = false;
            res.k_target = k_svd;
            for (int i = 0; i < static_cast<int>(eval_full.size()); ++i) {
                cumulative += eval_full[i];
                if (cumulative >= target_mass) {
                    res.k_target = i + 1;
                    crossed_target = true;
                    break;
                }
            }
            res.satisfied = crossed_target
                         && (res.k_target + ctx.woodbury_basis_EIG_k_buffer <= k_svd);
            // cumulative is the mass captured up to the crossing point when satisfied,
            // or the full eval_full sum (mass at the current k_svd) otherwise — exactly
            // the "how close are we" quantity the per-round log needs.
            res.metric = cumulative;
            break;
        }
        case WoodburyBasisSpec::Kind::VAR: {
            bool crossed_target = false;
            res.k_target = k_svd;
            double cum_sum = 0.0;
            double cum_sq  = 0.0;
            double last_rel_err = 0.0;
            for (int i = 0; i < static_cast<int>(eval_full.size()); ++i) {
                cum_sum += eval_full[i];
                cum_sq  += eval_full[i] * eval_full[i];
                const int rem_n = n - (i + 1);
                if (rem_n <= 0) break;
                const double tail_sum_sq = std::max(0.0, trace_K2 - cum_sq);
                const double lam_tail    = (trace_K_full - cum_sum) / static_cast<double>(rem_n);
                const double tail_var    = std::max(0.0, tail_sum_sq / static_cast<double>(rem_n) - lam_tail * lam_tail);
                const double tail_nonisotropic_energy = static_cast<double>(rem_n) * tail_var;
                const double relative_frobenius_error = (trace_K2 > 0.0)
                    ? std::sqrt(tail_nonisotropic_energy / trace_K2) : 0.0;
                last_rel_err = relative_frobenius_error;
                if (relative_frobenius_error < ctx.woodbury_basis_var_thresh) {
                    res.k_target = i + 1;
                    crossed_target = true;
                    break;
                }
            }
            res.satisfied = crossed_target;
            // last_rel_err is the error at the crossing point when satisfied, or the
            // smallest value reached so far (at the current k_svd) otherwise.
            res.metric = last_rel_err;
            break;
        }
        case WoodburyBasisSpec::Kind::Fixed:
        default:
            res.satisfied = true;
            res.k_target  = ctx.woodbury_basis ? ctx.woodbury_basis->fixed_rank : 0;
            break;
    }
    return res;
}

static int finalize_and_log_woodbury_rank(
    WoodburyBasisSpec::Kind mode,
    const RemlCtx& ctx,
    const Eigen::VectorXd& eval_full,
    int k_svd,
    int k_svd_budget_ceiling,
    double lambda_plus,
    double trace_K_full,
    double trace_K2,
    const RankEvalResult& eval_res)
{
    const int n = ctx.n;
    const bool k_max_is_hard_ceiling = (ctx.woodbury_basis_k_max > 0);
    const int k_cap = woodbury_rank_cap(ctx, k_svd_budget_ceiling);
    int k = 0;

    switch (mode) {
        case WoodburyBasisSpec::Kind::MP: {
            const int k_edge = eval_res.k_target;
            const int k_signal = eval_res.k_extra;
            k = std::max(20, std::min(k_svd, k_edge));
            const int band = std::max(0, k_edge - k_signal);
            LOGGER << "Woodbury MP-k: bulk edge lambda+=" << lambda_plus
                   << " (n=" << n << ")"
                   << ", eigenvalues above lambda+=" << k_signal
                   << ", edge band confirmed after " << band << " additional eigenvalue(s)"
                   << " (margin=" << (ctx.woodbury_basis_edge_margin * 100.0) << "%, confirm="
                   << ctx.woodbury_basis_edge_confirm << " consecutive)"
                   << ", using k=" << k << std::endl;
            if (!eval_res.satisfied && k_svd >= n - 1)
                LOGGER.e(0, "Woodbury MP-k: edge band not confirmed even at k=n-1=" + std::to_string(n - 1)
                         + "; this GRM has near-full effective rank and Woodbury may not offer a computational advantage here. Refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied && k_svd >= k_svd_budget_ceiling && !k_max_is_hard_ceiling)
                LOGGER.e(0, "Woodbury MP-k: edge band not confirmed within the memory-budget-implied ceiling k="
                         + std::to_string(k_svd) + " (--reml-woodbury-basis-mem-budget=" + std::to_string(ctx.svd_mem_budget_gb) + "GB). Raise the budget or lower --reml-woodbury-basis-edge-margin/-confirm; refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied)
                LOGGER.e(0, "Woodbury MP-k: edge band not confirmed within k_max=" + std::to_string(k_svd)
                         + ". Raise --reml-woodbury-basis-range's k_max; refusing to proceed with an unresolved basis.");
            break;
        }
        case WoodburyBasisSpec::Kind::EIG: {
            const int k_EIGMASS = eval_res.k_target;
            k = std::max(20, std::min(k_svd, k_EIGMASS + ctx.woodbury_basis_EIG_k_buffer));
            double cumulative = 0.0;
            for (int i = 0; i < k; ++i) cumulative += eval_full[i];
            const double rho = cumulative / trace_K_full;
            if (eval_res.satisfied) {
                LOGGER << "Woodbury EIG-k: trace(K)=" << trace_K_full
                       << ", raw " << ctx.woodbury_basis_eigen_mass * 100 << "% mass crossing at k=" << k_EIGMASS
                       << ", using k=" << k << " (+" << ctx.woodbury_basis_EIG_k_buffer << " eigenvalue buffer)"
                       << ", captured mass rho=" << rho << std::endl;
            } else {
                LOGGER << "Woodbury EIG-k: trace(K)=" << trace_K_full
                       << ", target mass (" << ctx.woodbury_basis_eigen_mass * 100 << "%) NOT reached within k=" << k_svd
                       << ", using fallback k=" << k
                       << ", captured mass rho=" << rho << std::endl;
            }
            if (!eval_res.satisfied && k_svd >= n - 1)
                LOGGER.e(0, "Woodbury EIG-k: mass target not reached even at k=n-1=" + std::to_string(n - 1)
                         + " (captured rho=" + std::to_string(rho) + "). Refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied && k_svd >= k_svd_budget_ceiling && !k_max_is_hard_ceiling)
                LOGGER.e(0, "Woodbury EIG-k: mass target not reached within memory budget ceiling k=" + std::to_string(k_svd)
                         + " (captured rho=" + std::to_string(rho) + "). Raise --reml-woodbury-basis-mem-budget; refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied && k_max_is_hard_ceiling)
                LOGGER.e(0, "Woodbury EIG-k: mass target not reached within k_max=" + std::to_string(k_cap)
                         + " (captured rho=" + std::to_string(rho) + "). Raise --reml-woodbury-basis-range's k_max; refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied)
                LOGGER.e(0, "Woodbury EIG-k: mass target not reached within k=" + std::to_string(k_svd)
                         + " (captured rho=" + std::to_string(rho) + "). Refusing to proceed with an unresolved basis.");
            break;
        }
        case WoodburyBasisSpec::Kind::VAR: {
            const int k_VAR = eval_res.k_target;
            k = std::max(20, std::min(k_svd, k_VAR));
            double cum_sum = 0.0;
            double cum_sq  = 0.0;
            for (int i = 0; i < k; ++i) {
                cum_sum += eval_full[i];
                cum_sq  += eval_full[i] * eval_full[i];
            }
            const int rem_n = n - k;
            const double tail_sum_sq = std::max(0.0, trace_K2 - cum_sq);
            const double lam_tail    = (rem_n > 0) ? ((trace_K_full - cum_sum) / static_cast<double>(rem_n)) : 0.0;
            const double tail_var    = (rem_n > 0) ? std::max(0.0, tail_sum_sq / static_cast<double>(rem_n) - lam_tail * lam_tail) : 0.0;
            const double tail_nonisotropic_energy = static_cast<double>(rem_n) * tail_var;
            const double relative_frobenius_error = (trace_K2 > 0.0)
                ? std::sqrt(tail_nonisotropic_energy / trace_K2) : 0.0;
            LOGGER << "Woodbury VAR-k: trace(K)=" << trace_K_full
                   << ", target relative Frobenius tail error < " << ctx.woodbury_basis_var_thresh
                   << " crossed at k=" << k_VAR << ", using k=" << k
                   << " (tail_d_var=" << tail_var
                   << ", tail non-isotropic energy=" << tail_nonisotropic_energy
                   << ", relative Frobenius error=" << relative_frobenius_error << ")" << std::endl;
            if (!eval_res.satisfied && k_svd >= n - 1)
                LOGGER.e(0, "Woodbury VAR-k: target relative Frobenius tail error not reached even at k=n-1=" + std::to_string(n - 1)
                         + " (relative Frobenius error=" + std::to_string(relative_frobenius_error) + "). Refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied && k_svd >= k_svd_budget_ceiling && !k_max_is_hard_ceiling)
                LOGGER.e(0, "Woodbury VAR-k: target relative Frobenius tail error not reached within memory budget ceiling k=" + std::to_string(k_svd)
                         + " (relative Frobenius error=" + std::to_string(relative_frobenius_error) + "). Raise --reml-woodbury-basis-mem-budget; refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied && k_max_is_hard_ceiling)
                LOGGER.e(0, "Woodbury VAR-k: target relative Frobenius tail error not reached within k_max=" + std::to_string(k_cap)
                         + " (relative Frobenius error=" + std::to_string(relative_frobenius_error) + "). Raise --reml-woodbury-basis-range's k_max; refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied)
                LOGGER.e(0, "Woodbury VAR-k: target relative Frobenius tail error not reached within k=" + std::to_string(k_svd)
                         + " (relative Frobenius error=" + std::to_string(relative_frobenius_error) + "). Refusing to proceed with an unresolved basis.");
            break;
        }
        case WoodburyBasisSpec::Kind::Fixed:
        default:
            k = ctx.woodbury_basis ? ctx.woodbury_basis->fixed_rank : 0;
            break;
    }
    return k;
}

// Exact lower bound on additional eigenvalues needed to reach target_mass,
// derived from eval_full being non-increasing (a property of eigenvalues,
// not a fitted assumption). Every undiscovered eigenvalue is <= lambda_k,
// so no k below this bound can possibly satisfy the EigMass criterion.
static int eigmass_min_k_next(const Eigen::VectorXd& eval_full, int k_svd, double S1_k, double target_mass) {
    const double lambda_k = eval_full[k_svd - 1];
    if (!(lambda_k > 0.0)) return k_svd;  // degenerate: no positive tail mass left to exploit
    const double mass_to_remove = target_mass - S1_k;
    if (!(mass_to_remove > 0.0)) return k_svd;  // already satisfied or overshot; shouldn't reach here
    const double m_needed = std::ceil(mass_to_remove / lambda_k);   // may be huge if lambda_k is tiny
    return k_svd + static_cast<int>(std::max(1.0, std::min(m_needed, 1.0e9)));
}

} // anonymous namespace

void compute_woodbury_basis(RemlCtx& ctx) {
    if (ctx.reml_mtd == RemlAlg::Fisher)
        LOGGER.e(0, "--reml-woodbury-basis is incompatible with Fisher-scoring REML.");
    if (!ctx.woodbury_basis.has_value())
        LOGGER.e(0, "--reml-woodbury-basis mode is not set.");
    if ((int)ctx.r_indx.size() != 2)
        LOGGER.e(0, "--reml-woodbury-basis supports only single-GRM models.");
    if (ctx.A[ctx.r_indx[0]].size() == 0 && ctx.grm_chunked_budget <= 0.0 && !ctx.grm_fp16)
        LOGGER.e(0, "--reml-woodbury: GRM component is identity; cannot compute basis.");
    if ((ctx.grm_chunked_budget > 0.0 || ctx.grm_fp16) && !ctx.grm_tile_reader)
        LOGGER.e(0, "--reml-woodbury: tiled GRM mode is set but ctx.grm_tile_reader is empty "
                    "— the GRM component wasn't actually loaded either way.");

    const auto mode = ctx.woodbury_basis->kind;
    const int  n = ctx.n;
    const bool grm_chunked = (ctx.grm_chunked_budget > 0.0) || ctx.grm_fp16;

    int k_svd_budget_ceiling = n - 1;
    if (grm_chunked && ctx.svd_mem_budget_gb > 0.0) {
        const double budget_bytes = ctx.svd_mem_budget_gb * 1e9;
        // Peak live rSVD sketch memory is ~2.2-2.4 * n * k_ext * sizeof(float)
        // (Q + AQ, with omega freed before power iterations and Q*Z rotated in-place).
        const int max_k_ext = static_cast<int>(budget_bytes / (2.4 * n * sizeof(float)));
        k_svd_budget_ceiling = std::min(k_svd_budget_ceiling, std::max(20, max_k_ext - 200));
        LOGGER << "--reml-woodbury-basis-mem-budget=" << ctx.svd_mem_budget_gb
               << "GB -> k_svd capped at " << k_svd_budget_ceiling << std::endl;
    }

    int grm_chunk_rows = 0;
    if (grm_chunked) {
        grm_chunk_rows = setup_chunked_grm_stream(ctx, "--reml-woodbury-basis");
        if (ctx.grm_chunked_budget > 0.0) {
            // Y is not reserved by setup_chunked_grm_stream -- report its worst-case
            // size (at the current k_svd ceiling) here so it's visible without being enforced.
            const double y_worst_case_gb = static_cast<double>(sizeof(float)) * n * static_cast<double>(k_svd_budget_ceiling) / 1e9;
            LOGGER << "--reml-woodbury-basis: at the current k_svd ceiling of " << k_svd_budget_ceiling
                   << ", the basis matrix (Y) may need up to ~" << y_worst_case_gb
                   << "GB, not covered by --grm-chunked-budget." << std::endl;
        }
    }


    const bool k_max_is_hard_ceiling = (ctx.woodbury_basis_k_max > 0);
    const int k_svd_cap = woodbury_rank_cap(ctx, k_svd_budget_ceiling);
    int k_svd = woodbury_initial_k_svd(ctx, mode, k_svd_budget_ceiling);

    if (mode == WoodburyBasisSpec::Kind::Fixed) {
        if (k_svd <= 0)
            LOGGER.e(0, "--reml-woodbury-basis rank must be positive for fixed-k mode.");
        if (k_svd > k_svd_budget_ceiling)
            LOGGER.e(0, "--reml-woodbury-basis " + std::to_string(k_svd) + " exceeds the k_svd ceiling ("
                        + std::to_string(k_svd_budget_ceiling) + ") implied by --reml-woodbury-basis-mem-budget="
                        + std::to_string(ctx.svd_mem_budget_gb) + "GB; raise the budget or lower the rank.");
        LOGGER << "\nComputing Woodbury low-rank basis (k=" << k_svd << ") ..." << std::endl;
    } else {
        LOGGER << "\nComputing Woodbury basis (" << woodbury_mode_name(mode)
               << ", k_init=" << k_svd << ", k_max=" << k_svd_cap
               << (k_max_is_hard_ceiling ? " [hard ceiling]" : " [soft cap]") << ") ..." << std::endl;
    }

    if (k_svd > k_svd_cap) {
        LOGGER.w(0, "--reml-woodbury-basis-range=(" + std::to_string(ctx.woodbury_basis_k_init) + "," + std::to_string(ctx.woodbury_basis_k_max) + ")" +
                    " exceeds the --reml-woodbury-basis-mem-budget-implied ceiling ("
                    + std::to_string(k_svd_budget_ceiling) + "); clamping to the effective cap "
                    + std::to_string(k_svd_cap) + ".");
        k_svd = k_svd_cap;
    }
    if (k_svd >= n) LOGGER.e(0, "--reml-woodbury-basis rank must be < n.");

    // K_f32 is upper-triangle-only storage (row <= col valid; see
    // grm_binary_io.hpp upper_only load path). No valid data below the
    // diagonal.
    // On the chunked path, trace(K) and trace(K^2) are accumulated during the
    // first rSVD matvec pass (inside chunked_symmetric_matvec_f32) rather than
    // running two separate pre-scans of .grm.bin before the first apply() call.
    const Eigen::MatrixXf& K_f32 = ctx.A[ctx.r_indx[0]];
    double trace_K_full = 0.0;
    double trace_K2 = 0.0;
    bool have_chunked_traces = !grm_chunked || ctx.grm_fp16;
    if (ctx.grm_fp16) {
        trace_K_full = ctx.grm_fp16_trace_K;
        trace_K2     = ctx.grm_fp16_trace_K2;
    } else if (!grm_chunked) {
        trace_K_full = K_f32.diagonal().cast<double>().sum();
        // Previously read K_f32.col(j).tail(n-j-1) -- rows j+1..n-1 of
        // column j, i.e. the LOWER triangle. That was a correctness bug
        // waiting to happen under upper-only storage (it happened to be
        // contiguous and correct only because the now-removed mirror pass
        // kept the lower triangle populated). Fixed the same way as
        // init_varcomp's trK2: col(j).head(j) reads rows 0..j-1 of column
        // j -- the UPPER triangle -- and is equally contiguous, so this is
        // a pure win, not a tradeoff.
        double diag_sq = K_f32.diagonal().cast<double>().squaredNorm();
        double off_sq  = 0.0;
        #pragma omp parallel for reduction(+:off_sq) schedule(guided)
        for (int j = 0; j < n; ++j)
            off_sq += K_f32.col(j).head(j).cast<double>().squaredNorm();
        trace_K2 = diag_sq + 2.0 * off_sq;
    }

    double lambda_plus = 0.0;
    if (mode == WoodburyBasisSpec::Kind::MP) {
        double M = 0.0;
        if (ctx.grm_N.rows() == n && ctx.grm_N.cols() == n) {
            if (grm_chunked)
                LOGGER.w(0, "--grm-chunked-budget: ctx.grm_N is a dense n x n matrix — this defeats "
                            "the memory savings from chunking K. If your SNP-count-per-pair GRM_N "
                            "is roughly constant, pass it as a 1x1 scalar via ctx.grm_N instead.");
            M = ctx.grm_N.diagonal().mean();
        } else if (ctx.grm_N.size() == 1) {
            M = ctx.grm_N(0, 0);
        }
        if (M <= 0.0)
            LOGGER.e(0, "--reml-woodbury-basis MP: cannot determine SNP count. Use --reml-woodbury-basis <k>.");
        const double gamma = static_cast<double>(n) / M;
        lambda_plus = std::pow(1.0 + std::sqrt(gamma), 2.0);
    }
    double target_mass = (mode == WoodburyBasisSpec::Kind::EIG && have_chunked_traces)
        ? (ctx.woodbury_basis_eigen_mass * trace_K_full) : 0.0;

    auto apply = [&](const Eigen::Ref<const Eigen::MatrixXf>& X, bool exact_fp32) -> Eigen::MatrixXf {
        const bool use_bf16 = !exact_fp32 && gcta_chunked::default_bf16_sketch();
        if (grm_chunked) {
            if (ctx.grm_tile_reader_f32) {
                double* tr_ptr  = have_chunked_traces ? nullptr : &trace_K_full;
                double* tr2_ptr = have_chunked_traces ? nullptr : &trace_K2;
                Eigen::MatrixXf Y = gcta_chunked::chunked_symmetric_matvec_f32(
                    ctx.grm_tile_reader_f32, n, grm_chunk_rows, X, tr_ptr, tr2_ptr, use_bf16);
                if (!have_chunked_traces) {
                    have_chunked_traces = true;
                    if (mode == WoodburyBasisSpec::Kind::EIG)
                        target_mass = ctx.woodbury_basis_eigen_mass * trace_K_full;
                }
                return Y;
            }
            if (!have_chunked_traces) {
                trace_K_full = gcta_chunked::chunked_diagonal(ctx.grm_tile_reader, n, grm_chunk_rows).sum();
                trace_K2 = gcta_chunked::chunked_trace_K_squared(ctx.grm_tile_reader, n, grm_chunk_rows);
                have_chunked_traces = true;
                if (mode == WoodburyBasisSpec::Kind::EIG)
                    target_mass = ctx.woodbury_basis_eigen_mass * trace_K_full;
            }
            return gcta_chunked::chunked_symmetric_matvec(
                ctx.grm_tile_reader, n, grm_chunk_rows, X.cast<double>()).cast<float>();
        }
        return gcta_chunked::dense_upper_symm_f32(K_f32, X, use_bf16);
    };

    Eigen::VectorXd eval_full;
    Eigen::MatrixXf evec_full;
    RankEvalResult eval_res;

    // Rank-expansion rounds lock the previous round's converged Ritz vectors and
    // sketch only the new directions (gcta_eigh::expand_symmetric_eigh), so each
    // column is touched once across rounds; that is what pays for q > 3.
    const int q_pow = std::max(1, ctx.svd_power_iter);
    // All k_ext Ritz pairs of a round are kept, not just the top k_svd: the oversample
    // vectors are already converged-ish and locking them means no column is ever
    // recomputed. The rank criterion only sees the leading k_svd values (eval_full).
    Eigen::MatrixXf V0;        // previous round's Ritz vectors (n x k_ext_prev, float32)
    Eigen::VectorXd th0;       // matching Ritz values (double)
    Eigen::VectorXd eval_all;  // this round's k_ext Ritz values (non-Nystrom, double)
    double prev_mass = 0.0;    // sum of the previous round's Ritz values
    int    prev_k    = 0;
    bool   round_expanded = false;
    if (!ctx.svd_nystrom && mode != WoodburyBasisSpec::Kind::Fixed)
        LOGGER << "Woodbury: rSVD power iterations = " << q_pow
               << "; rank expansions reuse converged Ritz vectors." << std::endl;

    if (ctx.svd_nystrom && ctx.Uk.size() > 0) {
        ctx.Uk.resize(0, 0);  // Nystrom does not consume warm-start vectors; release immediately
    }

    for (;;) {
        const int oversample = gcta_eigh::recommended_oversample(k_svd);
        const int k_ext = std::min(k_svd + oversample, n - 1);

        if (ctx.svd_nystrom) {
            try {
                gcta_eigh::EighResultF res = gcta_eigh::nystrom_symmetric_eigh<float>(
                    apply, n, k_svd,
                    gcta_eigh::recommended_oversample(k_svd));
                if (mode == WoodburyBasisSpec::Kind::EIG) {
                    LOGGER << "Woodbury EIG-k: refining Nystrom basis with one Rayleigh-Ritz projection ..."
                           << std::endl;
                    res = gcta_eigh::rayleigh_ritz_refine<float>(apply, std::move(res.eigenvectors), k_svd);
                }
                eval_full = std::move(res.eigenvalues);
                evec_full = std::move(res.eigenvectors);
            } catch (const std::exception& e) {
                LOGGER.e(0, std::string("Woodbury Nystrom: ") + e.what());
            }

            const double eval_sum = eval_full.sum();
            if (eval_sum > trace_K_full * 1.01)
                throw std::runtime_error(
                    "Woodbury Nystrom: sum of estimated eigenvalues (" + std::to_string(eval_sum) +
                    ") exceeds trace(K) (" + std::to_string(trace_K_full) + ") — impossible for real eigenvalues.");
        } else {
            round_expanded = false;
            if (static_cast<int>(V0.cols()) == k_ext) {
                // Near full rank (k_ext is capped at n-1): the locked basis already holds k_ext
                // exact Ritz pairs of its own span; nothing to compute.
                evec_full = std::move(V0);
                eval_all  = std::move(th0);
                round_expanded = true;
            } else if (V0.cols() > 0 && k_ext > static_cast<int>(V0.cols())) {
                try {
                    gcta_eigh::EighResultF res =
                        gcta_eigh::expand_symmetric_eigh<float>(apply, V0, th0, k_ext, k_ext, q_pow);
                    eval_all  = std::move(res.eigenvalues);
                    evec_full = std::move(res.eigenvectors);
                    round_expanded = true;
                } catch (const std::exception& e) {
                    LOGGER.w(0, std::string("Woodbury: locked expansion failed (") + e.what() +
                                "); recomputing this round from a fresh sketch.");
                }
            }
            if (!round_expanded) {
                const bool has_warm = (ctx.Uk.rows() == n && ctx.Uk.cols() > 0);
                if (has_warm) {
                    const int k_warm = std::min(static_cast<int>(ctx.Uk.cols()), k_ext);
                    LOGGER << "Woodbury: seeding initial rSVD sketch with " << k_warm
                           << " warm-start vector(s) (--reml-woodbury-basis-warm-start)." << std::endl;
                }
                // Copies ctx.Uk into omega and frees ctx.Uk BEFORE apply(omega) allocates Y.
                auto [omega, Y] = gcta_eigh::build_randomized_sketch_consume<float>(
                    apply, n, k_ext, has_warm ? &ctx.Uk : nullptr, /*exact_fp32=*/false);
                omega.resize(0, 0);
                const int k_proj = (mode == WoodburyBasisSpec::Kind::Fixed) ? k_svd : k_ext;
                try {
                    gcta_eigh::EighResultF res =
                        gcta_eigh::power_iterate_and_project<float>(apply, std::move(Y), k_proj, q_pow);
                    eval_all  = std::move(res.eigenvalues);
                    evec_full = std::move(res.eigenvectors);
                } catch (const std::exception& e) {
                    LOGGER.e(0, std::string("Woodbury: ") + e.what());
                }
            }
            V0.resize(0, 0);             // consumed (or moved from); free before the criterion runs
            th0.resize(0);
            eval_full = eval_all.head(k_svd);
        }

        if (mode != WoodburyBasisSpec::Kind::Fixed) {
            // Evaluate the rank criterion up front so the per-round log can report
            // each mode's actual tested quantity, not a generic proxy shared across
            // modes. Captured mass at any fixed rank must be non-decreasing across
            // locked-expansion rounds (Cauchy interlacing); a regression means
            // numerical trouble, not a statistical fluke.
            eval_res = evaluate_rank_criterion(mode, eval_full, k_svd, lambda_plus, target_mass, trace_K_full, trace_K2, ctx);
            const double mass_now = eval_full.sum();
            const char* expansion_tag = round_expanded ? " (locked expansion)" : "";

            switch (mode) {
                case WoodburyBasisSpec::Kind::MP:
                    // MP's test is the discrete bulk-edge band, not a smooth scalar:
                    // report the count above lambda+ and how far the confirmation
                    // run has gotten toward woodbury_basis_edge_confirm.
                    LOGGER << "Woodbury MP-k: k_svd=" << k_svd
                           << ", lambda+=" << lambda_plus
                           << ", eigenvalues above lambda+=" << eval_res.k_extra
                           << ", edge-band run=" << eval_res.run_len << "/" << ctx.woodbury_basis_edge_confirm
                           << (eval_res.satisfied ? " (confirmed)" : "")
                           << expansion_tag << std::endl;
                    break;
                case WoodburyBasisSpec::Kind::EIG:
                    LOGGER << "Woodbury EIG-k: k_svd=" << k_svd
                           << ", captured mass rho=" << (trace_K_full > 0.0 ? eval_res.metric / trace_K_full : 0.0)
                           << " (target " << ctx.woodbury_basis_eigen_mass * 100.0 << "%)"
                           << expansion_tag << std::endl;
                    break;
                case WoodburyBasisSpec::Kind::VAR:
                    // VAR's actual convergence test is the relative Frobenius tail
                    // error against woodbury_basis_var_thresh, not captured mass.
                    LOGGER << "Woodbury VAR-k: k_svd=" << k_svd
                           << ", relative Frobenius tail error=" << eval_res.metric
                           << " (target < " << ctx.woodbury_basis_var_thresh << ")"
                           << (eval_res.satisfied ? " (confirmed)" : "")
                           << expansion_tag << std::endl;
                    break;
                default:
                    break;
            }

            if (round_expanded && prev_k > 0 && static_cast<int>(eval_full.size()) >= prev_k) {
                const double mass_at_prev_rank = eval_full.head(prev_k).sum();
                if (mass_at_prev_rank < prev_mass - 1e-9 * trace_K_full)
                    LOGGER.w(0, "Woodbury: captured mass at rank " + std::to_string(prev_k) +
                                " decreased across a locked expansion (" + std::to_string(prev_mass) + " -> " +
                                std::to_string(mass_at_prev_rank) + "); please report this.");
            }
            prev_mass = mass_now;
            prev_k    = static_cast<int>(eval_full.size());
        }

        if (mode == WoodburyBasisSpec::Kind::Fixed) break;

        if (eval_res.satisfied || k_svd >= k_svd_cap || k_svd >= k_svd_budget_ceiling) break;

        // This is the actual expansion rank for the next round, not a ceiling or cap on anything computed here
        int k_svd_next = std::min({k_svd * 2, n - 1, k_svd_cap});
        std::string expand_note;
        if (mode == WoodburyBasisSpec::Kind::EIG && ctx.woodbury_basis_eigen_adaptive) {
            const double captured = eval_full.head(k_svd).sum();
            int k_svd_jump = eigmass_min_k_next(eval_full, k_svd, captured, target_mass);
            k_svd_next = std::min(std::max(k_svd_jump, k_svd_next), k_svd_cap);
            // Locked expansion recomputes nothing, so doubling can overshoot once the target is close.
            // Sufficient bound: the m largest of the n-k uncaptured eigenvalues average at least their mean
            // R/(n-k), so m = deficit*(n-k)/R more eigenvalues reach the target (+5% of k for Ritz lag).
            // It only binds near the target; far away it is larger than the doubling and changes nothing.
            const double deficit = target_mass - captured, remain = trace_K_full - captured;
            if (!ctx.svd_nystrom && deficit > 0.0 && remain > 0.0) {
                const double k_suff = std::max<double>(k_svd_jump, k_svd + std::ceil(deficit * (n - k_svd) / remain))
                                      + 0.05 * k_svd;
                k_svd_next = std::min(k_svd_next, static_cast<int>(std::min(k_suff, static_cast<double>(n))));
            }
            // mass-sufficient estimate vs. what's actually used next round (see note above).
            expand_note = ", mass-sufficient estimate k_svd=" + std::to_string(k_svd_jump);
        }
        const char* warm_status = ctx.svd_nystrom ? "full recompute" : "locked expansion, reusing converged Ritz vectors";
        LOGGER << "Woodbury " << woodbury_mode_name(mode)
               << ": k_svd=" << k_svd << " not sufficient; expanding to k_svd=" << k_svd_next
               << expand_note << " (" << warm_status << ") ..." << std::endl;
        if (!ctx.svd_nystrom) {
            V0  = std::move(evec_full);   // evec_full is reassigned next round
            th0 = std::move(eval_all);
        }
        k_svd = k_svd_next;
    }
    V0.resize(0, 0);                      // release before finalizing basis
    th0.resize(0);
    eval_all.resize(0);

    // Release dense/packed/chunked GRM storage and tile readers immediately post-basis.
    if (!ctx.woodbury_basis_posthoc_correction) {
        ctx.A[ctx.r_indx[0]].resize(0, 0);
        if (ctx.grm_fp16_release) ctx.grm_fp16_release();
        ctx.grm_tile_reader.release();
        ctx.grm_tile_reader = nullptr;
        ctx.grm_tile_reader_f32 = nullptr;
    }

    int k = finalize_and_log_woodbury_rank(mode, ctx, eval_full, k_svd, k_svd_budget_ceiling, lambda_plus, trace_K_full, trace_K2, eval_res);

    Eigen::VectorXd eval = eval_full.head(k);
    eval_full.resize(0);

    // Woodbury uses a PSD low-rank-plus-isotropic surrogate. Project the
    // signed Ritz spectrum before deriving its residual tail moments.
    eval = eval.cwiseMax(0.0);
    const double trace_K = trace_K_full;
    ctx.lambda_tail = (n - k > 0) ? ((trace_K - eval.sum()) / static_cast<double>(n - k)) : 0.0;
    if (ctx.lambda_tail < 0.0) {
        LOGGER.w(0, "Woodbury: lambda_tail < 0 (" + std::to_string(ctx.lambda_tail) + "); clamped to 0.");
        ctx.lambda_tail = 0.0;
    }

    const double tail_sum_sq = std::max(0.0, trace_K2 - eval.squaredNorm());
    ctx.tail_d_var = (n - k > 0) ? std::max(0.0, tail_sum_sq / static_cast<double>(n - k)
                                    - ctx.lambda_tail * ctx.lambda_tail) : 0.0;

    ctx.dk = std::move(eval);
    if (evec_full.cols() != k) {
        evec_full.conservativeResize(Eigen::NoChange, k);
    }
    ctx.Uk = std::move(evec_full);
    ctx.woodbury_basis_rank_ = k;
    ctx.Vi_use_woodbury_basis = true;
    ctx.wb_subspace_ready = false;

    LOGGER << "Woodbury basis: k=" << k
           << ", lambda_tail=" << ctx.lambda_tail
           << ", tail_d_var=" << ctx.tail_d_var << std::endl;
}

void compute(RemlCtx& ctx,
             const std::vector<double>& priors,
             const std::vector<double>& priors_var,
             bool no_constrain) {
    const bool priors_flag = !priors.empty() || !priors_var.empty();

    {
        RemlVec y_tmp = ctx.y.array() - ctx.y.mean();
        ctx.y_Ssq = y_tmp.squaredNorm() / (ctx.n - 1.0);
        if (!(std::fabs(ctx.y_Ssq) < 1e30))
            LOGGER.e(0, "phenotypic variance is infinite. Check phenotype file.");
    }

    if (!priors_var.empty() && (int)priors_var.size() < (int)ctx.r_indx.size() - 1) {
        std::ostringstream errmsg;
        errmsg << "in option --reml-priors-var. There are " << ctx.r_indx.size()
               << " variance components. At least " << ctx.r_indx.size() - 1
               << " prior values should be specified.";
        LOGGER.e(0, errmsg.str());
    }
    if (!priors.empty() && (int)priors.size() < (int)ctx.r_indx.size() - 1) {
        std::ostringstream errmsg;
        errmsg << "in option --reml-priors. There are " << ctx.r_indx.size()
               << " variance components. At least " << ctx.r_indx.size() - 1
               << " prior values should be specified.";
        LOGGER.e(0, errmsg.str());
    }

    LOGGER << "\nPerforming REML analysis ... (Note: may take hours depending on sample size)." << std::endl;
    if (ctx.n < 10) LOGGER.e(0, "sample size is too small.");
    LOGGER << ctx.n << " observations, " << ctx.X_c << " fixed effect(s), and "
           << ctx.r_indx.size() << " variance component(s) (including residual)." << std::endl;

    // Woodbury basis (must be done before init_varcomp for HE warm-start)
    if (ctx.woodbury_basis_preloaded) {
        // Basis supplied by the caller (--reml-woodbury-reuse): Uk/dk/lambda_tail/
        // woodbury_basis_rank_ are already set and ctx.A[GRM] is empty, i.e. the
        // same state compute_woodbury_basis() leaves behind.
        if (ctx.reml_mtd == RemlAlg::Fisher)
            LOGGER.e(0, "--reml-woodbury-reuse is incompatible with Fisher-scoring REML.");
        if ((int)ctx.r_indx.size() != 2)
            LOGGER.e(0, "--reml-woodbury-reuse supports only single-GRM models.");
        if (ctx.Uk.rows() != ctx.n || ctx.Uk.cols() != ctx.woodbury_basis_rank_
                || ctx.dk.size() != ctx.woodbury_basis_rank_)
            LOGGER.e(0, "--reml-woodbury-reuse: preloaded basis has inconsistent dimensions.");
        ctx.wb_subspace_ready = false;
        LOGGER << "Woodbury basis reused: k=" << ctx.woodbury_basis_rank_
               << ", lambda_tail=" << ctx.lambda_tail << std::endl;
    } else if (ctx.woodbury_basis.has_value()) {
        LOGGER.ts("woodbury");
        compute_woodbury_basis(ctx);
        float duration = LOGGER.tp("woodbury");
        LOGGER.i(0, "Woodbury basis computation took " + std::to_string(duration) + " seconds.");
    } else if (ctx.reml_trace_hutchpp && (ctx.grm_chunked_budget > 0.0 || ctx.grm_fp16)) {
        // Woodbury takes precedence when both are requested.
        if ((int)ctx.r_indx.size() != 2)
            LOGGER.e(0, "--grm-chunked-budget/--grm-fp16 with --reml-trace-hutchpp supports only single-GRM models.");
        ctx.grm_chunk_rows = setup_chunked_grm_stream(ctx, "--reml-trace-hutchpp");
    } else if (ctx.grm_chunked_budget > 0.0 || ctx.grm_fp16) {
        if ((int)ctx.r_indx.size() != 2)
            LOGGER.e(0, "--grm-chunked-budget/--grm-fp16 supports only single-GRM models.");
        ctx.grm_chunk_rows = setup_chunked_grm_stream(
            ctx, ctx.grm_fp16 ? "--grm-fp16" : "--grm-chunked-budget");
    }

    // Precompute spectral-subspace cache once before HE warm-start and iter 0.
    if (ctx.Vi_use_woodbury_basis) {
        ensure_woodbury_subspace_cache(ctx);
    }

    RemlMat Vi_X_out, Xt_Vi_X_i_out(ctx.X_c, ctx.X_c);
    RemlMat Hi(ctx.r_indx.size(), ctx.r_indx.size());
    RemlVec Py;
    if (!ctx.Vi_use_woodbury_basis) {
        Vi_X_out.resize(ctx.n, ctx.X_c);
        Py.resize(ctx.n);
    }
    RemlVec varcmp;
    ctx.he_warm_start_applied = init_varcomp(ctx, priors_var, priors, varcmp);

    const double lgL = reml_iteration(ctx, Vi_X_out, Xt_Vi_X_i_out, Hi, Py, varcmp, priors_flag, no_constrain);
    ctx.logL = lgL;
    ctx.has_logL = true;

    // Compute fixed effects: b = (X'V^{-1}X)^{-1} X'V^{-1}y
    if (ctx.Vi_use_woodbury_basis) {
        const RemlVec XtViy = ctx.Uk_Vi_X.transpose() * ctx.UkTy + ctx.wb_Xty_perp / ctx.sigma2_eff;
        ctx.b = Xt_Vi_X_i_out * XtViy;
    } else {
        ctx.b = Xt_Vi_X_i_out * (Vi_X_out.transpose() * ctx.y);
    }

    // Store variance components as a std::vector
    ctx.varcmp.resize(ctx.r_indx.size());
    for (int i = 0; i < (int)ctx.r_indx.size(); i++) ctx.varcmp[i] = varcmp[i];

    // Export a minimal non-bivariate REML summary used by MLMA-style .hsq output.
    const int m = static_cast<int>(ctx.r_indx.size());
    ctx.varcmp_se.assign(m, 0.0);
    for (int i = 0; i < m; ++i)
        ctx.varcmp_se[i] = std::sqrt(std::max(0.0, Hi(i, i)));

    const Eigen::Index me = static_cast<Eigen::Index>(m);
    const double Vp = varcmp.head(me).sum();
    const double VarVp = Hi.topLeftCorner(me, me).sum();
    ctx.Vp = Vp;
    ctx.Vp_se = std::sqrt(std::max(0.0, VarVp));

    const int ngen = std::max(0, m - 1);
    ctx.hsq.assign(ngen, 0.0);
    ctx.hsq_se.assign(ngen, 0.0);
    for (int i = 0; i < ngen; ++i) {
        const double V1 = varcmp[i];
        const double VarV1 = Hi(i, i);
        const double cov12 = Hi.row(i).head(me).sum();
        if (Vp > 0.0 && V1 > 0.0) {
            const double ratio = V1 / Vp;
            const double var_hsq = ratio * ratio *
                (VarV1 / (V1 * V1) + VarVp / (Vp * Vp) - (2.0 * cov12) / (V1 * Vp));
            ctx.hsq[i] = ratio;
            ctx.hsq_se[i] = std::sqrt(std::max(0.0, var_hsq));
        }
    }

    // Ensure V^{-1} is available for downstream MLMA streaming.
    // If neither Woodbury nor Hutch++ path: calcu_Vi with factorize_only=true so
    // we get L in ctx.Vi_L (or Vi in ctx.Vi) without needing P.
    if (!ctx.reml_trace_hutchpp && !ctx.Vi_use_woodbury_basis) {
        double logdet_dummy = 0.0;
        int iter_dummy = 0;
        calcu_Vi(ctx, varcmp, logdet_dummy, iter_dummy, /*factorize_only=*/true);
    }
}

RemlState build_reml_state(RemlCtx& ctx) {
    RemlState rs;
    rs.n           = static_cast<int32_t>(ctx.n);
    rs.x_c         = static_cast<int32_t>(ctx.X_c);
    rs.is_woodbury = ctx.Vi_use_woodbury_basis;

    // Design goal: keep the compact float-only RemlState as the final resident
    // store, and avoid holding a second full-size double buffer alive longer than
    // necessary. A true swap/move is impossible across Eigen's different scalar
    // types (MatrixXd vs MatrixXf), so the conversion must happen once while
        // filling the final float storage directly. Once the float copies are created,
    // we can immediately release the expensive double workspace: the streaming
    // association path consumes only the float state from here onward.
    rs.b.resize(ctx.b.size());
    rs.b.noalias() = ctx.b.cast<float>();

    rs.varcmp.resize(static_cast<Eigen::Index>(ctx.varcmp.size()));
    rs.varcmp.noalias() = Eigen::Map<const Eigen::VectorXd>(ctx.varcmp.data(),
                                                            static_cast<Eigen::Index>(ctx.varcmp.size()))
                            .cast<float>();

    if (ctx.Vi_use_woodbury_basis) {
        // WoodburyMLMACache stores Uk_f as k×n ColMajor for GEMM efficiency.
        // Cache-block (64x64) the n×k -> k×n float32 transpose to eliminate
        // 2 MB-stride cache/STLB misses (40-60 s -> ~0.2 s at N=500k, k=20k).
        const Eigen::Index k = ctx.Uk.cols();
        const Eigen::Index n = ctx.Uk.rows();
        rs.wb.Uk_f.resize(k, n);
        constexpr Eigen::Index kBlock = 64;
        const float* src_base = ctx.Uk.data();
        float* dst_base = rs.wb.Uk_f.data();
        #pragma omp parallel for collapse(2) schedule(static)
        for (Eigen::Index i0 = 0; i0 < n; i0 += kBlock) {
            for (Eigen::Index j0 = 0; j0 < k; j0 += kBlock) {
                const Eigen::Index i1 = std::min(n, i0 + kBlock);
                const Eigen::Index j1 = std::min(k, j0 + kBlock);
                alignas(64) float tile[kBlock][kBlock + 4];
                const Eigen::Index ilen = i1 - i0;
                const Eigen::Index jlen = j1 - j0;
                for (Eigen::Index j = 0; j < jlen; ++j) {
                    const float* src_col = src_base + i0 + static_cast<size_t>(j0 + j) * static_cast<size_t>(n);
                    for (Eigen::Index i = 0; i < ilen; ++i) {
                        tile[i][j] = src_col[i];
                    }
                }
                for (Eigen::Index i = 0; i < ilen; ++i) {
                    float* dst_col = dst_base + j0 + static_cast<size_t>(i0 + i) * static_cast<size_t>(k);
                    std::memcpy(dst_col, tile[i], static_cast<size_t>(jlen) * sizeof(float));
                }
            }
        }
        ctx.Uk.resize(0, 0); // free float32 ctx.Uk immediately now that Uk_f is populated

        rs.wb.ck_f.resize(ctx.ck.size());
        rs.wb.ck_f.noalias() = ctx.ck.cast<float>();
        ctx.ck.resize(0);

        rs.wb.sqrt_ck_f.resize(rs.wb.ck_f.size());
        rs.wb.sqrt_ck_f.noalias() = rs.wb.ck_f.cwiseSqrt();

        rs.wb.sigma2_eff_f = static_cast<float>(ctx.sigma2_eff);

        rs.dk_f.resize(ctx.dk.size());
        rs.dk_f.noalias() = ctx.dk.cast<float>();
        ctx.dk.resize(0);

        rs.lambda_tail_f = static_cast<float>(ctx.lambda_tail);
    } else if (ctx.Vi_use_llt) {
        // Vi_L holds the upper Cholesky factor U of V (V = U^T U, from dpotrf('U')).
        // Only cast the upper triangle (col(j).head(j + 1)) in parallel: MLMA_stream
        // only ever accesses CblasUpper / triangularView<Eigen::Upper>(), so leaving
        // the strict lower triangle untouched avoids faulting 2n² bytes of physical RAM.
        rs.is_llt = true;
        const Eigen::Index n = ctx.Vi_L.cols();
        rs.Vi_L_f.resize(ctx.Vi_L.rows(), n);
        #pragma omp parallel for schedule(guided)
        for (Eigen::Index j = 0; j < n; ++j) {
            rs.Vi_L_f.col(j).head(j + 1).noalias() =
                ctx.Vi_L.col(j).head(j + 1).cast<float>();
        }
        ctx.Vi_L.resize(0, 0);
    } else {
        gcta_blas_int blas_n_bs = static_cast<gcta_blas_int>(ctx.n);
        if (gcta_dpotrf(blas_n_bs, ctx.Vi.data(), blas_n_bs, 'U') != 0)
            LOGGER.e(0, "Vi is not positive definite when building REML state.");
        rs.is_llt = false;
        const Eigen::Index n = static_cast<Eigen::Index>(ctx.n);
        rs.Vi_L_f.resize(n, n);
        #pragma omp parallel for schedule(guided)
        for (Eigen::Index j = 0; j < n; ++j) {
            rs.Vi_L_f.col(j).head(j + 1).noalias() =
                ctx.Vi.col(j).head(j + 1).cast<float>();
        }
        ctx.Vi.resize(0, 0);
    }

    // Clear the remaining large, now-redundant double buffers after the float
    // copies are created. The downstream MLMA stream uses only the float state.
    ctx.A.clear();
    if (ctx.grm_fp16_release) ctx.grm_fp16_release();
    ctx.grm_tile_reader.release();
    ctx.grm_tile_reader = nullptr;
    ctx.grm_tile_reader_f32 = nullptr;
    ctx.grm_N.resize(0, 0);
    ctx.Vi_X.resize(0, 0);
    ctx.Xt_Vi_X_i.resize(0, 0);
    ctx.Uk_Vi_X.resize(0, 0);
    ctx.UkTX.resize(0, 0);
    ctx.UkTy.resize(0, 0);
    ctx.wb_subspace_ready = false;
    ctx.wb_XtX_perp.resize(0, 0);
    ctx.wb_Xty_perp.resize(0);
    ctx.wb_yty_perp = 0.0;
    ctx.hutchpp_S.resize(0, 0);
    ctx.hutchpp_G.resize(0, 0);
    ctx.hutchpp_qr_scratch.resize(0, 0);
    ctx.hutchpp_K.resize(0, 0);
    ctx.hutchpp_Q.resize(0, 0);
    ctx.hutchpp_QG.resize(0, 0);
    ctx.hutchpp_MQG.resize(0, 0);
    ctx.hutchpp_QtG.resize(0, 0);
    ctx.hutchpp_R.resize(0, 0);
    ctx.hutchpp_MR.resize(0, 0);
    ctx.grm_chunk_rows = 0;
    ctx.he_warm_start_applied = false;
    ctx.reml_tmp_n.resize(0);
    ctx.P.resize(0, 0);
    ctx.varcmp.clear();
    ctx.b.resize(0);
    ctx.Vi_use_woodbury_basis = false;
    ctx.Vi_use_llt = false;

    return rs;
}

} // namespace reml