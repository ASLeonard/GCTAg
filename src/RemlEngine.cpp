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
#include "mlma_woodbury.hpp"
#include "symmetric_eigendecomp.hpp"
#include "chunked_grm_matvec.hpp"

#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>
#include <random>
#include <algorithm>
#include <cmath>
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

// Woodbury helpers — mirrors gcta::woodbury_basis_Kv / woodbury_basis_Viv / woodbury_basis_ViZ
// but takes ctx by const ref instead of reading this->_ members.

RemlVec woodbury_basis_Kv(const RemlCtx& ctx, const RemlVec& v) {
    RemlVec Ukv = ctx.Uk.transpose() * v;
    Ukv.array() *= (ctx.dk.array() - ctx.lambda_tail);
    return ctx.lambda_tail * v + ctx.Uk * Ukv;
}

RemlMat woodbury_basis_KZ(const RemlCtx& ctx, const RemlMat& Z) {
    RemlMat UkZ = ctx.Uk.transpose() * Z;
    UkZ.array().colwise() *= (ctx.dk.array() - ctx.lambda_tail);
    return ctx.lambda_tail * Z + ctx.Uk * UkZ;
}

RemlVec woodbury_basis_Viv(const RemlCtx& ctx, const RemlVec& v) {
    RemlVec Ukv = ctx.Uk.transpose() * v;
    Ukv.array() *= ctx.ck.array();
    return (v - ctx.Uk * Ukv) / ctx.sigma2_eff;
}

RemlMat woodbury_basis_ViZ(const RemlCtx& ctx, const RemlMat& Z) {
    RemlMat UkZ = ctx.Uk.transpose() * Z;
    UkZ.array().colwise() *= ctx.ck.array();
    return (Z - ctx.Uk * UkZ) / ctx.sigma2_eff;
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

RemlMat applyP_mat(const RemlCtx& ctx, const RemlMat& Z) {
    RemlMat W;
    if (ctx.Vi_use_woodbury_basis) {
        W = woodbury_basis_ViZ(ctx, Z);
    } else if (ctx.Vi_use_llt) {
        W = Z;
        ctx.Vi_L.triangularView<Eigen::Upper>().adjoint().solveInPlace(W); // U^T z = v
        ctx.Vi_L.triangularView<Eigen::Upper>().solveInPlace(W);          // U w = z
    } else {
        W = RemlMat(ctx.Vi.selfadjointView<Eigen::Upper>() * Z);
    }
    const RemlMat A = ctx.Vi_X.transpose() * Z;
    W.noalias() -= ctx.Vi_X * (ctx.Xt_Vi_X_i.selfadjointView<Eigen::Lower>() * A);
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

//Returns true if variance components successfully warm-started via HE, false if not (e.g. single-GRM but priors specified)
bool init_varcomp(const RemlCtx& ctx,
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
        double trK = 0.0, trK2 = 0.0;
        RemlMat KX;
        RemlVec Ky(n);

        // Match REML's fixed-effect projection: for Q = I - X(X'X)^-1X',
        // solve E[y'QKQy] and E[y'Qy] for the two HE components.
        const RemlMat XtX = ctx.X.transpose() * ctx.X;
        Eigen::LDLT<RemlMat> XtX_ldlt(XtX);
        const int dof = n - ctx.X_c;
        if (XtX_ldlt.info() != Eigen::Success || dof <= 0) return false;
        const RemlVec y_r = ctx.y - ctx.X * XtX_ldlt.solve(ctx.X.transpose() * ctx.y);

        if (ctx.Vi_use_woodbury_basis) {
            const int k = ctx.woodbury_basis_rank_;
            trK  = ctx.dk.sum() + static_cast<double>(n - k) * ctx.lambda_tail;
            trK2 = ctx.dk.squaredNorm()
                 + static_cast<double>(n - k) * (ctx.tail_d_var + ctx.lambda_tail * ctx.lambda_tail);
            KX   = woodbury_basis_KZ(ctx, ctx.X);
            Ky   = woodbury_basis_Kv(ctx, y_r);
        } else if (ctx.grm_chunk_rows > 0) {
            // Exact GRM, streamed from disk in chunks -- ctx.A is empty by
            // design here, so this must come before (or instead of) the
            // ctx.A-based branch below. Mirrors the trace/matvec machinery
            // compute_woodbury_basis already uses for its chunked path.
            trK  = gcta_chunked::chunked_diagonal(ctx.grm_tile_reader, n, ctx.grm_chunk_rows).sum();
            trK2 = gcta_chunked::chunked_trace_K_squared(ctx.grm_tile_reader, n, ctx.grm_chunk_rows);
            KX   = gcta_chunked::chunked_symmetric_matvec(ctx.grm_tile_reader, n, ctx.grm_chunk_rows, ctx.X);
            Ky   = gcta_chunked::chunked_symmetric_matvec(ctx.grm_tile_reader, n, ctx.grm_chunk_rows, y_r);
        } else if (!ctx.A.empty() && ctx.A[ctx.r_indx[0]].size() > 0) {
            // Full exact GRM K = ctx.A[r_indx[0]]; K is upper-triangle-only
            // storage (see grm_binary_io.hpp upper_only load path) -- no
            // valid data below the diagonal.
            const auto& K = ctx.A[ctx.r_indx[0]];
            trK  = K.diagonal().sum();
            // K.squaredNorm() would undercount here: it sums whatever the
            // full n^2 buffer holds, but only the upper triangle (row<=col)
            // is valid -- the lower triangle is unfaulted/kernel-zero, not
            // K's actual values. Reconstruct tr(K^2) = sum(diag^2) +
            // 2*sum_{r<c} K(r,c)^2 explicitly. Column reads (col(j).head(j)
            // = K(0..j-1, j), the upper part of column j) are fully
            // contiguous -- strictly better than a row-based equivalent,
            // and squaring is order-independent so there's no accuracy
            // tradeoff either way.
            double diag_sq = K.diagonal().squaredNorm();
            double off_sq  = 0.0;
            #pragma omp parallel for reduction(+:off_sq) schedule(static)
            for (int j = 0; j < n; ++j)
                off_sq += K.col(j).head(j).squaredNorm();
            trK2 = diag_sq + 2.0 * off_sq;
            KX.noalias() = K.selfadjointView<Eigen::Upper>() * ctx.X;
            Ky.noalias() = K.selfadjointView<Eigen::Upper>() * y_r;
        }
        if (trK2 > 0.0) {
            const RemlMat XtKX = ctx.X.transpose() * KX;
            const RemlMat XtK2X = KX.transpose() * KX;
            const RemlMat B_inv_XtKX = XtX_ldlt.solve(XtKX);
            const double trQK = trK - B_inv_XtKX.trace();
            const double trQKQK = trK2 - 2.0 * XtX_ldlt.solve(XtK2X).trace()
                + (B_inv_XtKX * B_inv_XtKX).trace();
            const double yKy = y_r.dot(Ky);
            const double yy = y_r.squaredNorm();
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
bool verbose=false;


// Fill ctx.Vi (upper triangle + diagonal) with sum_ci varcmp[ci] * A[ci].
// Caller must have already called ctx.Vi.resize(n, n) and zeroed it.
// Identity-components (A.size()==0) are added to the diagonal only.
void assemble_V_upper(RemlCtx& ctx, const RemlVec& varcmp) {
    const int num_comp = static_cast<int>(ctx.r_indx.size());
    ctx.Vi.triangularView<Eigen::Upper>().setZero();

    if (ctx.grm_chunk_rows > 0) {
        // Single-GRM only (enforced where ctx.grm_chunk_rows is set) --
        // component 0 is the GRM, streamed tile-by-tile via the same
        // TileReader/chunk_rows already proven correct by
        // chunked_symmetric_matvec. Everything downstream of this function
        // (dpotrf, Vi_L, applyP_mat) is completely unchanged: ctx.Vi still
        // ends up fully dense -- this only changes how its upper triangle
        // gets filled in, so ctx.A[GRM] never needs to be resident.
        const double sg2 = varcmp[0];
        const gcta_chunked::BlockPartition part(ctx.n, ctx.grm_chunk_rows);
        const int m = part.num_blocks();
        for (int i = 0; i < m; ++i) {
            const int rs = part.block_start(i), re = part.block_end(i);
            for (int j = 0; j <= i; ++j) {
                const int cs = part.block_start(j), ce = part.block_end(j);
                const auto tile = ctx.grm_tile_reader(rs, re, cs, ce);  // (re-rs) x (ce-cs) view

                if (i == j) {
                    // Diagonal block: the tile's own LOWER triangle holds the
                    // valid data (row >= col) -- fixed by how
                    // ChunkedGrmReader/grm_binary_io.hpp read the GRM's
                    // on-disk packed-lower-triangular format.
                    for (int col = 0; col < re - rs; ++col)
                        ctx.Vi.col(cs + col).segment(rs, col + 1) +=
                            sg2 * tile.row(col).head(col + 1).transpose();

                } else {
                    // Off-diagonal block (i > j): block rows rs..re sit
                    // strictly below block cols cs..ce, i.e. this tile lives
                    // in the lower triangle. By symmetry the mirror-image
                    // block (rows cs..ce, cols rs..re) lives in the upper
                    // triangle, so write the transposed tile there instead.
                    ctx.Vi.block(cs, rs, ce - cs, re - rs) += sg2 * tile.transpose();
                }
            }
        }
    } else {
        // ctx.A is upper-triangle-only storage, so both the read
        // (ctx.A[...].col(j).head(j+1)) and the write
        // (ctx.Vi.col(j).head(j+1)) are now column-contiguous -- zero
        // stride, directly vectorizable.
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ctx.n; j++) {
            for (int ci = 0; ci < num_comp; ci++)
                if (ctx.A[ctx.r_indx[ci]].size() > 0)
                    ctx.Vi.col(j).head(j + 1) +=
                        varcmp[ci] * ctx.A[ctx.r_indx[ci]].col(j).head(j + 1);
        }
    }

    for (int ci = 0; ci < num_comp; ci++) {
        if (ctx.grm_chunk_rows > 0 && ci == 0)
            continue;  // GRM diagonal already added by the streamed tile loop above
        if (ctx.A[ctx.r_indx[ci]].size() == 0)
            ctx.Vi.diagonal().array() += varcmp[ci];
    }
}

// Returns false if V is not positive-definite and inversion failed.
bool calcu_Vi(RemlCtx& ctx, RemlVec& prev_varcmp, double& logdet, int& iter, bool factorize_only) {
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
            const double delta    = std::max(0.0, ctx.dk[j] - ctx.lambda_tail);
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
        ctx.Vi.triangularView<Eigen::Upper>().setZero();
        ctx.Vi.diagonal() = RemlVec::Constant(ctx.n, 1.0 / prev_varcmp[0]);
        logdet = ctx.n * std::log(prev_varcmp[0]);
    } else {
        assemble_V_upper(ctx, prev_varcmp);

        // LLT-only path (factorize_only, no diagV_adj)
        if (factorize_only && !ctx.reml_diagV_adj && !ctx.reml_force_dense_vi) {
            gcta_blas_int blas_n = static_cast<gcta_blas_int>(ctx.n);
            if (gcta_dpotrf(blas_n, ctx.Vi.data(), blas_n, 'U') == 0) {
                logdet = 2.0 * ctx.Vi.diagonal().array().log().sum();
                ctx.Vi_L.swap(ctx.Vi);
                ctx.Vi.resize(0, 0);
                ctx.Vi_use_llt = true;
                return true;
            }
            // dpotrf failed: reassemble from scratch (dpotrf may have partially overwritten Vi)
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
                // No mirror here: every downstream consumer of a dense ctx.Vi
                // (calcu_P_impl's Vi_X and its P construction) reads via
                // selfadjointView<Eigen::Upper>()/triangularView<Eigen::Upper>(),
                // exactly what dpotri('U') just populated -- mirroring the
                // lower triangle here would be a redundant O(n^2/2) copy
                // every REML iteration. The one consumer that genuinely
                // needs a fully dense ctx.Vi (the rare X'V^{-1}X-not-PD
                // fallback in calcu_P_impl) mirrors on demand there instead.
                return true;
            }
            // dpotrf/dpotri failed: reassemble for LU fallback
            ctx.Vi.resize(ctx.n, ctx.n);
            assemble_V_upper(ctx, prev_varcmp);
            ctx.Vi.triangularView<Eigen::Lower>() =
                ctx.Vi.triangularView<Eigen::Upper>().transpose();
            method_try = INV_LU;
        }

        if (!SquareMatrixInverse(ctx.Vi, logdet, rank, method_try)) {
            LOGGER << "Warning: V matrix is not invertible." << std::endl;
            if (ctx.reml_diagV_adj == 1) {
                LOGGER << "A small positive value is added to the diagonals. Results might not be reliable!" << std::endl;
                double d_buf = ctx.Vi.diagonal().mean() * ctx.reml_diag_mul;
                for (int j = 0; j < ctx.n; j++) ctx.Vi(j, j) += d_buf;
                if (!SquareMatrixInverse(ctx.Vi, logdet, rank, method_try)) {
                    LOGGER << "Still can't be inverted." << std::endl;
                    ret = false;
                }
            } else if (ctx.reml_diagV_adj == 2) {
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
    // Build Vi_X = V^{-1} X
    if (ctx.Vi_use_woodbury_basis) {
        if (ctx.UkTX.rows() != ctx.woodbury_basis_rank_ || ctx.UkTX.cols() != ctx.X_c)
            ctx.UkTX.noalias() = ctx.Uk.transpose() * ctx.X;
        if ((int)ctx.UkTy.size() != ctx.woodbury_basis_rank_)
            ctx.UkTy.noalias() = ctx.Uk.transpose() * ctx.y;
        const RemlMat ck_UkTX = ctx.ck.asDiagonal() * ctx.UkTX;
        ctx.Vi_X = (ctx.X - ctx.Uk * ck_UkTX) / ctx.sigma2_eff;
        ctx.Uk_Vi_X = (ctx.UkTX - ck_UkTX) / ctx.sigma2_eff;
    } else if (ctx.Vi_use_llt) {
        ctx.Vi_X = ctx.X;
        ctx.Vi_L.triangularView<Eigen::Upper>().adjoint().solveInPlace(ctx.Vi_X); // U^T z = X
        ctx.Vi_L.triangularView<Eigen::Upper>().solveInPlace(ctx.Vi_X);          // U w = z
    } else {
        ctx.Vi_X.noalias() = ctx.Vi.selfadjointView<Eigen::Upper>() * ctx.X;
    }
    ctx.Xt_Vi_X_i.noalias() = ctx.X.transpose() * ctx.Vi_X;

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
        RemlMat Uk_scaled = ctx.Uk;
        for (int j = 0; j < ctx.woodbury_basis_rank_; ++j)
            Uk_scaled.col(j) *= std::sqrt(ctx.ck[j] / ctx.sigma2_eff);
        ctx.Vi.selfadjointView<Eigen::Upper>().rankUpdate(Uk_scaled, -1.0);
        // No mirror: P's rank-update below only reads ctx.Vi's upper triangle.
    } else if (ctx.Vi_use_llt) {
        ctx.Vi.swap(ctx.Vi_L);
        gcta_blas_int blas_n_p = static_cast<gcta_blas_int>(ctx.n);
        if (gcta_dpotri(blas_n_p, ctx.Vi.data(), blas_n_p, 'U') != 0)
            LOGGER.e(0, "dpotri failed when materialising V^{-1} for P.");
        ctx.Vi_use_llt = false;
        // No mirror: P's rank-update below only reads ctx.Vi's upper triangle.
    }

    Eigen::LLT<RemlMat> llt(ctx.Xt_Vi_X_i);
    if (llt.info() == Eigen::Success) {
        RemlMat Z;
        Z.noalias() = ctx.Vi_X * llt.matrixL();
        P->swap(ctx.Vi);
        P->selfadjointView<Eigen::Upper>().rankUpdate(Z, -1.0);
        if (ctx.reml_mtd == 1)
            P->triangularView<Eigen::Lower>() = P->transpose();
    } else {
        // Rare path (X'V^{-1}X not positive definite): the subtraction below
        // is a plain dense op, not a triangular rank-update, so unlike the
        // common case above it needs a fully mirrored ctx.Vi. Mirror here,
        // on demand, rather than paying for it on every iteration above.
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
    const double sg2        = ctx.sg2;
    const double se2        = sigma2_eff - sg2 * ctx.lambda_tail;
    const double lambda_t   = ctx.lambda_tail;
    const int    n          = ctx.n;
    const int    k          = ctx.woodbury_basis_rank_;
    const double sum_ck     = ctx.ck.sum();
    const double tr_Vinv    = (n - sum_ck) / sigma2_eff;
    const double tr_Vinv_K  = (sg2 > 1e-15)
        ? (n * lambda_t / sigma2_eff + (se2 / (sg2 * sigma2_eff)) * sum_ck)
        : tr_Vinv * lambda_t;

    const int c = static_cast<int>(ctx.Vi_X.cols());
    RemlMat ViXTViX(c, c);
    ViXTViX.noalias() = ctx.Vi_X.transpose() * ctx.Vi_X;
    // trace(A*B) == (A.cwiseProduct(B.transpose())).sum() for any square A,B.
    // Both Xt_Vi_X_i and ViXTViX are c x c (c = covariate count, typically 1-5);
    // avoiding the c x c GEMM eliminates BLAS call overhead that would dominate
    // the actual arithmetic at such small sizes.
    const double tr_corr_I = ctx.Xt_Vi_X_i.cwiseProduct(ViXTViX.transpose()).sum();

    tr_PA(ncomp - 1) = tr_Vinv - tr_corr_I;

    RemlVec delta = ctx.dk.array() - ctx.lambda_tail;
    RemlMat delta_UkViX = delta.asDiagonal() * ctx.Uk_Vi_X;
    RemlMat ViX_K_ViX = lambda_t * ViXTViX;
    ViX_K_ViX.noalias() += ctx.Uk_Vi_X.transpose() * delta_UkViX;
    const double tr_corr_K = ctx.Xt_Vi_X_i.cwiseProduct(ViX_K_ViX.transpose()).sum();

    tr_PA(0) = tr_Vinv_K - tr_corr_K;

    // Delta-method (second-order Taylor) correction for tail heterogeneity: the
    // flat-tail approximation above treats every unretained eigenvalue as exactly
    // lambda_tail, which is only exact in expectation to first order. tail_d_var
    // (E[(d-lambda_tail)^2] over the tail) corrects the curvature terms this flat
    // approximation misses. Mirrors the existing logdet correction in calcu_Vi
    // (same r = sg2/sigma2_eff, same tail_d_var).
    //
    // NOT wired into tr_PA/live REML iteration -- that patching only this half 
    // of the score while leaving the AImatrix / quadratic score term 
    // (R, built in ai_reml from APy) uncorrected causes AI-REML to diverge once 
    // a component hits its clamp (score and curvature stop describing the same 
    // objective). This output is for a post-hoc, non-iterated correction only.
    // See compute_woodbury_posthoc_delta, which reuses this output.
    if (tr_PA_corrected) {
        tr_PA_corrected->resize(ncomp);
        double tr_Vinv_c   = tr_Vinv;
        double tr_Vinv_K_c = tr_Vinv_K;
        if (ctx.tail_d_var > 0.0 && n > k) {
            const double r = sg2 / sigma2_eff;
            tr_Vinv_c   += static_cast<double>(n - k) * r * r * ctx.tail_d_var / sigma2_eff;
            tr_Vinv_K_c -= static_cast<double>(n - k) * sg2 * se2 * ctx.tail_d_var
                          / (sigma2_eff * sigma2_eff * sigma2_eff);
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

    // Probe policy:
    // - reml_hutchpp_fixed_probes=false: redraw every call/iteration.
    // - reml_hutchpp_fixed_probes=true: keep probes fixed unless dimensions change.
    const bool need_resize = (ctx.hutchpp_S.rows() != ctx.n || ctx.hutchpp_S.cols() != k);
    if (need_resize) {
        ctx.hutchpp_S.resize(ctx.n, k);
        ctx.hutchpp_G.resize(ctx.n, k);

        // Persistent working-set scratch, sized once per (n,k) and reused
        // across every REML iteration and every component (ci) below --
        // this call sits in the hottest loop in the engine (once per ci per
        // AI-REML iteration), and every one of these was previously a fresh
        // stack RemlMat, i.e. malloc/free churn on each call. QG/MQG are
        // n x 2k: Q and G are pushed through the PA operator in one fused
        // call instead of two (see the ci loop below).
        ctx.hutchpp_qr_scratch.resize(ctx.n, k);
        ctx.hutchpp_K.resize(ctx.n, k);
        ctx.hutchpp_Q.resize(ctx.n, k);
        ctx.hutchpp_QG.resize(ctx.n, 2 * k);
        ctx.hutchpp_MQG.resize(ctx.n, 2 * k);
        ctx.hutchpp_QtG.resize(k, k);
        ctx.hutchpp_R.resize(ctx.n, k);
        ctx.hutchpp_MR.resize(ctx.n, k);
    }
    if (need_resize || !ctx.reml_hutchpp_fixed_probes) {
        std::uniform_int_distribution<int> coin(0, 1);
        for (int j = 0; j < k; j++)
            for (int r = 0; r < ctx.n; r++) {
                ctx.hutchpp_S(r, j) = coin(gcta_eigh::shared_rng()) ? 1.0 : -1.0;
                ctx.hutchpp_G(r, j) = coin(gcta_eigh::shared_rng()) ? 1.0 : -1.0;
            }
    }

    for (int ci = 0; ci < ncomp; ci++) {
        const bool is_I = (ctx.A[ctx.r_indx[ci]].size() == 0);

        auto applyPA_mat = [&](const RemlMat& Z) -> RemlMat {
            if (ctx.grm_chunk_rows > 0 && ci == 0) {
                return applyP_mat(ctx, RemlMat(gcta_chunked::chunked_symmetric_matvec(
                    ctx.grm_tile_reader, ctx.n, ctx.grm_chunk_rows, Z)));
            }
            if (ctx.Vi_use_woodbury_basis && ci == 0) {
                RemlMat UkZ = ctx.Uk.transpose() * Z;
                UkZ.array().colwise() *= (ctx.dk.array() - ctx.lambda_tail);
                RemlMat KZ = ctx.lambda_tail * Z + ctx.Uk * UkZ;
                return applyP_mat(ctx, KZ);
            }
            return is_I ? applyP_mat(ctx, Z)
                        : applyP_mat(ctx, RemlMat(ctx.A[ctx.r_indx[ci]].selfadjointView<Eigen::Upper>() * Z));
        };

        ctx.hutchpp_K.noalias() = applyPA_mat(ctx.hutchpp_S);
        // reml_trace_power_iter is 0 in normal operation, so this loop is a
        // no-op there; left on local temporaries rather than further hoisted
        // scratch -- revisit if power_iter is ever turned back on.
        for (int pw = 0; pw < ctx.reml_trace_power_iter; pw++) {
            Eigen::HouseholderQR<RemlMat> qr_pw(ctx.hutchpp_K);
            ctx.hutchpp_qr_scratch.setIdentity();
            ctx.hutchpp_K.noalias() = applyPA_mat(qr_pw.householderQ() * ctx.hutchpp_qr_scratch);
        }
        Eigen::HouseholderQR<RemlMat> qr(ctx.hutchpp_K);
        ctx.hutchpp_qr_scratch.setIdentity();
        ctx.hutchpp_Q.noalias() = qr.householderQ() * ctx.hutchpp_qr_scratch;

        // Fused MQ/MG evaluation: applyPA_mat is linear in its argument for
        // fixed ci, so applyPA_mat([Q G]) == [applyPA_mat(Q) applyPA_mat(G)]
        // exactly -- no numerical change from the previous two-call form.
        // One n x 2k call instead of two n x k calls halves the number of
        // applyP_mat invocations (each pays the Vi_X^T*Z / correction-term
        // overhead, plus a triangular solve on the LLT path) and gives the
        // underlying GEMMs double the RHS width to amortize call overhead
        // against. MQ/MG below are contiguous column-blocks of MQG (same
        // leading dimension as the parent matrix), so they still dispatch to
        // BLAS as plain submatrix GEMM operands, not a scalar fallback.
        ctx.hutchpp_QG.leftCols(k)  = ctx.hutchpp_Q;
        ctx.hutchpp_QG.rightCols(k) = ctx.hutchpp_G;
        ctx.hutchpp_MQG.noalias() = applyPA_mat(ctx.hutchpp_QG);
        const auto MQ = ctx.hutchpp_MQG.leftCols(k);
        const auto MG = ctx.hutchpp_MQG.rightCols(k);

        const double t_lr = ctx.hutchpp_Q.cwiseProduct(MQ).sum();

        ctx.hutchpp_QtG.noalias() = ctx.hutchpp_Q.transpose() * ctx.hutchpp_G;
        ctx.hutchpp_R.noalias()  = ctx.hutchpp_G - ctx.hutchpp_Q * ctx.hutchpp_QtG;
        ctx.hutchpp_MR.noalias() = MG - MQ * ctx.hutchpp_QtG;

        // per_col(j) is the j-th probe column's contribution to the residual
        // trace estimate; their mean is the estimate itself (unchanged from
        // before), their sample variance-of-the-mean is the noise-floor input.
        const RemlVec per_col = ctx.hutchpp_R.cwiseProduct(ctx.hutchpp_MR).colwise().sum();  // k values
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
            // Exact GRM, streamed: tr(P*K) = <P,K>_F (both symmetric), computed
            // from the same lower-triangular tiles assemble_V_upper reads, using
            // the same diagonal/2*off-diagonal decomposition the dense branch
            // below uses -- O(n^2), not a materialised O(n^3) K*P product.
            // P is upper-triangle-only at this point (mirrors the dense branch).
            const gcta_chunked::BlockPartition part(ctx.n, ctx.grm_chunk_rows);
            const int nb = part.num_blocks();
            double s = 0.0;
            // Serial, deliberately: ctx.grm_tile_reader is called directly here
            // (not via a gcta_chunked::* helper), and every other direct caller
            // in this file (assemble_V_upper's chunked block loop) is serial too
            // -- the raw reader is presumably a single stateful file/mmap
            // cursor, not proven safe for concurrent invocation the way the
            // gcta_chunked:: wrapper functions are.
            for (int bi = 0; bi < nb; ++bi) {
                const int rs = part.block_start(bi), re = part.block_end(bi);
                for (int bj = 0; bj <= bi; ++bj) {
                    const int cs = part.block_start(bj), ce = part.block_end(bj);
                    const auto tile = ctx.grm_tile_reader(rs, re, cs, ce);
                    if (bi == bj) {
                        for (int col = 0; col < re - rs; ++col) {
                            s += P(rs + col, rs + col) * tile(col, col);
                            if (col > 0)
                                s += 2.0 * P.col(rs + col).segment(rs, col)
                                            .dot(tile.row(col).head(col).transpose());
                        }
                    } else {
                        // Off-diagonal block: every (row,col) pair here is a
                        // distinct i<j pair (block bj strictly precedes block
                        // bi), visited exactly once across the (bi,bj) loop --
                        // doubled per the symmetric decomposition, same as the
                        // within-block off-diagonal term above.
                        s += 2.0 * P.block(cs, rs, ce - cs, re - rs)
                                    .cwiseProduct(tile.transpose()).sum();
                    }
                }
            }
            tr_PA(i) = s;
        } else if (ctx.A[ctx.r_indx[i]].size() == 0) {
            // Identity component: tr(PA) = tr(P). OMP parallel reduction over n
            // scalar loads has more overhead than gain; let Eigen vectorise it.
            tr_PA(i) = P.diagonal().sum();
        } else {
            // Ai is upper-triangle-only storage; Ai.col(col).tail(tail) read
            // rows col+1..n-1 of column col (lower triangle), no longer
            // valid. By symmetry Ai(col+1..n-1, col) == Ai(col, col+1..n-1)
            // -- read row col instead. Correctness fix only for now, same
            // as assemble_V_upper's pre-upper-triangle-swap fix -- not yet
            // perf-traced against a contiguous-copy alternative.
            const auto& Ai = ctx.A[ctx.r_indx[i]];
            double s = 0.0;
            #pragma omp parallel for reduction(+:s) schedule(guided)
            for (int col = 0; col < ctx.n; col++) {
                s += P(col, col) * Ai(col, col);
                if (col > 0)
                    s += 2.0 * P.col(col).head(col).dot(Ai.col(col).head(col));
            }
            tr_PA(i) = s;
        }
    }
}

void calcu_Hi(RemlCtx& ctx, RemlMat& P, RemlMat& Hi) {
    P.triangularView<Eigen::Lower>() = P.transpose();
    const int m = static_cast<int>(ctx.r_indx.size());

    std::vector<RemlMat> PA(m);
    for (int i = 0; i < m; i++) {
        if (ctx.Vi_use_woodbury_basis && i == 0) {
            // P is fully symmetrized above; use plain product for dgemm dispatch
            // (dsymm is slower than dgemm for rectangular Uk on AOCL/Zen4).
            RemlMat PUk(ctx.n, ctx.woodbury_basis_rank_);
            PUk.noalias() = P * ctx.Uk;
            RemlVec delta_v = ctx.dk.array() - ctx.lambda_tail;
            PA[i] = ctx.lambda_tail * P;
            PA[i].noalias() += PUk * delta_v.asDiagonal() * ctx.Uk.transpose();
        } else if (ctx.grm_chunk_rows > 0 && i == 0) {
            // Exact GRM, streamed. P is already fully symmetrized above, so
            // this is exactly the same chunked matvec used for APy/KX/Ky
            // elsewhere, just against the whole (now dense) P instead of a
            // handful of columns -- same O(n^3) cost class as the dense
            // Ai.selfadjointView*P branch below, just streamed from disk.
            PA[i] = gcta_chunked::chunked_symmetric_matvec(
                ctx.grm_tile_reader, ctx.n, ctx.grm_chunk_rows, P);
        } else if (ctx.A[ctx.r_indx[i]].size() == 0) {
            PA[i].resize(0, 0);
        } else {
            // ctx.A is upper-triangle only; use selfadjointView on A (not on the
            // already-full P) to avoid reading uninitialised lower-triangle elements.
            // A is symmetric so A*P == P*A; pass P.transpose() (==P) as the dense
            // RHS so dsymm fires with A as the symmetric operand. PA[i] is a full
            // (non-block) matrix, so this assignment is not subject to the
            // selfadjointView-on-block-target scalar-fallback issue.
            PA[i].noalias() = ctx.A[ctx.r_indx[i]].selfadjointView<Eigen::Upper>() * P;
        }
    }

    for (int i = 0; i < m; i++) {
        const bool i_id = (PA[i].size() == 0);
        for (int j = 0; j <= i; j++) {
            const bool j_id = (PA[j].size() == 0);
            double val;
            if (i_id && j_id)       val = P.squaredNorm();
            else if (i_id)          val = PA[j].cwiseProduct(P).sum();
            else if (j_id)          val = PA[i].cwiseProduct(P).sum();
            else                    val = PA[i].cwiseProduct(PA[j].transpose()).sum();
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

// Fisher-scoring REML update (legacy reml_equation equivalent).
// Requires explicit P materialisation; reml_iteration enforces that for mtd==1.
void reml_equation(RemlCtx& ctx, RemlMat& P, RemlMat& Hi, RemlVec& Py, RemlVec& varcmp) {
    calcu_Hi(ctx, P, Hi);
    if (ctx.reml_AI_not_invertible) return;

    Py.noalias() = P.selfadjointView<Eigen::Lower>() * ctx.y;

    const int m = static_cast<int>(ctx.r_indx.size());
    RemlVec R(m);
    if (ctx.reml_tmp_n.size() != ctx.n) ctx.reml_tmp_n.resize(ctx.n);
    for (int i = 0; i < m; i++) {
        if (ctx.A[ctx.r_indx[i]].size() == 0) {
            R(i) = Py.squaredNorm();
        } else {
            ctx.reml_tmp_n.noalias() = ctx.A[ctx.r_indx[i]].selfadjointView<Eigen::Upper>() * Py;
            R(i) = Py.dot(ctx.reml_tmp_n);
        }
    }

    varcmp = Hi * R;
    Hi = 2 * Hi;  // keep same SE convention as legacy path
}

// (1-alpha) quantile of the null (true-score == 0) distribution of
// lambda_sq = U'*AI^{-1}*U, given U ~ N(0, diag(var_U)) — the diagonal
// Hutch++ probe-noise covariance approximation. Under H0, lambda_sq is a
// weighted sum of independent chi-square(1) variables (a quadratic form in
// Gaussian noise), with weights = eigenvalues of
//     B = diag(sqrt(var_U)) * AI^{-1} * diag(sqrt(var_U))
// an m x m matrix (m = #variance components, so this eigendecomposition is
// negligible cost regardless of n). Approximated via Satterthwaite/Welch
// moment matching to a scaled chi-square g*Chi2_h, which matches the true
// first two moments of the weighted sum exactly:
//     mean = trace(B),  var = 2*trace(B^2)
//     g = trace(B^2)/trace(B),  h = trace(B)^2/trace(B^2)
// Returns 0 when var_U is (numerically) all-zero -- the deterministic
// exact/Woodbury case, where there is no noise term and the caller's
// logL-unit floor alone governs convergence.
double newton_decrement_null_quantile(const RemlMat& Hi, const RemlVec& var_U, double alpha) {
    const RemlVec sqrt_var = var_U.cwiseMax(0.0).cwiseSqrt();
    if (sqrt_var.squaredNorm() < 1e-300) return 0.0;   // fully deterministic; no noise term
    RemlMat B = sqrt_var.asDiagonal() * Hi * sqrt_var.asDiagonal();
    B = 0.5 * (B + B.transpose());   // guard tiny asymmetry from floating-point roundoff
    const double trB  = B.trace();
    const double trB2 = B.squaredNorm();   // == trace(B*B) for symmetric B
    if (trB2 < 1e-300) return 0.0;
    const double g = trB2 / trB;
    const double h = trB * trB / trB2;
    boost::math::chi_squared_distribution<double> chi2(h);
    return g * boost::math::quantile(chi2, 1.0 - alpha);
}

// lambda_sq / var_U are only meaningful on return when this call succeeds
// (i.e. reml_iteration hasn't already broken out via reml_AI_not_invertible)
// — see call site.
void ai_reml(RemlCtx& ctx, RemlMat& P, RemlMat& Hi, RemlVec& Py,
             RemlVec& prev_varcmp, RemlVec& varcmp, double dlogL,
             double& lambda_sq, RemlVec& var_U,
             bool last_step_trustworthy, double& step_scale_out) {
    const bool use_approx     = ctx.reml_trace_hutchpp;
    const bool woodbury_basis_active = ctx.Vi_use_woodbury_basis;

    if (use_approx || woodbury_basis_active)
        Py = applyP_vec(ctx, ctx.y);
    else
        Py.noalias() = P.selfadjointView<Eigen::Upper>() * ctx.y;

    const int m = static_cast<int>(ctx.r_indx.size());
    RemlMat APy(ctx.n, m);
    for (int i = 0; i < m; i++) {
        if (ctx.grm_chunk_rows > 0 && i == 0)
            APy.col(i) = gcta_chunked::chunked_symmetric_matvec(
                ctx.grm_tile_reader, ctx.n, ctx.grm_chunk_rows, Py);
        else if (woodbury_basis_active && i == 0)
            APy.col(i) = woodbury_basis_Kv(ctx, Py);
        else if (ctx.A[ctx.r_indx[i]].size() == 0)
            APy.col(i) = Py;
        else
            // APy.col(i) is a block-expression target -- selfadjointView
            // products assigned directly into a block silently fall back
            // to a scalar (non-BLAS) path in Eigen. Evaluate into a plain
            // RemlVec temporary first (forces the BLAS-dispatched product),
            // then copy into the block -- same pattern already used
            // elsewhere in this file (e.g. compute_woodbury_posthoc_delta).
            APy.col(i) = RemlVec(ctx.A[ctx.r_indx[i]].selfadjointView<Eigen::Upper>() * Py);
    }

    RemlVec R(m);
    if (use_approx || woodbury_basis_active) {
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

    RemlVec tr_PA;
    RemlVec tr_PA_var = RemlVec::Zero(m);   // deterministic (exact/Woodbury) => 0
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
    // Hi is now AI^{-1}. delta = AI^{-1}*U is both the Newton step and,
    // via lambda_sq = U'*delta, the Newton decrement -- no extra solve.
    RemlVec delta = Hi * R;
    lambda_sq = R.dot(delta);

    // Fraction-to-boundary rule (standard interior-point safeguard, e.g.
    // Wachter & Biegler / Nocedal & Wright Ch.19; tau=0.995 is the
    // conventional default, not tuned to any dataset). Applied unconditionally
    // (both legacy and --reml-ai-robust paths), independent of the trust/
    // curvature-confidence logic below: this is a geometry constraint, not a
    // model-confidence one. A raw Newton step that would drive a variance
    // component negative gets globally rescaled (not clamped component-wise
    // after the fact) so it lands just short of the boundary instead of
    // overrunning it and being force-floored by constrain_varcmp. This
    // matters because a hard-clamped, artificially degenerate point (rather
    // than wherever Newton's own trajectory would have stopped) produces an
    // unreliable AI-matrix estimate on the *next* call -- observed directly
    // on real data: a full step onto a clamped near-zero component was
    // followed by a step with a genuine logL decrease, costing several
    // iterations of recovery. This rule is a no-op whenever no component is
    // within tau of the boundary, which is the overwhelming majority of
    // steps.
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

    // Step-size damping. Legacy behaviour (default): GCTA's original fixed
    // 0.316 (~1/sqrt(10)) shrink whenever the *previous* iteration's logL
    // change exceeded 1.0 -- an undocumented magic constant, and a backward-
    // looking gate (dlogL says nothing about whether *this* step's local
    // quadratic model is trustworthy).
    //
    // Under --reml-ai-robust: replace it with a continuous damped-Newton
    // step scaled by the Newton decrement itself (lambda_sq, already computed
    // above for the robust stopping criterion, reused here rather than
    // introducing a second unrelated diagnostic). lambda_sq/2 is the predicted
    // logL improvement from a full step; large lambda_sq means the current
    // point is far enough from the optimum that the quadratic model backing
    // Hi shouldn't be trusted at full strength, so the step is shrunk
    // proportionally to sqrt(target_decrement / lambda_sq) rather than gated
    // on an unrelated threshold. Converges to a full Newton step (scale -> 1)
    // as lambda_sq -> 0, matching the legacy scheme's asymptotic behaviour
    // near convergence while differing (continuously, not via a step
    // function) far from it.
    double step_scale = 1.0;
    if (dlogL > 1.0) step_scale = 0.316;
    if (ctx.reml_mtd == 0 && ctx.reml_ai_robust) {
        // last_step_trustworthy: the *previous* iteration's own quadratic
        // model (lambda_sq/step_scale at the time) predicted a certain logL
        // gain; if the actual gain came in close to that prediction, Hi has
        // just been validated as locally trustworthy and this step takes
        // the full Newton step rather than re-damping from scratch. This is
        // one bit of memory, not a persistent radius -- no ratchet, no cap,
        // no expand/shrink schedule. Falls back to exactly the pre-existing
        // formula (unchanged target_decrement=0.5) whenever the previous
        // step hasn't (yet) proven itself.
        constexpr double target_decrement = 0.5;
        step_scale = last_step_trustworthy
            ? 1.0
            : std::min(1.0, std::sqrt(target_decrement / std::max(lambda_sq, 1e-12)));
    }
    step_scale = std::min(step_scale, fraction_to_boundary_cap);
    step_scale_out = step_scale;

    // Var(U_i) = 0.25 * Var(tr_PA_i) since only tr_PA is stochastic (Hutch++).
    // 0 for exact/Woodbury since tr_PA_var == 0. Left as a vector (not reduced
    // to trace(AI^{-1}*Var(U)) here) so the caller can build the full
    // null-distribution weight matrix, not just its trace.
    var_U = 0.25 * tr_PA_var;
    varcmp = prev_varcmp + step_scale * delta;
    if (verbose)
        LOGGER << "REML iteration: lambda_sq = " << std::to_string(lambda_sq) << ", step scale = " << std::to_string(step_scale) << ", delta = " << delta.transpose() << std::endl;
}

// Post-hoc, Woodbury-only tail correction. Called exactly once, only at
// convergence, only when ctx.woodbury_basis_posthoc_correction is set --
// NEVER from inside the AI-REML iteration loop, so it has zero effect
// (zero extra compute, zero behavioural change) on every iteration leading
// up to convergence, and zero cost at all when the flag is off.
//
// tr_PA's tail contribution gets the validated closed-form delta-method
// correction (calcu_tr_PA_woodbury's tr_PA_corrected output). R's tail
// contribution has no such closed form (see calcu_tr_PA_woodbury's own
// comment) -- instead R is computed exactly via one direct matvec against
// the true GRM (ctx.A) per score evaluation. Falls back to the flat-tail
// approximation with a warning if ctx.A has already been freed.
//
// Step size: NOT the raw Newton step Hi*U_corrected taken at full strength.
// Hi is the flat-tail model's curvature, evaluated at the flat-tail fixed
// point -- there is no reason to trust its *magnitude* once the score being
// solved is no longer the one it was derived from (confirmed empirically:
// the raw full step overshoots badly and by an increasing amount as the
// starting point approaches the true optimum). Instead of an arbitrary fixed
// damping factor, this evaluates the corrected score a second time, at the
// full-step trial endpoint, and takes a secant (regula falsi) step along the
// same direction Hi proposed -- i.e. it still trusts Hi's *direction*, but
// replaces its *magnitude* with one grounded in an actual second measurement
// of the corrected score, not an assumed constant. Costs one additional
// exact O(n^2) matvec (the trial-point score evaluation) on top of the one
// already paid for the current-point score -- still only paid once per
// converged run, still nothing paid unless the flag is on.
RemlVec compute_woodbury_posthoc_delta(RemlCtx& ctx, const RemlMat& Hi, const RemlVec& Py,
                                        const RemlVec& prev_varcmp, int iter) {
    const int m = static_cast<int>(ctx.r_indx.size());

    // Corrected score U(varcmp), evaluated at whatever point ctx is currently
    // set up for (Py must be V^{-1}-projected at that same point -- true both
    // for the current-point call below and the trial-point call further down).
    auto corrected_score = [&](const RemlVec& Py_at) -> RemlVec {
        RemlVec tr_PA_l(m), tr_PA_corrected_l(m);
        calcu_tr_PA_woodbury(ctx, tr_PA_l, &tr_PA_corrected_l);

        RemlVec R_raw_l(m);
        for (int i = 0; i < m; i++) {
            if (i == 0)
                R_raw_l(i) = Py_at.dot(woodbury_basis_Kv(ctx, Py_at));
            else if (ctx.A[ctx.r_indx[i]].size() == 0)
                R_raw_l(i) = Py_at.dot(Py_at);
            else
                R_raw_l(i) = Py_at.dot(RemlVec(ctx.A[ctx.r_indx[i]].selfadjointView<Eigen::Upper>() * Py_at));
        }

        RemlVec R_for_correction_l = R_raw_l;
        if (!ctx.A.empty() && ctx.A[ctx.r_indx[0]].size() > 0) {
            const RemlVec APy_exact_l = ctx.A[ctx.r_indx[0]].selfadjointView<Eigen::Upper>() * Py_at;   // the O(n^2) pass
            R_for_correction_l(0) = Py_at.dot(APy_exact_l);
        } else {
            LOGGER.w(0, "ctx.woodbury_basis_posthoc_correction is set but the exact GRM "
                        "(ctx.A) is not resident -- falling back to the flat-tail "
                        "approximation for R. Keep ctx.A resident through convergence "
                        "(skip the free in compute_woodbury_basis) to use this correction.");
        }
        return -0.5 * (tr_PA_corrected_l - R_for_correction_l);
    };

    // g(0): corrected score at the current (converged, flat-tail) point.
    // ctx is already correctly set up for prev_varcmp/Py on entry -- no
    // calcu_Vi call needed here, unlike the trial point below.
    const RemlVec U0 = corrected_score(Py);
    const RemlVec d  = U0; //Hi * U0;          // raw Newton direction (and, at t=1, the old full step)
    const double  g0 = d.dot(U0);

    // g(1): corrected score at the trial full-step endpoint. Requires moving
    // ctx to that point first (same skip_P=true, dense-P-free path the main
    // convergence refresh uses -- Woodbury cannot materialise dense P/Vi).
    RemlVec varcmp_trial = prev_varcmp + d;
    LOGGER << "REML: post-hoc secant trial point: varcmp_trial = " << varcmp_trial.transpose() << std::endl;

    double logdet_trial = 0.0;
    double t_star = 0.0;
    if (calcu_Vi(ctx, varcmp_trial, logdet_trial, iter, /*skip_P=*/true)) {
        calcu_P_impl(ctx, nullptr);
        const RemlVec Py_trial = applyP_vec(ctx, ctx.y);
        const RemlVec U1 = corrected_score(Py_trial);
        const double  g1 = d.dot(U1);

        constexpr double denom_eps = 1e-12;   // numerical-safety threshold, not a tuning knob
        if (std::fabs(g0 - g1) > denom_eps) {
            // Secant root of g(t) = g0 + t*(g1-g0) along the Hi-proposed
            // direction. Clamped to [0, 1] -- interpolate between "no
            // correction" (t=0, known-safe) and "full raw step" (t=1, known
            // to overshoot), never extrapolate beyond either.
            t_star = std::clamp(g0 / (g0 - g1), 0.0, 1.0);
        } else {
            LOGGER.w(0, "REML: post-hoc secant step denominator is degenerate "
                        "(corrected score barely changed over the trial step); "
                        "keeping the pre-correction estimate.");
        }
        LOGGER << "REML: post-hoc secant line search: g(0)=" << g0 << ", g(1)=" << g1
               << ", t*=" << t_star << std::endl;
    } else {
        LOGGER.w(0, "REML: post-hoc secant trial point is not positive-definite; "
                    "keeping the pre-correction estimate.");
    }

    // Restore ctx to the state it was in on entry (consistent with prev_varcmp
    // and the caller's Py) before returning, so this function has no side
    // effects the caller needs to know about or compensate for -- do not rely
    // on the caller happening to re-sync state right after this returns.
    // Local mutable copy: calcu_Vi requires non-const RemlVec& (it may
    // constrain/clamp values in place), and prev_varcmp is a const-ref
    // parameter here -- the mutated copy itself is discarded, only the side
    // effect on ctx state is wanted.
    RemlVec prev_varcmp_restore = prev_varcmp;
    double logdet_restore = 0.0;
    calcu_Vi(ctx, prev_varcmp_restore, logdet_restore, iter, /*skip_P=*/true);
    calcu_P_impl(ctx, nullptr);

    return t_star * d;
}

// Closed-form, non-iterative tail correction to the HE-moment variance
// component estimate. This is NOT a Newton step and does not touch the
// score, Hi, P, or Vi -- it recomputes sigma_g^2 via the same
// method-of-moments ratio used for the HE warm start (init_varcomp), but
// with tail statistics corrected using lambda_tail and tail_d_var.
//
// tr(K) and tr(K^2) are made fully EXACT here, not just improved: a sum of
// a degree-<=2 polynomial in the tail eigenvalues (tr(K) is degree 1,
// tr(K^2) is degree 2) is completely recoverable from the tail's mean and
// variance alone -- there is no Taylor remainder for these two terms, unlike
// log|V| or tr(V^-1 K), which needed higher moments for further accuracy.
//
// y'Ky cannot be made exact this way -- same obstruction as R: its tail
// contribution sum_{i>k} d_i*(w_i'y)^2 needs eigenvector alignment
// information tau^2 does not contain. Two options are given: the cheap
// flat-tail approximation (matches what the rest of REML already computes
// with, O(n*k)), or an exact O(n^2) pass against the true GRM if ctx.A is
// still resident. Neither is a correction to this specific term -- exact
// uses the real matrix, flat leaves the existing approximation as-is.
//
// The result is a standalone point estimate, entirely decoupled from
// whatever varcmp AI-REML converged to. It is NOT a refinement of that
// estimate and does not require it -- it can be computed and reported
// independently, e.g. purely as a diagnostic alongside the AI-REML result.
RemlVec compute_he_tail_corrected_varcmp(RemlCtx& ctx, bool use_exact_yKy) {
    const int k = ctx.woodbury_basis_rank_;
    const int n = ctx.n;

    // tr(K), tr(K^2): exact, from basis statistics already computed --
    // no extra cost, no matvec.
    const double trK  = ctx.dk.sum() + static_cast<double>(n - k) * ctx.lambda_tail;
    const double trK2 = ctx.dk.squaredNorm()
                       + static_cast<double>(n - k) * (ctx.tail_d_var + ctx.lambda_tail * ctx.lambda_tail);

    // y'Ky: flat-tail (cheap, O(n*k)) unless exact was requested and ctx.A
    // is available.
    double yKy;
    if (use_exact_yKy && !ctx.A.empty() && ctx.A[ctx.r_indx[0]].size() > 0) {
        yKy = ctx.y.dot(RemlVec(ctx.A[ctx.r_indx[0]].selfadjointView<Eigen::Upper>() * ctx.y));   // one O(n^2) pass
    } else {
        if (use_exact_yKy) {
            LOGGER.w(0, "compute_he_tail_corrected_varcmp: exact y'Ky requested but ctx.A "
                        "is not resident -- falling back to the flat-tail approximation.");
        }
        yKy = ctx.y.dot(woodbury_basis_Kv(ctx, ctx.y));
    }

    const double yy = ctx.y.squaredNorm();
    const double Vy = ctx.y_Ssq;
    const double denom = static_cast<double>(n) * trK2 - trK * trK;

    RemlVec varcmp_he(2);
    double sg_he = (denom > 1e-10)
        ? (static_cast<double>(n) * yKy - trK * yy) / denom
        : Vy * 0.5;
    // Same clamps as init_varcomp's HE warm start, for the same reason:
    // this is the identical estimator family and can fail the same way
    // (near-singular denominator, negative raw estimate).
    sg_he = std::max(sg_he, 0.01 * Vy);
    sg_he = std::min(sg_he, 0.99 * Vy);
    varcmp_he(0) = sg_he;
    varcmp_he(1) = std::max(Vy - sg_he, 0.01 * Vy);

    LOGGER << "REML: HE tail-corrected estimate (" << (use_exact_yKy ? "exact" : "flat") << " y'Ky): "
           << varcmp_he.transpose()
           << "  [trK=" << trK << " trK2=" << trK2 << " yKy=" << yKy << "]" << std::endl;

    return varcmp_he;
}

// tr_PA-only-corrected post-hoc step. No new matvec beyond what a converged
// AI-REML run already computed -- Hi is reused as-is, R stays the existing
// flat-tail approximation (woodbury_basis_Kv, O(n*k), no ctx.A needed).
// Applied as a single step, NOT iterated, NOT secant-refined: iterating or
// extrapolating this was shown earlier to accelerate rather than converge
// (growing corrections at successively closer starting points), so this is
// deliberately just one step from the actual converged point, nothing more.
RemlVec compute_woodbury_posthoc_delta_simple(RemlCtx& ctx, const RemlMat& Hi, const RemlVec& Py) {
    const int m = static_cast<int>(ctx.r_indx.size());

    RemlVec tr_PA(m), tr_PA_corrected(m);
    calcu_tr_PA_woodbury(ctx, tr_PA, &tr_PA_corrected);

    RemlVec R_flat(m);
    R_flat(0) = Py.dot(woodbury_basis_Kv(ctx, Py));
    for (int i = 1; i < m; i++) {
        R_flat(i) = (ctx.A[ctx.r_indx[i]].size() == 0)
            ? Py.dot(Py)
            : Py.dot(RemlVec(ctx.A[ctx.r_indx[i]].selfadjointView<Eigen::Upper>() * Py));
    }

    const RemlVec U_partial = -0.5 * (tr_PA_corrected - R_flat);
    return Hi * U_partial;
}

void em_reml(RemlCtx& ctx, RemlMat& P, RemlVec& Py,
             RemlVec& prev_varcmp, RemlVec& varcmp, double dlogL) {
    const bool woodbury_basis_active = ctx.Vi_use_woodbury_basis;
    const bool use_approx     = ctx.reml_trace_hutchpp;

    RemlVec tr_PA;
    if (woodbury_basis_active) {
        calcu_tr_PA_woodbury(ctx, tr_PA);
        Py = applyP_vec(ctx, ctx.y);
    } else if (use_approx) {
        const int eff_nprobes = (std::fabs(dlogL) < 1.0)
            ? ctx.reml_trace_hutchpp_nprobes
            : std::max(ctx.reml_trace_hutchpp_nprobes / 3, 30);
        RemlVec tr_PA_var_unused;   // EM-REML has no Newton-decrement consumer
        calcu_tr_PA_hutchpp(ctx, tr_PA, tr_PA_var_unused, eff_nprobes);
        Py = applyP_vec(ctx, ctx.y);
    } else {
        calcu_tr_PA(ctx, P, tr_PA);
        Py.noalias() = P.selfadjointView<Eigen::Upper>() * ctx.y;
    }

    const int m = static_cast<int>(ctx.r_indx.size());
    RemlVec R(m);
    if (ctx.reml_tmp_n.size() != ctx.n) ctx.reml_tmp_n.resize(ctx.n);
    for (int i = 0; i < m; i++) {
        if (ctx.grm_chunk_rows > 0 && i == 0) {
            ctx.reml_tmp_n = gcta_chunked::chunked_symmetric_matvec(
                ctx.grm_tile_reader, ctx.n, ctx.grm_chunk_rows, Py);
            R(i) = Py.dot(ctx.reml_tmp_n);
        } else if (woodbury_basis_active && i == 0) {
            ctx.reml_tmp_n = woodbury_basis_Kv(ctx, Py);
            R(i) = Py.dot(ctx.reml_tmp_n);
        } else if (ctx.A[ctx.r_indx[i]].size() == 0) {
            R(i) = Py.squaredNorm();
        } else {
            ctx.reml_tmp_n.noalias() = ctx.A[ctx.r_indx[i]].selfadjointView<Eigen::Upper>() * Py;
            R(i) = Py.dot(ctx.reml_tmp_n);
        }
        varcmp(i) = prev_varcmp(i) - prev_varcmp(i) * prev_varcmp(i) * (tr_PA(i) - R(i)) / ctx.n;
    }
    if (verbose)
        LOGGER << "REML iteration: step scale = 1.000000, delta = " << (varcmp - prev_varcmp).transpose() << std::endl;
}

double reml_iteration(RemlCtx& ctx,
                      RemlMat& Vi_X_out, RemlMat& Xt_Vi_X_i_out, RemlMat& Hi,
                      RemlVec& Py, RemlVec& varcmp,
                      bool prior_var_flag, bool no_constrain) {
    std::vector<std::string> mtd_str = {"AI-REML", "Fisher-scoring REML", "EM-REML"};
    const int m = static_cast<int>(ctx.r_indx.size());
    int constrain_num = 0, iter = 0;
    int reml_mtd_tmp = ctx.reml_mtd;
    double logdet = 0.0, logdet_Xt_Vi_X = 0.0;
    double prev_lgL = -1e20, lgL = -1e20, dlogL = 1000.0;
    double lambda_sq = 1e300;   // only meaningful when reml_mtd == 0
    RemlVec var_U;              // ditto -- Var(U) per component, from Hutch++ probe noise (0 for exact/Woodbury)
    RemlVec prev_prev_varcmp(varcmp), prev_varcmp(varcmp), varcomp_init(varcmp);

    // One-bit trust state for --reml-ai-robust's step scale (see ai_reml).
    // rho_threshold=0.75 is the standard trust-region "model was good"
    // cutoff (Nocedal & Wright), used here as a single yes/no gate rather
    // than a graduated radius. Unused when reml_ai_robust is off.
    //
    // Starts true (optimistic): attempt a full Newton step immediately on
    // iteration 0 rather than damping-by-default before there is any
    // evidence damping is needed. Self-correcting -- if iteration 0's full
    // step disappoints (rho <= rho_threshold, checked below), iteration 1
    // falls straight back to the damped formula. This does remove all
    // protection specifically on the single highest-risk step (zero prior
    // evidence on Hi's trustworthiness) -- validate against a poorly-
    // conditioned GRM, not just well-behaved data, before relying on this
    // for production runs.
    constexpr double rho_threshold = 0.75;
    bool last_step_trustworthy = true;
    double step_scale_taken = 1.0;
    bool have_predicted = false;
    double predicted_dlogL = 0.0;

    // ctx.reml_ai_robust_risk is the one number you set: the (Bonferroni,
    // approximate) probability that ANY iteration across this whole run
    // falsely declares convergence under H0, treating each iteration's fresh
    // Hutch++ probe draw as an independent test of "is the true score zero".
    // Dividing by reml_max_iter turns that into the per-iteration alpha the
    // test actually consumes -- this is the same alpha as before, just
    // reparameterised so it doesn't need re-justifying every time
    // reml_max_iter changes, and so the number you're setting is the
    // run-level quantity you actually care about rather than an intermediate
    // one. (Independence across iterations is reasonable here specifically
    // because fresh probes are redrawn every call -- see calcu_tr_PA_hutchpp
    // -- but this is still an approximation, not a rigorous sequential test.)
    const double ai_robust_alpha_per_iter =
        ctx.reml_ai_robust_risk / std::max(1, ctx.reml_max_iter);

    if (ctx.reml_trace_hutchpp && !ctx.Vi_use_woodbury_basis) {
        LOGGER << "Using Hutch++ stochastic trace estimator with "
               << ctx.reml_trace_hutchpp_nprobes << (ctx.reml_hutchpp_fixed_probes ? "fixed" : "fresh")  << " probes." << std::endl;
    }
    if (ctx.reml_mtd == 0 && ctx.reml_ai_robust) {
        LOGGER << "Using Newton-decrement convergence criterion (--reml-ai-robust, "
               << "logL tol=" << ctx.reml_ai_robust_tol
               << ", run-level false-stop risk=" << ctx.reml_ai_robust_risk
               << " => alpha/iter=" << ai_robust_alpha_per_iter << ")." << std::endl;
    }

    bool converged_flag = false;
    // small_delta_streak/M are only used by the legacy dlogL gate (mtd==1,
    // mtd==2, and mtd==0 when --reml-ai-robust is off): a single small
    // dlogL is a weak, uncalibrated signal there (no known false-stop rate),
    // so repetition is the only cheap way to build confidence. The AI-robust
    // path below doesn't need this -- the calibrated null-quantile test
    // already bounds the false-stop probability directly via alpha, so a
    // single iteration is sufficient by construction (see
    // newton_decrement_null_quantile for the caveats on that calibration).
    int small_delta_streak = 0;
    const int M = 1;   // consecutive small deltas required; small, fixed, not data-dependent
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
                // Fall back to EM-REML for round 0 whenever the run did NOT
                // actually start from an HE warm-start point -- either
                // because the user asked to skip it (--reml-no-he-start) or
                // because init_varcomp's warm-start preconditions weren't
                // met (multi-component model, priors supplied, or the HE
                // regression itself was ill-conditioned/degenerate and
                // declined to set varcmp). ctx.reml_no_HE_start alone isn't
                // enough here: it only reflects the user's request, not
                // whether a warm start was actually applied, and AI-REML's
                // Newton step from the naive equal-split starting point is
                // not reliably informative -- EM-REML is robust from any
                // starting point.
                if (ctx.reml_no_HE_start || !ctx.he_warm_start_applied) {
                  ctx.reml_mtd = 2;
                  LOGGER << "EM-REML ..." << std::endl;
                } else {
                  LOGGER << "AI-REML ..." << std::endl;
                }
            }
        }
        if (iter == 1) {
            ctx.reml_mtd = reml_mtd_tmp;
            LOGGER << "Running " << mtd_str[ctx.reml_mtd] << " algorithm ...\nIter.\tlogL\t";
            for (int i = 0; i < m; i++) LOGGER << ctx.var_name[ctx.r_indx[i]] << "\t";
            LOGGER << std::endl;
        }

        const bool skip_P_this_iter = (ctx.reml_trace_hutchpp || ctx.Vi_use_woodbury_basis) && ctx.reml_mtd != 1;
        if (!calcu_Vi(ctx, prev_varcmp, logdet, iter, skip_P_this_iter)) {
            LOGGER << "Warning: V matrix is not positive-definite." << std::endl;
            varcmp = prev_prev_varcmp;
            if (!calcu_Vi(ctx, varcmp, logdet, iter, false))
                LOGGER.e(0, "V matrix is not positive-definite.");
            // Rebuild P for Hi computation in break path
            RemlMat P_tmp(ctx.n, ctx.n);
            calcu_P_impl(ctx, &P_tmp);
            calcu_Hi(ctx, P_tmp, Hi);
            Hi = 2 * Hi;
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

        if (ctx.reml_mtd == 0)
            ai_reml(ctx, ctx.P, Hi, Py, prev_varcmp, varcmp, dlogL, lambda_sq, var_U,
                    last_step_trustworthy, step_scale_taken);
        else if (ctx.reml_mtd == 1)
            reml_equation(ctx, ctx.P, Hi, Py, varcmp);
        else if (ctx.reml_mtd == 2)
            em_reml(ctx, ctx.P, Py, prev_varcmp, varcmp, dlogL);
        // mtd 1 (Fisher scoring) requires full P. skip_P_this_iter is false for
        // mtd 1, so ctx.P is always materialised before reml_equation().

        lgL = -0.5 * (logdet_Xt_Vi_X + logdet + ctx.y.dot(Py));

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
            RemlMat P_tmp;
            calcu_P_impl(ctx, &P_tmp);
            calcu_Hi(ctx, P_tmp, Hi);
            Hi = 2 * Hi;
            break;
        }

        dlogL = lgL - prev_lgL;

        // Judge the step just taken against its own prediction, for use by
        // the *next* iteration's ai_reml call. Predicted gain for a step of
        // size s along the Newton direction is lambda_sq*s*(1-s/2) (from
        // delta'*AI*delta = delta'*U = lambda_sq, same identity used for the
        // Newton-decrement convergence test below) -- no extra solve, reuses
        // quantities ai_reml already computed. iter>0 guard: prev_lgL is
        // still the -1e20 sentinel at iter==0, so round 0's step has nothing
        // meaningful to be judged against yet -- last_step_trustworthy keeps
        // its optimistic initial value through iteration 0 and only starts
        // being judged from iteration 1 onward.
        const bool ai_robust_active_step = (ctx.reml_mtd == 0 && ctx.reml_ai_robust);
        if (ai_robust_active_step && iter > 0 && have_predicted && predicted_dlogL > 1e-12) {
            const double rho = dlogL / predicted_dlogL;
            last_step_trustworthy = (rho > rho_threshold);
            if (verbose) {
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

        // Convergence test. AI-REML (mtd==0) already computes AI^{-1} (Hi) and
        // the score U every iteration to take its Newton step, so the Newton
        // decrement lambda_sq = U'*AI^{-1}*U is a free byproduct with a known
        // closed-form meaning: for a Newton step delta = AI^{-1}*U under
        // curvature AI, the standard quadratic-approximation identity gives
        //     logL(theta*) - logL(theta) ~= 0.5 * U'*delta = 0.5 * lambda_sq
        // i.e. lambda_sq is (twice) the estimated log-likelihood gap remaining
        // to the optimum -- not an abstract unitless number. That lets the
        // floor be derived from an interpretable logL-unit tolerance
        // (ctx.reml_ai_robust_tol, same unit/default as the old absolute
        // dlogL<1e-4 gate) instead of being picked directly:
        //     lambda_sq_floor = 2 * reml_ai_robust_tol
        //
        // Under Hutch++, newton_decrement_null_quantile(Hi, var_U, alpha)
        // gives the (1-alpha) quantile of lambda_sq's own null distribution
        // (Satterthwaite-calibrated, see its docstring) -- so this is a
        // proper single-iteration hypothesis test with a chosen false-stop
        // probability, not a mean-comparison needing repetition to trust.
        // For exact/Woodbury, var_U == 0 so the quantile is 0 and this
        // reduces to lambda_sq <= lambda_sq_floor.
        //
        // Fisher-scoring (mtd==1) and EM-REML (mtd==2) don't materialise AI^{-1}
        // as part of their update (that's the point of EM), so no equally cheap
        // curvature estimate is available there; they keep the legacy logL-delta
        // gate, as does mtd==0 when --reml-ai-robust isn't set (this also
        // covers the iter==0 EM burn-in step used to pick priors).
        bool like_small;
        const bool ai_robust_active = (ctx.reml_mtd == 0 && ctx.reml_ai_robust);
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
            // Single-shot calibrated test -- no repetition needed (see above).
            converged_flag = like_small;
        } else {
            // Legacy path: require M consecutive small-delta iterations, not
            // just one. A single small delta can occur by chance -- e.g.
            // iteration 1 landing close to the prior logL purely because the
            // starting variance-component guess was already near-optimal,
            // with no evidence the underlying step itself has converged, and
            // dlogL carries no calibrated false-stop-rate guarantee the way
            // lambda_sq's null quantile does. Sustained small deltas are the
            // fallback signal; costs at most M-1 extra iterations in the rare
            // false-start case.
            small_delta_streak = like_small ? small_delta_streak + 1 : 0;
            converged_flag = (small_delta_streak >= M);
        }

        if (converged_flag) {
            //varcmp = best_varcmp;   // report best-seen, not last-iterate //Deactivated on purpose without testing, as this would no longer be a "convergence" method.
            if (ctx.Vi_use_woodbury_basis && ctx.reml_mtd == 0 && ctx.woodbury_basis_posthoc_correction) {
                const RemlVec delta_simple = compute_woodbury_posthoc_delta_simple(ctx, Hi, Py);   // side effect: logs the tr_PA correction
                 LOGGER << "REML: Woodbury post-hoc tail correction: delta = " << delta_simple.transpose() << std::endl;
            }   
            if (false) { //hard disable the secant step for now, as it is not well tested and may be unstable
                // Hi/Py here are exactly what the final ai_reml call above left
                // behind -- still consistent with prev_varcmp, the converged
                // (flat-tail) fixed point. Nothing above this point in the loop
                // pays any extra cost for this feature; this is the only place
                // it runs, and it runs at most once.
                const RemlVec delta_corrected = compute_woodbury_posthoc_delta(ctx, Hi, Py, prev_varcmp, iter);
                const RemlVec varcmp_post = prev_varcmp + delta_corrected;
                const double lgL_pre = lgL;
                varcmp = varcmp_post;

                // Refresh everything downstream to be self-consistent with the
                // corrected varcmp. Vi_X_out/Xt_Vi_X_i_out/Hi/lgL (-> ctx.logL)
                // are otherwise only captured once, below/by the caller, from
                // whatever state the last (pre-correction) iteration left
                // behind -- reassigning varcmp alone does not update any of
                // them, which is exactly what produced the earlier MLMA
                // regression from a hand-patched factorize_only attempt.
                //
                // Woodbury mode cannot materialize a dense P/Vi ("Woodbury REML
                // is incompatible with explicit V^{-1} materialisation") --
                // skip_P must stay true, and Py/Hi are rebuilt the same
                // implicit way ai_reml computes them every iteration
                // (applyP_vec/applyP_mat via the low-rank+tail basis), not via
                // calcu_Hi's dense-P path (that path is only ever exercised by
                // the reml_mtd==2 branch below, which is not reachable here).
                double logdet_post = 0.0;
                if (!calcu_Vi(ctx, varcmp, logdet_post, iter, /*skip_P=*/true)) {
                    LOGGER.w(0, "REML: post-hoc corrected variance components are not "
                                 "positive-definite; keeping the pre-correction estimate.");
                    varcmp = prev_varcmp;
                    calcu_Vi(ctx, varcmp, logdet_post, iter, true);
                }
                const double logdet_Xt_Vi_X_post = calcu_P_impl(ctx, nullptr);
                Py = applyP_vec(ctx, ctx.y);

                const int m_post = static_cast<int>(ctx.r_indx.size());
                RemlMat APy_post(ctx.n, m_post);
                for (int i = 0; i < m_post; i++) {
                    if (i == 0)
                        APy_post.col(i) = woodbury_basis_Kv(ctx, Py);
                    else if (ctx.A[ctx.r_indx[i]].size() == 0)
                        APy_post.col(i) = Py;
                    else
                        // Block-expression target (APy_post.col(i)) -- see ai_reml
                        // for why this must evaluate into a temporary first.
                        APy_post.col(i) = RemlVec(ctx.A[ctx.r_indx[i]].selfadjointView<Eigen::Upper>() * Py);
                }
                const RemlMat PAPy_post = applyP_mat(ctx, APy_post);
                Hi.noalias() = APy_post.transpose() * PAPy_post;
                Hi = 0.5 * Hi;
                if (!inverse_H(ctx, Hi)) {
                    LOGGER.w(0, "REML: AI matrix at the post-hoc corrected point is not "
                                 "invertible; SE reporting at the corrected point may be "
                                 "unreliable.");
                }
                lgL = -0.5 * (logdet_Xt_Vi_X_post + logdet_post + ctx.y.dot(Py));

                LOGGER << "REML: Woodbury post-hoc tail correction applied.\n"
                       << "\tConverged (flat-tail) estimate: " << prev_varcmp.transpose()
                       << "  (logL=" << lgL_pre << ")\n"
                       << "\tCorrected estimate:              " << varcmp.transpose()
                       << "  (logL=" << lgL << ")\n"
                       << "\tNote: reported SE(s) are the flat-tail model's curvature (AI^-1) "
                       << "evaluated at the corrected point, not a rigorously re-derived SE "
                       << "for the corrected estimator itself -- treat as approximate."
                       << std::endl;
            }
            if (ctx.reml_mtd == 2) {
                RemlMat P_tmp;
                calcu_P_impl(ctx, &P_tmp);
                calcu_Hi(ctx, P_tmp, Hi);
                Hi = 2 * Hi;
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
    // iter is zero-based in the loop; convert to total rounds executed.
    ctx.reml_iterations = std::max(0, iter + 1);
    ctx.P.resize(0, 0);

    // Export final Vi_X and Xt_Vi_X_i to output parameters
    Vi_X_out    = ctx.Vi_X;
    Xt_Vi_X_i_out = ctx.Xt_Vi_X_i;

    return lgL;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API implementations
// ─────────────────────────────────────────────────────────────────────────────

namespace reml {

namespace {

static const char* woodbury_mode_name(WoodburyMode mode) {
    switch (mode) {
        case WoodburyMode::MP:   return "MP-k";
        case WoodburyMode::EIG:  return "EIG-k";
        case WoodburyMode::VAR: return "VAR-k";
        case WoodburyMode::Fixed:    return "fixed-k";
        default:                     return "off";
    }
}

static int woodbury_rank_cap(const RemlCtx& ctx, int k_budget_ceiling) {
    const int n = ctx.n;
    const int user_cap = (ctx.woodbury_basis_k_max > 0) ? ctx.woodbury_basis_k_max : n - 1;
    return std::min({n - 1, k_budget_ceiling, user_cap});
}

static int woodbury_initial_k_svd(const RemlCtx& ctx, WoodburyMode mode, int k_budget_ceiling) {
    const int n = ctx.n;
    switch (mode) {
        case WoodburyMode::MP:
            return std::min({n - 1, k_budget_ceiling, ctx.woodbury_basis_k_init});
        case WoodburyMode::EIG:
            [[fallthrough]];
        case WoodburyMode::VAR: {
            const int start_guess = std::clamp(n / 20, ctx.woodbury_basis_k_init, ctx.woodbury_basis_k_max > 0 ? ctx.woodbury_basis_k_max : n - 1);
            return std::min({n - 1, k_budget_ceiling, start_guess});
        }
        case WoodburyMode::Fixed:
            [[fallthrough]];
        default:
            return ctx.woodbury_basis_rank;
    }
}

struct RankEvalResult {
    bool satisfied = false;
    int k_target   = 0;
    int k_extra    = 0;
};

static RankEvalResult evaluate_rank_criterion(
    WoodburyMode mode,
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
        case WoodburyMode::MP: {
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
            break;
        }
        case WoodburyMode::EIG: {
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
            break;
        }
        case WoodburyMode::VAR: {
            bool crossed_target = false;
            res.k_target = k_svd;
            double cum_sum = 0.0;
            double cum_sq  = 0.0;
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
                if (relative_frobenius_error < ctx.woodbury_basis_var_thresh) {
                    res.k_target = i + 1;
                    crossed_target = true;
                    break;
                }
            }
            res.satisfied = crossed_target;
            break;
        }
        case WoodburyMode::Fixed:
        default:
            res.satisfied = true;
            res.k_target  = ctx.woodbury_basis_rank;
            break;
    }
    return res;
}

static int finalize_and_log_woodbury_rank(
    WoodburyMode mode,
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
        case WoodburyMode::MP: {
            const int k_edge = eval_res.k_target;
            const int k_signal = eval_res.k_extra;
            k = std::max(20, std::min(k_svd, k_edge));
            const int band = std::max(0, k_edge - k_signal);
            LOGGER << "MP bulk edge lambda+ = " << lambda_plus
                   << " (n=" << n << ")"
                   << ", eigenvalues above lambda+ = " << k_signal
                   << ", edge band confirmed after " << band << " additional eigenvalue(s)"
                   << " (margin=" << (ctx.woodbury_basis_edge_margin * 100.0) << "%, confirm="
                   << ctx.woodbury_basis_edge_confirm << " consecutive)"
                   << ", using k = " << k << std::endl;
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
        case WoodburyMode::EIG: {
            const int k_EIGMASS = eval_res.k_target;
            k = std::max(20, std::min(k_svd, k_EIGMASS + ctx.woodbury_basis_EIG_k_buffer));
            double cumulative = 0.0;
            for (int i = 0; i < k; ++i) cumulative += eval_full[i];
            const double rho = cumulative / trace_K_full;
            if (eval_res.satisfied) {
                LOGGER << "EIG-k: trace(K)=" << trace_K_full
                       << ", raw " << ctx.woodbury_basis_eigen_mass * 100 << "% mass crossing at k=" << k_EIGMASS
                       << ", using k=" << k << " (+" << ctx.woodbury_basis_EIG_k_buffer << " eigenvalue buffer)"
                       << ", captured mass rho=" << rho << std::endl;
            } else {
                LOGGER << "EIG-k: trace(K)=" << trace_K_full
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
        case WoodburyMode::VAR: {
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
            LOGGER << "VARIANCE: trace(K)=" << trace_K_full
                   << ", target relative Frobenius tail error < " << ctx.woodbury_basis_var_thresh
                   << " crossed at k=" << k_VAR << ", using k=" << k
                   << " (tail_d_var=" << tail_var
                   << ", tail non-isotropic energy=" << tail_nonisotropic_energy
                   << ", relative Frobenius error=" << relative_frobenius_error << ")" << std::endl;
            if (!eval_res.satisfied && k_svd >= n - 1)
                LOGGER.e(0, "Woodbury VARIANCE: target relative Frobenius tail error not reached even at k=n-1=" + std::to_string(n - 1)
                         + " (relative Frobenius error=" + std::to_string(relative_frobenius_error) + "). Refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied && k_svd >= k_svd_budget_ceiling && !k_max_is_hard_ceiling)
                LOGGER.e(0, "Woodbury VARIANCE: target relative Frobenius tail error not reached within memory budget ceiling k=" + std::to_string(k_svd)
                         + " (relative Frobenius error=" + std::to_string(relative_frobenius_error) + "). Raise --reml-woodbury-basis-mem-budget; refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied && k_max_is_hard_ceiling)
                LOGGER.e(0, "Woodbury VARIANCE: target relative Frobenius tail error not reached within k_max=" + std::to_string(k_cap)
                         + " (relative Frobenius error=" + std::to_string(relative_frobenius_error) + "). Raise --reml-woodbury-basis-range's k_max; refusing to proceed with an unresolved basis.");
            else if (!eval_res.satisfied)
                LOGGER.e(0, "Woodbury VARIANCE: target relative Frobenius tail error not reached within k=" + std::to_string(k_svd)
                         + " (relative Frobenius error=" + std::to_string(relative_frobenius_error) + "). Refusing to proceed with an unresolved basis.");
            break;
        }
        case WoodburyMode::Fixed:
        default:
            k = ctx.woodbury_basis_rank;
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
    if (ctx.reml_mtd == 1)
        LOGGER.e(0, "--reml-woodbury-basis is incompatible with Fisher-scoring REML.");
    if ((int)ctx.r_indx.size() != 2)
        LOGGER.e(0, "--reml-woodbury-basis supports only single-GRM models.");
    if (ctx.A[ctx.r_indx[0]].size() == 0 && ctx.grm_chunked_budget <= 0.0)
        LOGGER.e(0, "--reml-woodbury: GRM component is identity; cannot compute basis.");
    if (ctx.grm_chunked_budget > 0.0 && !ctx.grm_tile_reader)
        LOGGER.e(0, "--reml-woodbury: --grm-chunked-budget is set but ctx.grm_tile_reader is empty "
                    "— the GRM component wasn't actually loaded either way.");

    const WoodburyMode mode = ctx.woodbury_mode();
    const int  n = ctx.n;
    const bool grm_chunked = ctx.grm_chunked_budget > 0.0;

    int k_svd_budget_ceiling = n - 1;
    if (grm_chunked && ctx.svd_mem_budget_gb > 0.0) {
        const double budget_bytes = ctx.svd_mem_budget_gb * 1e9;
        const int max_k_ext = static_cast<int>(budget_bytes / (5.0 * n * 8.0));
        k_svd_budget_ceiling = std::min(k_svd_budget_ceiling, std::max(20, max_k_ext - 200));
        LOGGER << "--reml-woodbury-basis-mem-budget=" << ctx.svd_mem_budget_gb
               << "GB -> k_svd capped at " << k_svd_budget_ceiling << std::endl;
    }

    int grm_chunk_rows = 0;
    if (grm_chunked) {
        grm_chunk_rows = setup_chunked_grm_stream(ctx, "--reml-woodbury-basis");
        // Y is not reserved by setup_chunked_grm_stream -- report its worst-case
        // size (at the current k_svd ceiling) here so it's visible without being enforced.
        const double y_worst_case_gb = 8.0 * n * static_cast<double>(k_svd_budget_ceiling) / 1e9;
        LOGGER << "--reml-woodbury-basis: at the current k_svd ceiling of " << k_svd_budget_ceiling
               << ", the basis matrix (Y) may need up to ~" << y_worst_case_gb
               << "GB, not covered by --grm-chunked-budget." << std::endl;
    }


    const bool k_max_is_hard_ceiling = (ctx.woodbury_basis_k_max > 0);
    const int k_svd_cap = woodbury_rank_cap(ctx, k_svd_budget_ceiling);
    int k_svd = woodbury_initial_k_svd(ctx, mode, k_svd_budget_ceiling);

    if (mode == WoodburyMode::Fixed) {
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

    // K_dbl is upper-triangle-only storage (row <= col valid; see
    // grm_binary_io.hpp upper_only load path). No valid data below the
    // diagonal.
    const Eigen::MatrixXd& K_dbl = ctx.A[ctx.r_indx[0]];
    const double trace_K_full = grm_chunked
        ? gcta_chunked::chunked_diagonal(ctx.grm_tile_reader, n, grm_chunk_rows).sum()
        : K_dbl.diagonal().sum();

    double trace_K2 = 0.0;
    if (grm_chunked) {
        trace_K2 = gcta_chunked::chunked_trace_K_squared(ctx.grm_tile_reader, n, grm_chunk_rows);
    } else {
        // Previously read K_dbl.col(j).tail(n-j-1) -- rows j+1..n-1 of
        // column j, i.e. the LOWER triangle. That was a correctness bug
        // waiting to happen under upper-only storage (it happened to be
        // contiguous and correct only because the now-removed mirror pass
        // kept the lower triangle populated). Fixed the same way as
        // init_varcomp's trK2: col(j).head(j) reads rows 0..j-1 of column
        // j -- the UPPER triangle -- and is equally contiguous, so this is
        // a pure win, not a tradeoff.
        double diag_sq = K_dbl.diagonal().squaredNorm();
        double off_sq  = 0.0;
        #pragma omp parallel for reduction(+:off_sq) schedule(static)
        for (int j = 0; j < n; ++j)
            off_sq += K_dbl.col(j).head(j).squaredNorm();
        trace_K2 = diag_sq + 2.0 * off_sq;
    }

    double lambda_plus = 0.0;
    if (mode == WoodburyMode::MP) {
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
    const double target_mass = (mode == WoodburyMode::EIG) ? (ctx.woodbury_basis_eigen_mass * trace_K_full) : 0.0;

    auto apply = [&](const auto& X) -> Eigen::MatrixXd {
        if (grm_chunked)
            return gcta_chunked::chunked_symmetric_matvec(ctx.grm_tile_reader, n, grm_chunk_rows, X);
        // Return-by-value forces full evaluation into a fresh Eigen::MatrixXd
        // temporary, not an in-place block write, so this is not subject to
        // the selfadjointView-on-block-target scalar-fallback issue.
        return K_dbl.selfadjointView<Eigen::Upper>() * X;
    };

    Eigen::VectorXd eval_full;
    Eigen::MatrixXd evec_full;
    RankEvalResult eval_res;

    // Rank-expansion rounds lock the previous round's converged Ritz vectors and
    // sketch only the new directions (gcta_eigh::expand_symmetric_eigh), so each
    // column is touched once across rounds; that is what pays for q > 3.
    const int q_pow = std::max(1, ctx.svd_power_iter);
    // All k_ext Ritz pairs of a round are kept, not just the top k_svd: the oversample
    // vectors are already converged-ish and locking them means no column is ever
    // recomputed. The rank criterion only sees the leading k_svd values (eval_full).
    Eigen::MatrixXd V0;        // previous round's Ritz vectors (n x k_ext_prev)
    Eigen::VectorXd th0;       // matching Ritz values
    Eigen::VectorXd eval_all;  // this round's k_ext Ritz values (non-Nystrom)
    double prev_mass = 0.0;    // sum of the previous round's Ritz values
    int    prev_k    = 0;
    bool   round_expanded = false;
    if (!ctx.svd_nystrom && mode != WoodburyMode::Fixed)
        LOGGER << "Woodbury: rSVD power iterations = " << q_pow
               << "; rank expansions reuse converged Ritz vectors." << std::endl;

    for (;;) {
        const int oversample = gcta_eigh::recommended_oversample(k_svd);
        const int k_ext = std::min(k_svd + oversample, n - 1);

        if (ctx.svd_nystrom) {
            try {
                gcta_eigh::EighResult res = gcta_eigh::nystrom_symmetric_eigh(
                    apply, n, k_svd,
                    gcta_eigh::recommended_oversample(k_svd));
                if (mode == WoodburyMode::EIG) {
                    LOGGER << "Woodbury EIG-k: refining Nystrom basis with one Rayleigh-Ritz projection ..."
                           << std::endl;
                    res = gcta_eigh::rayleigh_ritz_refine(apply, std::move(res.eigenvectors), k_svd);
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
                    gcta_eigh::EighResult res =
                        gcta_eigh::expand_symmetric_eigh(apply, V0, th0, k_ext, k_ext, q_pow);
                    eval_all  = std::move(res.eigenvalues);
                    evec_full = std::move(res.eigenvectors);
                    round_expanded = true;
                } catch (const std::exception& e) {
                    LOGGER.w(0, std::string("Woodbury: locked expansion failed (") + e.what() +
                                "); recomputing this round from a fresh sketch.");
                }
            }
            if (!round_expanded) {
                auto [omega, Y] = gcta_eigh::build_randomized_sketch(apply, n, k_ext, nullptr);
                omega.resize(0, 0);
                try {
                    gcta_eigh::EighResult res =
                        gcta_eigh::power_iterate_and_project(apply, std::move(Y), k_ext, q_pow);
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

        if (mode != WoodburyMode::Fixed) {
            // Per-round diagnostics. Captured mass at any fixed rank must be
            // non-decreasing across locked-expansion rounds (Cauchy interlacing);
            // a regression means numerical trouble, not a statistical fluke.
            const double mass_now = eval_full.sum();
            LOGGER << "Woodbury " << woodbury_mode_name(mode) << ": k_svd=" << k_svd
                   << ", captured mass rho=" << (trace_K_full > 0.0 ? mass_now / trace_K_full : 0.0)
                   << (round_expanded ? " (locked expansion)" : "") << std::endl;
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

        if (mode == WoodburyMode::Fixed) break;

        eval_res = evaluate_rank_criterion(mode, eval_full, k_svd, lambda_plus, target_mass, trace_K_full, trace_K2, ctx);

        if (eval_res.satisfied || k_svd >= k_svd_cap || k_svd >= k_svd_budget_ceiling) break;

        int k_svd_next = std::min({k_svd * 2, n - 1, k_svd_cap});
        if (mode == WoodburyMode::EIG && ctx.woodbury_basis_eigen_adaptive) {
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
            LOGGER.i(0, "Woodbury EIG-k: adaptive jump from k=" + std::to_string(k_svd) + " to k=" + std::to_string(k_svd_jump)
                        + " (bounded by " + std::to_string(k_svd_next) + ") to reach target mass (" + std::to_string(ctx.woodbury_basis_eigen_mass * 100.0) + "%)."
                        + "\n(Currently captured mass: " + std::to_string(captured) + ")");
        }
        const char* warm_status = ctx.svd_nystrom ? " (Nystrom: full recompute)"
                                                  : " (locked expansion: reusing converged Ritz vectors)";
        LOGGER << "Woodbury " << woodbury_mode_name(mode)
               << ": signal not resolved within k=" << k_svd
               << "; expanding budget to k=" << k_svd_next
               << warm_status << " ..." << std::endl;
        if (!ctx.svd_nystrom) {
            V0  = std::move(evec_full);   // evec_full is reassigned next round
            th0 = std::move(eval_all);
        }
        k_svd = k_svd_next;
    }
    V0.resize(0, 0);                      // release before the final basis copy
    th0.resize(0);
    eval_all.resize(0);

    int k = finalize_and_log_woodbury_rank(mode, ctx, eval_full, k_svd, k_svd_budget_ceiling, lambda_plus, trace_K_full, trace_K2, eval_res);

    Eigen::VectorXd eval = eval_full.head(k);
    Eigen::MatrixXd evec = evec_full.leftCols(k);
    eval_full.resize(0);
    evec_full.resize(0, 0);

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

    ctx.dk            = eval;
    ctx.Uk            = evec;
    ctx.woodbury_basis_rank_ = k;

    if (!ctx.woodbury_basis_posthoc_correction)
        ctx.A[ctx.r_indx[0]].resize(0, 0);
    ctx.Vi_use_woodbury_basis = true;

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
        if (ctx.reml_mtd == 1)
            LOGGER.e(0, "--reml-woodbury-reuse is incompatible with Fisher-scoring REML.");
        if ((int)ctx.r_indx.size() != 2)
            LOGGER.e(0, "--reml-woodbury-reuse supports only single-GRM models.");
        if (ctx.Uk.rows() != ctx.n || ctx.Uk.cols() != ctx.woodbury_basis_rank_
                || ctx.dk.size() != ctx.woodbury_basis_rank_)
            LOGGER.e(0, "--reml-woodbury-reuse: preloaded basis has inconsistent dimensions.");
        LOGGER << "Woodbury basis reused: k=" << ctx.woodbury_basis_rank_
               << ", lambda_tail=" << ctx.lambda_tail << std::endl;
    } else if (ctx.woodbury_basis_rank != 0) {
        LOGGER.ts("woodbury");
        compute_woodbury_basis(ctx);
        float duration = LOGGER.tp("main");
        LOGGER.i(0, "Woodbury basis computation took " + std::to_string(duration) + " seconds.");
    } else if (ctx.reml_trace_hutchpp && ctx.grm_chunked_budget > 0.0) {
        // Woodbury takes precedence when both are requested.
        if ((int)ctx.r_indx.size() != 2)
            LOGGER.e(0, "--grm-chunked-budget with --reml-trace-hutchpp supports only single-GRM models.");
        ctx.grm_chunk_rows = setup_chunked_grm_stream(ctx, "--reml-trace-hutchpp");
    } else if (ctx.grm_chunked_budget > 0.0) {
        // Plain exact (non-Hutch++, non-Woodbury) chunked GRM. Without this
        // branch ctx.grm_chunk_rows is never set for this path, so every
        // downstream "ctx.grm_chunk_rows > 0" gate (ai_reml, em_reml,
        // calcu_tr_PA, applyP_mat, init_varcomp's HE warm-start, ...) falls
        // through to the "ctx.A[...].size() == 0" branch instead -- which
        // silently treats the (empty, because it's streamed, not loaded)
        // GRM component as an identity/residual term for the entire run.
        if ((int)ctx.r_indx.size() != 2)
            LOGGER.e(0, "--grm-chunked-budget supports only single-GRM models.");
        ctx.grm_chunk_rows = setup_chunked_grm_stream(ctx, "--grm-chunked-budget");
    }

    RemlMat Vi_X_out(ctx.n, ctx.X_c), Xt_Vi_X_i_out(ctx.X_c, ctx.X_c);
    RemlMat Hi(ctx.r_indx.size(), ctx.r_indx.size());
    RemlVec Py(ctx.n);
    RemlVec varcmp;
    ctx.he_warm_start_applied = init_varcomp(ctx, priors_var, priors, varcmp);

    const double lgL = reml_iteration(ctx, Vi_X_out, Xt_Vi_X_i_out, Hi, Py, varcmp, priors_flag, no_constrain);
    ctx.logL = lgL;
    ctx.has_logL = true;

    // Compute fixed effects: b = (X'V^{-1}X)^{-1} X'V^{-1}y
    ctx.b = Xt_Vi_X_i_out * (Vi_X_out.transpose() * ctx.y);

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
        // WoodburyMLMACache stores Uk_f as k×n (transposed) for GEMM efficiency.
        // ctx.Uk is n×k — transpose before storing.
        rs.wb.Uk_f.resize(ctx.Uk.cols(), ctx.Uk.rows());
        rs.wb.Uk_f.noalias() = ctx.Uk.transpose().cast<float>();  // k×n
        ctx.Uk.resize(0, 0); //free double buffer immediately -- nothing below reads it again,
                              //same rationale as the ck/dk frees just below (was previously held
                              //alive through the ck/dk conversions for no reason; n×k doubles,
                              //typically the largest buffer in this branch)

        rs.wb.ck_f.resize(ctx.ck.size());
        rs.wb.ck_f.noalias() = ctx.ck.cast<float>(); //convert double to float
        ctx.ck.resize(0); //free double buffer immediately (both are temporarily alive for 1.5x the size of Uk_f)

        rs.wb.sqrt_ck_f.resize(rs.wb.ck_f.size());
        rs.wb.sqrt_ck_f.noalias() = rs.wb.ck_f.cwiseSqrt();

        rs.wb.sigma2_eff_f = static_cast<float>(ctx.sigma2_eff);

        rs.dk_f.resize(ctx.dk.size());
        rs.dk_f.noalias() = ctx.dk.cast<float>();
        ctx.dk.resize(0);

        rs.lambda_tail_f = static_cast<float>(ctx.lambda_tail);
    } else if (ctx.Vi_use_llt) {
        // Vi_L holds the upper Cholesky factor U of V (V = U^T U, from dpotrf).
        // Store it as float — the streaming code uses STRSV/STRSM directly,
        // avoiding dpotri (O(n³/3)) and a second Cholesky of V^{-1} (O(n³/3)).
        rs.is_llt = true;
        rs.Vi_L_f.resize(ctx.Vi_L.rows(), ctx.Vi_L.cols());
        rs.Vi_L_f.noalias() = ctx.Vi_L.cast<float>();   // n×n float, ~n²/2 significant bytes
        ctx.Vi_L.resize(0, 0);
    } else {
        // reml_diagV_adj fallback: ctx.Vi is already the dense explicit
        // V^{-1} (computed once, O(n³), inside calcu_Vi). Factor it here —
        // once — so the streaming code always applies V^{-1} via a
        // triangular product (TRMM/STRMM) instead of holding a dense matrix
        // and re-deriving a usable factor from it at association time.
        //
        // Factorize in place via gcta_dpotrf (as calcu_Vi already does)
        // rather than Eigen::LLT(ctx.Vi): Eigen::LLT's constructor copies
        // its input into internal storage before factorizing, so it can't
        // operate in place on a caller-owned buffer -- that doubles peak
        // RSS transiently (ctx.Vi + LLT's internal copy, both n×n doubles,
        // alive simultaneously) for no reason, since ctx.Vi is read via
        // selfadjointView<Upper> everywhere else in this file (only the
        // upper triangle is guaranteed valid, matching dpotrf('U')'s contract)
        // and is freed immediately below regardless.
        gcta_blas_int blas_n_bs = static_cast<gcta_blas_int>(ctx.n);
        if (gcta_dpotrf(blas_n_bs, ctx.Vi.data(), blas_n_bs, 'U') != 0)
            LOGGER.e(0, "Vi is not positive definite when building REML state.");
        rs.is_llt = false;
        rs.Vi_L_f.resize(ctx.n, ctx.n);
        rs.Vi_L_f.noalias() = ctx.Vi.cast<float>();   // upper triangle valid; lower is dpotrf
                                                       // leftover and never read downstream --
                                                       // same convention as the Vi_use_llt branch above
        ctx.Vi.resize(0, 0);
    }

    // Clear the remaining large, now-redundant double buffers after the float
    // copies are created. The downstream MLMA stream uses only the float state.
    ctx.A.clear();
    ctx.grm_N.resize(0, 0);
    ctx.Vi_X.resize(0, 0);
    ctx.Xt_Vi_X_i.resize(0, 0);
    ctx.Uk_Vi_X.resize(0, 0);
    ctx.UkTX.resize(0, 0);
    ctx.UkTy.resize(0, 0);
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
