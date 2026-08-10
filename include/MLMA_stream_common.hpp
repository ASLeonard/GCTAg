#pragma once

#include "Geno.h"
#include "Marker.h"
#include "Logger.h"
#include "RemlState.hpp"
#include "RemlCtx.hpp"
#include "cpu.h"
#include "mlma_woodbury.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <type_traits>
#include <span>
#include <string>
#include <vector>

template <typename T, typename IndexT,
          typename = std::enable_if_t<std::is_integral_v<IndexT>>>
inline std::vector<T> compact_sample_vector(const std::vector<T>& values,
                                            const std::vector<IndexT>& keep)
{
    std::vector<T> compacted;
    compacted.reserve(keep.size());
    for (IndexT index : keep) compacted.push_back(values[static_cast<size_t>(index)]);
    return compacted;
}

template <typename IndexT,
          typename = std::enable_if_t<std::is_integral_v<IndexT>>>
inline Eigen::VectorXd compact_sample_vector(const Eigen::VectorXd& values,
                                             const std::vector<IndexT>& keep)
{
    Eigen::VectorXd compacted(keep.size());
    for (size_t i = 0; i < keep.size(); ++i)
        compacted[static_cast<Eigen::Index>(i)] = values[static_cast<Eigen::Index>(keep[i])];
    return compacted;
}

template <typename IndexT,
          typename = std::enable_if_t<std::is_integral_v<IndexT>>>
inline Eigen::MatrixXd compact_sample_rows(const Eigen::MatrixXd& values,
                                           const std::vector<IndexT>& keep)
{
    Eigen::MatrixXd compacted(keep.size(), values.cols());
    for (size_t i = 0; i < keep.size(); ++i)
        compacted.row(static_cast<Eigen::Index>(i)) = values.row(static_cast<Eigen::Index>(keep[i]));
    return compacted;
}

// Resolve the streaming block width (SNPs per tile) from a memory budget.
// Two buffers dominate, both O(n * block):
//   - GenoBufItem::geno : n doubles/SNP (8 bytes) — the raw genotype call
//                         buffer Geno::getGenoDouble hands back per SNP,
//                         resident for every SNP currently in flight.
//   - X_block           : n floats/SNP  (4 bytes) — the centered, weighted
//                         float design block actually consumed by BLAS.
// Per-SNP scalars (Xt_Vi_y, xvx_diag, af_v, valid_v) are O(1) per column
// and negligible next to the O(n) terms above.
// budget_gb <= 0 preserves the previous fixed BLOCK=10000 behavior exactly
inline int resolve_mlma_block_size(int n, double budget_gb)
{
    if (budget_gb <= 0.0 || n <= 0)
        return 10000; // Somewhat arbitrary, but balance of peak RSS and throughout. Increasing may help STRSM efficiency but increases memory.

    constexpr double bytes_per_snp_per_sample = 8.0 /* GenoBufItem::geno */
                                               + 4.0 /* X_block */;
    constexpr int    min_block = 256;    // floor: keep BLAS3 tiles meaningful
    constexpr int    max_block = 65536;  // ceiling: cap per-column scalar creep

    const double budget_bytes  = budget_gb * 1e9;
    const double per_snp_bytes = static_cast<double>(n) * bytes_per_snp_per_sample;
    const int    block         = static_cast<int>(budget_bytes / per_snp_bytes);
    return std::clamp(block, min_block, max_block);
}

template <typename XVXDiagFn>
inline void run_mlma_stream_association_impl(const Eigen::VectorXf& Vi_y,
                                             XVXDiagFn&& compute_xvx_diag,
                                             const Eigen::VectorXf& w_sqrt,
                                             Geno* geno,
                                             Marker* marker,
                                             int n,
                                             int block_size,
                                             bool log_pval,
                                             std::ofstream& ofile)
{
    const uint32_t total_m = marker->count_extract();
    LOGGER << "\nStreaming association tests for " << total_m << " SNPs..." << std::endl;

    const int BLOCK = block_size;
    Eigen::MatrixXf X_block(n, BLOCK);
    Eigen::VectorXf Xt_Vi_y(BLOCK);
    Eigen::VectorXf xvx_diag(BLOCK);

    int  snp_done = 0;
    int  last_pct = -1;

    std::vector<GenoBufItem> gbuf_items(BLOCK);
    std::vector<uint8_t>     valid_v(BLOCK, 0);
    std::vector<float>       af_v(BLOCK, 0.0f);

    std::string io_buf;
    io_buf.reserve(4 << 20);

    auto flush_io = [&]() {
        if (!io_buf.empty()) {
            ofile.write(io_buf.data(), static_cast<std::streamsize>(io_buf.size()));
            io_buf.clear();
        }
    };

    auto callback = [&](uintptr_t* buf, std::span<const uint32_t> exIdx) {
        const int bs = static_cast<int>(exIdx.size());

        for (int i = 0; i < bs; ++i) {
            valid_v[i] = 0;
            af_v[i]    = 0.0f;
            GenoBufItem& item = gbuf_items[i];
            item.extractedMarkerIndex = exIdx[i];
            geno->getGenoDouble(buf, i, &item);
            if (!item.valid) { X_block.col(i).setZero(); continue; }
            if (static_cast<int>(item.geno.size()) != n) {
                LOGGER.e(0, "internal error: SNP " + std::to_string(exIdx[i])
                            + " returned geno.size=" + std::to_string(item.geno.size())
                            + " but expected " + std::to_string(n) + ".");
            }
            valid_v[i] = 1;
            af_v[i]    = static_cast<float>(item.additive_af);
            X_block.col(i) = (Eigen::Map<const Eigen::VectorXd>(item.geno.data(), n)
                                .cast<float>().array() * w_sqrt.array()).matrix();
        }

        Xt_Vi_y.head(bs).noalias() = X_block.leftCols(bs).transpose() * Vi_y;
        compute_xvx_diag(X_block, bs, xvx_diag);

        for (int i = 0; i < bs; ++i) {
            const uint32_t raw      = marker->getRawIndex(exIdx[i]);
            const std::string& chr  = marker->getRawChr(raw);
            const std::string& name = marker->getRawName(raw);
            const unsigned     bp   = static_cast<unsigned>(marker->getRawBp(raw));
            const std::string& a1   = marker->getRawA1(raw);
            const std::string& a2   = marker->getRawA2(raw);

            float beta_val = 0.0f, se_val = 0.0f;
            double pval_val = 0.0;
            const bool stat_ok = valid_v[i] &&
                mlma_snp_stat(Xt_Vi_y[i], xvx_diag[i], log_pval,
                              beta_val, se_val, pval_val);

            std::format_to(std::back_inserter(io_buf),
                "{}\t{}\t{}\t{}\t{}",
                chr, name, bp, a1, a2);

            if (!stat_ok) {
                io_buf += "\tNA\tNA\tNA\tNA\n";
            } else {
                std::format_to(std::back_inserter(io_buf),
                    "\t{:.6g}\t{:.6g}\t{:.6g}\t{:.6g}\n",
                    static_cast<double>(af_v[i]),
                    static_cast<double>(beta_val),
                    static_cast<double>(se_val),
                    pval_val);
            }
        }

        snp_done += bs;
        const int cur_pct = (total_m > 0)
            ? static_cast<int>((uint64_t)snp_done * 100 / total_m) : 100;
        if (cur_pct != last_pct) {
            LOGGER.p(0, std::to_string(snp_done) + " / " + std::to_string(total_m)
                        + " SNPs (" + std::to_string(cur_pct) + "%)");
            last_pct = cur_pct;
        }
    };

    const std::vector<uint32_t>& extractIndex = marker->get_extract_index();
    geno->loopDouble(extractIndex, BLOCK,
                     /*bMakeGeno*/   true,
                     /*bGenoCenter*/ true,
                     /*bGenoStd*/    false,
                     /*bMakeMiss*/   true,
                     {callback});

    flush_io();
}

inline void run_mlma_stream_association(RemlState& state,
                                        const Eigen::VectorXf& y_adj,
                                        const Eigen::VectorXf& w_sqrt,
                                        Geno* geno,
                                        Marker* marker,
                                        int n,
                                        bool log_pval,
                                        std::ofstream& ofile,
                                        double block_mem_budget_gb = 0.0)
{
    Eigen::VectorXf Vi_y(n);

    const bool use_wb  = state.is_woodbury;
    const bool use_llt = state.is_llt;  // selects TRSM (factor = L(V)) vs TRMM (factor = Li(Vi))

    if (use_wb) {
        LOGGER << "MLMA streaming: applying V^{-1} via Woodbury low-rank update "
            "(V^{-1} = D^{-1} - D^{-1} U_k C U_k^T D^{-1})." << std::endl;
        Vi_y = woodbury_apply_Vi_f(state.wb, y_adj);
    } else if (use_llt) {
        // V^{-1} y = L^{-T}(L^{-1} y) via two float triangular solves
        // against L, the Cholesky factor of V.
        LOGGER << "MLMA streaming: applying V^{-1} via triangular solve against L, "
            "where V = L L^T (STRSM/STRSV — forward/back substitution)." << std::endl;
        Vi_y = y_adj;
        cblas_strsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    n, state.Vi_L_f.data(), n, Vi_y.data(), 1);
        cblas_strsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                    n, state.Vi_L_f.data(), n, Vi_y.data(), 1);
    } else {
        // V^{-1} y = Li (Li^T y) via two float triangular products against
        // Li, the Cholesky factor of V^{-1} itself. Li was already computed
        // once at REML-exit/save time (see writeRemlStateFromCtx /
        // build_reml_state) — no factorization happens here, ever.
        LOGGER << "MLMA streaming: applying V^{-1} via triangular product against L_i, "
            "where V^{-1} = L_i L_i^T (STRMM — no linear solve)." << std::endl;
        Vi_y.noalias() = state.Vi_L_f.triangularView<Eigen::Lower>().transpose() * y_adj;
        Vi_y = state.Vi_L_f.triangularView<Eigen::Lower>() * Vi_y;
    }

    Vi_y.array() *= w_sqrt.array();
    const WoodburyMLMACache& wb = state.wb;

    auto compute_xvx_diag = [&](Eigen::MatrixXf& X_block, int bs, Eigen::VectorXf& xvx_diag) {
        if (use_wb) {
            woodbury_xvx_diag_block(wb, X_block, bs, xvx_diag);
        } else if (use_llt) {
            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                        n, bs, 1.0f,
                        state.Vi_L_f.data(), n,
                        X_block.data(), n);
            xvx_diag.head(bs) = X_block.leftCols(bs).colwise().squaredNorm();
        } else {
            cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                        n, bs, 1.0f,
                        state.Vi_L_f.data(), n,
                        X_block.data(), n);
            xvx_diag.head(bs) = X_block.leftCols(bs).colwise().squaredNorm();
        }
    };

    run_mlma_stream_association_impl(Vi_y, compute_xvx_diag, w_sqrt,
                                     geno, marker, n,
                                     resolve_mlma_block_size(n, block_mem_budget_gb),
                                     log_pval, ofile);
}

inline void run_mlma_stream_association(RemlCtx& ctx,
                                        const Eigen::VectorXf& y_adj,
                                        const Eigen::VectorXf& w_sqrt,
                                        Geno* geno,
                                        Marker* marker,
                                        int n,
                                        bool log_pval,
                                        std::ofstream& ofile,
                                        double block_mem_budget_gb = 0.0)
{
    Eigen::VectorXf Vi_y(n);

    if (ctx.Vi_use_woodbury_basis) {
        const Eigen::VectorXd y_d = y_adj.cast<double>();
        Eigen::VectorXd UkTy = ctx.Uk.transpose() * y_d;
        UkTy.array() *= ctx.ck.array();
        Vi_y = ((y_d - ctx.Uk * UkTy) / ctx.sigma2_eff).cast<float>();
    } else if (ctx.Vi_use_llt) {
        Eigen::VectorXd Vi_y_d = y_adj.cast<double>();
        ctx.Vi_L.triangularView<Eigen::Lower>().solveInPlace(Vi_y_d);
        ctx.Vi_L.triangularView<Eigen::Lower>().adjoint().solveInPlace(Vi_y_d);
        Vi_y = Vi_y_d.cast<float>();
    } else {
        gcta_blas_int blas_n = static_cast<gcta_blas_int>(n);
        if (gcta_dpotrf(blas_n, ctx.Vi.data(), blas_n) != 0)
            LOGGER.e(0, "inline REML V^{-1} is not positive definite for MLMA streaming.");
        const Eigen::VectorXd y_d = y_adj.cast<double>();
        const Eigen::VectorXd tmp = ctx.Vi.transpose().triangularView<Eigen::Upper>() * y_d;
        Vi_y = (ctx.Vi.triangularView<Eigen::Lower>() * tmp).cast<float>();
    }

    Vi_y.array() *= w_sqrt.array();

    auto compute_xvx_diag = [&](Eigen::MatrixXf& X_block, int bs, Eigen::VectorXf& xvx_diag) {
        if (ctx.Vi_use_woodbury_basis) {
            const Eigen::MatrixXd X_d = X_block.leftCols(bs).cast<double>();
            const Eigen::MatrixXd UkX = ctx.Uk.transpose() * X_d;
            const Eigen::VectorXd x_norm = X_d.colwise().squaredNorm().transpose();
            const Eigen::VectorXd correction =
                (UkX.array().square().colwise() * ctx.ck.array()).colwise().sum().transpose();
            xvx_diag.head(bs) = ((x_norm.array() - correction.array()) / ctx.sigma2_eff).cast<float>();
        } else if (ctx.Vi_use_llt) {
            Eigen::MatrixXd X_d = X_block.leftCols(bs).cast<double>();
            ctx.Vi_L.triangularView<Eigen::Lower>().solveInPlace(X_d);
            xvx_diag.head(bs) = X_d.colwise().squaredNorm().transpose().cast<float>();
        } else {
            const Eigen::MatrixXd X_d = X_block.leftCols(bs).cast<double>();
            const Eigen::MatrixXd proj = ctx.Vi.transpose().triangularView<Eigen::Upper>() * X_d;
            xvx_diag.head(bs) = proj.colwise().squaredNorm().transpose().cast<float>();
        }
    };

    run_mlma_stream_association_impl(Vi_y, compute_xvx_diag, w_sqrt,
                                     geno, marker, n,
                                     resolve_mlma_block_size(n, block_mem_budget_gb),
                                     log_pval, ofile);
}