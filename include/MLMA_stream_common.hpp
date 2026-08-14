#pragma once

#include "Geno.h"
#include "Marker.h"
#include "Logger.h"
#include "RemlEngine.hpp"
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
// The dominant O(n * block) allocation is now a single buffer:
//   - X_block : n floats/SNP (4 bytes) — the centered, weighted float design
//               block consumed by BLAS.  Written directly by getGenoFloat
//               (BED: GenoarrLookup256x4bx4; PGEN/BGEN: cast from double).
// The old GenoBufItem::geno double[] intermediate is no longer allocated on
// the MLMA hot path.
// Per-SNP scalars (Xt_Vi_y, xvx_diag, af_v, valid_v) are O(1) per column
// and negligible next to the O(n) term above.
// budget_gb <= 0 preserves the previous fixed BLOCK=10000 behavior exactly
inline int resolve_mlma_block_size(int n, double budget_gb)
{
    if (budget_gb <= 0.0 || n <= 0)
        return 10000; // Somewhat arbitrary, but balance of peak RSS and throughout. Increasing may help STRSM efficiency but increases memory.

    constexpr double bytes_per_snp_per_sample =
        static_cast<double>(sizeof(typename decltype(GenoBufItem::geno)::value_type)) /* GenoBufItem::geno */
        + static_cast<double>(sizeof(float)) /* X_block */;
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

    std::vector<uint8_t> valid_v(BLOCK, 0);
    std::vector<float>   af_v(BLOCK, 0.0f);
    std::vector<float>   additive_af_v(BLOCK, 0.0f);

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

        // Decode genotypes directly into X_block columns as float.
        // For BED format this uses GenoarrLookup256x4bx4 (SIMD, 4 KB L1 table).
        // For PGEN/BGEN it decodes via the double path then casts.
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < bs; ++i) {
            bool valid = false;
            float af = 0.0f, add_af = 0.0f;
            geno->getGenoFloat(buf, i, X_block.col(i).data(),
                               af, add_af, valid, exIdx[i]);
            if (valid) {
                X_block.col(i).array() *= w_sqrt.array();
            } else {
                X_block.col(i).setZero();
            }
            valid_v[i]       = static_cast<uint8_t>(valid);
            af_v[i]          = af;
            additive_af_v[i] = add_af;
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

            if (!stat_ok) {
                std::format_to(std::back_inserter(io_buf),
                    "{}\t{}\t{}\t{}\t{}\tNA\tNA\tNA\tNA\n",
                    chr, name, bp, a1, a2);
                std::format_to(std::back_inserter(io_buf),
                    "{}\t{}\t{}\t{}\t{}\tNA\tNA\tNA\tNA\n",
                    chr, name, bp, a1, a2);
            } else {
                std::format_to(std::back_inserter(io_buf),
                    "{}\t{}\t{}\t{}\t{}\t{:.6g}\t{:.6g}\t{:.6g}\t{:.6g}\n",
                    chr, name, bp, a1, a2,
                    static_cast<double>(additive_af_v[i]),
                    static_cast<double>(beta_val),
                    static_cast<double>(se_val),
                    pval_val);
            }
        }

        if (io_buf.size() >= (4 << 20)) {
            flush_io();
        }

        if (io_buf.size() >= (4 << 20)) {
            flush_io();
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

// The "--save-reml and --load-reml" path
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
    const int BLOCK = resolve_mlma_block_size(n, block_mem_budget_gb);
    Eigen::MatrixXf UkX_scratch = use_wb
        ? Eigen::MatrixXf(wb.Uk_f.rows(), BLOCK)
        : Eigen::MatrixXf();

    auto compute_xvx_diag = [&](Eigen::MatrixXf& X_block, int bs, Eigen::VectorXf& xvx_diag) {
        if (use_wb) {
            woodbury_xvx_diag_block(wb, X_block, bs, xvx_diag, UkX_scratch);
        } else if (use_llt) {
            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                        n, bs, 1.0f,
                        state.Vi_L_f.data(), n,
                        X_block.data(), n);
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < bs; ++j) {
                xvx_diag[j] = X_block.col(j).squaredNorm();
            }
        } else {
            cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                        n, bs, 1.0f,
                        state.Vi_L_f.data(), n,
                        X_block.data(), n);
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < bs; ++j) {
                xvx_diag[j] = X_block.col(j).squaredNorm();
            }
        }
    };

    run_mlma_stream_association_impl(Vi_y, compute_xvx_diag, w_sqrt,
                                     geno, marker, n,
                                     BLOCK,
                                     log_pval, ofile);
}

inline void release_reml_ctx_after_state_build(RemlCtx& ctx) {
    // Once the compact float-only RemlState is built, the large double-precision
    // REML workspace is no longer required for the association step. Releasing it
    // here cuts the RSS peak substantially without changing the downstream
    // arithmetic, because all streaming work already reads from the float state.
    ctx.A.clear();
    ctx.grm_N.resize(0, 0);
    ctx.Vi.resize(0, 0);
    ctx.Vi_L.resize(0, 0);
    ctx.Vi_X.resize(0, 0);
    ctx.Xt_Vi_X_i.resize(0, 0);
    ctx.Uk_Vi_X.resize(0, 0);
    ctx.UkTX.resize(0, 0);
    ctx.UkTy.resize(0, 0);
    ctx.hutchpp_S.resize(0, 0);
    ctx.hutchpp_G.resize(0, 0);
    ctx.P.resize(0, 0);
    ctx.Uk.resize(0, 0);
    ctx.dk.resize(0);
    ctx.ck.resize(0);
    ctx.Vi_use_woodbury_basis = false;
    ctx.Vi_use_llt = false;
}

//The "inline REML" path
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
    RemlState state = reml::build_reml_state(ctx);
    release_reml_ctx_after_state_build(ctx);
    run_mlma_stream_association(state, y_adj, w_sqrt, geno, marker, n, log_pval, ofile,
                                block_mem_budget_gb);
}