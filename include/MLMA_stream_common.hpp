#pragma once

#include "Geno.h"
#include "Marker.h"
#include "Logger.h"
#include "RemlEngine.hpp"
#include "cpu.h"
#include "mem.hpp"
#include "mlma_woodbury.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
// Accounts for all O(BLOCK) streaming buffers held simultaneously per tile:
//   - G_block_i8 (~n bytes, 64B-aligned column stride) vs X_block (4*n bytes)
//   - Backend double-buffered raw 2-bit slots (2 × ceil(max(n, raw_n)/4) ≈ 0.5*raw_n bytes)
//   - Woodbury projection scratch UkX_scratch (4 * wb_k bytes when Woodbury active)
// Per-SNP scalars (Xt_Vi_y, xvx_diag, af_v, valid_v) are O(1) per column.
// budget_gb <= 0 preserves the previous fixed BLOCK=10000 behavior exactly.
inline int resolve_mlma_block_size(int n, double budget_gb, bool use_int8_geno = false,
                                   int raw_n = 0, int wb_k = 0)
{
    int block = 10000; // Somewhat arbitrary, but balance of peak RSS and throughout. Increasing may help STRSM efficiency but increases memory.
    if (budget_gb > 0.0 && n > 0) {

      LOGGER << "MLMA streaming memory budget: " << budget_gb << " GB" << std::endl;
      const int raw_eff = std::max(n, raw_n);
      const double geno_bytes = use_int8_geno
          ? static_cast<double>((static_cast<size_t>(n) + 63) / 64 * 64) /* G_block_i8 */
          : static_cast<double>(sizeof(float)) * static_cast<double>(n); /* X_block */
      const double raw_io_bytes = 0.5 * static_cast<double>(raw_eff);
      const double scratch_bytes = (wb_k > 0)
          ? static_cast<double>(sizeof(float)) * static_cast<double>(wb_k)
          : 0.0;
      constexpr int    min_block = 256;    // floor: keep BLAS3 tiles meaningful
      constexpr int    max_block = 65536;  // ceiling: cap per-column scalar creep

      const double budget_bytes  = budget_gb * 1e9;
      const double per_snp_bytes = geno_bytes + raw_io_bytes + scratch_bytes;
      block   =  std::clamp(static_cast<int>(budget_bytes / per_snp_bytes), min_block, max_block);
    }
    LOGGER << "MLMA streaming block size: " << block << " SNPs per tile" << std::endl;
    return block;
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

// Int8 Woodbury streaming association path: stores raw additive BED genotypes
// in int8_t (1 byte/sample/SNP instead of 4 bytes/sample/SNP) and applies
// mean-centering and sparse missing-genotype corrections algebraically.
// When fast_mlma != FastMlmaMode::Off, pre-quantizes Uk into K-tiled INT8
// slices and computes U_k^T G_block_i8 directly via s8s8s32 GEMM without
// allocating X_panel_f32 or K x BLOCK UkX_scratch.
inline void run_mlma_stream_association_woodbury_i8(const WoodburyMLMACache& wb,
                                                    const Eigen::VectorXf& Vi_y,
                                                    const Eigen::VectorXf& w_sqrt,
                                                    Geno* geno,
                                                    Marker* marker,
                                                    int n,
                                                    int block_size,
                                                    bool log_pval,
                                                    std::ofstream& ofile,
                                                    FastMlmaMode fast_mlma = FastMlmaMode::Off)
{
    const uint32_t total_m = marker->count_extract();
    const bool use_fast_i8 = (fast_mlma != FastMlmaMode::Off);
    LOGGER << "\nStreaming association tests for " << total_m << " SNPs ("
           << (use_fast_i8
               ? (std::string("fast-mlma ") + fast_mlma_mode_name(fast_mlma) + " s8s8s32 Woodbury path")
               : std::string("int8 Woodbury path"))
           << ")..." << std::endl;

    bool unit_weights = true;
    for (int s = 0; s < n; ++s) {
        if (std::abs(w_sqrt[s] - 1.0f) > 1.0e-7f) {
            unit_weights = false;
            break;
        }
    }

    const Eigen::VectorXf vw = unit_weights ? Vi_y : Vi_y.cwiseProduct(w_sqrt);
    const float* vw_ptr = vw.data();
    const double S_vw   = vw.cast<double>().sum();

    Eigen::VectorXd w2;
    double S_w2 = static_cast<double>(n);
    if (!unit_weights) {
        w2   = w_sqrt.cast<double>().array().square().matrix();
        S_w2 = w2.sum();
    }
    const double* w2_ptr = unit_weights ? nullptr : w2.data();

    Eigen::VectorXf u_sum;
    QuantizedWoodburyUk q_uk;
    if (use_fast_i8) {
        q_uk = build_quantized_woodbury_uk(wb, w_sqrt, unit_weights, S_w2, fast_mlma);
        // Free float32 Uk_f immediately after quantization to cut resident memory.
        const_cast<WoodburyMLMACache&>(wb).Uk_f.resize(0, 0);
        LOGGER << "Fast MLMA quantized Uk (" << fast_mlma_mode_name(fast_mlma)
               << "): k=" << q_uk.k
               << ", tiles=" << q_uk.tiles.size()
               << ", Q_mem=" << std::format("{:.3f}", static_cast<double>(q_uk.total_q_bytes()) / (1024.0 * 1024.0 * 1024.0))
               << " GiB, delta_quant=" << std::format("{:.3e}", q_uk.delta_quant)
               << "." << std::endl;
    } else {
        u_sum = unit_weights
            ? wb.Uk_f.cast<double>().rowwise().sum().cast<float>().eval()
            : (wb.Uk_f.cast<double>() * w_sqrt.cast<double>()).cast<float>().eval();
    }

    const int BLOCK = block_size;
    // Pad column stride to 64 bytes so each SNP column starts on a fresh cacheline
    // (matches stdGenoLD in GRM.cpp and avoids false sharing across OMP threads).
    const int ldG = (n + 63) / 64 * 64;
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> G_block_i8(
        static_cast<size_t>(ldG) * static_cast<size_t>(BLOCK), 0);
    Eigen::VectorXf Xt_Vi_y(BLOCK);
    Eigen::VectorXf xvx_diag(BLOCK);
    Eigen::MatrixXf UkX_scratch;
    std::vector<float> X_panel_f32;
    std::vector<int32_t> P_i32;
    std::vector<float> scratch_A_f32, scratch_B_f32, scratch_C_f32;
    if (!use_fast_i8) {
        UkX_scratch.resize(wb.Uk_f.rows(), BLOCK);
    }

    int snp_done = 0;
    int last_pct = -1;

    std::vector<uint8_t> valid_v(BLOCK, 0);
    std::vector<float>   af_v(BLOCK, 0.0f);
    std::vector<float>   additive_af_v(BLOCK, 0.0f);
    std::vector<float>   mu_v(BLOCK, 0.0f);
    std::vector<float>   x_norm_v(BLOCK, 0.0f);
    std::vector<std::vector<int>> snp_miss_samples(BLOCK);

    std::string io_buf;
    io_buf.reserve(4 << 20);

    auto flush_io = [&]() {
        if (!io_buf.empty()) {
            ofile.write(io_buf.data(), static_cast<std::streamsize>(io_buf.size()));
            io_buf.clear();
        }
    };

    const int num_words = (n + 63) / 64;
    const uintptr_t last_word_mask = (n % 64 == 0)
        ? ~static_cast<uintptr_t>(0)
        : ((static_cast<uintptr_t>(1) << (n % 64)) - 1);

    auto callback = [&](uintptr_t* buf, std::span<const uint32_t> exIdx) {
        const int bs = static_cast<int>(exIdx.size());
        const int miss_words_alloc = std::max(geno->getMissPtrSize(), num_words);

        #pragma omp parallel
        {
            std::vector<uintptr_t, AlignedAllocator<uintptr_t, 32>> miss_buf(
                static_cast<size_t>(miss_words_alloc), 0);

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < bs; ++i) {
                bool valid = false;
                float af = 0.0f, add_af = 0.0f;
                double mu = 0.0;
                uint32_t nValidN = 0;
                int8_t* col_i8 = G_block_i8.data() + static_cast<size_t>(i) * static_cast<size_t>(ldG);

                geno->getGenoInt8_bed(buf, i, col_i8, af, add_af, mu, nValidN,
                                      miss_buf.data(), valid, exIdx[i]);
                valid_v[i]       = static_cast<uint8_t>(valid);
                af_v[i]          = af;
                additive_af_v[i] = add_af;
                mu_v[i]          = static_cast<float>(mu);
                snp_miss_samples[i].clear();

                if (!valid) {
                    Xt_Vi_y[i]  = 0.0f;
                    x_norm_v[i] = 0.0f;
                    continue;
                }

                double miss_vw = 0.0;
                double miss_w2 = 0.0;
                if (nValidN < static_cast<uint32_t>(n)) {
                    auto& miss_list = snp_miss_samples[i];
                    for (int w = 0; w < num_words; ++w) {
                        uintptr_t word = miss_buf[w];
                        if (w == num_words - 1) word &= last_word_mask;
                        while (word != 0) {
                            const int bit = std::countr_zero(word);
                            const int s = w * 64 + bit;
                            miss_list.push_back(s);
                            miss_vw += static_cast<double>(vw_ptr[s]);
                            if (!unit_weights) {
                                miss_w2 += w2_ptr[s];
                            }
                            word &= (word - 1);
                        }
                    }
                }

                if (unit_weights) {
                    double dot_vy  = 0.0;
                    int64_t sum_g  = 0;
                    int64_t sum_g2 = 0;
                    #pragma omp simd reduction(+:dot_vy, sum_g, sum_g2)
                    for (int s = 0; s < n; ++s) {
                        const int32_t g = col_i8[s];
                        dot_vy += static_cast<double>(g) * static_cast<double>(vw_ptr[s]);
                        sum_g  += g;
                        sum_g2 += g * g;
                    }
                    Xt_Vi_y[i] = static_cast<float>(dot_vy - mu * (S_vw - miss_vw));

                    const int64_t c2 = (sum_g2 - sum_g) / 2;
                    const int64_t c1 = sum_g - 2 * c2;
                    const int64_t c0 = static_cast<int64_t>(nValidN) - c1 - c2;
                    const double a0 = static_cast<double>(static_cast<float>(-mu));
                    const double a1 = static_cast<double>(static_cast<float>(1.0 - mu));
                    const double a2 = static_cast<double>(static_cast<float>(2.0 - mu));
                    const double x_norm_d = static_cast<double>(c0) * a0 * a0
                                          + static_cast<double>(c1) * a1 * a1
                                          + static_cast<double>(c2) * a2 * a2;
                    x_norm_v[i] = static_cast<float>(x_norm_d);
                } else {
                    double dot_vy    = 0.0;
                    double sum_w2_g  = 0.0;
                    double sum_w2_g2 = 0.0;
                    #pragma omp simd reduction(+:dot_vy, sum_w2_g, sum_w2_g2)
                    for (int s = 0; s < n; ++s) {
                        const int32_t g = col_i8[s];
                        const double gd = static_cast<double>(g);
                        dot_vy += gd * static_cast<double>(vw_ptr[s]);
                        const double w2_s = w2_ptr[s];
                        sum_w2_g  += w2_s * gd;
                        sum_w2_g2 += w2_s * gd * gd;
                    }
                    Xt_Vi_y[i] = static_cast<float>(dot_vy - mu * (S_vw - miss_vw));
                    const double x_norm_d =
                        sum_w2_g2 - 2.0 * mu * sum_w2_g + mu * mu * (S_w2 - miss_w2);
                    x_norm_v[i] = static_cast<float>(std::max(0.0, x_norm_d));
                }
            }
        }

        if (use_fast_i8) {
            woodbury_xvx_diag_block_fast_i8(wb, q_uk, G_block_i8.data(), ldG, bs,
                                            valid_v, mu_v, x_norm_v, snp_miss_samples,
                                            xvx_diag, P_i32,
                                            scratch_A_f32, scratch_B_f32, scratch_C_f32);
        } else {
            woodbury_xvx_diag_block_i8(wb, u_sum, w_sqrt, unit_weights,
                                       G_block_i8.data(), ldG, bs,
                                       valid_v, mu_v, x_norm_v, snp_miss_samples,
                                       xvx_diag, UkX_scratch, X_panel_f32);
        }

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
                                        double block_mem_budget_gb = 0.0,
                                        FastMlmaMode fast_mlma = FastMlmaMode::Off)
{
    Eigen::VectorXf Vi_y(n);

    const bool use_wb  = state.is_woodbury;
    const bool use_llt = state.is_llt;  // selects TRSM (factor = L(V)) vs TRMM (factor = Li(Vi))

    if (use_wb) {
        LOGGER << "MLMA streaming: applying V^{-1} via Woodbury low-rank update "
            "(V^{-1} = D^{-1} - D^{-1} U_k C U_k^T D^{-1})." << std::endl;
        Vi_y = woodbury_apply_Vi_f(state.wb, y_adj);
    } else if (use_llt) {
        // V^{-1} y = U^{-1}(U^{-T} y) via two float triangular solves
        // against U, the (upper) Cholesky factor of V.
        LOGGER << "MLMA streaming: applying V^{-1} via triangular solve against U, "
            "where V = U^T U (STRSM/STRSV — forward/back substitution)." << std::endl;
        Vi_y = y_adj;
        cblas_strsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                    n, state.Vi_L_f.data(), n, Vi_y.data(), 1);
        cblas_strsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    n, state.Vi_L_f.data(), n, Vi_y.data(), 1);
    } else {
        // V^{-1} y = Ui (Ui^T y) via two float triangular products against
        // Ui, the (upper) Cholesky factor of V^{-1} itself. Ui was already
        // computed once at REML-exit/save time (see writeRemlStateFromCtx /
        // build_reml_state) — no factorization happens here, ever.
        LOGGER << "MLMA streaming: applying V^{-1} via triangular product against U_i, "
            "where V^{-1} = U_i^T U_i (STRMM — no linear solve)." << std::endl;
        Vi_y.noalias() = state.Vi_L_f.triangularView<Eigen::Upper>() * y_adj;
        Vi_y = state.Vi_L_f.triangularView<Eigen::Upper>().transpose() * Vi_y;
    }

    Vi_y.array() *= w_sqrt.array();
    const WoodburyMLMACache& wb = state.wb;
    const bool use_int8_wb = use_wb && geno->canUseInt8Mlma() &&
                             (std::getenv("GCTA_DISABLE_INT8_MLMA") == nullptr);
    if (fast_mlma != FastMlmaMode::Off) {
        if (!use_wb) {
            LOGGER.w(0, "--fast-mlma requires a Woodbury REML basis; ignoring --fast-mlma.");
            fast_mlma = FastMlmaMode::Off;
        } else if (!use_int8_wb) {
            LOGGER.w(0, "--fast-mlma requires the additive BED int8 streaming path; falling back to float32 Woodbury.");
            fast_mlma = FastMlmaMode::Off;
        }
    }
    const int raw_n = static_cast<int>(geno->getRawSampleCT());
    const int wb_k  = (use_wb && fast_mlma == FastMlmaMode::Off) ? static_cast<int>(wb.Uk_f.rows()) : 0;
    const int BLOCK = resolve_mlma_block_size(n, block_mem_budget_gb, use_int8_wb, raw_n, wb_k);

    if (use_int8_wb) {
        run_mlma_stream_association_woodbury_i8(wb, Vi_y, w_sqrt,
                                                geno, marker, n,
                                                BLOCK, log_pval, ofile, fast_mlma);
        return;
    }

    Eigen::MatrixXf UkX_scratch = use_wb
        ? Eigen::MatrixXf(wb.Uk_f.rows(), BLOCK)
        : Eigen::MatrixXf();

    auto compute_xvx_diag = [&](Eigen::MatrixXf& X_block, int bs, Eigen::VectorXf& xvx_diag) {
        if (use_wb) {
            woodbury_xvx_diag_block(wb, X_block, bs, xvx_diag, UkX_scratch);
        } else if (use_llt) {
            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                        n, bs, 1.0f,
                        state.Vi_L_f.data(), n,
                        X_block.data(), n);
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < bs; ++j) {
                xvx_diag[j] = X_block.col(j).squaredNorm();
            }
        } else {
            cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
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

//The "inline REML" path
inline void run_mlma_stream_association(RemlCtx& ctx,
                                        const Eigen::VectorXf& y_adj,
                                        const Eigen::VectorXf& w_sqrt,
                                        Geno* geno,
                                        Marker* marker,
                                        int n,
                                        bool log_pval,
                                        std::ofstream& ofile,
                                        double block_mem_budget_gb = 0.0,
                                        FastMlmaMode fast_mlma = FastMlmaMode::Off)
{
    RemlState state = reml::build_reml_state(ctx);
    run_mlma_stream_association(state, y_adj, w_sqrt, geno, marker, n, log_pval, ofile,
                                block_mem_budget_gb, fast_mlma);
}
