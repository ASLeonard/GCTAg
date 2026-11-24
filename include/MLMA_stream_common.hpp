#pragma once

#include <Eigen/Dense>

#include "Geno.h"
#include "Marker.h"
#include "Logger.h"
#include "RemlEngine.hpp"
#include "async_seq_io.hpp"
#include "cpu.h"
#include "mem.hpp"
#include "mlma_woodbury.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <numeric>
#include <optional>
#include <ranges>
#include <type_traits>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif

template <typename T, std::ranges::sized_range IndexRange>
    requires std::integral<std::ranges::range_value_t<IndexRange>>
inline std::vector<T> compact_sample_vector(const std::vector<T>& values,
                                            const IndexRange& keep)
{
    std::vector<T> compacted;
    compacted.reserve(std::ranges::size(keep));
    for (const auto index : keep) {
        compacted.push_back(values[static_cast<size_t>(index)]);
    }
    return compacted;
}

template <std::ranges::sized_range IndexRange>
    requires std::integral<std::ranges::range_value_t<IndexRange>>
inline Eigen::VectorXd compact_sample_vector(const Eigen::VectorXd& values,
                                             const IndexRange& keep)
{
    Eigen::VectorXd compacted(std::ranges::size(keep));
    Eigen::Index dst = 0;
    for (const auto index : keep) {
        compacted[dst++] = values[static_cast<Eigen::Index>(index)];
    }
    return compacted;
}

template <std::ranges::sized_range IndexRange>
    requires std::integral<std::ranges::range_value_t<IndexRange>>
inline Eigen::MatrixXd compact_sample_rows(const Eigen::MatrixXd& values,
                                           const IndexRange& keep)
{
    Eigen::MatrixXd compacted(std::ranges::size(keep), values.cols());
    Eigen::Index dst = 0;
    for (const auto index : keep) {
        compacted.row(dst++) = values.row(static_cast<Eigen::Index>(index));
    }
    return compacted;
}

// Resolve the streaming block width (SNPs per tile) from a memory budget.
// Accounts for all O(BLOCK) streaming buffers held simultaneously per tile:
//   - G_block_i8 (~n bytes, 64B-aligned column stride) vs X_block (4*n bytes)
//   - Backend double-buffered raw slots: ~0.5*raw_n (BED), ~5.0*n (PGEN dosage), ~2.0*raw_n (BGEN)
//   - Woodbury projection scratch UkX_scratch (4 * wb_k bytes when Woodbury active)
// Per-SNP scalars (Xt_Vi_y, xvx_diag, af_v, valid_v) are O(1) per column.
// budget_gb <= 0 preserves the previous fixed BLOCK=10000 behavior exactly.
inline int resolve_mlma_block_size(int n, double budget_gb, bool use_int8_geno = false,
                                   int raw_n = 0, int wb_k = 0,
                                   const std::string& geno_format = "BED",
                                   int align_tile = 0)
{
    int block = 10000; // Somewhat arbitrary, but balance of peak RSS and throughput. Increasing may help STRSM efficiency but increases memory.
    if (budget_gb > 0.0 && n > 0) {

      LOGGER << "MLMA streaming memory budget: " << budget_gb << " GB" << std::endl;
      const int raw_eff = std::max(n, raw_n);
      size_t ldG_est = (static_cast<size_t>(n) + 63) / 64 * 64;
      if (ldG_est % 4096 == 0) ldG_est += 64;
      const double geno_bytes = use_int8_geno
          ? static_cast<double>(ldG_est) /* G_block_i8 */
          : static_cast<double>(sizeof(float)) * static_cast<double>(n); /* X_block */
      double raw_io_bytes = 0.5 * static_cast<double>(raw_eff);
      if (geno_format == "PGEN") {
          // 2 double-buffered slots of (2-bit geno + uint16 dosage_main + 1-bit dosage_present) ~ 5.0 * n bytes
          raw_io_bytes = 5.0 * static_cast<double>(n);
      } else if (geno_format == "BGEN") {
          raw_io_bytes = 2.0 * static_cast<double>(raw_eff);
      }
      const double scratch_bytes = (wb_k > 0)
          ? static_cast<double>(sizeof(float)) * static_cast<double>(wb_k)
          : 0.0;
      constexpr int    min_block = 256;    // floor: keep BLAS3 tiles meaningful
      constexpr int    max_block = 65536;  // ceiling: cap per-column scalar creep

      const double budget_bytes  = budget_gb * 1e9;
      const double per_snp_bytes = geno_bytes + raw_io_bytes + scratch_bytes;
      block = std::clamp(static_cast<int>(budget_bytes / per_snp_bytes), min_block, max_block);
      if (align_tile > 0) {
          const int step = (block >= align_tile) ? align_tile : min_block;
          block = std::clamp((block / step) * step, min_block, max_block);
      }
    }
    LOGGER << "MLMA streaming block size: " << block << " SNPs per tile" << std::endl;
    return block;
}

// Fast single-line TSV formatter for .mlma output using std::to_chars (general, 6) + memcpy.
inline void append_mlma_line(std::string& dst,
                             std::string_view chr,
                             std::string_view name,
                             unsigned bp,
                             std::string_view a1,
                             std::string_view a2,
                             bool stat_ok,
                             double freq,
                             double beta,
                             double se,
                             double pval)
{
    char bp_buf[20];
    const auto bp_res = std::to_chars(bp_buf, bp_buf + sizeof(bp_buf), bp);
    const size_t bp_len = static_cast<size_t>(bp_res.ptr - bp_buf);

    char stat_buf[144];
    char* sp = stat_buf;
    char* const send = stat_buf + sizeof(stat_buf);
    if (!stat_ok) {
        static constexpr char kNaTail[] = "NA\tNA\tNA\tNA\n";
        std::memcpy(sp, kNaTail, sizeof(kNaTail) - 1);
        sp += sizeof(kNaTail) - 1;
    } else {
        auto put_f6g = [&](double v) {
            auto [ptr, ec] = std::to_chars(sp, send, v, std::chars_format::general, 6);
            if (ec == std::errc()) {
                sp = ptr;
            } else {
                const int w = std::snprintf(sp, static_cast<size_t>(send - sp), "%.6g", v);
                if (w > 0) sp += w;
            }
        };
        put_f6g(freq); *sp++ = '\t';
        put_f6g(beta); *sp++ = '\t';
        put_f6g(se);   *sp++ = '\t';
        put_f6g(pval); *sp++ = '\n';
    }
    const size_t stat_len = static_cast<size_t>(sp - stat_buf);

    const size_t total_len = chr.size() + name.size() + bp_len + a1.size() + a2.size() + stat_len + 5;
    if (total_len <= 512) {
        char line[512];
        char* p = line;
        std::memcpy(p, chr.data(), chr.size());   p += chr.size();  *p++ = '\t';
        std::memcpy(p, name.data(), name.size()); p += name.size(); *p++ = '\t';
        std::memcpy(p, bp_buf, bp_len);           p += bp_len;      *p++ = '\t';
        std::memcpy(p, a1.data(), a1.size());     p += a1.size();   *p++ = '\t';
        std::memcpy(p, a2.data(), a2.size());     p += a2.size();   *p++ = '\t';
        std::memcpy(p, stat_buf, stat_len);
        dst.append(line, total_len);
    } else {
        dst.reserve(dst.size() + total_len);
        dst.append(chr.data(), chr.size());   dst.push_back('\t');
        dst.append(name.data(), name.size()); dst.push_back('\t');
        dst.append(bp_buf, bp_len);           dst.push_back('\t');
        dst.append(a1.data(), a1.size());     dst.push_back('\t');
        dst.append(a2.data(), a2.size());     dst.push_back('\t');
        dst.append(stat_buf, stat_len);
    }
}

struct MlmaAsyncOutState {
    char*  slot_ptr  = nullptr;
    size_t slot_cap  = 0;
    size_t slot_used = 0;

    void ensure_slot(gcta_io::AsyncSeqWriter& writer) {
        if (!slot_ptr) {
            slot_ptr = writer.acquire_slot(slot_cap);
            slot_used = 0;
        }
    }

    void append_bytes(gcta_io::AsyncSeqWriter& writer, const char* data, size_t len) {
        size_t off = 0;
        while (off < len) {
            ensure_slot(writer);
            const size_t room = slot_cap - slot_used;
            const size_t take = std::min(room, len - off);
            std::memcpy(slot_ptr + slot_used, data + off, take);
            slot_used += take;
            off += take;
            if (slot_used == slot_cap) {
                writer.submit_slot(slot_used);
                slot_ptr = nullptr;
                slot_used = 0;
            }
        }
    }

    void flush(gcta_io::AsyncSeqWriter& writer) {
        if (slot_ptr) {
            writer.submit_slot(slot_used);
            slot_ptr = nullptr;
            slot_used = 0;
        }
    }
};

// Parallel per-SNP stats + TSV formatting into per-thread pre-reserved buffers,
// copied in parallel via prefix-sum offsets directly into the active AsyncSeqWriter
// slot (16 MiB double-buffered, strictly 1 sequential writer thread).
inline void format_and_buffer_mlma_block(int bs,
                                         std::span<const uint32_t> exIdx,
                                         Marker* marker,
                                         const std::vector<uint8_t>& valid_v,
                                         const std::vector<float>& additive_af_v,
                                         const Eigen::VectorXf& Xt_Vi_y,
                                         const Eigen::VectorXf& xvx_diag,
                                         bool log_pval,
                                         std::vector<std::string>& thr_buf,
                                         MlmaAsyncOutState& out_state,
                                         gcta_io::AsyncSeqWriter& writer)
{
    if (bs <= 0) return;

    int n_active = 1;
    #pragma omp parallel
    {
        const int tid =
        #ifdef _OPENMP
            omp_get_thread_num();
        #else
            0;
        #endif
        const int nt =
        #ifdef _OPENMP
            omp_get_num_threads();
        #else
            1;
        #endif
        #pragma omp single
        {
            n_active = nt;
        }
        const int i_beg = static_cast<int>((static_cast<int64_t>(bs) * tid) / nt);
        const int i_end = static_cast<int>((static_cast<int64_t>(bs) * (tid + 1)) / nt);
        std::string& buf = thr_buf[static_cast<size_t>(tid)];
        buf.clear();
        for (int i = i_beg; i < i_end; ++i) {
            const uint32_t raw           = marker->getRawIndex(exIdx[static_cast<size_t>(i)]);
            const std::string_view chr   = marker->getRawChr(raw);
            const std::string_view name  = marker->getRawName(raw);
            const unsigned         bp    = static_cast<unsigned>(marker->getRawBp(raw));
            const std::string_view a1    = marker->getRawA1(raw);
            const std::string_view a2    = marker->getRawA2(raw);

            float beta_val = 0.0f, se_val = 0.0f;
            double pval_val = 0.0;
            const bool stat_ok = valid_v[static_cast<size_t>(i)] &&
                mlma_snp_stat(Xt_Vi_y[i], xvx_diag[i], log_pval,
                              beta_val, se_val, pval_val);

            append_mlma_line(buf, chr, name, bp, a1, a2, stat_ok,
                             static_cast<double>(additive_af_v[static_cast<size_t>(i)]),
                             static_cast<double>(beta_val),
                             static_cast<double>(se_val),
                             pval_val);
        }
    }

    std::vector<size_t> offsets(static_cast<size_t>(n_active) + 1, 0);
    for (int t = 0; t < n_active; ++t) {
        offsets[static_cast<size_t>(t) + 1] =
            offsets[static_cast<size_t>(t)] + thr_buf[static_cast<size_t>(t)].size();
    }
    const size_t total_bytes = offsets[static_cast<size_t>(n_active)];
    if (total_bytes == 0) return;

    out_state.ensure_slot(writer);
    if (out_state.slot_used > 0 && out_state.slot_used + total_bytes > out_state.slot_cap) {
        out_state.flush(writer);
        out_state.ensure_slot(writer);
    }

    if (out_state.slot_used + total_bytes <= out_state.slot_cap) {
        char* const dst_base = out_state.slot_ptr + out_state.slot_used;
        #pragma omp parallel for schedule(static) num_threads(n_active) if (total_bytes >= 4194304 && n_active > 1)
        for (int t = 0; t < n_active; ++t) {
            const std::string& tb = thr_buf[static_cast<size_t>(t)];
            if (!tb.empty()) {
                std::memcpy(dst_base + offsets[static_cast<size_t>(t)], tb.data(), tb.size());
            }
        }
        out_state.slot_used += total_bytes;
        if (out_state.slot_used >= out_state.slot_cap) {
            out_state.flush(writer);
        }
    } else {
        for (int t = 0; t < n_active; ++t) {
            const std::string& tb = thr_buf[static_cast<size_t>(t)];
            if (!tb.empty()) {
                out_state.append_bytes(writer, tb.data(), tb.size());
            }
        }
    }
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
                                             gcta_io::AsyncSeqWriter& writer)
{
    const uint32_t total_m = marker->count_extract();
    LOGGER << "\nStreaming association tests for " << total_m << " SNPs..." << std::endl;

    const int BLOCK = block_size;
    Eigen::MatrixXf X_block(n, BLOCK);
    Eigen::VectorXf Xt_Vi_y(BLOCK);
    Eigen::VectorXf xvx_diag(BLOCK);

    // Unweighted analyses (the common case) skip the per-column w_sqrt multiply.
    bool unit_weights = true;
    for (int s = 0; s < n; ++s) {
        if (std::abs(w_sqrt[s] - 1.0f) > 1.0e-7f) {
            unit_weights = false;
            break;
        }
    }

    const float* const vi_y_ptr = Vi_y.data();
    const float* const w_ptr    = w_sqrt.data();

    int  snp_done = 0;
    int  last_pct = -1;

    std::vector<uint8_t> valid_v(BLOCK, 0);
    std::vector<float>   af_v(BLOCK, 0.0f);
    std::vector<float>   additive_af_v(BLOCK, 0.0f);
    std::vector<float>   x_norm_v(BLOCK, 0.0f);

    const int max_threads =
    #ifdef _OPENMP
        std::max(1, omp_get_max_threads());
    #else
        1;
    #endif
    std::vector<std::string> thr_buf(static_cast<size_t>(max_threads));
    const size_t per_thr_reserve = std::max<size_t>(
        64ULL << 10,
        (static_cast<size_t>(BLOCK) / static_cast<size_t>(max_threads) + 16) * 96);
    for (auto& tb : thr_buf) tb.reserve(per_thr_reserve);

    MlmaAsyncOutState out_state;
    const std::string_view hdr_line = log_pval
        ? std::string_view("Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\tlog_p\n")
        : std::string_view("Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\tp\n");
    out_state.append_bytes(writer, hdr_line.data(), hdr_line.size());

    auto callback = [&](uintptr_t* buf, std::span<const uint32_t> exIdx) {
        const int bs = static_cast<int>(exIdx.size());

        // Decode genotypes directly into X_block columns as float, fusing
        // Xt_Vi_y and ||X_block.col(i)||^2 while each decoded column is cache-hot.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < bs; ++i) {
            bool valid = false;
            float af = 0.0f, add_af = 0.0f;
            float* const x_col = X_block.col(i).data();
            geno->getGenoFloat(buf, i, x_col, af, add_af, valid, exIdx[i]);
            if (valid) {
                double dot_vy  = 0.0;
                double norm_sq = 0.0;
                if (unit_weights) {
                    #pragma omp simd reduction(+:dot_vy, norm_sq)
                    for (int s = 0; s < n; ++s) {
                        const float xs = x_col[s];
                        dot_vy  += static_cast<double>(xs) * static_cast<double>(vi_y_ptr[s]);
                        norm_sq += static_cast<double>(xs) * static_cast<double>(xs);
                    }
                } else {
                    #pragma omp simd reduction(+:dot_vy, norm_sq)
                    for (int s = 0; s < n; ++s) {
                        const float xs = x_col[s] * w_ptr[s];
                        x_col[s] = xs;
                        dot_vy  += static_cast<double>(xs) * static_cast<double>(vi_y_ptr[s]);
                        norm_sq += static_cast<double>(xs) * static_cast<double>(xs);
                    }
                }
                Xt_Vi_y[i]  = static_cast<float>(dot_vy);
                x_norm_v[i] = static_cast<float>(norm_sq);
            } else {
                std::memset(x_col, 0, static_cast<size_t>(n) * sizeof(float));
                Xt_Vi_y[i]  = 0.0f;
                x_norm_v[i] = 0.0f;
            }
            valid_v[i]       = static_cast<uint8_t>(valid);
            af_v[i]          = af;
            additive_af_v[i] = add_af;
        }

        if constexpr (std::invocable<XVXDiagFn, Eigen::MatrixXf&, int, Eigen::VectorXf&, const float*>) {
            compute_xvx_diag(X_block, bs, xvx_diag, x_norm_v.data());
        } else {
            compute_xvx_diag(X_block, bs, xvx_diag);
        }

        format_and_buffer_mlma_block(bs, exIdx, marker, valid_v, additive_af_v,
                                     Xt_Vi_y, xvx_diag, log_pval,
                                     thr_buf, out_state, writer);

        snp_done += bs;
        const int cur_pct = (total_m > 0)
            ? static_cast<int>((uint64_t)snp_done * 100 / total_m) : 100;
        if (cur_pct != last_pct) {
            LOGGER.p(0, std::to_string(snp_done) + " / " + std::to_string(total_m)
                        + " SNPs (" + std::to_string(cur_pct) + "%)");
            last_pct = cur_pct;
        }
    };

    std::vector<uint32_t> extractIndex(total_m);
    std::iota(extractIndex.begin(), extractIndex.end(), 0u);
    geno->loopDouble(extractIndex, BLOCK,
                     /*bMakeGeno*/   true,
                     /*bGenoCenter*/ true,
                     /*bGenoStd*/    false,
                     /*bMakeMiss*/   false,
                     {callback});

    out_state.flush(writer);
}

// Int8 Woodbury streaming association path: stores raw additive BED genotypes
// in int8_t (1 byte/sample/SNP instead of 4 bytes/sample/SNP) and applies
// mean-centering and sparse missing-genotype corrections algebraically.
// When fast_mlma != FastMlmaMode::Off, uses prebuilt_q_uk if already streamed
// directly from TUN3, or quantizes Uk into K-tiled INT8 slices and computes
// U_k^T G_block_i8 directly via u8s8s32 GEMM.
inline void run_mlma_stream_association_woodbury_i8(const WoodburyMLMACache& wb,
                                                    const Eigen::VectorXf& Vi_y,
                                                    const Eigen::VectorXf& w_sqrt,
                                                    Geno* geno,
                                                    Marker* marker,
                                                    int n,
                                                    int block_size,
                                                    bool log_pval,
                                                    gcta_io::AsyncSeqWriter& writer,
                                                    FastMlmaMode fast_mlma = FastMlmaMode::Off,
                                                    std::optional<QuantizedWoodburyUk>* prebuilt_q_uk = nullptr,
                                                    const MlmaTuningOptions& tuning = {},
                                                    MlmaProfileStats* prof = nullptr)
{
    using Clock = std::chrono::steady_clock;
    const uint32_t total_m = marker->count_extract();
    const bool use_fast_i8 = (fast_mlma != FastMlmaMode::Off);
    LOGGER << "\nStreaming association tests for " << total_m << " SNPs ("
           << (use_fast_i8
               ? (std::string("fast-mlma ") + fast_mlma_mode_name(fast_mlma) + " u8s8s32 Woodbury path")
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

    const int max_threads =
    #ifdef _OPENMP
        std::max(1, omp_get_max_threads());
    #else
        1;
    #endif

    Eigen::VectorXf u_sum;
    QuantizedWoodburyUk q_uk;
    if (use_fast_i8) {
        const bool from_prebuilt = (prebuilt_q_uk != nullptr && prebuilt_q_uk->has_value());
        if (from_prebuilt) {
            q_uk = std::move(**prebuilt_q_uk);
            prebuilt_q_uk->reset();
        } else {
            const auto t_qb0 = prof ? Clock::now() : Clock::time_point{};
            q_uk = build_quantized_woodbury_uk(wb, w_sqrt, unit_weights, S_w2, fast_mlma, tuning.q_rows);
            if (prof) {
                prof->q_build_s += std::chrono::duration<double>(Clock::now() - t_qb0).count();
            }
            // Free float32 Uk_f immediately after quantization to cut resident memory
            // before optional AOCL pre-reordered Q buffers.
            const_cast<WoodburyMLMACache&>(wb).Uk_f.resize(0, 0);
        }
        if (tuning.reorder_q) {
            const auto t_qr0 = prof ? Clock::now() : Clock::time_point{};
            reorder_quantized_woodbury_uk(q_uk);
            if (prof) {
                prof->q_reorder_s += std::chrono::duration<double>(Clock::now() - t_qr0).count();
            }
        }
        if (prof) {
            prof->q_tiles = static_cast<int>(q_uk.tiles.size());
        }
        std::string reord_msg;
        if (q_uk.total_q_reordered_bytes() > 0) {
            reord_msg = ", Q_reordered_mem=" + std::format("{:.3f}",
                static_cast<double>(q_uk.total_q_reordered_bytes()) / (1024.0 * 1024.0 * 1024.0)) + " GiB";
        }
        LOGGER << "Fast MLMA quantized Uk (" << fast_mlma_mode_name(fast_mlma)
               << (from_prebuilt ? ", direct/streamed" : "")
               << "): k=" << q_uk.k
               << ", tiles=" << q_uk.tiles.size()
               << ", Q_mem=" << std::format("{:.3f}", static_cast<double>(q_uk.total_q_bytes()) / (1024.0 * 1024.0 * 1024.0))
               << " GiB" << reord_msg
               << ", delta_quant=" << std::format("{:.3e}", q_uk.delta_quant)
               << "." << std::endl;
    } else {
        // Compute u_sum = Uk_f * w_sqrt in double precision without materializing
        // a 40+ GB MatrixXd temporary (which Eigen's Uk_f.cast<double>() * w_sqrt
        // risks under EIGEN_USE_BLAS).
        const int k_wb = static_cast<int>(wb.Uk_f.rows());
        u_sum.resize(k_wb);
        if (k_wb > 0) {
            const float* const uk_base = wb.Uk_f.data();
            const float* const w_ptr   = w_sqrt.data();
            const size_t k_pad64 = (static_cast<size_t>(k_wb) + 7) / 8 * 8;
            std::vector<double> thr_sum(static_cast<size_t>(max_threads) * k_pad64, 0.0);
            int n_act = 1;
            #pragma omp parallel
            {
                const int tid =
                #ifdef _OPENMP
                    omp_get_thread_num();
                #else
                    0;
                #endif
                #pragma omp single
                {
                    n_act =
                    #ifdef _OPENMP
                        omp_get_num_threads();
                    #else
                        1;
                    #endif
                }
                double* const lsum = thr_sum.data() + static_cast<size_t>(tid) * k_pad64;
                if (unit_weights) {
                    #pragma omp for schedule(static)
                    for (int s = 0; s < n; ++s) {
                        const float* const uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k_wb);
                        #pragma omp simd
                        for (int r = 0; r < k_wb; ++r) {
                            lsum[r] += static_cast<double>(uk_col[r]);
                        }
                    }
                } else {
                    #pragma omp for schedule(static)
                    for (int s = 0; s < n; ++s) {
                        const float* const uk_col = uk_base + static_cast<size_t>(s) * static_cast<size_t>(k_wb);
                        const double ws = static_cast<double>(w_ptr[s]);
                        #pragma omp simd
                        for (int r = 0; r < k_wb; ++r) {
                            lsum[r] += static_cast<double>(uk_col[r]) * ws;
                        }
                    }
                }
            }
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < k_wb; ++r) {
                double acc = 0.0;
                for (int t = 0; t < n_act; ++t) {
                    acc += thr_sum[static_cast<size_t>(t) * k_pad64 + static_cast<size_t>(r)];
                }
                u_sum[r] = static_cast<float>(acc);
            }
        }
    }

    const int BLOCK = block_size;
    // Pad column stride to 64 bytes (avoiding 4096-byte multiples that cause
    // Zen 4 L1D cache-set aliasing) and allocate uninitialized without serial
    // zero-fill so parallel decode first-touches pages cleanly.
    int ldG = (n + 63) / 64 * 64;
    if (ldG % 4096 == 0) ldG += 64;
    const size_t g_block_bytes = static_cast<size_t>(ldG) * static_cast<size_t>(BLOCK);
    auto G_block_i8 = make_aligned_uninitialized<int8_t, 64>(g_block_bytes);
    if (g_block_bytes > 0 && !G_block_i8) {
        LOGGER.e(0, "Failed to allocate int8 genotype streaming block.");
    }
    Eigen::VectorXf Xt_Vi_y(BLOCK);
    Eigen::VectorXf xvx_diag(BLOCK);
    Eigen::MatrixXf UkX_scratch;
    AlignedRawBuffer<float, 64> X_panel_f32;
    AlignedRawBuffer<int32_t, 64> P_i32;
    AlignedRawBuffer<int32_t, 64> thr_miss_acc;
    std::vector<float> scratch_A_f32, scratch_B_f32, scratch_C_f32;
    std::vector<double> ukx_norm_sq(static_cast<size_t>(BLOCK), 0.0);

    std::vector<int>     compact_idx(static_cast<size_t>(BLOCK));
    std::vector<uint8_t> compact_valid(static_cast<size_t>(BLOCK), 1);
    std::vector<float>   compact_mu(static_cast<size_t>(BLOCK));
    std::vector<float>   compact_x_norm(static_cast<size_t>(BLOCK));
    Eigen::VectorXf      compact_xvx(BLOCK);

    int snp_done = 0;
    int last_pct = -1;
    if (prof) {
        prof->block_size = BLOCK;
    }

    std::vector<uint8_t> valid_v(BLOCK, 0);
    std::vector<float>   af_v(BLOCK, 0.0f);
    std::vector<float>   additive_af_v(BLOCK, 0.0f);
    std::vector<float>   mu_v(BLOCK, 0.0f);
    std::vector<float>   x_norm_v(BLOCK, 0.0f);
    std::vector<std::vector<int>> snp_miss_samples(BLOCK);

    std::vector<std::string> thr_buf(static_cast<size_t>(max_threads));
    const size_t per_thr_reserve = std::max<size_t>(
        64ULL << 10,
        (static_cast<size_t>(BLOCK) / static_cast<size_t>(max_threads) + 16) * 96);
    for (auto& tb : thr_buf) tb.reserve(per_thr_reserve);

    MlmaAsyncOutState out_state;
    const std::string_view hdr_line = log_pval
        ? std::string_view("Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\tlog_p\n")
        : std::string_view("Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\tp\n");
    out_state.append_bytes(writer, hdr_line.data(), hdr_line.size());

    // Persistent uninitialized per-thread miss_bits scratch: bed_simd::decode_i8
    // only writes/zeroes miss_bits when a SNP has missing calls (nmiss > 0), so
    // post-Beagle 0%-missing runs never fault or memset these pages at all.
    const int num_words = (n + 63) / 64;
    const size_t miss_words_stride = (static_cast<size_t>(std::max(geno->getMissPtrSize(), num_words)) + 7) / 8 * 8;
    AlignedRawBuffer<uintptr_t, 64> thr_miss_buf(
        static_cast<size_t>(max_threads) * miss_words_stride);

    auto callback = [&](uintptr_t* buf, std::span<const uint32_t> exIdx) {
        const int bs = static_cast<int>(exIdx.size());
        int block_has_miss = 0;

        const auto t_dec0 = prof ? Clock::now() : Clock::time_point{};
        #pragma omp parallel reduction(|:block_has_miss)
        {
            const int tid =
            #ifdef _OPENMP
                omp_get_thread_num();
            #else
                0;
            #endif
            uintptr_t* const miss_ptr = thr_miss_buf.data() + static_cast<size_t>(tid) * miss_words_stride;

            #pragma omp for schedule(static)
            for (int i = 0; i < bs; ++i) {
                bool valid = false;
                float af = 0.0f, add_af = 0.0f;
                double mu = 0.0;
                uint32_t nValidN = 0;
                int8_t* col_i8 = G_block_i8.get() + static_cast<size_t>(i) * static_cast<size_t>(ldG);

                // Single fused pass over the raw .bed record: int8 column,
                // exact (c0,c1,c2,c_miss), sum g*vw (+ w2 sums when weighted)
                // and the sparse missing list -- no re-scan of col_i8, and no
                // missing-bitmask work at all when the SNP has no missing calls.
                BedI8Fused fz;
                fz.vw = vw_ptr;
                fz.w2 = unit_weights ? nullptr : w2_ptr;
                fz.miss_list = &snp_miss_samples[i];
                geno->getGenoInt8_bed(buf, i, col_i8, af, add_af, mu, nValidN,
                                      miss_ptr, valid, exIdx[i], &fz);
                valid_v[i]       = static_cast<uint8_t>(valid);
                af_v[i]          = af;
                additive_af_v[i] = add_af;
                mu_v[i]          = static_cast<float>(mu);

                if (!valid) {
                    snp_miss_samples[i].clear();
                    Xt_Vi_y[i]  = 0.0f;
                    x_norm_v[i] = 0.0f;
                    continue;
                }

                if (fz.cmiss != 0) {
                    block_has_miss = 1;
                }

                const double miss_vw = (fz.cmiss != 0) ? fz.miss_vw : 0.0;
                Xt_Vi_y[i] = static_cast<float>(fz.dot_vw - mu * (S_vw - miss_vw));
                if (unit_weights) {
                    const int64_t c2 = static_cast<int64_t>(fz.c2);
                    const int64_t c1 = static_cast<int64_t>(fz.c1);
                    const int64_t c0 = static_cast<int64_t>(nValidN) - c1 - c2;
                    const double a0 = static_cast<double>(static_cast<float>(-mu));
                    const double a1 = static_cast<double>(static_cast<float>(1.0 - mu));
                    const double a2 = static_cast<double>(static_cast<float>(2.0 - mu));
                    const double x_norm_d = static_cast<double>(c0) * a0 * a0
                                          + static_cast<double>(c1) * a1 * a1
                                          + static_cast<double>(c2) * a2 * a2;
                    x_norm_v[i] = static_cast<float>(x_norm_d);
                } else {
                    const double miss_w2 = (fz.cmiss != 0) ? fz.miss_w2 : 0.0;
                    const double x_norm_d =
                        fz.dot2_w2 - 2.0 * mu * fz.dot_w2 + mu * mu * (S_w2 - miss_w2);
                    x_norm_v[i] = static_cast<float>(std::max(0.0, x_norm_d));
                }
            }
        }
        if (prof) {
            prof->decode_s += std::chrono::duration<double>(Clock::now() - t_dec0).count();
        }

        // Compact valid SNPs before Woodbury GEMM so filtered SNPs are skipped cleanly.
        int n_valid = 0;
        for (int i = 0; i < bs; ++i) {
            if (valid_v[i]) {
                compact_idx[static_cast<size_t>(n_valid)] = i;
                ++n_valid;
            }
        }

        if (n_valid == 0) {
            xvx_diag.head(bs).setZero();
        } else if (n_valid == bs) {
            if (use_fast_i8) {
                woodbury_xvx_diag_block_fast_i8(wb, q_uk, G_block_i8.get(), ldG, bs,
                                                valid_v, mu_v, x_norm_v, snp_miss_samples,
                                                xvx_diag, P_i32,
                                                scratch_A_f32, scratch_B_f32, scratch_C_f32,
                                                block_has_miss != 0, &ukx_norm_sq, &thr_miss_acc,
                                                tuning.p_budget_mb, tuning.snp_tile, prof);
            } else {
                woodbury_xvx_diag_block_i8(wb, u_sum, w_sqrt, unit_weights,
                                           G_block_i8.get(), ldG, bs,
                                           valid_v, mu_v, x_norm_v, snp_miss_samples,
                                           xvx_diag, UkX_scratch, X_panel_f32,
                                           block_has_miss != 0, &ukx_norm_sq,
                                           tuning.sample_tile, tuning.ukx_budget_mb, prof);
            }
        } else {
            for (int c = 0; c < n_valid; ++c) {
                const int src = compact_idx[static_cast<size_t>(c)];
                compact_mu[static_cast<size_t>(c)]     = mu_v[src];
                compact_x_norm[static_cast<size_t>(c)] = x_norm_v[src];
                if (src != c) {
                    std::memcpy(G_block_i8.get() + static_cast<size_t>(c) * static_cast<size_t>(ldG),
                                G_block_i8.get() + static_cast<size_t>(src) * static_cast<size_t>(ldG),
                                static_cast<size_t>(n));
                    if (block_has_miss != 0) {
                        snp_miss_samples[static_cast<size_t>(c)].swap(snp_miss_samples[static_cast<size_t>(src)]);
                    }
                }
            }
            if (use_fast_i8) {
                woodbury_xvx_diag_block_fast_i8(wb, q_uk, G_block_i8.get(), ldG, n_valid,
                                                compact_valid, compact_mu, compact_x_norm, snp_miss_samples,
                                                compact_xvx, P_i32,
                                                scratch_A_f32, scratch_B_f32, scratch_C_f32,
                                                block_has_miss != 0, &ukx_norm_sq, &thr_miss_acc,
                                                tuning.p_budget_mb, tuning.snp_tile, prof);
            } else {
                woodbury_xvx_diag_block_i8(wb, u_sum, w_sqrt, unit_weights,
                                           G_block_i8.get(), ldG, n_valid,
                                           compact_valid, compact_mu, compact_x_norm, snp_miss_samples,
                                           compact_xvx, UkX_scratch, X_panel_f32,
                                           block_has_miss != 0, &ukx_norm_sq,
                                           tuning.sample_tile, tuning.ukx_budget_mb, prof);
            }
            xvx_diag.head(bs).setZero();
            for (int c = 0; c < n_valid; ++c) {
                xvx_diag[compact_idx[static_cast<size_t>(c)]] = compact_xvx[c];
            }
        }

        const auto t_fmt0 = prof ? Clock::now() : Clock::time_point{};
        format_and_buffer_mlma_block(bs, exIdx, marker, valid_v, additive_af_v,
                                     Xt_Vi_y, xvx_diag, log_pval,
                                     thr_buf, out_state, writer);
        if (prof) {
            prof->format_write_s += std::chrono::duration<double>(Clock::now() - t_fmt0).count();
            ++prof->blocks;
            prof->valid_snps += static_cast<uint64_t>(n_valid);
            prof->total_snps += static_cast<uint64_t>(bs);
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

    std::vector<uint32_t> extractIndex(total_m);
    std::iota(extractIndex.begin(), extractIndex.end(), 0u);
    geno->loopDouble(extractIndex, BLOCK,
                     /*bMakeGeno*/   true,
                     /*bGenoCenter*/ true,
                     /*bGenoStd*/    false,
                     /*bMakeMiss*/   false,
                     {callback});

    out_state.flush(writer);
}

// The "--save-reml and --load-reml" path
inline void run_mlma_stream_association(RemlState& state,
                                        const Eigen::VectorXf& y_adj,
                                        const Eigen::VectorXf& w_sqrt,
                                        Geno* geno,
                                        Marker* marker,
                                        int n,
                                        bool log_pval,
                                        gcta_io::AsyncSeqWriter& writer,
                                        double block_mem_budget_gb = 0.0,
                                        FastMlmaMode fast_mlma = FastMlmaMode::Off,
                                        const MlmaTuningOptions& tuning = {},
                                        MlmaProfileStats* prof = nullptr)
{
    Eigen::VectorXf Vi_y(n);

    const bool use_wb  = state.is_woodbury;
    const bool use_llt = state.is_llt;  // selects TRSM (factor = L(V)) vs TRMM (factor = Li(Vi))

    if (use_wb) {
        if (state.has_cached_Vi_y && state.cached_Vi_y.size() == n) {
            LOGGER << "MLMA streaming: using precomputed exact V^{-1} y_adj from REML state." << std::endl;
            Vi_y = state.cached_Vi_y;
        } else {
            LOGGER << "MLMA streaming: applying V^{-1} via Woodbury low-rank update "
                "(V^{-1} = D^{-1} - D^{-1} U_k C U_k^T D^{-1})." << std::endl;
            Vi_y = woodbury_apply_Vi_f(state.wb, y_adj);
        }
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
    const int wb_k  = (use_wb && !use_int8_wb)
        ? static_cast<int>(wb.Uk_f.rows())
        : 0;
    const int align_tile = use_int8_wb ? 2048 : 0;
    const int BLOCK = (tuning.block_size > 0)
        ? tuning.block_size
        : resolve_mlma_block_size(n, block_mem_budget_gb, use_int8_wb, raw_n, wb_k,
                                  geno->getGenoFormat(), align_tile);
    if (tuning.block_size > 0) {
        LOGGER << "MLMA streaming block size (override): " << BLOCK << " SNPs per tile" << std::endl;
    }

    if (use_int8_wb) {
        run_mlma_stream_association_woodbury_i8(wb, Vi_y, w_sqrt,
                                                geno, marker, n,
                                                BLOCK, log_pval, writer, fast_mlma,
                                                state.prebuilt_q_uk.has_value() ? &state.prebuilt_q_uk : nullptr,
                                                tuning, prof);
        return;
    }

    Eigen::MatrixXf UkX_scratch = use_wb
        ? Eigen::MatrixXf(wb.Uk_f.rows(), BLOCK)
        : Eigen::MatrixXf();

    auto compute_xvx_diag = [&](Eigen::MatrixXf& X_block, int bs,
                                Eigen::VectorXf& xvx_diag,
                                const float* precomputed_x_norm) {
        if (use_wb) {
            woodbury_xvx_diag_block(wb, X_block, bs, xvx_diag, UkX_scratch, precomputed_x_norm);
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
                                     log_pval, writer);
}

// The "inline REML" path
inline void run_mlma_stream_association(RemlCtx& ctx,
                                        const Eigen::VectorXf& y_adj,
                                        const Eigen::VectorXf& w_sqrt,
                                        Geno* geno,
                                        Marker* marker,
                                        int n,
                                        bool log_pval,
                                        gcta_io::AsyncSeqWriter& writer,
                                        double block_mem_budget_gb = 0.0,
                                        FastMlmaMode fast_mlma = FastMlmaMode::Off,
                                        const MlmaTuningOptions& tuning = {},
                                        MlmaProfileStats* prof = nullptr)
{
    using Clock = std::chrono::steady_clock;
    const bool direct_fast_wb = ctx.Vi_use_woodbury_basis &&
                                (fast_mlma != FastMlmaMode::Off) &&
                                geno->canUseInt8Mlma() &&
                                (std::getenv("GCTA_DISABLE_INT8_MLMA") == nullptr);
    if (direct_fast_wb && !ctx.Uk_is_transposed && ctx.Uk.rows() == n && ctx.Uk.cols() > 0) {
        // Direct inline quantization from n x k ctx.Uk without ever allocating a
        // duplicate 4*k*n Uk_f matrix (cuts peak RSS from 80 GB -> 50-60 GB at N=500k, k=20k).
        const Eigen::VectorXf ck_f = ctx.ck.cast<float>();
        const Eigen::VectorXf sqrt_ck_f = ck_f.cwiseSqrt();
        const float sig2e_f = static_cast<float>(ctx.sigma2_eff);

        Eigen::VectorXf u_y = ctx.Uk.transpose() * y_adj;
        u_y.array() *= ck_f.array();
        Eigen::VectorXf vi_y = (y_adj - ctx.Uk * u_y) / sig2e_f;

        bool unit_weights = true;
        double S_w2 = static_cast<double>(n);
        for (int s = 0; s < n; ++s) {
            if (std::abs(w_sqrt[s] - 1.0f) > 1.0e-7f) {
                unit_weights = false;
                break;
            }
        }
        if (!unit_weights) {
            S_w2 = w_sqrt.cast<double>().array().square().sum();
        }

        const auto t_qb0 = prof ? Clock::now() : Clock::time_point{};
        QuantizedWoodburyUk q_uk = build_quantized_woodbury_uk_from_nxk(
            ctx.Uk, sqrt_ck_f, w_sqrt, unit_weights, S_w2, fast_mlma, tuning.q_rows);
        if (prof) {
            prof->q_build_s += std::chrono::duration<double>(Clock::now() - t_qb0).count();
        }
        ctx.Uk.resize(0, 0); // free n x k ctx.Uk BEFORE build_reml_state

        RemlState state = reml::build_reml_state(ctx);
        state.prebuilt_q_uk = std::move(q_uk);
        state.cached_Vi_y = std::move(vi_y);
        state.has_cached_Vi_y = true;
        run_mlma_stream_association(state, y_adj, w_sqrt, geno, marker, n, log_pval, writer,
                                    block_mem_budget_gb, fast_mlma, tuning, prof);
        return;
    }

    RemlState state = reml::build_reml_state(ctx);
    run_mlma_stream_association(state, y_adj, w_sqrt, geno, marker, n, log_pval, writer,
                                block_mem_budget_gb, fast_mlma, tuning, prof);
}
