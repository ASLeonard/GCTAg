#include "async_seq_io.hpp"
#include "grm_format.hpp"
#include "grm_binary_io.hpp"
#include "chunked_grm_matvec.hpp"
#include "PCA_stream.h"
#include "Pheno.h"
#include "symmetric_eigendecomp.hpp"
#include <cpu.h>
#include <mem.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <unordered_set>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <omp.h>

using std::string;
using std::to_string;
using std::vector;

namespace pca_stream_detail {

struct EvecHeader {
    char     magic[4]{'E', 'V', 'E', 'C'};
    uint32_t version  = 1u;
    uint32_t dtype    = 0u; // 0 = float32, 1 = row-scaled INT16
    int32_t  n        = 0;
    int32_t  k        = 0;
    uint32_t reserved = 0u;
    uint64_t id_hash  = 0;
    uint64_t id_bytes = 0;
};
static_assert(sizeof(EvecHeader) == 40, "EvecHeader must be 40 bytes");
static_assert(std::is_trivially_copyable_v<EvecHeader>, "EvecHeader must be trivially copyable");

void append_bytes_buffered(gcta_io::AsyncSeqWriter& writer,
                           AlignedRawBuffer<char, 64>*& slot,
                           size_t& slot_used,
                           size_t slot_cap,
                           const char* data,
                           size_t nbytes)
{
    size_t pos = 0;
    while (pos < nbytes) {
        if (slot == nullptr) {
            slot = &writer.acquire_slot();
            if (slot->size() < slot_cap) slot->resize(slot_cap);
            slot_used = 0;
        }
        const size_t avail = slot_cap - slot_used;
        const size_t take  = std::min(avail, nbytes - pos);
        std::memcpy(slot->data() + slot_used, data + pos, take);
        slot_used += take;
        pos += take;
        if (slot_used >= slot_cap) {
            writer.submit_slot(slot_used);
            slot = nullptr;
            slot_used = 0;
        }
    }
}

void append_thread_bufs_buffered(gcta_io::AsyncSeqWriter& writer,
                                 AlignedRawBuffer<char, 64>*& slot,
                                 size_t& slot_used,
                                 size_t slot_cap,
                                 const std::vector<std::string>& thread_bufs,
                                 int actual_nt)
{
    if (actual_nt <= 0) return;
    std::vector<size_t> offs(static_cast<size_t>(actual_nt) + 1, 0);
    for (int t = 0; t < actual_nt; ++t) {
        offs[static_cast<size_t>(t + 1)] = offs[static_cast<size_t>(t)] +
                                           thread_bufs[static_cast<size_t>(t)].size();
    }
    const size_t total = offs[static_cast<size_t>(actual_nt)];
    if (total == 0) return;

    if (total <= slot_cap) {
        if (slot != nullptr && slot_used + total > slot_cap) {
            writer.submit_slot(slot_used);
            slot = nullptr;
            slot_used = 0;
        }
        if (slot == nullptr) {
            slot = &writer.acquire_slot();
            if (slot->size() < slot_cap) slot->resize(slot_cap);
            slot_used = 0;
        }
        char* const dst_base = slot->data() + slot_used;
        if (actual_nt > 1 && total >= (1u << 18)) {
            #pragma omp parallel for schedule(static) num_threads(actual_nt)
            for (int t = 0; t < actual_nt; ++t) {
                const std::string& tb = thread_bufs[static_cast<size_t>(t)];
                if (!tb.empty()) {
                    std::memcpy(dst_base + offs[static_cast<size_t>(t)], tb.data(), tb.size());
                }
            }
        } else {
            for (int t = 0; t < actual_nt; ++t) {
                const std::string& tb = thread_bufs[static_cast<size_t>(t)];
                if (!tb.empty()) {
                    std::memcpy(dst_base + offs[static_cast<size_t>(t)], tb.data(), tb.size());
                }
            }
        }
        slot_used += total;
        if (slot_used >= slot_cap) {
            writer.submit_slot(slot_used);
            slot = nullptr;
            slot_used = 0;
        }
        return;
    }

    for (int t = 0; t < actual_nt; ++t) {
        const std::string& tb = thread_bufs[static_cast<size_t>(t)];
        if (!tb.empty()) {
            append_bytes_buffered(writer, slot, slot_used, slot_cap, tb.data(), tb.size());
        }
    }
}

} // namespace pca_stream_detail

void PCAStream::run(const gcta_cli::PcaOptions& opts, const std::string& out_prefix)
{
    using pca_stream_detail::EvecHeader;
    using pca_stream_detail::append_bytes_buffered;
    using pca_stream_detail::append_thread_bufs_buffered;
    const string& grm_pfx = opts.grm;
    std::optional<gcta_cli::PcaApprox> pca_approx = opts.approx;

    if (opts.seed.has_value()) {
        const auto seed = static_cast<std::mt19937::result_type>(*opts.seed);
        gcta_eigh::shared_rng().seed(seed);
        LOGGER.i(0, "Using random seed: " + to_string(seed) + ".");
    }

    // ---- Sample set: full GRM ID list, optionally filtered by --keep/--remove ----
    const vector<string> grm_ids = Pheno::read_sublist(grm_pfx + ".grm.id");

    vector<string> analysis_ids = grm_ids;  // preserves GRM file order, same as V1's _keep convention
    if (!opts.keep.empty()) {
        const vector<string> keep_ids = Pheno::read_sublist(opts.keep);
        const std::unordered_set<string> keep_set(keep_ids.begin(), keep_ids.end());
        vector<string> filtered;
        filtered.reserve(analysis_ids.size());
        for (const auto& id : analysis_ids)
            if (keep_set.count(id)) filtered.push_back(id);
        analysis_ids = std::move(filtered);
    }
    if (!opts.remove.empty()) {
        const vector<string> remove_ids = Pheno::read_sublist(opts.remove);
        const std::unordered_set<string> remove_set(remove_ids.begin(), remove_ids.end());
        vector<string> filtered;
        filtered.reserve(analysis_ids.size());
        for (const auto& id : analysis_ids)
            if (!remove_set.count(id)) filtered.push_back(id);
        analysis_ids = std::move(filtered);
    }
    const int n = static_cast<int>(analysis_ids.size());
    if (n == 0)
        LOGGER.e(0, "--pca: no individuals remain after --keep/--remove filtering.");

    int out_pc_num = opts.pcs;
    if (out_pc_num < 0)
        LOGGER.e(0, "--pca requires N >= 0 (0 means all, matching V1 semantics).");
    if (out_pc_num == 0 || out_pc_num > n) out_pc_num = n;

    const double grm_chunked_budget = opts.grm_chunked_budget.value_or(0.0);
    bool grm_chunked = grm_chunked_budget > 0.0;
    bool grm_fp16 = opts.grm_fp16;
    gcta_cli::EigenvecFormat evec_fmt = opts.eigenvec_format.value_or(gcta_cli::EigenvecFormat::Text);
    const int svd_power_iter = opts.svd_power_iter.value_or(3);

    if (opts.fast) {
        vector<string> fast_notes;
        if (opts.pcs == 0 || out_pc_num == n) {
            LOGGER.w(0, "[--fast] requires --pcs <N> (with N < n=" + to_string(n) +
                        ") for rSVD and FP16 GRM; falling back to exact full eigendecomposition.");
        } else {
            if (!pca_approx.has_value()) {
                pca_approx = gcta_cli::PcaApprox::rSVD;
                fast_notes.push_back("pca-approx=rSVD");
            }
            if (!grm_chunked && !grm_fp16) {
                grm_fp16 = true;
                fast_notes.push_back("grm-fp16=on");
            }
        }
        if (!opts.eigenvec_format.has_value()) {
            evec_fmt = gcta_cli::EigenvecFormat::Bin16;
            fast_notes.push_back("eigenvec-format=bin16");
        }
        if (!fast_notes.empty()) {
            string summary;
            for (size_t i = 0; i < fast_notes.size(); ++i) {
                if (i > 0) summary += ", ";
                summary += fast_notes[i];
            }
            LOGGER.i(0, "[--fast] enabled: " + summary + ".");
        }
    }

    if (out_pc_num == n && pca_approx.has_value()) {
        LOGGER.w(0, "--pca-approx is set, but all PCs requested. Falling back to full eigenvalue decomposition.");
        pca_approx.reset();
    }

    if (grm_chunked && grm_fp16)
        LOGGER.e(0, "--grm-fp16 and --grm-chunked-budget are mutually exclusive.");
    if (!pca_approx.has_value() && (grm_chunked || grm_fp16))
        LOGGER.e(0, "--pca exact mode (dsyevr/dsyevd) requires dense GRM loading. "
                    "Remove --grm-chunked-budget/--grm-fp16, or set --pca-approx to Lanczos/rSVD.");

    const auto resolved_grm = gcta_grm_io::resolve_grm_input(grm_pfx, grm_fp16);
    if (resolved_grm.format == gcta_grm_io::GrmDiskFormat::F16 &&
        !grm_chunked && !grm_fp16 && pca_approx.has_value()) {
        grm_fp16 = true;
        LOGGER.i(0, "[auto-detect] Found [" + resolved_grm.data_path + "]; enabling --grm-fp16.");
    }

    int chunk_size = 0;
    gcta_grm_io::ChunkedGrmHandle chunked_handle;  // only populated when grm_chunked
    gcta_grm_io::PackedGrmFp16Handle fp16_handle;  // only populated when grm_fp16
    if (grm_chunked) {
        chunked_handle = gcta_grm_io::make_chunked_grm_reader(
            grm_pfx, analysis_ids, grm_chunked_budget, /*k_ext=*/0, "--pca",
            /*need_m_snps=*/false, &grm_ids);
        chunk_size = chunked_handle.chunk_rows;
        if (chunk_size >= n) {
            LOGGER.w(0, "--grm-chunked-budget=" + to_string(grm_chunked_budget) +
                        "GB covers the full GRM (n=" + to_string(n) + ") in a single chunk. "
                        "Falling back to dense loading instead of paying chunking overhead "
                        "for no benefit.");
            chunked_handle = {};
            grm_chunked = false;
        }
    }

    LOGGER.i(0, "Running PCA v2 for " + to_string(out_pc_num) + " PC(s) from " + to_string(n) + " individuals");

    // ---- GRM access: chunked tile reader, packed fp16, or dense ----
    gcta_chunked::TileReader chunked_reader;
    gcta_chunked::TileReaderF32 chunked_reader_f32;
    Eigen::MatrixXd G_dense;      // populated only for exact mode (!grm_chunked && !pca_approx.has_value())
    Eigen::MatrixXf G_dense_f32;  // populated for --pca-approx when !grm_chunked

    double trace_total = 0.0;
    bool have_trace_total = false;

    if (grm_fp16) {
        fp16_handle = gcta_grm_io::load_packed_grm_fp16(
            grm_pfx, analysis_ids, /*need_m_snps=*/false, "--grm-fp16",
            (1ull << 30), &grm_ids);
        chunk_size = fp16_handle.chunk_rows;
        chunked_reader = std::move(fp16_handle.reader);
        chunked_reader_f32 = std::move(fp16_handle.reader_f32);
        trace_total = fp16_handle.trace_K;
        have_trace_total = true;
        grm_chunked = true;
    } else if (grm_chunked) {
        chunked_reader = std::move(chunked_handle.reader);
        chunked_reader_f32 = std::move(chunked_handle.reader_f32);
        chunked_handle.file.reset();  // reader closures own the shared_ptr
    } else if (pca_approx.has_value()) {
        double m_snps_unused = -1.0;
        gcta_grm_io::load_and_subset_grm_upper(
            grm_pfx, analysis_ids, G_dense_f32, m_snps_unused, "--pca",
            (1ull << 30), &trace_total, nullptr, &grm_ids);
        have_trace_total = true;
    } else {
        double m_snps_unused = -1.0;
        gcta_grm_io::load_and_subset_grm_upper(
            grm_pfx, analysis_ids, G_dense, m_snps_unused, "--pca",
            (1ull << 30), &trace_total, nullptr, &grm_ids);
        have_trace_total = true;
    }

    if (have_trace_total) {
        LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));
    } else if (pca_approx != gcta_cli::PcaApprox::rSVD) {
        // Direct diagonal pread: reads only n floats, zero off-diagonal I/O.
        trace_total = gcta_chunked::chunked_diagonal(chunked_reader, n, chunk_size).sum();
        have_trace_total = true;
        LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));
    }

    auto apply_f32 = [&](const Eigen::Ref<const Eigen::MatrixXf>& X, bool exact_fp32) -> Eigen::MatrixXf {
        const bool use_bf16 = !exact_fp32 && gcta_chunked::default_bf16_sketch();
        if (grm_chunked) {
            double* tr_ptr = have_trace_total ? nullptr : &trace_total;
            Eigen::MatrixXf Y = gcta_chunked::chunked_symmetric_matvec_f32(
                chunked_reader_f32, n, chunk_size, X, tr_ptr, nullptr, use_bf16);
            if (!have_trace_total) {
                have_trace_total = true;
                LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));
            }
            return Y;
        } else {
            return gcta_chunked::dense_upper_symm_f32(G_dense_f32, X, use_bf16);
        }
    };

    auto apply_f64 = [&](const Eigen::Ref<const Eigen::VectorXd>& x) -> Eigen::VectorXd {
        if (grm_chunked)
            return gcta_chunked::chunked_symmetric_matvec(chunked_reader, n, chunk_size, x);
        else
            return gcta_chunked::dense_upper_symm_f32(G_dense_f32, x.cast<float>(), false).cast<double>();
    };

    Eigen::VectorXd eval;
    Eigen::MatrixXf evec_f;       // populated by rSVD (stays float32: 3x less RAM, faster write)
    Eigen::MatrixXd evec_d;       // populated by Lanczos / exact dsyevr
    bool used_dsyevd = false;
    bool has_evec_f  = false;
    const double* raw_evec = nullptr;

    try {
        const bool pca_nystrom = opts.svd_method == gcta_cli::SvdMethod::Nystrom;
        if (pca_approx.has_value()) {
            if (*pca_approx == gcta_cli::PcaApprox::rSVD) {
                const int oversample = gcta_eigh::recommended_oversample(out_pc_num);
                gcta_eigh::EighResultF res_f;
                if (pca_nystrom) {
                    LOGGER.i(0, "--pca rSVD + --svd-method nystrom: signed single-pass sketch (1 GRM matvec). Accuracy near PC " +
                                std::to_string(out_pc_num) + " may be lower than power-iteration rSVD.");
                    res_f = gcta_eigh::nystrom_symmetric_eigh<float>(apply_f32, n, out_pc_num, oversample);
                } else {
                    res_f = gcta_eigh::randomized_symmetric_eigh<float>(apply_f32, n, out_pc_num, oversample, svd_power_iter);
                }
                G_dense_f32.resize(0, 0);
                eval = std::move(res_f.eigenvalues);
                evec_f = std::move(res_f.eigenvectors);
                has_evec_f = true;
            } else {
                const int ncv = std::min(n, std::max(3 * out_pc_num + 1, 30));
                gcta_eigh::EighResult res = gcta_eigh::lanczos_symmetric_eigh(apply_f64, n, out_pc_num, ncv);
                G_dense_f32.resize(0, 0);
                eval = std::move(res.eigenvalues);
                evec_d = std::move(res.eigenvectors);
            }
        } else {
            if (pca_nystrom)
                LOGGER.w(0, "--svd-method is ignored in exact mode (it only applies to --pca-approx rSVD).");

            double* grm_ptr = G_dense.data();
            if (out_pc_num == n) {
                Eigen::VectorXd w(n);
                const int info = gcta_dsyevd(static_cast<gcta_blas_int>(n), grm_ptr,
                                             static_cast<gcta_blas_int>(n), w.data());
                if (info != 0)
                    LOGGER.e(0, "dsyevd failed (info=" + std::to_string(info) +
                                "). For n > 32766, try --pca-approx.");
                eval = w.reverse();
                used_dsyevd = true;
                raw_evec = grm_ptr;
            } else {
                Eigen::VectorXd w(out_pc_num);
                Eigen::MatrixXd Z(n, out_pc_num);
                gcta_blas_int m_found = 0;
                std::vector<gcta_blas_int> isuppz(2 * static_cast<size_t>(out_pc_num));
                const gcta_blas_int il = static_cast<gcta_blas_int>(n - out_pc_num + 1);
                const gcta_blas_int iu = static_cast<gcta_blas_int>(n);
                const int info = gcta_dsyevr(static_cast<gcta_blas_int>(n), grm_ptr,
                                             static_cast<gcta_blas_int>(n),
                                             il, iu, &m_found,
                                             w.data(), Z.data(), static_cast<gcta_blas_int>(n),
                                             isuppz.data());
                if (info != 0)
                    LOGGER.e(0, "dsyevr failed (info=" + std::to_string(info) + ").");
                if (m_found != static_cast<gcta_blas_int>(out_pc_num))
                    LOGGER.e(0, "dsyevr returned " + std::to_string(m_found) +
                                " eigenvalues, expected " + std::to_string(out_pc_num) + ".");
                eval = w.reverse();
                evec_d.resize(n, out_pc_num);
                #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(out_pc_num) >= 65536)
                for (int j = 0; j < out_pc_num; ++j) {
                    std::memcpy(evec_d.col(j).data(), Z.col(out_pc_num - 1 - j).data(),
                                static_cast<size_t>(n) * sizeof(double));
                }
            }
        }
    } catch (const std::exception& e) {
        LOGGER.e(0, string("--pca failed: ") + e.what());
    }

    // Release GRM storage and reader handles before writing output files.
    if (!used_dsyevd) G_dense.resize(0, 0);
    chunked_reader.release();
    chunked_reader = nullptr;
    chunked_reader_f32 = nullptr;
    chunked_handle.file.reset();
    fp16_handle.file.reset();

    // ---- Write .eigenval / .eigenvec / .eigenvec.bin ----
    const string eval_file = out_prefix + ".eigenval";
    std::ofstream o_eval(eval_file);
    if (!o_eval) LOGGER.e(0, "cannot open the file [" + eval_file + "] to write.");
    for (int i = 0; i < out_pc_num; ++i) o_eval << eval(i) << "\n";
    o_eval.close();
    LOGGER.i(0, "Eigenvalues of " + to_string(n) + " individuals have been saved in [" + eval_file + "].");

    const bool write_text = (evec_fmt == gcta_cli::EigenvecFormat::Text ||
                             evec_fmt == gcta_cli::EigenvecFormat::Both);
    const bool write_bin  = (evec_fmt == gcta_cli::EigenvecFormat::Bin32 ||
                             evec_fmt == gcta_cli::EigenvecFormat::Bin16 ||
                             evec_fmt == gcta_cli::EigenvecFormat::Both);

    if (write_bin) {
        const uint32_t bin_dtype = (evec_fmt == gcta_cli::EigenvecFormat::Bin16) ? 1u : 0u;
        const string bin_file = out_prefix + ".eigenvec.bin";

        string ids_blob;
        size_t est_id_bytes = 0;
        for (const auto& id : analysis_ids) est_id_bytes += id.size() + 1;
        ids_blob.reserve(est_id_bytes);
        for (const auto& id : analysis_ids) {
            ids_blob.append(id);
            ids_blob.push_back('\n');
        }

        EvecHeader hdr{};
        hdr.version  = 1u;
        hdr.dtype    = bin_dtype;
        hdr.n        = n;
        hdr.k        = out_pc_num;
        hdr.id_hash  = gcta_io::hash_sample_ids(analysis_ids);
        hdr.id_bytes = static_cast<uint64_t>(ids_blob.size());

        std::vector<float> col_scale;
        std::vector<float> col_inv_scale;
        if (bin_dtype == 1u) {
            col_scale.assign(static_cast<size_t>(out_pc_num), 0.0f);
            col_inv_scale.assign(static_cast<size_t>(out_pc_num), 0.0f);
            const double* const d_src_base = used_dsyevd ? raw_evec : evec_d.data();
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(n) * static_cast<size_t>(out_pc_num) >= 65536)
            for (int j = 0; j < out_pc_num; ++j) {
                float max_abs = 0.0f;
                if (has_evec_f) {
                    const float* const col = evec_f.col(j).data();
                    for (int i = 0; i < n; ++i) {
                        const float a = std::fabs(col[i]);
                        if (a > max_abs) max_abs = a;
                    }
                } else {
                    const int src_col = used_dsyevd ? (out_pc_num - 1 - j) : j;
                    const double* const col = d_src_base + static_cast<size_t>(src_col) * static_cast<size_t>(n);
                    for (int i = 0; i < n; ++i) {
                        const float a = std::fabs(static_cast<float>(col[i]));
                        if (a > max_abs) max_abs = a;
                    }
                }
                if (max_abs > 0.0f) {
                    col_scale[static_cast<size_t>(j)]     = max_abs / 32767.0f;
                    col_inv_scale[static_cast<size_t>(j)] = 32767.0f / max_abs;
                }
            }
        }

        constexpr size_t kBinSlotBytes = 32ULL << 20; // 32 MiB double-buffered
        const size_t elem_bytes = (bin_dtype == 0u) ? sizeof(float) : sizeof(int16_t);
        const size_t col_bytes  = static_cast<size_t>(n) * elem_bytes;
        const int cols_per_slot = static_cast<int>(std::max<size_t>(1, kBinSlotBytes / std::max<size_t>(1, col_bytes)));
        const double* const d_src_base = used_dsyevd ? raw_evec : evec_d.data();

        try {
            gcta_io::AsyncSeqWriter writer(bin_file, kBinSlotBytes);
            AlignedRawBuffer<char, 64>* meta_slot = nullptr;
            size_t meta_used = 0;
            append_bytes_buffered(writer, meta_slot, meta_used, kBinSlotBytes,
                                  reinterpret_cast<const char*>(&hdr), sizeof(hdr));
            if (!ids_blob.empty()) {
                append_bytes_buffered(writer, meta_slot, meta_used, kBinSlotBytes,
                                      ids_blob.data(), ids_blob.size());
            }
            if (bin_dtype == 1u) {
                append_bytes_buffered(writer, meta_slot, meta_used, kBinSlotBytes,
                                      reinterpret_cast<const char*>(col_scale.data()),
                                      col_scale.size() * sizeof(float));
            }
            if (meta_slot != nullptr && meta_used > 0) {
                writer.submit_slot(meta_used);
                meta_slot = nullptr;
                meta_used = 0;
            }

            // Stream n x k ColMajor eigenvectors directly (ZERO transpose).
            for (int j0 = 0; j0 < out_pc_num; j0 += cols_per_slot) {
                const int j1   = std::min(out_pc_num, j0 + cols_per_slot);
                const int jlen = j1 - j0;
                const size_t bytes = static_cast<size_t>(jlen) * col_bytes;
                AlignedRawBuffer<char, 64>& bslot = writer.acquire_slot();
                if (bslot.size() < bytes) bslot.resize(bytes);

                if (bin_dtype == 0u) {
                    float* const dst_base = reinterpret_cast<float*>(bslot.data());
                    if (has_evec_f) {
                        if (jlen >= 4 && bytes >= (1u << 20)) {
                            #pragma omp parallel for schedule(static)
                            for (int dj = 0; dj < jlen; ++dj) {
                                std::memcpy(dst_base + static_cast<size_t>(dj) * static_cast<size_t>(n),
                                            evec_f.col(j0 + dj).data(),
                                            static_cast<size_t>(n) * sizeof(float));
                            }
                        } else {
                            std::memcpy(dst_base, evec_f.col(j0).data(), bytes);
                        }
                    } else if (jlen >= 4) {
                        #pragma omp parallel for schedule(static) if (bytes >= 65536)
                        for (int dj = 0; dj < jlen; ++dj) {
                            const int src_col = used_dsyevd ? (out_pc_num - 1 - (j0 + dj)) : (j0 + dj);
                            const double* const src = d_src_base + static_cast<size_t>(src_col) * static_cast<size_t>(n);
                            float* const dst = dst_base + static_cast<size_t>(dj) * static_cast<size_t>(n);
                            #pragma omp simd
                            for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
                        }
                    } else {
                        for (int dj = 0; dj < jlen; ++dj) {
                            const int src_col = used_dsyevd ? (out_pc_num - 1 - (j0 + dj)) : (j0 + dj);
                            const double* const src = d_src_base + static_cast<size_t>(src_col) * static_cast<size_t>(n);
                            float* const dst = dst_base + static_cast<size_t>(dj) * static_cast<size_t>(n);
                            #pragma omp parallel for schedule(static) if (n >= 32768)
                            for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
                        }
                    }
                } else {
                    int16_t* const dst_base = reinterpret_cast<int16_t*>(bslot.data());
                    auto quantize_col = [&](int dj) {
                        const int j = j0 + dj;
                        const float inv_s = col_inv_scale[static_cast<size_t>(j)];
                        int16_t* const dst = dst_base + static_cast<size_t>(dj) * static_cast<size_t>(n);
                        if (inv_s == 0.0f) {
                            std::memset(dst, 0, static_cast<size_t>(n) * sizeof(int16_t));
                            return;
                        }
                        if (has_evec_f) {
                            const float* const src = evec_f.col(j).data();
                            for (int i = 0; i < n; ++i) {
                                const int32_t q = std::clamp(
                                    static_cast<int32_t>(std::nearbyintf(src[i] * inv_s)),
                                    -32767, 32767);
                                dst[i] = static_cast<int16_t>(q);
                            }
                        } else {
                            const int src_col = used_dsyevd ? (out_pc_num - 1 - j) : j;
                            const double* const src = d_src_base + static_cast<size_t>(src_col) * static_cast<size_t>(n);
                            for (int i = 0; i < n; ++i) {
                                const int32_t q = std::clamp(
                                    static_cast<int32_t>(std::nearbyintf(static_cast<float>(src[i]) * inv_s)),
                                    -32767, 32767);
                                dst[i] = static_cast<int16_t>(q);
                            }
                        }
                    };
                    if (jlen >= 4) {
                        #pragma omp parallel for schedule(static) if (bytes >= 65536)
                        for (int dj = 0; dj < jlen; ++dj) quantize_col(dj);
                    } else {
                        for (int dj = 0; dj < jlen; ++dj) {
                            const int j = j0 + dj;
                            const float inv_s = col_inv_scale[static_cast<size_t>(j)];
                            int16_t* const dst = dst_base + static_cast<size_t>(dj) * static_cast<size_t>(n);
                            if (inv_s == 0.0f) {
                                std::memset(dst, 0, static_cast<size_t>(n) * sizeof(int16_t));
                            } else if (has_evec_f) {
                                const float* const src = evec_f.col(j).data();
                                #pragma omp parallel for schedule(static) if (n >= 32768)
                                for (int i = 0; i < n; ++i) {
                                    const int32_t q = std::clamp(
                                        static_cast<int32_t>(std::nearbyintf(src[i] * inv_s)),
                                        -32767, 32767);
                                    dst[i] = static_cast<int16_t>(q);
                                }
                            } else {
                                const int src_col = used_dsyevd ? (out_pc_num - 1 - j) : j;
                                const double* const src = d_src_base + static_cast<size_t>(src_col) * static_cast<size_t>(n);
                                #pragma omp parallel for schedule(static) if (n >= 32768)
                                for (int i = 0; i < n; ++i) {
                                    const int32_t q = std::clamp(
                                        static_cast<int32_t>(std::nearbyintf(static_cast<float>(src[i]) * inv_s)),
                                        -32767, 32767);
                                    dst[i] = static_cast<int16_t>(q);
                                }
                            }
                        }
                    }
                }
                writer.submit_slot(bytes);
            }
            writer.finish();
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }

        const size_t total_bin_bytes = sizeof(EvecHeader) + ids_blob.size() +
            (bin_dtype == 1u ? col_scale.size() * sizeof(float) : 0u) +
            static_cast<size_t>(out_pc_num) * col_bytes;
        LOGGER.i(0, to_string(n) + " individuals, " + to_string(out_pc_num) +
                    " eigenvectors saved in binary EVEC (" +
                    (bin_dtype == 1u ? "bin16" : "float32") + ") [" + bin_file + "] (" +
                    std::format("{:.3f}", static_cast<double>(total_bin_bytes) / 1e9) + " GB).");
    }

    if (write_text) {
        const string evec_file = out_prefix + ".eigenvec";

        auto append_id = [](string& buf, const string& raw_id) {
            const size_t tab = raw_id.find('\t');
            if (tab == string::npos) {
                buf.append(raw_id);
            } else {
                buf.append(raw_id.data(), tab);
                buf.push_back(' ');
                buf.append(raw_id.data() + tab + 1, raw_id.size() - tab - 1);
            }
        };

        // Format eigenvectors directly in parallel row chunks (L1-tiled transpose +
        // 2x16 MiB AsyncSeqWriter overlapping sequential I/O with formatting).
        const int max_threads = std::max(1, omp_get_max_threads());
        const int rows_per_chunk = std::clamp(
            static_cast<int>((4 * 1024 * 1024) / std::max(1, out_pc_num)),
            64, 4096);
        const bool use_slab_transpose = (out_pc_num >= 64);
        Eigen::MatrixXf slab_f;
        Eigen::MatrixXd slab_d;
        std::vector<string> thread_bufs(static_cast<size_t>(max_threads));

        constexpr size_t kTextSlotBytes = 16ULL << 20; // 16 MiB double-buffered
        try {
            gcta_io::AsyncSeqWriter writer(evec_file, kTextSlotBytes);
            AlignedRawBuffer<char, 64>* out_slot = nullptr;
            size_t out_used = 0;

            for (int i0 = 0; i0 < n; i0 += rows_per_chunk) {
                const int rows_c = std::min(rows_per_chunk, n - i0);

                if (use_slab_transpose) {
                    if (has_evec_f) {
                        constexpr int kB = 64;
                        slab_f.resize(out_pc_num, rows_c);
                        const float* src = evec_f.data();
                        float* dst = slab_f.data();
                        #pragma omp parallel for collapse(2) schedule(static)
                        for (int r0 = 0; r0 < rows_c; r0 += kB) {
                            for (int j0 = 0; j0 < out_pc_num; j0 += kB) {
                                const int r_len = std::min(kB, rows_c - r0);
                                const int j_len = std::min(kB, out_pc_num - j0);
                                alignas(64) float tile[kB][kB + 4];
                                for (int dj = 0; dj < j_len; ++dj) {
                                    const float* scol = src +
                                        static_cast<size_t>(j0 + dj) * static_cast<size_t>(n) +
                                        static_cast<size_t>(i0 + r0);
                                    for (int dr = 0; dr < r_len; ++dr) {
                                        tile[dr][dj] = scol[dr];
                                    }
                                }
                                for (int dr = 0; dr < r_len; ++dr) {
                                    std::memcpy(dst + static_cast<size_t>(r0 + dr) * static_cast<size_t>(out_pc_num) +
                                                static_cast<size_t>(j0),
                                                tile[dr], static_cast<size_t>(j_len) * sizeof(float));
                                }
                            }
                        }
                    } else {
                        constexpr int kB = 32; // 32 * 36 * 8B = 9 KiB (L1D-resident on all CPUs)
                        slab_d.resize(out_pc_num, rows_c);
                        const double* src = used_dsyevd ? raw_evec : evec_d.data();
                        double* dst = slab_d.data();
                        #pragma omp parallel for collapse(2) schedule(static)
                        for (int r0 = 0; r0 < rows_c; r0 += kB) {
                            for (int j0 = 0; j0 < out_pc_num; j0 += kB) {
                                const int r_len = std::min(kB, rows_c - r0);
                                const int j_len = std::min(kB, out_pc_num - j0);
                                alignas(64) double tile[kB][kB + 4];
                                for (int dj = 0; dj < j_len; ++dj) {
                                    const int src_col = used_dsyevd ? (out_pc_num - 1 - (j0 + dj)) : (j0 + dj);
                                    const double* scol = src +
                                        static_cast<size_t>(src_col) * static_cast<size_t>(n) +
                                        static_cast<size_t>(i0 + r0);
                                    for (int dr = 0; dr < r_len; ++dr) {
                                        tile[dr][dj] = scol[dr];
                                    }
                                }
                                for (int dr = 0; dr < r_len; ++dr) {
                                    std::memcpy(dst + static_cast<size_t>(r0 + dr) * static_cast<size_t>(out_pc_num) +
                                                static_cast<size_t>(j0),
                                                tile[dr], static_cast<size_t>(j_len) * sizeof(double));
                                }
                            }
                        }
                    }
                }

                const int active_threads = std::min(max_threads, std::max(1, rows_c));
                for (int t = 0; t < active_threads; ++t) {
                    thread_bufs[static_cast<size_t>(t)].clear();
                }
                int actual_nt = 1;
                #pragma omp parallel num_threads(active_threads)
                {
                    const int tid = omp_get_thread_num();
                    const int nt  = omp_get_num_threads();
                    #pragma omp single nowait
                    { actual_nt = nt; }
                    const int r_lo = (rows_c * tid) / nt;
                    const int r_hi = (rows_c * (tid + 1)) / nt;
                    string& local = thread_bufs[static_cast<size_t>(tid)];
                    local.clear();
                    const size_t est_bytes = static_cast<size_t>(r_hi - r_lo) *
                        (32u + static_cast<size_t>(out_pc_num) * (has_evec_f ? 14u : 22u));
                    if (local.capacity() < est_bytes) local.reserve(est_bytes);

                    char tmp[32];
                    for (int r = r_lo; r < r_hi; ++r) {
                        const int gi = i0 + r;
                        append_id(local, analysis_ids[static_cast<size_t>(gi)]);
                        if (has_evec_f) {
                            if (use_slab_transpose) {
                                const float* row = slab_f.col(r).data();
                                for (int j = 0; j < out_pc_num; ++j) {
                                    local.push_back(' ');
                                    const auto res = std::to_chars(tmp, tmp + sizeof(tmp), row[j]);
                                    local.append(tmp, static_cast<size_t>(res.ptr - tmp));
                                }
                            } else {
                                for (int j = 0; j < out_pc_num; ++j) {
                                    local.push_back(' ');
                                    const auto res = std::to_chars(tmp, tmp + sizeof(tmp), evec_f(gi, j));
                                    local.append(tmp, static_cast<size_t>(res.ptr - tmp));
                                }
                            }
                        } else {
                            if (use_slab_transpose) {
                                const double* row = slab_d.col(r).data();
                                for (int j = 0; j < out_pc_num; ++j) {
                                    local.push_back(' ');
                                    const auto res = std::to_chars(tmp, tmp + sizeof(tmp), row[j]);
                                    local.append(tmp, static_cast<size_t>(res.ptr - tmp));
                                }
                            } else {
                                for (int j = 0; j < out_pc_num; ++j) {
                                    local.push_back(' ');
                                    const double v = used_dsyevd
                                        ? raw_evec[static_cast<size_t>(out_pc_num - 1 - j) * static_cast<size_t>(n) + static_cast<size_t>(gi)]
                                        : evec_d(gi, j);
                                    const auto res = std::to_chars(tmp, tmp + sizeof(tmp), v);
                                    local.append(tmp, static_cast<size_t>(res.ptr - tmp));
                                }
                            }
                        }
                        local.push_back('\n');
                    }
                }

                append_thread_bufs_buffered(writer, out_slot, out_used, kTextSlotBytes,
                                            thread_bufs, actual_nt);
            }

            if (out_slot != nullptr && out_used > 0) {
                writer.submit_slot(out_used);
                out_slot = nullptr;
                out_used = 0;
            }
            writer.finish();
        } catch (const std::exception& e) {
            LOGGER.e(0, e.what());
        }

        LOGGER.i(0, to_string(n) + " individuals, " + to_string(out_pc_num) +
                    " eigenvectors saved in [" + evec_file + "].");
    }

    if (used_dsyevd) G_dense.resize(0, 0);
}
