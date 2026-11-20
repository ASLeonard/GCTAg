#include "grm_binary_io.hpp"
#include "chunked_grm_matvec.hpp"
#include "PCA_stream.h"
#include "symmetric_eigendecomp.hpp"
#include <cpu.h>
#include <mem.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <unordered_set>
#include <string>
#include <vector>
#include <omp.h>

using std::string;
using std::to_string;
using std::vector;

void PCAStream::run(const gcta_cli::PcaOptions& opts, const std::string& out_prefix)
{
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

    if (out_pc_num == n && pca_approx.has_value()) {
        LOGGER.w(0, "--pca-approx is set, but all PCs requested. Falling back to full eigenvalue decomposition.");
        pca_approx.reset();
    }

    const double grm_chunked_budget = opts.grm_chunked_budget.value_or(0.0);
    bool grm_chunked = grm_chunked_budget > 0.0;
    const bool grm_fp16 = opts.grm_fp16;
    const int svd_power_iter = opts.svd_power_iter.value_or(3);

    if (grm_chunked && grm_fp16)
        LOGGER.e(0, "--grm-fp16 and --grm-chunked-budget are mutually exclusive.");
    if (!pca_approx.has_value() && (grm_chunked || grm_fp16))
        LOGGER.e(0, "--pca exact mode (dsyevr/dsyevd) requires dense GRM loading. "
                    "Remove --grm-chunked-budget/--grm-fp16, or set --pca-approx to Lanczos/rSVD.");

    int chunk_size = 0;
    gcta_grm_io::ChunkedGrmHandle chunked_handle;  // only populated when grm_chunked
    gcta_grm_io::PackedGrmFp16Handle fp16_handle;  // only populated when grm_fp16
    if (grm_chunked) {
        chunked_handle = gcta_grm_io::make_chunked_grm_reader(
            grm_pfx, analysis_ids, grm_chunked_budget, /*k_ext=*/0, "--pca",
            /*need_m_snps=*/false);
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
            grm_pfx, analysis_ids, /*need_m_snps=*/false, "--grm-fp16");
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
            (1ull << 30), &trace_total, nullptr);
        have_trace_total = true;
    } else {
        double m_snps_unused = -1.0;
        gcta_grm_io::load_and_subset_grm_upper(
            grm_pfx, analysis_ids, G_dense, m_snps_unused, "--pca",
            (1ull << 30), &trace_total, nullptr);
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

    // ---- Write .eigenval / .eigenvec ----
    const string eval_file = out_prefix + ".eigenval";
    std::ofstream o_eval(eval_file);
    if (!o_eval) LOGGER.e(0, "cannot open the file [" + eval_file + "] to write.");
    for (int i = 0; i < out_pc_num; ++i) o_eval << eval(i) << "\n";
    o_eval.close();
    LOGGER.i(0, "Eigenvalues of " + to_string(n) + " individuals have been saved in [" + eval_file + "].");

    const string evec_file = out_prefix + ".eigenvec";
    std::ofstream o_evec(evec_file, std::ios::binary);
    if (!o_evec) LOGGER.e(0, "cannot open the file [" + evec_file + "] to write.");

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

    // Format eigenvectors directly in parallel row chunks (no double widening or
    // full-matrix transpose copy; ~3x less peak RAM and ~10x faster I/O formatting).
    const int max_threads = std::max(1, omp_get_max_threads());
    const int rows_per_chunk = std::clamp(
        static_cast<int>((4 * 1024 * 1024) / std::max(1, out_pc_num)),
        64, 4096);
    const bool use_slab_transpose = (out_pc_num >= 64);
    Eigen::MatrixXf slab_f;
    Eigen::MatrixXd slab_d;
    std::vector<string> thread_bufs(static_cast<size_t>(max_threads));

    for (int i0 = 0; i0 < n; i0 += rows_per_chunk) {
        const int rows_c = std::min(rows_per_chunk, n - i0);

        if (use_slab_transpose) {
            constexpr int kB = 64;
            if (has_evec_f) {
                slab_f.resize(out_pc_num, rows_c);
                const float* src = evec_f.data();
                float* dst = slab_f.data();
                #pragma omp parallel for collapse(2) schedule(static)
                for (int r0 = 0; r0 < rows_c; r0 += kB) {
                    for (int j0 = 0; j0 < out_pc_num; j0 += kB) {
                        const int r1 = std::min(rows_c, r0 + kB);
                        const int j1 = std::min(out_pc_num, j0 + kB);
                        for (int r = r0; r < r1; ++r) {
                            float* dcol = dst + static_cast<size_t>(r) * static_cast<size_t>(out_pc_num);
                            const int gi = i0 + r;
                            for (int j = j0; j < j1; ++j) {
                                dcol[j] = src[gi + static_cast<size_t>(j) * static_cast<size_t>(n)];
                            }
                        }
                    }
                }
            } else {
                slab_d.resize(out_pc_num, rows_c);
                const double* src = used_dsyevd ? raw_evec : evec_d.data();
                double* dst = slab_d.data();
                #pragma omp parallel for collapse(2) schedule(static)
                for (int r0 = 0; r0 < rows_c; r0 += kB) {
                    for (int j0 = 0; j0 < out_pc_num; j0 += kB) {
                        const int r1 = std::min(rows_c, r0 + kB);
                        const int j1 = std::min(out_pc_num, j0 + kB);
                        for (int r = r0; r < r1; ++r) {
                            double* dcol = dst + static_cast<size_t>(r) * static_cast<size_t>(out_pc_num);
                            const int gi = i0 + r;
                            for (int j = j0; j < j1; ++j) {
                                const int src_col = used_dsyevd ? (out_pc_num - 1 - j) : j;
                                dcol[j] = src[gi + static_cast<size_t>(src_col) * static_cast<size_t>(n)];
                            }
                        }
                    }
                }
            }
        }

        const int active_threads = std::min(max_threads, std::max(1, (rows_c + 15) / 16));
        #pragma omp parallel num_threads(active_threads)
        {
            const int tid = omp_get_thread_num();
            const int nt  = omp_get_num_threads();
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

        for (int t = 0; t < active_threads; ++t) {
            const string& local = thread_bufs[static_cast<size_t>(t)];
            if (!local.empty()) {
                o_evec.write(local.data(), static_cast<std::streamsize>(local.size()));
            }
        }
    }

    o_evec.close();
    if (!o_evec) LOGGER.e(0, "failed while writing [" + evec_file + "].");
    LOGGER.i(0, to_string(n) + " individuals, " + to_string(out_pc_num) +
                " eigenvectors saved in [" + evec_file + "].");

    if (used_dsyevd) G_dense.resize(0, 0);
}
