#include "grm_binary_io.hpp"
#include "chunked_grm_matvec.hpp"
#include "PCA_stream.h"
#include "symmetric_eigendecomp.hpp"
#include <cpu.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <unordered_set>
#include <string>
#include <vector>

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

        // Row-chunk size for streaming reads off the GRM (chunked_symmetric_matvec),
        // and the reader's own row-band cache reservation, are now both solved
        // inside make_chunked_grm_reader (see grm_binary_io.hpp) rather than
        // computed here and passed in piecemeal -- this file never touches
        // RemlCtx, so it has no other reason to duplicate that arithmetic.
        int chunk_size = 0;
        gcta_grm_io::ChunkedGrmHandle chunked_handle;  // only populated when grm_chunked
        gcta_grm_io::PackedGrmFp16Handle fp16_handle;  // only populated when grm_fp16
        if (grm_chunked) {
            // NOTE: this now opens the file and ID-matches against
            // .grm.id/.grm.N.bin unconditionally whenever grm_chunked is
            // true, even if the chunk_size >= n check below ends up
            // falling back to dense -- previously that check ran first and
            // skipped opening the file entirely in that case. The extra
            // work is cheap (ID matching + one N-mean read, no tile reads
            // yet) but it's a real behavior change worth knowing about.
            chunked_handle = gcta_grm_io::make_chunked_grm_reader(
                grm_pfx, analysis_ids, grm_chunked_budget, /*k_ext=*/0, "--pca",
                /*need_m_snps=*/false);
            chunk_size = chunked_handle.chunk_rows;
            if (chunk_size >= n) {
                // The whole GRM fits in a single block: chunking then buys nothing
                // (there's no second chunk to defer loading) but still pays the
                // diagonal-tile mirror's transient 2x n x n duplication (see
                // chunked_grm_matvec.hpp) across the entire matrix at once, since
                // that tile IS the whole matrix here. Strictly worse than dense.
                LOGGER.w(0, "--grm-chunked-budget=" + to_string(grm_chunked_budget) +
                            "GB covers the full GRM (n=" + to_string(n) + ") in a single chunk. "
                            "Falling back to dense loading instead of paying chunking overhead "
                            "for no benefit.");
                grm_chunked = false;
            }
        }

        LOGGER.i(0, "Running PCA v2 for " + to_string(out_pc_num) + " PC(s) from " + to_string(n) + " individuals");

        // ---- GRM access: chunked tile reader, or dense (fallback / comparison) ----
        gcta_chunked::TileReader chunked_reader;
        gcta_chunked::TileReaderF32 chunked_reader_f32;
        Eigen::MatrixXd G_dense;      // populated only for exact mode (!grm_chunked && !pca_approx.has_value())
        Eigen::MatrixXf G_dense_f32;  // populated for --pca-approx when !grm_chunked

        if (grm_fp16) {
            fp16_handle = gcta_grm_io::load_packed_grm_fp16(
                grm_pfx, analysis_ids, /*need_m_snps=*/false, "--grm-fp16");
            chunk_size = fp16_handle.chunk_rows;
            chunked_reader = std::move(fp16_handle.reader);
            chunked_reader_f32 = std::move(fp16_handle.reader_f32);
            grm_chunked = true;
        } else if (grm_chunked) {
            // make_chunked_grm_reader already logged the budget/chunk_size
            // split above; nothing further to log here.
            chunked_reader = std::move(chunked_handle.reader);
            chunked_reader_f32 = std::move(chunked_handle.reader_f32);
        } else if (pca_approx.has_value()) {
            double m_snps_unused = -1.0;
            gcta_grm_io::load_and_subset_grm_upper(
                grm_pfx, analysis_ids, G_dense_f32, m_snps_unused, "--pca");
        } else {
            double m_snps_unused = -1.0;
            gcta_grm_io::load_and_subset_grm_upper(
                grm_pfx, analysis_ids, G_dense, m_snps_unused, "--pca");
        }

        double trace_total = 0.0;
        bool have_trace_total = false;
        if (grm_fp16) {
            trace_total = fp16_handle.trace_K;
            have_trace_total = true;
            LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));
        } else if (!grm_chunked) {
            if (G_dense_f32.size() > 0) trace_total = G_dense_f32.diagonal().cast<double>().sum();
            else if (G_dense.size() > 0) trace_total = G_dense.diagonal().sum();
            have_trace_total = true;
            LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));
        } else if (pca_approx != gcta_cli::PcaApprox::rSVD) {
            trace_total = gcta_chunked::chunked_diagonal(chunked_reader, n, chunk_size).sum();
            have_trace_total = true;
            LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));
        }

        auto apply_f32 = [&](const Eigen::Ref<const Eigen::MatrixXf>& X) -> Eigen::MatrixXf {
            if (grm_chunked) {
                double* tr_ptr = have_trace_total ? nullptr : &trace_total;
                Eigen::MatrixXf Y = gcta_chunked::chunked_symmetric_matvec_f32(
                    chunked_reader_f32, n, chunk_size, X, tr_ptr);
                if (!have_trace_total) {
                    have_trace_total = true;
                    LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));
                }
                return Y;
            } else {
                return G_dense_f32.selfadjointView<Eigen::Upper>() * X;
            }
        };

        auto apply_f64 = [&](const Eigen::Ref<const Eigen::VectorXd>& x) -> Eigen::VectorXd {
            if (grm_chunked)
                return gcta_chunked::chunked_symmetric_matvec(chunked_reader, n, chunk_size, x);
            else
                return (G_dense_f32.selfadjointView<Eigen::Upper>() * x.cast<float>()).cast<double>();
        };

        Eigen::VectorXd eval;
        Eigen::MatrixXd evec;
        bool used_dsyevd = false;
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
                    evec = res_f.eigenvectors.cast<double>();
                } else {
                    const int ncv = std::min(n, std::max(3 * out_pc_num + 1, 30));
                    gcta_eigh::EighResult res = gcta_eigh::lanczos_symmetric_eigh(apply_f64, n, out_pc_num, ncv);
                    G_dense_f32.resize(0, 0);
                    eval = std::move(res.eigenvalues);
                    evec = std::move(res.eigenvectors);
                }
            } else {
                if (pca_nystrom)
                    LOGGER.w(0, "--svd-method is ignored in exact mode (it only applies to --pca-approx rSVD).");

                double* grm_ptr = G_dense.data();
                // No selfadjointView/triangle change needed here: gcta_dsyevd
                // and gcta_dsyevr (cpu.h) both hardcode uplo='U', so they
                // only ever read the upper triangle of grm_ptr on entry --
                // true before and after G_dense became upper-triangle-only.
                // On exit (jobz='V') they overwrite the whole buffer with
                // dense eigenvectors, which is why raw_evec = grm_ptr below
                // is valid regardless of the input triangle convention.
                if (out_pc_num == n) {
                    Eigen::VectorXd w(n);
                    const int info = gcta_dsyevd((gcta_blas_int)n, grm_ptr, (gcta_blas_int)n, w.data());
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
                    std::vector<gcta_blas_int> isuppz(2 * out_pc_num);
                    const gcta_blas_int il = (gcta_blas_int)(n - out_pc_num + 1);
                    const gcta_blas_int iu = (gcta_blas_int)n;
                    const int info = gcta_dsyevr((gcta_blas_int)n, grm_ptr, (gcta_blas_int)n,
                                                 il, iu, &m_found,
                                                 w.data(), Z.data(), (gcta_blas_int)n,
                                                 isuppz.data());
                    if (info != 0)
                        LOGGER.e(0, "dsyevr failed (info=" + std::to_string(info) + ").");
                    if (m_found != (gcta_blas_int)out_pc_num)
                        LOGGER.e(0, "dsyevr returned " + std::to_string(m_found) +
                                    " eigenvalues, expected " + std::to_string(out_pc_num) + ".");
                    eval = w.reverse();
                    evec = Z(Eigen::placeholders::all, Eigen::seq(out_pc_num - 1, 0, -1));
                }
            }
        } catch (const std::exception& e) {
            LOGGER.e(0, string("--pca failed: ") + e.what());
        }

        // Neither is needed past this point — G_dense (up to n x n) and the
        // chunked reader's mmap can both be released before the O(n) file
        // writes below, rather than sitting at peak size through them.
        if (!used_dsyevd) G_dense.resize(0, 0);
        chunked_reader = nullptr;
        chunked_reader_f32 = nullptr;
        fp16_handle.file.reset();

        // ---- Write .eigenval / .eigenvec — identical format to gcta::pca()'s writer ----
        auto append_double = [](string &buf, double v) {
            char tmp[32];
            buf.append(tmp, std::to_chars(tmp, tmp + sizeof(tmp), v).ptr);
        };

        const string eval_file = out_prefix + ".eigenval";
        std::ofstream o_eval(eval_file);
        if (!o_eval) LOGGER.e(0, "cannot open the file [" + eval_file + "] to write.");
        for (int i = 0; i < out_pc_num; ++i) o_eval << eval(i) << "\n";
        o_eval.close();
        LOGGER.i(0, "Eigenvalues of " + to_string(n) + " individuals have been saved in [" + eval_file + "].");

        const string evec_file = out_prefix + ".eigenvec";
        std::ofstream o_evec(evec_file);
        if (!o_evec) LOGGER.e(0, "cannot open the file [" + evec_file + "] to write.");

        // evec is n x out_pc_num, column-major: reading it row by row strides
        // by n elements between each PC value. Materialize transpose once so
        // each individual's PCs are contiguous for the output loop.
        const Eigen::MatrixXd evec_t = used_dsyevd ? Eigen::MatrixXd() : evec.transpose();

        string buf;
        buf.reserve(1 << 22);
        for (int i = 0; i < n; ++i) {
            // analysis_ids[i] is "FID\tIID" (Pheno::read_sublist convention,
            // same as match_ids_to_grm's expected input elsewhere); V1's
            // writer joins FID/IID with a space, so convert to match exactly.
            string id = analysis_ids[i];
            const size_t tab = id.find('\t');
            if (tab != string::npos) id[tab] = ' ';
            buf += id;
            for (int j = 0; j < out_pc_num; ++j) {
                buf += ' ';
                append_double(buf, used_dsyevd ? raw_evec[(size_t)(out_pc_num - 1 - j) * n + i]
                                                : evec_t(j, i));
            }
            buf += '\n';
            if (buf.size() >= (1 << 21)) { o_evec.write(buf.data(), (std::streamsize)buf.size()); buf.clear(); }
        }
        o_evec.write(buf.data(), (std::streamsize)buf.size());
        o_evec.close();
        LOGGER.i(0, to_string(n) + " individuals, " + to_string(out_pc_num) +
                    " eigenvectors saved in [" + evec_file + "].");

    if (used_dsyevd) G_dense.resize(0, 0);
}