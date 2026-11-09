#include "grm_binary_io.hpp"
#include "chunked_grm_matvec.hpp"
#include "PCA_stream.h"
#include "symmetric_eigendecomp.hpp"
#include <cpu.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <unordered_set>
#include <map>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::to_string;
using std::vector;

map<string, string> PCAStream::options;
map<string, double> PCAStream::options_d;
vector<string> PCAStream::processFunctions;

int PCAStream::registerOption(map<string, vector<string>>& options_in)
{
    // Always capture --out (shared flag)
    // Use the pre-processed "out" key (no dashes), which is always set by main.cpp
    if (!options_in["out"].empty())
        options["out"] = options_in["out"][0];

    const bool has_pca = options_in.find("--pca") != options_in.end();
    if (!has_pca)
        return 0;

    // PCAStream argument: --pca [N]
    // Omitted N means 0 (V1-style "all" sentinel), later mapped to n.
    {
        const auto& vals = options_in["--pca"];
        if (vals.empty()) {
            options_d["pca_out_num"] = 0.0;
        } else {
            options_d["pca_out_num"] = std::stod(vals[0]);
            if (vals.size() > 1)
                LOGGER.w(0, "--pca accepts at most one value; using the first one.");
        }
    }

    // Only reliable if PCAStream::registerOption runs before whatever else
    // might consume --grm-cutoff for an unrelated purpose (e.g. --make-grm
    // --grm-cutoff in the same invocation) — options_in is shared across all
    // registerOption calls, and this can only see keys not yet erased by an
    // earlier one. Same ordering dependency the whole registers[] list
    // already has; nothing new introduced here.
    if (options_in.find("--grm-cutoff") != options_in.end())
        LOGGER.e(0, "--pca does not support --grm-cutoff yet "
                    "(needs off-diagonal GRM values, not just IDs); use --pca-v1 "
                    "for relatedness pruning, or prune separately first.");

    if (options_in.find("--grm") == options_in.end() || options_in["--grm"].empty())
        LOGGER.e(0, "--pca requires --grm <prefix>.");
    options["grm"] = options_in["--grm"][0];
    options_in.erase("--grm");

    // Optional: if omitted, --pca runs exact dense PCA (dsyevr/dsyevd)
    // on the selected sample set. If present with no value, match legacy
    // default approximate method (Lanczos).
    if (options_in.find("--pca-approx") != options_in.end()) {
        const auto& vals = options_in["--pca-approx"];
        if (vals.empty() || vals[0].empty()) options["pca_approx"] = "Lanczos";
        else options["pca_approx"] = vals[0];
        options_in.erase("--pca-approx");
    }

    if (options_in.find("--keep") != options_in.end() && !options_in["--keep"].empty()) {
        options["keep"] = options_in["--keep"][0];
        options_in.erase("--keep");
    }
    if (options_in.find("--remove") != options_in.end() && !options_in["--remove"].empty()) {
        options["remove"] = options_in["--remove"][0];
        options_in.erase("--remove");
    }

    // Same name and meaning as --grm-chunked-budget in MLMA_stream.cpp /
    // RemlEngine.cpp — GB budget for streaming the GRM in row-chunks instead
    // of loading it densely — reused verbatim rather than introducing a
    // PCA-specific flag.
    if (options_in.find("--grm-chunked-budget") != options_in.end()) {
        const auto& vals = options_in["--grm-chunked-budget"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--grm-chunked-budget requires a GB argument (e.g. 20).");
        options_d["grm_chunked_budget"] = std::stod(vals[0]);
        options_in.erase("--grm-chunked-budget");
    }

    if (options_in.find("--svd-method") != options_in.end()) {
        const auto& vals = options_in["--svd-method"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--svd-method requires one argument: power or nystrom.");
        string method = vals[0];
        std::transform(method.begin(), method.end(), method.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (method == "nystrom") {
            options["pca_nystrom"] = "1";
        } else if (method != "power") {
            LOGGER.e(0, "Unsupported --svd-method value [" + vals[0] + "]. Allowed values: power, nystrom.");
        }
        if (vals.size() > 1)
            LOGGER.w(0, "--svd-method expects exactly one value; using the first one.");
        options_in.erase("--svd-method");
    }
    if (options_in.find("--svd-power-iter") != options_in.end()) {
        const auto& vals = options_in["--svd-power-iter"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--svd-power-iter requires one integer argument.");
        options["svd_power_iter"] = vals[0];
        if (vals.size() > 1)
            LOGGER.w(0, "--svd-power-iter expects exactly one value; using the first one.");
        options_in.erase("--svd-power-iter");
    }

    processFunctions.push_back("PCAStream");
    options_in.erase("--pca");
    return 1;
}

void PCAStream::processMain()
{
    for (const auto& pf : processFunctions) {
        if (pf != "PCAStream") continue;

        const string grm_pfx    = options.at("grm");
        const string out_prefix = options.at("out");
        string pca_approx = options.count("pca_approx") ? options.at("pca_approx") : string();

        // ---- Sample set: full GRM ID list, optionally filtered by --keep/--remove ----
        const vector<string> grm_ids = Pheno::read_sublist(grm_pfx + ".grm.id");

        vector<string> analysis_ids = grm_ids;  // preserves GRM file order, same as V1's _keep convention
        if (options.count("keep")) {
            const vector<string> keep_ids = Pheno::read_sublist(options.at("keep"));
            const std::unordered_set<string> keep_set(keep_ids.begin(), keep_ids.end());
            vector<string> filtered;
            filtered.reserve(analysis_ids.size());
            for (const auto& id : analysis_ids)
                if (keep_set.count(id)) filtered.push_back(id);
            analysis_ids = std::move(filtered);
        }
        if (options.count("remove")) {
            const vector<string> remove_ids = Pheno::read_sublist(options.at("remove"));
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

        int out_pc_num = options_d.count("pca_out_num")
            ? static_cast<int>(options_d.at("pca_out_num")) : 0;
        if (out_pc_num < 0)
            LOGGER.e(0, "--pca requires N >= 0 (0 means all, matching V1 semantics).");
        if (out_pc_num == 0 || out_pc_num > n) out_pc_num = n;

        if (out_pc_num == n && !pca_approx.empty()) {
            LOGGER.w(0, "--pca-approx is set, but all PCs requested. Falling back to full eigenvalue decomposition.");
            pca_approx.clear();
        }

        const double grm_chunked_budget = options_d.count("grm_chunked_budget")
            ? options_d.at("grm_chunked_budget") : 0.0;
        bool grm_chunked = grm_chunked_budget > 0.0;
        int svd_power_iter = options.count("svd_power_iter")
            ? std::stoi(options.at("svd_power_iter")) : 3;

        if (pca_approx.empty() && grm_chunked)
            LOGGER.e(0, "--pca exact mode (dsyevr/dsyevd) requires dense GRM loading. "
                        "Remove --grm-chunked-budget, or set --pca-approx to Lanczos/rSVD.");

        // Row-chunk size for streaming reads off the GRM (chunked_symmetric_matvec),
        // and the reader's own row-band cache reservation, are now both solved
        // inside make_chunked_grm_reader (see grm_binary_io.hpp) rather than
        // computed here and passed in piecemeal -- this file never touches
        // RemlCtx, so it has no other reason to duplicate that arithmetic.
        int chunk_size = 0;
        gcta_grm_io::ChunkedGrmHandle chunked_handle;  // only populated when grm_chunked
        if (grm_chunked) {
            // NOTE: this now opens the file and ID-matches against
            // .grm.id/.grm.N.bin unconditionally whenever grm_chunked is
            // true, even if the chunk_size >= n check below ends up
            // falling back to dense -- previously that check ran first and
            // skipped opening the file entirely in that case. The extra
            // work is cheap (ID matching + one N-mean read, no tile reads
            // yet) but it's a real behavior change worth knowing about.
            chunked_handle = gcta_grm_io::make_chunked_grm_reader(
                grm_pfx, analysis_ids, grm_chunked_budget, /*k_ext=*/0, "--pca");
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
        Eigen::MatrixXd G_dense;      // populated only for exact mode (!grm_chunked && pca_approx.empty())
        Eigen::MatrixXf G_dense_f32;  // populated for --pca-approx when !grm_chunked

        if (grm_chunked) {
            // make_chunked_grm_reader already logged the budget/chunk_size
            // split above; nothing further to log here.
            chunked_reader = std::move(chunked_handle.reader);
        } else if (!pca_approx.empty()) {
            double m_snps_unused = -1.0;
            gcta_grm_io::load_and_subset_grm_upper(
                grm_pfx, analysis_ids, G_dense_f32, m_snps_unused, "--pca");
        } else {
            double m_snps_unused = -1.0;
            gcta_grm_io::load_and_subset_grm_upper(
                grm_pfx, analysis_ids, G_dense, m_snps_unused, "--pca");
        }

        auto apply_f32 = [&](const Eigen::Ref<const Eigen::MatrixXf>& X) -> Eigen::MatrixXf {
            if (grm_chunked)
                return gcta_chunked::chunked_symmetric_matvec(
                    chunked_reader, n, chunk_size, X.cast<double>()).cast<float>();
            else
                return G_dense_f32.selfadjointView<Eigen::Upper>() * X;
        };

        auto apply_f64 = [&](const Eigen::Ref<const Eigen::VectorXd>& x) -> Eigen::VectorXd {
            if (grm_chunked)
                return gcta_chunked::chunked_symmetric_matvec(chunked_reader, n, chunk_size, x);
            else
                return (G_dense_f32.selfadjointView<Eigen::Upper>() * x.cast<float>()).cast<double>();
        };

        // Capture the full trace once here so that a partial spectrum can still be
        // reported as a proportion of total variance, rather than as if it were the
        // full sum of eigenvalues. For chunked execution we intentionally read only
        // the diagonal tiles, not the full matrix.
        // The GRM can be overwritten in place, so we must compute the trace first.
        const double trace_total = [&]() {
            if (grm_chunked) return gcta_chunked::chunked_diagonal(chunked_reader, n, chunk_size).sum();
            if (G_dense_f32.size() > 0) return G_dense_f32.diagonal().cast<double>().sum();
            if (G_dense.size() > 0) return G_dense.diagonal().sum();
            return 0.0;
        }();
        LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));

        Eigen::VectorXd eval;
        Eigen::MatrixXd evec;
        bool used_dsyevd = false;
        const double* raw_evec = nullptr;

        try {
            const bool pca_nystrom = options.count("pca_nystrom") > 0;
            if (!pca_approx.empty()) {
                if (pca_approx == "rSVD") {
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
                } else if (pca_approx == "Lanczos") {
                    const int ncv = std::min(n, std::max(3 * out_pc_num + 1, 30));
                    gcta_eigh::EighResult res = gcta_eigh::lanczos_symmetric_eigh(apply_f64, n, out_pc_num, ncv);
                    G_dense_f32.resize(0, 0);
                    eval = std::move(res.eigenvalues);
                    evec = std::move(res.eigenvectors);
                } else {
                    LOGGER.e(0, "--pca-approx: unrecognised method '" + pca_approx + "'. Use 'rSVD' or 'Lanczos'.");
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
}