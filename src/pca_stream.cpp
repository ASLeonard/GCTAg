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

    // Same name and meaning as --svd-chunked-budget in MLMA_stream.cpp /
    // RemlEngine.cpp — GB budget for streaming the GRM in row-chunks instead
    // of loading it densely — reused verbatim rather than introducing a
    // PCA-specific flag.
    if (options_in.find("--svd-chunked-budget") != options_in.end()) {
        const auto& vals = options_in["--svd-chunked-budget"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--svd-chunked-budget requires a GB argument (e.g. 20).");
        options_d["svd_chunked_budget"] = std::stod(vals[0]);
        options_in.erase("--svd-chunked-budget");
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

        const double svd_chunked_budget = options_d.count("svd_chunked_budget")
            ? options_d.at("svd_chunked_budget") : 0.0;
        bool svd_chunked = svd_chunked_budget > 0.0;

        if (pca_approx.empty() && svd_chunked)
            LOGGER.e(0, "--pca exact mode (dsyevr/dsyevd) requires dense GRM loading. "
                        "Remove --svd-chunked-budget, or set --pca-approx to Lanczos/rSVD.");

        // Row-chunk size for streaming reads off the GRM (chunked_symmetric_matvec /
        // matvec_blocked). Budget-driven rather than a guessed constant, same pattern
        // as RemlEngine.cpp's compute_woodbury_basis: solve for the number of rows
        // that fit chunk_rows x (n + k_ext) doubles in the budget. k_ext is sized for
        // whichever approx method got selected above — rSVD/Nystrom multiply block-wise
        // by out_pc_num + oversample columns at once; Lanczos is normally one vector at
        // a time (cols=1) but ncv is used here anyway as a conservative upper bound in
        // case the underlying implementation ever blocks its matvecs internally.
        int chunk_size = 0;
        int k_ext_hint = 0;
        if (svd_chunked) {
            k_ext_hint = (pca_approx == "Lanczos")
                ? std::min(n, std::max(3 * out_pc_num + 1, 30))
                : out_pc_num + gcta_eigh::recommended_oversample(out_pc_num);
            chunk_size = gcta_chunked::solve_chunk_rows(n, svd_chunked_budget, k_ext_hint);
            if (chunk_size < 1)
                LOGGER.e(0, "--svd-chunked-budget=" + to_string(svd_chunked_budget) +
                            "GB cannot fit even a single GRM row (n=" + to_string(n) +
                            ", k_ext=" + to_string(k_ext_hint) + " -> " +
                            to_string(8.0 * (n + k_ext_hint) / 1e9) + "GB/row); raise the budget.");
            if (chunk_size >= n) {
                // The whole GRM fits in a single block: chunking then buys nothing
                // (there's no second chunk to defer loading) but still pays the
                // diagonal-tile mirror's transient 2x n x n duplication (see
                // chunked_grm_matvec.hpp) across the entire matrix at once, since
                // that tile IS the whole matrix here. Strictly worse than dense.
                LOGGER.w(0, "--svd-chunked-budget=" + to_string(svd_chunked_budget) +
                            "GB covers the full GRM (n=" + to_string(n) + ", k_ext=" +
                            to_string(k_ext_hint) + ") in a single chunk. Falling back to "
                            "dense loading instead of paying chunking overhead for no benefit.");
                svd_chunked = false;
            }
        }

        LOGGER.i(0, "Running PCA v2 for " + to_string(out_pc_num) + " PC(s) from " + to_string(n) + " individuals");

        // ---- GRM access: chunked tile reader, or dense (fallback / comparison) ----
        gcta_chunked::TileReader chunked_reader;
        std::shared_ptr<const gcta_grm_io::ChunkedGrmMmap> chunked_file;
        Eigen::MatrixXd G_dense;  // left empty when svd_chunked

        if (svd_chunked) {
            gcta_grm_io::ChunkedGrmHandle handle = gcta_grm_io::make_chunked_grm_reader(grm_pfx, analysis_ids);
            chunked_reader = std::move(handle.reader);
            chunked_file = std::move(handle.file);
            LOGGER.i(0, "--pca: GRM will be read in " + to_string(chunk_size) +
                        "-row chunks from [" + grm_pfx + "] (--svd-chunked-budget=" +
                        to_string(svd_chunked_budget) + "GB, k_ext up to " + to_string(k_ext_hint) +
                        "), not loaded densely.");
        } else {
            vector<string> loaded_ids;
            double m_snps_unused = -1.0;
            Eigen::MatrixXd G_full;
            gcta_grm_io::read_grm_binary(grm_pfx, loaded_ids, G_full, m_snps_unused);
            const vector<int> kp = gcta_grm_io::match_ids_to_grm(analysis_ids, loaded_ids);
            for (int i = 0; i < n; ++i)
                if (kp[i] < 0)
                    LOGGER.e(0, "--pca: individual [" + analysis_ids[i] + "] not found in GRM.");

            // Fast path: no --keep/--remove (or a --keep/--remove that
            // happens to preserve GRM order) means kp is the identity — skip
            // the copy entirely rather than holding G_full and G_dense both
            // fully resident at once for no reason.
            const int n_grm = static_cast<int>(loaded_ids.size());
            bool is_identity = (n_grm == n);
            for (int i = 0; i < n && is_identity; ++i)
                if (kp[i] != i) is_identity = false;

            if (is_identity) {
                G_dense = std::move(G_full);
            } else {
                G_dense.resize(n, n);
                // Column-first traversal for column-major Eigen storage —
                // same pattern as MLMA_stream.cpp's G_n subsetting: for fixed
                // j, varying i reads down one column of G_full (contiguous
                // range, even though kp[i] visits it out of order), rather
                // than jumping across the whole matrix.
                for (int j = 0; j < n; ++j) {
                    const int src_col = kp[j];
                    for (int i = 0; i < n; ++i)
                        G_dense(i, j) = G_full(kp[i], src_col);
                }
                // G_full (n_grm x n_grm, potentially much larger than G_dense
                // when heavily filtered) goes out of scope at the end of this
                // else-block regardless, but free it explicitly right here —
                // no reason to keep it alive through the rest of this branch.
                G_full.resize(0, 0);
            }
        }

        auto apply = [&](const auto& X) -> Eigen::MatrixXd {
            const int cols = static_cast<int>(X.cols());
            if (svd_chunked)
            {
                Eigen::MatrixXd Y;
                if (cols == 1 && chunked_file) {
                    Y.resize(n, 1);
                    Y.col(0) = chunked_file->matvec_blocked(X.col(0), chunk_size);
                } else {
                    Y = gcta_chunked::chunked_symmetric_matvec(chunked_reader, n, chunk_size, X);
                }
                return Y;
            }

            Eigen::MatrixXd Y = G_dense * X;
            return Y;
        };

        Eigen::VectorXd eval;
        Eigen::MatrixXd evec;
        bool used_dsyevd = false;
        const double* raw_evec = nullptr;

        try {
            const bool pca_nystrom = options.count("pca_nystrom") > 0;
            if (!pca_approx.empty()) {
                gcta_eigh::EighResult res;
                if (pca_approx == "rSVD") {
                    const int oversample = gcta_eigh::recommended_oversample(out_pc_num);
                    if (pca_nystrom) {
                        LOGGER.i(0, "--pca rSVD + --svd-method nystrom: signed single-pass sketch (1 GRM matvec). Accuracy near PC " +
                                    std::to_string(out_pc_num) + " may be lower than power-iteration rSVD.");
                        res = gcta_eigh::nystrom_symmetric_eigh(apply, n, out_pc_num, oversample);
                    } else {
                        res = gcta_eigh::randomized_symmetric_eigh(apply, n, out_pc_num, oversample, 3);
                    }
                } else if (pca_approx == "Lanczos") {
                    const int ncv = std::min(n, std::max(3 * out_pc_num + 1, 30));
                    res = gcta_eigh::lanczos_symmetric_eigh(apply, n, out_pc_num, ncv);
                } else {
                    LOGGER.e(0, "--pca-approx: unrecognised method '" + pca_approx + "'. Use 'rSVD' or 'Lanczos'.");
                }
                eval = std::move(res.eigenvalues);
                evec = std::move(res.eigenvectors);
            } else {
                if (pca_nystrom)
                    LOGGER.w(0, "--svd-method is ignored in exact mode (it only applies to --pca-approx rSVD).");

                double* grm_ptr = G_dense.data();
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

        // Capture the full trace once here so that a partial spectrum can still be
        // reported as a proportion of total variance, rather than as if it were the
        // full sum of eigenvalues. For chunked execution we intentionally read only
        // the diagonal tiles, not the full matrix.
        const double trace_total = [&]() {
            if (svd_chunked) return gcta_chunked::chunked_diagonal(chunked_reader, n, chunk_size).sum();
            if (G_dense.size() == 0) return 0.0;
            return G_dense.diagonal().sum();
        }();
        LOGGER.i(0, "Total variance (tr(G)) = " + to_string(trace_total));

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