/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * MLMALoco — v2 LOCO MLMA.  Per-chromosome REML via reml::compute() + streaming
 * association test via Geno::loopDouble.
 *
 * Mandatory flags:
 *   --mlma-loco-stream    (trigger)
 *   --loco-manifest <f>   (TSV: chrom, bfile_prefix, grm_chr_prefix)
 *   --grm <prefix>        (all-autosome GRM)
 *   --pheno <f>
 *
 * Optional:
 *   --qcovar / --covar
 *   --reml-woodbury-basis [k]
 *   --reml-trace-approx [n]
 *   --log-pval
 *   --out <prefix>
 */

#include "MLMA_loco.h"
#include "RemlCtx.hpp"
#include "RemlEngine.hpp"
#include "RemlState.hpp"
#include "grm_binary_io.hpp"
#include "mlma_woodbury.hpp"
#include "Logger.h"
#include "cpu.h"
#include "Pheno.h"
#include "Marker.h"
#include "Geno.h"
#include "Covar.h"
#include "MLMA_stream_common.hpp"

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>

using std::map;
using std::string;
using std::vector;
using std::to_string;
using std::function;

// ---------------------------------------------------------------------------
// Static member definitions
// ---------------------------------------------------------------------------
map<string, string>  MLMALoco::options;
map<string, double>  MLMALoco::options_d;
map<string, vector<double>> MLMALoco::options_vd;
vector<string>       MLMALoco::processFunctions;

// ---------------------------------------------------------------------------
// GRM binary I/O helpers (file-local)
// ---------------------------------------------------------------------------
namespace {

// Shared I/O helpers live in include/grm_binary_io.hpp as inline functions.
using gcta_grm_io::read_grm_binary;
using gcta_grm_io::match_ids_to_grm;

// Manifest row
struct ManifestRow {
    int    chrom;
    string bfile;
    string grm_chr;
};

vector<ManifestRow> read_manifest(const string& filename)
{
    std::ifstream f(filename);
    if (!f.is_open()) LOGGER.e(0, "cannot open manifest file [" + filename + "].");
    vector<ManifestRow> rows;
    string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        // strip \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        ManifestRow r;
        string chrom_str;
        if (!(ss >> chrom_str >> r.bfile >> r.grm_chr)) {
            LOGGER.e(0, "manifest [" + filename + "] line " + to_string(lineno)
                      + ": expected 3 columns (chrom, bfile, grm_chr).");
        }
        r.chrom = std::stoi(chrom_str);
        rows.push_back(std::move(r));
    }
    if (rows.empty()) LOGGER.e(0, "manifest file [" + filename + "] has no data rows.");
    return rows;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// registerOption
// ---------------------------------------------------------------------------
int MLMALoco::registerOption(map<string, vector<string>>& options_in)
{
    // Common option: --out
    if (options_in.find("out") != options_in.end() && !options_in["out"].empty())
        options["out"] = options_in["out"][0];

    const bool has_flag = (options_in.find("--mlma-loco-stream") != options_in.end());
    if (!has_flag) return 0;

    if (options_in.find("--reml-alg") != options_in.end()
            && !options_in["--reml-alg"].empty()) {
        options_d["reml_alg"] = std::stod(options_in["--reml-alg"][0]);
        options_in.erase("--reml-alg");
    }
    if (options_in.find("--reml-no-constrain") != options_in.end()) {
        options["no_constrain"] = "1";
        options_in.erase("--reml-no-constrain");
    }
    if (options_in.find("--reml-diagV-adj") != options_in.end()
            && !options_in["--reml-diagV-adj"].empty()) {
        options_d["reml_diagV_adj"] = std::stod(options_in["--reml-diagV-adj"][0]);
        options_in.erase("--reml-diagV-adj");
    }

    if (options_in.find("--reml-priors-var") != options_in.end()) {
        options["reml_priors_var"] = "1";
        const auto& vals = options_in["--reml-priors-var"];
        for (const auto& s : vals) {
            if (!s.empty())
                options_vd["reml_priors_var"].push_back(std::stod(s));
        }
        options_in.erase("--reml-priors-var");
    }
    if (options_in.find("--reml-priors") != options_in.end()) {
        const auto& vals = options_in["--reml-priors"];
        for (const auto& s : vals) {
            if (!s.empty())
                options_vd["reml_priors"].push_back(std::stod(s));
        }
        options_in.erase("--reml-priors");
    }

    // --loco-manifest (mandatory)
    if (options_in.find("--loco-manifest") == options_in.end()
            || options_in["--loco-manifest"].empty())
        LOGGER.e(0, "--mlma-loco-stream requires --loco-manifest <file>.");
    options["manifest"] = options_in["--loco-manifest"][0];
    options_in.erase("--loco-manifest");

    // --grm (mandatory — all-chromosome GRM)
    if (options_in.find("--grm") != options_in.end() && !options_in["--grm"].empty()) {
        options["grm_all"] = options_in["--grm"][0];
        options_in.erase("--grm");
    } else {
        LOGGER.e(0, "--mlma-loco-stream requires --grm <prefix> (all-autosome GRM).");
    }

    // Optional flags
    if (options_in.find("--log-pval") != options_in.end()) {
        options["log_pval"] = "1";
        options_in.erase("--log-pval");
    }

    // --reml-woodbury-basis [k]  (k optional; -1 = MP-k, -2 = EIGMASS, -3 = variance)
    if (options_in.find("--reml-woodbury-basis") != options_in.end()) {
        const auto& vals = options_in["--reml-woodbury-basis"];
        if (!vals.empty() && !vals[0].empty()) {
            if (vals[0] == "EIG")
                options_d["woodbury_basis_rank"] = -2.0;  // Eigenvalue mass k
            else if (vals[0] == "VAR")
                options_d["woodbury_basis_rank"] = -3.0;  // Variance-k
            else if (vals[0] == "MP")  // Marchenko-Pastur
                options_d["woodbury_basis_rank"] = -1.0;  // MP-k
            else
                options_d["woodbury_basis_rank"] = std::stod(vals[0]);
        } else
            options_d["woodbury_basis_rank"] = -1.0;   // MP-k
        options_in.erase("--reml-woodbury-basis");
    }
    if (options_in.find("--reml-woodbury-basis-eigen-mass") != options_in.end()) {
        const auto& vals = options_in["--reml-woodbury-basis-eigen-mass"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--reml-woodbury-basis-eigen-mass requires a fractional argument (e.g. 0.99).");
        options_d["woodbury_basis_eigen_mass"] = std::stod(vals[0]);
        options_in.erase("--reml-woodbury-basis-eigen-mass");
    }
    if (options_in.find("--reml-woodbury-basis-var-thresh") != options_in.end()) {
        const auto& vals = options_in["--reml-woodbury-basis-var-thresh"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--reml-woodbury-basis-var-thresh requires a relative Frobenius error argument (e.g. 0.01).");
        options_d["woodbury_basis_var_thresh"] = std::stod(vals[0]);
        options_in.erase("--reml-woodbury-basis-var-thresh");
    }
    if (options_in.find("--reml-woodbury-basis-edge-margin") != options_in.end()) {
        const auto& vals = options_in["--reml-woodbury-basis-edge-margin"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--reml-woodbury-basis-edge-margin requires a fractional argument (e.g. 0.15).");
        options_d["woodbury_basis_edge_margin"] = std::stod(vals[0]);
        options_in.erase("--reml-woodbury-basis-edge-margin");
    }
    if (options_in.find("--reml-woodbury-basis-edge-confirm") != options_in.end()) {
        const auto& vals = options_in["--reml-woodbury-basis-edge-confirm"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--reml-woodbury-basis-edge-confirm requires an integer argument.");
        options_d["woodbury_basis_edge_confirm"] = std::stod(vals[0]);
        options_in.erase("--reml-woodbury-basis-edge-confirm");
    }
    if (options_in.find("--reml-woodbury-basis-EIGMASS-k-buffer") != options_in.end()) {
        const auto& vals = options_in["--reml-woodbury-basis-EIGMASS-k-buffer"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--reml-woodbury-basis-EIGMASS-k-buffer requires an integer argument.");
        options_d["woodbury_basis_EIGMASS_k_buffer"] = std::stod(vals[0]);
        options_in.erase("--reml-woodbury-basis-EIGMASS-k-buffer");
    }
    if (options_in.find("--svd-method") != options_in.end()) {
        const auto& vals = options_in["--svd-method"];
        if (vals.empty() || vals[0].empty())
            LOGGER.e(0, "--svd-method requires one argument: power or nystrom.");
        string method = vals[0];
        std::transform(method.begin(), method.end(), method.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (method == "nystrom") {
            options["svd_nystrom"] = "1";
        } else if (method != "power") {
            LOGGER.e(0, "Unsupported --svd-method value [" + vals[0] + "]. Allowed values: power, nystrom.");
        }
        if (vals.size() > 1)
            LOGGER.w(0, "--svd-method expects exactly one value; using the first one.");
        options_in.erase("--svd-method");
    }

    // --reml-trace-hutchpp [n]
    if (options_in.find("--reml-trace-hutchpp") != options_in.end()) {
        const auto& vals = options_in["--reml-trace-hutchpp"];
        options["trace_hutchpp"] = "1";
        if (!vals.empty() && !vals[0].empty())
            options_d["trace_hutchpp_nprobes"] = std::stod(vals[0]);
        options_in.erase("--reml-trace-hutchpp");
    }

    // --reml-trace-hutchpp-fixed-probes
    if (options_in.find("--reml-trace-hutchpp-fixed-probes") != options_in.end()) {
        options["trace_hutchpp_fixed_probes"] = "1";
        options_in.erase("--reml-trace-hutchpp-fixed-probes");
    }

    // --reml-maxit
    if (options_in.find("--reml-maxit") != options_in.end()
            && !options_in["--reml-maxit"].empty()) {
        options_d["reml_maxit"] = std::stod(options_in["--reml-maxit"][0]);
        options_in.erase("--reml-maxit");
    }

    processFunctions.push_back("MLMALoco");
    options_in.erase("--mlma-loco-stream");
    return 1;
}

// ---------------------------------------------------------------------------
// processMain
// ---------------------------------------------------------------------------
void MLMALoco::processMain()
{
    for (const auto& pf : processFunctions) {
        if (pf != "MLMALoco") continue;

        const string manifest_file = options.at("manifest");
        const string grm_all_pfx  = options.at("grm_all");
        const string out_prefix   = options.at("out");
        const bool   log_pval     = options.count("log_pval") > 0;

        // REML config
        const int    woodbury_basis_rank = (options_d.count("woodbury_basis_rank") > 0)
            ? static_cast<int>(options_d.at("woodbury_basis_rank")) : 0;
        const bool   trace_approx = (options.count("trace_approx") > 0);
        const int    trace_nprobes = (options_d.count("trace_approx_nprobes") > 0)
            ? static_cast<int>(options_d.at("trace_approx_nprobes")) : 90;
        const int    reml_maxit = (options_d.count("reml_maxit") > 0)
            ? static_cast<int>(options_d.at("reml_maxit")) : 100;
        const int    reml_alg = (options_d.count("reml_alg") > 0)
            ? static_cast<int>(options_d.at("reml_alg")) : 0;
        const int    reml_diagV_adj = (options_d.count("reml_diagV_adj") > 0)
            ? static_cast<int>(options_d.at("reml_diagV_adj")) : 0;
        const bool   no_constrain = options.count("no_constrain") > 0;
        const bool   svd_nystrom = options.count("svd_nystrom") > 0;
        const float  woodbury_basis_eigen_mass = options_d.count("woodbury_basis_eigen_mass")
            ? static_cast<float>(options_d.at("woodbury_basis_eigen_mass")) : 0.99f;
        const double woodbury_basis_edge_margin = options_d.count("woodbury_basis_edge_margin")
            ? options_d.at("woodbury_basis_edge_margin") : 0.15;
        const int    woodbury_basis_edge_confirm = options_d.count("woodbury_basis_edge_confirm")
            ? static_cast<int>(options_d.at("woodbury_basis_edge_confirm")) : 20;
        const int    woodbury_basis_EIGMASS_k_buffer = options_d.count("woodbury_basis_EIGMASS_k_buffer")
            ? static_cast<int>(options_d.at("woodbury_basis_EIGMASS_k_buffer")) : 0;
        const double woodbury_basis_var_thresh = options_d.count("woodbury_basis_var_thresh")
            ? options_d.at("woodbury_basis_var_thresh") : 0.01;

        if (reml_alg < 0 || reml_alg > 2)
            LOGGER.e(0, "--reml-alg should be 0, 1 or 2.");
        if (reml_diagV_adj < 0 || reml_diagV_adj > 2)
            LOGGER.e(0, "--reml-diagV-adj should be 0, 1, or 2.");
        if (woodbury_basis_rank != 0 && reml_alg == 1)
            LOGGER.e(0, "--reml-woodbury-basis is incompatible with Fisher-scoring REML (--reml-alg 1). Use AI-REML (default) or EM-REML (--reml-alg 2).");

        const vector<double> priors =
            options_vd.count("reml_priors") ? options_vd.at("reml_priors") : vector<double>{};
        const vector<double> priors_var_global =
            options_vd.count("reml_priors_var") ? options_vd.at("reml_priors_var") : vector<double>{};

        // ---- 1. Read manifest ----
        vector<ManifestRow> manifest = read_manifest(manifest_file);
        LOGGER << "[LOCO] Manifest: " << manifest.size() << " chromosome(s)." << std::endl;

        // ---- 2. Load all-chr GRM ----
        LOGGER << "\n[LOCO] Loading all-autosome GRM from [" << grm_all_pfx << "] ..." << std::endl;
        vector<string> grm_all_ids;
        Eigen::MatrixXd G_all;
        double m_all = 0.0;
        read_grm_binary(grm_all_pfx, grm_all_ids, G_all, m_all);
        if (m_all <= 0.0)
            LOGGER.e(0, "[LOCO] Could not read marker count from [" + grm_all_pfx
                      + ".grm.N.bin]. Rebuild GRM without --no-grm-N.");
        LOGGER << "[LOCO] G_all: " << grm_all_ids.size() << " individuals, "
               << static_cast<long long>(m_all) << " markers." << std::endl;

        // ---- 3. Load phenotype ----
        Pheno* pheno  = new Pheno();
        const uint32_t n_pheno_init = pheno->count_keep();
        vector<string> pheno_ids;
        vector<double> phenos_vec;
        pheno->get_pheno(pheno_ids, phenos_vec);
        if (pheno_ids.empty())
            LOGGER.e(0, "--mlma-loco-stream requires --pheno to be specified.");

        // ---- 4. Load covariates ----
        Covar covar;
        bool  has_covar = false;
        int   x_c       = 1;    // intercept only by default
        Eigen::MatrixXd X_design_full;  // n_pheno × x_c (before GRM intersection)

        {
            vector<uint32_t> remain_index, covar_index;
            has_covar = covar.getCommonSampleIndex(pheno_ids, remain_index, covar_index);

            if (has_covar) {
                LOGGER.i(0, to_string(remain_index.size()) +
                            " overlapping individuals with non-missing covariate data.");
                pheno->filter_keep_index(remain_index);
                phenos_vec = compact_sample_vector(phenos_vec, remain_index);
            } else if (covar.hasCovar()) {
                LOGGER.e(0, "no overlapping individuals with non-missing covariate data.");
            } else {
                remain_index.resize(pheno_ids.size());
                std::iota(remain_index.begin(), remain_index.end(), 0u);
            }

            const int n_new = static_cast<int>(pheno->count_keep());
            pheno->get_pheno(pheno_ids, phenos_vec);

            if (has_covar) {
                vector<string> final_ids = pheno->get_id(0, n_new - 1, "\t");
                vector<double> covar_X;
                vector<uint32_t> ck_dummy;
                covar.getCovarX(final_ids, covar_X, ck_dummy);
                const int covar_cols = (n_new > 0)
                    ? static_cast<int>(covar_X.size()) / n_new : 0;
                x_c = 1 + covar_cols;
                X_design_full.resize(n_new, x_c);
                X_design_full.col(0).setOnes();
                if (covar_cols > 0)
                    X_design_full.rightCols(covar_cols) =
                        Eigen::Map<const Eigen::MatrixXd>(covar_X.data(), n_new, covar_cols);
            } else {
                const int n_new2 = static_cast<int>(pheno->count_keep());
                X_design_full.resize(n_new2, 1);
                X_design_full.setOnes();
            }
        }

        // ---- 5. Intersect pheno+covar individuals with GRM ----
        // Match pheno IDs (FID\tIID) to GRM IDs (FID\tIID from read_sublist)
        vector<int> kp = match_ids_to_grm(pheno_ids, grm_all_ids);
        // Build the final keep list: only individuals present in GRM
        vector<int> final_keep;
        final_keep.reserve(pheno_ids.size());
        for (int i = 0; i < (int)pheno_ids.size(); ++i)
            if (kp[i] >= 0) final_keep.push_back(i);

        if (final_keep.empty())
            LOGGER.e(0, "[LOCO] No individuals overlap between phenotype and GRM. "
                        "Check that GRM was built from the same sample.");

        const int n = static_cast<int>(final_keep.size());
        LOGGER << "[LOCO] After GRM intersection: " << n << " individuals for analysis." << std::endl;

        pheno_ids  = compact_sample_vector(pheno_ids, final_keep);
        phenos_vec = compact_sample_vector(phenos_vec, final_keep);
        X_design_full = compact_sample_rows(X_design_full, final_keep);

        // Subset G_all to the n analysis individuals.
        // Fast path: if GRM sample == analysis sample (in order), move instead of copy.
        Eigen::MatrixXd G_all_n;
        {
            const int n_grm = static_cast<int>(grm_all_ids.size());
            vector<int> grm_rows(n);
            bool is_identity = (n_grm == n);
            for (int i = 0; i < n; ++i) {
                grm_rows[i] = kp[final_keep[i]];
                if (grm_rows[i] != i) is_identity = false;
            }
            if (is_identity) {
                G_all_n = std::move(G_all);  // O(1): GRM sample == analysis sample
            } else {
                G_all_n.resize(n, n);
                // Column-first traversal: accesses contiguous columns of G_all
                // (column-major Eigen storage), reducing cache misses by ~n×.
                for (int j = 0; j < n; ++j) {
                    const int src_col = grm_rows[j];
                    for (int i = 0; i < n; ++i)
                        G_all_n(i, j) = G_all(grm_rows[i], src_col);
                }
                G_all.resize(0, 0);
            }
        }

        // Subset y and X
        Eigen::VectorXd y_vec(n);
        for (int i = 0; i < n; ++i) {
            y_vec[i] = phenos_vec[i];
        }
        Eigen::MatrixXd X_mat = X_design_full;
        X_design_full.resize(0, 0);  // free

        // Also build the final analysis IDs (for GRM subset matching per chr)
        vector<string> analysis_ids = pheno_ids;

        // ---- 6. Open output file ----
        const string out_file = out_prefix + ".loco.mlma";
        LOGGER << "\n[LOCO] LOCO results will be streamed to [" << out_file << "] ..." << std::endl;
        std::ofstream ofile(out_file);
        if (!ofile) LOGGER.e(0, "cannot open [" + out_file + "] for writing.");
        ofile << "Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\t"
              << (log_pval ? "log_p" : "p") << "\n";

        // ---- 7. Per-chromosome LOCO loop ----
        LOGGER << "\n[LOCO] Running LOCO MLMA over " << manifest.size()
               << " chromosome(s) from manifest." << std::endl;
        if (woodbury_basis_rank != 0)
            LOGGER << "[LOCO] --reml-woodbury-basis active; basis warm-started between chromosomes." << std::endl;
        else
            LOGGER << "[LOCO] Tip: add --reml-woodbury-basis <k> for faster per-chr REML." << std::endl;

        // Woodbury basis for warm-starting across chromosomes
        Eigen::MatrixXd Uk_warmstart;
        Eigen::VectorXd dk_warmstart;
        std::vector<double> varcmp_warmstart;

        // Cache ID mapping when per-chr GRM ID ordering is identical across chromosomes.
        vector<string> cached_grm_chr_ids;
        vector<int>    kp_chr_cached;
        bool           has_kp_chr_cache = false;

        // Reused GRM read buffer to avoid repeated vector/matrix allocations.
        vector<string> grm_chr_ids;
        Eigen::MatrixXd G_chr_raw;

        for (size_t ci = 0; ci < manifest.size(); ++ci) {
            const ManifestRow& row = manifest[ci];
            LOGGER << "\n-----------------------------------\n#Chr " << row.chrom << ":" << std::endl;

            // ---- 7a. Load G_chr ----
            double m_chr = 0.0;
            LOGGER << "  Reading per-chr GRM from [" << row.grm_chr << "] ..." << std::endl;
            read_grm_binary(row.grm_chr, grm_chr_ids, G_chr_raw, m_chr);
            if (m_chr <= 0.0)
                LOGGER.e(0, "[LOCO] Chr " + to_string(row.chrom)
                          + ": could not read marker count from [" + row.grm_chr
                          + ".grm.N.bin]. Rebuild per-chr GRM without --no-grm-N.");

            // Match analysis_ids to chr GRM IDs
            vector<int> kp_chr;
            if (has_kp_chr_cache && grm_chr_ids == cached_grm_chr_ids) {
                kp_chr = kp_chr_cached;
            } else if (grm_chr_ids.size() == analysis_ids.size()
                       && std::equal(grm_chr_ids.begin(), grm_chr_ids.end(), analysis_ids.begin())) {
                kp_chr.resize(n);
                std::iota(kp_chr.begin(), kp_chr.end(), 0);
                cached_grm_chr_ids = grm_chr_ids;
                kp_chr_cached = kp_chr;
                has_kp_chr_cache = true;
            } else {
                kp_chr = match_ids_to_grm(analysis_ids, grm_chr_ids);
                cached_grm_chr_ids = grm_chr_ids;
                kp_chr_cached = kp_chr;
                has_kp_chr_cache = true;
            }
            for (int i = 0; i < n; ++i)
                if (kp_chr[i] < 0)
                    LOGGER.e(0, "[LOCO] Chr " + to_string(row.chrom)
                              + ": individual [" + analysis_ids[i]
                              + "] not found in per-chr GRM [" + row.grm_chr + "].");

            // Subset G_chr to analysis individuals.
            // Fast path: if chr-GRM sample == analysis sample (in order), move.
            Eigen::MatrixXd G_chr;
            {
                const int n_grm_chr = static_cast<int>(grm_chr_ids.size());
                bool chr_is_identity = (n_grm_chr == n);
                for (int i = 0; i < n && chr_is_identity; ++i)
                    if (kp_chr[i] != i) chr_is_identity = false;

                if (chr_is_identity) {
                    G_chr = std::move(G_chr_raw);
                } else {
                    G_chr.resize(n, n);
                    // Column-first traversal for column-major Eigen storage.
                    for (int j = 0; j < n; ++j) {
                        const int src_col = kp_chr[j];
                        for (int i = 0; i < n; ++i)
                            G_chr(i, j) = G_chr_raw(kp_chr[i], src_col);
                    }
                    G_chr_raw.resize(0, 0);
                }
            }

            // ---- 7b. Compute LOCO GRM in-place via G_chr ----
            // G_loco = (G_all_n * m_all - G_chr * m_chr) / m_loco
            // Reuse G_chr's buffer to avoid a 3rd n×n allocation (saves ~1.8 GB peak).
            const double m_loco = m_all - m_chr;
            if (m_loco <= 0.0)
                LOGGER.e(0, "[LOCO] Chr " + to_string(row.chrom) + ": m_chr=" +
                            to_string(static_cast<long long>(m_chr))
                          + " >= m_all=" + to_string(static_cast<long long>(m_all)) + ".");
            G_chr.array() *= (-m_chr / m_loco);
            G_chr.noalias() += (m_all / m_loco) * G_all_n;
            // G_chr now holds G_loco; G_chr_raw is already freed.

            // ---- 7c. Fill RemlCtx ----
            RemlCtx ctx;
            ctx.n   = n;
            ctx.X_c = x_c;
            ctx.X   = X_mat;
            ctx.y   = y_vec;
            ctx.A.resize(2);
            ctx.A[0] = std::move(G_chr);  // consumed by compute_woodbury_basis or used as-is
            ctx.grm_N.resize(1, 1);
            ctx.grm_N(0, 0) = m_loco;
            // ctx.A[1] left default (size 0 == identity convention for residual)
            ctx.r_indx = {0, 1};
            ctx.var_name = {"V(G)", "V(e)"};
            ctx.hsq_name = {"V(G)/Vp"};
            ctx.out = out_prefix + ".chr" + to_string(row.chrom);

            // REML config
            ctx.reml_mtd       = reml_alg;
            ctx.reml_max_iter  = reml_maxit;
            ctx.reml_inv_mtd   = 0;  // LLT
            ctx.reml_diagV_adj = reml_diagV_adj;
            ctx.woodbury_basis_rank            = woodbury_basis_rank;
            ctx.woodbury_basis_eigen_mass      = woodbury_basis_eigen_mass;
            ctx.woodbury_basis_edge_margin     = woodbury_basis_edge_margin;
            ctx.woodbury_basis_edge_confirm    = woodbury_basis_edge_confirm;
            ctx.woodbury_basis_eigmass_k_buffer = woodbury_basis_EIGMASS_k_buffer;
            ctx.woodbury_basis_var_thresh      = woodbury_basis_var_thresh;
            ctx.svd_nystrom         = svd_nystrom;
            ctx.reml_trace_hutchpp = trace_approx;
            ctx.reml_trace_hutchpp_nprobes = trace_nprobes;

            // Warm-start woodbury basis from previous chr
            if (woodbury_basis_rank != 0 && Uk_warmstart.rows() > 0) {
                ctx.Uk = Uk_warmstart;
                ctx.dk = dk_warmstart;
            }

            // ---- 7d. Run REML ----
            // Warm-start variance components from previous chromosome.
            // LOCO GRMs differ by one chromosome only, so V(G)/V(e) is typically close.
            std::vector<double> priors_var =
                (varcmp_warmstart.size() == ctx.r_indx.size()) ? varcmp_warmstart
                                                                : std::vector<double>{};
            if (priors_var.empty())
                priors_var = priors_var_global;
            reml::compute(ctx, priors, priors_var, no_constrain);

            // Save eigenvectors for warm-start in next chromosome
            if (ctx.Vi_use_woodbury_basis && ctx.Uk.rows() > 0) {
                Uk_warmstart = ctx.Uk;
                dk_warmstart = ctx.dk;
            }
            varcmp_warmstart = ctx.varcmp;

            LOGGER << "[LOCO] Chr " << row.chrom << " variance components: ";
            for (double vc : ctx.varcmp) LOGGER << vc << " ";
            LOGGER << std::endl;

            // ---- 7e. Build RemlState ----
            RemlState rs = reml::build_reml_state(ctx);
            release_reml_ctx_after_state_build(ctx);

            // ---- 7f. Pre-adjusted phenotype: y_adj = y - X*b ----
            Eigen::VectorXf y_adj(n);
            {
                Eigen::VectorXd b_d = ctx.b;
                Eigen::VectorXd y_adj_d = y_vec - X_mat * b_d;
                for (int i = 0; i < n; ++i) y_adj[i] = static_cast<float>(y_adj_d[i]);
            }

            // ---- 7h. Stream SNPs via Geno::loopDouble ----
            Marker::setLocoBfilePrefix(row.bfile);
            Geno::setLocoBfilePrefix(row.bfile);

            Pheno*  pheno_chr  = new Pheno();
            Marker* marker_chr = new Marker();
            Geno*   geno_chr   = new Geno(pheno_chr, marker_chr);

            // Per-chromosome bfiles must match analysis sample identity and order exactly.
            const int n_chr = static_cast<int>(pheno_chr->count_keep());
            if (n_chr != n) {
                LOGGER.e(0, "[LOCO] Chr " + to_string(row.chrom)
                          + ": bfile [" + row.bfile + "] has " + to_string(n_chr)
                          + " kept individuals, but LOCO analysis expects " + to_string(n) + ".");
            }
            if (n_chr > 0) {
                const vector<string> chr_ids = pheno_chr->get_id(0, n_chr - 1, "\t");
                if (static_cast<int>(chr_ids.size()) != n) {
                    LOGGER.e(0, "[LOCO] Chr " + to_string(row.chrom)
                              + ": unable to read consistent sample IDs from bfile ["
                              + row.bfile + "].");
                }
                for (int i = 0; i < n; ++i) {
                    if (chr_ids[i] != analysis_ids[i]) {
                        LOGGER.e(0, "[LOCO] Chr " + to_string(row.chrom)
                                  + ": sample order mismatch at index " + to_string(i)
                                  + ". Expected [" + analysis_ids[i] + "] but bfile has ["
                                  + chr_ids[i] + "] in [" + row.bfile + "].");
                    }
                }
            }
            Eigen::VectorXf w_sqrt = pheno->get_sqrt_weight_keep();
            run_mlma_stream_association(rs, y_adj, w_sqrt, geno_chr, marker_chr, n, log_pval, ofile);
            LOGGER << "[LOCO] Chr " << row.chrom << " done." << std::endl;

            delete geno_chr;
            delete marker_chr;
            delete pheno_chr;
        }

        LOGGER << "\n[LOCO] Results saved to [" << out_file << "]." << std::endl;
        ofile.close();

        delete pheno;
    }
}
