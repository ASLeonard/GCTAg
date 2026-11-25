#ifndef GCTA_CLI_HPP
#define GCTA_CLI_HPP

#include "ModeTypes.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gcta_cli {

using ::DiagVAdj;
using ::FastMlmaMode;
using ::RemlAlg;
using ::WoodburyBasisSpec;

enum class GrmAlg : int { Standard = 0, VanRaden = 1 };
enum class GrmStripLayout : uint8_t { ColMajor = 0, RowMajor = 1 };
enum class GrmDiagMode : uint8_t { Syrk = 0, Fused = 1 };
enum class PcaApprox : uint8_t { Lanczos = 1, rSVD = 2 };
enum class PcaMatvecMode : uint8_t { Auto = 0, Fused = 1, Tiled = 2 };
enum class PcaLoadMode : uint8_t { Auto = 0, Direct = 1, Async = 2 };
enum class SvdMethod : uint8_t { Power = 0, Nystrom = 1 };
enum class GenoModel : uint8_t { Additive = 0, Dominance = 1, NonAdditive = 2 };
enum class EigenvecFormat : uint8_t { Text = 0, Bin32 = 1, Bin16 = 2, Both = 3 };

constexpr const char* geno_model_name(GenoModel m) {
    switch (m) {
        case GenoModel::Additive:    return "additive";
        case GenoModel::Dominance:   return "dominance";
        case GenoModel::NonAdditive: return "nonadditive";
    }
    return "additive";
}

struct GlobalOptions {
    std::optional<std::string> out;
    std::optional<int> threads;  // nullopt = unset (fall back to OMP_NUM_THREADS or 1)
    bool verbose = false;

    std::string out_prefix() const { return out.value_or("output"); }
};

struct GenoInputOptions {
    std::string bfile;
    std::string bim;
    std::string fam;
    std::string pfile;
    std::string bpfile;
    std::string mbfile;
    std::string mpfile;
    std::string mbpfile;
    std::string bgen;
    std::string mbgen;
    std::string sample;
    std::string update_freq;
    std::string update_sex;
};

struct SampleFilterOptions {
    std::string keep;
    std::string remove;
    std::string sex_chr_file;
};

struct MarkerFilterOptions {
    std::string extract;
    std::string exclude;
    std::vector<std::string> chr;
    bool autosome = false;
    bool autosome_sex = false;
    bool chr_homogametic = false;
    std::vector<std::string> homogametic_chr;
    std::vector<std::string> heterogametic_chr;
    std::optional<double> maf;
    std::optional<double> max_maf;
    std::optional<double> geno;
    std::optional<double> info;
    std::optional<double> dc;
    std::optional<double> hard_call_thresh;
    bool dosage_call = false;
};

struct PhenoCovarOptions {
    std::string pheno;
    std::optional<int> mpheno;
    bool weighted_phenotypes = false;
    std::string qcovar;
    std::string covar;
    int covar_maxlevel = 33;
};

enum class RemlReuseStream { Auto = 0, Off = 1, On = 2 };
enum class RemlWoodburySolver { Auto = 0, Cholesky = 1, Eigh = 2 };
enum class RemlQrMode : uint8_t { Auto = 0, Sketched = 1, Standard = 2 };
enum class RemlSvdFilter : uint8_t { Power = 0, Chebyshev = 1 };

struct RemlOptions {
    std::string grm;
    std::string woodbury_reuse;
    std::string woodbury_basis_file;
    std::optional<double> grm_chunked_budget;
    bool grm_fp16 = false;
    bool fast = false;
    bool reml_standalone = false;

    std::optional<WoodburyBasisSpec> woodbury_basis;
    std::optional<double> woodbury_mp_margin;
    std::optional<int> woodbury_mp_confirm;
    std::optional<double> woodbury_eig_mass;
    std::optional<int> woodbury_eig_k_buffer;
    std::optional<double> woodbury_var_tail;
    std::optional<std::pair<int, int>> woodbury_range;
    std::optional<double> woodbury_mem_budget_gb;
    std::string woodbury_warm_start;
    bool woodbury_posthoc_correction = false;

    SvdMethod svd_method = SvdMethod::Power;
    std::optional<int> svd_power_iter;

    std::optional<int> trace_hutchpp;  // probe count when enabled (default: 100)
    bool trace_hutchpp_fixed_probes = false;

    RemlAlg alg = RemlAlg::AI;
    std::optional<int> maxit;
    bool no_constrain = false;
    std::vector<double> priors;
    std::vector<double> priors_var;
    DiagVAdj diagV_adj = DiagVAdj::None;
    bool no_he_start = false;
    bool force_dense_v = false;
    bool ai_robust = false;
    std::optional<double> ai_robust_tol;
    std::optional<double> ai_robust_risk;
    bool print_trajectory = false;
    std::optional<uint64_t> seed;

    // Hidden developer tuning flags (group("") in CLI)
    bool reml_profile = false;
    int reml_matvec_band_rows = 0;
    int reml_matvec_col_chunk = 0;
    int reml_qr_power_passes = 0;
    bool reml_eigh_fp32 = false;
    int reml_eigh_threads = 0;
    int reml_io_slot_mb = 256;
    RemlReuseStream reml_reuse_stream = RemlReuseStream::Auto;
    RemlWoodburySolver reml_woodbury_solver = RemlWoodburySolver::Auto;
    RemlQrMode reml_qr_mode = RemlQrMode::Auto;
    RemlSvdFilter reml_svd_filter = RemlSvdFilter::Power;
};

struct GrmMakeOptions {
    GenoInputOptions geno;
    SampleFilterOptions sample_filter;
    MarkerFilterOptions marker_filter;

    bool dominance = false;
    bool homogametic = false;
    std::optional<std::pair<int, int>> part;  // {total_parts, cur_part}
    GrmAlg alg = GrmAlg::Standard;
    bool alg_explicit = false;
    bool grm_fp16 = false;
    bool fast = false;
    bool impute_mean = false;
    std::optional<double> tile_budget_gb;
    std::optional<int> n_markers;
    std::optional<double> sparse_cutoff;

    // Hidden developer tuning options (group("") in CLI)
    std::optional<int> strip_rows;
    std::optional<GrmStripLayout> strip_layout;
    std::optional<int> gemm_col_chunk;
    std::optional<GrmDiagMode> diag_mode;
    std::optional<int> diag_block;
    std::optional<int> io_slot_mb;
    bool legacy_engine = false;
    bool profile = false;
};

struct GrmMergeOptions {
    std::string mgrm;
    int budget_gb = 0;  // 0 = default (~0.5 GB internal)
    bool grm_fp16 = false;
    bool fast = false;
};

struct GrmPruneOptions {
    std::string grm;
    std::optional<double> cutoff;
    std::optional<double> singleton;
    bool cutoff_detail = false;
    bool grm_fp16 = false;
    bool fast = false;
    std::string keep;
    std::string remove;
};

struct GrmSparseOptions {
    std::string grm;
    double cutoff = 0.05;
    std::optional<double> set_value;
    bool dense = false;
    bool grm_fp16 = false;
    bool fast = false;
    std::string keep;
    std::string remove;
};

struct GrmCombineOptions {
    std::string mgrm;
    bool grm_fp16 = false;
    bool fast = false;
    std::string keep;
    std::string remove;
};

struct PcaOptions {
    std::string grm;
    int pcs = 0;  // 0 = all eigenvalues
    std::optional<PcaApprox> approx;
    std::optional<SvdMethod> svd_method;
    std::optional<int> svd_power_iter;
    std::optional<double> grm_chunked_budget;
    bool grm_fp16 = false;
    bool fast = false;
    std::optional<EigenvecFormat> eigenvec_format;
    std::string keep;
    std::string remove;
    std::optional<uint64_t> seed;

    // Hidden developer tuning options (group("") in CLI)
    bool pca_profile = false;
    PcaMatvecMode pca_matvec_mode = PcaMatvecMode::Auto;
    int pca_fused_max_k = 128;
    int pca_j_block = 1024;
    PcaLoadMode pca_load_mode = PcaLoadMode::Auto;
    int pca_io_slot_mb = 0;   // 0 = default (256 MiB)
    int pca_tile_size = 0;    // 0 = default (8192 small-k / 4096 large-k)
};

enum class MlmaReorderQ { Auto = 0, Off = 1, On = 2 };

struct MlmaOptions {
    GenoInputOptions geno;
    SampleFilterOptions sample_filter;
    MarkerFilterOptions marker_filter;
    PhenoCovarOptions pheno_covar;
    RemlOptions reml;

    std::string load_reml;
    std::string woodbury_basis_file;
    bool save_reml = false;
    bool fast = false;
    std::optional<double> stream_budget_gb;
    std::optional<FastMlmaMode> fast_mlma;
    GenoModel model = GenoModel::Additive;
    bool log_pval = false;
    bool no_preadj_covar = false;

    // Hidden developer tuning flags
    bool mlma_profile = false;
    int mlma_p_budget_mb = 384;
    int mlma_q_rows = 8192;
    int mlma_snp_tile = 0;
    MlmaReorderQ mlma_reorder_q = MlmaReorderQ::Auto;
    int mlma_block_size = 0;
    int mlma_sample_tile = 8192;
    int mlma_ukx_budget_mb = 160;
    int mlma_tun3_slot_mb = 256;
};

struct LegacyOptions {
    std::vector<std::string> raw_args;
};

struct HelpOrVersion {};
struct GrmSubtractOptions : GrmCombineOptions {};
struct GrmUnifyOptions : GrmCombineOptions {};

using CommandPayload = std::variant<
    std::monostate,
    HelpOrVersion,
    GrmMakeOptions,
    GrmMergeOptions,
    GrmPruneOptions,
    GrmSparseOptions,
    GrmSubtractOptions,
    GrmUnifyOptions,
    PcaOptions,
    MlmaOptions,
    LegacyOptions>;

struct ParsedCli {
    GlobalOptions global;
    std::vector<std::string> raw_argv;
    CommandPayload command;
};

std::string format_version_string();

// Parse command-line arguments.
// Returns 0 if parsing succeeded (check std::holds_alternative<HelpOrVersion>(out.command)
// to exit cleanly without running an analysis), or a non-zero exit code on
// error (after printing diagnostic messages to stderr).
int parse_cli(int argc, char** argv, ParsedCli& out);

}  // namespace gcta_cli

#endif  // GCTA_CLI_HPP
