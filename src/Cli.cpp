#include "Cli.hpp"
#include "config.h"
#include "utils.hpp"

#include <CLI/CLI.hpp>

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace gcta_cli {

namespace {

const CLI::Range kOpenUnitRange(std::numeric_limits<double>::denorm_min(), std::nextafter(1.0, 0.0), "(0, 1)");
const CLI::Range kHalfOpenUnitRange(std::numeric_limits<double>::denorm_min(), 1.0, "(0, 1]");

template <typename Enum>
CLI::Validator make_enum_transformer(std::initializer_list<std::pair<std::string_view, Enum>> entries,
                                     std::string desc,
                                     bool allow_numeric = false) {
    std::vector<std::pair<std::string, Enum>> map;
    map.reserve(entries.size());
    for (const auto& [k, v] : entries) {
        map.emplace_back(CLI::detail::to_lower(std::string(k)), v);
    }
    return CLI::Validator(
        [map = std::move(map), desc, allow_numeric](std::string& input) -> std::string {
            const std::string lower = CLI::detail::to_lower(input);
            for (const auto& [k, v] : map) {
                if (lower == k || (allow_numeric && input == std::to_string(static_cast<int>(v)))) {
                    input = std::to_string(static_cast<int>(v));
                    return {};
                }
            }
            return "Check " + input + " value in " + desc + " FAILED";
        },
        std::move(desc));
}

class GctaFormatter : public CLI::Formatter {
  public:
    std::string make_expanded(const CLI::App* sub, CLI::AppFormatMode mode) const override {
        if (!sub->get_name().empty()) {
            return CLI::Formatter::make_expanded(sub, mode);
        }

        const auto min_opt = sub->get_require_option_min();
        const auto max_opt = sub->get_require_option_max();
        std::string badge;
        if (min_opt == 1 && max_opt == 1) {
            badge = " [choose exactly 1]";
        } else if (min_opt == 0 && max_opt == 1) {
            badge = " [choose at most 1]";
        } else if (min_opt > 0 && min_opt == max_opt) {
            badge = " [choose exactly " + std::to_string(min_opt) + "]";
        } else if (min_opt > 0 && max_opt > 0) {
            badge = " [choose " + std::to_string(min_opt) + ".." + std::to_string(max_opt) + "]";
        } else if (min_opt > 0) {
            badge = " [at least " + std::to_string(min_opt) + " required]";
        } else if (max_opt > 0) {
            badge = " [at most " + std::to_string(max_opt) + " allowed]";
        }

        std::vector<std::string> active_groups;
        std::vector<const CLI::Option*> single_group_opts;
        for (const std::string& grp : sub->get_groups()) {
            if (grp.empty()) {
                continue;
            }
            auto opts = sub->get_options([&grp](const CLI::Option* opt) {
                return opt->get_group() == grp && opt->nonpositional();
            });
            if (!opts.empty()) {
                active_groups.push_back(grp);
                if (active_groups.size() == 1) {
                    single_group_opts = std::move(opts);
                }
            }
        }

        if (active_groups.size() == 1) {
            return make_group(active_groups.front() + badge, false, single_group_opts);
        }
        return make_groups(sub, mode);
    }
};

enum class GenoInputMode {
    RequiredPrimary,
    OptionalPrimary,
    SampleManifest,
};

void add_global_options(CLI::App* app, GlobalOptions& global) {
    app->add_option_function<std::string>(
           "-o,--out",
           [&global](const std::string& val) {
               if (global.out.has_value() && *global.out != val) {
                   throw CLI::ValidationError("--out",
                                              "Conflicting values specified before and after subcommand ('" +
                                                  *global.out + "' vs '" + val + "').");
               }
               global.out = val;
           },
           "Output file prefix (default: output)")
        ->type_name("PREFIX")
        ->group("General");

    app->add_option_function<int>(
           "-t,--threads,--thread-num",
           [&global](const int& val) {
               if (global.threads.has_value() && *global.threads != val) {
                   throw CLI::ValidationError("--threads",
                                              "Conflicting values specified before and after subcommand (" +
                                                  std::to_string(*global.threads) + " vs " + std::to_string(val) +
                                                  ").");
               }
               global.threads = val;
           },
           "Number of OpenMP threads (default: $OMP_NUM_THREADS or 1)")
        ->check(CLI::PositiveNumber)
        ->group("General");

    app->add_flag_callback(
           "--verbose",
           [&global]() { global.verbose = true; },
           "Report peak memory usage at exit (Linux)")
        ->group("General");
}

bool has_geno_source(const GenoInputOptions& g) {
    return !g.bfile.empty() || !g.pfile.empty() || !g.bpfile.empty() ||
           !g.mbfile.empty() || !g.mpfile.empty() || !g.mbpfile.empty() ||
           !g.bgen.empty() || !g.mbgen.empty();
}

bool has_sample_manifest(const GenoInputOptions& g) {
    return !g.fam.empty() || !g.sample.empty() || has_geno_source(g);
}

void add_geno_input_options(CLI::App* sub,
                            GenoInputOptions& g,
                            GenoInputMode mode,
                            bool include_update_freq = true) {
    const bool sample_manifest_only = (mode == GenoInputMode::SampleManifest);
    const std::string primary_grp_name = sample_manifest_only ? "Sample Manifest (Genotype Fileset)" : "Genotype Input";
    const std::string override_grp_name = sample_manifest_only ? "Sample Files" : "Genotype Overrides";

    auto* primary_grp = sub->add_option_group(primary_grp_name);
    if (mode == GenoInputMode::RequiredPrimary) {
        primary_grp->require_option(1);
    } else {
        primary_grp->require_option(0, 1);
    }

    primary_grp->add_option("--bfile", g.bfile, "PLINK1 binary fileset prefix (.bed/.bim/.fam)")
        ->type_name("PREFIX")->group(primary_grp_name);
    primary_grp->add_option("--pfile", g.pfile, "PLINK2 fileset prefix (.pgen/.pvar/.psam)")
        ->type_name("PREFIX")->group(primary_grp_name);
    primary_grp->add_option("--bpfile", g.bpfile, "Mixed PLINK2/1 fileset prefix (.pgen/.bim/.fam)")
        ->type_name("PREFIX")->group(primary_grp_name);
    primary_grp->add_option("--mbfile", g.mbfile, "File listing multiple --bfile prefixes")
        ->check(CLI::ExistingFile)->group(primary_grp_name);
    primary_grp->add_option("--mpfile", g.mpfile, "File listing multiple --pfile prefixes")
        ->check(CLI::ExistingFile)->group(primary_grp_name);
    primary_grp->add_option("--mbpfile", g.mbpfile, "File listing multiple --bpfile prefixes")
        ->check(CLI::ExistingFile)->group(primary_grp_name);

    CLI::Option* o_bgen = nullptr;
    CLI::Option* o_mbgen = nullptr;
    if (!sample_manifest_only) {
        o_bgen = primary_grp->add_option("--bgen", g.bgen, "BGEN v1.2 genotype file (requires --sample)")
                     ->group(primary_grp_name);
        o_mbgen = primary_grp->add_option("--mbgen", g.mbgen, "File listing multiple .bgen files (requires --sample)")
                      ->group(primary_grp_name);
        sub->add_option("--bim", g.bim, "Override .bim marker file")
            ->check(CLI::ExistingFile)->group(override_grp_name);
    }

    sub->add_option("--fam", g.fam,
                    sample_manifest_only ? "PLINK .fam sample file" : "Override .fam sample file")
        ->check(CLI::ExistingFile)->group(override_grp_name);
    auto* o_sample = sub->add_option("--sample", g.sample, "Oxford/BGEN .sample file")
                         ->check(CLI::ExistingFile)->group(override_grp_name);

    if (o_bgen != nullptr && o_mbgen != nullptr) {
#ifdef BGEN_SUPPORT
        o_bgen->check(CLI::ExistingFile)->needs(o_sample);
        o_mbgen->check(CLI::ExistingFile)->needs(o_sample);
#else
        const CLI::Validator bgen_disabled(
            [](const std::string&) {
                return std::string("BGEN input (--bgen/--mbgen) is disabled in this build. "
                                   "Reconfigure CMake with -DBGEN_SUPPORT=ON.");
            },
            "DISABLED");
        o_bgen->check(bgen_disabled)->type_name("FILE");
        o_mbgen->check(bgen_disabled)->type_name("FILE");
#endif
    }

    if (!sample_manifest_only && include_update_freq) {
        sub->add_option("--update-freq", g.update_freq, "Override allele frequencies from file")
            ->check(CLI::ExistingFile)->group(override_grp_name);
    }
    sub->add_option("--update-sex", g.update_sex, "Update sample sex codes from file")
        ->check(CLI::ExistingFile)->group(override_grp_name);
}

void add_sample_filter_options(CLI::App* sub, SampleFilterOptions& s, bool include_sex_chr = true) {
    const std::string grp = "Sample Filtering";
    sub->add_option("--keep", s.keep, "Keep only sample IDs listed in file")
        ->check(CLI::ExistingFile)->group(grp);
    sub->add_option("--remove", s.remove, "Exclude sample IDs listed in file")
        ->check(CLI::ExistingFile)->group(grp);
    if (include_sex_chr) {
        sub->add_option("--sex-chr-file", s.sex_chr_file, "Sex chromosome & PLINK sex-code mapping file")
            ->check(CLI::ExistingFile)->group(grp);
    }
}

void add_marker_filter_options(CLI::App* sub,
                               MarkerFilterOptions& m,
                               const SampleFilterOptions& s,
                               bool include_bgen_opts = true) {
    auto* chr_grp = sub->add_option_group("Chromosome Scope");
    chr_grp->require_option(0, 1);
    chr_grp->add_option("--chr", m.chr, "Chromosome label or inclusive numeric range (1 or 2 args)")
        ->expected(1, 2)->type_name("CHR")->group("Chromosome Scope");
    chr_grp->add_flag("--autosome", m.autosome, "Include autosomes only")
        ->group("Chromosome Scope");
    chr_grp->add_flag("--autosome-sex", m.autosome_sex, "Include autosomes and sex chromosomes")
        ->group("Chromosome Scope");
    chr_grp->add_flag("--chr-homogametic", m.chr_homogametic,
                      "Include homogametic sex chromosome(s) only")
        ->group("Chromosome Scope");
    chr_grp->callback([&m, &s]() {
        if (m.chr_homogametic && m.homogametic_chr.empty() && s.sex_chr_file.empty()) {
            throw CLI::ValidationError("--chr-homogametic",
                                       "requires homogametic chromosome labels via --homogametic-chr or --sex-chr-file.");
        }
    });

    const std::string vf_grp_name = "Variant Filtering";
    auto* vf_grp = sub->add_option_group(vf_grp_name);
    vf_grp->add_option("--extract", m.extract, "Keep only variant IDs listed in file")
        ->check(CLI::ExistingFile)->group(vf_grp_name);
    vf_grp->add_option("--exclude", m.exclude, "Exclude variant IDs listed in file")
        ->check(CLI::ExistingFile)->group(vf_grp_name);
    vf_grp->add_option("--homogametic-chr", m.homogametic_chr, "Chromosome label(s) classified as homogametic (e.g. X, Z)")
        ->expected(1, -1)->type_name("CHR")->group(vf_grp_name);
    vf_grp->add_option("--heterogametic-chr", m.heterogametic_chr, "Chromosome label(s) classified as heterogametic (e.g. Y, W)")
        ->expected(1, -1)->type_name("CHR")->group(vf_grp_name);

    vf_grp->add_option("--maf", m.maf, "Minimum minor allele frequency [0, 0.5]")
        ->check(CLI::Range(0.0, 0.5, "[0, 0.5]"))->group(vf_grp_name);
    vf_grp->add_option("--max-maf", m.max_maf, "Maximum minor allele frequency [0, 0.5]")
        ->check(CLI::Range(0.0, 0.5, "[0, 0.5]"))->group(vf_grp_name);
    vf_grp->add_option("--geno", m.geno, "Maximum per-variant missingness rate [0, 1]")
        ->check(CLI::Range(0.0, 1.0, "[0, 1]"))->group(vf_grp_name);
    vf_grp->add_option("--dc", m.dc, "Dosage compensation parameter [-1, 1]")
        ->check(CLI::Range(-1.0, 1.0, "[-1, 1]"))->group(vf_grp_name);

    if (include_bgen_opts) {
        vf_grp->add_option("--info", m.info, "Minimum imputation INFO score [0, 1] (BGEN)")
            ->check(CLI::Range(0.0, 1.0, "[0, 1]"))->group(vf_grp_name);
        vf_grp->add_option("--hard-call-thresh", m.hard_call_thresh, "Hard-call probability threshold [0, 1] (BGEN, default: 0.9)")
            ->check(CLI::Range(0.0, 1.0, "[0, 1]"))->group(vf_grp_name);
        vf_grp->add_flag("--dosage-call", m.dosage_call, "Use expected dosages instead of hard calls (BGEN)")
            ->group(vf_grp_name);
    }

    vf_grp->callback([&m]() {
        const double min_maf = m.maf.value_or(0.0);
        const double max_maf = m.max_maf.value_or(0.5);
        if (min_maf > max_maf) {
            throw CLI::ValidationError("--maf",
                                       std::to_string(min_maf) + " cannot exceed --max-maf (" +
                                           std::to_string(max_maf) + ").");
        }
    });
}

void add_pheno_covar_options(CLI::App* sub, PhenoCovarOptions& p) {
    const std::string grp = "Phenotype & Covariates";
    auto* o_pheno = sub->add_option("--pheno", p.pheno, "Phenotype file (FID IID trait...)")
                        ->required()->check(CLI::ExistingFile)->group(grp);
    sub->add_option("--mpheno", p.mpheno, "1-based phenotype column index in --pheno")
        ->check(CLI::PositiveNumber)->needs(o_pheno)->group(grp);
    sub->add_flag("--weighted-phenotypes", p.weighted_phenotypes,
                  "Read per-sample residual weights from extra column in --pheno")
        ->needs(o_pheno)->group(grp);
    sub->add_option("--qcovar", p.qcovar, "Quantitative covariates file")
        ->check(CLI::ExistingFile)->group(grp);
    sub->add_option("--covar", p.covar, "Discrete/categorical covariates file")
        ->check(CLI::ExistingFile)->group(grp);
    sub->add_option("--covar-maxlevel", p.covar_maxlevel, "Maximum number of levels per discrete covariate (default: 33)")
        ->check(CLI::PositiveNumber)->group(grp);
}

struct RemlGroups {
    CLI::Option_group* model_source = nullptr;
    CLI::Option_group* reml = nullptr;
};

RemlGroups add_reml_options(CLI::App* sub, RemlOptions& r, bool add_subcmd_level_opts = true,
                            const bool* external_fast = nullptr) {
    const std::string g_src = "GRM / Model Source";
    const std::string g_grm = "GRM Loading";
    const std::string g_reml = "REML Solver";
    const std::string g_wb = "Woodbury Low-Rank Basis";

    auto* model_source = sub->add_option_group(g_src);
    model_source->require_option(1);

    auto* o_grm = model_source->add_option("--grm", r.grm, "Input GRM prefix (.grm.f16 or .grm.bin/.grm.id)")
                      ->type_name("PREFIX")->group(g_src);
    model_source->add_option("--woodbury-reuse,--reml-woodbury-reuse,--reml-woodbury-basis-reuse",
                             r.woodbury_reuse,
                             "Reuse Woodbury basis from a prior .reml file (reads no GRM)")
        ->check(CLI::ExistingFile)->group(g_src);

    if (add_subcmd_level_opts) {
        sub->add_flag("--fast", r.fast,
                      "Hyperscale fast preset: FP16 GRM + Woodbury MP basis")
            ->group(g_reml);
        sub->add_option("--woodbury-basis-file", r.woodbury_basis_file,
                        "Override basis .reml file path referenced by a thin TREF .reml")
            ->check(CLI::ExistingFile)->group(g_wb);
    }

    auto* reml = sub->add_option_group("REML & Woodbury");

    auto* o_chunked = reml->add_option("--grm-chunked-budget", r.grm_chunked_budget,
                                       "Stream GRM from disk within memory budget in GB")
                          ->check(CLI::PositiveNumber)->needs(o_grm)->type_name("GB")->group(g_grm);
    auto* o_fp16 = reml->add_flag("--grm-fp16", r.grm_fp16,
                                  "Hold packed lower-triangular GRM in IEEE fp16 in memory")
                       ->needs(o_grm)->group(g_grm);
    o_chunked->excludes(o_fp16);

    // REML solver options
    reml->add_option("--alg,--reml-alg", r.alg,
                     "REML algorithm: 0/AI (default), 1/Fisher, 2/EM")
        ->transform(make_enum_transformer<RemlAlg>(
            {{"AI", RemlAlg::AI}, {"Fisher", RemlAlg::Fisher}, {"EM", RemlAlg::EM}},
            "{0|AI,1|Fisher,2|EM}",
            /*allow_numeric=*/true))
        ->type_name("ALG")->group(g_reml);
    reml->add_option("--maxit,--reml-maxit", r.maxit,
                     "Maximum REML iterations (default: 100)")
        ->check(CLI::PositiveNumber)->group(g_reml);
    reml->add_flag("--no-constrain,--reml-no-constrain", r.no_constrain,
                   "Allow negative variance component estimates")->group(g_reml);
    auto* o_priors = reml->add_option("--priors,--reml-priors", r.priors,
                                      "Initial variance proportion priors")
                         ->expected(1, -1)->type_name("FLOAT...")->group(g_reml);
    auto* o_priors_var = reml->add_option("--priors-var,--reml-priors-var", r.priors_var,
                                          "Initial variance component priors")
                             ->expected(1, -1)->type_name("FLOAT...")->group(g_reml);
    o_priors->excludes(o_priors_var);

    reml->add_option("--diagV-adj,--reml-diagV-adj", r.diagV_adj,
                     "Diagonal V adjustment mode: 0/none (default), 1/jitter, or 2/bending")
        ->transform(make_enum_transformer<DiagVAdj>(
            {{"none", DiagVAdj::None}, {"jitter", DiagVAdj::Jitter}, {"bending", DiagVAdj::Bending}},
            "{0|none,1|jitter,2|bending}",
            /*allow_numeric=*/true))
        ->type_name("MODE")->group(g_reml);
    reml->add_flag("--no-he-start,--reml-no-HE-start", r.no_he_start,
                   "Disable Haseman-Elston starting values")->group(g_reml);
    reml->add_flag("--force-dense-V,--reml-force-dense-V", r.force_dense_v,
                   "Force storing dense V^-1 instead of Cholesky factor")->group(g_reml);
    reml->add_flag("--ai-robust,--reml-ai-robust", r.ai_robust,
                   "Curvature-based robust AI-REML convergence and step control")->group(g_reml);
    reml->add_option("--ai-robust-tol,--reml-ai-robust-tol", r.ai_robust_tol,
                     "Robust AI-REML convergence tolerance (default: 1e-4)")
        ->check(CLI::PositiveNumber)->group(g_reml);
    reml->add_option("--ai-robust-risk,--reml-ai-robust-risk", r.ai_robust_risk,
                     "Robust AI-REML step risk parameter (default: 0.01)")
        ->check(kOpenUnitRange)->group(g_reml);
    reml->add_flag("--print-trajectory,--reml-print-trajectory", r.print_trajectory,
                   "Print full REML optimization trajectory")->group(g_reml);

    auto* o_hutch = reml->add_option("--trace-hutchpp,--reml-trace-hutchpp", r.trace_hutchpp,
                                     "Use Hutch++ stochastic trace estimation [probes=100]")
                        ->expected(0, 1)->default_str("100")
                        ->check(CLI::PositiveNumber)->group(g_reml);
    reml->add_flag("--trace-hutchpp-fixed-probes,--reml-trace-hutchpp-fixed-probes",
                   r.trace_hutchpp_fixed_probes,
                   "Reuse fixed Hutch++ probes across REML iterations")
        ->needs(o_hutch)->group(g_reml);
    reml->add_option("--reml-pcg", r.reml_pcg,
                     "Exact Woodbury-preconditioned PCG + SLQ/trace AI-REML with preconditioner rank k_pc")
        ->expected(0, 1)->default_str("2000")->check(CLI::Range(10, 1000000))->needs(o_grm)->group(g_reml);
    reml->add_option("--seed", r.seed, "Random seed for rSVD / Hutch++")->group(g_reml);

    // Woodbury options
    const auto wb_validator =
        (CLI::IsMember({"MP", "EIG", "VAR"}, CLI::ignore_case) |
         (CLI::TypeValidator<int>() & CLI::PositiveNumber))
            .description("MP|EIG|VAR|k");
    reml->add_option_function<std::string>(
            "--woodbury-basis,--reml-woodbury-basis",
            [&r](const std::string& v) {
                if (v == "MP") {
                    r.woodbury_basis = WoodburyBasisSpec{WoodburyBasisSpec::Kind::MP, 0};
                } else if (v == "EIG") {
                    r.woodbury_basis = WoodburyBasisSpec{WoodburyBasisSpec::Kind::EIG, 0};
                } else if (v == "VAR") {
                    r.woodbury_basis = WoodburyBasisSpec{WoodburyBasisSpec::Kind::VAR, 0};
                } else {
                    r.woodbury_basis = WoodburyBasisSpec{WoodburyBasisSpec::Kind::Fixed, std::stoi(v)};
                }
            },
            "Woodbury low-rank basis mode: MP, EIG, VAR, or rank k [default: MP]")
        ->expected(0, 1)->default_str("MP")
        ->transform(wb_validator)
        ->needs(o_grm)->type_name("MODE")->group(g_wb);

    auto* o_wb_mp_margin = reml->add_option(
            "--woodbury-mp-margin,--reml-woodbury-basis-MP-margin",
            r.woodbury_mp_margin,
            "Marchenko-Pastur edge margin fraction (default: 0.15)")
        ->check(CLI::NonNegativeNumber)->group(g_wb);
    auto* o_wb_mp_confirm = reml->add_option(
            "--woodbury-mp-confirm,--reml-woodbury-basis-MP-confirm",
            r.woodbury_mp_confirm,
            "Confirmation eigenvalues beyond MP edge (default: 20)")
        ->check(CLI::PositiveNumber)->group(g_wb);
    auto* o_wb_eig_mass = reml->add_option(
            "--woodbury-eig-mass,--reml-woodbury-basis-EIG-mass",
            r.woodbury_eig_mass,
            "Target eigenvalue mass fraction for EIG mode (default: 0.99)")
        ->check(kHalfOpenUnitRange)->group(g_wb);
    auto* o_wb_eig_k_buffer = reml->add_option(
            "--woodbury-eig-k-buffer,--reml-woodbury-basis-EIG-k-buffer",
            r.woodbury_eig_k_buffer,
            "Extra eigenvalues added in EIG mode (default: 0)")
        ->check(CLI::NonNegativeNumber)->group(g_wb);
    auto* o_wb_var_tail = reml->add_option(
            "--woodbury-var-tail,--reml-woodbury-basis-VAR-tail",
            r.woodbury_var_tail,
            "Max relative Frobenius tail error for VAR mode (default: 0.001)")
        ->check(kOpenUnitRange)->group(g_wb);
    auto* o_wb_range = reml->add_option_function<std::pair<int, int>>(
            "--woodbury-range,--reml-woodbury-basis-range",
            [&r](const std::pair<int, int>& p) {
                if (p.first > p.second) {
                    throw CLI::ValidationError("--woodbury-range", "<init> must be <= <max>.");
                }
                r.woodbury_range = p;
            },
            "Initial and maximum rank for adaptive Woodbury basis (default: 2000 25000)")
        ->check(CLI::PositiveNumber)
        ->type_name("INIT MAX")->group(g_wb);
    auto* o_wb_mem_budget = reml->add_option(
            "--woodbury-mem-budget,--reml-woodbury-basis-mem-budget",
            r.woodbury_mem_budget_gb,
            "Hard memory budget (GB) capping rSVD sketch rank")
        ->check(CLI::PositiveNumber)
        ->type_name("GB")->group(g_wb);
    auto* o_wb_warm_start = reml->add_option(
            "--woodbury-warm-start,--reml-woodbury-basis-warm-start",
            r.woodbury_warm_start,
            "PCA output prefix (<prefix>.eigenvec) to warm-start Woodbury rSVD")
        ->type_name("PREFIX")->group(g_wb);
    auto* o_wb_posthoc = reml->add_flag(
            "--woodbury-posthoc-correction,--reml-woodbury-basis-posthoc-correction",
            r.woodbury_posthoc_correction,
            "Experimental post-hoc Woodbury tail correction")
        ->group(g_wb);
    auto* o_svd_method = reml->add_option(
            "--svd-method", r.svd_method,
            "Randomized SVD method for Woodbury basis: power (default) or nystrom")
        ->transform(make_enum_transformer<SvdMethod>(
            {{"power", SvdMethod::Power}, {"nystrom", SvdMethod::Nystrom}},
            "{power,nystrom}"))
        ->type_name("METHOD")->group(g_wb);
    auto* o_svd_power_iter = reml->add_option(
            "--svd-power-iter", r.svd_power_iter,
            "Number of rSVD power iterations (default: 3)")
        ->check(CLI::NonNegativeNumber)->group(g_wb);
    reml->add_flag("--reml-standalone", r.reml_standalone,
                   "Write full standalone .reml instead of thin TREF reference on --woodbury-reuse")
        ->group(g_wb);

    // Hidden developer tuning flags (work on both REML and Woodbury reuse paths)
    reml->add_flag("--reml-profile", r.reml_profile,
                   "Dev: Log detailed sub-phase timing and counter breakdown for reml")
        ->group("");
    reml->add_option("--reml-matvec-band-rows", r.reml_matvec_band_rows,
                     "Dev: Row-band height H for strip-coalesced PackedGrmFp16 matvec (0 = auto)")
        ->check(CLI::NonNegativeNumber)
        ->group("");
    reml->add_option("--reml-matvec-col-chunk", r.reml_matvec_col_chunk,
                     "Dev: Off-diagonal column chunk width C for PackedGrmFp16 matvec (0 = auto)")
        ->check(CLI::NonNegativeNumber)
        ->group("");
    reml->add_option("--reml-qr-power-passes", r.reml_qr_power_passes,
                     "Dev: CholeskyQR passes on intermediate rSVD power iterations (0 = auto, 1 or 2)")
        ->check(CLI::Range(0, 2))
        ->group("");
    reml->add_flag("--reml-eigh-fp32", r.reml_eigh_fp32,
                   "Dev: Use float32 ssyevd instead of double dsyevd on Rayleigh-Ritz Gram")
        ->group("");
    reml->add_option("--reml-eigh-threads", r.reml_eigh_threads,
                     "Dev: Cap worker threads during Rayleigh-Ritz symmetric eigensolve (0 = all)")
        ->check(CLI::NonNegativeNumber)
        ->group("");
    reml->add_option("--reml-io-slot-mb", r.reml_io_slot_mb,
                     "Dev: Async I/O slot size in MiB for GRM load and TUN3 save/reuse")
        ->check(CLI::PositiveNumber)
        ->group("");
    reml->add_option("--reml-reuse-stream", r.reml_reuse_stream,
                     "Dev: Zero-Uk streaming TREF reuse policy {auto,off,on}")
        ->transform(make_enum_transformer<RemlReuseStream>(
            {{"auto", RemlReuseStream::Auto},
             {"off", RemlReuseStream::Off},
             {"on", RemlReuseStream::On}},
            "RemlReuseStream", /*allow_numeric=*/false))
        ->group("");
    reml->add_option("--reml-woodbury-solver", r.reml_woodbury_solver,
                     "Dev: Woodbury REML solver: auto (default), cholesky (unrotated, no dsyevd), or eigh")
        ->transform(make_enum_transformer<RemlWoodburySolver>(
            {{"auto", RemlWoodburySolver::Auto},
             {"cholesky", RemlWoodburySolver::Cholesky},
             {"eigh", RemlWoodburySolver::Eigh}},
            "{auto,cholesky,eigh}"))
        ->group("");
    reml->add_option("--reml-qr-mode", r.reml_qr_mode,
                     "Dev: CholeskyQR mode {auto,sketched,standard}")
        ->transform(make_enum_transformer<RemlQrMode>(
            {{"auto", RemlQrMode::Auto},
             {"sketched", RemlQrMode::Sketched},
             {"standard", RemlQrMode::Standard}},
            "{auto,sketched,standard}",
            /*allow_numeric=*/false))
        ->group("");
    reml->add_option("--reml-svd-filter", r.reml_svd_filter,
                     "Dev: rSVD subspace filter {power,chebyshev}")
        ->transform(make_enum_transformer<RemlSvdFilter>(
            {{"power", RemlSvdFilter::Power},
             {"chebyshev", RemlSvdFilter::Chebyshev}},
            "{power,chebyshev}",
            /*allow_numeric=*/false))
        ->group("");

    const std::vector<CLI::Option*> wb_tuning_opts = {
        o_wb_mp_margin, o_wb_mp_confirm, o_wb_eig_mass, o_wb_eig_k_buffer,
        o_wb_var_tail, o_wb_range, o_wb_mem_budget, o_wb_warm_start,
        o_wb_posthoc, o_svd_method, o_svd_power_iter
    };

    reml->callback([&r, o_grm, external_fast, wb_tuning_opts]() {
        const bool fast_enabled = r.fast || (external_fast != nullptr && *external_fast);
        const bool wb_from_grm = (o_grm->count() > 0) && (r.woodbury_basis.has_value() || fast_enabled);
        for (const CLI::Option* opt : wb_tuning_opts) {
            if (opt->count() > 0 && !wb_from_grm) {
                if (o_grm->count() == 0) {
                    throw CLI::ValidationError(
                        opt->get_name(),
                        "requires --grm with --woodbury-basis or --fast (incompatible with --woodbury-reuse).");
                }
                throw CLI::ValidationError(opt->get_name(), "requires --woodbury-basis or --fast.");
            }
        }
        if ((wb_from_grm || !r.woodbury_reuse.empty()) && r.alg == RemlAlg::Fisher) {
            throw CLI::ValidationError("--alg",
                                       "Woodbury basis is incompatible with Fisher-scoring REML (--alg 1). "
                                       "Use AI-REML (default) or EM-REML (--alg 2).");
        }
    });

    return {model_source, reml};
}

std::string legacy_flat_flag_hint(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view tok(argv[i]);
        if (tok == "--make-grm" || tok == "--make-grm-part" ||
            tok == "--make-grm-d" || tok == "--make-grm-d-part" ||
            tok == "--make-grm-homogametic" || tok == "--make-grm-homogametic-part") {
            return "GCTAg now uses subcommands. Use 'GCTAg grm make ...' (run 'GCTAg grm make --help').";
        }
        if (tok == "--merge-grms") {
            return "GCTAg now uses subcommands. Use 'GCTAg grm merge ...' (run 'GCTAg grm merge --help').";
        }
        if (tok == "--grm-cutoff" || tok == "--grm-singleton") {
            return "GCTAg now uses subcommands. Use 'GCTAg grm prune ...' (run 'GCTAg grm prune --help').";
        }
        if (tok == "--make-bK-sparse" || tok == "--make-bK") {
            return "GCTAg now uses subcommands. Use 'GCTAg grm sparse ...' (run 'GCTAg grm sparse --help').";
        }
        if (tok == "--subtract-grm") {
            return "GCTAg now uses subcommands. Use 'GCTAg grm subtract ...' (run 'GCTAg grm subtract --help').";
        }
        if (tok == "--unify-grm") {
            return "GCTAg now uses subcommands. Use 'GCTAg grm unify ...' (run 'GCTAg grm unify --help').";
        }
        if (tok == "--pca" || tok == "--pca-stream") {
            return "GCTAg now uses subcommands. Use 'GCTAg pca ...' (run 'GCTAg pca --help').";
        }
        if (tok == "--mlma-stream") {
            return "GCTAg now uses subcommands. Use 'GCTAg mlma ...' (or 'GCTAg reml ...' for REML-only; run 'GCTAg mlma --help').";
        }
        if (tok.starts_with("--fastGWA") || tok == "--ld-matrix" ||
            tok == "--freq" || tok == "--make-bed" || tok == "--recodet" || tok == "--sum-geno-x") {
            return "The V2 '" + std::string(tok) + "' mode is temporarily disabled in this build (active V2 subcommands: grm, pca, reml, mlma, legacy).";
        }
        if (tok == "--reml" || tok == "--reml-bivar" || tok == "--mlma" || tok == "--mlma-loco" ||
            tok == "--pca-v1" || tok == "--he-reg" || tok == "--he-reg-bivar" ||
            tok.starts_with("--cojo-") || tok.starts_with("--gsmr") || tok.starts_with("--mtcojo") ||
            tok.starts_with("--simu-") || tok.starts_with("--fastBAT") || tok.starts_with("--mBAT")) {
            return "Legacy GCTA v1 analyses run under the 'legacy' subcommand: 'GCTAg legacy "
                   + std::string(tok) + " ...' (or use V2 subcommands 'GCTAg reml' / 'GCTAg mlma' / 'GCTAg pca').";
        }
    }
    return {};
}

}  // namespace

std::string format_version_string() {
    std::ostringstream oss;
    oss << "GCTAg " << GCTA_VERSION << " (" << getOSName() << ")\n";
    oss << "Built " << __DATE__ << " " << __TIME__;
#if defined(__clang__)
    oss << " with Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
    oss << " with GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#endif
#ifdef GCTA_TARGET_ARCH_STR
    oss << "\nTarget arch: " << GCTA_TARGET_ARCH_STR;
#endif
#ifdef GCTA_BLAS_BACKEND_STR
    oss << "\nBLAS/LAPACK: " << GCTA_BLAS_BACKEND_STR;
#ifdef GCTA_BLAS_VERSION_STR
    oss << " (" << GCTA_BLAS_VERSION_STR << ")";
#endif
#endif
    return oss.str();
}

int parse_cli(int argc, char** argv, ParsedCli& out) {
    out = ParsedCli{};
    out.raw_argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        out.raw_argv.emplace_back(argv[i]);
    }

    CLI::App app{
        "GCTAg — Genome-wide Complex Trait Analysis (Agriculture)\n"
        "Run 'GCTAg <subcommand> --help' for subcommand options.",
        "GCTAg"
    };
    auto fmt = std::make_shared<GctaFormatter>();
    fmt->column_width(36);
    app.formatter(fmt);
    app.set_version_flag("-v,--version", format_version_string(), "Display version and build information");
    app.require_subcommand(1);
    add_global_options(&app, out.global);

    GrmMakeOptions grm_make_opts;
    GrmMergeOptions grm_merge_opts;
    GrmPruneOptions grm_prune_opts;
    GrmSparseOptions grm_sparse_opts;
    GrmSubtractOptions grm_sub_opts;
    GrmUnifyOptions grm_uni_opts;
    PcaOptions pca_opts;
    MlmaOptions reml_opts;
    MlmaOptions mlma_opts;

    // ------------------------------------------------------------------
    // 1. grm (group)
    // ------------------------------------------------------------------
    auto* grm_cmd = app.add_subcommand("grm", "Build and manipulate Genetic Relationship Matrices");
    grm_cmd->require_subcommand(1);
    add_global_options(grm_cmd, out.global);

    // 1a. grm make
    auto* grm_make = grm_cmd->add_subcommand("make", "Compute a GRM from genotype data");
    add_global_options(grm_make, out.global);
    add_geno_input_options(grm_make, grm_make_opts.geno, GenoInputMode::RequiredPrimary);
    add_sample_filter_options(grm_make, grm_make_opts.sample_filter);
    add_marker_filter_options(grm_make, grm_make_opts.marker_filter, grm_make_opts.sample_filter);

    auto* gm_dom = grm_make->add_flag("--dominance,--make-grm-d", grm_make_opts.dominance,
                                      "Compute dominance GRM")->group("GRM Build");
    auto* gm_hom = grm_make->add_flag("--homogametic,--make-grm-homogametic", grm_make_opts.homogametic,
                                      "Compute homogametic sex-chromosome GRM")->group("GRM Build");
    gm_dom->excludes(gm_hom);

    auto* gm_part = grm_make->add_option_function<std::pair<int, int>>(
                "--part,--make-grm-part",
                [&grm_make_opts](const std::pair<int, int>& p) {
                    if (p.second > p.first) {
                        throw CLI::ValidationError("--part", "<part_index> must be <= <total_parts>.");
                    }
                    grm_make_opts.part = p;
                },
                "Partitioned GRM build: <total_parts> <part_index> (1-based)")
        ->check(CLI::PositiveNumber)->type_name("TOTAL INDEX")->group("GRM Build");
    auto* gm_alg = grm_make->add_option("--alg,--make-grm-alg", grm_make_opts.alg,
                         "GRM standardization algorithm: 0/standard (default) or 1/vanraden")
        ->transform(make_enum_transformer<GrmAlg>(
            {{"standard", GrmAlg::Standard}, {"vanraden", GrmAlg::VanRaden}},
            "{0,1}",
            /*allow_numeric=*/true))
        ->type_name("ALG")->group("GRM Build");
    grm_make->add_option("--tile-budget,--GRM-tile-budget", grm_make_opts.tile_budget_gb,
                         "Soft memory budget (GB) for automatic single-call GRM tiling")
        ->check(CLI::PositiveNumber)->type_name("GB")->group("GRM Build");
    grm_make->add_option("--n-markers,--nMarkers", grm_make_opts.n_markers,
                         "Markers per BLAS block (default: auto for BED VanRaden, 1024 otherwise)")
        ->check(CLI::PositiveNumber)->group("GRM Build");
    auto* gm_sparse = grm_make->add_option("--sparse-cutoff", grm_make_opts.sparse_cutoff,
                         "Emit sparse GRM entries above cutoff [default: 0.05]")
        ->expected(0, 1)->default_str("0.05")
        ->check(CLI::Range(0.0, 1.0, "[0, 1]"))->group("GRM Build");
    auto* gm_fp16 = grm_make->add_flag("--grm-fp16", grm_make_opts.grm_fp16,
                         "Write compact native FP16 GRM (<out>.grm.f16, exact float32 diagonal)")
        ->group("GRM Build");
    grm_make->add_flag("--fast", grm_make_opts.fast,
                       "Hyperscale fast preset: VanRaden INT8 (--alg 1) + native FP16 GRM (--grm-fp16)")
        ->group("GRM Build");
    grm_make->add_flag("--impute-mean,--grm-impute-mean", grm_make_opts.impute_mean,
                       "Mean-impute missing genotypes in INT8 VanRaden GRM (keeps 4 B/elem int32 path; omits .grm.N.bin for .grm.f16)")
        ->group("GRM Build");
    gm_fp16->excludes(gm_sparse, gm_part);

    // Hidden developer tuning options (omitted from --help via group(""))
    const CLI::Validator mult_of_64_validator(
        [](const std::string& input) -> std::string {
            try {
                size_t idx = 0;
                const long long v = std::stoll(input, &idx, 10);
                if (idx != input.size() || v <= 0 || (v % 64) != 0) {
                    return "Value must be a positive multiple of 64";
                }
            } catch (...) {
                return "Value must be a positive multiple of 64";
            }
            return {};
        },
        "POSITIVE_MULT_64");
    grm_make->add_option("--grm-strip-rows", grm_make_opts.strip_rows,
                         "Dev: GRM strip height in rows (multiple of 64)")
        ->check(CLI::PositiveNumber)->check(mult_of_64_validator)->group("");
    grm_make->add_option("--grm-layout", grm_make_opts.strip_layout,
                         "Dev: GRM accumulator strip layout {rowmajor,colmajor}")
        ->transform(make_enum_transformer<GrmStripLayout>(
            {{"colmajor", GrmStripLayout::ColMajor}, {"rowmajor", GrmStripLayout::RowMajor}},
            "{rowmajor,colmajor}",
            /*allow_numeric=*/false))
        ->group("");
    grm_make->add_option("--grm-gemm-col-chunk", grm_make_opts.gemm_col_chunk,
                         "Dev: Max columns per off-diagonal strip GEMM call (0 = unchunked)")
        ->check(CLI::NonNegativeNumber)->group("");
    grm_make->add_option("--grm-diag-mode", grm_make_opts.diag_mode,
                         "Dev: Strip diagonal accumulation mode {fused,syrk}")
        ->transform(make_enum_transformer<GrmDiagMode>(
            {{"syrk", GrmDiagMode::Syrk}, {"fused", GrmDiagMode::Fused}},
            "{fused,syrk}",
            /*allow_numeric=*/false))
        ->group("");
    grm_make->add_option("--grm-diag-block", grm_make_opts.diag_block,
                         "Dev: SYRK diagonal sub-block size")
        ->check(CLI::PositiveNumber)->group("");
    grm_make->add_option("--grm-io-slot-mb", grm_make_opts.io_slot_mb,
                         "Dev: AsyncSeqWriter output slot size in MiB")
        ->check(CLI::PositiveNumber)->group("");
    grm_make->add_flag("--grm-legacy-engine", grm_make_opts.legacy_engine,
                       "Dev: Bypass clean-room BED INT8 engine and use legacy GRM::processMakeGRM")
        ->group("");
    grm_make->add_flag("--grm-profile", grm_make_opts.profile,
                       "Dev: Log detailed phase timing breakdown for grm make")
        ->group("");
    grm_make->callback([&out, &grm_make_opts, gm_alg]() {
        grm_make_opts.alg_explicit = (gm_alg->count() > 0);
        if (grm_make_opts.fast) {
            if (!grm_make_opts.sparse_cutoff.has_value() && !grm_make_opts.part.has_value()) {
                grm_make_opts.grm_fp16 = true;
            }
            if (!grm_make_opts.alg_explicit && !grm_make_opts.dominance && !grm_make_opts.homogametic) {
                grm_make_opts.alg = GrmAlg::VanRaden;
            }
        }
        if (grm_make_opts.grm_fp16 && grm_make_opts.sparse_cutoff.has_value()) {
            throw CLI::ValidationError("--grm-fp16", "cannot be combined with --sparse-cutoff.");
        }
        if (grm_make_opts.grm_fp16 && grm_make_opts.part.has_value()) {
            throw CLI::ValidationError("--grm-fp16", "cannot be combined with --part (use --tile-budget instead).");
        }
        if (grm_make_opts.impute_mean) {
            if (grm_make_opts.dominance || grm_make_opts.homogametic) {
                throw CLI::ValidationError("--impute-mean", "is only supported for autosomal additive VanRaden GRM (--alg 1 or --fast).");
            }
            if (grm_make_opts.alg != GrmAlg::VanRaden) {
                throw CLI::ValidationError("--impute-mean", "requires --alg 1 (vanraden) or --fast.");
            }
            if (grm_make_opts.legacy_engine) {
                throw CLI::ValidationError("--impute-mean", "cannot be combined with --grm-legacy-engine.");
            }
            if (grm_make_opts.geno.bfile.empty() && grm_make_opts.geno.mbfile.empty()) {
                throw CLI::ValidationError("--impute-mean", "requires PLINK1 BED input (--bfile or --mbfile).");
            }
            if (!grm_make_opts.geno.update_freq.empty()) {
                throw CLI::ValidationError("--impute-mean", "cannot be combined with --update-freq.");
            }
        }
        out.command = std::move(grm_make_opts);
    });

    // 1b. grm merge
    auto* grm_merge = grm_cmd->add_subcommand("merge", "Merge multiple GRMs via streaming chunks");
    add_global_options(grm_merge, out.global);
    grm_merge->add_option("--mgrm", grm_merge_opts.mgrm, "File listing GRM prefixes to merge")
        ->required()->check(CLI::ExistingFile)->group("GRM Merge");
    grm_merge->add_option("--budget", grm_merge_opts.budget_gb,
                          "Streaming memory budget in GB (0 = default ~0.5 GB)")
        ->check(CLI::NonNegativeNumber)->group("GRM Merge");
    grm_merge->add_flag("--grm-fp16", grm_merge_opts.grm_fp16,
                        "Write compact native FP16 GRM (<out>.grm.f16)")->group("GRM Merge");
    grm_merge->add_flag("--fast", grm_merge_opts.fast,
                        "Hyperscale fast preset: native FP16 GRM output (--grm-fp16)")->group("GRM Merge");
    grm_merge->callback([&out, &grm_merge_opts]() {
        if (grm_merge_opts.fast) {
            grm_merge_opts.grm_fp16 = true;
        }
        out.command = std::move(grm_merge_opts);
    });

    // 1c. grm prune
    auto* grm_prune = grm_cmd->add_subcommand("prune", "Prune related individuals from a GRM");
    add_global_options(grm_prune, out.global);
    grm_prune->add_option("--grm", grm_prune_opts.grm, "Input GRM prefix")
        ->required()->type_name("PREFIX")->group("GRM Prune");
    auto* prune_mode = grm_prune->add_option_group("Pruning Mode");
    prune_mode->require_option(1);
    prune_mode->add_option("--cutoff,--grm-cutoff", grm_prune_opts.cutoff,
                           "Relatedness cutoff threshold (outputs pruned GRM)")
        ->group("Pruning Mode");
    prune_mode->add_option("--singleton,--grm-singleton", grm_prune_opts.singleton,
                           "Relatedness cutoff threshold (outputs singletons/families only)")
        ->group("Pruning Mode");
    grm_prune->add_flag("--cutoff-detail", grm_prune_opts.cutoff_detail,
                        "Write .family.txt and .singleton.txt details")->group("GRM Prune");
    grm_prune->add_flag("--grm-fp16", grm_prune_opts.grm_fp16,
                        "Write compact native FP16 GRM (<out>.grm.f16)")->group("GRM Prune");
    grm_prune->add_flag("--fast", grm_prune_opts.fast,
                        "Hyperscale fast preset: native FP16 GRM output (--grm-fp16)")->group("GRM Prune");
    grm_prune->add_option("--keep", grm_prune_opts.keep, "Keep only sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    grm_prune->add_option("--remove", grm_prune_opts.remove, "Exclude sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    grm_prune->callback([&out, &grm_prune_opts]() {
        if (grm_prune_opts.fast && !grm_prune_opts.singleton.has_value()) {
            grm_prune_opts.grm_fp16 = true;
        }
        if (grm_prune_opts.grm == out.global.out_prefix()) {
            throw CLI::ValidationError("--out",
                                       "input --grm and output --out cannot have the same prefix ('" +
                                           out.global.out_prefix() + "').");
        }
        out.command = std::move(grm_prune_opts);
    });

    // 1d. grm sparse
    auto* grm_sparse = grm_cmd->add_subcommand("sparse", "Build a sparse or thresholded block-diagonal GRM");
    add_global_options(grm_sparse, out.global);
    grm_sparse->add_option("--grm", grm_sparse_opts.grm, "Input GRM prefix")
        ->required()->type_name("PREFIX")->group("GRM Sparse");
    grm_sparse->add_option("--cutoff", grm_sparse_opts.cutoff, "Off-diagonal relatedness threshold")
        ->required()->group("GRM Sparse");
    grm_sparse->add_option("--set-value", grm_sparse_opts.set_value,
                           "Replacement value for elements below cutoff (optional)")
        ->group("GRM Sparse");
    grm_sparse->add_flag("--dense", grm_sparse_opts.dense,
                         "Write dense binary GRM (--make-bK) instead of sparse .grm.sp (--make-bK-sparse)")
        ->group("GRM Sparse");
    grm_sparse->add_flag("--grm-fp16", grm_sparse_opts.grm_fp16,
                         "Write compact native FP16 GRM (<out>.grm.f16) when --dense is set")
        ->group("GRM Sparse");
    grm_sparse->add_flag("--fast", grm_sparse_opts.fast,
                         "Hyperscale fast preset: native FP16 GRM output (--grm-fp16) when --dense is set")
        ->group("GRM Sparse");
    grm_sparse->add_option("--keep", grm_sparse_opts.keep, "Keep only sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    grm_sparse->add_option("--remove", grm_sparse_opts.remove, "Exclude sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    grm_sparse->callback([&out, &grm_sparse_opts]() {
        if (grm_sparse_opts.fast && grm_sparse_opts.dense) {
            grm_sparse_opts.grm_fp16 = true;
        }
        if (grm_sparse_opts.dense && grm_sparse_opts.grm == out.global.out_prefix()) {
            throw CLI::ValidationError("--out",
                                       "input --grm and output --out cannot have the same prefix when --dense is set.");
        }
        out.command = std::move(grm_sparse_opts);
    });

    // 1e. grm subtract
    auto* grm_subtract = grm_cmd->add_subcommand("subtract", "Subtract the second GRM from the first (for LOCO)");
    add_global_options(grm_subtract, out.global);
    grm_subtract->add_option("--mgrm", grm_sub_opts.mgrm, "File listing 2 GRM prefixes (full first, subset second)")
        ->required()->check(CLI::ExistingFile)->group("GRM Subtract");
    grm_subtract->add_flag("--grm-fp16", grm_sub_opts.grm_fp16,
                           "Write compact native FP16 GRM (<out>.grm.f16)")->group("GRM Subtract");
    grm_subtract->add_flag("--fast", grm_sub_opts.fast,
                           "Hyperscale fast preset: native FP16 GRM output (--grm-fp16)")->group("GRM Subtract");
    grm_subtract->add_option("--keep", grm_sub_opts.keep, "Keep only sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    grm_subtract->add_option("--remove", grm_sub_opts.remove, "Exclude sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    grm_subtract->callback([&out, &grm_sub_opts]() {
        if (grm_sub_opts.fast) {
            grm_sub_opts.grm_fp16 = true;
        }
        out.command = std::move(grm_sub_opts);
    });

    // 1f. grm unify
    auto* grm_unify = grm_cmd->add_subcommand("unify", "Intersect and align sample sets across multiple GRMs");
    add_global_options(grm_unify, out.global);
    grm_unify->add_option("--mgrm", grm_uni_opts.mgrm, "File listing GRM prefixes to unify")
        ->required()->check(CLI::ExistingFile)->group("GRM Unify");
    grm_unify->add_flag("--grm-fp16", grm_uni_opts.grm_fp16,
                        "Write compact native FP16 GRM (<out>.grm.f16)")->group("GRM Unify");
    grm_unify->add_flag("--fast", grm_uni_opts.fast,
                        "Hyperscale fast preset: native FP16 GRM output (--grm-fp16)")->group("GRM Unify");
    grm_unify->add_option("--keep", grm_uni_opts.keep, "Keep only sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    grm_unify->add_option("--remove", grm_uni_opts.remove, "Exclude sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    grm_unify->callback([&out, &grm_uni_opts]() {
        if (grm_uni_opts.fast) {
            grm_uni_opts.grm_fp16 = true;
        }
        out.command = std::move(grm_uni_opts);
    });

    // ------------------------------------------------------------------
    // 2. pca
    // ------------------------------------------------------------------
    auto* pca_cmd = app.add_subcommand("pca", "Principal Component Analysis on a GRM");
    add_global_options(pca_cmd, out.global);
    pca_cmd->add_option("--grm", pca_opts.grm, "Input GRM prefix (.grm.f16 or .grm.bin/.grm.id)")
        ->required()->type_name("PREFIX")->group("PCA");
    pca_cmd->add_option("--pcs,--n-pcs", pca_opts.pcs, "Number of top PCs to compute (0 = all eigenvalues)")
        ->check(CLI::NonNegativeNumber)->group("PCA");
    pca_cmd->add_option("--approx,--pca-approx", pca_opts.approx,
                        "Approximate eigensolver: Lanczos or rSVD [default: Lanczos]")
        ->expected(0, 1)->default_str("Lanczos")
        ->transform(make_enum_transformer<PcaApprox>(
            {{"Lanczos", PcaApprox::Lanczos}, {"rSVD", PcaApprox::rSVD}},
            "{Lanczos,rSVD}"))
        ->type_name("METHOD")->group("PCA");
    pca_cmd->add_option("--svd-method", pca_opts.svd_method,
                        "rSVD sketch method: power (default) or nystrom")
        ->transform(make_enum_transformer<SvdMethod>(
            {{"power", SvdMethod::Power}, {"nystrom", SvdMethod::Nystrom}},
            "{power,nystrom}"))
        ->type_name("METHOD")->group("PCA");
    pca_cmd->add_option("--svd-power-iter", pca_opts.svd_power_iter,
                        "Number of rSVD power iterations (default: 3)")
        ->check(CLI::NonNegativeNumber)->group("PCA");
    pca_cmd->add_option("--eigenvec-format", pca_opts.eigenvec_format,
                        "Eigenvector output format: text (default), bin, bin16, or both")
        ->transform(make_enum_transformer<EigenvecFormat>(
            {{"text", EigenvecFormat::Text},
             {"bin", EigenvecFormat::Bin32},
             {"bin16", EigenvecFormat::Bin16},
             {"both", EigenvecFormat::Both}},
            "{text,bin,bin16,both}"))
        ->type_name("FMT")->group("PCA");
    pca_cmd->add_flag("--fast", pca_opts.fast,
                      "Hyperscale fast preset: rSVD + packed FP16 GRM + compact binary eigenvectors")
        ->group("PCA");
    auto* pca_chunk = pca_cmd->add_option("--grm-chunked-budget", pca_opts.grm_chunked_budget,
                                          "Stream GRM from disk within memory budget in GB")
                          ->check(CLI::PositiveNumber)->type_name("GB")->group("GRM Loading");
    auto* pca_fp16 = pca_cmd->add_flag("--grm-fp16", pca_opts.grm_fp16,
                                       "Hold packed lower-triangular GRM in IEEE fp16 in memory")
                         ->group("GRM Loading");
    pca_chunk->excludes(pca_fp16);
    pca_cmd->add_option("--keep", pca_opts.keep, "Keep only sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    pca_cmd->add_option("--remove", pca_opts.remove, "Exclude sample IDs listed in file")
        ->check(CLI::ExistingFile)->group("Sample Filtering");
    pca_cmd->add_option("--seed", pca_opts.seed, "Random seed for rSVD")->group("PCA");
    // Hidden developer tuning flags
    pca_cmd->add_flag("--pca-profile", pca_opts.pca_profile,
                      "Dev: Log detailed phase timing breakdown for pca")
        ->group("");
    pca_cmd->add_option("--pca-matvec-mode", pca_opts.pca_matvec_mode,
                        "Dev: Packed FP16 matvec mode {auto,fused,tiled}")
        ->transform(make_enum_transformer<PcaMatvecMode>(
            {{"auto", PcaMatvecMode::Auto},
             {"fused", PcaMatvecMode::Fused},
             {"tiled", PcaMatvecMode::Tiled}},
            "{auto,fused,tiled}",
            /*allow_numeric=*/false))
        ->group("");
    pca_cmd->add_option("--pca-fused-max-k", pca_opts.pca_fused_max_k,
                        "Dev: Max X width for auto fused FP16 matvec")
        ->check(CLI::NonNegativeNumber)->group("");
    pca_cmd->add_option("--pca-j-block", pca_opts.pca_j_block,
                        "Dev: Column slab size JB inside fused FP16 matvec")
        ->check(CLI::PositiveNumber)->group("");
    pca_cmd->add_option("--pca-load-mode", pca_opts.pca_load_mode,
                        "Dev: Packed FP16 GRM loader mode {auto,direct,async}")
        ->transform(make_enum_transformer<PcaLoadMode>(
            {{"auto", PcaLoadMode::Auto},
             {"direct", PcaLoadMode::Direct},
             {"async", PcaLoadMode::Async}},
            "{auto,direct,async}",
            /*allow_numeric=*/false))
        ->group("");
    pca_cmd->add_option("--pca-io-slot-mb", pca_opts.pca_io_slot_mb,
                        "Dev: GRM reader slot/read chunk size in MiB (0 = default)")
        ->check(CLI::NonNegativeNumber)->group("");
    pca_cmd->add_option("--pca-tile-size", pca_opts.pca_tile_size,
                        "Dev: Override tiled symmetric matvec block size (0 = default)")
        ->check(CLI::NonNegativeNumber)->group("");
    pca_cmd->callback([&out, &pca_opts]() { out.command = std::move(pca_opts); });

    // ------------------------------------------------------------------
    // 3. reml (V2 REML-only, writes <out>.reml)
    // ------------------------------------------------------------------
    auto* reml_cmd = app.add_subcommand("reml", "Fit V2 REML variance components and save <out>.reml");
    add_global_options(reml_cmd, out.global);
    add_geno_input_options(reml_cmd, reml_opts.geno, GenoInputMode::SampleManifest, /*include_update_freq=*/false);
    add_sample_filter_options(reml_cmd, reml_opts.sample_filter);
    add_pheno_covar_options(reml_cmd, reml_opts.pheno_covar);
    add_reml_options(reml_cmd, reml_opts.reml, /*add_subcmd_level_opts=*/true);
    reml_cmd->callback([&out, &reml_opts]() {
        if (!has_sample_manifest(reml_opts.geno)) {
            throw CLI::ValidationError("Sample Manifest",
                                       "requires one of --fam, --bfile, --pfile, --bpfile, --sample, --mbfile, --mpfile, or --mbpfile.");
        }
        reml_opts.fast = reml_opts.reml.fast;
        reml_opts.woodbury_basis_file = reml_opts.reml.woodbury_basis_file;
        reml_opts.save_reml = true;
        out.command = std::move(reml_opts);
    });

    // ------------------------------------------------------------------
    // 4. mlma (V2 streaming MLMA)
    // ------------------------------------------------------------------
    auto* mlma_cmd = app.add_subcommand("mlma", "Streaming Mixed Linear Model Association (V2)");
    add_global_options(mlma_cmd, out.global);
    add_geno_input_options(mlma_cmd, mlma_opts.geno, GenoInputMode::OptionalPrimary);
    add_sample_filter_options(mlma_cmd, mlma_opts.sample_filter);
    add_marker_filter_options(mlma_cmd, mlma_opts.marker_filter, mlma_opts.sample_filter);
    add_pheno_covar_options(mlma_cmd, mlma_opts.pheno_covar);
    auto [grm_src, reml_grp] = add_reml_options(mlma_cmd, mlma_opts.reml, /*add_subcmd_level_opts=*/false,
                                                &mlma_opts.fast);

    const std::string g_assoc = "Association Scan";
    auto* mlma_load = grm_src->add_option("--load-reml", mlma_opts.load_reml,
                                          "Load pre-fitted .reml state (skips inline REML)")
                          ->check(CLI::ExistingFile)->group("GRM / Model Source");
    reml_grp->excludes(mlma_load);

    mlma_cmd->add_option("--woodbury-basis-file", mlma_opts.woodbury_basis_file,
                         "Override basis .reml file path referenced by a thin TREF .reml")
        ->check(CLI::ExistingFile)->group("GRM / Model Source");
    mlma_cmd->add_flag("--fast", mlma_opts.fast,
                       "Hyperscale fast preset: FP16 GRM + Woodbury MP + --fast-mlma adaptive")
        ->group(g_assoc);

    auto* mlma_save = mlma_cmd->add_flag("--save-reml", mlma_opts.save_reml,
                                         "Fit REML only and save state to <out>.reml (same as 'GCTAg reml')")
                          ->group(g_assoc);
    auto* mlma_budget = mlma_cmd->add_option("--stream-budget,--mlma-stream", mlma_opts.stream_budget_gb,
                                             "Genotype streaming memory budget in GB (optional)")
                            ->check(CLI::PositiveNumber)->type_name("GB")->group(g_assoc);
    auto* mlma_fast = mlma_cmd->add_option("--fast-mlma", mlma_opts.fast_mlma,
                                           "Quantized Woodbury association mode: adaptive, 2slice, hybrid, or 1slice [default: adaptive]")
                          ->expected(0, 1)->default_str("adaptive")
                          ->transform(make_enum_transformer<FastMlmaMode>(
                              {{"2slice", FastMlmaMode::Slice2},
                               {"hybrid", FastMlmaMode::Hybrid},
                               {"1slice", FastMlmaMode::Slice1},
                               {"adaptive", FastMlmaMode::Adaptive}},
                              "{adaptive,2slice,hybrid,1slice}"))
                          ->type_name("MODE")->group(g_assoc);
    auto* mlma_model = mlma_cmd->add_option("--model", mlma_opts.model,
                                            "Genotype coding model: additive (default), nonadditive, or dominance")
                           ->transform(make_enum_transformer<GenoModel>(
                               {{"additive", GenoModel::Additive},
                                {"nonadditive", GenoModel::NonAdditive},
                                {"dominance", GenoModel::Dominance}},
                               "{additive,nonadditive,dominance}"))
                           ->type_name("MODEL")->group(g_assoc);
    auto* mlma_logp = mlma_cmd->add_flag("--log-pval", mlma_opts.log_pval,
                                         "Output -log10(p) values beyond IEEE double precision")->group(g_assoc);
    mlma_cmd->add_flag("--no-preadj-covar,--mlma-no-preadj-covar", mlma_opts.no_preadj_covar,
                       "Disable covariate pre-adjustment (currently unsupported in V2 MLMA)")
        ->group(g_assoc);

    // Hidden developer tuning flags
    mlma_cmd->add_flag("--mlma-profile", mlma_opts.mlma_profile,
                       "Dev: Log detailed phase timing breakdown for mlma")
        ->group("");
    mlma_cmd->add_option("--mlma-p-budget-mb", mlma_opts.mlma_p_budget_mb,
                         "Dev: Persistent P_i32 tile buffer budget in MiB for quantized fast-mlma")
        ->check(CLI::PositiveNumber)
        ->group("");
    mlma_cmd->add_option("--mlma-q-rows", mlma_opts.mlma_q_rows,
                         "Dev: Max sample rows per quantized Q tile for 2slice/1slice/hybrid")
        ->check(CLI::Range(16, 1048576))
        ->group("");
    mlma_cmd->add_option("--mlma-snp-tile", mlma_opts.mlma_snp_tile,
                         "Dev: Override SNP sub-tile width in fast_i8 (0 = auto from P budget)")
        ->check(CLI::NonNegativeNumber)
        ->group("");
    mlma_cmd->add_option("--mlma-reorder-q", mlma_opts.mlma_reorder_q,
                         "Dev: AOCL Q reorder policy {auto,off,on}")
        ->transform(make_enum_transformer<MlmaReorderQ>(
            {{"auto", MlmaReorderQ::Auto},
             {"off", MlmaReorderQ::Off},
             {"on", MlmaReorderQ::On}},
            "MlmaReorderQ", /*allow_numeric=*/false))
        ->group("");
    mlma_cmd->add_option("--mlma-block-size", mlma_opts.mlma_block_size,
                         "Dev: Override SNP streaming block size (0 = auto)")
        ->check(CLI::NonNegativeNumber)
        ->group("");
    mlma_cmd->add_option("--mlma-sample-tile", mlma_opts.mlma_sample_tile,
                         "Dev: Sample tile size for std_i8 GEMM/UKX path")
        ->check(CLI::PositiveNumber)
        ->group("");
    mlma_cmd->add_option("--mlma-ukx-budget-mb", mlma_opts.mlma_ukx_budget_mb,
                         "Dev: Scratch budget in MiB for std_i8 UKX output panel")
        ->check(CLI::PositiveNumber)
        ->group("");
    mlma_cmd->add_option("--mlma-tun3-slot-mb", mlma_opts.mlma_tun3_slot_mb,
                         "Dev: Pre-allocated double-buffer slot size in MiB for TUN3 loading")
        ->check(CLI::PositiveNumber)
        ->group("");

    mlma_load->excludes(mlma_save);
    mlma_save->excludes(mlma_fast, mlma_budget, mlma_model, mlma_logp);

    mlma_cmd->callback([&out, &mlma_opts]() {
        if (!mlma_opts.save_reml && !has_geno_source(mlma_opts.geno)) {
            throw CLI::ValidationError("Genotype Input",
                                       "requires one of --bfile, --pfile, --bpfile, --mbfile, --mpfile, --mbpfile, --bgen, or --mbgen.");
        }
        if (mlma_opts.save_reml && !has_sample_manifest(mlma_opts.geno)) {
            throw CLI::ValidationError("Sample Manifest",
                                       "requires one of --fam, --bfile, --pfile, --bpfile, --sample, --mbfile, --mpfile, or --mbpfile.");
        }
        if (mlma_opts.fast) {
            mlma_opts.reml.fast = true;
        }
        if (!mlma_opts.woodbury_basis_file.empty()) {
            mlma_opts.reml.woodbury_basis_file = mlma_opts.woodbury_basis_file;
        }
        out.command = std::move(mlma_opts);
    });

    // ------------------------------------------------------------------
    // 5. legacy (explicit GCTA v1 passthrough)
    // ------------------------------------------------------------------
    auto* legacy_cmd = app.add_subcommand(
        "legacy",
        "Run legacy GCTA v1 analyses (--reml, --reml-bivar, --cojo-*, --gsmr, --mtcojo, --mlma-loco, --he-reg, --pca-v1, ...)");
    legacy_cmd->prefix_command();
    legacy_cmd->footer(
        "Examples:\n"
        "  GCTAg legacy --reml --grm data --pheno data.phen --out reml_v1\n"
        "  GCTAg legacy --cojo-slct --bfile ref --cojo-file sumstats.ma --out cojo\n"
        "  GCTAg legacy --mlma-loco --bfile data --grm data --pheno data.phen --out loco\n"
        "\n"
        "All options after 'legacy' are forwarded verbatim to the legacy GCTA v1 parser.");
    legacy_cmd->callback([legacy_cmd, &out]() {
        auto raw_args = legacy_cmd->remaining();
        if (raw_args.empty()) {
            throw CLI::ValidationError("legacy",
                                       "requires legacy GCTA v1 options (run 'GCTAg legacy --help' for examples).");
        }
        if (raw_args.size() == 1 && (raw_args[0] == "-h" || raw_args[0] == "--help")) {
            throw CLI::CallForHelp();
        }
        out.command = LegacyOptions{std::move(raw_args)};
    });

    // Handle bare `GCTAg` with no arguments cleanly without creating output.log
    if (argc <= 1) {
        std::cerr << app.help();
        return 1;
    }

    try {
        app.parse(argc, argv);
    } catch (const CLI::CallForHelp& e) {
        out.command = HelpOrVersion{};
        return app.exit(e);
    } catch (const CLI::CallForAllHelp& e) {
        out.command = HelpOrVersion{};
        return app.exit(e);
    } catch (const CLI::CallForVersion& e) {
        out.command = HelpOrVersion{};
        return app.exit(e);
    } catch (const CLI::ParseError& e) {
        const int rc = app.exit(e);
        if (app.get_subcommands().empty()) {
            const std::string hint = legacy_flat_flag_hint(argc, argv);
            if (!hint.empty()) {
                std::cerr << "Hint: " << hint << "\n";
            }
        }
        return rc != 0 ? rc : 1;
    }

    return 0;
}

}  // namespace gcta_cli
