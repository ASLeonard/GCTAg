/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * GCTA options
 *
 * 2010-Present by Jian Yang <jian.yang.qt@gmail.com> and others
 *
 * This file is distributed under the GNU General Public
 * License, Version 3.  Please see the file LICENSE for more
 * details
 *
 * Mocked by Zhili.
 *
 * This folder contains most majority functions of GCTA.
 * We are moving toward new version by huge mocks on original version.
 *
 */

#include <cstdio>
#include <filesystem>
#include <random>
#include <stdlib.h>
#include "gcta.h"
#include "Logger.h"
#include "constants.hpp"
#include "utils.hpp"
#include "../src/config.h"

void option(int option_num, char* option_str[]);

int main_v1(int argc, char* argv[])
{
    LOGGER << "*******************************************************************" << std::endl;
    LOGGER << "* Genome-wide Complex Trait Analysis -- Agricuture (GCTAg)" << std::endl;
    LOGGER << "* " + std::string(GCTA_VERSION) + " (" + getOSName() + ")" << std::endl;
    LOGGER << "* v1.0.0 \"Braunvieh\"" << std::endl;
    LOGGER << "* GCTA: (C) 2010-2021, Westlake University" << std::endl;
    LOGGER << "* GCTAg: (C) 2026-present, Alexander S. Leonard, ETH Zurich" << std::endl;
    LOGGER << "* MIT License" << std::endl;
    LOGGER << "* Please report bugs at https://github.com/ASLeonard/GCTAg/issues" << std::endl;
    LOGGER << "*******************************************************************" << std::endl;

    long int time_used = 0, start = time(NULL);
    time_t curr = time(0);
    LOGGER << "Analysis started: " << ctime(&curr) << std::endl;
    LOGGER << "Options:" << std::endl;
    try {
        option(argc, argv);
    } catch (const std::string &err_msg) {
        std::cerr << "\n" << err_msg << std::endl;
    } catch (const char *err_msg) {
        std::cerr << "\n" << err_msg << std::endl;
    }
    curr = time(0);
    LOGGER << "\nAnalysis finished: " << ctime(&curr);
    time_used = time(NULL) - start;
    LOGGER << "Computational time: " << time_used / 3600 << ":" << (time_used % 3600) / 60 << ":" << time_used % 60 << std::endl;

    return 0;
}

void option(int option_num, char* option_str[])
{
    int i = 0, j = 0;

    // Subcommand detection
    std::string_view subcommand;
    int arg_offset = 1;  // Start parsing flags from argv[1]

    // OpenMP
    bool thread_flag = false;
    int thread_num = omp_get_max_threads();

    // raw genotype data
    std::string RG_fname_file = "", RG_summary_file = "";
    double GC_cutoff = 0.7;

    // data management
    std::string bfile = "", bfile2 = "", bfile_list = "", update_sex_file = "", update_freq_file = "", update_refA_file = "", kp_indi_file = "", rm_indi_file = "", extract_snp_file = "", exclude_snp_file = "", extract_snp_name = "", exclude_snp_name = "", out = "gcta";
    bool SNP_major = false, make_bed_flag = false, dose_mach_flag = false, dose_mach_gz_flag = false, dose_beagle_flag = false, bfile2_flag = false, out_freq_flag = false, out_ssq_flag = false;
    bool ref_A = false, recode = false, recode_nomiss = false, recode_std = false, save_ram = false, autosome_flag = false;
    int bfile_flag = 0, autosome_num = 0, extract_chr_start = 0, extract_chr_end = 0, extract_region_chr = 0, extract_region_bp = 0, extract_region_wind = 0, exclude_region_chr = 0, exclude_region_bp = 0, exclude_region_wind = 0;
    bool autosome_num_explicit = false;
    std::string dose_file = "", dose_info_file = "", update_impRsq_file = "";
    double maf = 0.0, max_maf = 0.0, dose_Rsq_cutoff = 0.0;
    std::vector<std::string> multi_bfiles;

    // GRM
    bool ibc = false, ibc_all = false, grm_flag = false, grm_bin_flag = true, m_grm_flag = false, m_grm_bin_flag = true, make_grm_flag = false, make_grm_inbred_flag = false, dominance_flag = false, make_grm_homogametic_flag = false, grm_out_bin_flag = true, make_grm_f3_flag = false;
    bool align_grm_flag = false;
    bool pca_flag = false, pcl_flag = false;
    std::string pca_approx_flag = "";
    bool project_flag = false;
    std::string denseness_metric = "";
    double grm_adj_fac = -2.0, grm_cutoff = -2.0, rm_high_ld_cutoff = -1.0, bK_threshold = -10.0;
    int dosage_compen = -2, out_pc_num = 0, make_grm_mtd = 0;
    std::string grm_file = "", paa_file = "", pc_file = "";
    std::string genetic_model = ""; // genetic model for dosage calculation
    //pca projection
    std::string project_file = "";
    int project_N = 0;


    // LD
    std::string LD_file = "", ld_score_multi_file = "";
    bool LD = false, LD_search = false, LD_i = false, ld_score_flag = false, ld_max_rsq_flag = false, ld_mean_rsq_seg_flag = false, ldscore_adj_flag = false;
    int LD_step = 10;
    double LD_wind = 1e7, LD_sig = 0.05, LD_prune_rsq = -1.0, LD_rsq_cutoff = 0.0, LD_seg = 1e5;

    // initialize paramters for simulation based on real genotype data
    bool simu_qt_flag = false, simu_cc = false, simu_emb_flag = false, simu_output_causal = false;
    int simu_rep = 1, simu_case_num = 0, simu_control_num = 0, simu_eff_mod = 0;
    double simu_h2 = 0.1, simu_K = 0.1, simu_gener = 100, simu_seed = -static_cast<int>(std::random_device{}() & 0x7FFFFFFFu);
    std::string simu_causal = "";

    // simulate unlinked SNPs
    bool simu_unlinked_flag = false;
    int simu_unlinked_n = 1, simu_unlinked_m = 1;
    double simu_unlinked_maf = 0.0;

    // estimate genetic distance based on hapmap_data
    bool hapmap_genet_dst = false;
    std::string hapmap_genet_dst_file = "";

    // REML analysis
    bool prevalence_flag = false, reml_force_inv_fac_flag = false, reml_force_converge_flag = false, reml_no_converge_flag = false, reml_fixed_var_flag = false;
    bool logp_flag = false;
    int mphen = 1, mphen2 = 2, reml_mtd = 0, MaxIter = 100;
    bool reml_allow_constrain_run = false;
    bool reml_trace_approx = false;
    int  reml_trace_nprobes = 90;
    int  reml_trace_power_iter = 0;  // off by default; only helps GRMs with dominant low-rank structure
    int  reml_woodbury_rank = 0;     // >0 fixed k; -1 auto; 0 disabled
    double reml_woodbury_buffer = 2.5; // MP buffer factor for auto-k
    // Rationale: MP edge answers the logdet question (λ_i vs noise floor) but the
    // REML score equation is sensitive to λᵢ², so the effective rank for quadratic
    // form accuracy is ~k_signal×√k_signal ≈ k_signal^1.5, not just k_signal.
    // buffer=2.5 approximates k_signal^1.5/k_signal = k_signal^0.5 ≈ √380 ≈ 2.5
    // for a typical livestock dataset (k_signal ≈ 380), matching empirical k~1000.
    int  reml_woodbury_k_max = 0;    // SVD rank cap for auto-k (0 → min(n-1,2000))
    bool reml_woodbury_nystrom = false;
    double prevalence = -2.0, prevalence2 = -2.0;
    bool reml_flag = false, pred_rand_eff = false, est_fix_eff = false, est_fix_eff_var = false, blup_snp_flag = false, no_constrain = false, reml_lrt_flag = false, no_lrt = false, bivar_reml_flag = false, ignore_Ce = false, within_family = false, reml_bending = false, HE_reg_flag = false, reml_diag_one = false, bivar_no_constrain = false;
    int reml_diagV_adj = 0;
    double reml_diag_mul = 0.01;
    int reml_inv_method = 0;

    bool cv_blup = false;
    bool HE_reg_bivar_flag = false;
    std::string weight_file = "";
    std::string phen_file = "", qcovar_file = "", covar_file = "", qgxe_file = "", gxe_file = "", blup_indi_file = "";
    std::vector<double> reml_priors, reml_priors_var, fixed_rg_val;
    std::vector<int> reml_drop;
    reml_drop.push_back(1);

    // Joint analysis of GWAS MA
    std::string massoc_file = "", massoc_init_snplist = "", massoc_cond_snplist = "";
    int massoc_wind = 1e7, massoc_top_SNPs = -1, massoc_mld_slct_alg = 0;
    double massoc_p = 5e-8, massoc_collinear = 0.9, massoc_sblup_fac = -1, massoc_gc_val = -1;
    bool massoc_slct_flag = false, massoc_joint_flag = false, massoc_sblup_flag = false, massoc_gc_flag = false, massoc_actual_geno_flag = false;
    double massoc_out_pC_thresh = -1;

    // mixed linear model association
    bool mlma_flag = false, mlma_loco_flag = false, mlma_no_adj_covar = false;
    bool save_reml_flag = false;
    std::string subtract_grm_file = "", save_reml_file = "", load_reml_file = "";
    std::string grm_chr_prefix = "";  // prefix for per-chromosome GRM files (--grm-chr)

    // Fst
    bool fst_flag = false;
    std::string subpopu_file = "";

    // fastBAT gene-based association test
    bool sbat_seg_flag = false;
    double sbat_ld_cutoff = sqrt(0.9); //option to remove overly correlated snps in SBAT test
    bool sbat_write_snpset = false; //write snplist - used in conjunction with sbat_ld_cutoff
    std::string sbat_sAssoc_file = "", sbat_gAnno_file = "", sbat_snpset_file = "";
    int sbat_wind = 50000, sbat_seg_size = 1e5;

    // mBAT gene-based association test
    // alstep 1 . add flag
    double mbat_svd_gamma = 0.9; //option to remove overly correlated snps in mBAT test
    bool mbat_write_snpset = false; //write snplist _ used in conjunction with mbat_ld_cutoff
    std::string mbat_sAssoc_file = "", mbat_gAnno_file = "", mbat_snpset_file = "";
    int mbat_wind = 50000;
    bool mbat_print_all_p = false;

    // gene expression data
    std::string efile="", eR_file = "", ecojo_ma_file="";
    int make_erm_mtd = 1;
    double ecojo_p = 5e-6, ecojo_collinear = 0.9, ecojo_lambda = -1;
    bool efile_flag=false, eR_file_flag = false, ecojo_slct_flag = false, ecojo_blup_flag = false, make_erm_flag = false;

    // mtCOJO
    char chbuf = '\0';
    std::string mtcojolist_file="", mtcojo_bxy_file="", ref_ld_dirt="", w_ld_dirt="";
    int nsnp_gsmr=10, nsnp_read = 0;
    double freq_thresh = 0.2, gwas_thresh=5e-8, global_heidi_thresh = 0.01, std_heidi_thresh = 0.01, ld_fdr_thresh=0.05, clump_wind_size=10000, clump_r2_thresh=0.05;
    bool mtcojo_flag=false, ref_ld_flag=false, w_ld_flag=false, global_heidi_flag=false;

    // GSMR
    bool gsmr_flag = false, o_snp_instru_flag = false, gsmr_so_flag = false, gsmr_snp_update_flag = false;
    int gsmr_beta_version = 0, gsmr_alg_flag = 0, gsmr_so_alg = -9;
    std::string expo_file_list = "", outcome_file_list = "";
    
    // Adjustment for PC
    bool gwas_data_flag = false, gwas_adj_pc_flag = false;
    int pc_adj_wind_size = 10000;
    std::string pcadjust_list_file = "";

    int argc = option_num;
    std::vector<char *> argv(option_num + 2);
    for (i = 0; i < option_num; i++) argv[i] = option_str[i];
    argv[option_num] = const_cast<char*>("gcta");
    argv[option_num + 1] = const_cast<char*>("gcta");
    /*/
    // Detect subcommand (non-flag first argument)
    if (argc > 1 && argv[1][0] != '-') {
        subcommand = argv[1];
        arg_offset = 2;  // Start parsing from argv[2]

        // Set flags based on subcommand
        if (subcommand == "mlma") {
            mlma_flag = true;
        } else {
            LOGGER.e(0, "Unknown subcommand: " + std::string(subcommand) +
                        "\nCurrently supported: mlma\nUse 'gcta -h' for help.");
        }

        LOGGER << "Subcommand: " << subcommand << endl;
    }

    // Check for deprecated --mlma flag
    for (int check = 1; check < argc; ++check) {
        if (std::string_view(argv[check]) == "--mlma") {
            LOGGER.e(0, "The --mlma flag is deprecated.\nUse: gcta mlma <options>");
        }
    }

    LOGGER << "Accepted options:" << endl;
    
    // Early exit optimization for mlma: skip irrelevant flags
    constexpr std::array<std::string_view, 18> mlma_flags = {{
        "--bfile", "--pheno", "--mpheno", "--qcovar", "--covar", "--grm",
        "--mlma-loco", "--mlma-subtract-grm", "--mlma-no-preadj-covar",
        "--save-reml", "--load-reml", "--out", "--thread-num", "--threads",
        "--reml-maxit", "--reml-priors", "--reml-priors-var",
        "--reml-no-constrain"
    }};
    */
    

    for (i = arg_offset; i < argc; i++) {
        std::string_view flag = argv[i];
        /*
        // Early exit: skip unknown flags for mlma subcommand
        if (!subcommand.empty() && subcommand == "mlma") {
            if (std::ranges::find(mlma_flags, flag) == mlma_flags.end()) {
                // Skip unknown flag and its value (if present)
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    ++i;
                }
                continue;
            }
        }*/

        if (flag == "--thread-num") {
            thread_num = std::atoi(argv[++i]);
            LOGGER << "--thread-num " << thread_num << std::endl;
            if (thread_num < 1) LOGGER.e(0, "\n  --thread-num should be >= 1.\n");
        }
        else if (flag == "--threads") {
            thread_num = std::atoi(argv[++i]);
            LOGGER << "--threads " << thread_num << std::endl;
            if (thread_num < 1) LOGGER.e(0, "\n  --threads should be >= 1.\n");
        }// raw genotype data
        else if (flag == "--raw-files") {
            RG_fname_file = argv[++i];
            LOGGER << "--raw-files " << argv[i] << endl;
        } else if (flag == "--raw-summary") {
            RG_summary_file = argv[++i];
            LOGGER << "--raw-summary " << argv[i] << endl;
        } else if (flag == "--gencall") {
            GC_cutoff = std::atof(argv[++i]);
            LOGGER << "--gencall " << GC_cutoff << std::endl;
            if (GC_cutoff < 0.0 || GC_cutoff > 1.0) LOGGER.e(0, "\n  --gencall should be within the range from 0 to 1.\n");
        }            // data management
        else if (flag == "--bfile") {
            bfile_flag = 1;
            bfile = argv[++i];
            LOGGER << "--bfile " << argv[i] << endl;
        } else if (flag == "--mbfile") {
            bfile_flag = 2;
            bfile_list = argv[++i];
            LOGGER << "--mbfile " << argv[i] << endl;
        } else if (flag == "--make-bed") {
            make_bed_flag = true;
            LOGGER << "--make-bed " << endl;
        } else if (flag == "--bfile2") {
            bfile2_flag = true;
            bfile2 = argv[++i];
            LOGGER << "--bfile2 " << argv[i] << endl;
        } else if (flag == "--dosage-mach") {
            dose_mach_flag = true;
            dose_beagle_flag = false;
            dose_file = argv[++i];
            dose_info_file = argv[++i];
            LOGGER << "--dosage-mach " << dose_file << " " << dose_info_file << std::endl;
        } else if (flag == "--dosage-mach-gz") {
            dose_mach_gz_flag = true;
            dose_beagle_flag = false;
            dose_file = argv[++i];
            dose_info_file = argv[++i];
            LOGGER << "--dosage-mach-gz " << dose_file << " " << dose_info_file << std::endl;
        } else if (flag == "--dosage-beagle") {
            dose_beagle_flag = true;
            dose_mach_flag = false;
            dose_mach_gz_flag = false;
            dose_file = argv[++i];
            dose_info_file = argv[++i];
            LOGGER << "--dosage-beagle " << dose_file << " " << dose_info_file << std::endl;
        } else if (flag == "--imput-rsq") {
            dose_Rsq_cutoff = std::atof(argv[++i]);
            LOGGER << "--imput-rsq " << dose_Rsq_cutoff << std::endl;
            if (dose_Rsq_cutoff < 0.0 || dose_Rsq_cutoff > 1.0) LOGGER.e(0, "\n  --imput-rsq should be within the range from 0 to 1.\n");
        } else if (flag == "--update-imput-rsq") {
            update_impRsq_file = argv[++i];
            LOGGER << "--update-imput-rsq " << update_impRsq_file << std::endl;
            if (!std::filesystem::exists(update_impRsq_file)) LOGGER.e(0, "cannot open the file ["+update_impRsq_file+"] to read.");
        } else if (flag == "--update-freq") {
            update_freq_file = argv[++i];
            LOGGER << "--update-freq " << update_freq_file << std::endl;
            if (!std::filesystem::exists(update_freq_file)) LOGGER.e(0, "cannot open the file ["+update_freq_file+"] to read.");
        } else if (flag == "--update-ref-allele") {
            update_refA_file = argv[++i];
            LOGGER << "--update-ref-allele " << update_refA_file << std::endl;
            if (!std::filesystem::exists(update_refA_file)) LOGGER.e(0, "cannot open the file ["+update_refA_file+"] to read.");
        } else if (flag == "--keep") {
            kp_indi_file = argv[++i];
            LOGGER << "--keep " << kp_indi_file << std::endl;
            if (!std::filesystem::exists(kp_indi_file)) LOGGER.e(0, "cannot open the file ["+kp_indi_file+"] to read.");
        } else if (flag == "--remove") {
            rm_indi_file = argv[++i];
            LOGGER << "--remove " << rm_indi_file << std::endl;
            if (!std::filesystem::exists(rm_indi_file)) LOGGER.e(0, "cannot open the file ["+rm_indi_file+"] to read.");
        } else if (flag == "--update-sex") {
            update_sex_file = argv[++i];
            LOGGER << "--update-sex " << update_sex_file << std::endl;
            if (!std::filesystem::exists(update_sex_file)) LOGGER.e(0, "cannot open the file ["+update_sex_file+"] to read.");
        } else if (flag == "--chr") {
            extract_chr_start = extract_chr_end = std::atoi(argv[++i]);
            LOGGER << "--chr " << extract_chr_start << std::endl;
            if (extract_chr_start < 1) LOGGER.e(0, "\n --chr should be >= 1.\n");
        } else if (flag == "--autosome-num") {
            autosome_num = std::atoi(argv[++i]);
            autosome_num_explicit = true;
            LOGGER << "--autosome-num " << autosome_num << std::endl;
            if (autosome_num < 1) LOGGER.e(0, "\n  invalid number specified after the option --autosome-num (must be >= 1).\n");
        } else if (flag == "--autosome") {
            autosome_flag = true;
            LOGGER << "--autosome" << std::endl;
        } else if (flag == "--extract") {
            extract_snp_file = argv[++i];
            LOGGER << "--extract " << extract_snp_file << std::endl;
            if (!std::filesystem::exists(extract_snp_file)) LOGGER.e(0, "cannot open the file ["+extract_snp_file+"] to read.");
        } else if (flag == "--exclude") {
            exclude_snp_file = argv[++i];
            LOGGER << "--exclude " << exclude_snp_file << std::endl;
            if (!std::filesystem::exists(exclude_snp_file)) LOGGER.e(0, "cannot open the file ["+exclude_snp_file+"] to read.");
        } else if (flag == "--extract-snp") {
            extract_snp_name = argv[++i];
            LOGGER << "--extract-snp " << extract_snp_name << std::endl;
        } else if (flag == "--extract-region-snp") {
            extract_snp_name = argv[++i];
            extract_region_wind = std::atoi(argv[++i]);
            LOGGER << "--extract-region-snp " << extract_snp_name << " " << extract_region_wind << "Kb" << std::endl;
            extract_region_wind *= 1000;
            if(extract_region_wind < 1000 || extract_region_wind > 1e8) LOGGER.e(0, "\n the second parameter of --extract-region is distance in Kb unit. It should take value between 1 and 1e5.");
        } else if (flag == "--extract-region-bp") {
            extract_region_chr = std::atoi(argv[++i]);
            extract_region_bp = std::atoi(argv[++i]);
            extract_region_wind = std::atoi(argv[++i]);
            LOGGER << "--extract-region-bp " << extract_region_chr << " " << extract_region_bp << " " << extract_region_wind << "Kb" << std::endl;
            extract_region_wind *= 1000;
            if(extract_region_wind < 1000 || extract_region_wind > 1e8) LOGGER.e(0, "\n the second parameter of --extract-region is distance in Kb unit. It should take value between 1 and 1e5.");
        } else if (flag == "--exclude-snp") {
            exclude_snp_name = argv[++i];
            LOGGER << "--exclude-snp " << exclude_snp_name << std::endl;
        } else if (flag == "--exclude-region-snp") {
            exclude_snp_name = argv[++i];
            exclude_region_wind = std::atoi(argv[++i]);
            LOGGER << "--exclude-region-snp " << exclude_snp_name << exclude_region_wind << "Kb" << std::endl;
            exclude_region_wind *= 1000;
            if(exclude_region_wind < 1000 || exclude_region_wind > 1e8) LOGGER.e(0, "\n the second parameter of --exclude-region is distance in Kb unit. It should take value between 1 and 1e5.");
        } else if (flag == "--exclude-region-bp") {
            exclude_region_chr = std::atoi(argv[++i]);
            exclude_region_bp = std::atoi(argv[++i]);
            exclude_region_wind = std::atoi(argv[++i]);
            LOGGER << "--exclude-region-bp " << exclude_region_chr << " " << exclude_region_bp << " " << exclude_region_wind << "Kb" << std::endl;
            exclude_region_wind *= 1000;
            if(exclude_region_wind < 1000 || exclude_region_wind > 1e8) LOGGER.e(0, "\n the second parameter of --exclude-region is distance in Kb unit. It should take value between 1 and 1e5.");
        } else if (flag == "--maf") {
            maf = std::atof(argv[++i]);
            LOGGER << "--maf " << maf << std::endl;
            if (maf < 0 || maf > 0.5) LOGGER.e(0, "\n  --maf should be within the range from 0 to 0.5.\n");
        } else if (flag == "--max-maf") {
            max_maf = std::atof(argv[++i]);
            LOGGER << "--max-maf " << max_maf << std::endl;
            if (max_maf <= 0) LOGGER.e(0, "\n  --max-maf should be > 0.\n");
        } else if (flag == "--out") {
            out = argv[++i];
            LOGGER << "--out " << out << std::endl;
        } else if (flag == "--freq-v1") {
            out_freq_flag = true;
            thread_flag = true;
            LOGGER << "--freq-v1" << std::endl;
        } else if (flag == "--freq") {
            out_freq_flag = true;
            thread_flag = true;
            LOGGER << "--freq" << std::endl;
        } else if (flag == "--ssq") {
            out_ssq_flag = true;
            LOGGER << "--ssq" << std::endl;
        } else if (flag == "--recode") {
            recode = true;
            thread_flag = true;
            LOGGER << "--recode" << std::endl;
        } else if (flag == "--recode-nomiss") {
            recode_nomiss = true;
            thread_flag = true;
            LOGGER << "--recode-nomiss" << std::endl;
        } else if (flag == "--recode-std") {
            recode_std = true;
            thread_flag = true;
            LOGGER << "--recode-std" << std::endl;
        } else if (flag == "--save-ram") {
            save_ram = true;
            LOGGER << "--save-ram" << std::endl;
        }// GRM
        else if (flag == "--paa") {
            paa_file = argv[++i];
            LOGGER << "--paa " << paa_file << std::endl;
            if (!std::filesystem::exists(paa_file)) LOGGER.e(0, "cannot open the file ["+paa_file+"] to read.");
        } else if (flag == "--ibc") {
            ibc = true;
            LOGGER << "--ibc" << std::endl;
        } else if (flag == "--ibc-all") {
            ibc = ibc_all = true;
            LOGGER << "--ibc-all" << std::endl;
        } else if (flag == "--mgrm" || flag == "--mgrm-bin") {
            m_grm_flag = true;
            grm_file = argv[++i];
            LOGGER << argv[i - 1] << " " << grm_file << std::endl;
        } else if (flag == "--mgrm-gz") {
            m_grm_flag = true;
            m_grm_bin_flag = false;
            grm_bin_flag = false;
            grm_file = argv[++i];
            LOGGER << "--mgrm-gz " << grm_file << std::endl;
        } else if (flag == "--grm-chr") {
            grm_chr_prefix = argv[++i];
            LOGGER << "--grm-chr " << grm_chr_prefix << std::endl;
        } else if (flag == "--grm" || flag == "--grm-bin") {
            grm_flag = true;
            grm_file = argv[++i];
            LOGGER << argv[i - 1] << " " << grm_file << std::endl;
        } else if (flag == "--grm-gz") {
            grm_flag = true;
            m_grm_bin_flag = false;
            grm_bin_flag = false;
            grm_file = argv[++i];
            LOGGER << "--grm-gz " << grm_file << std::endl;
        } else if (flag == "--rm-high-ld") {
            rm_high_ld_cutoff = std::atof(argv[++i]);
            LOGGER << "--rm-high-ld " << rm_high_ld_cutoff << std::endl;
            if (rm_high_ld_cutoff <= 0 || rm_high_ld_cutoff >= 1) LOGGER.e(0, "\n the value to be specified after --rm-high-ld should be within the range from 0 to 1.\n");
        } else if (flag == "--make-grm" || flag == "--make-grm-v1" || flag == "--make-grm-bin") {
            make_grm_flag = true;
            thread_flag = true;
            LOGGER << argv[i] << std::endl;
        } else if (flag == "--make-grm-gz") {
            make_grm_flag = true;
            grm_out_bin_flag = false;
            thread_flag = true;
            LOGGER << "--make-grm-gz" << std::endl;
        } else if (strcmp(argv[i], "--make-grm-alg") == 0) {
            LOGGER << "--make-grm-gz" << std::endl;
        } else if (flag == "--make-grm-alg") {
            grm_out_bin_flag = false;
            thread_flag = true;
            LOGGER << "--make-grm-gz" << std::endl;
        } else if (flag == "--make-grm-f3") {
            make_grm_flag = true;
            make_grm_f3_flag = true;
            grm_out_bin_flag = true;
            thread_flag = true;
            LOGGER << "--make-grm-f3" << std::endl;
        } else if (flag == "--make-grm-d-v1" || flag == "--make-grm-d-bin") {
            make_grm_flag = true;
            dominance_flag = true;
            thread_flag = true;
            LOGGER << argv[i] << std::endl;
        } else if (flag == "--make-grm-d-gz") {
            make_grm_flag = true;
            dominance_flag = true;
            grm_out_bin_flag = false;
            thread_flag = true;
            LOGGER << "--make-grm-d-gz" << std::endl;
        } else if (flag == "--dominance") {
            dominance_flag = true;
            thread_flag = true;
            LOGGER <<"--dominance"<< std::endl;

        } else if (flag == "--model") {
            genetic_model = argv[++i];
            GeneticModel temp_model;
            if (!stringToGeneticModel(genetic_model, temp_model)) {
                LOGGER.e(0, "\n  --model should be either 'additive' or 'nonadditive'.\n");
            }
            LOGGER << "--model " << genetic_model << std::endl;
        } else if (flag == "--make-grm-homogametic" || flag == "--make-grm-homogametic-bin") {

            make_grm_flag = true;
            make_grm_homogametic_flag = true;
            thread_flag = true;
            LOGGER << argv[i] << std::endl;
        } else if (flag == "--make-grm-homogametic-gz") {
            make_grm_flag = true;
            make_grm_homogametic_flag = true;
            grm_out_bin_flag = false;
            thread_flag = true;
            LOGGER << "--make-grm-homogametic-gz" << std::endl;
        } else if (flag == "--make-grm-inbred" || flag == "--make-grm-inbred-bin") {
            make_grm_flag = true;
            make_grm_inbred_flag = true;
            thread_flag = true;
            LOGGER << argv[i] << std::endl;
            LOGGER << "--make-grm-inbred-gz" << std::endl;
        } else if (flag == "--grm-adj") {
            grm_adj_fac = std::atof(argv[++i]);
            LOGGER << "--grm-adj " << grm_adj_fac << std::endl;
            if (grm_adj_fac < 0 || grm_adj_fac > 1) LOGGER.e(0, "\n the value to be specified after --grm-adj should be within the range from 0 to 1.\n");
        } else if (flag == "--dc") {
            dosage_compen = std::atoi(argv[++i]);
            LOGGER << "--dc " << dosage_compen << std::endl;
            if (dosage_compen != 0 && dosage_compen != 1) LOGGER.e(0, "\n the value to be specified after --dc should be 0 or 1.\n");
        } else if (flag == "--grm-cutoff" || flag == "--grm-cutoff-v1") {
            grm_cutoff = std::atof(argv[++i]);
            if (grm_cutoff >= -1 && grm_cutoff <= 2) LOGGER << "--grm-cutoff" << grm_cutoff << std::endl;
            else grm_cutoff = -2;
        } else if (flag == "--grm-align") {
            align_grm_flag = true;
            thread_flag = true;
        } else if (flag == "--make-bK") {
            bK_threshold = std::atof(argv[++i]);
            if (bK_threshold < 0 || bK_threshold > 1) LOGGER.e(0, "\n --make-bK threshold should be range from 0 to 1.\n");
            else LOGGER << "--make-bK " << bK_threshold << std::endl;
        } else if (flag == "--denseness") {
            denseness_metric = argv[++i];
            LOGGER << "--denseness " << denseness_metric << std::endl;
        } else if (flag == "--pca-v1") {
            pca_flag = true;
            thread_flag = true;
            if (i + 1 < argc && std::string_view{argv[i+1]}.substr(0, 2) != "--") {
                out_pc_num = std::atoi(argv[++i]);
                if (out_pc_num < 0) LOGGER.e(0, "\n the value to be specified after --pca-v1 should be positive.\n");
                LOGGER << "--pca-v1 " << out_pc_num << std::endl;
            }
            else {
                LOGGER << "--pca-v1 (\"all by default\")" << std::endl;
            }
        } else if (flag == "--pca-approx") {
            i++;
            if (i >= argc || std::string_view{argv[i]}.substr(0, 2) == "--") {
                pca_approx_flag = "Lanczos";
                i--;
            } else {
                pca_approx_flag = argv[i];
            }
            LOGGER << "--pca-approx " << pca_approx_flag << std::endl;
        } else if (flag == "--pc-loading") {
            pcl_flag = true;
            thread_flag = true;
            pc_file = argv[++i];
            //pcl_grm_N = std::atoi(argv[++i]);
            LOGGER << "--pc-loading " << pc_file << std::endl;
            //if(pcl_grm_N < 1 || pcl_grm_N > 1e20) LOGGER.e(0, "\n invalid number of SNPs used to calculate PCs.");
        }else if (flag == "--project-loading" ){
            project_flag = true;
            thread_flag = true;
            project_file = argv[++i];
            project_N = std::atoi(argv[++i]);
            LOGGER << "--project-loading " << project_file << " " << project_N << std::endl;
            if(project_N < 1) LOGGER.e(0, "\n invalid number of PCs to output (must be >= 1)");
        }
        // estimation of LD structure
        else if (flag == "--ld") {
            LD = true;
            LD_file = argv[++i];
            LOGGER << "--ld " << LD_file << std::endl;
            if (!std::filesystem::exists(LD_file)) LOGGER.e(0, "cannot open the file ["+LD_file+"] to read.");
        } else if (flag == "--ld-step") {
            LD_search = true;
            LD_step = std::atoi(argv[++i]);
            LOGGER << "--ld-step " << LD_step << std::endl;
            if (LD_step < 1) LOGGER.e(0, "\n --ld-step should be >= 1.\n");
        } else if (flag == "--ld-wind" || flag == "--ld-pruning-wind" || flag == "--make-grm-wt-wind") {
            LD_wind = std::atof(argv[++i]);
            LOGGER << argv[i - 1] << " " << LD_wind << std::endl;
            LD_wind *= 1000;
            if (LD_wind < 1e3 || LD_wind > 2e7) {
                std::stringstream err_msg;
                err_msg << "\n  " << argv[i - 1] << " should be 1Kb or 20Mb.\n";
                LOGGER.e(0, err_msg.str());
            }
        } else if (flag == "--ld-sig") {
            LD_sig = std::atof(argv[++i]);
            LOGGER << "--ld-sig " << LD_sig << std::endl;
            if (LD_sig <= 0) LOGGER.e(0, "\n --ld-sig should be > 0.\n");
        } else if (flag == "--ld-i") {
            LD_i = true;
            LOGGER << "--ld-i" << std::endl;
        } else if (flag == "--ld-pruning") {
            thread_flag = true;
            LD_prune_rsq = std::atof(argv[++i]);
            LOGGER << "--ld-pruning " << LD_prune_rsq << std::endl;
            if (LD_prune_rsq < 0.0001 || LD_prune_rsq > 0.9999) LOGGER.e(0, "\n --ld-pruning should be within the range from 0.0001 to 0.9999.\n");
        } else if (flag == "--ld-score") {
            ld_score_flag = true;
            thread_flag = true;
            LOGGER << "--ld-score" << std::endl;
        } else if (flag == "--ld-score-adj") {
            ldscore_adj_flag = true;
            LOGGER << "--ld-score-adj" << std::endl;
        } else if (flag == "--ld-score-multi") {
            ld_score_flag = true;
            thread_flag = true;
            ld_score_multi_file = argv[++i];
            LOGGER << "--ld-score-multi " << ld_score_multi_file << std::endl;
            if (!std::filesystem::exists(ld_score_multi_file)) LOGGER.e(0, "cannot open the file ["+ld_score_multi_file+"] to read.");
        } else if (flag == "--ld-rsq-cutoff") {
            LD_rsq_cutoff = std::atof(argv[++i]);
            LOGGER << "--ld-rsq-cutoff " << LD_rsq_cutoff << std::endl;
            if (LD_rsq_cutoff < 0.0 || LD_rsq_cutoff > 1.0) {
                std::stringstream err_msg;
                err_msg << "\n  " << argv[i - 1] << " should be within the range from 0 to 1.\n";
                LOGGER.e(0, err_msg.str());
            }
        } else if (flag == "--ld-max-rsq") {
            ld_max_rsq_flag = true;
            thread_flag = true;
            LOGGER << "--ld-max-rsq" << std::endl;
        } else if (flag == "--ld-score-region") {
            ld_mean_rsq_seg_flag = true;
            thread_flag = true;
            i++;
            if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") {
                LD_seg = 200;
                i--;
            } else LD_seg = std::atoi(argv[i]);
            LOGGER << "--ld-score-region" << std::endl;
            if (LD_seg < 10) LOGGER.e(0, "\n the input value for --ld-score-region needs to be > 10.\n");
            LD_seg *= 1000;
        } else if (flag == "--ld-file") {
            LD_file = argv[++i];
            LOGGER << "--ld-file " << LD_file << std::endl;
            if (!std::filesystem::exists(LD_file)) LOGGER.e(0, "cannot open the file ["+LD_file+"] to read.");
        }
        // simulation based on real genotype data
        else if (flag == "--simu-qt") {
            simu_qt_flag = true;
            LOGGER << "--simu-qt" << std::endl;
        } else if (flag == "--simu-cc") {
            simu_cc = true;
            simu_case_num = std::atoi(argv[++i]);
            simu_control_num = std::atoi(argv[++i]);
            LOGGER << "--simu-cc " << simu_case_num << " " << simu_control_num << std::endl;
            if (simu_case_num < 10) LOGGER.e(0, "--simu-cc, Invalid number of cases. Minimum number 10.");
            if (simu_control_num < 10) LOGGER.e(0, "--simu-cc, Invalid number of controls. Minimum number 10.");
        } else if (flag == "--simu-rep") {
            simu_rep = std::atoi(argv[++i]);
            LOGGER << "--simu-rep " << simu_rep << std::endl;
            if (simu_rep < 1) LOGGER.e(0, "--simu-rep should be >= 1.");
        } else if (flag == "--simu-hsq") {
            simu_h2 = std::atof(argv[++i]);
            LOGGER << "--simu-hsq " << simu_h2 << std::endl;
            if (simu_h2 > 1.0 || simu_h2 < 0.0) LOGGER.e(0, "--simu-h2 should be within the range from 0 to 1.");
        } else if (flag == "--simu-k") {
            simu_K = std::atof(argv[++i]);
            LOGGER << "--simu-k " << simu_K << std::endl;
            if (simu_K > 0.5 || simu_K < 0.0001) LOGGER.e(0, "--simu-K should be within the range from 0.0001 to 0.5.");
        } else if (flag == "--simu-causal-loci") {
            simu_causal = argv[++i];
            LOGGER << "--simu-causal-loci " << simu_causal << std::endl;
            if (!std::filesystem::exists(simu_causal)) LOGGER.e(0, "cannot open the file ["+simu_causal+"] to read.");
        } else if (flag == "--simu-embayesb") { // internal
            simu_emb_flag = true;
            LOGGER << "--simu-embayesb" << std::endl;
        } else if (flag == "--simu-ouput-causal") { // internal
            simu_output_causal = true;
            LOGGER << "--simu-output-causal" << std::endl;
        } else if (flag == "--simu-seed") {
            simu_seed = std::atof(argv[++i]);
            LOGGER << "--simu-seed " << simu_seed << std::endl;
            if (simu_seed <= 100) LOGGER.e(0, "--simu-seed should be >100.");
        } else if (flag == "--simu-eff-mod") {
            simu_eff_mod = std::atoi(argv[++i]);
            LOGGER << "--simu-eff-mod " << simu_eff_mod << std::endl;
            if (simu_eff_mod != 0 && simu_eff_mod !=1) LOGGER.e(0, "--simu-eff-mod should be 0 or 1.");
        }
        else if (flag == "--hapmap-genet-dst") { // calculate genetic dst based on HapMap data
            hapmap_genet_dst = true;
            hapmap_genet_dst_file = argv[++i];
            LOGGER << "--hapmap-genet-dst " << hapmap_genet_dst_file << std::endl;
        }// estimate variance explained by all SNPs
        else if (flag == "--HEreg") {
            HE_reg_flag = true;
            thread_flag = true;
            LOGGER << "--HEreg" <<  std::endl;
        } else if (flag == "--HEreg-bivar") {
            HE_reg_bivar_flag = true;
            thread_flag = true;
            std::vector<int> mphen_buf;
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                mphen_buf.push_back(atoi(argv[i]));
            }
            i--;
            if (mphen_buf.size() < 2 && mphen_buf.size() > 0) LOGGER.e(0, "\n --HEreg-bivar. Please specify two traits for the bivariate HE regression analysis.");
            if (mphen_buf.size() == 0) {
                mphen = 1;
                mphen2 = 2;
            } else {
                mphen = mphen_buf[0];
                mphen2 = mphen_buf[1];
            }
            if (mphen < 1 || mphen2 < 1 || mphen == mphen2) LOGGER.e(0, "\n --HEreg-bivar. Invalid input parameters.");
            LOGGER << "--HEreg-bivar " << mphen << " " << mphen2 <<  std::endl;
        } else if (flag == "--reml") {
            reml_flag = true;
            thread_flag = true;
            LOGGER << "--reml" << std::endl;
            if (m_grm_flag) no_lrt = true;
        } else if (flag == "--prevalence") {
            prevalence_flag = true;
            prevalence = std::atof(argv[++i]);
            LOGGER << "--prevalence " << prevalence << std::endl;
            if (prevalence <= 0 || prevalence >= 1) LOGGER.e(0, "\n --prevalence should be between 0 to 1.\n");
        } else if (flag == "--reml-pred-rand") {
            pred_rand_eff = true;
            LOGGER << "--reml-pred-rand" <<  std::endl;
        } else if(flag == "--cvblup"){
            cv_blup = true;
            LOGGER << "--cvblup" <<  std::endl;
        } else if (flag == "--reml-est-fix") {
            est_fix_eff = true;
            LOGGER << "--reml-est-fix" << std::endl;
        } else if (flag == "--reml-est-fix-varcov") {
            est_fix_eff = true;
            est_fix_eff_var = true;
            LOGGER << "--reml-est-fix-varcov" << std::endl;
        } else if (flag == "--reml-alg") {
            reml_mtd = std::atoi(argv[++i]);
            LOGGER << "--reml-alg " << reml_mtd << std::endl;
            if (reml_mtd < 0 || reml_mtd > 2) LOGGER.e(0, "\n  --reml-alg should be 0, 1 or 2.\n");
        } else if (flag == "--reml-no-constrain") {
            reml_flag = true;
            no_constrain = true;
            LOGGER << "--reml-no-constrain" <<  std::endl;
        } else if (flag == "--reml-priors") {
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                reml_priors.push_back(atof(argv[i]));
            }
            i--;
            LOGGER << "--reml-priors ";
            bool err_flag = false;
            for (j = 0; j < reml_priors.size(); j++) {
                LOGGER << reml_priors[j] << " ";
                if (reml_priors[j] > 1.0 || reml_priors[j] < -10.0) err_flag = true;
            }
            LOGGER << std::endl;
            if (err_flag || reml_priors.empty()) LOGGER.e(0, "\n  --reml-priors. Prior values of variance explained should be between 0 and 1.\n");
        } else if (flag == "--reml-priors-var"  || flag == "--reml-fixed-var") {
             std::string s_buf = argv[i];
            if(s_buf == "--reml-fixed-var") reml_fixed_var_flag = true;
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                reml_priors_var.push_back(atof(argv[i]));
            }
            i--;
            LOGGER << s_buf << " ";
            bool err_flag = false;
            for (j = 0; j < reml_priors_var.size(); j++) {
                LOGGER << reml_priors_var[j] << " ";
                if (reml_priors_var[j] < 0.0) err_flag = true;
            }
            LOGGER << std::endl;
            if (reml_priors_var.empty()) LOGGER.e(0, "\n  " + s_buf + ". Prior values of variance components are required.\n");
        } else if (flag == "--reml-no-lrt") {
            no_lrt = true;
            LOGGER << "--reml-no-lrt" <<  std::endl;
        } else if (flag == "--reml-lrt") {
            no_lrt = false;
            reml_lrt_flag = true;
            reml_drop.clear();
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                reml_drop.push_back(atoi(argv[i]));
            }
            i--;
            LOGGER << "--reml-lrt ";
            bool err_flag = false;
            for (j = 0; j < reml_drop.size(); j++) {
                LOGGER << reml_drop[j] << " ";
                if (reml_drop[j] < 1) err_flag = true;
            }
            LOGGER << std::endl;
            if (err_flag || reml_drop.empty()) LOGGER.e(0, "\n invalid values specified after --reml-lrt.\n");
        } else if (flag == "--reml-maxit") {
            MaxIter = std::atoi(argv[++i]);
            LOGGER << "--reml-maxit " << MaxIter << std::endl;
            if (MaxIter < 1) LOGGER.e(0, "\n --reml-maxit should be >= 1.\n");
        } else if (flag == "--reml-bendV") {
            reml_force_inv_fac_flag = true;
            LOGGER << "--reml-bendV " <<  std::endl;
        } else if (flag == "--reml-force-converge") {
            reml_force_converge_flag = true;
            LOGGER << "--reml-force-converge " <<  std::endl;
        } else if (flag == "--reml-allow-no-converge") {
            reml_no_converge_flag = true;
            LOGGER << "--reml-allow-no-converge " <<  std::endl;
        } else if (flag == "--reml-bending") {
            reml_bending = true;
            LOGGER << "--reml-bending " <<  std::endl;
        }else if (flag == "--reml-amzvc"){
            reml_allow_constrain_run = true;
            LOGGER << "--reml-amzvc" <<  std::endl;
        } else if (flag == "--reml-trace-approx") {
            reml_trace_approx = true;
            LOGGER << "--reml-trace-approx" << std::endl;
        } else if (flag == "--reml-trace-nprobes") {
            reml_trace_nprobes = std::atoi(argv[++i]);
            reml_trace_approx = true;
            LOGGER << "--reml-trace-nprobes " << reml_trace_nprobes << std::endl;
            if (reml_trace_nprobes < 9) LOGGER.e(0, "\n  --reml-trace-nprobes should be >= 9.\n");
        } else if (flag == "--reml-trace-power-iter") {
            reml_trace_power_iter = std::atoi(argv[++i]);
            LOGGER << "--reml-trace-power-iter " << reml_trace_power_iter << std::endl;
            if (reml_trace_power_iter < 0) LOGGER.e(0, "\n  --reml-trace-power-iter must be >= 0.\n");
        } else if (flag == "--reml-woodbury") {
            if (i + 1 >= argc) LOGGER.e(0, "\n  --reml-woodbury requires an argument (e.g. --reml-woodbury 100 or --reml-woodbury auto).\n");
            std::string wb_arg = argv[++i];
            if (wb_arg.substr(0, 4) == "auto") {
                reml_woodbury_rank   = -1;   // auto-k mode sentinel
                reml_woodbury_buffer = 2.5;  // default MP buffer (see rationale above)
                reml_woodbury_k_max  = 0;    // 0 → min(n-1, 2000)
                // Optional suffixes: "auto:<k_max>" or "auto:<k_max>:<buffer>"
                if (wb_arg.size() > 4 && wb_arg[4] == ':') {
                    std::string rest = wb_arg.substr(5);
                    auto colon = rest.find(':');
                    if (colon == std::string::npos) {
                        reml_woodbury_k_max = std::atoi(rest.c_str());
                    } else {
                        reml_woodbury_k_max  = std::atoi(rest.substr(0, colon).c_str());
                        reml_woodbury_buffer = std::atof(rest.substr(colon + 1).c_str());
                    }
                }
                LOGGER << "--reml-woodbury auto (k_max=" << reml_woodbury_k_max
                       << ", buffer=" << reml_woodbury_buffer << ")" << std::endl;
                if (reml_woodbury_buffer < 1.0)
                    LOGGER.e(0, "\n  --reml-woodbury auto: buffer factor must be >= 1.0.\n");
            } else {
                reml_woodbury_rank = std::atoi(wb_arg.c_str());
                LOGGER << "--reml-woodbury " << reml_woodbury_rank << std::endl;
                if (reml_woodbury_rank < 1) LOGGER.e(0, "\n  --reml-woodbury rank must be >= 1.\n");
            }
        } else if (flag == "--reml-woodbury-nystrom") {
            reml_woodbury_nystrom = true;
            LOGGER << "--reml-woodbury-nystrom (single-pass Nyström sketch)" << std::endl;
        } else if (flag == "--reml-diag-one") {
            reml_diag_one = true;
            LOGGER << "--reml-diag-one " <<  std::endl;
        } else if (flag == "--reml-diagV-adj") {
            reml_diagV_adj = std::atoi(argv[++i]);
            LOGGER << "--reml-diagV-adj " << reml_diagV_adj <<  std::endl;
        } else if(flag == "--reml-diag-mul") {
            reml_diag_mul = std::stod(argv[++i]);
            LOGGER << "--reml-diag-mul " << reml_diag_mul <<  std::endl;
        } else if(flag == "--reml-inv-mtd"){
            reml_inv_method = std::stoi(argv[++i]);
            LOGGER << "--reml-inv-mtd " << reml_inv_method <<  std::endl;
        } else if (flag == "--log-pval") {
            logp_flag = true;
            LOGGER << "--log-pval" << std::endl;
        } else if (flag == "--pheno") {
            phen_file = argv[++i];
            LOGGER << "--pheno " << phen_file <<  std::endl;
            if (!std::filesystem::exists(phen_file)) LOGGER.e(0, "cannot open the file ["+phen_file+"] to read.");
        } else if (flag == "--mpheno") {
            mphen = std::atoi(argv[++i]);
            LOGGER << "--mpheno " << mphen <<  std::endl;
            if (mphen < 1) LOGGER.e(0, "--mpheno should be > 0.");
        } else if (flag == "--qcovar") {
            qcovar_file = argv[++i];
            LOGGER << "--qcovar " << qcovar_file <<  std::endl;
            if (!std::filesystem::exists(qcovar_file)) LOGGER.e(0, "cannot open the file ["+qcovar_file+"] to read.");
        } else if (flag == "--covar") {
            covar_file = argv[++i];
            LOGGER << "--covar " << covar_file <<  std::endl;
            if (!std::filesystem::exists(covar_file)) LOGGER.e(0, "cannot open the file ["+covar_file+"] to read.");
        } else if (flag == "--reml-res-diag"){
            weight_file = argv[++i];
            LOGGER << "--reml-res-diag " << weight_file <<  std::endl;
            if (!std::filesystem::exists(weight_file)) LOGGER.e(0, "cannot open the file ["+weight_file+"] to read.");
        } else if (flag == "--gxqe") {
            qgxe_file = argv[++i];
            LOGGER << "--gxqe " << qgxe_file <<  std::endl;
            if (!std::filesystem::exists(qgxe_file)) LOGGER.e(0, "cannot open the file ["+qgxe_file+"] to read.");
        } else if (flag == "--gxe") {
            gxe_file = argv[++i];
            LOGGER << "--gxe " << gxe_file <<  std::endl;
            if (!std::filesystem::exists(gxe_file)) LOGGER.e(0, "cannot open the file ["+gxe_file+"] to read.");
        } else if (flag == "--blup-snp") {
            blup_snp_flag = true;
            blup_indi_file = argv[++i];
            LOGGER << "--blup-snp " << blup_indi_file <<  std::endl;
            if (!std::filesystem::exists(blup_indi_file)) LOGGER.e(0, "cannot open the file ["+blup_indi_file+"] to read.");
        } else if (flag == "--reml-wfam") {
            reml_flag = true;
            within_family = true;
            LOGGER << "--reml-wfam " <<  std::endl;
        } else if (flag == "--reml-bivar") {
            bivar_reml_flag = true;
            thread_flag = true;
            std::vector<int> mphen_buf;
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                mphen_buf.push_back(atoi(argv[i]));
            }
            i--;
            if (mphen_buf.size() < 2 && mphen_buf.size() > 0) LOGGER.e(0, "\n --reml-bivar. Please specify two traits for the bivariate REML analysis.");
            if (mphen_buf.size() == 0) {
                mphen = 1;
                mphen2 = 2;
            } else {
                mphen = mphen_buf[0];
                mphen2 = mphen_buf[1];
            }
            if (mphen < 1 || mphen2 < 1 || mphen == mphen2) LOGGER.e(0, "\n --reml-bivar. Invalid input parameters.");
            LOGGER << "--reml-bivar " << mphen << " " << mphen2 <<  std::endl;
        } else if (flag == "--reml-bivar-prevalence") {
             std::vector<double> K_buf;
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                K_buf.push_back(atof(argv[i]));
            }
            i--;
            if (K_buf.size() < 1 || K_buf.size() > 2) LOGGER.e(0, "\n  --reml-bivar-prevalence. Please specify the prevalence of the two diseases.");
            if (K_buf.size() == 2) {
                if (K_buf[0] < 0.0 || K_buf[0] > 1.0 || K_buf[1] < 0.0 || K_buf[1] > 1.0) LOGGER.e(0, "\n  --reml-bivar-prevalence. Disease prevalence should be between 0 and 1.");
                LOGGER << "--reml-bivar-prevalence " << K_buf[0] << " " << K_buf[1] << std::endl;
                prevalence = K_buf[0];
                prevalence2 = K_buf[1];
            } else {
                if (K_buf[0] < 0.0 || K_buf[0] > 1.0) LOGGER.e(0, "\n --reml-bivar-prevalence. Disease prevalence should be between 0 and 1.");
                LOGGER << "--reml-bivar-prevalence " << K_buf[0] << std::endl;
                prevalence = prevalence2 = K_buf[0];
            }
        } else if (flag == "--reml-bivar-nocove") {
            ignore_Ce = true;
            LOGGER << "--reml-bivar-nocove" << std::endl;
        } else if (flag == "--reml-bivar-lrt-rg") {
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                fixed_rg_val.push_back(atof(argv[i]));
            }
            i--;
            LOGGER << "--reml-bivar-lrt-rg ";
            bool err_flag = false;
            for (j = 0; j < fixed_rg_val.size(); j++) {
                LOGGER << fixed_rg_val[j] << " ";
                if (fixed_rg_val[j] > 1.0 || fixed_rg_val[j]<-1.0) err_flag = true;
            }
            LOGGER << std::endl;
            if (err_flag || fixed_rg_val.empty()) LOGGER.e(0, "\n --reml-bivar-lrt-rg. Any input parameter should be within the range from -1 to 1.\n");
            bool haveZero = false;
            if (CommFunc::FloatEqual(fixed_rg_val[0], 0.0)) haveZero = true;
            for (j = 1; j < fixed_rg_val.size(); j++) {
                if ((CommFunc::FloatNotEqual(fixed_rg_val[0], 0.0) && haveZero) || (CommFunc::FloatEqual(fixed_rg_val[0], 0.0) && !haveZero)) LOGGER.e(0, "\n --reml-bivar-lrt-rg. Input parameters should be all zero or all non-zero values.\n");
            }
        } else if (flag == "--reml-bivar-no-constrain") {
            bivar_no_constrain = true;
            LOGGER << "--reml-bivar-no-constrain" << std::endl;
        } else if (flag == "--cojo-file") {
            massoc_file = argv[++i];
            LOGGER << "--cojo-file " << massoc_file << std::endl;
            if (!std::filesystem::exists(massoc_file)) LOGGER.e(0, "cannot open the file ["+massoc_file+"] to read.");
        } else if (flag == "--cojo-slct") {
            massoc_slct_flag = true;
            massoc_mld_slct_alg = 0;
            LOGGER << "--cojo-slct" <<  std::endl;
        } else if (flag == "--cojo-stepwise") {
            massoc_slct_flag = true;
            massoc_mld_slct_alg = 0;
            LOGGER << "--cojo-stepwise" << std::endl;
        } else if (flag == "--cojo-forward") {
            massoc_slct_flag = true;
            massoc_mld_slct_alg = 1;
            LOGGER << "--cojo-forward" << std::endl;
        } else if (flag == "--cojo-backward") {
            massoc_slct_flag = true;
            massoc_mld_slct_alg = 2;
            LOGGER << "--cojo-backward" << std::endl;
        } else if (flag == "--cojo-top-SNPs") {
            massoc_slct_flag = true;
            massoc_top_SNPs = std::atoi(argv[++i]);
            LOGGER << "--cojo-top-SNPs " << massoc_top_SNPs << std::endl;
            if (massoc_top_SNPs < 1) LOGGER.e(0, "\n --cojo-top-SNPs should be >= 1.\n");
        } else if (flag == "--cojo-actual-geno") {
            massoc_actual_geno_flag = false;
            LOGGER << "--cojo-actual-geno is deprecated currently." << std::endl;
        } else if (flag == "--cojo-p") {
            massoc_p = std::atof(argv[++i]);
            LOGGER << "--cojo-p " << massoc_p << std::endl;
            if (massoc_p > 0.05 || massoc_p <= 0) LOGGER.e(0, "\n --cojo-p should be within the range from 0 to 0.05.\n");
        } else if (flag == "--restrict-output-pC") {
            massoc_out_pC_thresh = strtod(argv[++i], NULL);
        } else if (flag == "--cojo-collinear") {
            massoc_collinear = std::atof(argv[++i]);
            LOGGER << "--cojo-collinear " << massoc_collinear << std::endl;
            if (massoc_collinear > 0.99 || massoc_collinear < 0.01) LOGGER.e(0, "\n --cojo-collinear should be within the ragne from 0.01 to 0.99.\n");
        } else if (flag == "--cojo-wind") {
            massoc_wind = std::atoi(argv[++i]);
            LOGGER << "--cojo-wind " << massoc_wind << std::endl;
            if (massoc_wind <= 0) LOGGER.e(0, "\n invalid value for --cojo-wind. It should be > 0.\n");
            massoc_wind *= 1000;
        } else if (flag == "--cojo-joint") {
            massoc_joint_flag = true;
            LOGGER << "--cojo-joint" << std::endl;
        } else if (flag == "--cojo-cond") {
            massoc_cond_snplist = argv[++i];
            LOGGER << "--cojo-cond " << massoc_cond_snplist << std::endl;
        } else if (flag == "--cojo-gc") {
            massoc_gc_flag = true;
            i++;
            if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") {
                massoc_gc_val = -1;
                i--;
            } else {
                massoc_gc_val = std::atof(argv[i]);
                if (massoc_gc_val < 1) LOGGER.e(0, "\n invalid value specified after --cojo-gc (must be >= 1).\n");
            }
            LOGGER << "--cojo-gc " << ((massoc_gc_val < 0) ? "" : argv[i]) << std::endl;
        } else if (flag == "--cojo-sblup") {
            massoc_sblup_flag = true;
            massoc_sblup_fac = std::atof(argv[++i]);
            LOGGER << "--cojo-sblup " << massoc_sblup_fac << std::endl;
            if (massoc_sblup_fac < 0) LOGGER.e(0, "\n invalid value for --cojo-sblup.\n");
        } else if (flag == "--mlma") {
            reml_flag = false;
            mlma_flag = true;
            thread_flag = true;
            LOGGER << "--mlma " << std::endl;
        } else if (flag == "--mlma-subtract-grm") {
            subtract_grm_file = argv[++i];
            LOGGER << "--mlma-subtract-grm " << subtract_grm_file << std::endl;
        } else if (flag == "--mlma-loco") {
            reml_flag = false;
            mlma_loco_flag = true;
            thread_flag = true;
            LOGGER << "--mlma-loco " << std::endl;
        } else if (flag == "--mlma-no-adj-covar") {
            mlma_no_adj_covar = true;
            LOGGER << "--mlma-no-adj-covar (use --mlma-no-preadj-covar instead)" << std::endl;
        } else if (flag == "--mlma-no-preadj-covar") {
            mlma_no_adj_covar = true;
            LOGGER << "--mlma-no-preadj-covar" << std::endl;
        } else if (flag == "--save-reml") {
            save_reml_flag = true;
            LOGGER << "--save-reml" << std::endl;
        } else if (flag == "--load-reml") {
            load_reml_file = argv[++i];
            LOGGER << "--load-reml " << load_reml_file << std::endl;
            if (!std::filesystem::exists(load_reml_file)) LOGGER.e(0, "cannot open the file ["+load_reml_file+"] to read.");
        } else if (flag == "--fst") {
            fst_flag = true;
            LOGGER << "--fst " << std::endl;
        } else if (flag == "--sub-popu") {
            subpopu_file = argv[++i];
            LOGGER << "--sub-popu " << subpopu_file << std::endl;
            if (!std::filesystem::exists(subpopu_file)) LOGGER.e(0, "cannot open the file ["+subpopu_file+"] to read.");
        }
        else if (flag == "--fastBAT-ld-cutoff") {
            sbat_ld_cutoff = sqrt(atof(argv[++i]));
            LOGGER << "--fastBAT-ld-cutoff " << sbat_ld_cutoff * sbat_ld_cutoff << std::endl;
            if (sbat_ld_cutoff <= 0.1) LOGGER.e(0, "\n --fastBAT_ld_cutoff should be > 0.1\n");
        } else if (flag == "--fastBAT-write-snpset") {
            sbat_write_snpset = true;
            LOGGER << "--fastBAT-write-snpset" << std::endl;
        } else if (flag == "--fastBAT") {
            sbat_sAssoc_file = argv[++i];
            LOGGER << "--fastBAT " << sbat_sAssoc_file << std::endl;
            if (!std::filesystem::exists(sbat_sAssoc_file)) LOGGER.e(0, "cannot open the file ["+sbat_sAssoc_file+"] to read.");
        } else if (flag == "--fastBAT-gene-list") {
            sbat_gAnno_file = argv[++i];
            LOGGER << "--fastBAT-gene-list " << sbat_gAnno_file << std::endl;
            if (!std::filesystem::exists(sbat_gAnno_file)) LOGGER.e(0, "cannot open the file ["+sbat_gAnno_file+"] to read.");
        } else if (flag == "--fastBAT-set-list") {
            sbat_snpset_file = argv[++i];
            LOGGER << "--fastBAT-set-list " << sbat_snpset_file << std::endl;
            if (!std::filesystem::exists(sbat_snpset_file)) LOGGER.e(0, "cannot open the file ["+sbat_snpset_file+"] to read.");
        } else if (flag == "--fastBAT-wind") {
            sbat_wind = std::atoi(argv[++i]);
            LOGGER << "--fastBAT-wind " << sbat_wind << std::endl;
            if (sbat_wind < 0) LOGGER.e(0, "\n invalid value for --fastBAT-wind. Valid range: >= 0\n");
            sbat_wind *= 1000;
        } else if (flag == "--fastBAT-seg") {
            sbat_seg_flag = true;
            thread_flag = true;
            i++;
            if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") {
                sbat_seg_size = 100;
                i--;
            } else sbat_seg_size = std::atoi(argv[i]);
            LOGGER << "--fastBAT-seg " << sbat_seg_size << std::endl;
            if (sbat_seg_size < 1) LOGGER.e(0, "\n invalid value for --fastBAT-seg. Valid range: >= 1\n");
            sbat_seg_size *= 1000;
        }
        else if (flag == "--mBAT-svd-gamma") {
            mbat_svd_gamma = std::atof(argv[++i]);
            LOGGER << "--mBAT-svd-gamma " << mbat_svd_gamma << std::endl;
            if (mbat_svd_gamma <= 0.8) LOGGER.e(0, "\n --mBAT-svd-gamma recommend to be 0.9\n");
        } else if (flag == "--mBAT-write-snpset") {
            mbat_write_snpset = true;
            LOGGER << "--mBAT-write-snpset" << std::endl;
        } else if (flag == "--mBAT-print-all-p") {
            mbat_print_all_p = true;
            LOGGER << "--mBAT-print-all-p" << std::endl;
        } else if (flag == "--mBAT-combo") {
            mbat_sAssoc_file = argv[++i];
            LOGGER << "--mBAT-combo " << mbat_sAssoc_file << std::endl;
            if (!std::filesystem::exists(mbat_sAssoc_file)) LOGGER.e(0, "cannot open the file ["+mbat_sAssoc_file+"] to read.");
        } else if (flag == "--mBAT-gene-list") {
            mbat_gAnno_file = argv[++i];
            LOGGER << "--mBAT-gene-list " << mbat_gAnno_file << std::endl;
            if (!std::filesystem::exists(mbat_gAnno_file)) LOGGER.e(0, "cannot open the file ["+mbat_gAnno_file+"] to read.");
        } else if (flag == "--mBAT-set-list") {
            mbat_snpset_file = argv[++i];
            LOGGER << "--mBAT-set-list " << mbat_snpset_file << std::endl;
            if (!std::filesystem::exists(mbat_snpset_file)) LOGGER.e(0, "cannot open the file ["+mbat_snpset_file+"] to read.");
        } else if (flag == "--mBAT-wind") {
            mbat_wind = std::atoi(argv[++i]);
            LOGGER << "--mBAT-wind " << mbat_wind << std::endl;
            if (mbat_wind < 0) LOGGER.e(0, "\n invalid value for --mBAT-wind. Valid range: >= 0\n");
            mbat_wind *= 1000;
        }
        else if (flag == "--efile") {
            efile = argv[++i];
            efile_flag = true;
            LOGGER << "--efile " << efile << std::endl;
            if (!std::filesystem::exists(efile)) LOGGER.e(0, "cannot open the file ["+efile+"] to read.");
        }
        else if (flag == "--e-cor") {
            eR_file = argv[++i];
            eR_file_flag = true;
            LOGGER << "--e-cor " << eR_file << std::endl;
            if (!std::filesystem::exists(eR_file)) LOGGER.e(0, "cannot open the file ["+eR_file+"] to read.");
        }
        else if (flag == "--ecojo") {
            ecojo_ma_file = argv[++i];
            LOGGER << "--ecojo " << ecojo_ma_file << std::endl;
            if (!std::filesystem::exists(ecojo_ma_file)) LOGGER.e(0, "cannot open the file ["+ecojo_ma_file+"] to read.");
        }
        else if (flag == "--ecojo-slct") {
            ecojo_slct_flag = true;
            LOGGER << "--ecojo-slct" << std::endl;
        }
        else if (flag == "--ecojo-p") {
            ecojo_p = std::atof(argv[++i]);
            LOGGER << "--ecojo-p " << ecojo_p << std::endl;
            if (ecojo_p > 0.05 || ecojo_p <= 0) LOGGER.e(0, "\n  --ecojo-p should be within the range from 0 to 0.05.\n");
        }
        else if (flag == "--ecojo-collinear") {
            ecojo_collinear = std::atof(argv[++i]);
            LOGGER << "--ecojo-collinear " << ecojo_collinear << std::endl;
            if (ecojo_collinear > 1 || ecojo_collinear < 0.01) LOGGER.e(0, "\n --ecojo-collinear should be within the range from 0.01 to 0.99.\n");
        }
        else if (flag == "--ecojo-blup") {
            ecojo_blup_flag = true;
            ecojo_lambda = std::atof(argv[++i]);
            LOGGER << "--ecojo-blup " << ecojo_lambda << std::endl;
            if (ecojo_lambda < 0.01 || ecojo_lambda > 0.99) LOGGER.e(0, "\n --ecojo-blup should be within the range from 0.01 to 0.99.\n");
        }
        else if (flag == "--make-erm") {
            make_erm_flag = true;
            thread_flag = true;
            LOGGER << argv[i] << std::endl;
        }
        else if (flag == "--make-erm-gz") {
            make_erm_flag = true;
            grm_out_bin_flag = false;
            thread_flag = true;
            LOGGER << "--make-erm-gz" << std::endl;
        }
        else if (flag == "--make-erm-alg") {
            make_erm_flag = true;
            make_erm_mtd = std::atoi(argv[++i]);
            thread_flag = true;
            LOGGER << "--make-erm-alg " << make_erm_mtd << std::endl;
            if (make_erm_mtd < 1 || make_erm_mtd > 3) LOGGER.e(0, "\n --make-erm-alg should be 1, 2 or 3.\n");
        } else if (flag == "--gsmr-file" ) {
            gsmr_flag = true;

            std::vector<std::string> gsmr_file_list;
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                gsmr_file_list.push_back(argv[i]);
            }
            i--;
            if (gsmr_file_list.size() != 2)
                LOGGER.e(0, "--gsmr-file, please specify the GWAS summary data for the exposure(s) and the outcome(s).");

            expo_file_list = gsmr_file_list[0];
            outcome_file_list = gsmr_file_list[1];
            LOGGER << "--gsmr-file " << expo_file_list << " " << outcome_file_list << std::endl;
            if (!std::filesystem::exists(expo_file_list)) LOGGER.e(0, "cannot open the file ["+expo_file_list+"] to read.");
            if (!std::filesystem::exists(outcome_file_list)) LOGGER.e(0, "cannot open the file ["+outcome_file_list+"] to read.");
        } else if(flag == "--gsmr2-beta") {
            gsmr_beta_version = 1;
            LOGGER << "--gsmr2-beta" << std::endl;
        } else if (flag == "--gsmr-direction") {
            gsmr_alg_flag = std::atoi(argv[++i]);
            if(gsmr_alg_flag < 0 || gsmr_alg_flag > 2)
               LOGGER.e(0, "--gsmr-direction should be 0 (forward-GSMR), 1 (reverse-GSMR) or 2 (bi-GSMR).");
            LOGGER << "--gsmr-direction " << gsmr_alg_flag << std::endl;
        } else if (flag == "--gsmr-alg") {
            LOGGER.e(0, "--gsmr-alg has been superseded by --gsmr-direction.");
        } else if (flag == "--gsmr-so") {
            gsmr_so_flag = true;
            //gsmr_so_alg = std::atoi(argv[++i]);
            if(gsmr_so_alg < 0 || gsmr_so_alg > 1)
                LOGGER.e(0, "--gsmr-so should be 0 (LD score regression) or 1 (correlation of SNP effects).");
            LOGGER << "--gsmr-so " << gsmr_so_alg << std::endl;
        } else if (flag == "--effect-plot") {
            o_snp_instru_flag = true;
            LOGGER << "--effect-plot" << std::endl;
        } else if (flag == "--mtcojo-file") {
            mtcojo_flag = true;
            mtcojolist_file = argv[++i];
            LOGGER << "--mtcojo-file " << mtcojolist_file << std::endl;
            if (!std::filesystem::exists(mtcojolist_file)) LOGGER.e(0, "cannot open the file ["+mtcojolist_file+"] to read.");
        } else if (flag == "--mtcojo-bxy") {
            mtcojo_bxy_file = argv[++i];
            LOGGER << "--mtcojo-bxy " << mtcojo_bxy_file << std::endl;
            if (!std::filesystem::exists(mtcojo_bxy_file)) LOGGER.e(0, "cannot open the file ["+mtcojo_bxy_file+"] to read.");
        } else if (flag == "--ref-ld-chr") {
            ref_ld_flag = true;
            ref_ld_dirt = argv[++i];
            chbuf = ref_ld_dirt.back();

#ifdef _WIN32
	    if(chbuf != '\\') ref_ld_dirt = ref_ld_dirt + '\\';
#elif defined __linux__ || defined __APPLE__
	    if(chbuf != '/') ref_ld_dirt = ref_ld_dirt + '/';
#else
#error Only Windows, Mac and Linux are supported.
#endif
            LOGGER << "--ref-ld-chr " << ref_ld_dirt << std::endl;
        } else if (flag == "--w-ld-chr") {
            w_ld_flag = true;
            w_ld_dirt = argv[++i];
            chbuf = w_ld_dirt.back();
#ifdef _WIN32
            if(chbuf != '\\') w_ld_dirt = w_ld_dirt + '\\';
#elif defined __linux__ || defined __APPLE__
            if(chbuf != '/') w_ld_dirt = w_ld_dirt + '/';
#else
#error Only Windows, Mac and Linux are supported.
#endif

            LOGGER << "--w-ld-chr " << w_ld_dirt << std::endl;
        } else if (flag == "--diff-freq") {
            freq_thresh = std::atof(argv[++i]);
            if(freq_thresh <0 || freq_thresh >1)
                LOGGER.e(0, "--diff-freq, Invalid threshold used to check allele frequency difference.");
            LOGGER<<"--diff-freq "<<freq_thresh<<std::endl;

        } else if (flag == "--gwas-thresh") {
            gwas_thresh = std::atof(argv[++i]);
            if(gwas_thresh <0 || gwas_thresh >1)
                LOGGER.e(0, "--gwas-thresh, Invalid GWAS p-value threshold.");
            LOGGER<<"--gwas-thresh "<<gwas_thresh<<std::endl;
        } else if (flag == "--heidi-thresh") {
            std::vector<string> thresh_list;
            while (1) {
                i++;
                if (flag == "gcta" || std::string_view{argv[i]}.substr(0, 2) == "--") break;
                thresh_list.push_back(argv[i]);
            }
            i--;
            if (thresh_list.size() < 1 || thresh_list.size() > 2)
                LOGGER.e(0, "--heidi-thresh, please specify p-value threshold(s) for the HEIDI-outlier analysis.");
            std_heidi_thresh = std::atof(thresh_list[0].c_str());
            if(thresh_list.size() > 1) {
                global_heidi_thresh = std::atof(thresh_list[1].c_str());
                global_heidi_flag = true;
            }
            if(std_heidi_thresh <0 || std_heidi_thresh >1)
                LOGGER.e(0, "--heidi-thresh, Invalid p-value threshold for single-SNP-based HEIDI-outlier test.");
            if(global_heidi_thresh <0 || global_heidi_thresh >1)
                LOGGER.e(0, "--heidi-thresh, Invalid p-value threshold for multi-SNP-based HEIDI-outlier test.");

            LOGGER<<"--heidi-thresh "<<std_heidi_thresh;
            if(thresh_list.size() > 1) LOGGER<<" "<<global_heidi_thresh;
            LOGGER<<std::endl;
        } else if (flag == "--heidi-snp") {
            LOGGER.e(0, "--heidi-snp is discontinued. Please use --gsmr-snp-min to specify minimum number of SNP instruments for the HEIDI-outlier analysis.");
        } else if ((flag == "--gsmr-snp") || (flag == "--gsmr-snp-min")) {
            if(flag == "--gsmr-snp") gsmr_snp_update_flag = true;
            nsnp_gsmr = std::atoi(argv[++i]);
            if(nsnp_gsmr < 0)
                LOGGER.e(0, "--gsmr-snp-min, Invalid SNP number threshold for the GSMR analysis.");
            LOGGER<<"--gsmr-snp-min "<<nsnp_gsmr<<std::endl;
        } else if (flag == "--gsmr-ld-fdr") {
            ld_fdr_thresh = std::atoi(argv[++i]);
            if(ld_fdr_thresh < 0 || ld_fdr_thresh > 1)
                LOGGER.e(0, "--gsmr-ld-fdr, Invalid FDR threshold for LD correlation matrix.");
        } else {
            std::stringstream errmsg;
            errmsg << "\n  invalid option \"" << argv[i] << "\".\n";
            LOGGER.e(0, errmsg.str());
        }
    }
    if (save_reml_flag) save_reml_file = out + ".reml";

    // conflicted options
    LOGGER << std::endl;
    if (bfile2_flag && !bfile_flag) LOGGER.e(0, "the option --bfile2 should always go with the option --bfile.");
    if(bfile_flag && grm_cutoff>-1.0) LOGGER.e(0, "the --grm-cutoff option is invalid when used in combination with the --bfile option.");
    if (m_grm_flag) {
        if (grm_flag) {
            grm_flag = false;
            LOGGER.w(0, "--grm option suppressed by the --mgrm option.");
        }
        if (grm_cutoff>-1.0) {
            grm_cutoff = -2.0;
            LOGGER.w(0, "--grm-cutoff option suppressed by the --mgrm option.");
        }
    }
    if (pca_flag) {
        if (grm_adj_fac>-1.0) {
            grm_adj_fac = -2.0;
            LOGGER.w(0, "--grm-adj option suppressed by the --pca-v1 option.");
        } else if (dosage_compen>-1) {
            grm_adj_fac = -2;
            LOGGER.w(0, "--dosage-compen option suppressed by the --pca-v1 option.");
        }
    }
    if (!gxe_file.empty() && !grm_flag && !m_grm_flag) {
        LOGGER.w(0, "--gxe option is ignored because there is no --grm or --mgrm option specified.");
        gxe_file = "";
    }
    if (pred_rand_eff && !grm_flag && !m_grm_flag) {
        LOGGER.w(0, "--reml-pred-rand option is ignored because there is no --grm or --mgrm option specified.");
        pred_rand_eff = false;
    }
    if (cv_blup && !grm_flag && !m_grm_flag) {
        LOGGER.w(0, "--cvblup option is ignored because there is no --grm or --mgrm option specified.");
        cv_blup = false;
    }
    if(cv_blup && pred_rand_eff){
        LOGGER.w(0, "--reml-pred-rand options is ignored because --cvblup does more than this option");
        pred_rand_eff = false;
    }

    if (dosage_compen>-1 && update_sex_file.empty()) LOGGER.e(0, "you need to specify the sex information for the individuals by the option --update-sex because of the use of the –dc option.");
    if (bfile2_flag && update_freq_file.empty()) LOGGER.e(0, "you need to update the allele frequency by the option --update-freq because there are two datasets.");
    if ((dose_beagle_flag || dose_mach_flag || dose_mach_gz_flag) && dominance_flag) LOGGER.e(0, "unable to calculate the GRM for dominance effect using imputed dosage data.");
    if (make_grm_homogametic_flag && dominance_flag) LOGGER.e(0, "unable to calculate the GRM for dominance effect for homogametic chromosomes.");
    if (mlma_flag || mlma_loco_flag) {
        if (!gxe_file.empty()) LOGGER.w(0, "the option --gxe option is disabled in this analysis.");
        if (!update_sex_file.empty()) LOGGER.w(0, "the option --update-sex option is disabled in this analysis.");
        if (grm_adj_fac>-1.0) LOGGER.w(0, "the option --grm-adj option is disabled in this analysis.");
        if (dosage_compen>-1.0) LOGGER.w(0, "the option --dc option is disabled in this analysis.");
        if (est_fix_eff) LOGGER.w(0, "the option --reml-est-fix option is disabled in this analysis.");
        if (est_fix_eff_var) LOGGER.w(0, "the option --reml-est-fix-varcov option is disabled in this analysis.");
        if (pred_rand_eff) LOGGER.w(0, "the option --reml-pred-rand option is disabled in this analysis.");
        if(cv_blup) LOGGER.w(0, "the option --cvblup option is disabled in this analysis.");
        if (reml_lrt_flag) LOGGER.w(0, "the option --reml-lrt option is disabled in this analysis.");
        if (!save_reml_file.empty() && !load_reml_file.empty()) LOGGER.e(0, "--save-reml and --load-reml cannot be used together.");
        if ((!save_reml_file.empty() || !load_reml_file.empty()) && mlma_loco_flag) LOGGER.e(0, "--save-reml and --load-reml are not supported with --mlma-loco.");
        if (!load_reml_file.empty() && !mlma_flag) LOGGER.e(0, "--load-reml can only be used with --mlma.");
        if (!save_reml_file.empty() && !mlma_flag) LOGGER.e(0, "--save-reml can only be used with --mlma.");
        if (!save_reml_file.empty() && !grm_flag && !m_grm_flag) LOGGER.e(0, "--save-reml requires an explicit GRM via --grm or --mgrm.");
    }
    if(bivar_reml_flag && prevalence_flag) LOGGER.e(0, "--prevalence option is not compatible with --reml-bivar option. Please check the --reml-bivar-prevalence option!");
    if(gsmr_flag || mtcojo_flag){
        if(ref_ld_flag && !w_ld_flag) LOGGER.e(0, "--ref-ld-chr, please specify the directory of the LD score files.");
        if(!ref_ld_flag && w_ld_flag) LOGGER.e(0, "--w-ld-chr, please specify the directory of the LD scores for the regression weights.");
        if(gsmr_snp_update_flag) LOGGER.w(0, "--gsmr-snp has been superseded by --gsmr-snp-min.");
        if(nsnp_gsmr < 5) LOGGER.w(0, "The number of SNP instruments included in the analysis is too small. There might not be enough SNPs to perform the HEIDI-outlier analysis.");
        // if(!gsmr_so_flag && ref_ld_flag && w_ld_flag) { gsmr_so_alg = 0; LOGGER.w(0, "--gsmr-so is not specified. The default value is 0. GSMR analysis will perform LD score regression to estimate sample overlap."); }
        // if(gsmr_so_alg == 1 && ref_ld_flag && w_ld_flag) { gsmr_so_alg = 0; LOGGER.w(0, "The LD score regression instead of correlation method will be used to estimate sample overlap."); }
        // if(gsmr_so_alg == 0 && !ref_ld_flag && !w_ld_flag) LOGGER.e(0, "Please specify the directory of LD score files to perform the LD score regression analysis.");
        // if(!gsmr_so_flag && !ref_ld_flag && !w_ld_flag) LOGGER.w(0, "The GSMR analysis will be performed assuming no sample overlap between the GWAS data for exposure and outcome.");
    }
    if(gsmr_beta_version && !global_heidi_flag) {
        LOGGER.w(0, "The threshold of multi-SNP-based HEIDI-outlier analysis is not specified. The default value is " + std::to_string(global_heidi_thresh).substr(0,5) + ".");
    }
    if(!gsmr_beta_version && global_heidi_flag) {
        LOGGER.w(0, "--gsmr2-beta is not specified. GCTA will perform single-SNP-based HEIDI-outlier analysis, which was published in Zhu et al. 2018 Nature Communications. The threshold of multi-SNP-based HEIDI-outlier analysis will not be accepted.");
    }
    if(pcl_flag && gwas_data_flag) {
        pcl_flag = false; gwas_adj_pc_flag = true; thread_flag = false;
    }
    // OpenMP
    if (thread_flag) {
        if (thread_num == 1) LOGGER.i(0, "This is a multi-thread program. You could specify the number of threads by the --thread-num option to speed up the computation if there are multiple processors in your machine.", "Note:");
        else LOGGER.i(0, "the program will be running on " + std::to_string(thread_num) + " threads.", "Note:");
    }

    // set autosome
    if (autosome_flag) {
        if(!autosome_num_explicit){
            LOGGER.w(0, "--autosome was specified without --autosome-num; all positive numeric chromosome labels are treated as autosomal by default.");
        } else {
            if(extract_chr_start == extract_chr_end && extract_chr_start != 0){
                LOGGER.w(0, "--autosome option omitted. You have specified the chromosome to analysis.");
            }else{
                extract_chr_start = 1;
                extract_chr_end = autosome_num;
            }
        }
    }
    if (make_grm_homogametic_flag) {
        if(!(extract_chr_start == extract_chr_end && extract_chr_start > 0)){
            LOGGER.e(0, "--make-grm-homogametic in the legacy pipeline requires --chr to specify the homogametic chromosome explicitly.");
        }
    }

    // Implement
    LOGGER << std::endl;
    gcta *pter_gcta = new gcta(autosome_num, rm_high_ld_cutoff, out);

    if (!genetic_model.empty()) pter_gcta->set_genetic_model(genetic_model);
    if(ldscore_adj_flag) pter_gcta->set_ldscore_adj_flag(ldscore_adj_flag);
    if(reml_force_inv_fac_flag) pter_gcta->set_reml_force_inv();
    if(logp_flag) pter_gcta->set_log_pval(true);
    if(reml_force_converge_flag) pter_gcta->set_reml_force_converge();
    if(reml_no_converge_flag) pter_gcta->set_reml_no_converge();
    if(reml_fixed_var_flag) pter_gcta->set_reml_fixed_var();
    if(reml_allow_constrain_run) pter_gcta->set_reml_allow_constrain_run();
    if(reml_trace_approx) pter_gcta->set_reml_trace_approx(true, reml_trace_nprobes);
    pter_gcta->set_reml_trace_power_iter(reml_trace_power_iter);
    if(reml_woodbury_rank > 0)
        pter_gcta->set_reml_woodbury_rank(reml_woodbury_rank);
    else if(reml_woodbury_rank < 0)
        pter_gcta->set_reml_woodbury_auto(reml_woodbury_buffer, reml_woodbury_k_max);
    if(reml_woodbury_nystrom) pter_gcta->set_reml_woodbury_nystrom();
    if(reml_mtd != 0) pter_gcta->set_reml_mtd(reml_mtd);
    if(reml_inv_method != 0) pter_gcta->set_reml_inv_method(reml_inv_method);
    pter_gcta->set_reml_diagV_adj(reml_diagV_adj);
    pter_gcta->set_reml_diag_mul(reml_diag_mul);
    pter_gcta->set_diff_freq(freq_thresh);
    if (grm_bin_flag || m_grm_bin_flag) pter_gcta->enable_grm_bin_flag();
    //if(simu_unlinked_flag) pter_gcta->simu_geno_unlinked(simu_unlinked_n, simu_unlinked_m, simu_unlinked_maf);
    if (!RG_fname_file.empty()) {
        if (RG_summary_file.empty()) LOGGER.e(0, "please input the summary information for the raw data files by the option --raw-summary.");
        pter_gcta->read_IRG_fnames(RG_summary_file, RG_fname_file, GC_cutoff);
    }
    else if(efile_flag){
        pter_gcta->read_efile(efile);
        if (make_erm_flag) pter_gcta->make_erm(make_erm_mtd - 1, grm_out_bin_flag);
        else if(ecojo_slct_flag) pter_gcta->run_ecojo_slct(ecojo_ma_file, ecojo_p, ecojo_collinear);
        else if(ecojo_blup_flag) pter_gcta->run_ecojo_blup_efile(ecojo_ma_file, ecojo_lambda);
    }
    else if(eR_file_flag){
        pter_gcta->read_eR(eR_file);
        pter_gcta->run_ecojo_blup_eR(ecojo_ma_file, ecojo_lambda);
    }
    else if (bfile_flag) {
        if (hapmap_genet_dst) pter_gcta->genet_dst(bfile, hapmap_genet_dst_file);
        else {
            if (bfile2_flag) {
                LOGGER << "There are two datasets specified (in PLINK binary PED format).\nReading dataset 1 ..." << std::endl;
                if (update_freq_file.empty()) LOGGER.e(0, "since there are two datasets, you should update the allele frequencies calculated from the combined dataset.");
            }
            // Read the list, if there are multiple bfiles
            if(bfile_flag==2) multi_bfiles = pter_gcta->read_bfile_list(bfile_list);
            // For --save-reml --mlma only the .fam is needed to build the sample intersection;
            // the .bim and .bed data are not loaded.
            const bool fam_only = save_reml_flag && mlma_flag;
            // Start to read the genotypes
            if(bfile_flag==1) pter_gcta->read_famfile(bfile + ".fam");
            else pter_gcta->read_multi_famfiles(multi_bfiles);
            if (!kp_indi_file.empty()) pter_gcta->keep_indi(kp_indi_file);
            if (!rm_indi_file.empty()) pter_gcta->remove_indi(rm_indi_file);
            if (!update_sex_file.empty()) pter_gcta->update_sex(update_sex_file);
            if (!blup_indi_file.empty()) pter_gcta->read_indi_blup(blup_indi_file);
            if (!fam_only) {
                if(bfile_flag==1) pter_gcta->read_bimfile(bfile + ".bim");
                else pter_gcta->read_multi_bimfiles(multi_bfiles);
                if (!extract_snp_file.empty()) pter_gcta->extract_snp(extract_snp_file);
                if (extract_chr_start > 0) pter_gcta->extract_chr(extract_chr_start, extract_chr_end);
                if(extract_region_chr>0) pter_gcta->extract_region_bp(extract_region_chr, extract_region_bp, extract_region_wind);
                if (!extract_snp_name.empty()){
                    if(extract_region_wind>0) pter_gcta->extract_region_snp(extract_snp_name, extract_region_wind);
                    else pter_gcta->extract_single_snp(extract_snp_name);
                }
                if (!exclude_snp_file.empty()) pter_gcta->exclude_snp(exclude_snp_file);
                if(exclude_region_chr>0) pter_gcta->exclude_region_bp(exclude_region_chr, exclude_region_bp, exclude_region_wind);
                if (!exclude_snp_name.empty()) {
                    if(exclude_region_wind>0) pter_gcta->exclude_region_snp(exclude_snp_name, exclude_region_wind);
                    else pter_gcta->exclude_single_snp(exclude_snp_name);
                }
                if (!update_refA_file.empty()) pter_gcta->update_allele_ref(update_refA_file);
                if (LD) pter_gcta->read_LD_target_SNPs(LD_file);
                if(gsmr_flag) pter_gcta->read_gsmrfile(expo_file_list, outcome_file_list, gwas_thresh, nsnp_gsmr, gsmr_so_alg);
                if(mtcojo_flag) nsnp_read = pter_gcta->read_mtcojofile(mtcojolist_file, gwas_thresh, nsnp_gsmr);
                if(mtcojo_flag && nsnp_read>0) {
                    if(bfile_flag==1) {
                        if (genetic_model != "") {
                            pter_gcta->read_bed_dosage(bfile + ".bed");
                        } else {
                            pter_gcta->read_bedfile(bfile + ".bed");
                        }
                    } else {
                        pter_gcta->read_multi_bedfiles(multi_bfiles);
                    }
                }
                if(!mtcojo_flag){
                    if(bfile_flag==1) {
                        if (genetic_model != "") {
                            pter_gcta->read_bed_dosage(bfile + ".bed");
                        } else {
                            pter_gcta->read_bedfile(bfile + ".bed");
                        }
                    } else {
                        pter_gcta->read_multi_bedfiles(multi_bfiles);
                    }
                }
            } // end !fam_only

            if (!update_impRsq_file.empty()) pter_gcta->update_impRsq(update_impRsq_file);
            if (!update_freq_file.empty()) pter_gcta->update_freq(update_freq_file);
            if (dose_Rsq_cutoff > 0.0) pter_gcta->filter_impRsq(dose_Rsq_cutoff);
            if (maf > 0) pter_gcta->filter_snp_maf(maf);
            if (max_maf > 0.0) pter_gcta->filter_snp_max_maf(max_maf);
            if (out_freq_flag) pter_gcta->save_freq(out_ssq_flag);
            else if (!paa_file.empty()) pter_gcta->paa(paa_file);
            else if (ibc) pter_gcta->ibc(ibc_all);
            else if (make_grm_flag) pter_gcta->make_grm(dominance_flag, make_grm_homogametic_flag, make_grm_inbred_flag, grm_out_bin_flag, make_grm_mtd, false, make_grm_f3_flag, subpopu_file);
            else if (recode || recode_nomiss || recode_std) pter_gcta->save_XMat(recode_nomiss, recode_std);
            else if (LD) pter_gcta->LD_Blocks(LD_step, LD_wind, LD_sig, LD_i, save_ram);
            else if (LD_prune_rsq>-1.0) pter_gcta->LD_pruning(LD_prune_rsq, LD_wind);
            else if (ld_score_flag){
                if(ld_score_multi_file.empty()) pter_gcta->calcu_mean_rsq(LD_wind, LD_rsq_cutoff, dominance_flag);
                else pter_gcta->calcu_mean_rsq_multiSet(ld_score_multi_file, LD_wind, LD_rsq_cutoff, dominance_flag);
            }
            else if (ld_mean_rsq_seg_flag) pter_gcta->ld_seg(LD_file, LD_seg, LD_wind, LD_rsq_cutoff, dominance_flag);
            else if (ld_max_rsq_flag) pter_gcta ->calcu_max_ld_rsq(LD_wind, LD_rsq_cutoff, dominance_flag);
            else if (blup_snp_flag) pter_gcta->blup_snp_geno();
            else if (mlma_flag) {
                // Lazy load genotypes for mlma if deferred (when using external GRM or pre-computed REML)
                if (grm_file.empty() && load_reml_file.empty()) {
                    // Need to load genotypes for GRM computation
                    LOGGER << "Loading genotype data for GRM computation..." << endl;
                    if(bfile_flag==1) pter_gcta->read_bedfile(bfile + ".bed");
                    else pter_gcta->read_multi_bedfiles(multi_bfiles);
                } else {
                    LOGGER << "Using external GRM or pre-computed REML (genotypes loaded as needed for association testing)" << endl;
                }
                pter_gcta->mlma(grm_file, m_grm_flag, subtract_grm_file, phen_file, qcovar_file, covar_file, mphen, MaxIter, reml_priors, reml_priors_var, no_constrain, within_family, make_grm_inbred_flag, mlma_no_adj_covar, weight_file, save_reml_file, load_reml_file);
            }
            else if (mlma_loco_flag) {
                if (!grm_file.empty() && !m_grm_flag) {
                    // Memory-efficient path: G_loco_c = (G_all*m_all − G_chr_c*m_c)/(m_all−m_c).
                    // Peak RAM = O(3·n²) regardless of n_chr, vs O(n_chr·n²) for the legacy path.
                    pter_gcta->mlma_loco_v2(grm_file, grm_chr_prefix, phen_file, qcovar_file, covar_file, mphen, MaxIter, reml_priors, reml_priors_var, no_constrain, make_grm_inbred_flag, mlma_no_adj_covar);
                } else {
                    if (!grm_file.empty() && m_grm_flag)
                        LOGGER.w(0, "--mlma-loco with --mgrm: multiple-GRM lists are not supported by the memory-efficient path. Falling back to legacy all-in-memory LOCO. Use --grm (single all-autosome GRM) to enable the efficient path.");
                    pter_gcta->mlma_loco(phen_file, qcovar_file, covar_file, mphen, MaxIter, reml_priors, reml_priors_var, no_constrain, make_grm_inbred_flag, mlma_no_adj_covar);
                }
            }
            else if (massoc_slct_flag | massoc_joint_flag) {pter_gcta->set_massoc_pC_thresh(massoc_out_pC_thresh); pter_gcta->run_massoc_slct(massoc_file, massoc_wind, massoc_p, massoc_collinear, massoc_top_SNPs, massoc_joint_flag, massoc_gc_flag, massoc_gc_val, massoc_actual_geno_flag, massoc_mld_slct_alg);}
            else if (!massoc_cond_snplist.empty()) {pter_gcta->set_massoc_pC_thresh(massoc_out_pC_thresh); pter_gcta->run_massoc_cond(massoc_file, massoc_cond_snplist, massoc_wind, massoc_collinear, massoc_gc_flag, massoc_gc_val, massoc_actual_geno_flag);}
            else if (massoc_sblup_flag) pter_gcta->run_massoc_sblup(massoc_file, massoc_wind, massoc_sblup_fac);
            else if (gsmr_flag) pter_gcta->gsmr(gsmr_alg_flag, ref_ld_dirt, w_ld_dirt, freq_thresh, gwas_thresh, clump_wind_size, clump_r2_thresh, std_heidi_thresh, global_heidi_thresh, ld_fdr_thresh, nsnp_gsmr, o_snp_instru_flag, gsmr_so_alg, gsmr_beta_version);
            else if (mtcojo_flag) pter_gcta->mtcojo(mtcojo_bxy_file, ref_ld_dirt, w_ld_dirt, freq_thresh, gwas_thresh, clump_wind_size, clump_r2_thresh, std_heidi_thresh, global_heidi_thresh, ld_fdr_thresh, nsnp_gsmr, gsmr_beta_version);
            else if (simu_qt_flag || simu_cc) pter_gcta->GWAS_simu(bfile, simu_rep, simu_causal, simu_case_num, simu_control_num, simu_h2, simu_K, simu_seed, simu_output_causal, simu_emb_flag, simu_eff_mod);
            else if (make_bed_flag) pter_gcta->save_plink();
            else if (fst_flag) pter_gcta->Fst(subpopu_file);
            else if (!sbat_sAssoc_file.empty()){
                //if(!sbat_gAnno_file.empty()) pter_gcta->sbat_gene_old(sbat_sAssoc_file, sbat_gAnno_file, sbat_wind, sbat_ld_cutoff, sbat_write_snpset);
                if(!sbat_gAnno_file.empty()) pter_gcta->sbat_gene( sbat_sAssoc_file,sbat_gAnno_file, sbat_wind, sbat_ld_cutoff, sbat_write_snpset, massoc_gc_flag, massoc_gc_val);
                else if(!sbat_snpset_file.empty()) pter_gcta->sbat(sbat_sAssoc_file, sbat_snpset_file, sbat_ld_cutoff, sbat_write_snpset,massoc_gc_flag, massoc_gc_val);
                else if(sbat_seg_flag) pter_gcta->sbat_seg(sbat_sAssoc_file, sbat_seg_size, sbat_ld_cutoff, sbat_write_snpset,massoc_gc_flag, massoc_gc_val);
            }
            else if(!mbat_sAssoc_file.empty()){
               if(!mbat_gAnno_file.empty()) pter_gcta->mbat_gene(mbat_sAssoc_file, mbat_gAnno_file, mbat_wind,mbat_svd_gamma, sbat_ld_cutoff, mbat_write_snpset,massoc_gc_flag, massoc_gc_val,mbat_print_all_p);
               else if(!mbat_snpset_file.empty()) pter_gcta->mbat(mbat_sAssoc_file, mbat_snpset_file, mbat_svd_gamma, sbat_ld_cutoff, mbat_write_snpset,massoc_gc_flag, massoc_gc_val,mbat_print_all_p);
            }
            else if(gwas_adj_pc_flag) { pcl_flag=false; pter_gcta->pc_adjust(pcadjust_list_file, pc_file, freq_thresh, pc_adj_wind_size); }
            else if(pcl_flag) pter_gcta->snp_pc_loading(pc_file);
            else if(project_flag) pter_gcta->project_loading(project_file, project_N);
        }
    } else if (dose_beagle_flag || dose_mach_flag || dose_mach_gz_flag) {
        if (massoc_slct_flag | massoc_joint_flag | !massoc_cond_snplist.empty()) LOGGER.e(0, "the --dosage option can't be used in combination with the --cojo options.");
        if (dose_mach_flag) pter_gcta->read_imp_info_mach(dose_info_file);
        else if (dose_mach_gz_flag) pter_gcta->read_imp_info_mach_gz(dose_info_file);
        else if (dose_beagle_flag) pter_gcta->read_imp_info_beagle(dose_info_file);
        if (!extract_snp_file.empty()) pter_gcta->extract_snp(extract_snp_file);
        if (!exclude_snp_file.empty()) pter_gcta->exclude_snp(exclude_snp_file);
        if (!extract_snp_name.empty()) pter_gcta->extract_single_snp(extract_snp_name);
        if (!exclude_snp_name.empty()) pter_gcta->exclude_single_snp(exclude_snp_name);
        if (extract_chr_start > 0) LOGGER.w(0, "the option --chr, --autosome or --nonautosome is inactive for dosage data.");
        if (!update_refA_file.empty()) pter_gcta->update_allele_ref(update_refA_file);
        if (dose_mach_flag) pter_gcta->read_imp_dose_mach(dose_file, kp_indi_file, rm_indi_file, blup_indi_file);
        else if (dose_mach_gz_flag) pter_gcta->read_imp_dose_mach_gz(dose_file, kp_indi_file, rm_indi_file, blup_indi_file);
        else if (dose_beagle_flag) pter_gcta->read_imp_dose_beagle(dose_file, kp_indi_file, rm_indi_file, blup_indi_file);
        if (!update_sex_file.empty()) pter_gcta->update_sex(update_sex_file);
        if (!update_impRsq_file.empty()) pter_gcta->update_impRsq(update_impRsq_file);
        if (!update_freq_file.empty()) pter_gcta->update_freq(update_freq_file);
        if (dose_Rsq_cutoff > 0.0) pter_gcta->filter_impRsq(dose_Rsq_cutoff);
        if (maf > 0.0) pter_gcta->filter_snp_maf(maf);
        if (max_maf > 0.0) pter_gcta->filter_snp_max_maf(max_maf);
        if (out_freq_flag) pter_gcta->save_freq(out_ssq_flag);
        else if (make_grm_flag) pter_gcta->make_grm(dominance_flag, make_grm_homogametic_flag, make_grm_inbred_flag, grm_out_bin_flag, make_grm_mtd, false, make_grm_f3_flag, subpopu_file);
        else if (recode || recode_nomiss || recode_std) pter_gcta->save_XMat(recode_nomiss, recode_std);
        else if (LD_prune_rsq>-1.0) pter_gcta->LD_pruning(LD_prune_rsq, LD_wind);
        else if (ld_score_flag){
                if(ld_score_multi_file.empty()) pter_gcta->calcu_mean_rsq(LD_wind, LD_rsq_cutoff, dominance_flag);
                else pter_gcta->calcu_mean_rsq_multiSet(ld_score_multi_file, LD_wind, LD_rsq_cutoff, dominance_flag);
        }
        else if (ld_max_rsq_flag) pter_gcta ->calcu_max_ld_rsq(LD_wind, LD_rsq_cutoff, dominance_flag);
        else if (blup_snp_flag) pter_gcta->blup_snp_dosage();
        else if (massoc_sblup_flag) {pter_gcta->set_diff_freq(freq_thresh);pter_gcta->run_massoc_sblup(massoc_file, massoc_wind, massoc_sblup_fac);}
        else if (simu_qt_flag || simu_cc) pter_gcta->GWAS_simu(bfile, simu_rep, simu_causal, simu_case_num, simu_control_num, simu_h2, simu_K, simu_seed, simu_output_causal, simu_emb_flag, simu_eff_mod);
        else if (make_bed_flag) pter_gcta->save_plink();
        else if (fst_flag) pter_gcta->Fst(subpopu_file);
        else if (mlma_flag) pter_gcta->mlma(grm_file, m_grm_flag, subtract_grm_file, phen_file, qcovar_file, covar_file, mphen, MaxIter, reml_priors, reml_priors_var, no_constrain, within_family, make_grm_inbred_flag, mlma_no_adj_covar, weight_file, save_reml_file, load_reml_file);
        else if (mlma_loco_flag) pter_gcta->mlma_loco(phen_file, qcovar_file, covar_file, mphen, MaxIter, reml_priors, reml_priors_var, no_constrain, make_grm_inbred_flag, mlma_no_adj_covar);
    } else if (HE_reg_flag) pter_gcta->HE_reg(grm_file, m_grm_flag, phen_file, kp_indi_file, rm_indi_file, mphen);
    else if (HE_reg_bivar_flag) pter_gcta->HE_reg_bivar(grm_file, m_grm_flag, phen_file, kp_indi_file, rm_indi_file, mphen, mphen2);
    else if ((reml_flag || bivar_reml_flag) && phen_file.empty()) LOGGER.e(0, "\n a phenotype file is required for reml analysis.\n");
    else if (bivar_reml_flag) {
        pter_gcta->set_cv_blup(cv_blup);
        pter_gcta->fit_bivar_reml(grm_file, phen_file, qcovar_file, covar_file, kp_indi_file, rm_indi_file, update_sex_file, mphen, mphen2, grm_cutoff, grm_adj_fac, dosage_compen, m_grm_flag, pred_rand_eff, est_fix_eff, est_fix_eff_var, reml_mtd, MaxIter, reml_priors, reml_priors_var, reml_drop, no_lrt, prevalence, prevalence2, no_constrain, ignore_Ce, fixed_rg_val, bivar_no_constrain);
    } else if (reml_flag) {
        pter_gcta->set_cv_blup(cv_blup);
        pter_gcta->fit_reml(grm_file, phen_file, qcovar_file, covar_file, qgxe_file, gxe_file, kp_indi_file, rm_indi_file, update_sex_file, mphen, grm_cutoff, grm_adj_fac, dosage_compen, m_grm_flag, pred_rand_eff, est_fix_eff, est_fix_eff_var, reml_mtd, MaxIter, reml_priors, reml_priors_var, reml_drop, no_lrt, prevalence, no_constrain, mlma_flag, within_family, reml_bending, reml_diag_one, weight_file);
    } else if (grm_flag || m_grm_flag) {
        if (pca_flag) pter_gcta->pca(grm_file, kp_indi_file, rm_indi_file, grm_cutoff, m_grm_flag, out_pc_num, pca_approx_flag);
        else if (make_grm_flag) pter_gcta->save_grm(grm_file, kp_indi_file, rm_indi_file, update_sex_file, grm_cutoff, grm_adj_fac, dosage_compen, m_grm_flag, grm_out_bin_flag);
        else if (align_grm_flag) pter_gcta->align_grm(grm_file);
        else if (bK_threshold > -1) pter_gcta->grm_bK(grm_file, kp_indi_file, rm_indi_file, bK_threshold, grm_out_bin_flag);
        else if (!denseness_metric.empty()) pter_gcta->grm_denseness(grm_file, kp_indi_file, rm_indi_file, m_grm_flag, denseness_metric);
    }
    else if (ld_mean_rsq_seg_flag) pter_gcta->ld_seg(LD_file, LD_seg, LD_wind, LD_rsq_cutoff, dominance_flag);
    else LOGGER.e(0, "no analysis has been launched by the option(s).\n");

    delete pter_gcta;
}
