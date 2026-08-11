/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * Interface to all the GCTA functions
 *
 * 2010 by Jian Yang <jian.yang.qt@gmail.com>
 *
 * This file is distributed under the GNU General Public
 * License, Version 3.  Please see the file COPYING for more
 * details
 */

#ifndef _GCTA_H
#define _GCTA_H

#include "cpu.h"
#include <cstdio>
#include "CommFunc.h"
#include "StrFunc.h"
#include "StatFunc.h"
#include <fstream>
#include <iomanip>
#include <bitset>
#include <map>
#include <span>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/IterativeSolvers>
#include <omp.h>
#include "Logger.h"
#include "Matrix.hpp"
#include "mlma_woodbury.hpp"

#ifdef SINGLE_PRECISION
typedef Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic> eigenDiagMat;
typedef Eigen::MatrixXf eigenMatrix;
typedef Eigen::VectorXf eigenVector;
typedef Eigen::SparseMatrix<float> eigenDynSparseMat;
typedef Eigen::SparseMatrix<float, Eigen::ColMajor, long long> eigenSparseMat;
#else
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenDiagMat;
typedef Eigen::MatrixXd eigenMatrix;
typedef Eigen::VectorXd eigenVector;
typedef Eigen::SparseMatrix<double> eigenDynSparseMat;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor, long long> eigenSparseMat;
#endif

enum class GeneticModel {
    ADDITIVE,
    NONADDITIVE
};

// Sentinel value for missing dosage entries (infinity compares > any valid dosage)
inline const float DOSAGE_NA = std::numeric_limits<float>::infinity();

// Helper functions for GeneticModel enum
inline bool stringToGeneticModel(const std::string& str, GeneticModel& model) {
    if (str == "additive") {
        model = GeneticModel::ADDITIVE;
        return true;
    } else if (str == "nonadditive") {
        model = GeneticModel::NONADDITIVE;
        return true;
    }
    return false;
}

inline std::string geneticModelToString(GeneticModel model) {
    switch (model) {
        case GeneticModel::ADDITIVE:    return "additive";
        case GeneticModel::NONADDITIVE: return "nonadditive";
        default: return "unknown";
    }
}

class gcta {
public:
    gcta(int autosome_num, double rm_ld_cutoff, std::string out);
    gcta();
    virtual ~gcta();

    void read_famfile(std::string famfile);
    void read_bimfile(std::string bimfile);
    void read_bedfile(std::string bedfile);
    void read_bed_dosage(std::string bedfile); // Read bed file and fill _geno_dose with dosage calculated from genotypes
    std::vector<std::string> read_bfile_list(std::string bfile_list);
    void read_multi_famfiles(std::vector<std::string> multi_bfiles);
    void read_multi_bimfiles(std::vector<std::string> multi_bfiles);
    void read_multi_bedfiles(std::vector<std::string> multi_bfiles);
    void read_imp_info_mach_gz(std::string zinfofile);
    void read_imp_info_mach(std::string infofile);
    void read_imp_dose_mach_gz(std::string zdosefile, std::string kp_indi_file, std::string rm_indi_file, std::string blup_indi_file);
    void read_imp_dose_mach(std::string dosefile, std::string kp_indi_file, std::string rm_indi_file, std::string blup_indi_file);
    void read_imp_info_beagle(std::string zinfofile);
    void read_imp_dose_beagle(std::string zdosefile, std::string kp_indi_file, std::string rm_indi_file, std::string blup_indi_file);
    void update_allele_ref(std::string ref_A_file);
    void update_impRsq(std::string zinfofile);
    void update_freq(std::string freq);
    void save_freq(bool ssq_flag);
    void extract_snp(std::string snplistfile);
    void extract_single_snp(std::string snpname);
    void extract_region_snp(std::string snpname, int wind_size);
    void extract_region_bp(int chr, int bp, int wind_size);
    void exclude_snp(std::string snplistfile);
    void exclude_single_snp(std::string snpname);
    void exclude_region_snp(std::string snpname, int wind_size);
    void exclude_region_bp(int chr, int bp, int wind_size);
    void extract_chr(int chr_start, int chr_end);
    void filter_snp_maf(double maf);
    void filter_snp_max_maf(double max_maf);
    void filter_impRsq(double rsq_cutoff);
    void keep_indi(std::string indi_list_file);
    void remove_indi(std::string indi_list_file);
    void update_sex(std::string sex_file);
    void read_indi_blup(std::string blup_indi_file);
    void save_XMat(bool miss_with_mu, bool std);

    void make_grm(bool grm_d_flag, bool grm_homogametic_flag, bool inbred, bool output_bin, int grm_mtd, bool mlmassoc, bool diag_f3_flag = false, std::string subpopu_file = "");
    //void make_grm_pca(bool grm_d_flag, bool grm_homogametic_flag, bool inbred, bool output_bin, int grm_mtd, double wind_size, bool mlmassoc);
    void save_grm(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, std::string sex_file, double grm_cutoff, double adj_grm_fac, int dosage_compen, bool merge_grm_flag, bool output_grm_bin);
    void align_grm(std::string m_grm_file);
    void pca(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, double grm_cutoff, bool merge_grm_flag, int out_pc_num, std::string pca_approx = "");
    void snp_pc_loading(std::string pc_file);
    void project_loading(std::string pc_load, int N); 

    // bigK + smallK method
    void grm_bK(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, double threshold, bool grm_out_bin_flag);

    void grm_denseness(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, bool merge_grm_flag, std::string metric);

    void enable_grm_bin_flag();
    void fit_reml(std::string grm_file, std::string phen_file, std::string qcovar_file, std::string covar_file, std::string qGE_file, std::string GE_file, std::string keep_indi_file, std::string remove_indi_file, std::string sex_file, int mphen, double grm_cutoff, double adj_grm_fac, int dosage_compen, bool m_grm_flag, bool pred_rand_eff, bool est_fix_eff, bool est_fix_eff_var, int reml_mtd, int MaxIter, std::vector<double> reml_priors, std::vector<double> reml_priors_var, std::vector<int> drop, bool no_lrt, double prevalence, bool no_constrain, bool mlmassoc = false, bool within_family = false, bool reml_bending = false, bool reml_diag_one = false, std::string weight_file = "");
    //void HE_reg(std::string grm_file, std::string phen_file, std::string keep_indi_file, std::string remove_indi_file, int mphen); // old HE regression method
    void HE_reg(std::string grm_file, bool m_grm_flag, std::string phen_file, std::string keep_indi_file, std::string remove_indi_file, int mphen); // allow multiple regression
    void HE_reg_bivar(std::string grm_file, bool m_grm_flag, std::string phen_file, std::string keep_indi_file, std::string remove_indi_file, int mphen, int mphen2); // estimate genetic covariance between two traits
    void blup_snp_geno();
    void blup_snp_dosage();
    void set_reml_force_inv();
    void set_reml_force_converge();
    void set_genetic_model(std::string model);
    void set_cv_blup(bool cv_blup);
    void set_reml_no_converge();
    void set_reml_fixed_var();
    void set_reml_mtd(int reml_mtd);
    void set_reml_allow_constrain_run();
    void set_reml_diag_mul(double value);
    void set_reml_diagV_adj(int method);
    void set_reml_trace_approx(bool on, int nprobes);
    void set_reml_trace_power_iter(int q);
    void set_log_pval(bool log_pval);
    void set_reml_inv_method(int method);

    // bivariate REML analysis
    void fit_bivar_reml(std::string grm_file, std::string phen_file, std::string qcovar_file, std::string covar_file, std::string keep_indi_file, std::string remove_indi_file, std::string sex_file, int mphen, int mphen2, double grm_cutoff, double adj_grm_fac, int dosage_compen, bool m_grm_flag, bool pred_rand_eff, bool est_fix_eff, bool est_fix_eff_var, int reml_mtd, int MaxIter, std::vector<double> reml_priors, std::vector<double> reml_priors_var, std::vector<int> drop, bool no_lrt, double prevalence, double prevalence2, bool no_constrain, bool ignore_Ce, std::vector<double> &fixed_rg_val, bool bivar_no_constrain);

    // LD
    void read_LD_target_SNPs(std::string snplistfile);
    void LD_Blocks(int stp, double wind_size, double alpha, bool IncldQ = true, bool save_ram = false);
    void calcu_mean_rsq(int wind_size, double rsq_cutoff, bool dominance_flag);
    void calcu_mean_rsq_multiSet(std::string snpset_filenames_file, int wind_size, double rsq_cutoff, bool dominance_flag);
    void calcu_max_ld_rsq(int wind_size, double rsq_cutoff, bool dominance_flag);
    void ld_seg(std::string i_ld_file, int seg_size, int wind_size, double rsq_cutoff, bool dominance_flag);
    void set_ldscore_adj_flag(bool ldscore_adj);

    void genet_dst(std::string bfile, std::string hapmap_genet_map);

    void GWAS_simu(std::string bfile, int simu_num, std::string qtl_file, int case_num, int control_num, double hsq, double K, int seed, bool output_causal, bool simu_emb_flag, int eff_mod=0);
    //void simu_geno_unlinked(int N, int M, double maf);

    void set_diff_freq(double freq_diff);
    void run_massoc_slct(std::string metafile, int wind_size, double p_cutoff, double collinear, int64_t top_SNPs, bool joint_only, bool GC, double GC_val, bool actual_geno, int mld_slct_alg);
    void run_massoc_cond(std::string metafile, std::string snplistfile, int wind_size, double collinear, bool GC, double GC_val, bool actual_geno);
    void run_massoc_sblup(std::string metafile, int wind_size, double lambda);
    void set_massoc_pC_thresh(double thresh);

    void save_plink();
    void dose2bed();

    void read_IRG_fnames(std::string snp_info_file, std::string fname_file, double GC_cutoff);

    // population genetics
    void Fst(std::string filename);
    void paa(std::string aa_file);
    void ibc(bool ibc_all);

    // LD pruning
    void LD_pruning(double rsq_cutoff, int wind_size);
    //void make_grm_wt_mkl(string i_ld_file, int wind_m, double wt_ld_cut, bool grm_homogametic_flag, bool inbred, bool output_bin, int grm_mtd=0, int wt_mtd=0, bool mlmassoc=false, bool impData_flag=false, int ttl_snp_num=-1);

    void make_uni_id(std::vector<std::string> &uni_id, std::map<std::string, int> &uni_id_map);

    // mlma
    void mlma(std::string grm_file, bool m_grm_flag, std::string subtract_grm_file, std::string phen_file, std::string qcovar_file, std::string covar_file, int mphen, int MaxIter, std::vector<double> reml_priors, std::vector<double> reml_priors_var, bool no_constrain, bool within_family, bool inbred, bool no_adj_covar, std::string weight_file, std::string save_reml_file, std::string load_reml_file);
    void mlma_loco(std::string phen_file, std::string qcovar_file, std::string covar_file, int mphen, int MaxIter, std::vector<double> reml_priors, std::vector<double> reml_priors_var, bool no_constrain, bool inbred, bool no_adj_covar);
    // Memory-efficient LOCO: requires pre-built all-autosome GRM via --grm.
    // Peak RAM = O(2·n²) instead of O(n_chr·n²) for the legacy path above.
    void mlma_loco_v2(std::string grm_file, std::string grm_chr_prefix, std::string phen_file, std::string qcovar_file, std::string covar_file, int mphen, int MaxIter, std::vector<double> reml_priors, std::vector<double> reml_priors_var, bool no_constrain, bool inbred, bool no_adj_covar);
    void save_reml_state(const std::string& filename, bool no_adj_covar);
    void load_reml_state(const std::string& filename, bool no_adj_covar);

    // gene based association test
    void sbat_gene(std::string sAssoc_file, std::string gAnno_file, int sbat_wind, double sbat_ld_cutoff, bool sbat_write_snpset, bool GC, double GC_val);
    void sbat(std::string sAssoc_file, std::string snpset_file, double sbat_ld_cutoff, bool sbat_write_snpset,bool GC, double GC_val);
    void sbat_seg(std::string sAssoc_file, int seg_size, double sbat_ld_cutoff, bool sbat_write_snpset,bool GC, double GC_val);

    // gene based association
    //////////////////////////////
    void mbat_gene(std::string mbat_sAssoc_file, std::string mbat_gAnno_file, int mbat_wind, double mbat_svd_gamma, double sbat_ld_cutoff, bool mbat_write_snpset, bool GC, double GC_val,bool mbat_print_all_p);
    void svdDecomposition( Eigen::MatrixXf &X,double &prop, int &eigenvalueNum, Eigen::VectorXd &eigenvalueUsed,Eigen::MatrixXd &U_prop);
    void mbat_ACATO(double &mbat_svd_pvalue,double &fastbat_pvalue, double &mbat_pvalue);
    void mbat_calcu_lambda(std::vector<int> &snp_indx, Eigen::MatrixXf &rval, Eigen::VectorXd &eigenval, int &snp_count, double sbat_ld_cutoff, std::vector<int> &sub_indx);
    void mbat(std::string mbat_sAssoc_file, std::string snpset_file, double mbat_svd_gamma, double sbat_ld_cutoff, bool mbat_write_snpset, bool GC, double GC_val,bool mbat_print_all_p); 

    ////////////////////////////////
    // GSMR
    void read_gsmrfile(std::string expo_file_list, std::string outcome_file_list, double gwas_thresh, int nsnp_gsmr, int gsmr_so_alg);
    void gsmr(int gsmr_alg_flag, std::string ref_ld_dirt, std::string w_ld_dirt, double freq_thresh, double gwas_thresh, double clump_wind_size, double clump_r2_thresh, double std_heidi_thresh, double global_heidi_thresh, double ld_fdr_thresh, int nsnp_gsmr, bool o_snp_instru_flag, int gsmr_so_alg, int gsmr_beta_version);
    std::vector<std::vector<double>> forward_gsmr(std::stringstream &ss, std::map<std::string,int> &snp_instru_map, double gwas_thresh, double clump_wind_size, double clump_r2_thresh, double std_heidi_thresh, double global_heidi_thresh, double ld_fdr_thresh, int nsnp_gsmr, std::stringstream &ss_pleio);
    std::vector<std::vector<double>> reverse_gsmr(std::stringstream &ss, std::map<std::string,int> &snp_instru_map, double gwas_thresh, double clump_wind_size, double clump_r2_thresh, double std_heidi_thresh, double global_heidi_thresh, double ld_fdr_thresh, int nsnp_gsmr, std::stringstream &ss_pleio);
    eigenMatrix rho_sample_overlap(std::vector<std::vector<bool>> snp_val_flag, eigenMatrix snp_b, eigenMatrix snp_se, eigenMatrix snp_pval, eigenMatrix snp_n, int nexpo, int noutcome, std::vector<std::string> snp_name, std::vector<int> snp_remain, std::string ref_ld_dirt, std::string w_ld_dirt, std::vector<std::string> trait_name, int gsmr_so_alg);

    eigenMatrix sample_overlap_ldsc(std::vector<std::vector<bool>> snp_val_flag, eigenMatrix snp_b, eigenMatrix snp_se, eigenMatrix snp_n, int nexpo, int noutcome, std::vector<std::string> snp_name, std::vector<int> snp_remain, std::string ref_ld_dirt, std::string w_ld_dirt, std::vector<std::string> trait_name);
    eigenMatrix sample_overlap_rb(std::vector<std::vector<bool>> snp_val_flag, eigenMatrix snp_b, eigenMatrix snp_se, eigenMatrix snp_pval, eigenMatrix snp_n, int nexpo, int noutcome, std::vector<std::string> snp_name, std::vector<int> snp_remain, std::vector<std::string> trait_name);

    // mtCOJO
    void mtcojo(std::string mtcojo_bxy_file, std::string ref_ld_dirt, std::string w_ld_dirt, double freq_thresh, double gwas_thresh, int clump_wind_size, double clump_r2_thresh, double std_heidi_thresh, double global_heidi_thresh, double ld_fdr_thresh, int nsnp_gsmr, int gsmr_beta_version);
    bool mtcojo_ldsc(std::vector<std::vector<bool>> snp_val_flag, eigenMatrix snp_b, eigenMatrix snp_se, eigenMatrix snp_n, int ntrait, std::vector<std::string> snp_name, std::vector<int> snp_remain, std::string ref_ld_dirt, std::string w_ld_dirt, std::vector<std::string> trait_name, eigenMatrix &ldsc_intercept, eigenMatrix &ldsc_slope);
    int read_mtcojofile(std::string mtcojolist_file, double gwas_thresh, int nsnp_gsmr);
    double read_single_metafile_txt(std::string metafile, std::map<std::string, int> id_map, std::vector<std::string> &snp_a1, std::vector<std::string> &snp_a2, 
                                eigenVector &snp_freq, eigenVector &snp_b, eigenVector &snp_se, eigenVector &snp_pval, eigenVector &snp_n, std::vector<bool> &snpflag);
    double read_single_metafile_gz(std::string metafile, std::map<std::string, int> id_map, std::vector<std::string> &snp_a1, std::vector<std::string> &snp_a2, 
                                eigenVector &snp_freq, eigenVector &snp_b, eigenVector &snp_se, eigenVector &snp_pval, eigenVector &snp_n, std::vector<bool> &snpflag);
    std::vector<std::string> read_snp_metafile_txt(std::string metafile, std::map<std::string,int> &gws_snp_name_map, double thresh);
    std::vector<std::string> read_snp_metafile_gz(std::string metafile, std::map<std::string,int> &gws_snp_name_map, double thresh);
    
    // Adjusted for PC
    void pc_adjust(std::string pcadjust_list_file, std::string eigenvalue_file, double freq_thresh, int wind_size);
    void read_pc_adjust_file(std::string pcadjust_list_file, std::string pc_file);

    /////////////////////////
    // gene expresion data
    void read_efile(std::string efile);

    // ecojo
    void read_eR(std::string eR_file);
    void run_ecojo_slct(std::string e_metafile, double p_cutoff, double collinear);
    void run_ecojo_blup_efile(std::string e_metafile, double lambda);
    void run_ecojo_blup_eR(std::string e_metafile, double lambda);

    // ERM
    void make_erm(int erm_mtd, bool output_bin);

    // Woodbury low-rank REML
    void set_reml_woodbury_rank(int k) { _woodbury_rank = k; }
    void set_reml_woodbury_auto(double buf, int k_max) {
        _woodbury_rank = -1;
        _woodbury_buffer_factor = buf;
        _woodbury_k_max = k_max;
    }
    void set_reml_woodbury_nystrom() { _woodbury_nystrom = true; }

    // WoodburyMLMACache is defined in mlma_woodbury.hpp (k×n Uk layout).
    using WoodburyMLMACache = ::WoodburyMLMACache;
    WoodburyMLMACache build_woodbury_mlma_cache() const;

private:
    void init_keep();
    void init_include();
    void get_rsnp(std::vector<int> &rsnp);
    void get_rindi(std::vector<int> &rindi);

    void save_famfile();
    void save_bimfile();
    void save_bedfile();

    void update_bim(std::vector<int> &rsnp);
    void update_fam(std::vector<int> &rindi);
    void compact_indi_data();
    void compact_snp_data();
    void compact_dosage_data();

    void update_include(std::vector<int> chr_buf, std::vector<std::string> snpid_buf, std::vector<double> gd_buf, std::vector<int> bp_buf, std::vector<std::string> a1_buf, std::vector<std::string> a2_buf, int file_indx);
    void update_keep(std::vector<std::string> fid_buf, std::vector<std::string> pid_buf, std::vector<std::string> fa_id_buf, std::vector<std::string> mo_id_buf, std::vector<int> sex_buf, std::vector<double> pheno_buf, std::string famfile);
    
    void update_id_map_kp(const std::vector<std::string> &id_list, std::map<std::string, int> &id_map, std::vector<int> &keep);
    void update_id_map_rm(const std::vector<std::string> &id_list, std::map<std::string, int> &id_map, std::vector<int> &keep);
    void read_snplist(std::string snplistfile, std::vector<std::string> &snplist, std::string msg = "SNPs");
    void read_indi_list(std::string indi_list_file, std::vector<std::string> &indi_list);

    bool make_XMat(Eigen::MatrixXf &X);
    bool make_XMat_d(Eigen::MatrixXf &X);
    //void make_XMat_SNPs(vector< vector<float> > &X, bool miss_with_mu);
    void std_XMat(Eigen::MatrixXf &X, eigenVector &sd_SNP, bool grm_homogametic_flag, bool miss_with_mu, bool divid_by_std);
    void std_XMat_subpopu(std::string subpopu_file, Eigen::MatrixXf &X, eigenVector &sd_SNP, bool grm_homogametic_flag, bool miss_with_mu, bool divid_by_std);
    void std_XMat_d(Eigen::MatrixXf &X, eigenVector &sd_SNP, bool miss_with_mu, bool divid_by_std);
    //void std_XMat(vector< vector<float> > &X, vector<double> &sd_SNP, bool grm_homogametic_flag, bool divid_by_std = true);
    void makex_eigenVector(int j, eigenVector &x, bool resize = true, bool minus_2p = false);
    void makex_eigenVector_std(int j, eigenVector &x, bool resize = true, double snp_std = 1.0);
    //void make_XMat_eigenMatrix(MatrixXf &X);
    bool make_XMat_subset(Eigen::MatrixXf &X, std::vector<int> &snp_indx, bool divid_by_std);
    bool make_XMat_d_subset(Eigen::MatrixXf &X, std::vector<int> &snp_indx, bool divid_by_std);
    bool make_XMat_subset_dense(Eigen::MatrixXf &X, std::vector<int> &snp_indx, bool divid_by_std);
    bool make_XMat_d_subset_dense(Eigen::MatrixXf &X, std::vector<int> &snp_indx, bool divid_by_std);


    void calcu_mu(bool ssq_flag = false);
    void calcu_maf();
    void mu_func(int j, std::vector<double> &fac);
    void check_autosome();
    void check_homogametic_chromosomes();
    void check_sex();

    // grm
    void calcu_grm_var(double &diag_m, double &diag_v, double &off_m, double &off_v);
    int read_grm_id(std::string grm_file, std::vector<std::string> &grm_id, bool out_id_log, bool read_id_only);
    void read_grm(std::string grm_file, std::vector<std::string> &grm_id, bool out_id_log = true, bool read_id_only = false, bool dont_read_N = false);
    void read_grm_gz(std::string grm_file, std::vector<std::string> &grm_id, bool out_id_log = true, bool read_id_only = false);
    void read_grm_bin(std::string grm_file, std::vector<std::string> &grm_id, bool out_id_log = true, bool read_id_only = false, bool dont_read_N = false);
    void read_grm_filenames(std::string merge_grm_file, std::vector<std::string> &grm_files, bool out_log = true);
    void merge_grm(std::string merge_grm_file);
    void rm_cor_indi(double grm_cutoff);
    void adj_grm(double adj_grm_fac);
    void dc(int dosage_compen);
    void manipulate_grm(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, std::string sex_file, double grm_cutoff, double adj_grm_fac, int dosage_compen, bool merge_grm_flag, bool dont_read_N = false);
    void output_grm_vec(std::vector< std::vector<float> > &A, std::vector< std::vector<int> > &A_N, bool output_grm_bin);
    void output_grm(bool output_grm_bin);

    // reml
    void read_phen(std::string phen_file, std::vector<std::string> &phen_ID, std::vector< std::vector<std::string> > &phen_buf, int mphen, int mphen2 = 0);
    int read_fac(std::ifstream &ifstrm, std::vector<std::string> &ID, std::vector< std::vector<std::string> > &fac);
    int read_covar(std::string covar_file, std::vector<std::string> &covar_ID, std::vector< std::vector<std::string> > &covar, bool qcovar_flag);
    int read_GE(std::string GE_file, std::vector<std::string> &GE_ID, std::vector< std::vector<std::string> > &GE, bool qGE_flag = false);
    bool check_case_control(double &ncase, eigenVector &y);
    double transform_hsq_L(double P, double K, double hsq);
    int constrain_varcmp(eigenVector &varcmp);
    void drop_comp(std::vector<int> &drop);
    void construct_X(int n, std::map<std::string, int> &uni_id_map, bool qcovar_flag, int qcovar_num, std::vector<std::string> &qcovar_ID, std::vector< std::vector<std::string> > &qcovar, bool covar_flag, int covar_num, std::vector<std::string> &covar_ID, std::vector< std::vector<std::string> > &covar, std::vector<eigenMatrix> &E_float, eigenMatrix &qE_float);
    void coeff_mat(const std::vector<std::string> &vec, eigenMatrix &coeff_mat, std::string errmsg1, std::string errmsg2);
    void reml(bool pred_rand_eff, bool est_fix_eff, bool est_fix_eff_var, std::vector<double> &reml_priors, std::vector<double> &reml_priors_var, double prevalence, double prevalence2, bool no_constrain, bool no_lrt, bool mlmassoc = false);
    double reml_iteration(eigenMatrix &Vi_X, eigenMatrix &Xt_Vi_X_i, eigenMatrix &Hi, eigenVector &Py, eigenVector &varcmp, bool prior_var_flag, bool no_constrain, bool reml_bivar_fix_rg = false);
    void init_varcomp(std::vector<double> &reml_priors_var, std::vector<double> &reml_priors, eigenVector &varcmp);
    bool calcu_Vi(eigenMatrix &Vi, eigenVector &prev_varcmp, double &logdet, int &iter, bool factorize_only = false);
    bool inverse_H(eigenMatrix &H);
    void bend_V(eigenMatrix &Vi);
    void bend_A();
    bool bending_eigenval(eigenVector &eval);
    double calcu_P(eigenMatrix &Vi, eigenMatrix &Vi_X, eigenMatrix &Xt_Vi_X_i, eigenMatrix &P);
    double calcu_P_nomatrix(eigenMatrix &Vi, eigenMatrix &Vi_X, eigenMatrix &Xt_Vi_X_i);
    double calcu_P_impl(eigenMatrix &Vi, eigenMatrix &Vi_X, eigenMatrix &Xt_Vi_X_i, eigenMatrix *P);
    void calcu_Hi(eigenMatrix &P, eigenMatrix &Hi);
    void reml_equation(eigenMatrix &P, eigenMatrix &Hi, eigenVector &Py, eigenVector &varcmp);
    double lgL_reduce_mdl(bool no_constrain);
    void em_reml(eigenMatrix &P, eigenVector &Py, eigenVector &prev_varcmp, eigenVector &varcmp, double dlogL = 1000.0);
    void ai_reml(eigenMatrix &P, eigenMatrix &Hi, eigenVector &Py, eigenVector &prev_varcmp, eigenVector &varcmp, double dlogL);
    void calcu_tr_PA(eigenMatrix &P, eigenVector &tr_PA);
    eigenVector applyP_vec(const eigenVector &v) const;
    eigenMatrix applyP_mat(const eigenMatrix &Z) const;  // batch DTRSM/DSYMM version
    void calcu_tr_PA_hutchpp(eigenVector &tr_PA, int m_probes);
    void calcu_tr_PA_woodbury(eigenVector &tr_PA); // exact O(nkc) trace for woodbury path
    // Woodbury helpers
    void compute_woodbury_basis(int k, double buffer_factor, int k_max);
    eigenVector woodbury_Kv(const eigenVector &v) const;
    eigenVector woodbury_Viv(const eigenVector &v) const;
    eigenMatrix woodbury_ViZ(const eigenMatrix &Z) const;
    void calcu_Vp(double &Vp, double &Vp2, double &VarVp, double &VarVp2, const eigenVector &varcmp, const eigenMatrix &Hi);
    void calcu_hsq(int i, double Vp, double Vp2, double VarVp, double VarVp2, double &hsq, double &var_hsq, const eigenVector &varcmp, const eigenMatrix &Hi);
    void calcu_sum_hsq(double Vp, double VarVp, double &sum_hsq, double &var_sum_hsq, const eigenVector &varcmp, const eigenMatrix &Hi);
    void output_blup_snp(eigenMatrix &b_SNP);

    void read_weight(std::string phen_file, std::vector<std::string> &phen_ID, std::vector<double> &weights);
    
    // within-family reml analysis
    void detect_family();
    bool calcu_Vi_within_family(eigenMatrix &Vi, eigenVector &prev_varcmp, double &logdet, int &iter);

    // bivariate REML analysis
    void calcu_rg(eigenVector &varcmp, eigenMatrix &Hi, eigenVector &rg, eigenVector &rg_var, std::vector<std::string> &rg_name);
    void update_A(eigenVector &prev_varcmp);
    void constrain_rg(eigenVector &varcmp);
    double lgL_fix_rg(eigenVector &prev_varcmp, bool no_constrain);
    bool calcu_Vi_bivar(eigenMatrix &Vi, eigenVector &prev_varcmp, double &logdet, int &iter);

    // GWAS simulation
    void kosambi();
    int read_QTL_file(std::string qtl_file, std::vector<std::string> &qtl_name, std::vector<int> &qtl_pos, std::vector<double> &qtl_eff, std::vector<int> &have_eff);
    void output_simu_par(std::vector<std::string> &qtl_name, std::vector<int> &qtl_pos, std::vector<double> &qtl_eff, double Vp);
    void save_phenfile(std::vector< std::vector<double> > &y);
    // not usefully any more
    void GenerCases(std::string bfile, std::string qtl_file, int case_num, int control_num, double hsq, double K, bool curr_popu = false, double gnrt = 100);

    // LD
    void EstLD(std::vector<int> &smpl, double wind_size, std::vector< std::vector<std::string> > &snp, std::vector< std::vector<double> > &r, std::vector<double> &r2, std::vector<double> &md_r2, std::vector<double> &max_r2, std::vector<std::string> &max_r2_snp, std::vector<double> &dL, std::vector<double> &dR, std::vector<int> &K, std::vector<std::string> &L_SNP, std::vector<std::string> &R_SNP, double alpha, bool IncldQ);
    eigenMatrix reg(std::vector<double> &y, std::vector<double> &x, std::vector<double> &rst, bool table = false);
    void rm_cor_snp(int m, int start, float *rsq, double rsq_cutoff, std::vector<int> &rm_snp_ID1);
    void get_ld_blk_pnt(std::vector<int> &brk_pnt1, std::vector<int> &brk_pnt2, std::vector<int> &brk_pnt3, int wind_bp, int wind_snp = 0);
    void get_ld_blk_pnt_max_limit(std::vector<int> &brk_pnt1, std::vector<int> &brk_pnt2, std::vector<int> &brk_pnt3, int wind_bp, int wind_snp);
    void get_lds_brkpnt(std::vector<int> &brk_pnt1, std::vector<int> &brk_pnt2, int ldwt_seg, int wind_snp_num=0);
    void calcu_ld_blk(std::vector<int> &brk_pnt, std::vector<int> &brk_pnt3, eigenVector &mean_rsq, eigenVector &snp_num, eigenVector &max_rsq, bool second, double rsq_cutoff, bool dominance_flag = false);
    void calcu_ld_blk_split(int size, int size_limit, Eigen::MatrixXf &X_sub, eigenVector &ssx_sqrt_i_sub, double rsq_cutoff, eigenVector &rsq_size, eigenVector &mean_rsq_sub, eigenVector &max_rsq_sub, int s1, int s2, bool second);
    void calcu_ssx_sqrt_i(eigenVector &ssx_sqrt_i);
    void calcu_max_ld_rsq_blk(eigenVector &multi_rsq, eigenVector &multi_rsq_adj, eigenVector &max_rsq, std::vector<int> &max_pos, std::vector<int> &brk_pnt, double rsq_cutoff, bool dominance_flag);
    bool bending_eigenval_Xf(Eigen::VectorXf &eval);
    void calcu_ld_blk_multiSet(std::vector<int> &brk_pnt, std::vector<int> &brk_pnt3, std::vector< std::vector<bool> > &set_flag, std::vector<eigenVector> &mean_rsq, std::vector<eigenVector> &snp_num, std::vector<eigenVector> &max_rsq, bool second, double rsq_cutoff, bool dominance_flag);
    void calcu_ld_blk_split_multiSet(int size, int size_limit, Eigen::MatrixXf &X_sub, Eigen::MatrixXf &X_sub2, std::vector<int> &used_in_this_set, eigenVector &ssx_sqrt_i_sub, double rsq_cutoff, eigenVector &rsq_size, eigenVector &mean_rsq_sub, eigenVector &max_rsq_sub, int s1, int s2, bool second);

    // Joint analysis of GWAS MA results
    void read_metafile(std::string metafile, bool GC, double GC_val);
    void init_massoc(std::string metafile, bool GC, double GC_val);
    void read_fixed_snp(std::string snplistfile, std::string msg, std::vector<int> &pgiven, std::vector<int> &remain);
    std::vector<double> eigenVector2Vector(const eigenVector &x);
    //double crossprod(int indx1, int indx2);
    void get_x_vec(float *x, int indx);
    void get_x_mat(float *x, std::vector<int> &indx);
    void vec_t_mat(float *vec, int nrow, float *mat, int ncol, eigenVector &out); // 1 x n vector multiplied by m x n matrix
    void stepwise_slct(std::vector<int> &slct, std::vector<int> &remain, eigenVector &bC, eigenVector &bC_se, eigenVector &pC, int mld_slct_alg, uint64_t top_SNPs);
    bool slct_entry(std::vector<int> &slct, std::vector<int> &remain, eigenVector &bC, eigenVector &bC_se, eigenVector &pC);
    void slct_stay(std::vector<int> &slct, eigenVector &bJ, eigenVector &bJ_se, eigenVector &pJ);
    double massoc_calcu_Ve(const std::vector<int> &slct, eigenVector &bJ, eigenVector &b);
    void massoc_cond(const std::vector<int> &slct, const std::vector<int> &remain, eigenVector &bC, eigenVector &bC_se, eigenVector &pC);
    void massoc_joint(const std::vector<int> &indx, eigenVector &bJ, eigenVector &bJ_se, eigenVector &pJ);
    bool init_B(const std::vector<int> &indx);
    void init_Z(const std::vector<int> &indx);
    bool insert_B_and_Z(const std::vector<int> &indx, int insert_indx);
    void erase_B_and_Z(const std::vector<int> &indx, int erase_indx);
    void LD_rval(const std::vector<int> &indx, eigenMatrix &rval);
    bool massoc_sblup(double lambda, eigenVector &bJ);
    void massoc_slct_output(bool joint_only, std::vector<int> &slct, eigenVector &bJ, eigenVector &bJ_se, eigenVector &pJ, eigenMatrix &rval);
    void massoc_cond_output(std::vector<int> &remain, eigenVector &bC, eigenVector &bC_se, eigenVector &pC);

    // read raw genotype data (Illumina)
    char flip_allele(char a);
    void read_one_IRG(std::ofstream &oped, int ind, std::string IRG_fname, double GC_cutoff);


    bool comput_inverse_logdet_LDLT(eigenMatrix &Vi, double &logdet);
    bool comput_inverse_logdet_LU(eigenMatrix &Vi, double &logdet);
    bool comput_inverse_logdet_LU_array(int n, float *Vi, double &logdet);
    void LD_pruning_blk(const Eigen::MatrixXf &X, std::vector<int> &brk_pnt, double rsq_cutoff, std::vector<int> &rm_snp_ID1);
    

    // mlma
    //struct MlmaResult { std::vector<float> beta, se, pval; };
    struct MlmaResult { std::vector<float> beta, se; std::vector<double> pval; };
    MlmaResult mlma_calcu_stat(std::span<const float> y, unsigned long m);
    MlmaResult mlma_calcu_stat_covar(std::span<const float> y, unsigned long m);
    void grm_minus_grm(float *grm, float *sub_grm);

    // population
    void read_subpopu(std::string filename, std::vector<std::string> &subpopu, std::vector<std::string> &subpopu_name);

    /*
    // weighting GRM: ldwt_wind = window size for mean LD calculation; ld_seg = block size;
    void calcu_ld_blk_ldwt(eigenVector &ssx_sqrt_i, vector<int> &brk_pnt, vector<int> &brk_pnt3, eigenVector &mean_rsq, eigenVector &snp_num, eigenVector &max_rsq, bool second, double rsq_cutoff, bool adj);
    void calcu_ld_blk_split_ldwt(int size, int size_limit, int s_pnt, eigenVector &ssx_sqrt_i_sub, double rsq_cutoff, eigenVector &rsq_size, eigenVector &mean_rsq_sub, eigenVector &max_rsq_sub, int s1, int s2, bool second, bool adj);
    void calcu_lds(string i_ld_file, eigenVector &wt, int ldwt_wind, int ldwt_seg, double rsq_cutoff);
    void calcu_ldak(eigenVector &wt, int ldwt_seg, double rsq_cutoff);
    void calcu_ldak_blk(eigenVector &wt, eigenVector &sum_rsq, eigenVector &ssx_sqrt_i, vector<int> &brk_pnt, bool second, double rsq_cutoff);
    void calcu_ldwt(string i_ld_file, eigenVector &wt, int wind_size, double rsq_cutoff);
    void read_mrsq_mb(string i_ld_file, vector<float> &seq, vector<double> &mrsq_mb, eigenVector &wt, eigenVector &snp_m);
    void adj_wt_4_maf(eigenVector &wt);
    void cal_sum_rsq_mb(eigenVector &sum_rsq_mb);
    void col_std(Eigen::MatrixXf &X);
    void assign_snp_2_mb(vector<float> &seq, vector< vector<int> > &maf_bin_pos, int mb_num);
    void make_grm_pca_blk(vector<int> & maf_bin_pos_i, int ldwt_seg, double &trace);
    void glpk_simplex_solver(Eigen::MatrixXf &rsq, eigenVector &mrsq, eigenVector &wt, int maxiter);
    void calcu_grm_wt_mkl(string i_ld_file, float *X, vector<double> &sd_SNP, eigenVector &wt, int wind_size, double rsq_cutoff, int wt_mtd, int ttl_snp_num);
    */

    // gene based association test
    void sbat_read_snpAssoc(std::string snpAssoc_file, std::vector<std::string> &snp_name, std::vector<int> &snp_chr, std::vector<int> &snp_bp, std::vector<double> &snp_pval);
    void sbat_read_geneAnno(std::string gAnno_file, std::vector<std::string> &gene_name, std::vector<int> &gene_chr, std::vector<int> &gene_bp1, std::vector<int> &gene_bp2);
    void sbat_read_snpset(std::string snpset_file, std::vector<std::string> &set_name, std::vector< std::vector<std::string> > &snpset);
    void sbat_calcu_lambda(std::vector<int> &snp_indx, Eigen::VectorXd &eigenval, int &snp_count, double sbat_ld_cutoff, std::vector<int> &sub_indx);
    void get_sbat_seg_blk(int seg_size, std::vector< std::vector<int> > &snp_set_indx, std::vector<int> &set_chr, std::vector<int> &set_start_bp, std::vector<int> &set_end_bp);
    void rm_cor_sbat(Eigen::MatrixXf &R, double R_cutoff, int m, std::vector<int> &rm_ID1);

	// gene based association test 
	void gbat_read_snpAssoc(std::string snpAssoc_file, std::vector<std::string>& snp_name, std::vector<int>& snp_chr, std::vector<int>& snp_bp, std::vector<double>& snp_pval);
	void gbat_read_geneAnno(std::string gAnno_file, std::vector<std::string>& gene_name, std::vector<int>& gene_chr, std::vector<int>& gene_bp1, std::vector<int>& gene_bp2);
	void gbat_calcu_ld(Eigen::MatrixXf & X, eigenVector & sumsq_x, int snp1_indx, int snp2_indx, Eigen::MatrixXf & C);
	void gbat(std::string sAssoc_file, std::string gAnno_file, int wind, int simu_num);
	double gbat_simu_p(int & seed, int size, eigenMatrix & L, int simu_num, double chisq_o);


    //////////////////////
    // gene expresion data
    void init_e_include();
    void std_probe(std::vector< std::vector<bool> > &X_bool, bool divid_by_std);
    void std_probe_ind(std::vector< std::vector<bool> > &X_bool, bool divid_by_std);
    
    // ecojo
    void calcu_eR();
    void read_e_metafile(std::string e_metafile);
    void ecojo_slct_output(bool joint_only, std::vector<int> &slct, eigenVector &bJ, eigenVector &bJ_se, eigenVector &pJ);
    void ecojo_slct(std::vector<int> &slct, std::vector<int> &remain, eigenVector &bC, eigenVector &bC_se, eigenVector &pC);
    bool ecojo_slct_entry(std::vector<int> &slct, std::vector<int> &remain, eigenVector &bC, eigenVector &bC_se, eigenVector &pC);
    void ecojo_slct_stay(std::vector<int> &slct, eigenVector &bJ, eigenVector &bJ_se, eigenVector &pJ);
    void ecojo_joint(const std::vector<int> &slct, eigenVector &bJ, eigenVector &bJ_se, eigenVector &pJ);
    void ecojo_cond(const std::vector<int> &slct, const std::vector<int> &remain, eigenVector &bC, eigenVector &bC_se, eigenVector &pC);
    bool ecojo_init_R(const std::vector<int> &slct);
    void ecojo_init_RC(const std::vector<int> &slct, const std::vector<int> &remain);
    bool ecojo_insert_R(const std::vector<int> &slct, int insert_indx);
    void ecojo_erase_R(const std::vector<int> &slct);
    void ecojo_inv_R();
    void ecojo_blup(double lambda);   

    // mtCOJO and GSMR
    void init_meta_snp_map(std::vector<std::string> snplist, std::map<std::string, int> &snp_name_map, std::vector<std::string> &snp_name, std::vector<int> &remain_snp);
    void init_gwas_variable(std::vector<std::vector<std::string>> &snp_a1, std::vector<std::vector<std::string>> &snp_a2, eigenMatrix &snp_freq, eigenMatrix &snp_b, eigenMatrix &snp_se, eigenMatrix &snp_pval, eigenMatrix &n, int npheno, int nsnp);
    void update_meta_snp_list(std::vector<std::string> &snplist, std::map<std::string, int> snp_id_map);
    void update_meta_snp_map(std::vector<std::string> snplist, std::map<std::string, int> &snp_id_map, std::vector<std::string> &snp_id, std::vector<int> &snp_indx, bool indx_flag);
    void update_meta_snp(std::map<std::string,int> &snp_name_map, std::vector<std::string> &snp_name, std::vector<int> &snp_remain);
    std::vector<std::string> remove_bad_snps(std::vector<std::string> snp_name, std::vector<int> snp_remain, std::vector<std::vector<bool>> snp_flag, std::vector<std::vector<std::string>> &snp_a1, std::vector<std::vector<std::string>> &snp_a2, eigenMatrix &snp_freq,  eigenMatrix &snp_b, eigenMatrix snp_se, eigenMatrix snp_pval, eigenMatrix snp_n, std::map<std::string,int> plink_snp_name_map, std::vector<std::string> snp_ref_a1, std::vector<std::string> snp_ref_a2, int ntarget, int ncovar, std::string outfile_name);
    std::vector<std::string> remove_freq_diff_snps(std::vector<std::string> meta_snp_name, std::vector<int> meta_snp_remain, std::map<std::string,int> snp_name_map, std::vector<double> ref_freq, eigenMatrix meta_freq, std::vector<std::vector<bool>> snp_flag, int ntrait, double freq_thresh, std::string outfile_name);
    std::vector<std::string> remove_mono_snps(std::map<std::string,int> snp_name_map, std::vector<double> ref_snpfreq, std::string outfile_name);
    std::vector<std::string> filter_meta_snp_pval(std::vector<std::string> snp_name, std::vector<int> remain_snp_indx,  eigenMatrix snp_pval, int start_indx, int end_indx, std::vector<std::vector<bool>> snp_flag, double pval_thresh);
    std::vector<double> gsmr_meta(std::vector<std::string> &snp_instru, eigenVector bzx, eigenVector bzx_se, eigenVector bzx_pval, eigenVector bzy, eigenVector bzy_se, eigenVector bzy_pval, double rho_pheno, std::vector<bool> snp_flag, double gwas_thresh, int wind_size, double r2_thresh, double std_heidi_thresh, double global_heidi_thresh, double ld_fdr_thresh, int nsnp_gsmr, std::string &pleio_snps, std::string &err_msg);
    std::vector<std::string> clumping_meta(eigenVector snp_chival, std::vector<bool> snp_flag, double pval_thresh, int wind_size, double r2_thresh);
    void update_mtcojo_snp_rm(std::vector<std::string> adjsnps, std::map<std::string,int> &snp_id_map, std::vector<int> &remain_snp_indx);
    std::vector<std::string> read_snp_ldsc(std::map<std::string,int> ldsc_snp_name_map, std::vector<std::string> snp_name, std::vector<int> snp_remain, int &ttl_mk_num, std::string ref_ld_dirt, std::string w_ld_dirt, std::vector<double> &ref_ld_vec, std::vector<double> &w_ld_vec);
    void reorder_snp_effect(std::vector<int> snp_remain, eigenMatrix &bhat_z, eigenMatrix &bhat_n, eigenMatrix snp_b, eigenMatrix snp_se, eigenMatrix snp_n, std::vector<std::vector<bool>> &snp_flag, std::vector<std::vector<bool>> snp_val_flag, std::vector<int> &nsnp_cm_trait, std::vector<std::string> cm_ld_snps, std::map<std::string,int> ldsc_snp_name_map, eigenVector &ref_ld, eigenVector &w_ld, std::vector<double> ref_ld_vec, std::vector<double> w_ld_vec, int ntrait);
    eigenMatrix ldsc_snp_h2(eigenMatrix bhat_z, eigenMatrix bhat_n, eigenVector ref_ld, eigenVector w_ld, std::vector<std::vector<bool>> snp_flag, std::vector<int> nsnp_cm_trait, int n_cm_ld_snps, int ttl_mk_num, std::vector<std::string> trait_name, int ntrait);
    eigenMatrix ldsc_snp_rg(eigenMatrix ldsc_var_h2, eigenMatrix bhat_z, eigenMatrix bhat_n, eigenVector ref_ld, eigenVector w_ld, std::vector<std::vector<bool>> snp_flag, std::vector<int> trait_indx1, std::vector<int> trait_indx2, int n_cm_ld_snps, int ttl_mk_num, std::vector<std::string> trait_name);

    // Ajust summarydata for PC
    void adjust_snp_effect_for_pc(eigenVector &bzy_adj, eigenVector &bzx_hat, eigenVector bzy, eigenVector bxy_hat, int wind_size);

    // inline functions
    template<typename ElemType>
    void makex(int j, std::vector<ElemType> &x, bool minus_2p = false) {
        int i = 0;
        x.resize(_keep.size());
        for (i = 0; i < _keep.size(); i++) {
            if (!_snp_1[_include[j]][_keep[i]] || _snp_2[_include[j]][_keep[i]]) {
                x[i] = (_snp_1[_include[j]][_keep[i]] + _snp_2[_include[j]][_keep[i]]);
            } else x[i] = _mu[_include[j]];
            if (minus_2p) x[i] -= _mu[_include[j]];
        }
    }
    
private:
    bool autosomeBoundSpecified() const { return _autosome_num > 0; }
    bool isAutosomalChr(int chr) const { return chr > 0 && (!autosomeBoundSpecified() || chr <= _autosome_num); }

    // read in plink files
    // bim file
    int _autosome_num;
    std::vector<int> _chr;
    std::vector<std::string> _snp_name;
    std::map<std::string, int> _snp_name_map;
    std::map<std::string, std::string> _snp_name_per_chr;
    std::vector<double> _genet_dst;
    std::vector<int> _bp;
    std::vector<std::string> _allele_ref;
    std::vector<std::string> _allele_alt;
    int _snp_num;
    std::vector<double> _rc_rate;
    std::vector<int> _include; // initialized in the read_bimfile()
    eigenVector _maf;

    // fam file
    std::vector<std::string> _fid;
    std::vector<std::string> _pid;
    std::map<std::string, int> _id_map;
    std::vector<std::string> _fa_id;
    std::vector<std::string> _mo_id;
    std::vector<int> _sex;
    std::vector<double> _pheno;
    int _indi_num;
    std::vector<int> _keep; // initialized in the read_famfile()
    eigenMatrix _varcmp_Py; // BLUP solution to the total genetic effects of individuals

    // bfile path for lazy loading
    string _bfile;

    // bed file
    std::vector< std::vector<bool> > _snp_1;
    std::vector< std::vector<bool> > _snp_2;

    // imputed data
    bool _dosage_flag;
    // When true, make_XMat/make_XMat_subset skip compact_snp_data() so that
    // _snp_1/_snp_2 are preserved across multiple different extract_chr() calls
    // (e.g. the LOCO per-chromosome loop in mlma_loco_v2).
    bool _make_XMat_no_compact = false;
    // Legacy path: selected homogametic chromosome ID for sex-aware scaling.
    // 0 means not set (treat all markers as autosomal in calcu_mu).
    int _legacy_homogametic_chr = 0;
    GeneticModel _genetic_model;
    Eigen::MatrixXf _geno_dose;  // (n_indi, n_snp), column-major; dense indices after compaction
    std::vector<double> _impRsq;

    // genotypes
    Eigen::MatrixXf _geno;

    // QC
    double _rm_ld_cutoff;

    // grm 
    Eigen::MatrixXf _grm_N;
    eigenMatrix _grm;
    bool _grm_bin_flag;

    // reml
    int _n;
    int _X_c;
    std::vector<int> _r_indx;
    std::vector<int> _r_indx_drop;
    int _reml_max_iter;
    int _reml_mtd;
    int _reml_inv_mtd;
    double _reml_diag_mul;
    int _reml_diagV_adj;
    bool _cv_blup;
    eigenMatrix _X;
    std::vector<eigenMatrix> _A;
    eigenVector _y;
    eigenMatrix _Vi;
    Eigen::LLT<eigenMatrix> _Vi_llt;  // unused — kept for ABI compat; remove in next major version
    eigenMatrix _Vi_L;                 // Cholesky factor L (lower tri) from in-place dpotrf; replaces _Vi_llt
    bool _Vi_use_llt = false;          // true when _Vi_L is valid and _Vi is empty
    // Woodbury low-rank REML: K ≈ U_k D_k U_k^T + λ_tail(I − U_k U_k^T)
    bool        _Vi_use_woodbury = false;  // true when woodbury basis is active
    int         _woodbury_rank   = 0;      // >0 fixed k; -1 auto-k; 0 disabled
    double      _woodbury_buffer_factor = 0.0;  // MP buffer for auto-k (default 1.5)
    int         _woodbury_k_max  = 0;           // SVD rank cap for auto-k (0 → min(n−1,2000))
    bool        _woodbury_nystrom = false;       // true → single-pass Nyström instead of power-iter
    eigenMatrix _Uk;              // n×k leading eigenvectors of K
    eigenVector _dk;              // k eigenvalues (clamped ≥ 0)
    double      _lambda_tail = 0.0;  // average bulk eigenvalue
    double      _tail_d_var  = 0.0;  // var of tail eigenvalues: Σ(d_j−λ_tail)²/(n−k)
    double      _sigma2_eff  = 0.0;  // σ²_g·λ_tail + σ²_e (updated per iteration)
    double      _sg2         = 0.0;  // σ²_g (genetic variance component, cached per calcu_Vi)
    eigenVector _ck;              // Woodbury correction: c_j = σ²_g(d_j−λ)/(σ²_eff+σ²_g(d_j−λ))
    eigenMatrix _Vi_X;        // cached V^{-1} X for implicit P matvecs
    eigenMatrix _Xt_Vi_X_i;   // cached (X' V^{-1} X)^{-1} for implicit P matvecs
    eigenMatrix _Uk_Vi_X;     // cached U_k^T V^{-1} X  (k×c), set in calcu_P_impl
    eigenMatrix _UkTX;        // cached U_k^T X  (k×c), constant across REML iterations
    eigenVector _UkTy;        // cached U_k^T y  (k),   constant across REML iterations
    eigenMatrix _P;
    bool _reml_trace_approx = false;
    int  _reml_trace_approx_nprobes = 90;  // ~1% relative trace error
    int  _reml_trace_power_iter = 0;       // power iterations for range sketch; 0=off (no benefit for bulk-spectrum GRMs)
    eigenMatrix _hutchpp_S;   // cached Rademacher probes (n × k) for Hutch++ range sketch
    eigenMatrix _hutchpp_G;   // cached Rademacher probes (n × k) for Hutch++ residual
    eigenVector _b;
    std::vector<std::string> _var_name;
    std::vector<double> _varcmp;
    std::vector<std::string> _hsq_name;
    double _y_Ssq;
    double _ncase;
    bool _flag_CC;
    bool _flag_CC2;
    bool _reml_diag_one;
    bool _reml_have_bend_A;
    int _V_inv_mtd;
    bool _reml_force_inv;
    bool _reml_AI_not_invertible;
    bool _reml_force_converge;
    bool _reml_no_converge;
    bool _reml_fixed_var;
    bool _reml_allow_constrain_run = false;
    bool _log_pval;

    // within-family reml analysis
    bool _within_family;
    std::vector<int> _fam_brk_pnt;

    // bivariate reml
    bool _bivar_reml;
    bool _ignore_Ce;
    bool _bivar_no_constrain;
    double _y2_Ssq;
    double _ncase2;
    std::vector< std::vector<int> > _bivar_pos;
    std::vector< std::vector<int> > _bivar_pos_prev;
    std::vector< eigenSparseMat > _Asp;
    std::vector< eigenSparseMat > _Asp_prev;
    std::vector<eigenMatrix> _A_prev;
    std::vector<double> _fixed_rg_val;

    std::vector<double> _mu;
    // Real (additive-scale) allele frequency mean, captured right after sample
    // compaction in read_bed_dosage() but before the nonadditive recoding is
    // applied to _geno_dose. Unlike _mu (which calcu_mu() may later recompute
    // directly from _geno_dose and would then reflect the recoded — not real —
    // values under the nonadditive model), _additive_mu always mirrors the true
    // allele frequency and should be used for reporting/output purposes.
    std::vector<double> _additive_mu;
    std::string _out;
    bool _save_ram;

    // LD
    std::vector<std::string> _ld_target_snp;
    bool _ldscore_adj;

    // joint analysis of META
    bool _jma_actual_geno;
    int _jma_wind_size;
    double _jma_p_cutoff;
    double _jma_collinear;
    double _jma_Vp;
    double _jma_Ve;
    double _GC_val;
    int _jma_snpnum_backward;
    int _jma_snpnum_collienar;
    eigenVector _freq;
    eigenVector _beta;
    eigenVector _beta_se;
    eigenVector _chisq;
    eigenVector _pval;
    eigenVector _N_o;
    eigenVector _Nd;
    eigenVector _MSX;
    eigenVector _MSX_B;
    eigenSparseMat _B_N;
    eigenSparseMat _B;
    eigenSparseMat _B_N_i;
    eigenSparseMat _B_i;
    eigenVector _D_N;
    eigenSparseMat _Z_N;
    eigenSparseMat _Z;
    double g_massoc_out_thresh = -1.0;
    double _diff_freq = 0.2;
    
    // GSMR analysis
    int _expo_num;
    int _outcome_num;
    int _n_gsmr_rst_item = 5;
    int _gsmr_beta_version = 0;
    eigenMatrix _r_pheno_sample;
    std::vector<std::string> _gwas_trait_name;
    std::vector<std::vector<bool>> _snp_val_flag;
    
    // mtCOJO analysis
    std::string _target_pheno_name="";
    std::vector<std::string> _meta_snp_name;
    std::vector<int> _meta_remain_snp;
    std::map<std::string,int> _meta_snp_name_map;
    std::vector<std::string> _covar_pheno_name;
    std::vector<std::string> _meta_snp_a1;
    std::vector<std::string> _meta_snp_a2;
    eigenMatrix _meta_snp_freq;
    eigenMatrix _meta_snp_b;
    eigenMatrix _meta_snp_se;
    eigenMatrix _meta_snp_pval;
    eigenMatrix _meta_snp_n_o;
    eigenVector _meta_vp_trait;
    std::vector<double> _meta_popu_prev;
    std::vector<double> _meta_smpl_prev;
    
    // gene-trait association
    std::map<std::string, int> _probe_name_map;
    std::vector<std::string> _probe_name;
    int _probe_num;
    eigenMatrix _probe_data;
    std::vector<int> _e_include;
    eigenVector _ecojo_z;
    eigenVector _ecojo_b;
    eigenVector _ecojo_se;
    eigenVector _ecojo_n;
    eigenVector _ecojo_pval;
    eigenMatrix _ecojo_R;
    eigenMatrix _ecojo_RC;
    eigenMatrix _ecojo_wholeR;
    double _ecojo_p_cutoff;
    double _ecojo_collinear;

    // PC adjustment
    double _ttl_mk_num;
    std::vector<double> _eigen_value;
};

class locus_bp {
public:
    std::string locus_name;
    int chr;
    int bp;

    locus_bp(std::string locus_name_buf, int chr_buf, int bp_buf) {
        locus_name = locus_name_buf;
        chr = chr_buf;
        bp = bp_buf;
    }

    bool operator()(const locus_bp & other) {
        return (chr == other.chr && bp <= other.bp);
    }
};

#endif
