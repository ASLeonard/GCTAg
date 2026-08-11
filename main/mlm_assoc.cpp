/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * Implementations of functions for mixed linera model association analysis
 *
 * 2013 by Jian Yang <jian.yang.qt@gmail.com>
 *
 * This file is distributed under the GNU General Public
 * License, Version 3.  Please see the file LICENSE for more
 * details
 */

#include "gcta.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string_view>

// ---------------------------------------------------------------------------
// Woodbury MLMA helpers (shared between mlma_calcu_stat and mlma_calcu_stat_covar)
// ---------------------------------------------------------------------------

gcta::WoodburyMLMACache gcta::build_woodbury_mlma_cache() const {
    WoodburyMLMACache wb;
    wb.Uk_f         = _Uk.transpose().cast<float>();  // k×n layout
    wb.ck_f         = _ck.cast<float>();
    wb.sigma2_eff_f = static_cast<float>(_sigma2_eff);
    // _ck is clamped >= 0 in calcu_Vi, so sqrt is safe here.
    wb.sqrt_ck_f    = wb.ck_f.cwiseSqrt();
    return wb;
}

void gcta::mlma(std::string grm_file, bool m_grm_flag, std::string subtract_grm_file, std::string phen_file, std::string qcovar_file, std::string covar_file, int mphen, int MaxIter, std::vector<double> reml_priors, std::vector<double> reml_priors_var, bool no_constrain, bool within_family, bool inbred, bool no_adj_covar, std::string weight_file, std::string save_reml_file, std::string load_reml_file)
{
    LOGGER.w(0, "The --mlma option has largely been superseded by the --mlma-stream option.");
    _within_family=within_family;
    _reml_max_iter=MaxIter;
    unsigned long i = 0, j = 0, k = 0;
    bool grm_flag=(!grm_file.empty());
    bool qcovar_flag=(!qcovar_file.empty());
    bool covar_flag=(!covar_file.empty());
    if (!qcovar_flag && !covar_flag) no_adj_covar=false;
    if (m_grm_flag) grm_flag = false;
    bool subtract_grm_flag = (!subtract_grm_file.empty());
    const bool skip_grm_loading = !load_reml_file.empty();
    if (subtract_grm_flag && m_grm_flag) LOGGER.e(0, "the --mlma-subtract-grm option cannot be used in combination with the --mgrm option.");
    
    // Read data
    int qcovar_num=0, covar_num=0;
    std::vector<std::string> phen_ID, qcovar_ID, covar_ID, grm_id;
    std::vector< std::vector<std::string> > phen_buf, qcovar, covar; // save individuals by column
    std::vector<std::string> grm_files;
    
    if(phen_file.empty()){
        LOGGER.e(0, "no file name in --pheno.");
    }
    read_phen(phen_file, phen_ID, phen_buf, mphen);
    update_id_map_kp(phen_ID, _id_map, _keep);
    if(qcovar_flag){
        qcovar_num=read_covar(qcovar_file, qcovar_ID, qcovar, true);
        update_id_map_kp(qcovar_ID, _id_map, _keep);
    }
    if(covar_flag){
        covar_num=read_covar(covar_file, covar_ID, covar, false);
        update_id_map_kp(covar_ID, _id_map, _keep);
    }
    // grm operations will overwrite the _keep
    if(_keep.size() < 1){
        LOGGER.e(0, "no individual is in common among the input files.");
    }

    const bool woodbury_needs_N = (_woodbury_rank == -1);  // only auto-k needs the SNP count

    if (skip_grm_loading) {
        if (!grm_file.empty() || m_grm_flag || subtract_grm_flag) {
            LOGGER << "Skipping GRM loading because --load-reml is set. Ensure the saved REML state matches the current sample set." << std::endl;
        }
    } else {
        if(subtract_grm_flag){
            grm_files.push_back(grm_file);
            grm_files.push_back(subtract_grm_file);
            for (i = 0; i < grm_files.size(); i++) {
                read_grm(grm_files[i], grm_id, false, true, true);
                update_id_map_kp(grm_id, _id_map, _keep);
            }
        }
        else{
            if(grm_flag){
                grm_files.push_back(grm_file);
                read_grm(grm_file, grm_id, true, false, !woodbury_needs_N);
                update_id_map_kp(grm_id, _id_map, _keep);
            }
            else if (m_grm_flag) {
                read_grm_filenames(grm_file, grm_files, false);
                for (i = 0; i < grm_files.size(); i++) {
                    read_grm(grm_files[i], grm_id, false, true, true);
                    update_id_map_kp(grm_id, _id_map, _keep);
                }
            }
            else{
                grm_files.push_back("NA");
                make_grm(false, false, inbred, true, 0, true);
                for(i=0; i<_keep.size(); i++) grm_id.push_back(_fid[_keep[i]]+":"+_pid[_keep[i]]);
            }
        }
    }
    
    std::vector<std::string> uni_id;
    std::map<std::string, int> uni_id_map;
    compact_indi_data();
    make_uni_id(uni_id, uni_id_map);
    _n=_keep.size();
    if(_n<1) LOGGER.e(0, "no individual is in common in the input files.");
    LOGGER<<_n<<" individuals are in common in these files."<<std::endl;
    
    // construct model terms
    _y.setZero(_n);
    for(size_t i = 0; i < phen_ID.size(); ++i){
        if(auto iter = uni_id_map.find(phen_ID[i]); iter != uni_id_map.end()) {
            _y[iter->second] = std::stod(phen_buf[i][mphen-1]);
        }
    }

    _r_indx.clear();
    std::vector<int> kp;
    if (!skip_grm_loading) {
        if (subtract_grm_flag) {
            _r_indx = {0, 1};
            _A.resize(_r_indx.size());

            LOGGER << "\nReading the primary GRM from [" << grm_files[1] << "] ..." << std::endl;
            read_grm(grm_files[1], grm_id, true, false, false);

            StrFunc::match(uni_id, grm_id, kp);
            (_A[0]).resize(_n, _n);
            eigenMatrix A_N_buf(_n, _n);
            {
                // read_grm_bin/gz fill both triangles; no selfadjointView copy needed for _grm.
                // _grm_N is float so a cast<double>() temporary is unavoidable, but we
                // avoid the intermediate MatrixXf symmetrization copy.
                Eigen::Map<const Eigen::VectorXi> kp_idx(kp.data(), _n);
                (_A[0]) = _grm(kp_idx, kp_idx);
                A_N_buf = _grm_N.cast<double>()(kp_idx, kp_idx);
            }

            LOGGER << "\nReading the secondary GRM from [" << grm_files[0] << "] ..." << std::endl;
            read_grm(grm_files[0], grm_id, true, false, false);
            LOGGER<<"\nSubtracting [" << grm_files[1] << "] from [" << grm_files[0] << "] ..." << std::endl;
            StrFunc::match(uni_id, grm_id, kp);
            {
                Eigen::Map<const Eigen::VectorXi> kp_idx(kp.data(), _n);
                eigenMatrix grm2_slice = _grm(kp_idx, kp_idx);
                eigenMatrix grm2_N_slice = _grm_N.cast<double>()(kp_idx, kp_idx);
                _A[0] = ((_A[0].array() * A_N_buf.array()) - (grm2_slice.array() * grm2_N_slice.array()))
                         / (A_N_buf.array() - grm2_N_slice.array());
            }
            _grm.resize(0,0);
            _grm_N.resize(0,0);
        }
        else {
            _r_indx.resize(grm_files.size() + 1);
            std::iota(_r_indx.begin(), _r_indx.end(), 0);
            _A.resize(_r_indx.size());
            if(grm_flag){
                StrFunc::match(uni_id, grm_id, kp);
                bool identity_kp = (static_cast<int>(_n) == _grm.rows());
                if (identity_kp) {
                    for (int ii = 0; ii < static_cast<int>(_n) && identity_kp; ii++)
                        if (kp[ii] != ii) identity_kp = false;
                }
                if (identity_kp) {
                    _A[0].swap(_grm);  // O(1): transfers ownership, _grm becomes empty
                    // _grm_N already matches _n/order 1:1 when identity_kp — nothing to subset.
                } else {
                    // read_grm_bin/gz fill both triangles, so no selfadjointView copy needed.
                    Eigen::Map<const Eigen::VectorXi> kp_idx(kp.data(), _n);
                    (_A[0]) = _grm(kp_idx, kp_idx);
                    _grm.resize(0,0);
                    if (woodbury_needs_N)
                        _grm_N = Eigen::MatrixXf(_grm_N(kp_idx, kp_idx));  // keep in sync with _A[0]'s order
                }
                if (!woodbury_needs_N)
                    _grm_N.resize(0,0);  // not needed after _A[0] is filled, unless Woodbury auto-k needs it later
            }
            else if(m_grm_flag){
                LOGGER << "There are " << grm_files.size() << " GRM file names specified in the file [" + grm_file + "]." << std::endl;
                for (i = 0; i < grm_files.size(); i++) {
                    LOGGER << "Reading the GRM from the " << i + 1 << "th file ..." << std::endl;
                    read_grm(grm_files[i], grm_id, true, false, true);
                    StrFunc::match(uni_id, grm_id, kp);
                    bool identity_kp_m = (static_cast<int>(_n) == _grm.rows());
                    if (identity_kp_m) {
                        for (int ii = 0; ii < static_cast<int>(_n) && identity_kp_m; ii++)
                            if (kp[ii] != ii) identity_kp_m = false;
                    }
                    if (identity_kp_m) {
                        _A[i].swap(_grm);
                    } else {
                        // read_grm_bin/gz fill both triangles, so no selfadjointView copy needed.
                        Eigen::Map<const Eigen::VectorXi> kp_idx(kp.data(), _n);
                        (_A[i]) = _grm(kp_idx, kp_idx);
                        _grm.resize(0,0);
                    }
                    _grm_N.resize(0,0);  // not needed after _A[i] is filled
                }
            }
            else{
                StrFunc::match(uni_id, grm_id, kp);
                {
                    eigenMatrix grm_sym(_grm.selfadjointView<Eigen::Lower>());
                    Eigen::Map<const Eigen::VectorXi> kp_idx(kp.data(), _n);
                    (_A[0]) = grm_sym(kp_idx, kp_idx);
                }
                _grm.resize(0,0);
            }
        }
        // _A[last] left size-0 (identity convention) unless weights override the diagonal.
        if(!weight_file.empty()){
            _A[_r_indx.size()-1]=eigenMatrix::Identity(_n, _n);
            std::vector<std::string> weight_ID;
            std::vector<double> weights;

            read_weight(weight_file, weight_ID, weights);
            update_id_map_kp(weight_ID, _id_map, _keep);

        //if(!weight_file.empty()){
            // contruct weight
            Eigen::VectorXd v_weight(_n);
            for (size_t i = 0; i < weight_ID.size(); ++i) {
                if (auto it = uni_id_map.find(weight_ID[i]); it != uni_id_map.end()) {
                    v_weight(it->second) = weights[i];
                }
            }
            //v_weight = 1.0 / v_weight.array();
            //v_weight = 1.0 / v_weight.array() - (1.0 / v_weight.array()).mean() + 1;
            /*
            std::ofstream o_test("weight_out.txt");
            for(int i = 0; i < v_weight.size(); i++){
                o_test << v_weight[i] << "\t" << _y[i] << std::endl;
            }
            o_test.close();
            */

            _A[_r_indx.size() - 1].diagonal() = v_weight;
        }
    }

    // construct X matrix
    std::vector<eigenMatrix> E_float;
    eigenMatrix qE_float;
    construct_X(_n, uni_id_map, qcovar_flag, qcovar_num, qcovar_ID, qcovar, covar_flag, covar_num, covar_ID, covar, E_float, qE_float);

    
    // names of variance component
    if (!skip_grm_loading) {
        for (size_t i = 0; i < grm_files.size(); ++i) {
            const std::string suffix = (grm_files.size() == 1) ? "" : std::to_string(i + 1);
            _var_name.emplace_back("V(G" + suffix + ")");
            _hsq_name.emplace_back("V(G" + suffix + ")/Vp");
        }
        _var_name.push_back("V(e)");
        
        // within family
        if(_within_family) detect_family();
    }
    
    // run REML algorithm
    LOGGER << "\nPerforming MLM association analyses" << (subtract_grm_flag?"":" (including the candidate SNP)") << " ..."<<std::endl;
    unsigned long n=_keep.size(), m=_include.size();
	
    if(!load_reml_file.empty()) {
            // Load REML state from file
            LOGGER << "Loading REML state from [" << load_reml_file << "] ..." << std::endl;
            load_reml_state(load_reml_file, no_adj_covar);
    } else {
        // Run REML estimation
        if (_woodbury_rank > 0)
            compute_woodbury_basis(_woodbury_rank, 0.0, 0);
        else if (_woodbury_rank < 0)
            compute_woodbury_basis(0, _woodbury_buffer_factor, _woodbury_k_max);
        reml(false, true, true, reml_priors, reml_priors_var, -2.0, -2.0, no_constrain, true, true);
        
        if(!save_reml_file.empty()) {
            // Save REML state and exit - no need for genotype data beyond this point
            LOGGER << "Saving REML state to [" << save_reml_file << "] ..." << std::endl;
            save_reml_state(save_reml_file, no_adj_covar);
            LOGGER << "REML estimation completed. Use --load-reml to perform association tests." << std::endl;
            
            // Clean up matrices that won't be needed
            _P.resize(0,0);
            _A.clear();
            return;
        }
    }
    
    _P.resize(0,0);
    _A.clear();
    LOGGER << "\nRegression coefficient(s) (e.g., general mean): " <<_b <<std::endl;

    std::vector<float> y(n);
    if(!no_adj_covar)
        Eigen::Map<Eigen::VectorXf>(y.data(), n) = (_y - _X*_b).cast<float>(); // adjust phenotype for covariates
    else
        Eigen::Map<Eigen::VectorXf>(y.data(), n) = _y.cast<float>();
    
/*    if(grm_flag || m_grm_flag){
        LOGGER<<std::endl;
        _geno_mkl=new float[n*m];
        make_XMat_mkl(_geno_mkl, false);
        #pragma omp parallel for private(j, k)
        for(i=0; i<n; i++){
            for(j=0; j<m; j++){
                k=i*m+j;
                if(_geno_mkl[k]<1e5) _geno_mkl[k]-=_mu[_include[j]];
                else _geno_mkl[k]=0.0;
            }
        }
    }*/
    compact_snp_data();
    _mu.clear();
    if (_mu.empty()) calcu_mu();
    // Real (additive-scale) allele frequency for the Freq column: _additive_mu is
    // captured in read_bed_dosage() before the nonadditive recoding is applied, so
    // (unlike _mu, which calcu_mu() recomputes directly from the possibly-recoded
    // _geno_dose) it always reflects the true allele frequency. Falls back to _mu
    // when unavailable (e.g. the additive model, which doesn't populate it).
    const std::vector<double>& freq_mu = _additive_mu.empty() ? _mu : _additive_mu;

    auto [beta, se, pval] = no_adj_covar
        ? mlma_calcu_stat_covar(std::span<const float>(y), m)
        : mlma_calcu_stat(std::span<const float>(y), m);
    _geno.resize(0, 0);
    
    const std::string filename = _out + ".mlma";
    LOGGER<<"\nSaving the results of the mixed linear model association analyses of "<<m<<" SNPs to ["+filename+"] ..."<<std::endl;

    std::ofstream ofile(filename);
    if(!ofile) LOGGER.e(0, "cannot open the file ["+filename+"] to write.");
    ofile<<"Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\t"<<(_log_pval ? "log_p" : "p")<<std::endl;
    std::string chunk;
    chunk.reserve(1 << 22); // 4 MB chunk

    for (size_t i = 0; i < m; ++i) {
        std::format_to(
            std::back_inserter(chunk),
            "{}\t{}\t{}\t{}\t{}\t",
            _chr[i],
            _snp_name[i],
            _bp[i],
            _allele_alt[i],
            _allele_ref[i]
        );

        if (pval[i] > 1.0) {
            chunk += "NA\tNA\tNA\tNA\n";
        } else {
            std::format_to(
                std::back_inserter(chunk),
                "{:.6g}\t{:.6g}\t{:.6g}\t{:.6g}\n",
                0.5 * freq_mu[i],
                beta[i],
                se[i],
                pval[i]
            );
        }

        // Flush when chunk is large enough
        if (chunk.size() >= (1 << 21)) { // 2 MB threshold
            ofile.write(chunk.data(), chunk.size());
            chunk.clear();
        }
    }
    // Final flush
    if (!chunk.empty()) {
        ofile.write(chunk.data(), chunk.size());
    }
}

gcta::MlmaResult gcta::mlma_calcu_stat(std::span<const float> y, unsigned long m)
{
    // When _Vi_use_llt=true (both exact and Hutch++ paths after their respective REML
    // preps), _Vi_L holds the lower Cholesky factor L of V (from dpotrf).
    // We cast L to float and use triangular solves directly, skipping dpotri and the
    // float LLT entirely.  Mathematical identities used:
    //   V^{-1} y  = L^{-T}(L^{-1} y)   [two STRSV — forward + backward solve on L]
    //   x^T V^{-1} x = ||L^{-1} x||^2  [STRSM: L^{-1} X in-place, then squared norms]
    // Saves one dpotri O(n³/3 double) + one float LLT O(n³/3 float) per MLMA run.
    // The fallback (!use_L) preserves the old dpotri path for any caller that still
    // arrives with _Vi_use_llt=false and a valid _Vi.
    const bool use_woodbury = _Vi_use_woodbury;
    const bool use_L = _Vi_use_llt;

    const auto n = static_cast<Eigen::Index>(y.size());
    constexpr Eigen::Index max_block_size = 10000;
    unsigned long i = 0;
    std::vector<float> beta(m, 0.0f), se(m, 0.0f);
    std::vector<double> pval(m, 2.0);

    Eigen::Map<const Eigen::VectorXf> y_vec(y.data(), n);
    Eigen::VectorXf Vi_y(n);

    Eigen::MatrixXf L_f;                         // lower Cholesky of V  (when use_L)
    Eigen::LLT<Eigen::MatrixXf> Vi_llt;          // Cholesky of V^{-1}  (when !use_L)

    // Precomputed woodbury float-precision quantities for the block loop
    gcta::WoodburyMLMACache wb;

    if (use_woodbury) {
        wb   = build_woodbury_mlma_cache();
        Vi_y = woodbury_apply_Vi_f(wb, y_vec);
    } else if (use_L) {
        L_f = _Vi_L.cast<float>();
        _Vi_L.resize(0, 0);
        _Vi_use_llt = false;
        // Vi_y = V^{-1} y = L^{-T}(L^{-1} y) via two float triangular solves.
        Vi_y = y_vec;
        cblas_strsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), L_f.data(), static_cast<int>(n), Vi_y.data(), 1);
        cblas_strsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                    static_cast<int>(n), L_f.data(), static_cast<int>(n), Vi_y.data(), 1);
    } else {
        if (_Vi_use_llt) {
            _Vi.swap(_Vi_L);  // O(1): _Vi gets L's storage, _Vi_L becomes empty
            gcta_blas_int blas_n_m = static_cast<gcta_blas_int>(_n);
            if (gcta_dpotri(blas_n_m, _Vi.data(), blas_n_m) != 0)
                LOGGER.e(0, "dpotri failed materialising V^{-1} for mlma_calcu_stat.");
            _Vi.triangularView<Eigen::Upper>() = _Vi.transpose();
            _Vi_use_llt = false;
        }
        // Symmetrise Vi into a plain dense float matrix for SSYMV below.
        Eigen::MatrixXf Vi = _Vi.cast<float>().selfadjointView<Eigen::Upper>();
        _Vi.resize(0,0);
        Vi_y.noalias() = Vi * y_vec;
        // Cholesky-factor Vi for the per-block STRMM below.
        Vi_llt = Eigen::LLT<Eigen::MatrixXf>(Vi);
        if (Vi_llt.info() != Eigen::Success)
            LOGGER.e(0, "mlma_calcu_stat: Vi is not positive definite.");
        Vi.resize(0, 0);
    }

    LOGGER<<"\nRunning association tests for "<<m<<" SNPs ..."<<std::endl;

    compact_indi_data();
    if (!_make_XMat_no_compact) compact_snp_data();

    int k = 0, l = 0;
    // Pre-allocate at max_block_size — no heap activity inside the hot loop.
    // leftCols(bs) selects the active columns for partial (last) blocks.
    // The STRMM/STRSM overwrites X_block in-place (B is overwritten, A is read-only),
    // eliminating the separate UX_block allocation (~600 MB for n=15k).
    Eigen::MatrixXf X_block(n, max_block_size);
    Eigen::VectorXf Xt_Vi_y_block(max_block_size);
    Eigen::VectorXf xvx_diag(max_block_size);    // diag(X^T Vi X)

    std::vector<int> indx;
    indx.reserve(max_block_size);   // single alloc before the loop

    int last_pct = -1;
    for(i = 0; i < m; ){
        int cur_pct = static_cast<int>((i * 100UL) / m);
        if(cur_pct != last_pct){
            LOGGER.p(0, std::to_string(i) + " / " + std::to_string(m) + " SNPs (" + std::to_string(cur_pct) + "%)");
            last_pct = cur_pct;
        }
        const Eigen::Index bs = static_cast<Eigen::Index>(
            std::min(static_cast<unsigned long>(max_block_size), m - i));
        indx.resize(bs);
        std::iota(indx.begin(), indx.end(), static_cast<int>(i));
        make_XMat_subset_dense(X_block, indx, false);  // X_block is n×bs (written into pre-alloc'd buffer)

        // GEMV first — X_block is still hot in cache from make_XMat_subset.
        // X^T Vi_y (bs×1) — single GEMV into pre-allocated target.
        Xt_Vi_y_block.head(bs).noalias() = X_block.leftCols(bs).transpose() * Vi_y;

        if (use_woodbury) {
            woodbury_xvx_diag_block(wb, X_block, bs, xvx_diag);
        } else if (use_L) {
            // In-place STRSM: X_block = L^{-1} * X_block.
            // x^T V^{-1} x = ||L^{-1} x||^2  (L is lower Cholesky of V).
            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                        static_cast<int>(n), static_cast<int>(bs), 1.0f,
                        L_f.data(), static_cast<int>(n),
                        X_block.data(), static_cast<int>(n));
            xvx_diag.head(bs) = X_block.leftCols(bs).colwise().squaredNorm();
        } else {
            // In-place STRMM: X_block = L_vit^T * X_block  (L_vit = lower Cholesky of V^{-1}).
            // x^T V^{-1} x = ||L_vit^T x||^2.
            // cblas_strmm is safe to call in-place (B is overwritten, A is read-only).
            cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                        static_cast<int>(n), static_cast<int>(bs), 1.0f,
                        Vi_llt.matrixLLT().data(), static_cast<int>(n),
                        X_block.data(), static_cast<int>(n));
            xvx_diag.head(bs) = X_block.leftCols(bs).colwise().squaredNorm();
        }

        for(l = 0; l < bs; l++){
            double pval_d;
            if (mlma_snp_stat(Xt_Vi_y_block[l], xvx_diag[l], _log_pval,
                              beta[i + l], se[i + l], pval_d))
                pval[i + l] = static_cast<double>(pval_d);
        }

        i += bs;
    }
    LOGGER.p(0, std::to_string(m) + " / " + std::to_string(m) + " SNPs (100%)");
    return {beta, se, pval};
}

gcta::MlmaResult gcta::mlma_calcu_stat_covar(std::span<const float> y, unsigned long m)
{
    // When _Vi_use_llt=true (both exact and Hutch++ paths after their respective REML
    // preps), _Vi_L holds the lower Cholesky factor L of V (from dpotrf).
    // Cast L to float and use triangular solves directly, skipping dpotri and the
    // float LLT entirely.  Mathematical identities:
    //   V^{-1} y  = L^{-T}(L^{-1} y)   [two STRSV — forward + backward solve on L]
    //   V^{-1} C  = L^{-T}(L^{-1} C)   [two STRSM on n×p]
    //   x^T V^{-1} x = ||L^{-1} x||^2  [STRSM: L^{-1} X in-place, then squared norms]
    // Saves one dpotri O(n³/3 double) + one float LLT O(n³/3 float) per MLMA run.
    const bool use_woodbury = _Vi_use_woodbury;
    const bool use_L = _Vi_use_llt;

    const auto n = static_cast<Eigen::Index>(y.size());
    const Eigen::Index p = static_cast<Eigen::Index>(_X_c);  // number of fixed covariates
    constexpr Eigen::Index max_block_size = 10000;
    unsigned long i = 0;
    std::vector<float> beta(m, 0.0f), se(m, 0.0f);
    std::vector<double> pval(m, 2.0);

    Eigen::Map<const Eigen::VectorXf> y_vec(y.data(), n);
    Eigen::VectorXf Vi_y(n);
    Eigen::MatrixXf C = _X.cast<float>();
    Eigen::MatrixXf Vi_C(n, p);

    Eigen::MatrixXf L_f;                         // lower Cholesky of V  (when use_L)
    Eigen::LLT<Eigen::MatrixXf> Vi_llt;          // Cholesky of V^{-1}  (when !use_L)

    // Precomputed woodbury float-precision quantities for the block loop
    gcta::WoodburyMLMACache wb;

    if (use_woodbury) {
        wb   = build_woodbury_mlma_cache();
        Vi_y = woodbury_apply_Vi_f(wb, y_vec);
        Vi_C = woodbury_apply_Vi_mat_f(wb, C);
    } else if (use_L) {
        L_f = _Vi_L.cast<float>();
        _Vi_L.resize(0, 0);
        _Vi_use_llt = false;
        // Vi_y = V^{-1} y = L^{-T}(L^{-1} y) via two float triangular solves.
        Vi_y = y_vec;
        cblas_strsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), L_f.data(), static_cast<int>(n), Vi_y.data(), 1);
        cblas_strsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                    static_cast<int>(n), L_f.data(), static_cast<int>(n), Vi_y.data(), 1);
        // Vi_C = V^{-1} C = L^{-T}(L^{-1} C) via two float STRSM on n×p.
        Vi_C = C;
        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), static_cast<int>(p), 1.0f,
                    L_f.data(), static_cast<int>(n), Vi_C.data(), static_cast<int>(n));
        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                    static_cast<int>(n), static_cast<int>(p), 1.0f,
                    L_f.data(), static_cast<int>(n), Vi_C.data(), static_cast<int>(n));
    } else {
        if (_Vi_use_llt) {
            _Vi.swap(_Vi_L);  // O(1): _Vi gets L's storage, _Vi_L becomes empty
            gcta_blas_int blas_n_mc = static_cast<gcta_blas_int>(_n);
            if (gcta_dpotri(blas_n_mc, _Vi.data(), blas_n_mc) != 0)
                LOGGER.e(0, "dpotri failed materialising V^{-1} for mlma_calcu_stat_covar.");
            _Vi.triangularView<Eigen::Upper>() = _Vi.transpose();
            _Vi_use_llt = false;
        }
        // Symmetrise Vi once into a plain dense float matrix.
        Eigen::MatrixXf Vi = _Vi.cast<float>().selfadjointView<Eigen::Upper>();
        _Vi.resize(0,0);
        Vi_y.noalias() = Vi * y_vec;
        // Vi_C = Vi * C (n×p): retained for the hot loop D = Vi_C^T * X.
        Vi_C.noalias() = Vi * C;
        // Cholesky-factor Vi for the per-block STRMM below.
        Vi_llt = Eigen::LLT<Eigen::MatrixXf>(Vi);
        if (Vi_llt.info() != Eigen::Success)
            LOGGER.e(0, "mlma_calcu_stat_covar: Vi is not positive definite.");
        Vi.resize(0, 0);
    }

    // A = C^T Vi C = Vi_C^T C  (p×p); same formula regardless of path since
    // Vi_C = V^{-1} C is already computed above.
    Eigen::MatrixXf A_mat(p, p);
    A_mat.noalias() = Vi_C.transpose() * C;
    Eigen::LLT<Eigen::MatrixXf> A_llt(A_mat);
    if(A_llt.info() != Eigen::Success)
        LOGGER.e(0, "covariate matrix C^T Vi C is not positive-definite/invertible.");

    // t = A^{-1} (C^T Vi_y)  (p×1)
    Eigen::VectorXf t_vec(p);
    t_vec.noalias() = C.transpose() * Vi_y;
    t_vec = A_llt.solve(t_vec);

    LOGGER << "\nRunning association tests for " << m << " SNPs ..." << std::endl;

    int k = 0, l = 0;
    // The STRMM/STRSM overwrites X_block in-place after all operations using the
    // original genotype data are complete.
    Eigen::MatrixXf X_block(n, max_block_size);
    std::vector<int> indx;
    indx.reserve(max_block_size);

    // Pre-allocated block temporaries — zero heap activity per SNP
    Eigen::MatrixXf D_block(p, max_block_size);      // Vi_C^T * X_block  (p×bs)
    Eigen::MatrixXf E_block(p, max_block_size);      // A^{-1} * D_block  (p×bs)
    Eigen::VectorXf f_vec(max_block_size);           // X^T Vi_y          (bs×1)
    Eigen::VectorXf Dt_t_vec(max_block_size);        // D^T t             (bs×1)
    Eigen::VectorXf xvx_diag(max_block_size);        // diag(X^T Vi X)    (bs×1)
    Eigen::VectorXf d_dot_e_diag(max_block_size);    // diag(D^T E)       (bs×1)

    int last_pct_c = -1;
    for(i = 0; i < m; ){
        int cur_pct_c = static_cast<int>(i * 100 / m);
        if(cur_pct_c != last_pct_c){
            LOGGER.p(0, std::to_string(i) + " / " + std::to_string(m) + " SNPs (" + std::to_string(cur_pct_c) + "%)");
            last_pct_c = cur_pct_c;
        }
        const Eigen::Index bs = static_cast<Eigen::Index>(
            std::min(static_cast<unsigned long>(max_block_size), m - i));
        indx.resize(bs);
        for(k = 0; k < bs; k++) indx[k] = i + k;
        make_XMat_subset_dense(X_block, indx, false);  // X_block is n×bs

        // All operations using the original genotype values must come first,
        // before the in-place STRMM/STRSM overwrites X_block.

        // D = C^T Vi X = Vi_C^T X  (p×bs)
        D_block.leftCols(bs).noalias() = Vi_C.transpose() * X_block;

        // f = X^T Vi_y  (bs×1)
        f_vec.head(bs).noalias() = X_block.transpose() * Vi_y;

        // E = A^{-1} * D  (p×bs)  — does not need X_block
        E_block.leftCols(bs).noalias() = A_llt.solve(D_block.leftCols(bs));

        // Dt_t = D^T t  (bs×1)  — does not need X_block
        Dt_t_vec.head(bs).noalias() = D_block.leftCols(bs).transpose() * t_vec;

        // diag(D^T E)  — does not need X_block
        d_dot_e_diag.head(bs) = (D_block.leftCols(bs).cwiseProduct(E_block.leftCols(bs))).colwise().sum();

        if (use_woodbury) {
            woodbury_xvx_diag_block(wb, X_block, bs, xvx_diag);
        } else if (use_L) {
            // In-place STRSM: X_block = L^{-1} * X_block.
            // x^T V^{-1} x = ||L^{-1} x||^2  (L is lower Cholesky of V).
            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                        static_cast<int>(n), static_cast<int>(bs), 1.0f,
                        L_f.data(), static_cast<int>(n),
                        X_block.data(), static_cast<int>(n));
            xvx_diag.head(bs) = X_block.leftCols(bs).colwise().squaredNorm();
        } else {
            // In-place STRMM: X_block = L_vit^T * X_block  (L_vit = lower Cholesky of V^{-1}).
            // x^T V^{-1} x = ||L_vit^T x||^2.
            cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                        static_cast<int>(n), static_cast<int>(bs), 1.0f,
                        Vi_llt.matrixLLT().data(), static_cast<int>(n),
                        X_block.data(), static_cast<int>(n));
            xvx_diag.head(bs) = X_block.leftCols(bs).colwise().squaredNorm();
        }

        for(l = 0; l < bs; l++){
            const float S         = xvx_diag[l] - d_dot_e_diag[l];  // Schur complement
            const float numerator = f_vec[l] - Dt_t_vec[l];
            double pval_d;
            if (mlma_snp_stat(numerator, S, _log_pval,
                              beta[i + l], se[i + l], pval_d))
                pval[i + l] = static_cast<float>(pval_d);
        }

        i += bs;
    }
    LOGGER.p(0, std::to_string(m) + " / " + std::to_string(m) + " SNPs (100%)");
    return {beta, se, pval};
}

void gcta::mlma_loco(std::string phen_file, std::string qcovar_file, std::string covar_file, int mphen, int MaxIter, std::vector<double> reml_priors, std::vector<double> reml_priors_var, bool no_constrain, bool inbred, bool no_adj_covar)
{
    unsigned long i=0, j=0, k=0, c1=0, c2=0, n=0;
    _reml_max_iter=MaxIter;
    bool qcovar_flag=(!qcovar_file.empty());
    bool covar_flag=(!covar_file.empty());
    if(!qcovar_flag && !covar_flag) no_adj_covar=false;
    
    // Read data
    int qcovar_num=0, covar_num=0;
    std::vector<std::string> phen_ID, qcovar_ID, covar_ID, grm_id;
    std::vector< std::vector<std::string> > phen_buf, qcovar, covar; // save individuals by column

    if(phen_file.empty()){
        LOGGER.e(0, "no file name in --pheno.");
    }
    
    read_phen(phen_file, phen_ID, phen_buf, mphen);
    update_id_map_kp(phen_ID, _id_map, _keep);
    if(qcovar_flag){
        qcovar_num=read_covar(qcovar_file, qcovar_ID, qcovar, true);
        update_id_map_kp(qcovar_ID, _id_map, _keep);
    }
    if(covar_flag){
        covar_num=read_covar(covar_file, covar_ID, covar, false);
        update_id_map_kp(covar_ID, _id_map, _keep);
    }
    n=_keep.size();
    _n=_keep.size();
    if(_n<1) LOGGER.e(0, "no individual is in common among the input files.");
    LOGGER<<_n<<" individuals are in common in these files."<<std::endl;
    
    std::vector<int> chrs, vi_buf(_chr);
    std::ranges::sort(vi_buf);
    auto unique_range = std::ranges::unique(vi_buf);
    vi_buf.erase(unique_range.begin(), vi_buf.end());
    if(vi_buf.size()<2) LOGGER.e(0, "There is only one chromosome. The MLM leave-on-chromosome-out (LOCO) analysis requires at least two chromosomes.");
    for(i=0; i<vi_buf.size(); i++){
        if(isAutosomalChr(vi_buf[i])) chrs.push_back(vi_buf[i]);
    }
    std::vector<int> include_o(_include);
    std::map<std::string, int> snp_name_map_o(_snp_name_map);
    std::vector<float> m_chrs_f(chrs.size());
    std::vector<eigenMatrix> grm_chrs(chrs.size());
    std::vector< std::vector<int> > icld_chrs(chrs.size());
    LOGGER<<std::endl;
    if(_mu.empty()) calcu_mu();
    LOGGER<<"\nCalculating the genetic relationship matrix for each of the "<<chrs.size()<<" chromosomes ... "<<std::endl;
    for(c1=0; c1<chrs.size(); c1++){
        LOGGER<<"Chr "<<chrs[c1]<<":"<<std::endl;
        extract_chr(chrs[c1], chrs[c1]);
        make_grm(false, false, inbred, true, 0, true);
        
        m_chrs_f[c1]=(float)_include.size();
        icld_chrs[c1]=_include;
        _include=include_o;
        _snp_name_map=snp_name_map_o;
        
        grm_chrs[c1]=std::move(_grm);
        _geno.resize(0, 0);
    }
    for(i=0; i<_keep.size(); i++) grm_id.push_back(_fid[_keep[i]]+":"+_pid[_keep[i]]);
    
    std::vector<std::string> uni_id;
    std::map<std::string, int> uni_id_map;
    std::map<std::string, int>::iterator iter;
    make_uni_id(uni_id, uni_id_map);
    
    // construct model terms
    _y.setZero(_n);
    for(i=0; i<phen_ID.size(); i++){
        iter=uni_id_map.find(phen_ID[i]);
        if(iter==uni_id_map.end()) continue;
        _y[iter->second]=atof(phen_buf[i][mphen-1].c_str());
    }
    
    // construct X matrix
    std::vector<eigenMatrix> E_float;
    eigenMatrix qE_float;
    construct_X(_n, uni_id_map, qcovar_flag, qcovar_num, qcovar_ID, qcovar, covar_flag, covar_num, covar_ID, covar, E_float, qE_float);
    
    // names of variance component
    _var_name.push_back("V(G)");
    _hsq_name.push_back("V(G)/Vp");
    _var_name.push_back("V(e)");
    
    // MLM association
    LOGGER<<"\nPerforming MLM association analyses (leave-one-chromosome-out) ..."<<std::endl;
    
    std::vector<int> kp;
    StrFunc::match(uni_id, grm_id, kp);
    _r_indx.resize(2);
    for(i=0; i<2; i++) _r_indx[i]=i;
    _A.resize(_r_indx.size());
    // _A[1] left size-0 (identity convention); hot paths handle this implicitly.
    
    eigenVector y_buf=_y;
    std::vector<float> y(_n);
    std::vector<std::vector<float>> beta(chrs.size()), se(chrs.size());
    std::vector<std::vector<double>> pval(chrs.size());
    for(c1=0; c1<chrs.size(); c1++){
        LOGGER<<"\n-----------------------------------\n#Chr "<<chrs[c1]<<":"<<std::endl;
        extract_chr(chrs[c1], chrs[c1]);
        
        _A[0] = eigenMatrix::Zero(_n, _n);
        double d_buf = 0;
        {
            Eigen::Map<const Eigen::VectorXi> kp_idx(kp.data(), _n);
            for(c2=0; c2<chrs.size(); c2++){
                if(chrs[c1]==chrs[c2]) continue;
                eigenMatrix grm_sym(grm_chrs[c2].selfadjointView<Eigen::Lower>());
                _A[0] += grm_sym(kp_idx, kp_idx) * static_cast<double>(m_chrs_f[c2]);
                d_buf += m_chrs_f[c2];
            }
        }
        _A[0] /= d_buf;
        
        // run REML algorithm
        reml(false, true, true, reml_priors, reml_priors_var, -2.0, -2.0, no_constrain, true, true);
        if(!no_adj_covar) y_buf=_y.array()-(_X*_b).array(); // adjust phenotype for covariates
        for(i=0; i<_n; i++) y[i]=y_buf[i];
        reml_priors.clear();
        reml_priors_var=_varcmp;
        _P.resize(0,0);
        _A[0].resize(0,0);

        auto [b, s, p] = no_adj_covar
            ? mlma_calcu_stat_covar(std::span<const float>(y), _include.size())
            : mlma_calcu_stat(std::span<const float>(y), _include.size());
        beta[c1] = std::move(b); se[c1] = std::move(s); pval[c1] = std::move(p);
        
        _include=include_o;
        _snp_name_map=snp_name_map_o;
        LOGGER<<"-----------------------------------"<<std::endl;
    }
    
    std::string filename=_out+".loco.mlma";
    LOGGER<<"\nSaving the results of the mixed linear model association analyses of "<<_include.size()<<" SNPs to ["+filename+"] ..."<<std::endl;
    std::ofstream ofile(filename.c_str());
    if(!ofile) LOGGER.e(0, "cannot open the file ["+filename+"] to write.");
    ofile<<"Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\t"<<(_log_pval ? "log_p" : "p")<<std::endl;
    for(c1=0; c1<chrs.size(); c1++){
        for(i=0; i<icld_chrs[c1].size(); i++){
            j=icld_chrs[c1][i];
            ofile<<_chr[j]<<"\t"<<_snp_name[j]<<"\t"<<_bp[j]<<"\t"<<_allele_alt[j]<<"\t"<<_allele_ref[j]<<"\t";
            const double freq_j = (!_additive_mu.empty() ? _additive_mu[j] : _mu[j]);
            if(pval[c1][i]>1.5) ofile<<"NA\tNA\tNA\tNA"<<std::endl;
            else ofile<<0.5*freq_j<<"\t"<<beta[c1][i]<<"\t"<<se[c1][i]<<"\t"<<pval[c1][i]<<std::endl;
        }
    }
    ofile.close();
}

// ---------------------------------------------------------------------------
// mlma_loco_v2 — memory-efficient LOCO MLMA
//
// Requires a pre-built all-autosome GRM supplied via --grm <prefix>.
//
// Algorithm per chromosome c:
//   G_loco_c = (G_all·m_all − G_chr_c·m_c) / (m_all − m_c)
//
// Peak RAM = O(3·n²·8) during the G_loco_c computation step, versus the
// legacy mlma_loco which holds all n_chr GRMs simultaneously O(n_chr·n²·8).
// For n=15 000, 29 autosomes: ~5 GB peak vs ~49 GB.
//
// When --reml-woodbury <k> is set (recommended), the Woodbury eigenvectors
// from chromosome c are used as a warm start for chromosome c+1, reducing
// power iterations from ~3 to ~1 and cutting total REML time by ~3×.
// ---------------------------------------------------------------------------
void gcta::mlma_loco_v2(std::string grm_file, std::string grm_chr_prefix,
                         std::string phen_file, std::string qcovar_file,
                         std::string covar_file, int mphen, int MaxIter,
                         std::vector<double> reml_priors,
                         std::vector<double> reml_priors_var,
                         bool no_constrain, bool inbred, bool no_adj_covar)
{
    unsigned long i = 0, c1 = 0;
    _reml_max_iter = MaxIter;
    const bool qcovar_flag = (!qcovar_file.empty());
    const bool covar_flag  = (!covar_file.empty());
    if (!qcovar_flag && !covar_flag) no_adj_covar = false;

    // ---- 1. Load whole-genome GRM (G_all) ----
    // read_grm populates _grm (lower-triangular n_grm×n_grm) and _grm_N.
    // update_id_map_kp restricts _keep to individuals in the GRM file.
    std::vector<std::string> grm_id_all;
    LOGGER << "\n[LOCO] Loading all-autosome GRM from [" << grm_file << "] ..." << std::endl;
    read_grm(grm_file, grm_id_all, /*out_id_log=*/true, /*read_id_only=*/false,
             /*dont_read_N=*/false);
    update_id_map_kp(grm_id_all, _id_map, _keep);

    if (_grm_N.size() == 0)
        LOGGER.e(0, "--mlma-loco: GRM N file [" + grm_file + ".grm.N.bin] is missing or empty. "
                    "Rebuild the GRM without --no-grm-N.");
    // Diagonal entry (0,0) equals the all-autosome marker count for individual 0.
    const double m_all = static_cast<double>(_grm_N(0, 0));
    if (m_all <= 0.0)
        LOGGER.e(0, "--mlma-loco: marker count read from N matrix is zero or negative.");
    LOGGER << "[LOCO] G_all: " << static_cast<long long>(m_all) << " markers." << std::endl;

    // ---- 2. Read phenotype and optional covariates ----
    int qcovar_num = 0, covar_num = 0;
    std::vector<std::string> phen_ID, qcovar_ID, covar_ID;
    std::vector<std::vector<std::string>> phen_buf, qcovar, covar;

    if (phen_file.empty()) LOGGER.e(0, "no file name in --pheno.");
    read_phen(phen_file, phen_ID, phen_buf, mphen);
    update_id_map_kp(phen_ID, _id_map, _keep);
    if (qcovar_flag) {
        qcovar_num = read_covar(qcovar_file, qcovar_ID, qcovar, true);
        update_id_map_kp(qcovar_ID, _id_map, _keep);
    }
    if (covar_flag) {
        covar_num = read_covar(covar_file, covar_ID, covar, false);
        update_id_map_kp(covar_ID, _id_map, _keep);
    }
    _n = _keep.size();
    if (_n < 1) LOGGER.e(0, "no individual is in common among the input files.");
    LOGGER << _n << " individuals are in common in these files." << std::endl;

    // ---- 3. Build ID maps and subset G_all to the final _keep individuals ----
    std::vector<std::string> uni_id;
    std::map<std::string, int> uni_id_map;
    make_uni_id(uni_id, uni_id_map);

    // kp_grm[i] = row/col in the loaded _grm corresponding to uni_id[i]
    std::vector<int> kp_grm;
    StrFunc::match(uni_id, grm_id_all, kp_grm);
    for (i = 0; i < kp_grm.size(); i++) {
        if (kp_grm[i] < 0)
            LOGGER.e(0, "--mlma-loco: individual " + uni_id[i] +
                        " is present in phenotype/genotype data but absent from ["
                        + grm_file + ".grm.id]. "
                        "Rebuild the all-autosome GRM to include all analysis individuals.");
    }

    eigenMatrix grm_all;
    {
        // Symmetrize _grm (lower triangle → full symmetric) then subset to _n.
        eigenMatrix grm_sym(_grm.selfadjointView<Eigen::Lower>());
        Eigen::Map<const Eigen::VectorXi> kp_idx(kp_grm.data(),
                                                  static_cast<Eigen::Index>(_n));
        grm_all = grm_sym(kp_idx, kp_idx);  // n×n submatrix (full symmetric)
    }
    _grm.resize(0, 0);
    _grm_N.resize(0, 0);
    {
        const double gb_one  = static_cast<double>(_n) * _n * 8.0 / 1e9;
        const double gb_peak = gb_one * 3.0;   // grm_all + grm_c_sym + _A[0]
        LOGGER << "[LOCO] G_all subset to " << _n << "×" << _n << "  ("
               << std::fixed << std::setprecision(2) << gb_one
               << " GB). Peak LOCO RAM ≈" << std::setprecision(1) << gb_peak << " GB." << std::endl;
        if (gb_peak > 10.0)
            LOGGER << "[LOCO][WARNING] Peak RAM ~" << std::setprecision(1) << gb_peak
                   << " GB.  Ensure sufficient system RAM before proceeding." << std::endl;
    }

    // ---- 4. Identify autosomes and collect per-chromosome SNP lists ----
    std::vector<int> chrs, vi_buf(_chr);
    std::ranges::sort(vi_buf);
    { auto ur = std::ranges::unique(vi_buf); vi_buf.erase(ur.begin(), vi_buf.end()); }
    if (vi_buf.size() < 2)
        LOGGER.e(0, "There is only one chromosome. LOCO analysis requires ≥2 chromosomes.");
    for (const int c : vi_buf)
        if (isAutosomalChr(c)) chrs.push_back(c);
    if (chrs.size() < 2)
        LOGGER.e(0, "LOCO analysis requires at least two autosomal chromosomes.");

    std::vector<int> include_o(_include);
    std::map<std::string, int> snp_name_map_o(_snp_name_map);

    if (_mu.empty()) calcu_mu();

    std::vector<std::vector<int>> icld_chrs(chrs.size());
    for (c1 = 0; c1 < chrs.size(); c1++) {
        extract_chr(chrs[c1], chrs[c1]);
        icld_chrs[c1] = _include;
        _include      = include_o;
        _snp_name_map = snp_name_map_o;
    }

    // ---- 5. Phenotype vector, covariate matrix, REML bookkeeping ----
    // uni_id and grm_id_keep are both derived from _keep in the same order,
    // so kp would always be the identity permutation.  No need to construct it.

    _y.setZero(_n);
    for (i = 0; i < phen_ID.size(); i++) {
        auto iter = uni_id_map.find(phen_ID[i]);
        if (iter == uni_id_map.end()) continue;
        _y[iter->second] = std::stod(phen_buf[i][mphen - 1]);
    }

    std::vector<eigenMatrix> E_float;
    eigenMatrix qE_float;
    construct_X(_n, uni_id_map, qcovar_flag, qcovar_num, qcovar_ID, qcovar,
                covar_flag, covar_num, covar_ID, covar, E_float, qE_float);

    _var_name.push_back("V(G)");
    _hsq_name.push_back("V(G)/Vp");
    _var_name.push_back("V(e)");

    _r_indx.resize(2);
    std::iota(_r_indx.begin(), _r_indx.end(), 0);
    _A.resize(2);

    // Disable any stale Woodbury state from a prior analysis so that reml()
    // takes the correct non-Woodbury path when _woodbury_rank == 0.
    if (_woodbury_rank == 0) _Vi_use_woodbury = false;

    // Prevent compact_snp_data() from destroying _snp_1/_snp_2 between
    // per-chromosome make_grm calls.  The scatter-gather path in make_XMat /
    // make_XMat_subset is correct without compaction; std_XMat now uses
    // _mu[_include[j]] which is equivalent post-compact and correct pre-compact.
    if (_dosage_flag)
        LOGGER.e(0, "--mlma-loco with --grm does not support dosage data.");
    _make_XMat_no_compact = true;

    LOGGER << "\n[LOCO] Running LOCO MLMA over " << chrs.size() << " autosomes." << std::endl;
    if (_woodbury_rank != 0)
        LOGGER << "[LOCO] --reml-woodbury active; eigenvectors warm-started from each "
                  "previous chromosome after the first." << std::endl;
    else
        LOGGER << "[LOCO] Tip: add --reml-woodbury <k> (e.g. k=200) for faster per-chr "
                  "REML and lower peak RAM (Woodbury path)." << std::endl;

    eigenVector y_buf = _y;
    std::vector<float> y(_n);

    // ---- 6a. Open output file for streaming (write per-chromosome as it completes) ----
    const std::string filename = _out + ".loco.mlma";
    LOGGER << "\nLOCO results will be streamed to [" << filename << "] ..." << std::endl;
    std::ofstream ofile(filename);
    if (!ofile) LOGGER.e(0, "cannot open [" + filename + "] to write.");
    ofile << "Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\t"
          << (_log_pval ? "log_p" : "p") << "\n";

    // ---- 6b. Per-chromosome LOCO loop ----
    for (c1 = 0; c1 < chrs.size(); c1++) {
        LOGGER << "\n-----------------------------------\n#Chr " << chrs[c1] << ":" << std::endl;
        extract_chr(chrs[c1], chrs[c1]);

        // Build or load G_chr_c for this chromosome.
        double m_c;
        if (grm_chr_prefix.empty()) {
            // Build G_chr_c via DSYRK from genotypes.
            // mlmassoc=true suppresses disk output and keeps _geno scaled; we free
            // _geno immediately because mlma_calcu_stat reads from _snp_1/_snp_2,
            // not _geno.  This avoids holding n×m_chr genotypes through the REML.
            make_grm(false, false, inbred, /*output_bin=*/false, /*grm_mtd=*/0,
                     /*mlmassoc=*/true);
            _geno.resize(0, 0);   // free n×m_c float genotype matrix
            m_c = static_cast<double>(_include.size());
        } else {
            // Read pre-built per-chr GRM from disk (fast: no genotype I/O or DSYRK).
            const std::string grm_chr_file = grm_chr_prefix + std::to_string(chrs[c1]);
            LOGGER << "  Reading per-chr GRM from [" << grm_chr_file << "] ..." << std::endl;
            std::vector<std::string> grm_chr_ids;
            read_grm(grm_chr_file, grm_chr_ids, /*out_id_log=*/false,
                     /*read_id_only=*/false, /*dont_read_N=*/false);
            if (_grm_N.size() == 0)
                LOGGER.e(0, "--grm-chr: N file [" + grm_chr_file + ".grm.N.bin] missing or "
                            "empty. Rebuild the per-chr GRM without --no-grm-N.");
            m_c = static_cast<double>(_grm_N(0, 0));
            // Match analysis individuals (uni_id) to the per-chr GRM IDs.
            // Usually identity (same bfile for all GRM builds), but we handle the
            // general case to be safe.
            std::vector<int> kp_chr;
            StrFunc::match(uni_id, grm_chr_ids, kp_chr);
            for (size_t ii = 0; ii < _n; ii++) {
                if (kp_chr[ii] < 0)
                    LOGGER.e(0, "--grm-chr chr" + std::to_string(chrs[c1])
                                + ": individual [" + uni_id[ii]
                                + "] not found in per-chr GRM. Ensure the per-chr GRM was "
                                  "built from the same sample as --grm.");
            }
            // Symmetrize _grm (lower-triangular) and subset to analysis individuals.
            eigenMatrix grm_c_full(_grm.selfadjointView<Eigen::Lower>());
            _grm.resize(0, 0);
            _grm_N.resize(0, 0);
            // Check whether kp_chr is the identity permutation (common case).
            bool id_kp = true;
            for (size_t ii = 0; ii < _n && id_kp; ii++)
                if (kp_chr[ii] != static_cast<int>(ii)) id_kp = false;
            // Store in _grm as a symmetric matrix (lower+upper) for the subtraction below.
            if (id_kp) {
                _grm = grm_c_full;
            } else {
                Eigen::Map<const Eigen::VectorXi> kp_chr_idx(kp_chr.data(),
                                                              static_cast<Eigen::Index>(_n));
                _grm = grm_c_full(kp_chr_idx, kp_chr_idx);
            }
        }

        const double m_loco = m_all - m_c;
        if (m_loco <= 0.0)
            LOGGER.e(0, "Chr " + std::to_string(chrs[c1]) + " has " +
                        std::to_string(static_cast<long long>(m_c)) +
                        " markers ≥ total markers in G_all (" +
                        std::to_string(static_cast<long long>(m_all)) + "). "
                        "The --grm file must cover all autosomes.");

        // G_loco_c = (G_all·m_all − G_chr_c·m_c) / m_loco
        // For the make_grm path: _grm is lower-triangular; selfadjointView symmetrizes.
        // For the --grm-chr path: _grm is already fully symmetric (stored above).
        // Either way: free _grm before _A[0] to keep peak at 3×n²×8.
        {
            eigenMatrix grm_c_sym(_grm.selfadjointView<Eigen::Lower>());
            _grm.resize(0, 0);   // free G_chr_c before _A[0] to cap peak at 3×n²×8
            _A[0] = (grm_all * m_all - grm_c_sym * m_c) / m_loco;
        }

        // Compute a fresh Woodbury basis for this chromosome's G_loco_c.
        // compute_woodbury_basis() consumes _A[0] (frees it internally) and
        // stores the top-k eigenvectors in _Uk / eigenvalues in _dk.
        // Warm-start: _Uk from the previous chr is used as the initial sketch
        // inside compute_woodbury_basis (see est_hsq.cpp warm-start logic), so
        // power-iteration converges in ~1 step rather than ~3 for chrs 2..n_chr.
        if (_woodbury_rank != 0) {
            if (_woodbury_rank > 0)
                compute_woodbury_basis(_woodbury_rank, 0.0, 0);
            else  // auto-k
                compute_woodbury_basis(0, _woodbury_buffer_factor, _woodbury_k_max);
            // _A[0] is now freed; reml() will use the Woodbury path.
        }

        // Run REML.  Variance components from the previous chr seed this one.
        reml(false, true, true, reml_priors, reml_priors_var,
             -2.0, -2.0, no_constrain, /*no_lrt=*/true, /*mlmassoc=*/true);

        if (!no_adj_covar) y_buf = _y.array() - (_X * _b).array();
        for (i = 0; i < _n; i++) y[i] = static_cast<float>(y_buf[i]);
        reml_priors.clear();
        reml_priors_var = _varcmp;
        _P.resize(0, 0);
        if (_A[0].size() > 0) _A[0].resize(0, 0);   // no-op if Woodbury already freed it

        auto [b, s, p] = no_adj_covar
            ? mlma_calcu_stat_covar(std::span<const float>(y), icld_chrs[c1].size())
            : mlma_calcu_stat(std::span<const float>(y), icld_chrs[c1].size());

        // Stream this chromosome's results immediately so output is written
        // progressively and peak RAM is not inflated by large result buffers.
        for (i = 0; i < icld_chrs[c1].size(); i++) {
            const int j = icld_chrs[c1][i];
            ofile << _chr[j] << "\t" << _snp_name[j] << "\t" << _bp[j] << "\t"
                  << _allele_alt[j] << "\t" << _allele_ref[j] << "\t";
            const double freq_j = (!_additive_mu.empty() ? _additive_mu[j] : _mu[j]);
            if (p[i] > 1.5)
                ofile << "NA\tNA\tNA\tNA\n";
            else
                ofile << 0.5 * freq_j << "\t" << b[i] << "\t"
                      << s[i] << "\t" << p[i] << "\n";
        }

        _include      = include_o;
        _snp_name_map = snp_name_map_o;
        LOGGER << "-----------------------------------" << std::endl;
    }

    // ---- 7. Finalise output ----
    ofile.close();
    LOGGER << "\n[LOCO] Results saved to [" << filename << "]" << std::endl;
    _make_XMat_no_compact = false;   // restore for any subsequent analyses
    LOGGER << "[LOCO] Done." << std::endl;
}

void gcta::grm_minus_grm(float *grm, float *sub_grm)
{
    int i=0, j=0, k=0, n=_n;
    
    #pragma omp parallel for private(j,k)
    for(i=0; i<n; i++){
		for(j=0; j<=i; j++){
            k=i*n+j;
            sub_grm[k]=grm[k]-sub_grm[k];
		}
	}

}

void gcta::save_reml_state(const std::string& filename, bool no_adj_covar)
{
    // Ensure _Vi is materialised regardless of which REML path was taken:
    //   - Default path:    _Vi_use_llt == false, _Vi is already an n×n matrix.
    //   - Hutch++ path:    _Vi_use_llt == true, _Vi was released to save RAM.
    //                      _Vi_L holds L; dpotri on a copy gives V^{-1}.
    //   - Woodbury path:   _Vi_use_woodbury == true; V^{-1} is stored implicitly
    //                      as (I − U_k diag(c_k) U_k^T) / σ²_eff.
    //                      Materialise it explicitly here before packing.
    // ---- Woodbury path: compact "TUNA" format ----
    // V^{-1} is never needed as a dense matrix in the Woodbury MLMA path —
    // mlma_calcu_stat{,_covar} use _Uk/_dk/_ck/_sigma2_eff directly.
    // Save just the Woodbury factors (~n*k floats) instead of the packed
    // lower triangle (~n²/2 floats), saving ~100× in file size.
    if (_Vi_use_woodbury) {
        std::ofstream outfile(filename, std::ios::binary);
        if(!outfile.is_open()) LOGGER.e(0, "cannot open the file ["+filename+"] to write.");

        struct WBHeader {
            char    magic[4]    = {'T','U','N','A'};
            int32_t n           = 0;
            int32_t x_c         = 0;
            int32_t num_varcmp  = 0;
            int32_t num_r_indx  = 0;
            int32_t woodbury_k  = 0;
        } hdr;
        hdr.n          = static_cast<int32_t>(_n);
        hdr.x_c        = static_cast<int32_t>(_X_c);
        hdr.num_varcmp = static_cast<int32_t>(_varcmp.size());
        hdr.num_r_indx = static_cast<int32_t>(_r_indx.size());
        hdr.woodbury_k = static_cast<int32_t>(_woodbury_rank);
        outfile.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        // lambda_tail (double — keep full precision; small)
        outfile.write(reinterpret_cast<const char*>(&_lambda_tail), sizeof(double));

        // _Uk stored as k×n (transposed) for cache-efficient GEMM in mlma-stream hot path.
        {
            Eigen::MatrixXf Uk_f = _Uk.transpose().cast<float>();
            const size_t sz = static_cast<size_t>(_n) * _woodbury_rank;
            outfile.write(reinterpret_cast<const char*>(Uk_f.data()),
                          static_cast<std::streamsize>(sz * sizeof(float)));
        }

        // _dk (k, float)
        {
            Eigen::VectorXf dk_f = _dk.cast<float>();
            outfile.write(reinterpret_cast<const char*>(dk_f.data()),
                          static_cast<std::streamsize>(_woodbury_rank * sizeof(float)));
        }

        // _b (float, only when covariates are present)
        if(!no_adj_covar) {
            Eigen::VectorXf b_f = _b.cast<float>();
            outfile.write(reinterpret_cast<const char*>(b_f.data()),
                          static_cast<std::streamsize>(hdr.x_c * sizeof(float)));
        }

        // _varcmp (float)
        {
            Eigen::VectorXf vc_f =
                Eigen::Map<const Eigen::VectorXd>(_varcmp.data(), hdr.num_varcmp).cast<float>();
            outfile.write(reinterpret_cast<const char*>(vc_f.data()),
                          static_cast<std::streamsize>(hdr.num_varcmp * sizeof(float)));
        }

        // _r_indx (int32)
        outfile.write(reinterpret_cast<const char*>(_r_indx.data()),
                      static_cast<std::streamsize>(hdr.num_r_indx * sizeof(int)));

        outfile.flush();
        if(!outfile)
            LOGGER.e(0, "write error on ["+filename+"]: disk full or I/O failure.");

        const size_t wb_bytes  = static_cast<size_t>(_n) * _woodbury_rank * sizeof(float);
        const size_t tri_bytes = static_cast<size_t>(_n) * (_n + 1) / 2 * sizeof(float);
        LOGGER << "Saved Woodbury REML state (n=" << _n << ", k=" << _woodbury_rank
               << ", covariates=" << _X_c << ", variance components=" << hdr.num_varcmp
               << ") to [" << filename << "] ("
               << (wb_bytes >> 20) << " MiB factors vs "
               << (tri_bytes >> 20) << " MiB dense triangle)." << std::endl;
        return;
    }

    // ---- Dense path: "GOBY" packed-triangle format ----
    if (_Vi_use_llt) {
        _Vi.swap(_Vi_L);  // O(1): _Vi gets L's storage, _Vi_L becomes empty
        gcta_blas_int blas_n_s = static_cast<gcta_blas_int>(_n);
        if (gcta_dpotri(blas_n_s, _Vi.data(), blas_n_s) != 0)
            LOGGER.e(0, "dpotri failed materialising V^{-1} in save_reml_state.");
        _Vi.triangularView<Eigen::Upper>() = _Vi.transpose();
        _Vi_use_llt = false;
    }

    std::ofstream outfile(filename, std::ios::binary);
    if(!outfile.is_open()) LOGGER.e(0, "cannot open the file ["+filename+"] to write.");

    // ---- Header ----
    // Magic "GOBY": packed-triangle symmetric format.
    struct Header {
        char    magic[4]    = {'G','O','B','Y'};
        int32_t n           = 0;
        int32_t x_c         = 0;
        int32_t num_varcmp  = 0;
        int32_t num_r_indx  = 0;
    } hdr;
    hdr.n          = static_cast<int32_t>(_n);
    hdr.x_c        = static_cast<int32_t>(_X_c);
    hdr.num_varcmp = static_cast<int32_t>(_varcmp.size());
    hdr.num_r_indx = static_cast<int32_t>(_r_indx.size());
    outfile.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // ---- _Vi: packed lower triangle (float) ----
    // _Vi is symmetric positive-definite, so only the n(n+1)/2 lower-triangle
    // elements are unique.  We save those directly rather than the full n×n matrix.
    //
    // We save _Vi (the inverse) rather than the LLT factor L for two reasons:
    //   1. On the default (non-Hutch++) path _Vi_llt is not reliably populated
    //      after convergence — only _Vi is guaranteed to exist.
    //   2. Serialising L as float and reconstructing L Lᵀ introduces squared
    //      rounding errors before the re-factorisation, degrading numerical quality.
    {
        const int32_t n = hdr.n;
        const size_t tri_size = static_cast<size_t>(n) * (n + 1) / 2;
        std::vector<float> tri(tri_size);
        size_t idx = 0;
        for(int32_t j = 0; j < n; ++j)
            for(int32_t i = j; i < n; ++i)
                tri[idx++] = static_cast<float>(_Vi(i, j));
        outfile.write(reinterpret_cast<const char*>(tri.data()),
                      static_cast<std::streamsize>(tri_size * sizeof(float)));
    }

    // ---- _b (float, only when covariates are present) ----
    if(!no_adj_covar) {
        Eigen::VectorXf b_f = _b.cast<float>();
        outfile.write(reinterpret_cast<const char*>(b_f.data()),
                      static_cast<std::streamsize>(hdr.x_c * sizeof(float)));
    }

    // ---- _varcmp (float) ----
    {
        Eigen::VectorXf vc_f =
            Eigen::Map<const Eigen::VectorXd>(_varcmp.data(), hdr.num_varcmp).cast<float>();
        outfile.write(reinterpret_cast<const char*>(vc_f.data()),
                      static_cast<std::streamsize>(hdr.num_varcmp * sizeof(float)));
    }

    // ---- _r_indx (int32) ----
    outfile.write(reinterpret_cast<const char*>(_r_indx.data()),
                  static_cast<std::streamsize>(hdr.num_r_indx * sizeof(int)));

    outfile.flush();
    if(!outfile)
        LOGGER.e(0, "write error on ["+filename+"]: disk full or I/O failure.");

    const size_t tri_bytes  = static_cast<size_t>(hdr.n) * (hdr.n + 1) / 2 * sizeof(float);
    const size_t full_bytes = static_cast<size_t>(hdr.n) * hdr.n * sizeof(float);
    LOGGER << "Saved REML state (n=" << _n << ", covariates=" << _X_c
           << ", variance components=" << hdr.num_varcmp << ") to [" << filename << "] ("
           << tri_bytes / (1<<20) << " MiB vs " << full_bytes / (1<<20) << " MiB dense)." << std::endl;
}

void gcta::load_reml_state(const std::string& filename, bool no_adj_covar)
{
    std::ifstream infile(filename, std::ios::binary);
    if(!infile.is_open()) LOGGER.e(0, "cannot open the file ["+filename+"] to read. "
        "Make sure you have run --mlma --save-reml first with matching --out prefix.");

    auto must_read = [&](void* dst, std::streamsize n){
        infile.read(reinterpret_cast<char*>(dst), n);
        if(infile.gcount() != n)
            LOGGER.e(0, "unexpected end of file reading ["+filename+"] — file may be truncated.");
    };

    // ---- Header ----
    struct Header {
        char    magic[4];
        int32_t n, x_c, num_varcmp, num_r_indx;
    } hdr;
    must_read(&hdr, sizeof(hdr));

    // ---- Woodbury "TUNA" format ----
    if(std::string_view(hdr.magic, 4) == "TUNA") {
        // The Woodbury-format header has one extra int32 (woodbury_k).
        int32_t woodbury_k = 0;
        must_read(&woodbury_k, sizeof(int32_t));

        if(hdr.n <= 0 || hdr.x_c < 0 || hdr.num_varcmp <= 0 || hdr.num_r_indx <= 0 || woodbury_k <= 0)
            LOGGER.e(0, "file ["+filename+"] contains invalid Woodbury dimensions — file may be corrupt.");
        if(hdr.n != static_cast<int32_t>(_n))
            LOGGER.e(0, "sample size mismatch: REML state has n=" + std::to_string(hdr.n)
                      + " but current dataset has n=" + std::to_string(_n));
        if(hdr.x_c != static_cast<int32_t>(_X_c))
            LOGGER.e(0, "number of covariates mismatch: REML state has "
                      + std::to_string(hdr.x_c) + " but current dataset has " + std::to_string(_X_c));

        // lambda_tail (double)
        must_read(&_lambda_tail, sizeof(double));

        // _Uk: on-disk layout is k×n (transposed, matching save_reml_state).
        // Read into k×n, then transpose to n×k for _Uk.
        {
            Eigen::MatrixXf Uk_kn(woodbury_k, hdr.n);
            must_read(Uk_kn.data(),
                      static_cast<std::streamsize>(static_cast<size_t>(hdr.n) * woodbury_k * sizeof(float)));
            _Uk = Uk_kn.transpose().cast<eigenMatrix::Scalar>();
        }

        // _dk (k, float → eigenVector)
        {
            Eigen::VectorXf dk_f(woodbury_k);
            must_read(dk_f.data(),
                      static_cast<std::streamsize>(woodbury_k * sizeof(float)));
            _dk = dk_f.cast<eigenVector::Scalar>();
        }

        // _b
        if(!no_adj_covar) {
            Eigen::VectorXf b_f(hdr.x_c);
            must_read(b_f.data(), static_cast<std::streamsize>(hdr.x_c * sizeof(float)));
            _b = b_f.cast<double>();
        }

        // _varcmp
        {
            Eigen::VectorXf vc_f(hdr.num_varcmp);
            must_read(vc_f.data(), static_cast<std::streamsize>(hdr.num_varcmp * sizeof(float)));
            _varcmp.resize(hdr.num_varcmp);
            Eigen::Map<Eigen::VectorXd>(_varcmp.data(), hdr.num_varcmp) = vc_f.cast<double>();
        }

        // _r_indx
        _r_indx.resize(hdr.num_r_indx);
        must_read(_r_indx.data(),
                  static_cast<std::streamsize>(hdr.num_r_indx * sizeof(int)));

        // Recompute _sigma2_eff and _ck from the loaded _varcmp + Woodbury basis
        _woodbury_rank = woodbury_k;
        const double sg2 = _varcmp[0];
        const double se2 = _varcmp.back();
        _sigma2_eff = sg2 * _lambda_tail + se2;
        _ck.resize(woodbury_k);
        for (int j = 0; j < woodbury_k; ++j) {
            const double delta    = std::max(0.0, static_cast<double>(_dk[j]) - _lambda_tail);
            const double sig_delta = sg2 * delta;
            _ck[j] = static_cast<eigenVector::Scalar>(sig_delta / (_sigma2_eff + sig_delta));
        }
        _Vi_use_woodbury = true;

        LOGGER << "Loaded Woodbury REML state from [" << filename << "]: n=" << hdr.n
               << ", k=" << woodbury_k << ", covariates=" << hdr.x_c
               << ", variance components=" << hdr.num_varcmp << std::endl;
        return;
    }

    // ---- Dense "GOBY" packed-triangle format ----
    if(std::string_view(hdr.magic, 4) != "GOBY")
        LOGGER.e(0, "file ["+filename+"] is not a valid REML state file.");
    if(hdr.n <= 0 || hdr.x_c < 0 || hdr.num_varcmp <= 0 || hdr.num_r_indx <= 0)
        LOGGER.e(0, "file ["+filename+"] contains invalid dimensions — file may be corrupt.");
    if(hdr.n != static_cast<int32_t>(_n))
        LOGGER.e(0, "sample size mismatch: REML state has n=" + std::to_string(hdr.n)
                  + " but current dataset has n=" + std::to_string(_n));
    if(hdr.x_c != static_cast<int32_t>(_X_c))
        LOGGER.e(0, "number of covariates mismatch: REML state has "
                  + std::to_string(hdr.x_c) + " but current dataset has " + std::to_string(_X_c));

    // ---- _Vi: packed lower triangle → full symmetric matrix ----
    // Read float triangle, upcast to double, and reconstruct the full n×n _Vi
    // by reflecting the lower triangle into the upper half.
    {
        const int32_t n = hdr.n;
        const size_t tri_size = static_cast<size_t>(n) * (n + 1) / 2;
        std::vector<float> tri(tri_size);
        must_read(tri.data(), static_cast<std::streamsize>(tri_size * sizeof(float)));

        _Vi.resize(n, n);
        size_t idx = 0;
        for(int32_t j = 0; j < n; ++j) {
            for(int32_t i = j; i < n; ++i) {
                const double v = static_cast<double>(tri[idx++]);
                _Vi(i, j) = v;
                _Vi(j, i) = v;   // reflect to upper triangle
            }
        }
        _Vi_use_llt = false;
    }

    // ---- _b ----
    if(!no_adj_covar) {
        Eigen::VectorXf b_f(hdr.x_c);
        must_read(b_f.data(), static_cast<std::streamsize>(hdr.x_c * sizeof(float)));
        _b = b_f.cast<double>();
    }

    // ---- _varcmp ----
    {
        Eigen::VectorXf vc_f(hdr.num_varcmp);
        must_read(vc_f.data(), static_cast<std::streamsize>(hdr.num_varcmp * sizeof(float)));
        _varcmp.resize(hdr.num_varcmp);
        Eigen::Map<Eigen::VectorXd>(_varcmp.data(), hdr.num_varcmp) = vc_f.cast<double>();
    }

    // ---- _r_indx ----
    _r_indx.resize(hdr.num_r_indx);
    must_read(_r_indx.data(),
              static_cast<std::streamsize>(hdr.num_r_indx * sizeof(int)));

    LOGGER << "Loaded REML state from [" << filename << "]: n=" << hdr.n
           << ", covariates=" << hdr.x_c
           << ", variance components=" << hdr.num_varcmp << std::endl;
}
