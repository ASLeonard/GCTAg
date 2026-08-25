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
#include <cstdint>

void gcta::mlma(string grm_file, bool m_grm_flag, string subtract_grm_file, string phen_file, string qcovar_file, string covar_file, int mphen, int MaxIter, vector<double> reml_priors, vector<double> reml_priors_var, bool no_constrain, bool within_family, bool inbred, bool no_adj_covar)
{
    _within_family=within_family;
    _reml_max_iter=MaxIter;
    unsigned long i = 0, j = 0, k = 0;
    bool grm_flag=(!grm_file.empty());
    bool qcovar_flag=(!qcovar_file.empty());
    bool covar_flag=(!covar_file.empty());
    if (!qcovar_flag && !covar_flag) no_adj_covar=false;
    if (m_grm_flag) grm_flag = false;
    bool subtract_grm_flag = (!subtract_grm_file.empty());
    if (subtract_grm_flag && m_grm_flag) LOGGER.e(0, "the --mlma-subtract-grm option cannot be used in combination with the --mgrm option.");
    
    // Read data
    int qcovar_num=0, covar_num=0;
    vector<string> phen_ID, qcovar_ID, covar_ID, grm_id;
    vector< vector<string> > phen_buf, qcovar, covar; // save individuals by column
    vector<string> grm_files;
    
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
            read_grm(grm_file, grm_id, true, false, true);
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
            make_grm_mkl(false, false, inbred, true, 0, true);
            for(i=0; i<_keep.size(); i++) grm_id.push_back(_fid[_keep[i]]+":"+_pid[_keep[i]]);
        }
    }
    
    vector<string> uni_id;
	map<string, int> uni_id_map;
    map<string, int>::iterator iter;
	for(i=0; i<_keep.size(); i++){
	    uni_id.push_back(_fid[_keep[i]]+":"+_pid[_keep[i]]);
	    uni_id_map.insert(pair<string,int>(_fid[_keep[i]]+":"+_pid[_keep[i]], i));
	}
    _n=_keep.size();
    if(_n<1) LOGGER.e(0, "no individual is in common in the input files.");
    LOGGER<<_n<<" individuals are in common in these files."<<endl;
    
    // construct model terms
    _y.setZero(_n);
    for(i=0; i<phen_ID.size(); i++){
        iter=uni_id_map.find(phen_ID[i]);
        if(iter==uni_id_map.end()) continue;
        _y[iter->second]=atof(phen_buf[i][mphen-1].c_str());
    }

    _r_indx.clear();
    vector<int> kp;
    if (subtract_grm_flag) {
        for(i=0; i < 2; i++) _r_indx.push_back(i);
        _A.resize(_r_indx.size());

        LOGGER << "\nReading the primary GRM from [" << grm_files[1] << "] ..." << endl;
        read_grm(grm_files[1], grm_id, true, false, false);

        StrFunc::match(uni_id, grm_id, kp);
        (_A[0]).resize(_n, _n);
        MatrixXf A_N_buf(_n, _n);
        #pragma omp parallel for private(k)
        for (j = 0; j < _n; j++) {
            for (k = 0; k <= j; k++) {
                if (kp[j] >= kp[k]){
                    (_A[0])(k, j) = (_A[0])(j, k) = _grm(kp[j], kp[k]);
                    A_N_buf(k, j) = A_N_buf(j, k) = _grm_N(kp[j], kp[k]);
                }
                else{
                    (_A[0])(k, j) = (_A[0])(j, k) = _grm(kp[k], kp[j]);
                    A_N_buf(k, j) = A_N_buf(j, k) = _grm_N(kp[k], kp[j]);
                }
            }
        }

        LOGGER << "\nReading the secondary GRM from [" << grm_files[0] << "] ..." << endl;
        read_grm(grm_files[0], grm_id, true, false, false);
        LOGGER<<"\nSubtracting [" << grm_files[1] << "] from [" << grm_files[0] << "] ..." << endl;
        StrFunc::match(uni_id, grm_id, kp);
        #pragma omp parallel for private(k)
        for (j = 0; j < _n; j++) {
            for (k = 0; k <= j; k++) {
                if (kp[j] >= kp[k]) (_A[0])(k, j) = (_A[0])(j, k) = ((_A[0])(j, k) * A_N_buf(j, k)  - _grm(kp[j], kp[k]) * _grm_N(kp[j], kp[k])) / (A_N_buf(j, k) - _grm_N(kp[j], kp[k]));
                else (_A[0])(k, j) = (_A[0])(j, k) = ((_A[0])(j, k) * A_N_buf(j, k) - _grm(kp[k], kp[j]) * _grm_N(kp[k], kp[j])) / (A_N_buf(j, k) - _grm_N(kp[k], kp[j]));
            }
        }
        _grm.resize(0,0);
        _grm_N.resize(0,0);
    }
    else {
        for(i=0; i < grm_files.size() + 1; i++) _r_indx.push_back(i);
        _A.resize(_r_indx.size());
        if(grm_flag){
            StrFunc::match(uni_id, grm_id, kp);
            (_A[0]).resize(_n, _n);
            #pragma omp parallel for private(j)
            for(i=0; i<_n; i++){
                for(j=0; j<=i; j++) (_A[0])(j,i)=(_A[0])(i,j)=_grm(kp[i],kp[j]);
            }
            _grm.resize(0,0);
        }
        else if(m_grm_flag){
            LOGGER << "There are " << grm_files.size() << " GRM file names specified in the file [" + grm_file + "]." << endl;
            for (i = 0; i < grm_files.size(); i++) {
                LOGGER << "Reading the GRM from the " << i + 1 << "th file ..." << endl;
                read_grm(grm_files[i], grm_id, true, false, true);
                StrFunc::match(uni_id, grm_id, kp);
                (_A[i]).resize(_n, _n);
                #pragma omp parallel for private(k)
                for (j = 0; j < _n; j++) {
                    for (k = 0; k <= j; k++) {
                        if (kp[j] >= kp[k]) (_A[i])(k, j) = (_A[i])(j, k) = _grm(kp[j], kp[k]);
                        else (_A[i])(k, j) = (_A[i])(j, k) = _grm(kp[k], kp[j]);
                    }
                }
            }
        }
        else{
            StrFunc::match(uni_id, grm_id, kp);
            (_A[0]).resize(_n, _n);
            #pragma omp parallel for private(j)
            for(i=0; i<_n; i++){
                for(j=0; j<=i; j++) (_A[0])(j,i)=(_A[0])(i,j)=_grm_mkl[kp[i]*_n+kp[j]];
            }
            delete[] _grm_mkl;
        }
    }
    _A[_r_indx.size()-1]=eigenMatrix::Identity(_n, _n);
    
    // construct X matrix
    vector<eigenMatrix> E_float;
    eigenMatrix qE_float;
    construct_X(_n, uni_id_map, qcovar_flag, qcovar_num, qcovar_ID, qcovar, covar_flag, covar_num, covar_ID, covar, E_float, qE_float);
    
    // names of variance component
    for (i = 0; i < grm_files.size(); i++) {
        stringstream strstrm;
        if (grm_files.size() == 1) strstrm << "";
        else strstrm << i + 1;
        _var_name.push_back("V(G" + strstrm.str() + ")");
        _hsq_name.push_back("V(G" + strstrm.str() + ")/Vp");
    }
    _var_name.push_back("V(e)");
    
    // within family
    if(_within_family) detect_family();
    
    // run REML algorithm
    LOGGER << "\nPerforming MLM association analyses" << (subtract_grm_flag?"":" (including the candidate SNP)") << " ..."<<endl;
    unsigned long n=_keep.size(), m=_include.size();
	reml(false, true, true, reml_priors, reml_priors_var, -2.0, -2.0, no_constrain, true, true);
    _P.resize(0,0);
    _A.clear();
    float *y=new float[n];
    eigenVector y_buf=_y;
    if(!no_adj_covar) y_buf=_y.array()-(_X*_b).array(); // adjust phenotype for covariates
    for(i=0; i<n; i++) y[i]=y_buf[i];
    
/*    if(grm_flag || m_grm_flag){
        LOGGER<<endl;
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
    
    if (_mu.empty()) calcu_mu();
    eigenVector beta, se, pval;
    if(no_adj_covar) mlma_calcu_stat_covar(y, _geno_mkl, n, m, beta, se, pval);
    else mlma_calcu_stat(y, _geno_mkl, n, m, beta, se, pval);
    delete[] y;
    delete[] _geno_mkl;
    
    string filename=_out+".mlma";
    LOGGER<<"\nSaving the results of the mixed linear model association analyses of "<<m<<" SNPs to ["+filename+"] ..."<<endl;
    ofstream ofile(filename.c_str());
    if(!ofile) LOGGER.e(0, "cannot open the file ["+filename+"] to write.");
    ofile<<"Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\tp"<<endl;
	for(i=0; i<m; i++){
        j=_include[i];
        ofile<<_chr[j]<<"\t"<<_snp_name[j]<<"\t"<<_bp[j]<<"\t"<<_ref_A[j]<<"\t"<<_other_A[j]<<"\t";
        if(pval[i]>1.5) ofile<<"NA\tNA\tNA\tNA"<<endl;
        else ofile<<0.5*_mu[j]<<"\t"<<beta[i]<<"\t"<<se[i]<<"\t"<<pval[i]<<endl;
    }
    ofile.close();
}

void gcta::mlma_calcu_stat(float *y, float *geno_mkl, unsigned long n, unsigned long m, eigenVector &beta, eigenVector &se, eigenVector &pval)
{
    int max_block_size = 10000;
    unsigned long i=0, j=0;
    double Xt_Vi_X=0.0, chisq=0.0;
    float *X=new float[n];
    float *Vi_X=new float[n];
    float *Vi=new float[n*n];
    #pragma omp parallel for private(j)
    for(i=0; i<n; i++){
        for(j=0; j<n; j++) Vi[i*n+j]=_Vi(i,j);
    }
    _Vi.resize(0,0);
    
    beta.resize(m);
    se=eigenVector::Zero(m);
    pval=eigenVector::Constant(m,2);
    LOGGER<<"\nRunning association tests for "<<m<<" SNPs ..."<<endl;
    int new_start = 0, block_size = 0, block_col = 0, k = 0, l = 0;
    MatrixXf X_block;
    vector<int> indx;
    for(i = 0; i < m; i++, block_col++){
        // get a block of SNPs
        if(i == new_start){
            block_col = 0;
            new_start = i + max_block_size;
            block_size = max_block_size;
            if(new_start > m) block_size = m - i;
            indx.resize(block_size);
            for(k = i, l = 0; l < block_size; k++, l++) indx[l] = k;
            make_XMat_subset(X_block, indx, false);
        }

        for(j = 0; j < n; j++) X[j] = X_block(j, block_col);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, Vi, n, X, 1, 0.0, Vi_X, 1);
        Xt_Vi_X=cblas_sdot(n, X, 1, Vi_X, 1);
        se[i]=1.0/Xt_Vi_X;
        beta[i]=se[i]*cblas_sdot(n, y, 1, Vi_X, 1);
        if(se[i]>1.0e-30){
            se[i]=sqrt(se[i]);
            chisq=beta[i]/se[i];
            pval[i]=StatFunc::pchisq(chisq*chisq, 1);
        }
    }
    delete[] X;
    delete[] Vi_X;
    delete[] Vi;
}

void gcta::mlma_calcu_stat_covar(float *y, float *geno_mkl, unsigned long n, unsigned long m, eigenVector &beta, eigenVector &se, eigenVector &pval)
{
    int max_block_size = 10000;
    unsigned long i=0, j=0, col_num=_X_c+1;
    double chisq=0.0, d_buf=0.0;
    float *Vi=new float[n*n];
    float *X=new float[n*col_num];
    float *Vi_X=new float[n*col_num];
    float *Xt_Vi_X=new float[col_num*col_num];
    float *Xt_Vi_y=new float[col_num];
    float *b_vec=new float[col_num];
    #pragma omp parallel for private(j)
    for(i=0; i<n; i++){
        for(j=0; j<n; j++) Vi[i*n+j]=_Vi(i,j);
    }
    _Vi.resize(0,0);
    for(i=0; i<n; i++){
        for(j=0; j<_X_c; j++) X[i*col_num+j]=_X(i,j);
        X[i*col_num+_X_c]=0.0;
    }

    beta.resize(m);
    se=eigenVector::Zero(m);
    pval=eigenVector::Constant(m,2);
    LOGGER<<"\nRunning association tests for "<<m<<" SNPs ..."<<endl;
    int new_start = 0, block_size = 0, block_col = 0, k = 0, l = 0;
    MatrixXf X_block;
    vector<int> indx;
    for(i = 0; i < m; i++, block_col++){
        // get a block of SNPs
        if(i == new_start){
            block_col = 0;
            new_start = i + max_block_size;
            block_size = max_block_size;
            if(new_start > m) block_size = m - i;
            indx.resize(block_size);
            for(k = i, l = 0; l < block_size; k++, l++) indx[l] = k;
            make_XMat_subset(X_block, indx, false);
        }

        for(j = 0; j < n; j++) X[j*col_num+_X_c] = X_block(j, block_col);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, col_num, n, 1.0, Vi, n, X, col_num, 0.0, Vi_X, col_num);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, col_num, col_num, n, 1.0, X, col_num, Vi_X, col_num, 0.0, Xt_Vi_X, col_num);
        if(!comput_inverse_logdet_LU_mkl_array(col_num, Xt_Vi_X, d_buf)) LOGGER.e(0, "Xt_Vi_X is not invertible.");
        cblas_sgemv(CblasRowMajor, CblasTrans, n, col_num, 1.0, Vi_X, col_num, y, 1, 0.0, Xt_Vi_y, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, col_num, col_num, 1.0, Xt_Vi_X, col_num, Xt_Vi_y, 1, 0.0, b_vec, 1);
        se[i]=Xt_Vi_X[_X_c*col_num+_X_c];
        beta[i]=b_vec[_X_c];
        if(se[i]>1.0e-30){
            se[i]=sqrt(se[i]);
            chisq=beta[i]/se[i];
            pval[i]=StatFunc::pchisq(chisq*chisq, 1);
        }
    }
    delete[] Vi;
    delete[] X;
    delete[] Vi_X;
    delete[] Xt_Vi_X;
    delete[] Xt_Vi_y;
    delete[] b_vec;
}

void gcta::mlma_loco(string phen_file, string qcovar_file, string covar_file, int mphen, int MaxIter, vector<double> reml_priors, vector<double> reml_priors_var, bool no_constrain, bool inbred, bool no_adj_covar)
{
    unsigned long i=0, j=0, k=0, c1=0, c2=0, n=0;
    _reml_max_iter=MaxIter;
    bool qcovar_flag=(!qcovar_file.empty());
    bool covar_flag=(!covar_file.empty());
    if(!qcovar_flag && !covar_flag) no_adj_covar=false;
    
    // Read data
    int qcovar_num=0, covar_num=0;
    vector<string> phen_ID, qcovar_ID, covar_ID, grm_id;
    vector< vector<string> > phen_buf, qcovar, covar; // save individuals by column

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
    LOGGER<<_n<<" individuals are in common in these files."<<endl;
    
    vector<int> chrs, vi_buf(_chr);
    stable_sort(vi_buf.begin(), vi_buf.end());
	vi_buf.erase(unique(vi_buf.begin(), vi_buf.end()), vi_buf.end());
    if(vi_buf.size()<2) LOGGER.e(0, "There is only one chromosome. The MLM leave-on-chromosome-out (LOCO) analysis requires at least two chromosomes.");
    for(i=0; i<vi_buf.size(); i++){
        if(vi_buf[i]<=_autosome_num) chrs.push_back(vi_buf[i]);
    }
    vector<int> include_o(_include);
    map<string, int> snp_name_map_o(_snp_name_map);
    vector<float> m_chrs_f(chrs.size());
    vector<float *> grm_chrs(chrs.size());
    vector<float *> geno_chrs(chrs.size());
    vector< vector<int> > icld_chrs(chrs.size());
    LOGGER<<endl;
    if(_mu.empty()) calcu_mu();
    LOGGER<<"\nCalculating the genetic relationship matrix for each of the "<<chrs.size()<<" chromosomes ... "<<endl;
    for(c1=0; c1<chrs.size(); c1++){
        LOGGER<<"Chr "<<chrs[c1]<<":"<<endl;
        extract_chr(chrs[c1], chrs[c1]);
        make_grm_mkl(false, false, inbred, true, 0, true);
        
        m_chrs_f[c1]=(float)_include.size();
        icld_chrs[c1]=_include;
        _include=include_o;
        _snp_name_map=snp_name_map_o;
        
        geno_chrs[c1]=_geno_mkl;
        _geno_mkl=NULL;
        grm_chrs[c1]=_grm_mkl;
        _grm_mkl=NULL;
    }
    for(i=0; i<_keep.size(); i++) grm_id.push_back(_fid[_keep[i]]+":"+_pid[_keep[i]]);
    
    vector<string> uni_id;
	map<string, int> uni_id_map;
    map<string, int>::iterator iter;
	for(i=0; i<_keep.size(); i++){
	    uni_id.push_back(_fid[_keep[i]]+":"+_pid[_keep[i]]);
	    uni_id_map.insert(pair<string,int>(_fid[_keep[i]]+":"+_pid[_keep[i]], i));
	}
    
    // construct model terms
    _y.setZero(_n);
    for(i=0; i<phen_ID.size(); i++){
        iter=uni_id_map.find(phen_ID[i]);
        if(iter==uni_id_map.end()) continue;
        _y[iter->second]=atof(phen_buf[i][mphen-1].c_str());
    }
    
    // construct X matrix
    vector<eigenMatrix> E_float;
    eigenMatrix qE_float;
    construct_X(_n, uni_id_map, qcovar_flag, qcovar_num, qcovar_ID, qcovar, covar_flag, covar_num, covar_ID, covar, E_float, qE_float);
    
    // names of variance component
    _var_name.push_back("V(G)");
    _hsq_name.push_back("V(G)/Vp");
    _var_name.push_back("V(e)");
    
    // MLM association
    LOGGER<<"\nPerforming MLM association analyses (leave-one-chromosome-out) ..."<<endl;
    
    vector<int> kp;
    StrFunc::match(uni_id, grm_id, kp);
    _r_indx.resize(2);
    for(i=0; i<2; i++) _r_indx[i]=i;
    _A.resize(_r_indx.size());
    _A[1]=eigenMatrix::Identity(_n, _n);
    
    eigenVector y_buf=_y;
    float *y=new float[_n];
    vector<eigenVector> beta(chrs.size()), se(chrs.size()), pval(chrs.size());
    for(c1=0; c1<chrs.size(); c1++){
        LOGGER<<"\n-----------------------------------\n#Chr "<<chrs[c1]<<":"<<endl;
        extract_chr(chrs[c1], chrs[c1]);
        
        _A[0]=eigenMatrix::Zero(_n, _n);
        double d_buf=0;
        for(c2=0; c2<chrs.size(); c2++){
            if(chrs[c1]==chrs[c2]) continue;
            #pragma omp parallel for private(j)
            for(i=0; i<_n; i++){
                for(j=0; j<=i; j++){
                    (_A[0])(i,j)+=(grm_chrs[c2])[kp[i]*_n+kp[j]]*m_chrs_f[c2];
                }
            }
            d_buf+=m_chrs_f[c2];
        }
        
        #pragma omp parallel for private(j)
        for(i=0; i<_n; i++){
            for(j=0; j<=i; j++){
                (_A[0])(i,j)/=d_buf;
                (_A[0])(j,i)=(_A[0])(i,j);
            }
        }
        
        // run REML algorithm
        reml(false, true, true, reml_priors, reml_priors_var, -2.0, -2.0, no_constrain, true, true);
        if(!no_adj_covar) y_buf=_y.array()-(_X*_b).array(); // adjust phenotype for covariates
        for(i=0; i<_n; i++) y[i]=y_buf[i];
        reml_priors.clear();
        reml_priors_var=_varcmp;
        _P.resize(0,0);
        _A[0].resize(0,0);

        if(no_adj_covar)  mlma_calcu_stat_covar(y, (geno_chrs[c1]), n, _include.size(), beta[c1], se[c1], pval[c1]);
        else mlma_calcu_stat(y, (geno_chrs[c1]), n, _include.size(), beta[c1], se[c1], pval[c1]);
        
        _include=include_o;
        _snp_name_map=snp_name_map_o;
        LOGGER<<"-----------------------------------"<<endl;
    }
    
    delete[] y;
    for(c1=0; c1<chrs.size(); c1++){
        delete[] (grm_chrs[c1]);
        delete[] (geno_chrs[c1]);
    }
    
    string filename=_out+".loco.mlma";
    LOGGER<<"\nSaving the results of the mixed linear model association analyses of "<<_include.size()<<" SNPs to ["+filename+"] ..."<<endl;
    ofstream ofile(filename.c_str());
    if(!ofile) LOGGER.e(0, "cannot open the file ["+filename+"] to write.");
    ofile<<"Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\tp"<<endl;
    for(c1=0; c1<chrs.size(); c1++){
        for(i=0; i<icld_chrs[c1].size(); i++){
            j=icld_chrs[c1][i];
            ofile<<_chr[j]<<"\t"<<_snp_name[j]<<"\t"<<_bp[j]<<"\t"<<_ref_A[j]<<"\t"<<_other_A[j]<<"\t";
            if(pval[c1][i]>1.5) ofile<<"NA\tNA\tNA\tNA"<<endl;
            else ofile<<0.5*_mu[j]<<"\t"<<beta[c1][i]<<"\t"<<se[c1][i]<<"\t"<<pval[c1][i]<<endl;
        }
    }
    ofile.close();
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
// ---------------------------------------------------------------------
// Serialization helpers
//
// Plain binary dump, no compression, no versioned schema evolution --
// deliberately unoptimized/minimal, matching the rest of this split.
// Layout:
//   magic[8] "GCTAREML"
//   int32    format version
//   uint64   n            (individuals)
//   int32    X_c           (fixed-effect columns, from construct_X)
//   int32    n_vc          (number of variance components, _varcmp.size())
//   n x  { uint32 len, char[len] }   individual IDs, "fid:pid", in the
//                                     order rows of _Vi/_X/_y correspond to
//   lower triangle (incl. diagonal) of _Vi, column by column, double
//   n*X_c double  _X,  row-major
//   n    double   _y
//   X_c  double   _b
//   n_vc double   _varcmp
//
// _Vi is symmetric (it's V^-1, V being the fitted variance matrix), so
// only its lower triangle is stored -- this is a storage-layout choice,
// not a numerical one: the full matrix reconstructed on load is bit-
// identical to the one that would have been written in full. Values stay
// double throughout; no precision is traded away.
//
// Reading/writing goes straight from/into eigenMatrix's own column-major
// buffer (each column's lower part is already contiguous in memory), so
// there is no per-element copy loop on the hot path -- just one write()/
// read() per column.
// ---------------------------------------------------------------------

static const char REML_MAGIC[8] = {'G', 'C', 'T', 'A', 'R', 'E', 'M', 'L'};
static const int32_t REML_FMT_VERSION = 1;

// The direct-pointer column I/O below assumes eigenMatrix uses Eigen's
// default (column-major) storage. Fails to compile rather than silently
// writing garbage if that assumption is ever violated.
static_assert(!eigenMatrix::IsRowMajor, "save/read_reml_state assumes column-major eigenMatrix storage");

void gcta::save_reml_state(string filename, vector<string> &uni_id)
{
    uint64_t n = (uint64_t)_n;
    int32_t X_c = (int32_t)_X_c;
    int32_t n_vc = (int32_t)_varcmp.size();

    if ((uint64_t)uni_id.size() != n) LOGGER.e(0, "internal error: uni_id size does not match _n when saving REML state.");
    if ((uint64_t)_Vi.rows() != n || (uint64_t)_Vi.cols() != n) LOGGER.e(0, "internal error: _Vi is not n x n when saving REML state.");
    if ((uint64_t)_X.rows() != n || _X.cols() != X_c) LOGGER.e(0, "internal error: _X dimensions do not match when saving REML state.");
    if ((uint64_t)_y.size() != n) LOGGER.e(0, "internal error: _y size does not match _n when saving REML state.");

    ofstream out(filename.c_str(), ios::binary);
    if (!out) LOGGER.e(0, "cannot open the file [" + filename + "] to write.");

    out.write(REML_MAGIC, sizeof(REML_MAGIC));
    out.write(reinterpret_cast<const char *>(&REML_FMT_VERSION), sizeof(REML_FMT_VERSION));
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(&X_c), sizeof(X_c));
    out.write(reinterpret_cast<const char *>(&n_vc), sizeof(n_vc));

    for (uint64_t i = 0; i < n; i++) {
        uint32_t len = (uint32_t)uni_id[i].size();
        out.write(reinterpret_cast<const char *>(&len), sizeof(len));
        out.write(uni_id[i].data(), len);
    }

    // _Vi: lower triangle only, one column at a time. For column i,
    // rows i..n-1 are already contiguous in _Vi's own buffer (column-
    // major dense matrix), so this writes directly out of Eigen's
    // memory -- no intermediate copy.
    for (uint64_t i = 0; i < n; i++) {
        const double *col_tail = _Vi.data() + i * n + i;
        out.write(reinterpret_cast<const char *>(col_tail), (n - i) * sizeof(double));
    }

    // _X, row by row
    if (X_c > 0) {
        vector<double> row_buf(X_c);
        for (uint64_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < X_c; j++) row_buf[j] = _X(i, j);
            out.write(reinterpret_cast<const char *>(row_buf.data()), X_c * sizeof(double));
        }
    }

    // _y (contiguous)
    {
        vector<double> y_buf(n);
        for (uint64_t i = 0; i < n; i++) y_buf[i] = _y[i];
        out.write(reinterpret_cast<const char *>(y_buf.data()), n * sizeof(double));
    }

    // _b (contiguous)
    if (X_c > 0) {
        vector<double> b_buf(X_c);
        for (int32_t i = 0; i < X_c; i++) b_buf[i] = _b[i];
        out.write(reinterpret_cast<const char *>(b_buf.data()), X_c * sizeof(double));
    }

    // _varcmp
    if (n_vc > 0) {
        vector<double> vc_buf(n_vc);
        for (int32_t i = 0; i < n_vc; i++) vc_buf[i] = _varcmp[i];
        out.write(reinterpret_cast<const char *>(vc_buf.data()), n_vc * sizeof(double));
    }

    out.close();
}

// Reads a .reml file into local (non-member) buffers; the caller is
// responsible for reordering rows/cols to match the current --keep order
// and assigning into _Vi/_X/_y/_b/_varcmp.
static void read_reml_state(string filename, vector<string> &saved_ids, eigenMatrix &Vi, eigenMatrix &X, eigenVector &y, eigenVector &b, vector<double> &varcmp, int32_t &X_c)
{
    ifstream in(filename.c_str(), ios::binary);
    if (!in) LOGGER.e(0, "cannot open the file [" + filename + "] to read.");

    char magic[8];
    int32_t version;
    uint64_t n;
    int32_t n_vc;

    in.read(magic, sizeof(magic));
    if (memcmp(magic, REML_MAGIC, sizeof(magic)) != 0) LOGGER.e(0, "[" + filename + "] does not look like a GCTA .reml file.");
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != REML_FMT_VERSION) LOGGER.e(0, "[" + filename + "] was written by an incompatible .reml format version.");
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    in.read(reinterpret_cast<char *>(&X_c), sizeof(X_c));
    in.read(reinterpret_cast<char *>(&n_vc), sizeof(n_vc));

    saved_ids.resize(n);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t len;
        in.read(reinterpret_cast<char *>(&len), sizeof(len));
        string id(len, '\0');
        in.read(&id[0], len);
        saved_ids[i] = id;
    }

    // Read the lower triangle straight into Vi's own buffer (same
    // zero-copy shape as the write side), then mirror it into the
    // upper triangle so downstream code (mlma_calcu_stat and friends,
    // plus the --keep reordering step below) can keep indexing Vi(i,j)
    // for arbitrary i,j as a plain full matrix, same as before.
    Vi.resize(n, n);
    for (uint64_t i = 0; i < n; i++) {
        double *col_tail = Vi.data() + i * n + i;
        in.read(reinterpret_cast<char *>(col_tail), (n - i) * sizeof(double));
    }
    for (uint64_t i = 0; i < n; i++)
        for (uint64_t j = i + 1; j < n; j++) Vi(i, j) = Vi(j, i);

    X.resize(n, X_c);
    if (X_c > 0) {
        vector<double> row_buf(X_c);
        for (uint64_t i = 0; i < n; i++) {
            in.read(reinterpret_cast<char *>(row_buf.data()), X_c * sizeof(double));
            for (int32_t j = 0; j < X_c; j++) X(i, j) = row_buf[j];
        }
    }

    y.resize(n);
    {
        vector<double> y_buf(n);
        in.read(reinterpret_cast<char *>(y_buf.data()), n * sizeof(double));
        for (uint64_t i = 0; i < n; i++) y[i] = y_buf[i];
    }

    b.resize(X_c);
    if (X_c > 0) {
        vector<double> b_buf(X_c);
        in.read(reinterpret_cast<char *>(b_buf.data()), X_c * sizeof(double));
        for (int32_t i = 0; i < X_c; i++) b[i] = b_buf[i];
    }

    varcmp.resize(n_vc);
    if (n_vc > 0) {
        vector<double> vc_buf(n_vc);
        in.read(reinterpret_cast<char *>(vc_buf.data()), n_vc * sizeof(double));
        for (int32_t i = 0; i < n_vc; i++) varcmp[i] = vc_buf[i];
    }

    in.close();
}

// ---------------------------------------------------------------------
// Stage 1: REML fit only. Identical to gcta::mlma() up through the
// reml() call (where _Vi, _b, _X, _varcmp become available), then
// serializes that state to <_out>.reml instead of scanning SNPs.
// ---------------------------------------------------------------------

void gcta::mlma_reml_stage(string grm_file, bool m_grm_flag, string subtract_grm_file, string phen_file, string qcovar_file, string covar_file, int mphen, int MaxIter, vector<double> reml_priors, vector<double> reml_priors_var, bool no_constrain, bool within_family, bool inbred)
{
    _within_family = within_family;
    _reml_max_iter = MaxIter;
    unsigned long i = 0, j = 0, k = 0;
    bool grm_flag = (!grm_file.empty());
    bool qcovar_flag = (!qcovar_file.empty());
    bool covar_flag = (!covar_file.empty());
    if (m_grm_flag) grm_flag = false;
    bool subtract_grm_flag = (!subtract_grm_file.empty());
    if (subtract_grm_flag && m_grm_flag) LOGGER.e(0, "the --mlma-subtract-grm option cannot be used in combination with the --mgrm option.");

    // Read data
    int qcovar_num = 0, covar_num = 0;
    vector<string> phen_ID, qcovar_ID, covar_ID, grm_id;
    vector<vector<string> > phen_buf, qcovar, covar;
    vector<string> grm_files;

    if (phen_file.empty()) {
        LOGGER.e(0, "no file name in --pheno.");
    }
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
    if (_keep.size() < 1) {
        LOGGER.e(0, "no individual is in common among the input files.");
    }

    if (subtract_grm_flag) {
        grm_files.push_back(grm_file);
        grm_files.push_back(subtract_grm_file);
        for (i = 0; i < grm_files.size(); i++) {
            read_grm(grm_files[i], grm_id, false, true, true);
            update_id_map_kp(grm_id, _id_map, _keep);
        }
    }
    else {
        if (grm_flag) {
            grm_files.push_back(grm_file);
            read_grm(grm_file, grm_id, true, false, true);
            update_id_map_kp(grm_id, _id_map, _keep);
        }
        else if (m_grm_flag) {
            read_grm_filenames(grm_file, grm_files, false);
            for (i = 0; i < grm_files.size(); i++) {
                read_grm(grm_files[i], grm_id, false, true, true);
                update_id_map_kp(grm_id, _id_map, _keep);
            }
        }
        else {
            grm_files.push_back("NA");
            make_grm_mkl(false, false, inbred, true, 0, true);
            for (i = 0; i < _keep.size(); i++) grm_id.push_back(_fid[_keep[i]] + ":" + _pid[_keep[i]]);
        }
    }

    vector<string> uni_id;
    map<string, int> uni_id_map;
    map<string, int>::iterator iter;
    for (i = 0; i < _keep.size(); i++) {
        uni_id.push_back(_fid[_keep[i]] + ":" + _pid[_keep[i]]);
        uni_id_map.insert(pair<string, int>(_fid[_keep[i]] + ":" + _pid[_keep[i]], i));
    }
    _n = _keep.size();
    if (_n < 1) LOGGER.e(0, "no individual is in common in the input files.");
    LOGGER << _n << " individuals are in common in these files." << endl;

    // construct model terms
    _y.setZero(_n);
    for (i = 0; i < phen_ID.size(); i++) {
        iter = uni_id_map.find(phen_ID[i]);
        if (iter == uni_id_map.end()) continue;
        _y[iter->second] = atof(phen_buf[i][mphen - 1].c_str());
    }

    _r_indx.clear();
    vector<int> kp;
    if (subtract_grm_flag) {
        for (i = 0; i < 2; i++) _r_indx.push_back(i);
        _A.resize(_r_indx.size());

        LOGGER << "\nReading the primary GRM from [" << grm_files[1] << "] ..." << endl;
        read_grm(grm_files[1], grm_id, true, false, false);

        StrFunc::match(uni_id, grm_id, kp);
        (_A[0]).resize(_n, _n);
        MatrixXf A_N_buf(_n, _n);
        #pragma omp parallel for private(k)
        for (j = 0; j < _n; j++) {
            for (k = 0; k <= j; k++) {
                if (kp[j] >= kp[k]) {
                    (_A[0])(k, j) = (_A[0])(j, k) = _grm(kp[j], kp[k]);
                    A_N_buf(k, j) = A_N_buf(j, k) = _grm_N(kp[j], kp[k]);
                }
                else {
                    (_A[0])(k, j) = (_A[0])(j, k) = _grm(kp[k], kp[j]);
                    A_N_buf(k, j) = A_N_buf(j, k) = _grm_N(kp[k], kp[j]);
                }
            }
        }

        LOGGER << "\nReading the secondary GRM from [" << grm_files[0] << "] ..." << endl;
        read_grm(grm_files[0], grm_id, true, false, false);
        LOGGER << "\nSubtracting [" << grm_files[1] << "] from [" << grm_files[0] << "] ..." << endl;
        StrFunc::match(uni_id, grm_id, kp);
        #pragma omp parallel for private(k)
        for (j = 0; j < _n; j++) {
            for (k = 0; k <= j; k++) {
                if (kp[j] >= kp[k]) (_A[0])(k, j) = (_A[0])(j, k) = ((_A[0])(j, k) * A_N_buf(j, k) - _grm(kp[j], kp[k]) * _grm_N(kp[j], kp[k])) / (A_N_buf(j, k) - _grm_N(kp[j], kp[k]));
                else (_A[0])(k, j) = (_A[0])(j, k) = ((_A[0])(j, k) * A_N_buf(j, k) - _grm(kp[k], kp[j]) * _grm_N(kp[k], kp[j])) / (A_N_buf(j, k) - _grm_N(kp[k], kp[j]));
            }
        }
        _grm.resize(0, 0);
        _grm_N.resize(0, 0);
    }
    else {
        for (i = 0; i < grm_files.size() + 1; i++) _r_indx.push_back(i);
        _A.resize(_r_indx.size());
        if (grm_flag) {
            StrFunc::match(uni_id, grm_id, kp);
            (_A[0]).resize(_n, _n);
            #pragma omp parallel for private(j)
            for (i = 0; i < _n; i++) {
                for (j = 0; j <= i; j++) (_A[0])(j, i) = (_A[0])(i, j) = _grm(kp[i], kp[j]);
            }
            _grm.resize(0, 0);
        }
        else if (m_grm_flag) {
            LOGGER << "There are " << grm_files.size() << " GRM file names specified in the file [" + grm_file + "]." << endl;
            for (i = 0; i < grm_files.size(); i++) {
                LOGGER << "Reading the GRM from the " << i + 1 << "th file ..." << endl;
                read_grm(grm_files[i], grm_id, true, false, true);
                StrFunc::match(uni_id, grm_id, kp);
                (_A[i]).resize(_n, _n);
                #pragma omp parallel for private(k)
                for (j = 0; j < _n; j++) {
                    for (k = 0; k <= j; k++) {
                        if (kp[j] >= kp[k]) (_A[i])(k, j) = (_A[i])(j, k) = _grm(kp[j], kp[k]);
                        else (_A[i])(k, j) = (_A[i])(j, k) = _grm(kp[k], kp[j]);
                    }
                }
            }
        }
        else {
            StrFunc::match(uni_id, grm_id, kp);
            (_A[0]).resize(_n, _n);
            #pragma omp parallel for private(j)
            for (i = 0; i < _n; i++) {
                for (j = 0; j <= i; j++) (_A[0])(j, i) = (_A[0])(i, j) = _grm_mkl[kp[i] * _n + kp[j]];
            }
            delete[] _grm_mkl;
        }
    }
    _A[_r_indx.size() - 1] = eigenMatrix::Identity(_n, _n);

    // construct X matrix
    vector<eigenMatrix> E_float;
    eigenMatrix qE_float;
    construct_X(_n, uni_id_map, qcovar_flag, qcovar_num, qcovar_ID, qcovar, covar_flag, covar_num, covar_ID, covar, E_float, qE_float);

    // names of variance component
    for (i = 0; i < grm_files.size(); i++) {
        stringstream strstrm;
        if (grm_files.size() == 1) strstrm << "";
        else strstrm << i + 1;
        _var_name.push_back("V(G" + strstrm.str() + ")");
        _hsq_name.push_back("V(G" + strstrm.str() + ")/Vp");
    }
    _var_name.push_back("V(e)");

    if (_within_family) detect_family();

    // run REML algorithm -- this is the entire point of stage 1. Once
    // this returns, _Vi, _b, _X and _varcmp hold everything stage 2
    // needs to run the SNP scan.
    LOGGER << "\nFitting the REML model" << (subtract_grm_flag ? "" : " (including the candidate SNP)") << " ..." << endl;
    reml(false, true, true, reml_priors, reml_priors_var, -2.0, -2.0, no_constrain, true, true);
    _A.clear();

    string reml_file = _out + ".reml";
    save_reml_state(reml_file, uni_id);
    LOGGER << "\nREML variance components and fitted model for " << _n << " individuals saved to [" << reml_file << "]." << endl;
    LOGGER << "Run the association scan with --load-reml " << _out << " (pointing --bfile/--out at the genotype data and desired output) to complete the analysis." << endl;
}

// ---------------------------------------------------------------------
// Stage 2: association scan only. Identical to the second half of
// gcta::mlma() (everything after the reml() call), except _Vi/_X/_y/_b
// come from a previously-saved .reml file instead of an in-process fit.
//
// Assumes standard CLI genotype loading has already populated _include,
// _chr/_bp/_snp_name/_ref_A/_other_A, _geno_mkl and _fid/_pid/_id_map/
// _keep from the .fam file, exactly as it does ahead of gcta::mlma().
// ---------------------------------------------------------------------

void gcta::mlma_assoc_stage(string load_reml_name, bool no_adj_covar)
{
    string reml_file = load_reml_name;
    LOGGER << "Reading the REML fit from [" << reml_file << "] ..." << endl;

    vector<string> saved_ids;
    eigenMatrix Vi_loaded, X_loaded;
    eigenVector y_loaded, b_loaded;
    vector<double> varcmp_loaded;
    int32_t X_c = 0;
    read_reml_state(reml_file, saved_ids, Vi_loaded, X_loaded, y_loaded, b_loaded, varcmp_loaded, X_c);

    // Restrict/reorder _keep to the individuals present in the saved fit,
    // in the saved fit's own order-independent canonical order (same
    // helper every read_* routine in this codebase uses to intersect
    // _keep against an incoming ID list).
    update_id_map_kp(saved_ids, _id_map, _keep);
    if (_keep.size() < 1) LOGGER.e(0, "no individual in the genotype data matches the saved REML fit in [" + reml_file + "].");

    unsigned long i = 0, j = 0;
    vector<string> uni_id;
    for (i = 0; i < _keep.size(); i++) uni_id.push_back(_fid[_keep[i]] + ":" + _pid[_keep[i]]);
    _n = _keep.size();
    if (_n != saved_ids.size()) {
        LOGGER << "Warning: " << saved_ids.size() - _n << " individual(s) present in [" << reml_file << "] are missing from the genotype data and will be dropped." << endl;
    }
    LOGGER << _n << " individuals are in common between the genotype data and the saved REML fit." << endl;

    // Map current --keep order back into the saved (loaded) order.
    vector<int> kp;
    StrFunc::match(uni_id, saved_ids, kp);

    _Vi.resize(_n, _n);
    for (i = 0; i < _n; i++)
        for (j = 0; j < _n; j++) _Vi(i, j) = Vi_loaded(kp[i], kp[j]);
    Vi_loaded.resize(0, 0);

    _X_c = X_c;
    _X.resize(_n, _X_c);
    for (i = 0; i < _n; i++)
        for (j = 0; j < (unsigned long)_X_c; j++) _X(i, j) = X_loaded(kp[i], j);
    X_loaded.resize(0, 0);

    _y.resize(_n);
    for (i = 0; i < _n; i++) _y[i] = y_loaded[kp[i]];

    _b = b_loaded;
    eigenVector varcmp_ev = Map<eigenVector>(varcmp_loaded.data(), varcmp_loaded.size());
    eigenVector2Vector(varcmp_ev, _varcmp);

    // If the saved fit had no covariates beyond the intercept, adjustment
    // vs. not is moot -- force the plain (pre-adjusted) path, mirroring
    // gcta::mlma()'s own "!qcovar_flag && !covar_flag -> no_adj_covar=false".
    if (_X_c <= 1) no_adj_covar = false;

    unsigned long n = _n, m = _include.size();
    float *y = new float[n];
    eigenVector y_buf = _y;
    if (!no_adj_covar) y_buf = _y.array() - (_X * _b).array(); // adjust phenotype for covariates
    for (i = 0; i < n; i++) y[i] = y_buf[i];

    if (_mu.empty()) calcu_mu();
    eigenVector beta, se, pval;
    if (no_adj_covar) mlma_calcu_stat_covar(y, _geno_mkl, n, m, beta, se, pval);
    else mlma_calcu_stat(y, _geno_mkl, n, m, beta, se, pval);
    delete[] y;
    delete[] _geno_mkl;

    string filename = _out + ".mlma";
    LOGGER << "\nSaving the results of the mixed linear model association analyses of " << m << " SNPs to [" + filename + "] ..." << endl;
    ofstream ofile(filename.c_str());
    if (!ofile) LOGGER.e(0, "cannot open the file [" + filename + "] to write.");
    ofile << "Chr\tSNP\tbp\tA1\tA2\tFreq\tb\tse\tp" << endl;
    for (i = 0; i < m; i++) {
        j = _include[i];
        ofile << _chr[j] << "\t" << _snp_name[j] << "\t" << _bp[j] << "\t" << _ref_A[j] << "\t" << _other_A[j] << "\t";
        if (pval[i] > 1.5) ofile << "NA\tNA\tNA\tNA" << endl;
        else ofile << 0.5 * _mu[j] << "\t" << beta[i] << "\t" << se[i] << "\t" << pval[i] << endl;
    }
    ofile.close();
}
