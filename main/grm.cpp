/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * Implementations of functions for estimating the genetic relationship matrix
 *
 * 2010 by Jian Yang <jian.yang.qt@gmail.com>
 *
 * This file is distributed under the GNU General Public
 * License, Version 3.  Please see the file LICENSE for more
 * details
 */

#include "gcta.h"
#include <fstream>
#include <iterator>
#include <ranges>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <cpu.h>   // gcta_dsyevd / gcta_dsyevr wrappers
#include <Eigen/Core>
#include "symmetric_eigendecomp.hpp"

void gcta::enable_grm_bin_flag() {
    _grm_bin_flag = true;
}

void gcta::check_autosome() {
    for (int i = 0; i < _include.size(); i++) {
        if (!isAutosomalChr(_chr[_include[i]])) LOGGER.e(0, "this option requires autosomal SNPs only under the current autosome definition. Check --autosome/--autosome-num.");
    }
}

void gcta::check_homogametic_chromosomes() {
    if (_include.empty()) {
        LOGGER.e(0, "no SNPs are included for homogametic chromosome analysis.");
    }
    const int target_chr = _chr[_include[0]];
    for (int i = 1; i < _include.size(); i++) {
        if (_chr[_include[i]] != target_chr) {
            LOGGER.e(0, "this option requires SNPs from exactly one chromosome. Use --chr to select the homogametic chromosome explicitly.");
        }
    }
    _legacy_homogametic_chr = target_chr;
}

void gcta::check_sex() {
    for (int i = 0; i < _keep.size(); i++) {
        if (_sex[_keep[i]] != 1 && _sex[_keep[i]] != 2) LOGGER.e(0, "sex information of the individual \"" + _fid[_keep[i]] + " " + _pid[_keep[i]] + "\" is missing.\nUse --update-sex option to update the sex information of the individuals.");
    }
}

void gcta::make_grm(bool grm_d_flag, bool grm_homogametic_flag, bool inbred, bool output_bin, int grm_mtd, bool mlmassoc, bool diag_f3_flag, std::string subpopu_file)
{
    bool have_mis = false;

    if (!grm_d_flag && grm_homogametic_flag) check_homogametic_chromosomes();
    else {
        _legacy_homogametic_chr = 0;
        check_autosome();
    }

    unsigned long i = 0, j = 0, k = 0, n = _keep.size(), m = _include.size();
    if(grm_d_flag) have_mis = make_XMat_d(_geno);
    else  have_mis = make_XMat(_geno);

    eigenVector sd_SNP;
    if (grm_mtd == 0) {
        if (grm_d_flag) std_XMat_d(_geno, sd_SNP, false, true);
        else{
            if(subpopu_file.empty()) std_XMat(_geno, sd_SNP, grm_homogametic_flag, false, true);
            else std_XMat_subpopu(subpopu_file, _geno, sd_SNP, grm_homogametic_flag, false, true);
        }
    } 
    else {
        if (grm_d_flag) std_XMat_d(_geno, sd_SNP, false, false);
        else std_XMat(_geno, sd_SNP, grm_homogametic_flag, false, false);
    }

    if (!mlmassoc) LOGGER << "\nCalculating the" << ((grm_d_flag) ? " dominance" : "") << " genetic relationship matrix (GRM)" << (grm_homogametic_flag ? " for homogametic chromosomes" : "") << (_dosage_flag ? " using imputed dosage data" : "") << " ... (Note: default speed-optimized mode, may use huge RAM)" << std::endl;
    else LOGGER << "\nCalculating the genetic relationship matrix (GRM) ... " << std::endl;

    // count the number of missing genotypes
    std::vector< std::vector<int> > miss_pos;
    std::vector< std::vector<bool> > X_bool;
    if(have_mis){
        miss_pos.resize(n);
        X_bool.resize(n);
        #pragma omp parallel for private(j)
        for (i = 0; i < n; i++) {
            X_bool[i].resize(m);
            for (j = 0; j < m; j++) {
                if (_geno(i,j) < 1e5) X_bool[i][j] = true;
                else {
                    _geno(i,j) = 0.0;
                    miss_pos[i].push_back(j);
                    X_bool[i][j] = false;
                }
            }
        }
    }

    // Calculate A_N matrix
    if(have_mis){
        _grm_N.resize(n, n);
        #pragma omp parallel for private(j, k)
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                int miss_j = 0;
                for (k = 0; k < miss_pos[j].size(); k++) miss_j += (int) X_bool[i][miss_pos[j][k]];
                _grm_N(i,j) = m - miss_pos[i].size() - miss_j;
                _grm_N(j,i) = _grm_N(i,j);
            }
        }
    }
    else _grm_N = Eigen::MatrixXf::Constant(n,n,m);

    // Calcuate WW'
    #ifdef SINGLE_PRECISION
    _grm = _geno * _geno.transpose();
    #else
    _grm = (_geno * _geno.transpose()).cast<double>();
    #endif

    // Calculate A matrix
    if (grm_mtd == 1) _grm_N = _grm_N.array() * sd_SNP.mean();

    #pragma omp parallel for private(j)
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            if(_grm_N(i,j) > 0) _grm(i,j) /= _grm_N(i,j);
            else _grm(i,j) = 0.0;
            _grm(j,i) = _grm(i,j);
        }
    }

    // GRM summary
    double diag_m = 0.0, diag_v = 0.0, off_m = 0.0, off_v = 0.0;
    calcu_grm_var(diag_m, diag_v, off_m, off_v);
    LOGGER<<"\nSummary of the GRM:" << std::endl;
    LOGGER<<"Mean of diagonals = "<<diag_m<<std::endl;
    LOGGER<<"Variance of diagonals = "<<diag_v<<std::endl;
    LOGGER<<"Mean of off-diagonals = " << off_m <<std::endl;
    LOGGER<<"Variance of off-diagonals = " << off_v <<std::endl;

    // re-calcuate the diagonals (Fhat3+1)
    if (diag_f3_flag) {
        #pragma omp parallel for private(j)
        for (i = 0; i < n; i++) {
            _grm(i,i) = 0.0;
            double non_missing = 0.0;
            for (j = 0; j < m; j++) {
                if (_geno(i,j) < 1e5){
                    _grm(i,i) += _geno(i,j)*(_geno(i,j)+(_mu[_include[j]] - 1.0) * sd_SNP[j]);
                    non_missing += 1.0;
                } 
            }
            _grm(i,i) /= non_missing; 
        }
    }

    if (inbred) {
        #pragma omp parallel for private(j)
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) _grm(i,j) *= 0.5;
        }
    }

    if (mlmassoc && grm_mtd == 0) {
        for (j = 0; j < m; j++) {
            if (fabs(sd_SNP[j]) < 1.0e-50) sd_SNP[j] = 0.0;
            else sd_SNP[j] = 1.0 / sd_SNP[j];
        }
        #pragma omp parallel for private(j)
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                if (_geno(i,j) < 1e5) _geno(i,j) *= sd_SNP[j];
                else _geno(i,j) = 0.0;
            }
        }
    } 
    else {
        // Output A_N and A
        std::string out_buf = _out;
        if (grm_d_flag) _out += ".d";
        output_grm(output_bin);
        _out = out_buf;
    }
}

void gcta::calcu_grm_var(double &diag_m, double &diag_v, double &off_m, double &off_v)
{
    int i = 0, n = _keep.size();
    diag_m = _grm.diagonal().mean();
    diag_v = (_grm.diagonal() - eigenVector::Constant(n, diag_m)).squaredNorm() / ((double)n - 1.0);
    double off_num = 0.5*n*(n - 1.0);
    off_m = 0.0;
    for (i = 1; i < n; i++) off_m += _grm.row(i).segment(0, i).sum();
    off_m /= off_num;
    off_v = 0.0;
    for (i = 1; i < n; i++) off_v += (_grm.row(i).segment(0, i) -  eigenVector::Constant(i, off_m).transpose()).squaredNorm();
    off_v /= (off_num - 1.0);    
}

void gcta::output_grm(bool output_grm_bin)
{
    int i = 0, j = 0;
    std::string grm_file;
    if (output_grm_bin) {
        // Save matrix A in binary file
        grm_file = _out + ".grm.bin";
        std::fstream A_Bin(grm_file.c_str(), std::ios::out | std::ios::binary);
        if (!A_Bin) LOGGER.e(0, "cannot open the file [" + grm_file + "] to write.");
        float f_buf = 0.0;
        int size = sizeof (float);
        for (i = 0; i < _keep.size(); i++) {
            for (j = 0; j <= i; j++) {
                f_buf = (float) (_grm(i, j));
                A_Bin.write((char*) &f_buf, size);
            }
        }
        A_Bin.close();
        LOGGER << "GRM of " << _keep.size() << " individuals has been saved in the file [" + grm_file + "] (in binary format)." << std::endl;

        std::string grm_N_file = _out + ".grm.N.bin";
        std::fstream N_Bin(grm_N_file.c_str(), std::ios::out | std::ios::binary);
        if (!N_Bin) LOGGER.e(0, "cannot open the file [" + grm_N_file + "] to write.");
        f_buf = 0.0;
        size = sizeof (int);
        for (i = 0; i < _keep.size(); i++) {
            for (j = 0; j <= i; j++) {
                f_buf = (float) (_grm_N(i, j));
                N_Bin.write((char*) &f_buf, size);
            }
        }
        N_Bin.close();
        LOGGER << "Number of SNPs to calculate the genetic relationship between each std::pair of individuals has been saved in the file [" + grm_N_file + "] (in binary format)." << std::endl;
    } 
    else {
        // Save A matrix in txt format
        grm_file = _out + ".grm.gz";
        std::ofstream raw_file(grm_file, std::ios::binary);
        if (!raw_file.is_open()) LOGGER.e(0, "cannot open the file [" + grm_file + "] to write.");
        boost::iostreams::filtering_ostream zoutf;
        zoutf.push(boost::iostreams::gzip_compressor());
        zoutf.push(raw_file);
        LOGGER << "Saving the genetic relationship matrix to the file [" + grm_file + "] (in compressed text format)." << std::endl;
        zoutf.setf(std::ios::scientific);
        zoutf.precision(6);
        for (i = 0; i < _keep.size(); i++) {
            if (_grm_N.rows() > 0){
                zoutf.setf(std::ios::scientific);
                zoutf.precision(6);
                for (j = 0; j <= i; j++) zoutf << i + 1 << '\t' << j + 1 << '\t' << _grm_N(i, j) << '\t' << _grm(i, j) << std::endl;
            }
            else{ 
                for (j = 0; j <= i; j++) zoutf << i + 1 << '\t' << j + 1 << "\t0\t" << _grm(i, j) << std::endl;
            }
        }
        zoutf.reset();
        raw_file.close();
        LOGGER << "The genetic relationship matrix has been saved in the file [" + grm_file + "] (in compressed text format)." << std::endl;
    }

    std::string famfile = _out + ".grm.id";
    std::ofstream Fam(famfile.c_str());
    if (!Fam) LOGGER.e(0, "cannot open the file [" + famfile + "] to write.");
    for (i = 0; i < _keep.size(); i++) Fam << _fid[_keep[i]] + "\t" + _pid[_keep[i]] << std::endl;
    Fam.close();
    LOGGER << "IDs for the GRM file [" + grm_file + "] have been saved in the file [" + famfile + "]." << std::endl;
}

int gcta::read_grm_id(std::string grm_file, std::vector<std::string> &grm_id, bool out_id_log, bool read_id_only)
{
    // read GRM IDs
    std::string grm_id_file = grm_file + ".grm.id";
    if (out_id_log) LOGGER << "Reading IDs of the GRM from [" + grm_id_file + "]." << std::endl;
    int fd = open(grm_id_file.c_str(), O_RDONLY);
    if (fd < 0) LOGGER.e(0, "cannot open the file [" + grm_id_file + "] to read.");
    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;
    std::vector<std::string> fid, pid;
    grm_id.clear();
    if (file_size > 0) {
        const char* data = (const char*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (data == MAP_FAILED) LOGGER.e(0, "mmap failed for [" + grm_id_file + "].");
        madvise((void*)data, file_size, MADV_SEQUENTIAL);
        const char* p = data;
        const char* end = data + file_size;
        while (p < end) {
            while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
            if (p >= end) break;
            const char* tok = p;
            while (p < end && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') ++p;
            std::string f(tok, p);
            while (p < end && (*p == ' ' || *p == '\t')) ++p;
            tok = p;
            while (p < end && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') ++p;
            std::string id(tok, p);
            while (p < end && *p != '\n') ++p;
            fid.push_back(f);
            pid.push_back(id);
            grm_id.push_back(f + ":" + id);
        }
        munmap((void*)data, file_size);
    } else {
        close(fd);
    }
    int n = grm_id.size();
    if (out_id_log) LOGGER << n << " IDs are read from [" + grm_id_file + "]." << std::endl;

    if (_id_map.empty()) {
        _fid = fid;
        _pid = pid;
        _indi_num = _fid.size();
        _sex.resize(_fid.size());
        init_keep();
    }

    return (n);
}

void gcta::read_grm(std::string grm_file, std::vector<std::string> &grm_id, bool out_id_log, bool read_id_only, bool dont_read_N)
{
    if (_grm_bin_flag) read_grm_bin(grm_file, grm_id, out_id_log, read_id_only, dont_read_N);
    else read_grm_gz(grm_file, grm_id, out_id_log, read_id_only);
}

void gcta::read_grm_gz(std::string grm_file, std::vector<std::string> &grm_id, bool out_id_log, bool read_id_only) {
    long n = read_grm_id(grm_file, grm_id, out_id_log, read_id_only);

    if (read_id_only) return;

    std::string grm_gzfile = grm_file + ".grm.gz", str_buf;
    std::ifstream raw_file(grm_gzfile, std::ios::binary);
    if (!raw_file.is_open()) LOGGER.e(0, "cannot open the file [" + grm_gzfile + "] to read.");
    boost::iostreams::filtering_istream zinf;
    zinf.push(boost::iostreams::gzip_decompressor());
    zinf.push(raw_file);

    long indx1 = 0, indx2 = 0, nline = 0;
    double grm_buf = 0.0, grm_N_buf;
    std::string errmsg = "failed to read [" + grm_gzfile + "]. The format of the GRM file has been changed?\nError occurs in line:\n";
    LOGGER << "Reading the GRM from [" + grm_gzfile + "]." << std::endl;
    _grm.resize(n, n);
    _grm_N.resize(n, n);
    std::string line;
    while (std::getline(zinf, line)) {
        std::stringstream ss(line);
        if (!(ss >> indx1)) LOGGER.e(0, errmsg + line);
        if (!(ss >> indx2)) LOGGER.e(0, errmsg + line);
        if (!(ss >> grm_N_buf)) LOGGER.e(0, errmsg + line);
        if (!(ss >> grm_buf)) LOGGER.e(0, errmsg + line);
        if (indx1 < indx2 || indx1 > n || indx2 > n) LOGGER.e(0, errmsg + line);
        if (grm_N_buf == 0) LOGGER.w(0, line);
        _grm_N(indx1 - 1, indx2 - 1) = _grm_N(indx2 - 1, indx1 - 1) = grm_N_buf;
        _grm(indx1 - 1, indx2 - 1) = _grm(indx2 - 1, indx1 - 1) = grm_buf;
        nline++;
        if (ss >> str_buf) LOGGER.e(0, errmsg + line);
    }
    zinf.reset();
    raw_file.close();
    if (!_within_family && nline != (long) n * (n + 1)*0.5){
        std::stringstream errmsg_tmp;
        errmsg_tmp << "there are " << nline << " lines in the [" << grm_gzfile << "] file. The expected number of lines is " << (long) (n * (n + 1)*0.5) << "." << std::endl;
        LOGGER.e(0, errmsg_tmp.str());
        //LOGGER.e(0, "incorrect number of lines in the grm file. Are *.grm.gz file and *.grm.id file mismatched?");
    }
    LOGGER << "GRM for " << n << " individuals are included from [" + grm_gzfile + "]." << std::endl;
}

void gcta::read_grm_bin(std::string grm_file, std::vector<std::string> &grm_id, bool out_id_log, bool read_id_only, bool dont_read_N)
{
    int i = 0, j = 0, n = read_grm_id(grm_file, grm_id, out_id_log, read_id_only);

    if (read_id_only) return;

    std::string grm_binfile = grm_file + ".grm.bin";
    _grm.resize(n, n);
    LOGGER << "Reading the GRM from [" + grm_binfile + "]." << std::endl;
    {
        long long n_tri = (long long)n * (n + 1) / 2;
        int fd = open(grm_binfile.c_str(), O_RDONLY);
        if (fd < 0) LOGGER.e(0, "cannot open the file [" + grm_binfile + "] to read.");
        struct stat st;
        fstat(fd, &st);
        if ((long long)st.st_size < n_tri * (long long)sizeof(float))
            LOGGER.e(0, "Is the size of the [" + grm_binfile + "] file incorrect?");
        const float* buf = (const float*)mmap(nullptr, n_tri * sizeof(float), PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (buf == MAP_FAILED) LOGGER.e(0, "mmap failed for [" + grm_binfile + "].");
        madvise((void*)buf, n_tri * sizeof(float), MADV_SEQUENTIAL | MADV_WILLNEED);

        // Batch convert float→double: sequential read + sequential write,
        // compiler-vectorisable (AVX2 vcvtps2pd). Decouples I/O stream
        // from the scatter into _grm so neither fights the other's prefetch.
        const size_t byte_len = static_cast<size_t>(n_tri) * sizeof(float);
        std::vector<double> dbuf(static_cast<size_t>(n_tri));
        std::transform(buf, buf + n_tri, dbuf.begin(),
                       [](float f) noexcept { return static_cast<double>(f); });
        munmap((void*)buf, byte_len);

        // Fill lower triangle only from dbuf, then mirror via selfadjointView.
        // Eliminates n²/2 cache-hostile scattered writes into upper triangle
        // columns (column-major Eigen; upper-triangle writes stride by n doubles).
        for (i = 0; i < n; i++) {
            const double* row = dbuf.data() + (long long)i * (i + 1) / 2;
            for (j = 0; j <= i; j++)
                _grm(i, j) = row[j];
        }
        _grm = _grm.selfadjointView<Eigen::Lower>();
    }


    if(!dont_read_N){
        std::string grm_Nfile = grm_file + ".grm.N.bin";
        _grm_N.resize(n, n);
        LOGGER << "Reading the number of SNPs for the GRM from [" + grm_Nfile + "]." << std::endl;
        {
            long long n_tri = (long long)n * (n + 1) / 2;
            int fd = open(grm_Nfile.c_str(), O_RDONLY);
            if (fd < 0) LOGGER.e(0, "cannot open the file [" + grm_Nfile + "] to read.");
            struct stat st;
            fstat(fd, &st);
            if ((long long)st.st_size < n_tri * (long long)sizeof(float))
                LOGGER.e(0, "Is the size of the [" + grm_Nfile + "] file incorrect?");
            const float* buf = (const float*)mmap(nullptr, n_tri * sizeof(float), PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (buf == MAP_FAILED) LOGGER.e(0, "mmap failed for [" + grm_Nfile + "].");
            madvise((void*)buf, n_tri * sizeof(float), MADV_SEQUENTIAL);
            #pragma omp parallel for schedule(dynamic, 64) private(j)
            for (i = 0; i < n; i++) {
                const float* row = buf + (long long)i * (i + 1) / 2;
                for (j = 0; j <= i; j++) {
                    _grm_N(j, i) = _grm_N(i, j) = row[j];
                }
            }
            munmap((void*)buf, n_tri * sizeof(float));
        }
    }

    LOGGER << "GRM for " << n << " individuals are included from [" + grm_binfile + "]." << std::endl;
}

void gcta::rm_cor_indi(double grm_cutoff) {
    LOGGER << "Pruning the GRM with a cutoff of " << grm_cutoff << " ..." << std::endl;

    int i = 0, j = 0, i_buf = 0;

    // identify the positions where you see a value > than the threshold
    std::vector<int> rm_grm_ID1, rm_grm_ID2;
    for (i = 0; i < _keep.size(); i++) {
        for (j = 0; j < i; j++) {
            if (_grm(_keep[i], _keep[j]) > grm_cutoff) {
                rm_grm_ID1.push_back(_keep[i]);
                rm_grm_ID2.push_back(_keep[j]);
            }
        }
    }

    // count the number of appearance of each "position" in the std::vector, which involves a few steps
    std::vector<int> rm_uni_ID(rm_grm_ID1);
    rm_uni_ID.insert(rm_uni_ID.end(), rm_grm_ID2.begin(), rm_grm_ID2.end());
    std::stable_sort(rm_uni_ID.begin(), rm_uni_ID.end());
    rm_uni_ID.erase(std::unique(rm_uni_ID.begin(), rm_uni_ID.end()), rm_uni_ID.end());
    std::map<int, int> rm_uni_ID_count;
    for (i = 0; i < rm_uni_ID.size(); i++) {
        i_buf = std::count(rm_grm_ID1.begin(), rm_grm_ID1.end(), rm_uni_ID[i]) + std::count(rm_grm_ID2.begin(), rm_grm_ID2.end(), rm_uni_ID[i]);
        rm_uni_ID_count.insert(std::pair<int, int>(rm_uni_ID[i], i_buf));
    }

    // swapping
    for (i = 0; i < rm_grm_ID1.size(); i++) {
        auto iter1 = rm_uni_ID_count.find(rm_grm_ID1[i]);
        auto iter2 = rm_uni_ID_count.find(rm_grm_ID2[i]);
        if (iter1->second < iter2->second) {
            i_buf = rm_grm_ID1[i];
            rm_grm_ID1[i] = rm_grm_ID2[i];
            rm_grm_ID2[i] = i_buf;
        }
    }
    
    std::stable_sort(rm_grm_ID1.begin(), rm_grm_ID1.end());
    rm_grm_ID1.erase(std::unique(rm_grm_ID1.begin(), rm_grm_ID1.end()), rm_grm_ID1.end());
    std::vector<std::string> removed_ID;
    for (i = 0; i < rm_grm_ID1.size(); i++) removed_ID.push_back(_fid[rm_grm_ID1[i]] + ":" + _pid[rm_grm_ID1[i]]);

    // update _keep and _id_map
    update_id_map_rm(removed_ID, _id_map, _keep);

    LOGGER << "After pruning the GRM, there are " << _keep.size() << " individuals (" << removed_ID.size() << " individuals removed)." << std::endl;
}

void gcta::adj_grm(double adj_grm_fac) {
    LOGGER << "Adjusting the GRM for sampling errors ..." << std::endl;
    int i = 0, j = 0, n = _keep.size();
    double off_mean = 0.0, diag_mean = 0.0, off_var = 0.0, diag_var = 0.0, d_buf = 0.0;
    for (i = 0; i < n; i++) {
        diag_mean += _grm(_keep[i], _keep[i]);
        for (j = 0; j < i; j++) off_mean += _grm(_keep[i], _keep[j]);
    }
    diag_mean /= n;
    off_mean /= 0.5 * n * (n - 1.0);
    for (i = 0; i < n; i++) {
        d_buf = _grm(_keep[i], _keep[i]) - diag_mean;
        diag_var += d_buf*d_buf;
        for (j = 0; j < i; j++) {
            d_buf = _grm(_keep[i], _keep[j]) - off_mean;
            off_var += d_buf*d_buf;
        }
    }
    diag_var /= n - 1.0;
    off_var /= 0.5 * n * (n - 1.0) - 1.0;
    for (i = 0; i < _keep.size(); i++) {
        d_buf = 1.0 - (adj_grm_fac + 1.0 / _grm_N(_keep[i], _keep[i])) / diag_var;
        if (_grm(_keep[i], _keep[i]) > 0) _grm(_keep[i], _keep[i]) = 1.0 + d_buf * (_grm(_keep[i], _keep[i]) - 1.0);
        for (j = 0; j < i; j++) {
            if (_grm_N(_keep[i], _keep[j]) > 0) _grm(_keep[i], _keep[j]) *= 1.0 - (adj_grm_fac + 1.0 / _grm_N(_keep[i], _keep[j])) / off_var;
        }
    }
}

void gcta::dc(int dosage_compen) {
    LOGGER << "Parameterizing the GRM under the assumption of ";
    if (dosage_compen == 1) LOGGER << "full dosage compensation ..." << std::endl;
    else if (dosage_compen == 0) LOGGER << "no dosage compensation ..." << std::endl;

    int i = 0, j = 0, i_buf = 0;
    double c1 = 1.0, c2 = 1.0;
    if (dosage_compen == 1) {
        c1 = 2.0;
        c2 = sqrt(2.0);
    }// full dosage compensation
    else if (dosage_compen == 0) {
        c1 = 0.5;
        c2 = sqrt(0.5);
    } // on dosage compensation
    for (i = 0; i < _keep.size(); i++) {
        for (j = 0; j <= i; j++) {
            i_buf = _sex[_keep[i]] * _sex[_keep[j]];
            if (i_buf == 1) _grm(i, j) *= c1;
            else if (i_buf == 2) _grm(i, j) *= c2;
        }
    }
}

void gcta::manipulate_grm(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, std::string sex_file, double grm_cutoff, double adj_grm_fac, int dosage_compen, bool merge_grm_flag, bool dont_read_N)
{
    int i = 0, j = 0;

    std::vector<std::string> grm_id;
    if (merge_grm_flag) merge_grm(grm_file);
    else read_grm(grm_file, grm_id, true, false, dont_read_N);

    if (!keep_indi_file.empty()) keep_indi(keep_indi_file);
    if (!remove_indi_file.empty()) remove_indi(remove_indi_file);
    if (grm_cutoff>-1.0) rm_cor_indi(grm_cutoff);
    if (!sex_file.empty()) update_sex(sex_file);
    if (adj_grm_fac>-1.0) adj_grm(adj_grm_fac);
    if (dosage_compen>-1) dc(dosage_compen);
    if (grm_cutoff>-1.0 || !keep_indi_file.empty() || !remove_indi_file.empty()) {
        eigenMatrix grm_buf(_grm);
        _grm.resize(_keep.size(), _keep.size());
        for (i = 0; i < _keep.size(); i++) {
            for (j = 0; j <= i; j++) _grm(i, j) = grm_buf(_keep[i], _keep[j]);
        }
        grm_buf.resize(0,0);
        if(!dont_read_N){
            Eigen::MatrixXf grm_N_buf = _grm_N;
            _grm_N.resize(_keep.size(), _keep.size());
            for (i = 0; i < _keep.size(); i++) {
                for (j = 0; j <= i; j++) _grm_N(i, j) = grm_N_buf(_keep[i], _keep[j]);
            }
        }
    }
}

void gcta::save_grm(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, std::string sex_file, double grm_cutoff, double adj_grm_fac, int dosage_compen, bool merge_grm_flag, bool output_grm_bin) {
    if (dosage_compen>-1) check_sex();
    manipulate_grm(grm_file, keep_indi_file, remove_indi_file, sex_file, grm_cutoff, adj_grm_fac, dosage_compen, merge_grm_flag);
    output_grm(output_grm_bin);
}

void gcta::merge_grm(std::string merge_grm_file) {
    std::vector<std::string> grm_files, grm_id;
    read_grm_filenames(merge_grm_file, grm_files);
    int K = (int)grm_files.size();

    int f = 0, i = 0, j = 0;
    for (f = 0; f < K; f++) {
        read_grm(grm_files[f], grm_id, false, true);
        update_id_map_kp(grm_id, _id_map, _keep);
    }
    std::vector<std::string> uni_id;
    for (i = 0; i < (int)_keep.size(); i++) uni_id.push_back(_fid[_keep[i]] + ":" + _pid[_keep[i]]);
    _n = uni_id.size();
    if (_n == 0) LOGGER.e(0, "no individual is in common among the GRM files.");
    else LOGGER << _n << " individuals in common in the GRM files." << std::endl;

    // Check whether all files share the same individuals in the same order.
    // If so, files can be streamed row-by-row (O(K*n) extra memory instead of O(K*n^2)).
    bool streaming = true;
    for (f = 0; f < K && streaming; f++) {
        std::vector<std::string> file_id;
        int nf = read_grm_id(grm_files[f], file_id, false, false);
        if (nf != _n) { streaming = false; break; }
        std::vector<int> kp;
        StrFunc::match(uni_id, file_id, kp);
        for (i = 0; i < _n; i++) {
            if (kp[i] != i) { streaming = false; break; }
        }
    }

    if (streaming) {
        LOGGER << "Sample order is consistent across files; using streaming merge." << std::endl;
        _grm.resize(_n, _n);
        _grm_N.resize(_n, _n);

        // Open all source file descriptors up front; reads will be sequential.
        std::vector<int> bin_fds(K, -1), N_fds(K, -1);
        for (f = 0; f < K; f++) {
            bin_fds[f] = open((grm_files[f] + ".grm.bin").c_str(), O_RDONLY);
            if (bin_fds[f] < 0) LOGGER.e(0, "cannot open [" + grm_files[f] + ".grm.bin] to read.");
            N_fds[f] = open((grm_files[f] + ".grm.N.bin").c_str(), O_RDONLY);
            if (N_fds[f] < 0) LOGGER.e(0, "cannot open [" + grm_files[f] + ".grm.N.bin] to read.");
        }

        std::vector<float> vbuf(_n), nbuf(_n);
        std::vector<double> wsum(_n), wtN(_n);

        for (i = 0; i < _n; i++) {
            int row_len = i + 1;
            ssize_t expect = (ssize_t)row_len * sizeof(float);
            std::fill(wsum.begin(), wsum.begin() + row_len, 0.0);
            std::fill(wtN.begin(),  wtN.begin()  + row_len, 0.0);
            for (f = 0; f < K; f++) {
                if (::read(bin_fds[f], vbuf.data(), expect) != expect)
                    LOGGER.e(0, "unexpected end of file in [" + grm_files[f] + ".grm.bin].");
                if (::read(N_fds[f],   nbuf.data(), expect) != expect)
                    LOGGER.e(0, "unexpected end of file in [" + grm_files[f] + ".grm.N.bin].");
                for (j = 0; j < row_len; j++) {
                    wsum[j] += (double)vbuf[j] * (double)nbuf[j];
                    wtN[j]  += (double)nbuf[j];
                }
            }
            for (j = 0; j < row_len; j++) {
                double v = (wtN[j] == 0.0) ? 0.0 : wsum[j] / wtN[j];
                _grm(i, j) = _grm(j, i) = v;
                _grm_N(i, j) = _grm_N(j, i) = (float)wtN[j];
            }
        }

        for (f = 0; f < K; f++) { close(bin_fds[f]); close(N_fds[f]); }
    } else {
        LOGGER << "Warning: sample ordering not consistent across GRM files; falling back to bulk merge." << std::endl;

        std::vector<int> kp;
        eigenMatrix grm   = eigenMatrix::Zero(_n, _n);
        eigenMatrix grm_N = eigenMatrix::Zero(_n, _n);
        for (f = 0; f < K; f++) {
            LOGGER << "Reading the GRM from the " << f + 1 << "th file ..." << std::endl;
            read_grm(grm_files[f], grm_id);
            StrFunc::match(uni_id, grm_id, kp);
            #pragma omp parallel for schedule(dynamic, 64) private(j)
            for (i = 0; i < _n; i++) {
                for (j = 0; j <= i; j++) {
                    int r = std::max(kp[i], kp[j]);
                    int c = std::min(kp[i], kp[j]);
                    grm(i, j)   += _grm(r, c) * _grm_N(r, c);
                    grm_N(i, j) += _grm_N(r, c);
                }
            }
        }
        _grm.resize(_n, _n);
        _grm_N.resize(_n, _n);
        #pragma omp parallel for schedule(dynamic, 64) private(j)
        for (i = 0; i < _n; i++) {
            for (j = 0; j <= i; j++) {
                double v = (grm_N(i, j) == 0) ? 0.0 : grm(i, j) / grm_N(i, j);
                _grm(i, j) = _grm(j, i) = v;
                _grm_N(i, j) = _grm_N(j, i) = (float)grm_N(i, j);
            }
        }
    }

    LOGGER << "\n" << K << " GRMs have been merged together." << std::endl;
}

void gcta::align_grm(std::string m_grm_file) {
    std::vector<std::string> grm_files, grm_id;
    read_grm_filenames(m_grm_file, grm_files);
    
    int f = 0, i = 0, j = 0;
    for (f = 0; f < grm_files.size(); f++) {
        read_grm(grm_files[f], grm_id, false, true);
        update_id_map_kp(grm_id, _id_map, _keep);
    }
    std::vector<std::string> uni_id;
    for (i = 0; i < _keep.size(); i++) uni_id.push_back(_fid[_keep[i]] + ":" + _pid[_keep[i]]);
    _n = uni_id.size();
    if (_n == 0) LOGGER.e(0, "no individual is in common among the GRM files.");
    else LOGGER << _n << " individuals are in common among the GRM files." << std::endl;
    
    std::string _out_save = _out;
    
    std::vector<int> kp;
    eigenMatrix grm = eigenMatrix::Zero(_n, _n);
    eigenMatrix grm_N = eigenMatrix::Zero(_n, _n);
    for (f = 0; f < grm_files.size(); f++) {
        LOGGER << "Reading the GRM from the " << f + 1 << "th file ..." << std::endl;
        grm.setZero(_n, _n);
        grm_N.setZero(_n, _n);
        read_grm(grm_files[f], grm_id);
        StrFunc::match(uni_id, grm_id, kp);
        for (i = 0; i < _n; i++) {
            for (j = 0; j <= i; j++) {
                if (kp[i] >= kp[j]) {
                    grm(i, j) = _grm(kp[i], kp[j]) * _grm_N(kp[i], kp[j]);
                    grm_N(i, j) = _grm_N(kp[i], kp[j]);
                } else {
                    grm(i, j) = _grm(kp[j], kp[i]) * _grm_N(kp[j], kp[i]);
                    grm_N(i, j) = _grm_N(kp[j], kp[i]);
                }
            }
        }
        for (i = 0; i < _n; i++) {
            for (j = 0; j <= i; j++) {
                if (grm_N(i, j) == 0) _grm(i, j) = 0;
                else _grm(i, j) = grm(i, j) / grm_N(i, j);
                _grm_N(i, j) = grm_N(i, j);
            }
        }
        
        _out = grm_files[f] + ".aligned";
        output_grm(true);
    }
    
    _out = _out_save;
    
    grm.resize(0, 0);
    grm_N.resize(0, 0);
    LOGGER << "\n" << grm_files.size() << " GRMs have been aligned." << std::endl;
}


void gcta::read_grm_filenames(std::string merge_grm_file,std::vector<std::string> &grm_files, bool out_log) {
    std::ifstream merge_grm(merge_grm_file.c_str());
    if (!merge_grm) LOGGER.e(0, "cannot open the file [" + merge_grm_file + "] to read.");
    std::string str_buf;
    grm_files.clear();
    std::vector<std::string> vs_buf;
    while (std::getline(merge_grm, str_buf)) {
        if (!str_buf.empty()) {
            if (StrFunc::split_string(str_buf, vs_buf) == 1) grm_files.push_back(vs_buf[0]);
        }
    }
    if (out_log) LOGGER << "There are " << grm_files.size() << " GRM file names specified in [" + merge_grm_file + "]." << std::endl;
    if (grm_files.size() > 1000) LOGGER.e(0, "too many GRM file names specified in [" + merge_grm_file + "]. The maximum number is 1000.");
    if (grm_files.size() < 1) LOGGER.e(0, "no GRM file name is found in [" + merge_grm_file + "].");
}

void gcta::grm_bK(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, double threshold, bool grm_out_bin_flag)
{
    int i = 0, j = 0;
    std::vector<std::string> grm_id;
    read_grm(grm_file, grm_id);
    if (!keep_indi_file.empty()) keep_indi(keep_indi_file);
    if (!remove_indi_file.empty()) remove_indi(remove_indi_file);
    if (!keep_indi_file.empty() || !remove_indi_file.empty()) {
        eigenMatrix grm_buf(_grm);
        _grm.resize(_keep.size(), _keep.size());
        for (i = 0; i < _keep.size(); i++) {
            for (j = 0; j <= i; j++) _grm(i, j) = grm_buf(_keep[i], _keep[j]);
        }
        grm_buf.resize(0,0);
        Eigen::MatrixXf grm_N_buf = _grm_N;
        _grm_N.resize(_keep.size(), _keep.size());
        for (i = 0; i < _keep.size(); i++) {
            for (j = 0; j <= i; j++) _grm_N(i, j) = grm_N_buf(_keep[i], _keep[j]);
        }
    }

    LOGGER << "\nThe off-diagonals that are < " << threshold << " are std::set to zero.\n" << std::endl;
    for (i = 0; i < _keep.size(); i++) {
        for (j = 0; j < i; j++){
            if(_grm(i, j) < threshold) _grm(i, j) = 0.0;
        }
    }

    output_grm(grm_out_bin_flag);
}

void gcta::grm_denseness(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, bool merge_grm_flag, std::string metric)
{
    static const std::vector<std::string> valid_metrics = {"all", "conditioning", "offdiag", "effective-rank"};
    if (std::find(valid_metrics.begin(), valid_metrics.end(), metric) == valid_metrics.end())
        LOGGER.e(0, "--denseness: unrecognised metric '" + metric + "'. Use 'conditioning', 'offdiag', 'effective-rank', or 'all'.");

    manipulate_grm(grm_file, keep_indi_file, remove_indi_file, "", -2.0, -2.0, -2, merge_grm_flag, true);
    _grm_N.resize(0, 0);
    int n = _keep.size();

    Eigen::MatrixXd grm_dbl_storage;
    const Eigen::MatrixXd& grm_dbl = [&]() -> const Eigen::MatrixXd& {
        if constexpr (std::is_same_v<eigenMatrix, Eigen::MatrixXd>) {
            return _grm;
        } else {
            grm_dbl_storage = _grm.cast<double>();
            return grm_dbl_storage;
        }
    }();

    bool do_all     = (metric == "all");
    bool do_offdiag = do_all || (metric == "offdiag");
    bool do_cond    = do_all || (metric == "conditioning");
    bool do_effrank = do_all || (metric == "effective-rank");

    LOGGER << "\nGRM density metrics for " << n << " individuals:" << std::endl;

    // -------------------------------------------------------------------------
    // Off-diagonal statistics (lower triangle only, single pass)
    // -------------------------------------------------------------------------
    if (do_offdiag) {
        double off_num         = 0.5 * n * (n - 1.0);
        double off_sum         = 0.0;
        double off_abs_sum     = 0.0;
        double off_sum_sq      = 0.0;
        double off_max_abs     = 0.0;
        long long n_above_005  = 0;
        long long n_above_0125 = 0;

        const double threshold_nonzero = 0.05;
        const double threshold_cousin  = 0.125;

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                double val     = grm_dbl(i, j);
                double abs_val = std::abs(val);

                off_sum     += val;
                off_abs_sum += abs_val;
                off_sum_sq  += val * val;

                if (abs_val > off_max_abs)       off_max_abs = abs_val;
                if (abs_val > threshold_nonzero) n_above_005++;
                if (abs_val > threshold_cousin)  n_above_0125++;
            }
        }

        double off_mean     = off_sum     / off_num;
        double off_abs_mean = off_abs_sum / off_num;
        // Single-pass variance: Var = E[x^2] - E[x]^2
        double off_var      = (off_sum_sq / off_num) - (off_mean * off_mean);

        LOGGER << "\nOff-diagonal statistics:" << std::endl;
        LOGGER << "  Mean:                    " << off_mean     << std::endl;
        LOGGER << "  Mean absolute value:     " << off_abs_mean << "  (primary density metric)" << std::endl;
        LOGGER << "  Variance:                " << off_var      << std::endl;
        LOGGER << "  SD:                      " << std::sqrt(off_var) << std::endl;
        LOGGER << "  Max absolute value:      " << off_max_abs  << std::endl;
        LOGGER << "  Proportion |off-diag| > " << threshold_nonzero << ": "
               << (double)n_above_005  / off_num
               << " (" << n_above_005  << " / " << (long long)off_num << ")" << std::endl;
        LOGGER << "  Proportion |off-diag| > " << threshold_cousin  << " (~first-cousin): "
               << (double)n_above_0125 / off_num
               << " (" << n_above_0125 << " / " << (long long)off_num << ")" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Eigenvalue-based metrics
    // Branching strategy:
    //   - effective-rank needs all eigenvalues -> full SelfAdjointEigenSolver
    //     with EigenvaluesOnly (skip eigenvector computation for speed)
    //   - conditioning alone -> Spectra to get only lambda_max and lambda_min,
    //     O(n^2 * k) instead of O(n^3)
    // -------------------------------------------------------------------------
    if (do_cond || do_effrank) {

        // Helper: shared negative-eigenvalue warning, printed once regardless
        // of which sub-metric triggered the decomposition.
        auto warn_negative = [&](const Eigen::VectorXd& evals) {
            int n_neg = (int)(evals.array() < 0.0).count();
            if (n_neg > 0)
                LOGGER << "\n  Warning: " << n_neg
                       << " negative eigenvalue(s) detected (GRM is not positive semi-definite)." << std::endl;
        };

        // Helper: conditioning output, reused by both paths.
        auto log_conditioning = [&](double eval_max, double eval_min) {
            double cond_num = (std::abs(eval_min) > 0.0)
                ? eval_max / std::abs(eval_min)
                : std::numeric_limits<double>::infinity();
            LOGGER << "\nConditioning:" << std::endl;
            LOGGER << "  Largest eigenvalue:              " << eval_max << std::endl;
            LOGGER << "  Smallest eigenvalue:             " << eval_min << std::endl;
            LOGGER << "  Condition number (max / |min|):  " << cond_num << std::endl;
        };

        if (do_effrank) {
            // -------------------------------------------------------------------
            // Full decomposition path (EigenvaluesOnly — no eigenvectors computed)
            // -------------------------------------------------------------------
            LOGGER << "\nComputing eigendecomposition (values only)..." << std::endl;

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(grm_dbl, Eigen::EigenvaluesOnly);
            if (eigs.info() != Eigen::Success)
                LOGGER.e(0, "eigenvalue decomposition failed.");

            const Eigen::VectorXd& evals = eigs.eigenvalues(); // ascending order
            warn_negative(evals);

            if (do_cond)
                log_conditioning(evals(n - 1), evals(0));

            // Effective rank -------------------------------------------------------
            Eigen::VectorXd abs_evals = evals.cwiseAbs();
            double sum1 = abs_evals.sum();
            double sum2 = abs_evals.squaredNorm();

            // Participation ratio == n_eff:
            //   (sum lambda_i)^2 / sum(lambda_i^2)
            //   Equals n for identity (all equal / unrelated), falls toward 1
            //   for highly structured or related populations.
            double participation_ratio = (sum2 > 0.0) ? (sum1 * sum1) / sum2 : 0.0;

            // Entropy-based effective rank: exp(Shannon entropy of normalised eigenvalues).
            // More sensitive to the full distribution shape than participation ratio.
            // Guard log(0) with .select() — zero eigenvalues contribute 0 by convention.
            Eigen::VectorXd p     = abs_evals / sum1;
            Eigen::VectorXd log_p = (p.array() > 0.0).select(p.array().log(),
                                                               Eigen::VectorXd::Zero(n));
            double entropy      = -(p.array() * log_p.array()).sum();
            double entropy_rank = std::exp(entropy);

            LOGGER << "\nEffective rank:" << std::endl;
            LOGGER << "  Participation ratio (n_eff):     " << participation_ratio
                   << "  (out of " << n << ")" << std::endl;
            LOGGER << "  Entropy-based effective rank:    " << entropy_rank
                   << "  (out of " << n << ")" << std::endl;

        } else {
            // -------------------------------------------------------------------
            // Fast path: conditioning only via Spectra (largest + smallest eval)
            // O(n^2 * k) vs O(n^3) for full decomposition — significant for
            // large GRMs (n > ~3000).
            // -------------------------------------------------------------------
            LOGGER << "\nComputing condition number (Spectra fast path)..." << std::endl;

            Spectra::DenseSymMatProd<double> op(grm_dbl);
            int ncv = std::min(n, 20); // Krylov subspace size; 20 is sufficient for k=1

            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> solver_max(op, 1, ncv);
            solver_max.init();
            solver_max.compute(Spectra::SortRule::LargestAlge);

            Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> solver_min(op, 1, ncv);
            solver_min.init();
            solver_min.compute(Spectra::SortRule::SmallestAlge);

            if (solver_max.info() != Spectra::CompInfo::Successful ||
                solver_min.info() != Spectra::CompInfo::Successful)
                LOGGER.e(0, "Spectra eigenvalue solver failed.");

            double eval_max = solver_max.eigenvalues()(0);
            double eval_min = solver_min.eigenvalues()(0);

            // Spectra doesn't give us all eigenvalues, so warn only if eval_min < 0
            if (eval_min < 0.0)
                LOGGER << "\n  Warning: smallest eigenvalue is negative (" << eval_min
                       << "); GRM may not be positive semi-definite." << std::endl;

            log_conditioning(eval_max, eval_min);
        }
    }
}
void gcta::pca(std::string grm_file, std::string keep_indi_file, std::string remove_indi_file, double grm_cutoff, bool merge_grm_flag, int out_pc_num, std::string pca_approx)
{
    manipulate_grm(grm_file, keep_indi_file, remove_indi_file, "", grm_cutoff, -2.0, -2, merge_grm_flag, true);
    _grm_N.resize(0, 0);
    int n = _keep.size();
    if (out_pc_num > n || out_pc_num == 0) out_pc_num = n;
    LOGGER << "\nPerforming principal component analysis ..." << _grm.rows() << "x" << _grm.cols() << std::endl;
    LOGGER << "\nNote: PCA V1 is deprecated, you should be hitting the V2 path by default." << std::endl;

    //_grm is either MatrixXf or MatrixXd; if it's MatrixXf, we need to cast it to MatrixXd. If it's already MatrixXd, we can use it directly.
    Eigen::MatrixXd grm_dbl_storage;
    const Eigen::MatrixXd& grm_dbl = [&]() -> const Eigen::MatrixXd& {
        if constexpr (std::is_same_v<eigenMatrix, Eigen::MatrixXd>) {
            return _grm;
        } else {
            grm_dbl_storage = _grm.cast<double>();
            return grm_dbl_storage;
        }
    }();
    Eigen::VectorXd eval;
    Eigen::MatrixXd evec;
    bool used_dsyevd = false;
    const double* raw_evec = nullptr;

    if (out_pc_num == n && !pca_approx.empty()) {
        LOGGER << "Warning: --pca-approx is set, but all PCs requested. Falling back to full eigenvalue decomposition." << std::endl;
        pca_approx="";
    }
    if (!pca_approx.empty()) {

        // Shared matvec: grm_dbl is fully populated (both triangles), so a
        // full GEMV/GEMM is used rather than a triangular BLAS2 dsymv/dsymm —
        // full-matrix multithreaded GEMM tends to beat symmetric BLAS2/3
        // routines in practice despite the 2x flop count. Generic on X so it
        // serves both the block sketches (rSVD) and single vectors (Lanczos).
        auto apply = [&](const auto& X) -> Eigen::MatrixXd {
            const int cols = static_cast<int>(X.cols());
            if (X.cols() == 1) {
                Eigen::MatrixXd Y(n, 1);
                Y.col(0) = gcta_eigh::threaded_dense_matvec(grm_dbl, X.col(0));
                return Y;
            }
            Eigen::MatrixXd Y = grm_dbl * X;
            return Y;
        };

        try {
            gcta_eigh::EighResult res;
            if (pca_approx == "rSVD") {
                const int oversample     = gcta_eigh::recommended_oversample(out_pc_num);
                const int pca_power_iter = 3;
                res = gcta_eigh::randomized_symmetric_eigh(apply, n, out_pc_num, oversample, pca_power_iter);
            } else if (pca_approx == "Lanczos") {
                const int ncv = std::min(n, std::max(3 * out_pc_num + 1, 30));
                res = gcta_eigh::lanczos_symmetric_eigh(apply, n, out_pc_num, ncv);
            } else {
                LOGGER.e(0, "--pca-approx: unrecognised method '" + pca_approx + "'. Use 'Lanczos' or 'rSVD'.");
            }
            eval = std::move(res.eigenvalues);
            evec = std::move(res.eigenvectors);
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("--pca-approx failed: ") + e.what());
        }
    } else {
        if (out_pc_num == n && n >= 32766)
            LOGGER << "Warning: n = " << n << " may exceed dsyevd's safe workspace limit "
                      "(n >= 32766). Consider using --pca-approx for large GRMs." << std::endl;

        // Pass the GRM storage directly to LAPACK (it is overwritten in-place).
        // grm_dbl is a const-ref to _grm (or grm_dbl_storage), both genuinely
        // non-const; we accept the destruction because PCA no longer needs the GRM.
        double* grm_ptr = const_cast<double*>(grm_dbl.data());

        if (out_pc_num == n+1) {
            // Full spectrum via dsyevd: overwrites GRM in-place; eigenvectors are read
            // back via raw_evec (no extra n×n copy). Eigenvalues stored ascending → reversed.
            Eigen::VectorXd w(n);
            int info = gcta_dsyevd((gcta_blas_int)n, grm_ptr, (gcta_blas_int)n, w.data());
            if (info != 0)
                LOGGER.e(0, "dsyevd failed (info=" + std::to_string(info) +
                             "). For n > 32766, try --pca-approx.");
            eval = w.reverse();
            used_dsyevd = true;
            raw_evec = grm_ptr;
        } else if (out_pc_num == n) {
            Eigen::VectorXd w(n);
int info = gcta_dsyev((gcta_blas_int)n, grm_ptr, (gcta_blas_int)n, w.data());
if (info != 0)
    LOGGER.e(0, "dsyev failed (info=" + std::to_string(info) + ").");
eval = w.reverse();
used_dsyevd = false;   // not D&C — flag if this bool gates any downstream logic
raw_evec = grm_ptr;
        } else {
            // Partial spectrum: dsyevr overwrites the GRM in-place but writes the
            // requested eigenvectors to the separate Z buffer (n × out_pc_num),
            // which is much smaller than n×n.
            Eigen::VectorXd            w(out_pc_num);
            Eigen::MatrixXd            Z(n, out_pc_num);
            gcta_blas_int              m_found = 0;
            std::vector<gcta_blas_int> isuppz(2 * out_pc_num);
            gcta_blas_int il = (gcta_blas_int)(n - out_pc_num + 1);
            gcta_blas_int iu = (gcta_blas_int)n;
            int info = gcta_dsyevr((gcta_blas_int)n, grm_ptr, (gcta_blas_int)n,
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

    std::string eval_file = _out + ".eigenval";
    std::ofstream o_eval(eval_file.c_str());
    if (!o_eval) LOGGER.e(0, "cannot open the file [" + eval_file + "] to read.");
    for (int i = 0; i < out_pc_num; i++) o_eval << eval(i) << std::endl;
    o_eval.close();
    LOGGER << "Eigenvalues of " << n << " individuals have been saved in [" + eval_file + "]." << std::endl;
    std::string evec_file = _out + ".eigenvec";
    std::ofstream o_evec(evec_file.c_str());
    if (!o_evec) LOGGER.e(0, "cannot open the file [" + evec_file + "] to read.");
    for (int i = 0; i < n; i++) {
        o_evec << _fid[_keep[i]] << " " << _pid[_keep[i]];
        if (used_dsyevd)
            for (int j = 0; j < out_pc_num; j++)
                o_evec << " " << raw_evec[(size_t)(out_pc_num - 1 - j) * n + i];
        else
            for (int j = 0; j < out_pc_num; j++)
                o_evec << " " << evec(i, j);
        o_evec << std::endl;
    }
    o_evec.close();
    LOGGER << "The first " << out_pc_num << " eigenvectors of " << n << " individuals have been saved in [" + evec_file + "]." << std::endl;
}

void gcta::snp_pc_loading(std::string pc_file)
{
    // read eigenvectors and eigenvalues
    std::string eigenval_file = pc_file + ".eigenval";
    std::ifstream in_eigenval(eigenval_file.c_str());
    if (!in_eigenval) LOGGER.e(0, "cannot open the file [" + eigenval_file + "] to read.");
    std::string eigenvec_file = pc_file + ".eigenvec";
    std::ifstream in_eigenvec(eigenvec_file.c_str());
    if (!in_eigenvec) LOGGER.e(0, "cannot open the file [" + eigenvec_file + "] to read.");
  
    LOGGER << "Reading eigenvectors from [" + eigenvec_file + "]." << std::endl;
    std::vector<std::string> eigenvec_ID;
    std::vector< std::vector<std::string> > eigenvec_str;
    int eigenvec_num = read_fac(in_eigenvec, eigenvec_ID, eigenvec_str);
    LOGGER << eigenvec_num << " eigenvectors of " << eigenvec_ID.size() << " individuals are included from [" + eigenvec_file + "]." << std::endl;
    update_id_map_kp(eigenvec_ID, _id_map, _keep);

    LOGGER << "\nReading eigenvalues from [" + eigenval_file + "]." << std::endl;
    std::vector<double> eigenval_buf;
    double d_buf = 0.0;
    int eigenval_num = 0;
    while(in_eigenval && eigenval_num < eigenvec_num){
        in_eigenval >> d_buf;
        if(d_buf > 1e10 || d_buf < 1e-10) LOGGER.e(0, "invalid eigenvalue in the file [" + eigenval_file + "].");
        eigenval_buf.push_back(d_buf);
        eigenval_num++;
    }
    if(eigenvec_num != eigenval_num) LOGGER.e(0, "inconsistent numbers of eigenvalues and eigenvectors in the files [" + eigenval_file + "] and [" + eigenvec_file + "]");
    LOGGER << eigenval_num << " eigenvalues are read from [" + eigenval_file + "]" << std::endl;  

    int i = 0, j = 0;
    std::vector<std::string> uni_id;
    std::map<std::string, int> uni_id_map;
    make_uni_id(uni_id, uni_id_map);
    _n = _keep.size();
    int m = _include.size();
    if(_n < 1) LOGGER.e(0, "no individual is in common among the input files.");
    LOGGER << _n << " individuals in common between the input files are included in the analysis."<<std::endl;
    
    eigenMatrix eigenvec(eigenvec_num, _n);
    //GCC13 has a bug preventing the more optimal structured binding approach here, so we use a traditional indexed loop instead.
    // for (const auto& [id, evec_row] : std::views::zip(eigenvec_ID, eigenvec_str)) {
    //   if (auto it = uni_id_map.find(id); it != uni_id_map.end()) {
    for (size_t _i = 0; _i < eigenvec_ID.size(); ++_i) {
      const auto& id      = eigenvec_ID[_i];
      const auto& evec_row = eigenvec_str[_i];
      if (auto it = uni_id_map.find(id); it != uni_id_map.end()) {
            for (j = 0; j < eigenvec_num; j++)
                eigenvec(j, it->second) = std::stod(evec_row[j]);
        }
    }

    eigenVector inv_eigenval(eigenval_num);
    for(i = 0; i < eigenval_num; i++)  inv_eigenval(i) = 1.0 / (eigenval_buf[i] * m);

    // calculating SNP loading
    if (_mu.empty()) calcu_mu();
    LOGGER << "\nCalculating PC loadings of SNPs..." << std::endl;
    eigenMatrix snp_loading(m, eigenvec_num);
    eigenVector x(_n);
    for(j = 0; j < m ; j++) {
        makex_eigenVector(j, x, false, true);
        x = x.array() / sqrt(_mu[_include[j]]*(1.0 - 0.5*_mu[_include[j]]));
        snp_loading.row(j) = (eigenvec * x).array() * inv_eigenval.array();
    }

    std::string filename = _out + ".pcl";
    LOGGER << "\nSaving the PC loadings of " << m << " SNPs to [" + filename + "] ..." << std::endl;
    std::ofstream ofile(filename.c_str());
    if(!ofile) LOGGER.e(0, "cannot open the file [" + filename + "] to write.");
    ofile << "SNP\tA1\tA2\tmu";
    for(i = 0; i < eigenval_num; i++) ofile << "\tpc" << i+1 << "_loading";
    ofile << std::endl;
    for(i = 0; i < m; i++){
        ofile << _snp_name[_include[i]] << "\t" << _allele_alt[_include[i]] << "\t" << _allele_ref[_include[i]] << "\t" <<  _mu[_include[i]];
        for(j = 0; j < eigenvec_num; j++) ofile << "\t" << snp_loading(i, j);
        ofile << "\n";
    }
    ofile.close();
}

//This function changes the original _geno, _include, never use genotype variable after it!!
void gcta::project_loading(std::string pc_load, int N){

    #ifdef SINGLE_PRECISION
    typedef float t_val;
    #else
    typedef double t_val;
    #endif

    std::string f_pc_load = pc_load + ".pcl";
    std::ifstream h_pc_load(f_pc_load.c_str());
    if (!h_pc_load) LOGGER.e(0, "cannot open the loading file [" + f_pc_load + "] to read.");
    
    // output eigenvec. Moving this up to save the time for user when running in an unwritable directory.
    std::string out_filename = _out + ".proj.eigenvec";
    //LOGGER << "\nOpen the project file [" << out_filename << "] to write."<< std::endl;
    std::ofstream ofile(out_filename.c_str());
    if(!ofile) LOGGER.e(0, "failed to open the file [" + out_filename + "] to write.");

    LOGGER << "Reading PC loadings of SNPs from [" + f_pc_load + "]." << std::endl;
    std::string buf;
    std::vector<std::string> vsec_buf;
    std::string header;
    std::getline(h_pc_load,header);
    size_t N_loading_file = count(header.begin(),header.end(),'p');
    LOGGER << "Number of PC loading vectors: " << N_loading_file << std::endl;
    while(h_pc_load >> buf){
        vsec_buf.push_back(buf);
    }
    h_pc_load.close();
    buf.clear();
    header.clear();

    int len_col = N_loading_file + 4;
    int num_snp = vsec_buf.size() / len_col;
    LOGGER << "Number of SNPs in loading array: " << num_snp << "." << std::endl; 
    if(vsec_buf.size() % len_col != 0){
        LOGGER.e(0, "invalid number of columns in the PC loading file. Please check.");
    }
    if(N > N_loading_file){
        LOGGER.e(0, "only " + std::to_string(N_loading_file) + " vectors of PC loadings available, thus not able to project onto " + std::to_string(N) + " PCs.");
    }

    std::vector<std::string> snps(num_snp);
    std::vector<std::string> A1(num_snp);
    std::vector<std::string> A2(num_snp);
    std::vector<t_val> mu(num_snp);
    std::vector<t_val> snp_loading (num_snp * N);
    for(int read_snp_index=0; read_snp_index < num_snp; read_snp_index++){
       int base_index = read_snp_index * len_col;
       snps[read_snp_index] = vsec_buf[base_index];
       A1[read_snp_index] = vsec_buf[base_index + 1];
       A2[read_snp_index] = vsec_buf[base_index + 2];
       #ifdef SINGLE_PRECISION
       mu[read_snp_index] = atof(vsec_buf[base_index + 3].c_str());
       #else
       mu[read_snp_index] = stod(vsec_buf[base_index + 3]);
       #endif
       for(int N_index=0; N_index<N; N_index++){
           // atof .c_str()
            #ifdef SINGLE_PRECISION
            snp_loading[read_snp_index*N + N_index] = atof(vsec_buf[base_index+4+N_index].c_str());
            #else
            snp_loading[read_snp_index*N + N_index] = stod(vsec_buf[base_index+4+N_index]);
            #endif
        }
    }

    vsec_buf.clear();
    vsec_buf.shrink_to_fit();
    
    LOGGER << "Matching alleles..." << std::endl;
    std::vector<int> snp_index_include(snps.size());
    StrFunc::match(snps,_snp_name,snp_index_include);
    // _include should be fixed here, it cause maf caculation go vain; 
    std::unordered_set<int> ori_SNPs(_include.begin(),_include.end());
    _include.clear();

    std::vector<t_val> filter_snp_loading;
    filter_snp_loading.reserve(num_snp * N);

    LOGGER << "Adjusting A1" << std::endl;
    bool remove_flag = true;
    std::vector<std::string> missnp_list;
    std::vector<t_val> mu_adj;
    for(int snp_index=0; snp_index < snps.size(); snp_index++){
        int cur_snp_index = snp_index_include[snp_index];
        if(cur_snp_index >= 0 && ori_SNPs.find(cur_snp_index) != ori_SNPs.end() ){
            if((StrFunc::i_compare(A1[snp_index],_allele_alt[cur_snp_index]) && 
                StrFunc::i_compare(A2[snp_index],_allele_ref[cur_snp_index])) 
               || (StrFunc::i_compare(A1[snp_index],_allele_ref[cur_snp_index]) && 
                StrFunc::i_compare(A2[snp_index],_allele_alt[cur_snp_index]))){
                     _include.push_back(cur_snp_index);
                     mu_adj.push_back(mu[snp_index]);
                     _allele_ref[cur_snp_index] = A1[snp_index];
                     filter_snp_loading.insert(filter_snp_loading.end(), snp_loading.begin() + N*snp_index, snp_loading.begin() + N*snp_index + N );
                     continue;
            }
        }
        missnp_list.push_back(snps[snp_index]);
    }

    snp_loading.clear();
    snp_loading.shrink_to_fit();

    //Map the std::vector to Matrix, share the same memory, thus to save the memory.
    eigenMatrix m_snp_loading = Eigen::Map< Eigen::Matrix<t_val,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> > (filter_snp_loading.data(), _include.size(), N);
    LOGGER << " " << m_snp_loading.rows() << " SNPs are included for loading" << std::endl;

    if(missnp_list.size() > 0){
        LOGGER.w(0, std::to_string(missnp_list.size()) + " SNPs are not found or alleles mismatch in the target genotype"); 
        std::string miss_file = _out + ".proj.missnp";
        LOGGER << " See [" << miss_file << "] for details. If there are many missing SNPs, the projection might be biased." << std::endl;
        std::ofstream h_miss(miss_file);
        std::ostream_iterator<std::string> output_iterator(h_miss,"\n");
        copy(missnp_list.begin(), missnp_list.end(), output_iterator);
    }
    missnp_list.clear();
    missnp_list.shrink_to_fit();

    //if(_mu.empty()) calcu_mu();
    LOGGER << "Standardizing genotypes and projecting PCs..." << std::endl;
    LOGGER << "Total number of subjects: " << _keep.size() << "\n" << std::endl;
    //std::ofstream demo(_out + ".proj.matrix");
    LOGGER << "Processing subject number: " << std::endl;
    eigenMatrix PCs(_keep.size(),N);
    #pragma omp parallel for ordered schedule(dynamic)
    for(int ind_index=0; ind_index < _keep.size(); ind_index++){
        LOGGER <<  std::to_string(ind_index+1) + "\r" << std::flush;
        Eigen::Matrix<t_val,1,Eigen::Dynamic> geno(_include.size());
        for(int snp_index=0; snp_index < _include.size(); snp_index++){
            if (!_snp_1[_include[snp_index]][_keep[ind_index]] || _snp_2[_include[snp_index]][_keep[ind_index]]) {
                geno(snp_index) = _snp_1[_include[snp_index]][_keep[ind_index]] + _snp_2[_include[snp_index]][_keep[ind_index]];                geno(snp_index) = (geno(snp_index) - mu_adj[snp_index]) / sqrt(mu_adj[snp_index]*(1.0 - 0.5*mu_adj[snp_index]));
            }else{
                geno(snp_index) = 0.0;
            }
        }
        PCs.row(ind_index) = geno * m_snp_loading;
        /*
        if(ind_index==0){
            demo << "geno:" << std::endl;
            demo << geno << std::endl;
            demo << "m_snp_loading" << std::endl;
            demo << m_snp_loading << std::endl;
            demo << "PC" << std::endl;
            demo << PCs << std::endl;
            demo.close();
        }
        */
        geno.resize(0);
    }
    
    // Output the values
    for(int ind_index=0; ind_index < _keep.size(); ind_index++){
        ofile << _fid[_keep[ind_index]] << "\t" << _pid[_keep[ind_index]] << "\t";
        for(int pc_index=0; pc_index<N; pc_index++){
            ofile << PCs(ind_index,pc_index) << "\t"; 
        }
        ofile << "\n";
    }
    ofile.close();
    
    LOGGER << "\nFinished, and the PCs have all been saved to " << out_filename << std::endl;
}
