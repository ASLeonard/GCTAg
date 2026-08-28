/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * Implementations of functions for data management
 *
 * 2010 by Jian Yang <jian.yang.qt@gmail.com>
 *
 * This file is distributed under the GNU General Public
 * License, Version 3.  Please see the file LICENSE for more
 * details
 */

#include <sstream>
#include <charconv>
#include <iterator>
#include <set>
#include <fstream>
#include <limits>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include "gcta.h"
#include "Logger.h"
#include "StrFunc.h"

// (DOSAGE_NA is declared in gcta.h)

gcta::gcta(int autosome_num, double rm_ld_cutoff, std::string out)
{
    // autosome_num <= 0 means no explicit autosome upper bound was provided.
    _autosome_num = autosome_num;
    _legacy_homogametic_chr = 0;
    _rm_ld_cutoff = rm_ld_cutoff;
    _out = out;
    _dosage_flag = false;
    _genetic_model = GeneticModel::ADDITIVE;
    _grm_bin_flag = false;
    _reml_mtd = 0;
    _reml_inv_mtd = 0;
    _reml_max_iter = 30;
    _jma_actual_geno = false;
    _jma_wind_size = 1e7;
    _jma_p_cutoff = 5e-8;
    _jma_collinear = 0.9;
    _jma_Vp = 1;
    _GC_val = 1;
    _bivar_reml = false;
    _ignore_Ce = false;
    _bivar_no_constrain = false;
    _y_Ssq = 0.0;
    _y2_Ssq = 0.0;
    _ncase = 0;
    _ncase2 = 0;
    _flag_CC = false;
    _flag_CC2 = false;
    _within_family = false;
    _reml_have_bend_A = false;
    _V_inv_mtd = 0;
    _reml_force_inv = false;
    _reml_AI_not_invertible = false;
    _reml_force_converge = false;
    _reml_no_converge = false;
    _reml_fixed_var = false;
    _ldscore_adj = false;
    _log_pval = false;
}

gcta::gcta() {
    _legacy_homogametic_chr = 0;
    _dosage_flag = false;
    _genetic_model = GeneticModel::ADDITIVE;
    _grm_bin_flag = false;
    _reml_mtd = 0;
    _reml_inv_mtd = 0;
    _reml_max_iter = 30;
    _jma_actual_geno = false;
    _jma_wind_size = 1e7;
    _jma_p_cutoff = 5e-8;
    _jma_collinear = 0.9;
    _jma_Vp = 1;
    _GC_val = 1;
    _bivar_reml = false;
    _ignore_Ce = false;
    _bivar_no_constrain = false;
    _y_Ssq = 0.0;
    _y2_Ssq = 0.0;
    _ncase = 0;
    _ncase2 = 0;
    _flag_CC = false;
    _flag_CC2 = false;
    _within_family = false;
    _reml_have_bend_A = false;
    _V_inv_mtd = 0;
    _reml_force_inv = false;
    _reml_AI_not_invertible = false;
    _reml_force_converge = false;
    _reml_no_converge = false;
    _reml_fixed_var = false;
    _ldscore_adj = false;
    _log_pval = false;
}

gcta::~gcta() {

}

void gcta::set_genetic_model(std::string model) {
    if (!stringToGeneticModel(model, _genetic_model)) {
        LOGGER.e(0, "genetic model must be either 'additive' or 'nonadditive'.");
    }
    LOGGER << "Genetic model set to: " << model << std::endl;
}

void gcta::read_famfile(std::string famfile) {
    std::ifstream Fam(famfile.c_str());
    if (!Fam) LOGGER.e(0, "cannot open the file [" + famfile + "] to read.");
    LOGGER << "Reading PLINK FAM file from [" + famfile + "]." << std::endl;

    int i = 0;
    std::string str_buf;
    _fid.clear();
    _pid.clear();
    _fa_id.clear();
    _mo_id.clear();
    _sex.clear();
    _pheno.clear();
    while (Fam) {
        Fam >> str_buf;
        if (Fam.eof()) break;
        _fid.push_back(str_buf);
        Fam >> str_buf;
        _pid.push_back(str_buf);
        Fam >> str_buf;
        _fa_id.push_back(str_buf);
        Fam >> str_buf;
        _mo_id.push_back(str_buf);
        Fam >> str_buf;
        _sex.push_back(atoi(str_buf.c_str()));
        Fam >> str_buf;
        _pheno.push_back(atoi(str_buf.c_str()));
    }
    Fam.clear();
    Fam.close();
    _indi_num = _fid.size();
    LOGGER << _indi_num << " individuals to be included from [" + famfile + "]." << std::endl;

    // Initialize _keep
    init_keep();
}

void gcta::make_uni_id(std::vector<std::string> &uni_id, std::map<std::string, int> &uni_id_map) {
    uni_id.resize(_keep.size());
    for (size_t i = 0; i < _keep.size(); i++) {
        uni_id[i] = _fid[_keep[i]] + ":" + _pid[_keep[i]];
        uni_id_map.emplace(uni_id[i], static_cast<int>(i));
    }
}   

void gcta::init_keep() {
    _keep.clear();
    _keep.resize(_indi_num);
    _id_map.clear();
    int i = 0, size = 0;
    for (i = 0; i < _indi_num; i++) {
        _keep[i] = i;
        _id_map.insert(std::pair<std::string, int>(_fid[i] + ":" + _pid[i], i));
        if (size == _id_map.size()) LOGGER.e(0, "Duplicate individual ID found: \"" + _fid[i] + "\t" + _pid[i] + "\".");
        size = _id_map.size();
    }
}

void gcta::read_bimfile(std::string bimfile) {
    // Read bim file: recombination rate is defined between SNP i and SNP i-1
    int ibuf = 0;
    std::string cbuf = "0";
    double dbuf = 0.0;
    std::string str_buf;
    std::ifstream Bim(bimfile.c_str());
    if (!Bim) LOGGER.e(0, "cannot open the file [" + bimfile + "] to read.");
    LOGGER << "Reading PLINK BIM file from [" + bimfile + "]." << std::endl;
    _chr.clear();
    _snp_name.clear();
    _genet_dst.clear();
    _bp.clear();
    _allele_alt.clear();
    _allele_ref.clear();
    while (Bim) {
        Bim >> ibuf;
        if (Bim.eof()) break;
        _chr.push_back(ibuf);
        Bim >> str_buf;
        _snp_name.push_back(str_buf);
        Bim >> dbuf;
        _genet_dst.push_back(dbuf);
        Bim >> ibuf;
        _bp.push_back(ibuf);
        Bim >> cbuf;
        StrFunc::to_upper(cbuf);
        _allele_alt.push_back(cbuf);
        Bim >> cbuf;
        StrFunc::to_upper(cbuf);
        _allele_ref.push_back(cbuf);
    }
    Bim.close();
    _snp_num = _chr.size();
    LOGGER << _snp_num << " SNPs to be included from [" + bimfile + "]." << std::endl;

    // Initialize _include
    init_include();
}

void gcta::init_include()
{
    _include.clear();
    _include.resize(_snp_num);
    _snp_name_map.clear();
    int i = 0, size = 0;
    for (i = 0; i < _snp_num; i++) {
        _include[i] = i;
        if(_snp_name_map.find(_snp_name[i]) != _snp_name_map.end()){
            std::string orig = _snp_name[i];
            std::stringstream ss;
            ss << _snp_name[i] << "_" << i + 1;
            _snp_name[i] = ss.str();
            LOGGER.w(0, "Duplicated SNP ID \"" + orig + "\" has been changed to \"" + _snp_name[i] + "\".");
        }
        _snp_name_map.insert(std::pair<std::string, int>(_snp_name[i], i));
    }
}

// some code are adopted from PLINK with modifications
void gcta::read_bedfile(std::string bedfile)
{
    int i = 0, j = 0, k = 0;

    // Flag for reading individuals and SNPs
    std::vector<int> rindi, rsnp;
    get_rindi(rindi);
    get_rsnp(rsnp);

    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");
    if (_keep.size() == 0) LOGGER.e(0, "no individual is retained for analysis.");

    // Read bed file
    const size_t bytes_per_snp = ((size_t)_indi_num + 3) / 4;
    const int m = (int)_include.size();
    const int n_keep = (int)_keep.size();

    _snp_1.resize(m);
    _snp_2.resize(m);
    for (i = 0; i < m; i++) {
        _snp_1[i].assign(n_keep, false);
        _snp_2[i].assign(n_keep, false);
    }

    LOGGER << "Reading PLINK BED file from [" + bedfile + "] in SNP-major format ..." << std::endl;

    // mmap the bed file — zero-copy, OS pages in on demand
    int bed_fd = open(bedfile.c_str(), O_RDONLY);
    if (bed_fd < 0) LOGGER.e(0, "cannot open the file [" + bedfile + "] to read.");
    struct stat bed_st;
    fstat(bed_fd, &bed_st);
    size_t file_size = (size_t)bed_st.st_size;
    if (file_size < 3 + (size_t)_snp_num * bytes_per_snp)
        LOGGER.e(0, "problem with the BED file ... has the FAM/BIM file been changed?");
    const unsigned char* bed_buf = (const unsigned char*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, bed_fd, 0);
    close(bed_fd);
    if (bed_buf == MAP_FAILED) LOGGER.e(0, "mmap failed for [" + bedfile + "].");
    // Hint: we'll touch every page, so preload in background
    madvise((void*)bed_buf, file_size, MADV_WILLNEED);

    // Pre-compute file offset for each included SNP (dense index 0..m-1)
    std::vector<size_t> snp_offsets(m);
    for (j = 0, k = 0; j < _snp_num; j++) {
        if (rsnp[j]) snp_offsets[k++] = 3 + (size_t)j * bytes_per_snp;
    }

    // Pre-compute byte offset and bit shift within a SNP block for each kept individual.
    // Eliminates the per-individual rindi[] branch from the hot loop.
    std::vector<int> indi_byte(n_keep), indi_shift(n_keep);
    for (j = 0, k = 0; j < _indi_num; j++) {
        if (rindi[j]) {
            indi_byte[k]  = j / 4;
            indi_shift[k] = (j % 4) * 2;
            k++;
        }
    }

    // Parallel decode: each SNP slot is independent — no shared writes
    #pragma omp parallel for schedule(static) private(j)
    for (i = 0; i < m; i++) {
        const unsigned char* snp_data = bed_buf + snp_offsets[i];
        for (j = 0; j < n_keep; j++) {
            int genotype = (snp_data[indi_byte[j]] >> indi_shift[j]) & 3;
            _snp_2[i][j] = !(genotype & 1);
            _snp_1[i][j] = !((genotype >> 1) & 1);
        }
    }

    munmap((void*)bed_buf, file_size);
    LOGGER << "Genotype data for " << _keep.size() << " individuals and " << _include.size() << " SNPs to be included from [" + bedfile + "]." << std::endl;

    update_fam(rindi);
    update_bim(rsnp);
}

/**
 * Read PLINK bed file and fill _geno_dose matrix with dosage values
 * calculated from genotypes and allele frequencies.
 * 
 * This function performs two passes over the bed file:
 * Pass 1: Calculate allele frequencies from the genotype data (for the reference allele)
 * Pass 2: Convert genotypes to dosages using the calculated frequencies
 * 
 * Genotype encoding in PLINK bed format (SNP-major):
 *   00 -> Homozygous for allele 1 (coded as 2)
 *   01 -> Missing (coded as DOSAGE_NA)
 *   10 -> Heterozygous (coded as 1)
 *   11 -> Homozygous for allele 2 (coded as 0)
 *
 * Reference allele frequency:
 *   - Always calculated for the allele specified in _allele_ref
 *   - Stored in _mu as mean allele count (0-2 range)
 *   - To get allele frequency in 0-1 range: divide _mu by 2
 * 
 * Dosage calculation depends on the genetic model (_genetic_model):
 * 
 * Additive model (default):
 *   - RR (homozygous for reference) = 0
 *   - RA (heterozygous) = 1
 *   - AA (homozygous for alternate) = 2
 * 
 * Nonadditive model:
 *   - RR (homozygous for reference) = 0
 *   - RA (heterozygous) = 2p (where p is allele frequency)
 *   - AA (homozygous for alternate) = 4p - 2
 *   - Missing values are set to DOSAGE_NA
 * 
 * Prerequisites:
 *   - read_famfile() must be called first to initialize individuals
 *   - read_bimfile() must be called first to initialize SNPs and alleles
 *   - set_genetic_model() can be called to std::set the model (default: "additive")
 * 
 * @param bedfile Path to the PLINK .bed file
 * 
 * Example usage:
 *   gcta mydata;
 *   mydata.read_famfile("mydata.fam");
 *   mydata.read_bimfile("mydata.bim");
 *   mydata.set_genetic_model("nonadditive");  // Optional
 *   mydata.read_bed_dosage("mydata.bed");
 *   // Now _geno_dose matrix is filled with dosage values
 */

void gcta::read_bed_dosage(std::string bedfile)
{
    std::vector<int> rindi, rsnp;
    get_rindi(rindi);
    get_rsnp(rsnp);

    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");
    if (_keep.size() == 0) LOGGER.e(0, "no individual is retained for analysis.");

    _dosage_flag = true;

    const int n = static_cast<int>(_keep.size());
    const int m = static_cast<int>(_include.size());
    const size_t bytes_per_snp = (static_cast<size_t>(_indi_num) + 3) / 4;

    LOGGER << "Reading PLINK BED file from [" + bedfile + "] in SNP-major format (dosage mode) ..." << std::endl;

    int bed_fd = open(bedfile.c_str(), O_RDONLY);
    if (bed_fd < 0) LOGGER.e(0, "cannot open the file [" + bedfile + "] to read.");
    struct stat bed_st;
    fstat(bed_fd, &bed_st);
    size_t file_size = static_cast<size_t>(bed_st.st_size);
    if (file_size < 3 + static_cast<size_t>(_snp_num) * bytes_per_snp) {
        close(bed_fd);
        LOGGER.e(0, "problem with the BED file ... has the FAM/BIM file been changed?");
    }
    const unsigned char* bed_buf = static_cast<const unsigned char*>(
        mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, bed_fd, 0));
    close(bed_fd);
    if (bed_buf == MAP_FAILED) LOGGER.e(0, "mmap failed for [" + bedfile + "].");
    madvise(const_cast<unsigned char*>(bed_buf), file_size, MADV_WILLNEED);

    _geno_dose.setZero(n, m);
    _mu.assign(m, 0.0);
    _additive_mu.assign(m, 0.0);

    static constexpr int8_t lut[4] = { 2, -1, 1, 0 };

    // Precompute the compact list of kept SNP indices once, so the parallel
    // loop can index snp_indx directly instead of relying on a serial
    // increment-under-`continue` (which doesn't parallelize).
    std::vector<int> kept_snp_idx;
    kept_snp_idx.reserve(m);
    for (int j = 0; j < _snp_num; j++) if (rsnp[j]) kept_snp_idx.push_back(j);

    std::atomic<bool> missing_warned{false};

    // Each iteration owns a distinct column of _geno_dose (column-major, so
    // no cross-thread false sharing) and distinct _mu[snp_indx]/_additive_mu[snp_indx]
    // entries. Per-SNP cost is uniform (same _indi_num scan every time), so
    // static scheduling avoids dynamic-scheduling overhead for no loss of balance.
    #pragma omp parallel for schedule(static)
    for (int snp_indx = 0; snp_indx < m; snp_indx++) {
        const int j = kept_snp_idx[snp_indx];
        const size_t snp_offset = 3 + static_cast<size_t>(j) * bytes_per_snp;
        const unsigned char* snp_data = bed_buf + snp_offset;

        int allele_count = 0, valid_count = 0;
        int indi_indx = 0;

        for (int i = 0; i < _indi_num; ) {
            const unsigned char byte = snp_data[i / 4];
            for (int bit = 0; bit < 8 && i < _indi_num; bit += 2, i++) {
                if (!rindi[i]) continue;
                const int8_t g = lut[(byte >> bit) & 3];
                if (g < 0) {
                    if (!missing_warned.exchange(true, std::memory_order_relaxed)) {
                        LOGGER.w(0, "missing values detected in the genotype data.");
                    }
                    _geno_dose(indi_indx, snp_indx) = DOSAGE_NA;
                } else {
                    _geno_dose(indi_indx, snp_indx) = static_cast<float>(g);
                    allele_count += g;
                    valid_count++;
                }
                indi_indx++;
            }
        }

        const double mu = (valid_count > 0) ? static_cast<double>(allele_count) / valid_count : 0.0;
        _mu[snp_indx] = mu;
        _additive_mu[snp_indx] = mu;

        // Fused, branchless nonadditive transform — runs while this column
        // is still hot in L2/L3, instead of a second full-matrix pass later.
        if (_genetic_model == GeneticModel::NONADDITIVE) {
            const float p2 = static_cast<float>(mu);
            auto col = _geno_dose.col(snp_indx).array();
            col = (col < DOSAGE_NA).select(col * (p2 + 1.0f - col), col);
        }
    }

    munmap(const_cast<unsigned char*>(bed_buf), file_size);

    LOGGER << "Dosage data for " << n << " individuals and " << m
           << " SNPs loaded from [" + bedfile + "] ("
           << geneticModelToString(_genetic_model) << " model)." << std::endl;

    update_fam(rindi);
    update_bim(rsnp);
}

void gcta::get_rsnp(std::vector<int> &rsnp) {
    rsnp.clear();
    rsnp.resize(_snp_num);
    for (int i = 0; i < _snp_num; i++) {
        if (_snp_name_map.find(_snp_name[i]) != _snp_name_map.end()) rsnp[i] = 1;
        else rsnp[i] = 0;
    }
}

void gcta::get_rindi(std::vector<int> &rindi) {
    rindi.clear();
    rindi.resize(_indi_num);
    for (int i = 0; i < _indi_num; i++) {
        if (_id_map.find(_fid[i] + ":" + _pid[i]) != _id_map.end()) rindi[i] = 1;
        else rindi[i] = 0;
    }
}

void gcta::update_bim(std::vector<int> &rsnp) {
    int write = 0;
    for (int i = 0; i < _snp_num; i++) {
        if (!rsnp[i]) continue;
        if (write != i) {
            _chr[write]       = _chr[i];
            _snp_name[write]  = std::move(_snp_name[i]);
            _genet_dst[write] = _genet_dst[i];
            _bp[write]        = _bp[i];
            _allele_alt[write]   = std::move(_allele_alt[i]);
            _allele_ref[write]   = std::move(_allele_ref[i]);
            if (!_impRsq.empty()) _impRsq[write] = _impRsq[i];
        }
        write++;
    }
    _chr.resize(write);
    _snp_name.resize(write);
    _genet_dst.resize(write);
    _bp.resize(write);
    _allele_alt.resize(write);
    _allele_ref.resize(write);
    if (!_impRsq.empty()) _impRsq.resize(write);

    _snp_num = write;
    _include.resize(_snp_num);
    _snp_name_map.clear();
    for (int i = 0; i < _snp_num; i++) {
        _include[i] = i;
        _snp_name_map.insert({_snp_name[i], i});
    }
}

void gcta::update_fam(std::vector<int> &rindi) {
    int write = 0;
    for (int i = 0; i < _indi_num; i++) {
        if (!rindi[i]) continue;
        if (write != i) {
            _fid[write]   = std::move(_fid[i]);
            _pid[write]   = std::move(_pid[i]);
            _fa_id[write] = std::move(_fa_id[i]);
            _mo_id[write] = std::move(_mo_id[i]);
            _sex[write]   = _sex[i];
            _pheno[write] = _pheno[i];
        }
        write++;
    }
    _fid.resize(write);
    _pid.resize(write);
    _fa_id.resize(write);
    _mo_id.resize(write);
    _sex.resize(write); 
    _pheno.resize(write);

    _indi_num = write;
    _keep.resize(_indi_num);
    _id_map.clear();
    for (int i = 0; i < _indi_num; i++) {
        _keep[i] = i;
        _id_map.insert({_fid[i] + ":" + _pid[i], i});
    }
}

// Compact individual-major data to the current _keep order and reset _keep to
// identity. This physically discards samples outside _keep and enables dense
// row access for downstream paths.
void gcta::compact_indi_data() {
    const int n_old = _indi_num;
    const int n_new = static_cast<int>(_keep.size());
    if (n_new < 1) return;

    bool identity_keep = (n_new == n_old);
    if (identity_keep) {
        for (int i = 0; i < n_new; i++) {
            if (_keep[i] != i) {
                identity_keep = false;
                break;
            }
        }
    }
    if (identity_keep) return;

    const std::vector<int> keep_old(_keep);

    auto compact_vec_string = [&](std::vector<std::string> &v) {
        std::vector<std::string> out;
        out.resize(n_new);
        for (int i = 0; i < n_new; i++) out[i] = std::move(v[keep_old[i]]);
        v = std::move(out);
    };
    auto compact_vec_int = [&](std::vector<int> &v) {
        std::vector<int> out(n_new);
        for (int i = 0; i < n_new; i++) out[i] = v[keep_old[i]];
        v = std::move(out);
    };
    auto compact_vec_double = [&](std::vector<double> &v) {
        std::vector<double> out(n_new);
        for (int i = 0; i < n_new; i++) out[i] = v[keep_old[i]];
        v = std::move(out);
    };

    compact_vec_string(_fid);
    compact_vec_string(_pid);
    compact_vec_string(_fa_id);
    compact_vec_string(_mo_id);
    compact_vec_int(_sex);
    compact_vec_double(_pheno);

    if (_varcmp_Py.size() > 0 && _varcmp_Py.rows() == n_old) {
        eigenMatrix varcmp_buf(n_new, _varcmp_Py.cols());
        for (int i = 0; i < n_new; i++) varcmp_buf.row(i) = _varcmp_Py.row(keep_old[i]);
        _varcmp_Py.swap(varcmp_buf);
    }

    if (_dosage_flag && _geno_dose.size() > 0 && _geno_dose.rows() == n_old) {
        Eigen::MatrixXf dose_buf(n_new, _geno_dose.cols());
        for (int i = 0; i < n_new; i++) dose_buf.row(i) = _geno_dose.row(keep_old[i]);
        _geno_dose = std::move(dose_buf);
    }

    if (!_dosage_flag && !_snp_1.empty()) {
        for (size_t j = 0; j < _snp_1.size(); j++) {
            if (static_cast<int>(_snp_1[j].size()) != n_old || static_cast<int>(_snp_2[j].size()) != n_old) continue;
            std::vector<bool> s1_buf(n_new), s2_buf(n_new);
            for (int i = 0; i < n_new; i++) {
                s1_buf[i] = _snp_1[j][keep_old[i]];
                s2_buf[i] = _snp_2[j][keep_old[i]];
            }
            _snp_1[j] = std::move(s1_buf);
            _snp_2[j] = std::move(s2_buf);
        }
    }

    _indi_num = n_new;
    _mu.clear();   // force recomputation on the compacted sample set
    _keep.resize(n_new);
    _id_map.clear();
    for (int i = 0; i < n_new; i++) {
        _keep[i] = i;
        _id_map.insert({_fid[i] + ":" + _pid[i], i});
    }
}

// Compact _geno_dose column (SNP) dimension so that _include[j]==j after this call.
// All loaders store _geno_dose(dense_indi, dense_snp_within_include); after a
// post-load MAF/Rsq filter _include becomes sparse.  This repacks columns to
// the current m SNPs, then calls update_bim to reset _include to identity.
void gcta::compact_dosage_data() {
    if (!_dosage_flag) return;
    const int n = static_cast<int>(_keep.size());
    const int m = static_cast<int>(_include.size());
    if (m == 0 || n == 0 || _geno_dose.size() == 0) return;

    // Early exit: already compact (cols == m and _include is identity).
    if (_geno_dose.cols() == m) {
        bool identity = true;
        for (int j = 0; j < m; j++) {
            if (_include[j] != j) { identity = false; break; }
        }
        if (identity) return;
    }

    // Repack columns: gather the _include[j]-th column into position j.
    // _keep[i]==i is an invariant of the dosage loading paths.
    Eigen::MatrixXf new_dose(n, m);
    for (int j = 0; j < m; j++) new_dose.col(j) = _geno_dose.col(_include[j]);
    _geno_dose = std::move(new_dose);

    // Compact _mu if already computed.
    if (!_mu.empty()) {
        std::vector<double> mu(m);
        for (int j = 0; j < m; j++) mu[j] = _mu[_include[j]];
        _mu = std::move(mu);
    }

    // Compact _additive_mu the same way — it must stay aligned with _mu/_include.
    if (!_additive_mu.empty()) {
        std::vector<double> additive_mu(m);
        for (int j = 0; j < m; j++) additive_mu[j] = _additive_mu[_include[j]];
        _additive_mu = std::move(additive_mu);
    }

    // Compact metadata and reset _include[j]=j via update_bim.
    std::vector<int> rsnp(_snp_num, 0);
    for (int idx : _include) rsnp[idx] = 1;
    update_bim(rsnp);
}

// Compact _snp_1/_snp_2 and _mu so that their outer index equals the filtered
// SNP index directly (i.e. _include[j] == j after this call).  This eliminates
// the scatter-gather indirection in every subsequent hot loop.  Safe to call
// multiple times — returns immediately when already compact.
void gcta::compact_snp_data() {
    if (_dosage_flag) { compact_dosage_data(); return; }
    const int m = static_cast<int>(_include.size());
    if (m == 0) return;

    // Early exit: already compact when _snp_1 has exactly m rows and _include is identity.
    if (static_cast<int>(_snp_1.size()) == m) {
        bool identity = true;
        for (int j = 0; j < m; j++) {
            if (_include[j] != j) { identity = false; break; }
        }
        if (identity) return;
    }

    // Reorder _snp_1/_snp_2 rows to match the current _include order.
    // _keep is always identity after update_fam, so no inner reindexing needed.
    std::vector<std::vector<bool>> s1(m), s2(m);
    for (int j = 0; j < m; j++) {
        s1[j] = std::move(_snp_1[_include[j]]);
        s2[j] = std::move(_snp_2[_include[j]]);
    }
    _snp_1 = std::move(s1);
    _snp_2 = std::move(s2);

    // Compact _mu if already computed.
    if (!_mu.empty()) {
        std::vector<double> mu(m);
        for (int j = 0; j < m; j++) mu[j] = _mu[_include[j]];
        _mu = std::move(mu);
    }

    // Compact _additive_mu the same way — it must stay aligned with _mu/_include.
    if (!_additive_mu.empty()) {
        std::vector<double> additive_mu(m);
        for (int j = 0; j < m; j++) additive_mu[j] = _additive_mu[_include[j]];
        _additive_mu = std::move(additive_mu);
    }

    // Compact all metadata arrays and reset _include[j]=j via update_bim.
    std::vector<int> rsnp(_snp_num, 0);
    for (int idx : _include) rsnp[idx] = 1;
    update_bim(rsnp);
}

// Read the multiple bfiles
std::vector<std::string>  gcta::read_bfile_list(std::string bfile_list)
{
    std::ifstream bin_list(bfile_list.c_str());
    if (!bin_list)
        LOGGER.e(0, "cannot open the file [" + bfile_list + "] to read.");
    
    std::string strbuf = "";
    std::vector<std::string> multi_bfiles;
    
    while(std::getline(bin_list, strbuf))
    {
        if(strbuf != "")
            multi_bfiles.push_back(strbuf);
    }
    bin_list.close();

    LOGGER.i(0, "There are " + std::to_string(multi_bfiles.size()) + " PLINK genotype files specified in [" + bfile_list + "].");
    return(multi_bfiles);
}

void read_single_famfile(std::string famfile, std::vector<std::string> &fid, std::vector<std::string> &pid, std::vector<std::string> &fa_id, std::vector<std::string> &mo_id, std::vector<int> &sex, std::vector<double> &pheno, bool msg_flag) {
    std::ifstream Fam(famfile.c_str());
    if(!Fam) LOGGER.e(0, "cannot open the file [" + famfile + "] to read.");
    if(msg_flag) LOGGER.i(0, "Reading PLINK FAM file from [" + famfile + "].");

    int i = 0;
    std::string str_buf;
    fid.clear(); pid.clear(); fa_id.clear(); mo_id.clear(); sex.clear(); pheno.clear();
    while (Fam) {
        Fam >> str_buf;
        if (Fam.eof()) break;
        fid.push_back(str_buf);
        Fam >> str_buf;
        pid.push_back(str_buf);
        Fam >> str_buf;
        fa_id.push_back(str_buf);
        Fam >> str_buf;
        mo_id.push_back(str_buf);
        Fam >> str_buf;
        sex.push_back(atoi(str_buf.c_str()));
        Fam >> str_buf;
        pheno.push_back(atoi(str_buf.c_str()));
    }
    Fam.clear();
    Fam.close();
    
    if(msg_flag) {
        int indi_num = fid.size();
        LOGGER.i(0, std::to_string(indi_num) + " individuals to be included from [" + famfile + "].");
    }
}

void gcta::read_multi_famfiles(std::vector<std::string> multi_bfiles)
{
    LOGGER.i(0, "\nReading the PLINK FAM files ....");

    int i=0, nindi_buf = 0, nbfiles = multi_bfiles.size();
    std::string famfile = "";
    std::vector<std::string> tfid, tpid, tfa_id, tmo_id;
    std::vector<int> tsex;
    std::vector<double> tpheno;

    _fid.clear(); _pid.clear(); _fa_id.clear(); _mo_id.clear(); _sex.clear(); _pheno.clear();
    for( i=0; i<nbfiles; i++ ) {
        famfile = multi_bfiles[i]+".fam";
        // Read the fam file
        read_single_famfile(famfile, tfid, tpid, tfa_id, tmo_id, tsex, tpheno, false); 
        // Individual ID
        update_keep(tfid, tpid, tfa_id, tmo_id, tsex, tpheno, famfile);
    }

    // Sample size
    _indi_num = _fid.size();

    LOGGER.i(0, std::to_string(_indi_num) + " individuals have been included from the PLINK FAM files.");
}

void gcta::update_keep(std::vector<std::string> fid_buf, std::vector<std::string> pid_buf, std::vector<std::string> fa_id_buf, std::vector<std::string> mo_id_buf, std::vector<int> sex_buf, std::vector<double> pheno_buf, std::string famfile) {
    int i = 0, indx = 0, nindi_buf = fid_buf.size();
    std::string indi_str = "", strbuf = "";
    std::set<int> dup_ids;

    // Initial sample size
    if(_fid.size() == 0) {
        // Empty std::vector
        // Initialize variables
        _fid = fid_buf;
        _pid = pid_buf;
        _fa_id = fa_id_buf;
        _mo_id = mo_id_buf;
        _sex = sex_buf;
        _pheno = pheno_buf;
        // Initialize the id std::map per chr
        int size=0;
        _keep.clear(); _id_map.clear();
        for(i = 0; i < nindi_buf; i++) {
            _keep.push_back(i);
            // id std::map
            _id_map.insert(std::pair<std::string,int>(_fid[i] + ":" + _pid[i], i));
            if (size == _id_map.size()) LOGGER.e(0, "duplicated individual IDs found: " + _fid[i] + " " + _pid[i] + ".");
            size = _id_map.size();
        }
    } else {
        // Add new individuals
        for(i=0; i<nindi_buf; i++) {
            // Search conflicted information of individuals 
            indi_str = fid_buf[i] + ":" + pid_buf[i];
            std::map<std::string,int>::iterator iter = _id_map.find(indi_str);

            if(iter!=_id_map.end()) {
                // already existed
                indx = iter->second;
                if(fa_id_buf[i] != _fa_id[indx]) LOGGER.e(0, "inconsistent paternal IDs found, " + fid_buf[i] + " " + pid_buf[i] + ", from [" + famfile + "].");
                if(mo_id_buf[i] != _mo_id[indx]) LOGGER.e(0, "inconsistent maternal IDs found, " + fid_buf[i] + " " + pid_buf[i] + ", from [" + famfile + "].");
                if(sex_buf[i] != _sex[indx]) LOGGER.e(0, "inconsistent gender found, " + fid_buf[i] + " " + pid_buf[i] + ", from [" + famfile + "].");
                if(pheno_buf[i] != _pheno[indx]) LOGGER.e(0, "inconsistent phenotype found, " + fid_buf[i] + " " + pid_buf[i] + ", from [" + famfile + "].");
                if(i!=indx) LOGGER.e(0, "inconsistent order of individuals found from [" + famfile + "]. Please make sure that the order of individuals is the same across the fam files.");
            } else {
                // not existed
                LOGGER.e(0, "unexpected individual ID found, " + fid_buf[i] + " " + pid_buf[i] + ", from [" + famfile + "].");
            }   
        }
    }
}

// Read .bim files
std::string duplicated_snp_name(std::string tsnp_name, int tchr, int tbp, std::string ta1, std::string ta2) {
    std::string strbuf = tsnp_name;
    std::stringstream ss;
    ss << "chr" << tchr+1 << ":" << tbp << ":" << ta1 << ta2;
    tsnp_name = ss.str();
    LOGGER.w(0, "Duplicated SNP ID " + strbuf + " has been changed to " + tsnp_name + ".");
    return(tsnp_name);
}

void gcta::update_include(std::vector<int> chr_buf, std::vector<std::string> snpid_buf, std::vector<double> gd_buf, std::vector<int> bp_buf, std::vector<std::string> a1_buf, std::vector<std::string> a2_buf, int file_indx) {
    int i = 0, nsnp_buf = chr_buf.size();

    for(i=0; i<nsnp_buf; i++) {
        // Duplicated SNPs
        if(_snp_name_per_chr.find(snpid_buf[i]) != _snp_name_per_chr.end()) {
            snpid_buf[i] = duplicated_snp_name(snpid_buf[i], chr_buf[i], bp_buf[i], a1_buf[i], a2_buf[i]);
        }
        // SNP name std::map per chr
        _snp_name_per_chr.insert(std::pair<std::string,std::string>(snpid_buf[i], std::to_string(file_indx)+":"+std::to_string(i)));
    }

    // Add the new SNP
    _chr.insert(_chr.end(), chr_buf.begin(), chr_buf.end());
    _snp_name.insert(_snp_name.end(), snpid_buf.begin(), snpid_buf.end());
    _genet_dst.insert(_genet_dst.end(), gd_buf.begin(), gd_buf.end());
    _bp.insert(_bp.end(), bp_buf.begin(), bp_buf.end());
    _allele_alt.insert(_allele_alt.end(), a1_buf.begin(), a1_buf.end());
    _allele_ref.insert(_allele_ref.end(), a2_buf.begin(), a2_buf.end());
}

void read_single_bimfile(std::string bimfile, std::vector<int> &chr, std::vector<std::string> &snp_name, std::vector<double> &genet_dst, std::vector<int> &bp, std::vector<std::string> &allele1, std::vector<std::string> &allele2, bool msg_flag) {
    // Read bim file: recombination rate is defined between SNP i and SNP i-1
    int ibuf = 0;
    std::string cbuf = "0";
    double dbuf = 0.0;
    std::string str_buf;
    std::ifstream Bim(bimfile.c_str());
    if(!Bim) LOGGER.e(0, "cannot open the file [" + bimfile + "] to read.");
    if(msg_flag) LOGGER.i(0, "Reading PLINK BIM file from [" + bimfile + "].");

    chr.clear(); snp_name.clear(); genet_dst.clear(); bp.clear(); allele1.clear(); allele2.clear();
    while (Bim) {
        Bim >> ibuf;
        if (Bim.eof()) break;
        chr.push_back(ibuf);
        Bim >> str_buf;
        snp_name.push_back(str_buf);
        Bim >> dbuf;
        genet_dst.push_back(dbuf);
        Bim >> ibuf;
        bp.push_back(ibuf);
        Bim >> cbuf;
        StrFunc::to_upper(cbuf);
        allele1.push_back(cbuf);
        Bim >> cbuf;
        StrFunc::to_upper(cbuf);
        allele2.push_back(cbuf);
    }
    Bim.close();

    if(msg_flag) {
        int snp_num = chr.size();
        LOGGER.i(0, std::to_string(snp_num) + " SNPs to be included from [" + bimfile + "].");
    }
}

void gcta::read_multi_bimfiles(std::vector<std::string> multi_bfiles)
{
    LOGGER.i(0, "Reading the PLINK BIM files ...");

    int i=0, nbfiles = multi_bfiles.size();
    std::string bimfile = "";
    std::vector<std::string> tsnp_name, tallele1, tallele2;
    std::vector<int> tchr, tbp;
    std::vector<double> tgenet_dst;

    _snp_name_per_chr.clear();
    for( i=0; i<nbfiles; i++ ) {
        bimfile = multi_bfiles[i]+".bim";
        read_single_bimfile(bimfile, tchr, tsnp_name, tgenet_dst, tbp, tallele1, tallele2, false);
        update_include(tchr, tsnp_name, tgenet_dst, tbp, tallele1, tallele2, i);
    }

    // Initialize the variables
    _snp_num = _snp_name.size();
    _include.clear(); _include.resize(_snp_num);
    for(i=0; i<_snp_num; i++) {
        _snp_name_map.insert(std::pair<std::string,int>(_snp_name[i], i));
        _include[i] = i;
    }
    LOGGER.i(0, std::to_string(_snp_num) + " SNPs to be included from PLINK BIM files.");
}

// Read multiple .bed files
void update_id_chr_map(std::map<std::string, std::string> &chr_map, std::map<std::string, int> id_map) {
    int i = 0;
    std::map<std::string, std::string> chr_map_buf(chr_map);
    for(const auto& [k, v] : id_map) chr_map_buf.erase(k);
    for(const auto& [k, v] : chr_map_buf) chr_map.erase(k);
}

void retrieve_snp(std::map<std::string,std::string> snp_chr_map, std::map<std::string,int> snp_id_map, std::vector<std::vector<std::pair<int,int>>> &rsnp, int nbfiles) {
    int i = 0, j=0, snp_indx = 0;
    std::string snp_indx_str = "";
    std::vector<std::string> vs_buf;
    std::map<std::string,std::string>::iterator iter1;
    std::map<std::string,int>::iterator iter2;

    rsnp.clear(); rsnp.resize(nbfiles);

    for(iter1=snp_chr_map.begin(), iter2=snp_id_map.begin(); iter1 != snp_chr_map.end(); iter1++, iter2++) {
        vs_buf.clear();
        snp_indx_str = iter1->second;
        StrFunc::split_string(snp_indx_str, vs_buf, ":");
        snp_indx = iter2->second;
        rsnp[atoi(vs_buf[0].c_str())].push_back(std::make_pair(atoi(vs_buf[1].c_str()), snp_indx));
    }

    for(i=0; i<nbfiles; i++) std::stable_sort(rsnp[i].begin(), rsnp[i].end());
}

void read_single_bedfile(std::string bedfile, std::vector<std::pair<int,int>> rsnp, std::vector<int> rindi, std::vector<std::vector<bool>> &snp1, std::vector<std::vector<bool>> &snp2, bool msg_flag)
{
    int i = 0, j = 0, k = 0, t1 = 0, t2= 0, nsnp_chr = rsnp.size(), nindi_chr = rindi.size();

    // Read bed file
    std::bitset<8> b;
    std::fstream BIT(bedfile.c_str(), std::ios::in | std::ios::binary);
    if(!BIT) LOGGER.e(0, "cannot open the file [" + bedfile + "] to read.");
    if(msg_flag) LOGGER.i(0, "Reading single PLINK BED file from [" + bedfile + "] in SNP-major format ...");

    int bytes_per_snp = (nindi_chr + 3) / 4;
    std::vector<char> snp_buf(bytes_per_snp);

    // skip the first three bytes
    char header[3];
    BIT.read(header, 3);
    int snp_indx = 0, indi_indx = 0;
    // Read genotype in SNP-major mode, 00: homozygote AA; 11: homozygote BB; 01: hetezygote; 10: missing
    for(j = 0, t1 = 0, snp_indx = 0; t1 < nsnp_chr; j++) { 
        if(j!=rsnp[t1].first) {
            BIT.seekg(bytes_per_snp, std::ios::cur);
            continue;
        }
        snp_indx = rsnp[t1].second;
        BIT.read(snp_buf.data(), bytes_per_snp);
        if (!BIT) LOGGER.e(0, "problem with the BED file ... has the FAM/BIM file been changed?");
        int byte_idx = 0;
        for (i = 0, indi_indx = 0; i < nindi_chr;) {
            b = snp_buf[byte_idx++];
            k = 0;
            while (k < 7 && i < nindi_chr) { // change code: 11 for AA; 00 for BB;
                if (!rindi[i]) k += 2;
                else {
                    snp2[snp_indx][indi_indx] = (!b[k++]);
                    snp1[snp_indx][indi_indx] = (!b[k++]);
                    indi_indx++;
                }
                i++;
            }
        }
        t1++;
    }
    BIT.clear();
    BIT.close();

    if(msg_flag) LOGGER.i(0, "Genotype data for " + std::to_string(nindi_chr) + " individuals and " + std::to_string(nsnp_chr) + " SNPs to be included from [" + bedfile + "].");
}

void gcta::read_multi_bedfiles(std::vector<std::string> multi_bfiles) {
    int i=0, nbfiles = multi_bfiles.size();
    std::string bedfile = "";
    std::vector<std::vector<std::pair<int, int>>> rsnp;
    std::vector<int> rindi_flag, rsnp_flag;

    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");
    if (_keep.size() == 0) LOGGER.e(0, "no individual is retained for analysis.");

    LOGGER.i(0, "Reading PLINK BED files ...");
    // Initialize the matrix
    _snp_1.clear(); _snp_2.clear();
    _snp_1.resize(_include.size()); _snp_2.resize(_include.size());
    for (i = 0; i < _include.size(); i++) {
        _snp_1[i].reserve(_keep.size());
        _snp_2[i].reserve(_keep.size());
    }

    // Update the std::map to retrieve individuals and SNPs
    update_id_chr_map(_snp_name_per_chr, _snp_name_map);

    // Reset variables
    get_rindi(rindi_flag);
    get_rsnp(rsnp_flag);
    update_fam(rindi_flag); 
    update_bim(rsnp_flag);
    
    retrieve_snp(_snp_name_per_chr, _snp_name_map, rsnp, nbfiles);

    // Read the coded genotypes
    for( i=0; i<nbfiles; i++) {        
        if(rsnp[i].size()==0) { 
            LOGGER.i(0, "Skip reading " + multi_bfiles[i] + ".bed, no SNPs retained on this chromosome.");
            continue;
        }
        bedfile = multi_bfiles[i] + ".bed";
        read_single_bedfile(bedfile, rsnp[i], rindi_flag, _snp_1, _snp_2, false);
    }

    LOGGER.i(0, "Genotype data for " + std::to_string(_keep.size()) + " individuals and " + std::to_string(_include.size()) + " SNPs have been included.");
}

void gcta::read_imp_info_mach_gz(std::string zinfofile)
{
    _dosage_flag = true;

    int i = 0;
    std::ifstream raw_file(zinfofile, std::ios::binary);
    if (!raw_file.is_open()) LOGGER.e(0, "cannot open the file [" + zinfofile + "] to read.");
    boost::iostreams::filtering_istream zinf;
    zinf.push(boost::iostreams::gzip_decompressor(15, 1 << 18)); // 256KB internal buffer
    zinf.push(raw_file);

    std::string buf, str_buf, errmsg = "Reading dosage data failed. Please check the format of the std::map file.";
    std::string c_buf;
    double f_buf = 0.0;
    LOGGER << "Reading map file of the imputed dosage data from [" + zinfofile + "]." << std::endl;
    std::getline(zinf, buf); // skip the header
    std::vector<std::string> vs_buf;
    int col_num = StrFunc::split_string(buf, vs_buf, " \t\n");
    if (col_num < 7) LOGGER.e(0, errmsg);
    if (vs_buf[6] != "Rsq") LOGGER.e(0, errmsg);
    _snp_name.clear();
    _allele_alt.clear();
    _allele_ref.clear();
    _impRsq.clear();
    _mu.clear();
    while (std::getline(zinf, buf)) {
        std::stringstream ss(buf);
        std::string nerr = errmsg + "\nError occurs in line: " + ss.str();
        if (!(ss >> str_buf)) break;
        _snp_name.push_back(str_buf);
        if (!(ss >> c_buf)) LOGGER.e(0, nerr);
        _allele_alt.push_back(c_buf);
        if (!(ss >> c_buf)) LOGGER.e(0, nerr);
        _allele_ref.push_back(c_buf);
        double freq1 = 0.0;
        if (!(ss >> freq1)) LOGGER.e(0, nerr);   // Freq1 → _mu (as 2*freq since _mu is 0-2 scale)
        for (i = 0; i < 3; i++) if (!(ss >> f_buf)) LOGGER.e(0, nerr);  // MAF, Quality, Rsq
        _mu.push_back(2.0 * freq1);
        _impRsq.push_back(f_buf);
    }
    zinf.reset();
    raw_file.close();
    _snp_num = _snp_name.size();
    _chr.resize(_snp_num);
    _bp.resize(_snp_num);
    _genet_dst.resize(_snp_num);

    // Initialize _include
    init_include();
    LOGGER << _snp_num << " SNPs to be included from [" + zinfofile + "]." << std::endl;
}

void gcta::read_imp_info_mach(std::string infofile)
{
    _dosage_flag = true;    
    if(infofile.substr(infofile.length()-3,3)==".gz") LOGGER.e(0, "the --dosage-mach option doesn't support .gz file any more. Please check the --dosage-mach-gz option.");

    int i = 0;
    std::ifstream inf(infofile.c_str());
    if (!inf.is_open()) LOGGER.e(0, "cannot open the file [" + infofile + "] to read.");

    std::string buf, str_buf, errmsg = "Reading dosage data failed. Please check the format of the std::map file.";
    std::string c_buf;
    double f_buf = 0.0;
    LOGGER << "Reading map file of the imputed dosage data from [" + infofile + "]." << std::endl;
    std::getline(inf, buf); // skip the header
    std::vector<std::string> vs_buf;
    int col_num = StrFunc::split_string(buf, vs_buf, " \t\n");
    if (col_num < 7) LOGGER.e(0, errmsg);
    if (vs_buf[6] != "Rsq" && vs_buf[6] != "Rsq_hat") LOGGER.e(0, errmsg);
    _snp_name.clear();
    _allele_alt.clear();
    _allele_ref.clear();
    _impRsq.clear();
    while (std::getline(inf, buf)) {
        std::stringstream ss(buf);
        std::string nerr = errmsg + "\nError occurs in line: " + ss.str();
        if (!(ss >> str_buf)) break;
        _snp_name.push_back(str_buf);
        if (!(ss >> c_buf)) LOGGER.e(0, nerr);
        _allele_alt.push_back(c_buf);
        if (!(ss >> c_buf)) LOGGER.e(0, nerr);
        _allele_ref.push_back(c_buf);
        for (i = 0; i < 3; i++) if (!(ss >> f_buf)) LOGGER.e(0, nerr);
        _impRsq.push_back(f_buf);
    }
    inf.close();
    _snp_num = _snp_name.size();
    _chr.resize(_snp_num);
    _bp.resize(_snp_num);
    _genet_dst.resize(_snp_num);

    // Initialize _include
    init_include();
    LOGGER << _snp_num << " SNPs to be included from [" + infofile + "]." << std::endl;
}

void gcta::read_imp_dose_mach_gz(std::string zdosefile, std::string kp_indi_file, std::string rm_indi_file, std::string blup_indi_file) {
    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");

    int i = 0, j = 0, k = 0, line = 0;
    std::vector<int> rsnp;
    get_rsnp(rsnp);

    // Helper: pop the next whitespace-delimited token from a string_view, advancing it.
    // Returns empty string_view when exhausted.
    auto next_token = [](std::string_view& sv) -> std::string_view {
        size_t start = sv.find_first_not_of(' ');
        if (start == std::string_view::npos) { sv = {}; return {}; }
        sv.remove_prefix(start);
        size_t end = sv.find(' ');
        std::string_view tok = sv.substr(0, end == std::string_view::npos ? sv.size() : end);
        sv.remove_prefix(end == std::string_view::npos ? sv.size() : end);
        return tok;
    };

    std::ifstream raw_file(zdosefile, std::ios::binary);
    if (!raw_file.is_open()) LOGGER.e(0, "cannot open the file [" + zdosefile + "] to read.");
    boost::iostreams::filtering_istream zinf;
    zinf.push(boost::iostreams::gzip_decompressor(15, 1 << 18)); // 256KB internal buffer
    zinf.push(raw_file);

    std::vector<std::string> indi_ls;
    std::map<std::string, int> kp_id_map, blup_id_map, rm_id_map;
    bool kp_indi_flag = !kp_indi_file.empty(), blup_indi_flag = !blup_indi_file.empty(), rm_indi_flag = !rm_indi_file.empty();
    if (kp_indi_flag) read_indi_list(kp_indi_file, indi_ls);
    for (i = 0; i < indi_ls.size(); i++) kp_id_map.insert(std::pair<std::string, int>(indi_ls[i], i));
    if (blup_indi_flag) read_indi_list(blup_indi_file, indi_ls);
    for (i = 0; i < indi_ls.size(); i++) blup_id_map.insert(std::pair<std::string, int>(indi_ls[i], i));
    if (rm_indi_flag) read_indi_list(rm_indi_file, indi_ls);
    for (i = 0; i < indi_ls.size(); i++) rm_id_map.insert(std::pair<std::string, int>(indi_ls[i], i));

    bool missing = false;
    std::string buf, id_buf, err_msg = "reading dosage data failed. Are the map file and the dosage file matched?";
    std::vector<std::string> kept_id, vs_buf;
    LOGGER << "Reading dosage data from [" + zdosefile + "] in individual-major format (Note: may use huge RAM)." << std::endl;
    _fid.clear();
    _pid.clear();
    _geno_dose.resize(0, 0);

    std::vector<int> kp_it;
    while (std::getline(zinf, buf)) {
        bool kp_flag = true;
        std::string_view sv(buf);
        std::string_view id_tok = next_token(sv);
        if (id_tok.empty()) break;

        int ibuf = StrFunc::split_string(std::string(id_tok), vs_buf, ">");
        if (ibuf > 1) {
            if (vs_buf[0].empty()) LOGGER.e(0, "the family ID of the individual [" + std::string(id_tok) + "] is missing.");
            else vs_buf[0].erase(vs_buf[0].end() - 1);
        } else if (ibuf == 1) vs_buf.push_back(vs_buf[0]);
        else break;
        id_buf = vs_buf[0] + ":" + vs_buf[1];
        if (kp_indi_flag && kp_id_map.find(id_buf) == kp_id_map.end()) kp_flag = false;
        if (kp_flag && blup_indi_flag && blup_id_map.find(id_buf) == blup_id_map.end()) kp_flag = false;
        if (kp_flag && rm_indi_flag && rm_id_map.find(id_buf) != rm_id_map.end()) kp_flag = false;
        if (kp_flag) {
            kp_it.push_back(1);
            _fid.push_back(vs_buf[0]);
            _pid.push_back(vs_buf[1]);
            kept_id.push_back(id_buf);
        } else kp_it.push_back(0);
    }
    zinf.reset();
    raw_file.close();
    LOGGER << "(Imputed dosage data for " << kp_it.size() << " individuals detected)." << std::endl;
    _indi_num = _fid.size();

    std::ifstream raw_file2(zdosefile, std::ios::binary);
    if (!raw_file2.is_open()) LOGGER.e(0, "cannot open the file [" + zdosefile + "] to read.");
    boost::iostreams::filtering_istream zinf2;
    zinf2.push(boost::iostreams::gzip_decompressor(15, 1 << 18));
    zinf2.push(raw_file2);

    // Single-row staging buffer: matches the file's row-wise layout for a
    // sequential, cache-friendly parse target, without materializing a
    // second full-size n*m copy alongside _geno_dose (previously a
    // RowMajor n*m staging matrix + a bulk `_geno_dose = tmp` transpose,
    // which doubled peak memory for the duration of the assignment).
    const int m_cols = static_cast<int>(_include.size());
    _geno_dose.resize(_indi_num, m_cols);
    std::vector<float> row_buf(m_cols);

    for (line = 0, k = 0; line < kp_it.size(); line++) {
        std::getline(zinf2, buf);
        if (kp_it[line] == 0) continue;

        std::string_view sv(buf);
        if (next_token(sv).empty()) break; // family/individual ID token
        if (next_token(sv).empty()) break; // second header token

        for (i = 0, j = 0; i < _snp_num; i++) {
            std::string_view tok = next_token(sv);
            float f_buf;

            if (tok == "X" || tok == "NA") {
                if (!missing) {
                    LOGGER.w(0, "missing values detected in the dosage data.");
                    missing = true;
                }
                f_buf = DOSAGE_NA;
            } else {
                auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), f_buf);
                if (ec != std::errc()) {
                    if (!missing) {
                        LOGGER.w(0, "missing values detected in the dosage data.");
                        missing = true;
                    }
                    f_buf = DOSAGE_NA;
                }
            }

            if (rsnp[i]) {
                row_buf[j] = f_buf;
                j++;
            }
        }
        // One strided row-write into the column-major matrix per individual,
        // instead of a full-matrix RowMajor->ColMajor copy at the end.
        _geno_dose.row(k) = Eigen::Map<const Eigen::RowVectorXf>(row_buf.data(), m_cols);
        k++;
    }
    zinf2.reset();
    raw_file2.close();

    if (_geno_dose.rows() != _indi_num || _geno_dose.cols() != m_cols)
    LOGGER.e(0, "internal error: _geno_dose dimension mismatch after dosage load.");

    LOGGER << "Imputed dosage data for " << kept_id.size() << " individuals are included from [" << zdosefile << "]." << std::endl;
    _fa_id.resize(_indi_num);
    _mo_id.resize(_indi_num);
    _sex.resize(_indi_num);
    _pheno.resize(_indi_num);
    for (i = 0; i < _indi_num; i++) {
        _fa_id[i] = _mo_id[i] = "0";
        _sex[i] = -9;
        _pheno[i] = -9;
    }

    // initialize keep
    init_keep();
    update_id_map_kp(kept_id, _id_map, _keep);
    if (_keep.size() == 0) LOGGER.e(0, "no individual is retained for analysis.");

    if (blup_indi_flag) read_indi_blup(blup_indi_file);

    // update data
    update_bim(rsnp);
}

void gcta::read_imp_dose_mach(std::string dosefile, std::string kp_indi_file, std::string rm_indi_file, std::string blup_indi_file) {
    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");
    if(dosefile.substr(dosefile.length()-3,3)==".gz") LOGGER.e(0, "the --dosage-mach option doesn't support .gz file any more. Please check the --dosage-mach-gz option.");

    int i = 0, j = 0, k = 0, line = 0;
    std::vector<int> rsnp;
    get_rsnp(rsnp);

    // Helper: pop the next whitespace-delimited token from a string_view, advancing it.
    // Returns empty string_view when exhausted. Same tokenizer used by the gz reader,
    // avoids stringstream's per-token locale/allocation overhead.
    auto next_token = [](std::string_view& sv) -> std::string_view {
        size_t start = sv.find_first_not_of(' ');
        if (start == std::string_view::npos) { sv = {}; return {}; }
        sv.remove_prefix(start);
        size_t end = sv.find(' ');
        std::string_view tok = sv.substr(0, end == std::string_view::npos ? sv.size() : end);
        sv.remove_prefix(end == std::string_view::npos ? sv.size() : end);
        return tok;
    };

    std::ifstream idose(dosefile.c_str());
    if (!idose) LOGGER.e(0, "cannot open the file [" + dosefile + "] to read.");

    std::vector<std::string> indi_ls;
    std::map<std::string, int> kp_id_map, blup_id_map, rm_id_map;
    bool kp_indi_flag = !kp_indi_file.empty(), blup_indi_flag = !blup_indi_file.empty(), rm_indi_flag = !rm_indi_file.empty();
    if (kp_indi_flag) read_indi_list(kp_indi_file, indi_ls);
    for (i = 0; i < indi_ls.size(); i++) kp_id_map.insert(std::pair<std::string, int>(indi_ls[i], i));
    if (blup_indi_flag) read_indi_list(blup_indi_file, indi_ls);
    for (i = 0; i < indi_ls.size(); i++) blup_id_map.insert(std::pair<std::string, int>(indi_ls[i], i));
    if (rm_indi_flag) read_indi_list(rm_indi_file, indi_ls);
    for (i = 0; i < indi_ls.size(); i++) rm_id_map.insert(std::pair<std::string, int>(indi_ls[i], i));

    bool missing = false;
    std::string buf, id_buf, err_msg = "reading dosage data failed. Are the map file and the dosage file matched?";
    std::vector<std::string> kept_id, vs_buf;
    LOGGER << "Reading dosage data from [" + dosefile + "] in individual-major format (Note: may use huge RAM)." << std::endl;
    _fid.clear();
    _pid.clear();
    _geno_dose.resize(0, 0);

    std::vector<int> kp_it;
    while (std::getline(idose, buf)) {
        bool kp_flag = true;
        std::string_view sv(buf);
        std::string_view id_tok = next_token(sv);
        if (id_tok.empty()) break;

        int ibuf = StrFunc::split_string(std::string(id_tok), vs_buf, ">");
        if (ibuf > 1) {
            if (vs_buf[0].empty()) LOGGER.e(0, "the family ID of the individual [" + std::string(id_tok) + "] is missing.");
            else vs_buf[0].erase(vs_buf[0].end() - 1);
        } else if (ibuf == 1) vs_buf.push_back(vs_buf[0]);
        else break;
        id_buf = vs_buf[0] + ":" + vs_buf[1];
        if (kp_indi_flag && kp_id_map.find(id_buf) == kp_id_map.end()) kp_flag = false;
        if (kp_flag && blup_indi_flag && blup_id_map.find(id_buf) == blup_id_map.end()) kp_flag = false;
        if (kp_flag && rm_indi_flag && rm_id_map.find(id_buf) != rm_id_map.end()) kp_flag = false;
        if (kp_flag) {
            kp_it.push_back(1);
            _fid.push_back(vs_buf[0]);
            _pid.push_back(vs_buf[1]);
            kept_id.push_back(id_buf);
        } else kp_it.push_back(0);
    }
    idose.close();
    LOGGER << "(Imputed dosage data for " << kp_it.size() << " individuals detected)." << std::endl;
    _indi_num = _fid.size();

    idose.open(dosefile.c_str());
    if (!idose) LOGGER.e(0, "cannot open the file [" + dosefile + "] to read.");

    // Single-row staging buffer: matches the file's row-wise layout for a
    // sequential, cache-friendly parse target, without materializing a
    // second full-size n*m copy alongside _geno_dose (previously a
    // RowMajor n*m staging matrix + a bulk `_geno_dose = tmp` transpose,
    // which doubled peak memory for the duration of the assignment).
    const int m_cols = static_cast<int>(_include.size());
    _geno_dose.resize(_indi_num, m_cols);
    std::vector<float> row_buf(m_cols);

    for (line = 0, k = 0; line < kp_it.size(); line++) {
        std::getline(idose, buf);
        if (kp_it[line] == 0) continue;

        std::string_view sv(buf);
        if (next_token(sv).empty()) break; // family/individual ID token
        if (next_token(sv).empty()) break; // second header token

        for (i = 0, j = 0; i < _snp_num; i++) {
            std::string_view tok = next_token(sv);
            float f_buf;

            if (tok == "X" || tok == "NA") {
                if (!missing) {
                    LOGGER.w(0, "missing values detected in the dosage data.");
                    missing = true;
                }
                f_buf = DOSAGE_NA;
            } else {
                auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), f_buf);
                if (ec != std::errc()) {
                    if (!missing) {
                        LOGGER.w(0, "missing values detected in the dosage data.");
                        missing = true;
                    }
                    f_buf = DOSAGE_NA;
                }
            }

            if (rsnp[i]) {
                row_buf[j] = f_buf;
                j++;
            }
        }
        // One strided row-write into the column-major matrix per individual,
        // instead of a full-matrix RowMajor->ColMajor copy at the end.
        _geno_dose.row(k) = Eigen::Map<const Eigen::RowVectorXf>(row_buf.data(), m_cols);
        k++;
    }
    idose.close();

    if (_geno_dose.rows() != _indi_num || _geno_dose.cols() != m_cols)
    LOGGER.e(0, "internal error: _geno_dose dimension mismatch after dosage load.");

    LOGGER << "Imputed dosage data for " << kept_id.size() << " individuals are included from [" << dosefile << "]." << std::endl;
    _fa_id.resize(_indi_num);
    _mo_id.resize(_indi_num);
    _sex.resize(_indi_num);
    _pheno.resize(_indi_num);
    for (i = 0; i < _indi_num; i++) {
        _fa_id[i] = _mo_id[i] = "0";
        _sex[i] = -9;
        _pheno[i] = -9;
    }

    // initialize keep
    init_keep();
    update_id_map_kp(kept_id, _id_map, _keep);
    if (_keep.size() == 0) LOGGER.e(0, "no individual is retained for analysis.");

    if (blup_indi_flag) read_indi_blup(blup_indi_file);

    // update data
    update_bim(rsnp);
}

void gcta::read_imp_info_beagle(std::string zinfofile) {
    _dosage_flag = true;

    std::string buf;
    std::string str_buf, errmsg = "Reading SNP summary information filed? Please check the format of [" + zinfofile + "].";

    std::string c_buf;
    int i_buf;
    double f_buf = 0.0;
    std::ifstream raw_file(zinfofile, std::ios::binary);
    if (!raw_file.is_open()) LOGGER.e(0, "cannot open the file [" + zinfofile + "] to read.");
    boost::iostreams::filtering_istream zinf;
    zinf.push(boost::iostreams::gzip_decompressor());
    zinf.push(raw_file);
    LOGGER << "Reading summary information of the imputed SNPs (BEAGLE output) ..." << std::endl;
    std::getline(zinf, buf); // skip the header
    while (std::getline(zinf, buf)) {
        std::stringstream ss(buf);
        std::string nerr = errmsg + "\nError line: " + ss.str();
        if (!(ss >> i_buf)) LOGGER.e(0, nerr);
        _chr.push_back(i_buf);
        if (!(ss >> str_buf)) LOGGER.e(0, nerr);
        _snp_name.push_back(str_buf);
        if (!(ss >> i_buf)) LOGGER.e(0, nerr);
        _bp.push_back(i_buf);
        if (!(ss >> c_buf)) LOGGER.e(0, nerr);
        _allele_alt.push_back(c_buf);
        if (!(ss >> c_buf)) LOGGER.e(0, nerr);
        _allele_ref.push_back(c_buf);
        if (!(ss >> str_buf)) LOGGER.e(0, nerr);
        if (!(ss >> str_buf)) LOGGER.e(0, nerr);
        if (!(ss >> str_buf)) LOGGER.e(0, nerr);
        if (!(ss >> f_buf)) LOGGER.e(0, nerr);
        if (!(ss >> f_buf)) LOGGER.e(0, nerr);
        if (!(ss >> f_buf)) LOGGER.e(0, nerr);
        _impRsq.push_back(f_buf);
        if (!(ss >> f_buf)) LOGGER.e(0, nerr);
        if (!(ss >> f_buf)) LOGGER.e(0, nerr);
        if (ss >> f_buf) LOGGER.e(0, nerr);
    }
    zinf.reset();
    raw_file.close();
    _snp_num = _snp_name.size();
    LOGGER << _snp_num << " SNPs to be included from [" + zinfofile + "]." << std::endl;
    _genet_dst.resize(_snp_num);

    // Initialize _include
    init_include();
}

void gcta::read_imp_dose_beagle(std::string zdosefile, std::string kp_indi_file, std::string rm_indi_file, std::string blup_indi_file) {
    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");
    int i = 0, j = 0;
    std::vector<int> rsnp;
    get_rsnp(rsnp);

    std::string buf;
    std::string str_buf;

    std::ifstream raw_file(zdosefile, std::ios::binary);
    if (!raw_file.is_open()) LOGGER.e(0, "cannot open the file [" + zdosefile + "] to read.");
    boost::iostreams::filtering_istream zinf;
    zinf.push(boost::iostreams::gzip_decompressor());
    zinf.push(raw_file);
    LOGGER << "Reading imputed dosage scores (BEAGLE output) ..." << std::endl;
    std::getline(zinf, buf);
    std::stringstream ss(buf);
    for (i = 0; i < 3; i++) ss >> str_buf;
    while (ss >> str_buf) {
        _fid.push_back(str_buf);
    }
    _pid = _fid;
    _indi_num = _fid.size();
    _fa_id.resize(_indi_num);
    _mo_id.resize(_indi_num);
    _sex.resize(_indi_num);
    _pheno.resize(_indi_num);
    LOGGER << _indi_num << " individuals to be included from [" + zdosefile + "]." << std::endl;
    init_keep();
    if (!kp_indi_file.empty()) keep_indi(kp_indi_file);
    if (!blup_indi_file.empty()) read_indi_blup(blup_indi_file);
    if (!rm_indi_file.empty()) remove_indi(rm_indi_file);

    _geno_dose.setZero(static_cast<int>(_keep.size()), static_cast<int>(_include.size()));

    std::vector<int> rindi;
    get_rindi(rindi);

    int line = 0;
    int k = 0;
    double d_buf = 0.0;
    while (std::getline(zinf, buf)) {
        if (!rsnp[line++]) continue;
        std::stringstream ss(buf);
        ss >> str_buf;
        if (str_buf != _snp_name[line - 1]) {
            std::stringstream errmsg;
            errmsg << "the " << line << " th SNP [" + _snp_name[line - 1] + "] in the summary file doesn't match to that in the dosage file." << std::endl;
            LOGGER.e(0, errmsg.str());
        }
        ss >> str_buf >> str_buf;
        for (i = 0, j = 0; i < _indi_num; i++) {
            ss >> d_buf;
            if (rindi[i]) {
                _geno_dose(j, k) = static_cast<float>(d_buf);
                j++;
            }
        }
        k++;
    }
    zinf.reset();
    raw_file.close();
}

void gcta::save_plink() {
    if (_dosage_flag) dose2bed();
    save_famfile();
    save_bimfile();
    save_bedfile();
}

void gcta::save_bedfile() {
    int i = 0, pos = 0, j = 0;
    std::string OutBedFile = _out + ".bed";
    std::fstream OutBed(OutBedFile.c_str(), std::ios::out | std::ios::binary);
    if (!OutBed) LOGGER.e(0, "cannot open the file [" + OutBedFile + "] to write.");
    LOGGER << "Writing genotypes to PLINK BED file [" + OutBedFile + "] ..." << std::endl;
    std::bitset<8> b;
    char ch[1];
    b.reset();
    b.set(2);
    b.set(3);
    b.set(5);
    b.set(6);
    ch[0] = (char) b.to_ulong();
    OutBed.write(ch, 1);
    b.reset();
    b.set(0);
    b.set(1);
    b.set(3);
    b.set(4);
    ch[0] = (char) b.to_ulong();
    OutBed.write(ch, 1);
    b.reset();
    b.set(0);
    ch[0] = (char) b.to_ulong();
    OutBed.write(ch, 1);
    for (i = 0; i < _include.size(); i++) {
        pos = 0;
        b.reset();
        for (j = 0; j < _keep.size(); j++) {
            b[pos++] = (!_snp_2[_include[i]][_keep[j]]);
            b[pos++] = (!_snp_1[_include[i]][_keep[j]]);
            if (pos > 7 || j == _keep.size() - 1) {
                ch[0] = (char) b.to_ulong();
                OutBed.write(ch, 1);
                pos = 0;
                b.reset();
            }
        }
    }
    OutBed.close();
}

void gcta::save_famfile() {
    std::string famfile = _out + ".fam";
    std::ofstream Fam(famfile.c_str());
    if (!Fam) LOGGER.e(0, "cannot open the fam file " + famfile + " to save!");
    LOGGER << "Writing PLINK FAM file to [" + famfile + "] ..." << std::endl;
    int i = 0;
    for (i = 0; i < _keep.size(); i++) {
        Fam << _fid[_keep[i]] << "\t" << _pid[_keep[i]] << "\t" << _fa_id[_keep[i]] << "\t" << _mo_id[_keep[i]] << "\t" << _sex[_keep[i]] << "\t" << _pheno[_keep[i]] << std::endl;
    }
    Fam.close();
    LOGGER << _keep.size() << " individuals to be saved to [" + famfile + "]." << std::endl;
}

void gcta::save_bimfile() {
    int i = 0;
    std::string bimfile = _out + ".bim";
    std::ofstream Bim(bimfile.c_str());
    if (!Bim) LOGGER.e(0, "cannot open the file [" + bimfile + "] to write.");
    LOGGER << "Writing PLINK BIM file to [" + bimfile + "] ..." << std::endl;
    for (i = 0; i < _include.size(); i++) {
        Bim << _chr[_include[i]] << "\t" << _snp_name[_include[i]] << "\t" << _genet_dst[_include[i]] << "\t" << _bp[_include[i]] << "\t" << _allele_alt[_include[i]] << "\t" << _allele_ref[_include[i]] << std::endl;
    }
    Bim.close();
    LOGGER << _include.size() << " SNPs to be saved to [" + bimfile + "]." << std::endl;
}

void gcta::dose2bed() {
    int i = 0, j = 0;
    double d_buf = 0.0;

    compact_dosage_data(); // ensure _include[j]==j and _keep[i]==i
    LOGGER << "Converting dosage data into PLINK binary PED format ... " << std::endl;
    _snp_1.resize(_snp_num);
    _snp_2.resize(_snp_num);
    for (i = 0; i < _snp_num; i++){
        _snp_1[i].resize(_indi_num);
        _snp_2[i].resize(_indi_num);       
    }  
    for (i = 0; i < _include.size(); i++) {  
        for (j = 0; j < _keep.size(); j++) {
           d_buf = _geno_dose(j, i);
            if (d_buf >= DOSAGE_NA) {
                _snp_2[_include[i]][_keep[j]] = false;
                _snp_1[_include[i]][_keep[j]] = true;
            } else if (d_buf >= 1.5) _snp_1[_include[i]][_keep[j]] = _snp_2[_include[i]][_keep[j]] = true;
            else if (d_buf > 0.5) {
                _snp_2[_include[i]][_keep[j]] = true;
                _snp_1[_include[i]][_keep[j]] = false;
            } else if (d_buf <= 0.5) _snp_1[_include[i]][_keep[j]] = _snp_2[_include[i]][_keep[j]] = false;
        }
    }
}

void gcta::update_id_map_kp(const std::vector<std::string> &id_list, std::map<std::string, int> &id_map, std::vector<int> &keep) {
    int i = 0;
    std::map<std::string, int> id_map_buf(id_map);
    for (i = 0; i < id_list.size(); i++) id_map_buf.erase(id_list[i]);
    for (const auto& [k, v] : id_map_buf) id_map.erase(k);

    keep.clear();
    for (const auto& [k, v] : id_map) keep.push_back(v);
    std::stable_sort(keep.begin(), keep.end());
}

void gcta::update_id_map_rm(const std::vector<std::string> &id_list, std::map<std::string, int> &id_map, std::vector<int> &keep)
{
    int i = 0;
    for (i = 0; i < id_list.size(); i++) id_map.erase(id_list[i]);

    keep.clear();
    std::map<std::string, int>::iterator iter;
    for (iter = id_map.begin(); iter != id_map.end(); iter++) keep.push_back(iter->second);
    std::stable_sort(keep.begin(), keep.end());
}

void gcta::read_snplist(std::string snplistfile, std::vector<std::string> &snplist, std::string msg)
{
    // Read snplist file
    snplist.clear();
    std::string StrBuf;
    std::ifstream i_snplist(snplistfile.c_str());
    if (!i_snplist) LOGGER.e(0, "cannot open the file [" + snplistfile + "] to read.");
    LOGGER << "Reading a list of " << msg << " from [" + snplistfile + "]." << std::endl;
    while (i_snplist >> StrBuf) {
        snplist.push_back(StrBuf);
        std::getline(i_snplist, StrBuf);
    }
    i_snplist.close();
}

void gcta::extract_snp(std::string snplistfile)
{
    std::vector<std::string> snplist;
    read_snplist(snplistfile, snplist);
    update_id_map_kp(snplist, _snp_name_map, _include);
    LOGGER << _include.size() << " SNPs are extracted from [" + snplistfile + "]." << std::endl;
}

void gcta::extract_single_snp(std::string snpname)
{
    std::vector<std::string> snplist;
    snplist.push_back(snpname);
    update_id_map_kp(snplist, _snp_name_map, _include);
    if (_include.empty()) LOGGER.e(0, "cannot find the SNP [" + snpname + "] in the data.");
    else LOGGER << "Only the SNP [" + snpname + "] is included in the analysis." << std::endl;
}

void gcta::extract_region_snp(std::string snpname, int wind_size)
{
    LOGGER << "Extracting SNPs " << wind_size/1000 << "kb away from the SNP [" << snpname << "] in either direction ..." << std::endl;
    auto iter = _snp_name_map.find(snpname);
    int i = 0, j = 0;
    std::vector<std::string> snplist;
    if(iter==_snp_name_map.end()) LOGGER.e(0, "cannot find the SNP [" + snpname + "] in the data.");
    else{
        int bp = _bp[iter->second];
        int chr = _chr[iter->second];
        for(i = 0; i < _include.size(); i++){
            j = _include[i];
            if(_chr[j] == chr && abs(_bp[j]-bp) <= wind_size) snplist.push_back(_snp_name[j]);
        }
    }
    if(snplist.empty()) LOGGER.e(0, "no SNP found in this region.");
    update_id_map_kp(snplist, _snp_name_map, _include);
    LOGGER << _include.size() << " SNPs are extracted." << std::endl;
}

void gcta::extract_region_bp(int chr, int bp, int wind_size)
{
    LOGGER << "Extracting SNPs " << wind_size/1000 << "kb away from the position [chr=" << chr <<"; bp="<< bp << "] in either direction ..." << std::endl;
    int i = 0, j = 0;
    std::vector<std::string> snplist;
    for(i = 0; i < _include.size(); i++){
        j = _include[i];
        if(_chr[j] == chr && abs(_bp[j]-bp) <= wind_size) snplist.push_back(_snp_name[j]);
    }
    if(snplist.empty()) LOGGER.e(0, "no SNP found in this region.");
    update_id_map_kp(snplist, _snp_name_map, _include);
    LOGGER << _include.size() << " SNPs are extracted." << std::endl;
}

void gcta::exclude_snp(std::string snplistfile)
{
    std::vector<std::string> snplist;
    read_snplist(snplistfile, snplist);
    int prev_size = _include.size();
    update_id_map_rm(snplist, _snp_name_map, _include);
    LOGGER << prev_size - _include.size() << " SNPs are excluded from [" + snplistfile + "] and there are " << _include.size() << " SNPs remaining." << std::endl;
}

void gcta::exclude_region_snp(std::string snpname, int wind_size)
{
    LOGGER << "Excluding SNPs " << wind_size/1000 << "kb away from the SNP [" << snpname << "] in either direction ..." << std::endl;
    auto iter = _snp_name_map.find(snpname);
    int i = 0, j = 0;
    std::vector<std::string> snplist;
    if(iter==_snp_name_map.end()) LOGGER.e(0, "cannot find the SNP [" + snpname + "] in the data.");
    else{
        int bp = _bp[iter->second];
        int chr = _chr[iter->second];
        for(i = 0; i < _include.size(); i++){
            j = _include[i];
            if(_chr[j] == chr && abs(_bp[j]-bp) <= wind_size) snplist.push_back(_snp_name[j]);
        }
    }
    if(snplist.empty()) LOGGER.e(0, "no SNP found in this region.");
    update_id_map_rm(snplist, _snp_name_map, _include);
    LOGGER << _include.size() << " SNPs have been excluded." << std::endl;
}

void gcta::exclude_region_bp(int chr, int bp, int wind_size)
{
    LOGGER << "Excluding SNPs " << wind_size/1000 << "kb away from the position [chr=" << chr <<"; bp="<< bp << "] in either direction ..." << std::endl;
    int i = 0, j = 0;
    std::vector<std::string> snplist;
    for(i = 0; i < _include.size(); i++){
        j = _include[i];
        if(_chr[j] == chr && abs(_bp[j]-bp) <= wind_size) snplist.push_back(_snp_name[j]);
    }
    if(snplist.empty()) LOGGER.e(0, "no SNP found in this region.");
    update_id_map_rm(snplist, _snp_name_map, _include);
    LOGGER << _include.size() << " SNPs are excluded." << std::endl;
}

void gcta::exclude_single_snp(std::string snpname)
{
    std::vector<std::string> snplist;
    snplist.push_back(snpname);
    int include_size = _include.size();
    update_id_map_rm(snplist, _snp_name_map, _include);
    if (_include.size() == include_size) LOGGER.e(0, "cannot find the SNP [" + snpname + "] in the data.");
    else LOGGER << "The SNP [" + snpname + "] has been excluded from the analysis." << std::endl;
}

void gcta::extract_chr(int chr_start, int chr_end)
{
    std::map<std::string, int> id_map_buf(_snp_name_map);
    _snp_name_map.clear();
    _include.clear();
    for (const auto& [k, v] : id_map_buf) {
        if (_chr[v] >= chr_start && _chr[v] <= chr_end) {
            _snp_name_map.emplace(k, v);
            _include.push_back(v);
        }
    }
    std::stable_sort(_include.begin(), _include.end());
    if (chr_start != chr_end) LOGGER << _include.size() << " SNPs from chromosome " << chr_start << " to chromosome " << chr_end << " are included in the analysis." << std::endl;
    else LOGGER << _include.size() << " SNPs on chromosome " << chr_start << " are included in the analysis." << std::endl;
}

void gcta::filter_snp_maf(double maf)
{
    if (_mu.empty()) calcu_mu();

    LOGGER << "Filtering SNPs with MAF > " << maf << " ..." << std::endl;
    std::map<std::string, int> id_map_buf(_snp_name_map);
    int prev_size = _include.size();
    double fbuf = 0.0;
    _include.clear();
    _snp_name_map.clear();
    for (const auto& [k, v] : id_map_buf) {
        fbuf = _mu[v]*0.5;
        if (fbuf <= maf || (1.0 - fbuf) <= maf) continue;
        _snp_name_map.emplace(k, v);
        _include.push_back(v);
    }
    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");
    else {
        std::stable_sort(_include.begin(), _include.end());
        LOGGER << "After filtering SNPs with MAF > " << maf << ", there are " << _include.size() << " SNPs (" << prev_size - _include.size() << " SNPs with MAF < " << maf << ")." << std::endl;
    }
}

void gcta::filter_snp_max_maf(double max_maf)
{
    if (_mu.empty()) calcu_mu();

    LOGGER << "Filtering SNPs with MAF < " << max_maf << " ..." << std::endl;
    std::map<std::string, int> id_map_buf(_snp_name_map);
    int prev_size = _include.size();
    double fbuf = 0.0;
    _include.clear();
    _snp_name_map.clear();
    for (const auto& [k, v] : id_map_buf) {
        fbuf = _mu[v]*0.5;
        if (fbuf > max_maf && 1.0 - fbuf > max_maf) continue;
        _snp_name_map.emplace(k, v);
        _include.push_back(v);
    }
    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");
    else {
        std::stable_sort(_include.begin(), _include.end());
        LOGGER << "After filtering SNPs with MAF < " << max_maf << ", there are " << _include.size() << " SNPs (" << prev_size - _include.size() << " SNPs with MAF > " << max_maf << ")." << std::endl;
    }
}

void gcta::filter_impRsq(double rsq_cutoff)
{
    if (_impRsq.empty()) LOGGER.w(0, "the option --imput-rsq is inactive because GCTA can't find the imputation quality scores for the SNPs. Use the option --update-imput-rsq to input the imputation quality scores.");
    LOGGER << "Filtering SNPs with imputation Rsq > " << rsq_cutoff << " ..." << std::endl;
    std::map<std::string, int> id_map_buf(_snp_name_map);
    int prev_size = _include.size();
    _include.clear();
    _snp_name_map.clear();
    for (const auto& [k, v] : id_map_buf) {
        if (_impRsq[v] < rsq_cutoff) continue;
        _snp_name_map.emplace(k, v);
        _include.push_back(v);
    }
    if (_include.size() == 0) LOGGER.e(0, "no SNP is retained for analysis.");
    else {
        std::stable_sort(_include.begin(), _include.end());
        LOGGER << "After filtering for imputation Rsq > " << rsq_cutoff << ", there are " << _include.size() << " SNPs (" << prev_size - _include.size() << " SNPs with imputation Rsq < " << rsq_cutoff << ")." << std::endl;
    }
}

void gcta::read_indi_list(std::string indi_list_file, std::vector<std::string> &indi_list)
{
    std::ifstream i_indi_list(indi_list_file.c_str());
    if (!i_indi_list) LOGGER.e(0, "cannot open the file [" + indi_list_file + "] to read.");
    std::string str_buf, id_buf;
    indi_list.clear();
    while (i_indi_list) {
        i_indi_list >> str_buf;
        if (i_indi_list.eof()) break;
        id_buf = str_buf + ":";
        i_indi_list >> str_buf;
        id_buf += str_buf;
        indi_list.push_back(id_buf);
        std::getline(i_indi_list, str_buf);
    }
    i_indi_list.close();
}

void gcta::keep_indi(std::string indi_list_file) {
    std::vector<std::string> indi_list;
    read_indi_list(indi_list_file, indi_list);
    update_id_map_kp(indi_list, _id_map, _keep);
    LOGGER << _keep.size() << " individuals are kept from [" + indi_list_file + "]." << std::endl;
}

void gcta::remove_indi(std::string indi_list_file) {
    std::vector<std::string> indi_list;
    read_indi_list(indi_list_file, indi_list);
    int prev_size = _keep.size();
    update_id_map_rm(indi_list, _id_map, _keep);
    LOGGER << prev_size - _keep.size() << " individuals are removed from [" + indi_list_file + "] and there are " << _keep.size() << " individuals remaining." << std::endl;
}

void gcta::update_sex(std::string sex_file) {
    std::ifstream isex(sex_file.c_str());
    if (!isex) LOGGER.e(0, "cannot open the file [" + sex_file + "] to read.");
    int sex_buf = 0, icount = 0;
    std::string str_buf, fid, pid;
    LOGGER << "Reading sex information from [" + sex_file + "]." << std::endl;
    _sex.clear();
    _sex.resize(_indi_num);
    std::vector<int> confirm(_indi_num);
    while (isex) {
        isex >> fid;
        if (isex.eof()) break;
        isex >> pid;
        isex >> str_buf;
        if (str_buf != "1" && str_buf != "2" && str_buf != "M" && str_buf != "F") LOGGER.e(0, "unrecognized sex code: \"" + fid + " " + pid + " " + str_buf + "\" in [" + sex_file + "].");
        auto iter = _id_map.find(fid + ":" + pid);
        if (iter != _id_map.end()) {
            if (str_buf == "M" || str_buf == "1") _sex[iter->second] = 1;
            else if (str_buf == "F" || str_buf == "2") _sex[iter->second] = 2;
            confirm[iter->second] = 1;
            icount++;
        }
        std::getline(isex, str_buf);
    }
    isex.close();

    for (int i = 0; i < _keep.size(); i++) {
        if (confirm[_keep[i]] != 1) LOGGER.e(0, "the sex information for all of the included individuals should be updated.");
    }
    LOGGER << "Sex information for " << icount << " individuals are update from [" + sex_file + "]." << std::endl;
}

void gcta::update_allele_ref(std::string ref_A_file) {
    std::ifstream i_allele_ref(ref_A_file.c_str());
    if (!i_allele_ref) LOGGER.e(0, "cannot open the file [" + ref_A_file + "] to read.");
    int i = 0;
    std::string str_buf, ref_A_buf;
    LOGGER << "Reading reference alleles of SNPs from [" + ref_A_file + "]." << std::endl;
    int icount = 0;
    while (i_allele_ref) {
        i_allele_ref >> str_buf;
        if (i_allele_ref.eof()) break;
        auto iter = _snp_name_map.find(str_buf);
        i_allele_ref >> ref_A_buf;
        if (iter != _snp_name_map.end()) {
            _allele_ref[iter->second] = ref_A_buf;
            icount++;
        }
        std::getline(i_allele_ref, str_buf);
    }
    i_allele_ref.close();
    LOGGER << "Reference alleles of " << icount << " SNPs are updated from [" + ref_A_file + "]." << std::endl;
    if (icount != _snp_num) LOGGER.w(0, "reference alleles of " + std::to_string(_snp_num - icount) + " SNPs have not been updated.");
}

void gcta::calcu_mu(bool ssq_flag) {
    compact_indi_data();
    compact_snp_data();
    const int n = static_cast<int>(_keep.size());
    const int m = static_cast<int>(_include.size());

    // Build sex-aware weight vectors
    bool no_sex_info = false;
    Eigen::VectorXf auto_fac = Eigen::VectorXf::Ones(n);
    Eigen::VectorXf xfac(n);
    for (int i = 0; i < n; i++) {
        const int sex = _sex[i];
        if      (sex == 1) xfac[i] = 0.5f;
        else if (sex == 2) xfac[i] = 1.0f;
        else             { xfac[i] = 1.0f; no_sex_info = true; }
    }

    LOGGER << "Calculating allele frequencies ..." << std::endl;
    _mu.assign(_snp_num, 0.0);

    bool flag_x_problem = false;

    #pragma omp parallel for reduction(||:flag_x_problem)
    for (int j = 0; j < m; j++) {
        const bool is_x = (_legacy_homogametic_chr != 0 && _chr[j] == _legacy_homogametic_chr);
        if (is_x && no_sex_info) flag_x_problem = true;
        const Eigen::VectorXf& fac = is_x ? xfac : auto_fac;

        if (_dosage_flag) {
            auto col = _geno_dose.col(j);
            auto mask = (col.array() < DOSAGE_NA);
            const float fcount = mask.select(fac, 0.0f).sum();
            if (fcount > 0.0f)
                _mu[j] = static_cast<double>(
                    mask.select((col.array() * fac.array()).matrix(), 0.0f).sum() / fcount);
        } else {
            const auto &s1 = _snp_1[j];
            const auto &s2 = _snp_2[j];
            float fcount = 0.0f, mu_acc = 0.0f;
            for (int i = 0; i < n; i++) {
                if (!s1[i] || s2[i]) {
                    float g = static_cast<float>(s1[i] + s2[i]);
                    mu_acc += fac[i] * g;
                    fcount += fac[i];
                }
            }
            if (fcount > 0.0f)
                _mu[j] = static_cast<double>(mu_acc / fcount);
        }
    }

    if (flag_x_problem)
        LOGGER.w(0, "sex-code information (the 5th column of the .fam file) is required for analysis on the selected homogametic chromosome. GCTA assumes missing codes are homogametic (code 2).");
}

void gcta::calcu_maf()
{
    if (_mu.empty()) calcu_mu();

    int i = 0, m = _include.size();
    _maf.resize(m);
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        _maf[i] = 0.5*_mu[i];
        if(_maf[i] > 0.5) _maf[i] = 1.0 - _maf[i];
    }
}

void gcta::update_impRsq(std::string zinfofile) {
    std::ifstream iRsq(zinfofile.c_str());
    if (!iRsq) LOGGER.e(0, "cannot open the file [" + zinfofile + "] to read.");

    std::string snp_name_buf, str_buf;
    double fbuf = 0.0;
    LOGGER << "Reading imputation Rsq of the SNPs from [" + zinfofile + "]." << std::endl;
    _impRsq.clear();
    _impRsq.resize(_snp_num, 0.0);
    int icount = 0;
    while (iRsq) {
        iRsq >> snp_name_buf;
        if (iRsq.eof()) break;
        auto iter = _snp_name_map.find(snp_name_buf);
        iRsq >> str_buf;
        fbuf = atof(str_buf.c_str());
        if (iter != _snp_name_map.end()) {
            if (fbuf > 2.0 || fbuf < 0.0) LOGGER.e(0, "invalid value of imputation Rsq for the SNP " + snp_name_buf + ".");
            _impRsq[iter->second] = fbuf;
            icount++;
        }
        std::getline(iRsq, str_buf);
    }
    iRsq.close();

    LOGGER << "Imputation Rsq of " << icount << " SNPs are updated from [" + zinfofile + "]." << std::endl;
    if (icount != _snp_num) LOGGER.w(0, "imputation Rsq of " + std::to_string(_snp_num - icount) + " SNPs have not been updated.");
}

void gcta::update_freq(std::string freq) {
    std::ifstream ifreq(freq.c_str());
    if (!ifreq) LOGGER.e(0, "cannot open the file [" + freq + "] to read.");
    int i = 0;
    std::string ref_A_buf;
    double fbuf = 0.0;
    std::string snp_name_buf, str_buf;
    LOGGER << "Reading allele frequencies of the SNPs from [" + freq + "]." << std::endl;
    _mu.clear();
    _mu.resize(_snp_num, 0.0);
    int icount = 0;
    while (ifreq) {
        ifreq >> snp_name_buf;
        if (ifreq.eof()) break;
        auto iter = _snp_name_map.find(snp_name_buf);
        ifreq >> ref_A_buf;
        ifreq >> str_buf;
        fbuf = atof(str_buf.c_str());
        if (iter != _snp_name_map.end()) {
            if (fbuf > 1.0 || fbuf < 0.0) LOGGER.e(0, "invalid value of allele frequency for the SNP " + snp_name_buf + ".");
            _mu[iter->second] = fbuf * 2.0;
            icount++;
        }
        std::getline(ifreq, str_buf);
    }
    ifreq.close();

    LOGGER << "Allele frequencies of " << icount << " SNPs are updated from [" + freq + "]." << std::endl;
    if (icount != _snp_num) LOGGER.w(0, "allele frequencies of " + std::to_string(_snp_num - icount) + " SNPs have not been updated.");
}

void gcta::save_freq(bool ssq_flag) {
    if (_mu.empty()) 
      calcu_mu(ssq_flag);
    // Prefer _additive_mu (real allele frequency, captured before any nonadditive
    // recoding) when available; falls back to _mu for the additive model.
    const std::vector<double>& freq_mu = _additive_mu.empty() ? _mu : _additive_mu;
    std::string save_freq = _out + ".freq";
    std::ofstream ofreq(save_freq.c_str());
    if (!ofreq) LOGGER.e(0, "cannot open the file [" + save_freq + "] to write.");
    int i = 0;
    LOGGER << "Writing allele frequencies of " << _include.size() << " SNPs to [" + save_freq + "]." << std::endl;
    for (i = 0; i < _include.size(); i++) {
        ofreq << _snp_name[_include[i]] << "\t" << _allele_ref[_include[i]] << "\t" << std::setprecision(15) << freq_mu[_include[i]]*0.5;
        //        if(ssq_flag) ofreq<<"\t"<<_ssq[_include[i]]<<"\t"<<_w[_include[i]];
        ofreq << std::endl;
    }
    ofreq.close();
    LOGGER << "Allele frequencies of " << _include.size() << " SNPs have been saved in the file [" + save_freq + "]." << std::endl;
}

void gcta::read_indi_blup(std::string blup_indi_file) {
    std::vector< std::vector<std::string> > g_buf;
    std::ifstream i_indi_blup(blup_indi_file.c_str());
    if (!i_indi_blup) LOGGER.e(0, "cannot open the file [" + blup_indi_file + "] to read.");
    std::string str_buf, id_buf;
    std::vector<std::string> id, vs_buf;
    int i = 0, j = 0, k = 0, col_num = 0;
    while (i_indi_blup) {
        i_indi_blup >> str_buf;
        if (i_indi_blup.eof()) break;
        id_buf = str_buf + ":";
        i_indi_blup >> str_buf;
        id_buf += str_buf;
        std::getline(i_indi_blup, str_buf);
        col_num = StrFunc::split_string(str_buf, vs_buf, " \t\n");
        if (col_num < 1) continue;
        id.push_back(id_buf);
        g_buf.push_back(vs_buf);
    }
    i_indi_blup.close();

    update_id_map_kp(id, _id_map, _keep);
    std::map<std::string, int> uni_id_map;
    for (i = 0; i < _keep.size(); i++) uni_id_map.insert(std::pair<std::string, int>(_fid[_keep[i]] + ":" + _pid[_keep[i]], i));
    _varcmp_Py.setZero(_keep.size(), col_num / 2);
    for (i = 0; i < id.size(); i++) {
        auto iter = uni_id_map.find(id[i]);
        if (iter == uni_id_map.end()) continue;
        for (j = 0, k = 0; j < col_num; j += 2, k++) _varcmp_Py(iter->second, k) = atof(g_buf[i][j].c_str());
    }
    LOGGER << "BLUP solution to the total genetic effects for " << _keep.size() << " individuals have been read from [" + blup_indi_file + "]." << std::endl;
}

bool gcta::make_XMat(Eigen::MatrixXf &X)
{
    compact_indi_data();
    if (!_make_XMat_no_compact) compact_snp_data();
    if (_mu.empty()) calcu_mu();

    LOGGER << "Recoding genotypes (individual major mode) ..." << std::endl;
    bool have_mis = false;
    unsigned long i = 0, j = 0, n = _keep.size(), m = _include.size();

    X.resize(0,0);
    X.resize(n, m);
    #pragma omp parallel for private(j)
    for (i = 0; i < n; i++) {
        if (_dosage_flag) {
            for (j = 0; j < m; j++) {
                const float d = _geno_dose(i, j);
                if (d < DOSAGE_NA) {
                    X(i,j) = d;
                } else {
                    X(i,j) = DOSAGE_NA;
                    have_mis = true;
                }
            }
        }
        else {
            for (j = 0; j < m; j++) {
                if (!_snp_1[j][i] || _snp_2[j][i]) {
                    X(i,j) = _snp_1[j][i] + _snp_2[j][i];
                } 
                else {
                    X(i,j) = DOSAGE_NA;
                    have_mis = true;
                }
            }
        }
    }
    if (_dosage_flag) _geno_dose.resize(0, 0); // free after copy into X
    return have_mis;
}

bool gcta::make_XMat_d(Eigen::MatrixXf &X)
{
    compact_indi_data();
    compact_snp_data();
    if (_mu.empty()) calcu_mu();

    LOGGER << "Recoding genotypes for dominance effects (individual major mode) ..." << std::endl;
    unsigned long i = 0, j = 0, n = _keep.size(), m = _include.size();
    bool have_mis = false;

    X.resize(0,0);
    X.resize(n, m);
    #pragma omp parallel for private(j)
    for (i = 0; i < n; i++) {
        if (_dosage_flag) {
            for (j = 0; j < m; j++) {
                const float d = _geno_dose(i, j);
                if (d < DOSAGE_NA) {
                    float g = d;
                    if      (g < 0.5f) g = 0.0f;
                    else if (g < 1.5f) g = static_cast<float>(_mu[j]);
                    else               g = static_cast<float>(2.0 * _mu[j] - 2.0);
                    X(i,j) = g;
                } else {
                    X(i,j) = DOSAGE_NA;
                    have_mis = true;
                }
            }
        }
        else {
            for (j = 0; j < m; j++) {
                if (!_snp_1[j][i] || _snp_2[j][i]) {
                    X(i,j) = _snp_1[j][i] + _snp_2[j][i];
                    if (X(i,j) < 0.5) X(i,j) = 0.0;
                    else if (X(i,j) < 1.5) X(i,j) = _mu[j];
                    else X(i,j) = (2.0 * _mu[j] - 2.0);
                } 
                else{
                    X(i,j) = DOSAGE_NA;
                    have_mis = true;
                }
            }
        }
    }
    if (_dosage_flag) _geno_dose.resize(0, 0); // free after copy into X
    return have_mis;
}

void gcta::std_XMat(Eigen::MatrixXf &X, eigenVector &sd_SNP, bool grm_homogametic_flag, bool miss_with_mu, bool divid_by_std)
{
    compact_indi_data();
    compact_snp_data();
    if (_mu.empty()) calcu_mu();

    unsigned long i = 0, j = 0, n = _keep.size(), m = _include.size();
    sd_SNP.resize(m);
    if (_dosage_flag) {
        for (j = 0; j < m; j++)  sd_SNP[j] = (X.col(j) - Eigen::VectorXf::Constant(n, _mu[j])).squaredNorm() / (n - 1.0);
    } 
    else {
        for (j = 0; j < m; j++) sd_SNP[j] = _mu[j]*(1.0 - 0.5 * _mu[j]);
    }
    if (divid_by_std) {
        for (j = 0; j < m; j++) {
            if (fabs(sd_SNP[j]) < 1.0e-50) sd_SNP[j] = 0.0;
            else sd_SNP[j] = sqrt(1.0 / sd_SNP[j]);
        }
    }

    #pragma omp parallel for private(j)
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (X(i,j) < DOSAGE_NA) {
                X(i,j) -= _mu[j];
                if (divid_by_std) X(i,j) *= sd_SNP[j];
            } 
            else if (miss_with_mu) X(i,j) = 0.0;
        }
    }

    if (!grm_homogametic_flag) return;

    // for homogametic chromosomes
    check_sex();
    double f_buf = sqrt(0.5);

    #pragma omp parallel for private(j)
    for (i = 0; i < n; i++) {
        if (_sex[i] == 1) {
            for (j = 0; j < m; j++) {
                if (X(i,j) < DOSAGE_NA) X(i,j) *= f_buf;
                else if (miss_with_mu) X(i,j) = 0.0;
            }
        }
    }
}

void gcta::std_XMat_d(Eigen::MatrixXf &X, eigenVector &sd_SNP, bool miss_with_mu, bool divid_by_std)
{
    compact_indi_data();
    compact_snp_data();
    if (_mu.empty()) calcu_mu();

    unsigned long i = 0, j = 0, n = _keep.size(), m = _include.size();
    sd_SNP.resize(m);
    if (_dosage_flag) {
        #pragma omp parallel for private(i)
        for (j = 0; j < m; j++) {
            for (i = 0; i < n; i++) {
                double d_buf = (X(i,j) - _mu[j]);
                sd_SNP[j] += d_buf*d_buf;
            }
            sd_SNP[j] /= (n - 1.0);
        }
    } 
    else {
        for (j = 0; j < m; j++) sd_SNP[j] = _mu[j]*(1.0 - 0.5 * _mu[j]);
    }
    if (divid_by_std) {
        for (j = 0; j < m; j++) {
            if (fabs(sd_SNP[j]) < 1.0e-50) sd_SNP[j] = 0.0;
            else sd_SNP[j] = 1.0 / sd_SNP[j];
        }
    } 
    else {
        for (j = 0; j < m; j++) sd_SNP[j] = sd_SNP[j] * sd_SNP[j];
    }
    std::vector<double> psq(m);
    for (j = 0; j < m; j++) psq[j] = 0.5 * _mu[j] * _mu[j];

    #pragma omp parallel for private(j)
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (X(i,j) < DOSAGE_NA) {
                X(i,j) -= psq[j];
                if (divid_by_std) X(i,j) *= sd_SNP[j];
            } 
            else if (miss_with_mu) X(i,j) = 0.0;
        }
    }
}

void gcta::makex_eigenVector(int j, eigenVector &x, bool resize, bool minus_2p)
{
    int i = 0;
    compact_indi_data();
    compact_snp_data();
    if (resize) x.resize(_keep.size());
    #pragma omp parallel for
    for (i = 0; i < _keep.size(); i++) {
        if (!_snp_1[j][i] || _snp_2[j][i]) {
            x[i] = (_snp_1[j][i] + _snp_2[j][i]);
        }
        else x[i] = _mu[j];
        if (minus_2p) x[i] -= _mu[j];
    }
}

//change here: returns standardized genotypes
void gcta::makex_eigenVector_std(int j, eigenVector &x, bool resize, double snp_std)
{
    int i = 0;
    compact_indi_data();
    compact_snp_data();
    if (resize) x.resize(_keep.size());
    #pragma omp parallel for
    for (i = 0; i < _keep.size(); i++) {
        if (!_snp_1[j][i] || _snp_2[j][i]) {
            x[i] = (_snp_1[j][i] + _snp_2[j][i]);
        }
        else x[i] = _mu[j];
        // change here: subtract mean and divide by std
        x[i] -= _mu[j];
        x[i] /= snp_std;
    }
}


void gcta::save_XMat(bool miss_with_mu, bool std)
{
    if(std && _dosage_flag) LOGGER.e(0, "the --recode-std is invalid for dosage data.");
    compact_indi_data();
    if (_dosage_flag) compact_dosage_data();
    else compact_snp_data();
    if ( (miss_with_mu || std) && _mu.empty()) calcu_mu();

    int i = 0, j = 0, m = _include.size();
    eigenVector sd_SNP;
    if(std){
        sd_SNP.resize(m);
        for (j = 0; j < m; j++) {
            sd_SNP(j) = _mu[j]*(1.0 - 0.5 * _mu[j]);
            if (fabs(sd_SNP(j)) < 1.0e-50) sd_SNP(j) = 0.0;
            else sd_SNP(j) = sqrt(1.0 / sd_SNP(j));
        }
    }

    // Save matrix X
    double x_buf = 0.0;
    std::string X_zFile = _out + ".xmat.gz";
    std::ofstream raw_file(X_zFile, std::ios::binary);
    if (!raw_file.is_open()) LOGGER.e(0, "cannot open the file [" + X_zFile + "] to write.");
    boost::iostreams::filtering_ostream zoutf;
    zoutf.push(boost::iostreams::gzip_compressor());
    zoutf.push(raw_file);
    LOGGER << "Saving the recoded genotype matrix to the file [" + X_zFile + "]." << std::endl;
    zoutf << "FID IID ";
    for (j = 0; j < m; j++) zoutf << _snp_name[j] << " ";
    zoutf << std::endl;
    zoutf << "Reference Allele ";
    for (j = 0; j < m; j++) zoutf << _allele_ref[j] << " ";
    zoutf << std::endl;
    for (i = 0; i < _keep.size(); i++) {
        zoutf << _fid[i] << ' ' << _pid[i] << ' ';
        if (_dosage_flag) {
            for (j = 0; j < m; j++) {
                const float d = _geno_dose(i, j);
                if (d < DOSAGE_NA) {
                    x_buf = d;
                    if(std) x_buf = (x_buf - _mu[j]) * sd_SNP(j);
                    zoutf << x_buf << ' ';
                } else {
                    if(std) zoutf << "0 ";
                    else{
                        if (miss_with_mu) zoutf << _mu[j] << ' ';
                        else zoutf << "NA ";
                    }
                }
            }
        } else {
            for (j = 0; j < m; j++) {
                if (!_snp_1[j][i] || _snp_2[j][i]) {
                    x_buf = _snp_1[j][i] + _snp_2[j][i];
                    if(std) x_buf = (x_buf - _mu[j]) * sd_SNP(j);
                    zoutf << x_buf << ' ';                    
                } else {
                    if(std) zoutf << "0 ";
                    else {
                        if (miss_with_mu) zoutf << _mu[j] << ' ';
                        else zoutf << "NA ";
                    }
                }
            }
        }
        zoutf << std::endl;
    }
    zoutf.reset();
    raw_file.close();
    LOGGER << "The recoded genotype matrix has been saved in the file [" + X_zFile + "] (in compressed text format)." << std::endl;
}

bool gcta::make_XMat_subset(Eigen::MatrixXf &X, std::vector<int> &snp_indx, bool divid_by_std)
{
    if(snp_indx.empty()) return false;
    compact_indi_data();
    if (!_make_XMat_no_compact) compact_snp_data();
    return make_XMat_subset_dense(X, snp_indx, divid_by_std);
}

bool gcta::make_XMat_subset_dense(Eigen::MatrixXf &X, std::vector<int> &snp_indx, bool divid_by_std)
{
    if(snp_indx.empty()) return false;
    const int n = _keep.size(), m = snp_indx.size();

    // Guard: only resize when strictly necessary.  If the caller pre-allocated
    // X(n, max_block_size), this is a no-op for every full block and avoids a
    // reallocation on the last partial block (where m < max_block_size).
    if (X.rows() != n || X.cols() < m)
        X.resize(n, m);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < m; j++) {
        const int k = snp_indx[j];
        auto col = X.col(j);
        if (_dosage_flag) {
            const float mu_k = static_cast<float>(_mu[k]);
            float scale = 1.0f;
            if (divid_by_std) {
                const double sd = _mu[k] * (1.0 - 0.5 * _mu[k]);
                if (sd > 1.0e-50) scale = static_cast<float>(1.0 / std::sqrt(sd));
            }
            for (int i = 0; i < n; i++) {
                const float d = _geno_dose(i, k);
                if (d < DOSAGE_NA) col(i) = (d - mu_k) * scale;
                else col(i) = 0.0f;
            }
        } else {
            const auto& s1 = _snp_1[k];
            const auto& s2 = _snp_2[k];
            const float mu_k = static_cast<float>(_mu[k]);
            float scale = 1.0f;
            if (divid_by_std) {
                const double sd = _mu[k] * (1.0 - 0.5 * _mu[k]);
                if (sd > 1.0e-50) scale = static_cast<float>(1.0 / std::sqrt(sd));
            }
            for (int i = 0; i < n; i++) {
                if (!s1[i] || s2[i]) {
                    float geno = static_cast<float>(s1[i]) + static_cast<float>(s2[i]);
                    col(i) = (geno - mu_k) * scale;
                } else {
                    col(i) = 0.0f;
                }
            }
        }
    }

    return true;
}

bool gcta::make_XMat_d_subset(Eigen::MatrixXf &X, std::vector<int> &snp_indx, bool divid_by_std)
{
    if(snp_indx.empty()) return false;
    compact_indi_data();
    compact_snp_data();
    return make_XMat_d_subset_dense(X, snp_indx, divid_by_std);
}

bool gcta::make_XMat_d_subset_dense(Eigen::MatrixXf &X, std::vector<int> &snp_indx, bool divid_by_std)
{
    if(snp_indx.empty()) return false;
    if (_mu.empty()) calcu_mu();

    const int n = _keep.size(), m = snp_indx.size();
    X.resize(n, m);

    // Column-parallel fill (column-major layout: each col is contiguous)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < m; j++) {
        const int k = snp_indx[j];
        auto col = X.col(j);
        if (_dosage_flag) {
            const float mu_k = static_cast<float>(_mu[k]);
            const float psq = 0.5f * mu_k * mu_k;
            for (int i = 0; i < n; i++) {
                if (_geno_dose(i, k) < DOSAGE_NA) {
                    float g = _geno_dose(i, k);
                    if      (g < 0.5f) g = 0.0f;
                    else if (g < 1.5f) g = mu_k;
                    else               g = 2.0f * mu_k - 2.0f;
                    col(i) = g - psq;
                } else {
                    col(i) = 0.0f;
                }
            }
        } else {
            const auto& s1 = _snp_1[k];
            const auto& s2 = _snp_2[k];
            const float mu_k = static_cast<float>(_mu[k]);
            const float psq = 0.5f * mu_k * mu_k;
            for (int i = 0; i < n; i++) {
                if (!s1[i] || s2[i]) {
                    float g = static_cast<float>(s1[i]) + static_cast<float>(s2[i]);
                    if      (g < 0.5f) g = 0.0f;
                    else if (g < 1.5f) g = mu_k;
                    else               g = 2.0f * mu_k - 2.0f;
                    col(i) = g - psq;
                } else {
                    col(i) = 0.0f;
                }
            }
        }
    }

    if (divid_by_std) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < m; j++) {
            const int k = snp_indx[j];
            const double sd = _mu[k] * (1.0 - 0.5 * _mu[k]);
            if (sd > 1.0e-50)
                X.col(j) *= static_cast<float>(1.0 / sd);
        }
    }

    return true;
}

