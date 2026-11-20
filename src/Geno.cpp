/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   New implementation: read and process genotype of plink format in block way.

   Depends on the class of marker and phenotype

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/

#define NOMINMAX
#include "Geno.h"
#include "constants.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iterator>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "utils.hpp"
#include "omp.h"
#include "GenoBackendFactory.h"
#include <cstring>
#include "OptionIO.h"
#include "zlib.h"
#include "zstd.h"
#include <cstring>
#include "cpu.h"
#include <Eigen/Eigen>
#include <algorithm>
#include "third_party/Pgenlib/PgenReader.h"
#include <numeric>
#include <array>
#include <bit>

#ifdef _WIN64
  #include <intrin.h>
  uint32_t __inline CTZ64U(uint64_t value){
      unsigned long tz = 0;
      _BitScanForward64(&tz, value);
      return tz;
  }
  
  uint32_t __inline CLZ64U(uint64_t value){
      unsigned long lz = 0;
      _BitScanReverse64(&lz, value);
      return 63 - lz;
  }
#else
  //#define CTZU __builtin_ctz
  //#define CLZU __builtin_clz
  // __builtin_ctzll/__builtin_clzll are compiler builtins (GCC/Clang), not x86
  // intrinsics. Both are portable across x86 and ARM64; no multiversioning needed.
  uint32_t CTZ64U(uint64_t value){
      return __builtin_ctzll(value);
  }

  uint32_t CLZ64U(uint64_t value){
      return __builtin_clzll(value);
  }

#endif

// Spreads the low 32 bits of x into alternating bit positions (bit deposit).
// On x86 Linux: function multiversioning selects _pdep_u64 (BMI2) at runtime.
// On ARM64: scalar fallback only. SVE2 has BDEP (exact equivalent of _pdep_u64)
// but it is absent on Apple Silicon and not universally available on Linux ARM64.
#if defined(__linux__) && GCTA_CPU_x86
__attribute__((target("default")))
#endif
uint64_t fill_inter_zero(uint64_t x) {
   uint64_t t;
   t = (x ^ (x >> 16)) & 0x00000000FFFF0000;
   x = x ^ t ^ (t << 16);
   t = (x ^ (x >> 8)) & 0x0000FF000000FF00;
   x = x ^ t ^ (t << 8);
   t = (x ^ (x >> 4)) & 0x00F000F000F000F0;
   x = x ^ t ^ (t << 4);
   t = (x ^ (x >> 2)) & 0x0C0C0C0C0C0C0C0C;
   x = x ^ t ^ (t << 2);
   t = (x ^ (x >> 1)) & 0x2222222222222222;
   x = x ^ t ^ (t << 1);
   return x;
}
#if defined(__linux__) && GCTA_CPU_x86
#include <x86intrin.h>
__attribute__((target("bmi2")))
uint64_t fill_inter_zero(uint64_t x) {
    return _pdep_u64(x, 0x5555555555555555U);
}
#endif


typedef uint32_t halfword_t;
const uintptr_t k1LU = (uintptr_t)1;

using std::thread;
using std::to_string;

Geno::Geno(Pheno* pheno,
           Marker* marker,
           const gcta_cli::GenoInputOptions& geno_opts,
           const gcta_cli::MarkerFilterOptions& marker_filter,
           const std::string& out_prefix)
    : out_prefix_(out_prefix),
      opt_min_maf(marker_filter.maf.value_or(0.0)),
      opt_max_maf(marker_filter.max_maf.value_or(0.5)),
      opt_geno_rate(marker_filter.geno.value_or(1.0)),
      opt_info_score(marker_filter.info.value_or(0.0)),
      opt_dos_dc(marker_filter.dc.value_or(-1.0)),
      opt_hard_call_thresh(marker_filter.hard_call_thresh.value_or(0.9)),
      opt_dosage_call(marker_filter.dosage_call) {
    geno_files.clear();
    int num_geno = 0;
    if (!geno_opts.bfile.empty()) {
        const string f = geno_opts.bfile + ".bed";
        requireFileReadable("--bfile", f);
        genoFormat = "BED";
        geno_files.push_back(f);
        num_geno++;
        hasInfo = false;
    }

    if (!geno_opts.mbfile.empty()) {
        genoFormat = "BED";
        geno_files = expandFileList(geno_opts.mbfile, ".bed");
        num_geno++;
        hasInfo = false;
    }

    if (!geno_opts.pfile.empty()) {
        const string f = geno_opts.pfile + ".pgen";
        requireFileReadable("--pfile", f);
        genoFormat = "PGEN";
        geno_files.push_back(f);
        num_geno++;
        hasInfo = false;
    }

    if (!geno_opts.bpfile.empty()) {
        const string f = geno_opts.bpfile + ".pgen";
        requireFileReadable("--bpfile", f);
        genoFormat = "PGEN";
        geno_files.push_back(f);
        num_geno++;
        hasInfo = false;
    }

    if (!geno_opts.mpfile.empty()) {
        genoFormat = "PGEN";
        geno_files = expandFileList(geno_opts.mpfile, ".pgen");
        num_geno++;
        hasInfo = false;
    }

    if (!geno_opts.mbpfile.empty()) {
        genoFormat = "PGEN";
        geno_files = expandFileList(geno_opts.mbpfile, ".pgen");
        num_geno++;
        hasInfo = false;
    }

    if (!geno_opts.bgen.empty()) {
        requireFileReadable("--bgen", geno_opts.bgen);
        genoFormat = "BGEN";
        geno_files.push_back(geno_opts.bgen);
        num_geno++;
        hasInfo = true;
    }

    if (!geno_opts.mbgen.empty()) {
        genoFormat = "BGEN";
        geno_files = expandFileList(geno_opts.mbgen, ".bgen");
        num_geno++;
        hasInfo = true;
    }

    if (num_geno == 0) {
        LOGGER.e(0, "no genotype file is specified");
    }

    this->pheno = pheno;
    this->marker = marker;

    // Resolve the unpacking function once so getGenoDouble() is a direct
    // member-function call on every marker, not a hash-map lookup.
    if      (genoFormat == "BED")  getGenoDoubleFunc = &Geno::getGenoDouble_bed;
    else if (genoFormat == "PGEN") getGenoDoubleFunc = &Geno::getGenoDouble_pgen;
    else if (genoFormat == "BGEN") getGenoDoubleFunc = &Geno::getGenoDouble_bgen;
    else LOGGER.e(0, "Geno: unknown genoFormat '" + genoFormat + "' — cannot resolve getGenoDouble.");

    if      (genoFormat == "BED")  getGenoFloatFunc = &Geno::getGenoFloat_bed;
    else if (genoFormat == "PGEN") getGenoFloatFunc = &Geno::getGenoFloat_pgen;
    else if (genoFormat == "BGEN") getGenoFloatFunc = &Geno::getGenoFloat_bgen;

    string alleleFileName;
    if (!geno_opts.update_freq.empty()) {
        requireFileReadable("--update-freq", geno_opts.update_freq);
        alleleFileName = geno_opts.update_freq;
    }

    setMAF(opt_min_maf);
    setMaxMAF(opt_max_maf);
    setFilterInfo(opt_info_score);
    setFilterMiss(1.0 - opt_geno_rate);

    string filterprompt = "Threshold to filter variants:";
    bool outFilterPrompt = false;
    if (opt_min_maf != 0.0) {
        filterprompt += " MAF > " + to_string(opt_min_maf);
        outFilterPrompt = true;
    }
    if (opt_max_maf != 0.5) {
        filterprompt += string(outFilterPrompt ? "," : "") + " MAF < " + to_string(opt_max_maf);
        outFilterPrompt = true;
    }
    if (opt_info_score != 0.0) {
        filterprompt += string(outFilterPrompt ? "," : "") + " imputation INFO score > " + to_string(opt_info_score);
        outFilterPrompt = true;
    }

    if (opt_geno_rate != 1.0) {
        filterprompt += string(outFilterPrompt ? "," : "") + " missingness rate < " + to_string(opt_geno_rate);
        outFilterPrompt = true;
    }
    if (outFilterPrompt) {
        LOGGER << filterprompt << "." << std::endl;
    }
    if (opt_dos_dc == 1.0) {
        iGRMdc = 1;
        iDC = 1;
        LOGGER << "Switch to the full dosage compensation mode." << std::endl;
    } else if (opt_dos_dc == 0.0) {
        iGRMdc = 0;
        iDC = 0;
        LOGGER << "Switch to the no dosage compensation mode." << std::endl;
    } else {
        // default equal variance mode for grm
        iGRMdc = -1;
        // compensation mode for x
        iDC = 1;
    }

    init_AF(alleleFileName);
}

Geno::~Geno(){
}

uint32_t Geno::getTotalMarker(){
    return total_markers;
}


//
//true:  filtered; false: not necessary to filter
bool Geno::filterMAF(){
    if((opt_min_maf != 0.0) || (opt_max_maf != 0.5)){
        LOGGER.i(0, "Computing allele frequencies...");

        int N = static_cast<int>(marker->count_extract());
        AFA1.assign(N, 0.0);
        countMarkers.assign(N, 0);
        vector<uint32_t> extractIdx(N);
        std::iota(extractIdx.begin(), extractIdx.end(), 0);

        // Temporarily open all filters so getGenoDouble computes AF for every
        // marker regardless of MAF/missingness thresholds.
        double sv_min = min_maf, sv_max = max_maf, sv_miss = dFilterMiss;
        min_maf = 0.0; max_maf = 1.0; dFilterMiss = 0.0;

        loopDouble(extractIdx, Constants::NUM_MARKER_READ, false, false, false, false,
            {[this](uintptr_t *buf, std::span<const uint32_t> exIdx) {
                int n = static_cast<int>(exIdx.size());
                #pragma omp parallel for schedule(static)
                for(int i = 0; i < n; ++i){
                    GenoBufItem item;
                    item.extractedMarkerIndex = exIdx[i];
                    getGenoDouble(buf, i, &item);
                    if(item.valid){
                        AFA1[exIdx[i]]         = item.af;
                        countMarkers[exIdx[i]] = item.nValidAllele;
                    }
                }
            }}, false);

        min_maf = sv_min; max_maf = sv_max; dFilterMiss = sv_miss;

        // Apply MAF filter
        double min_maf_eps = opt_min_maf * (1.0 - Constants::SMALL_EPSILON);
        double max_maf_eps = opt_max_maf * (1.0 + Constants::SMALL_EPSILON);
        LOGGER.d(0, "min_maf: " + to_string(min_maf_eps) + " max_maf: " + to_string(max_maf_eps));
        vector<uint32_t> extract_index;
        double cur_AF;

        for(int index = 0; index != static_cast<int>(AFA1.size()); index++){
            cur_AF = AFA1[index];
            if(cur_AF > 0.5) cur_AF = 1.0 - cur_AF;
            if((cur_AF > min_maf_eps) && (cur_AF < max_maf_eps)){
                extract_index.push_back(index);
                LOGGER.d(0, to_string(index) + ": " + to_string(cur_AF));
            }
        }

        vector<double> AFA1o = AFA1;
        vector<uint32_t> countMarkerso = countMarkers;

        AFA1.resize(extract_index.size());
        countMarkers.resize(extract_index.size());

        #pragma omp parallel for
        for(uint32_t index = 0; index < extract_index.size(); index++){
            uint32_t cur_index = extract_index[index];
            AFA1[index]        = AFA1o[cur_index];
            countMarkers[index] = countMarkerso[cur_index];
        }

        marker->keep_extracted_index(extract_index);

        LOGGER.i(0, to_string(extract_index.size()) + " SNPs remain from --maf or --max-maf,  ");
        return true;
    }else{
        return false;
    }
}

void Geno::init_AF(string alleleFileName) {
    AFA1.clear();
    //countA1A2.clear();
    //countA1A1.clear();
    //countA2A2.clear();
    countMarkers.clear();
    //RDev.clear();
    if(!alleleFileName.empty()){
        LOGGER.i(0, "Reading frequencies from [" + alleleFileName + "]...");
        vector<int> field_return = {2};
        vector<string> fields;
        marker->matchSNPListFile(alleleFileName, 3, field_return, fields);

        AFA1.resize(fields.size());
        vector<uint32_t> extract_index;
        bool filterByMaf = false;
        for(int i = 0; i < fields.size(); i++){
            double af;
            try{
                af = stod(fields[i]);
            }catch(std::out_of_range &){
                LOGGER.e(0, "the third column should be numeric");
            }
            if(af < 0 || af > 1.0){
                LOGGER.e(0, "frequency values should range from 0 to 1");
            }
            double maf = std::min(af, 1.0 - af);
            if(maf > min_maf && maf < max_maf){
                extract_index.push_back(i);
            }else{
                filterByMaf = true;
            }
            AFA1[i] = af;
        }
        LOGGER.i(0, "Frequencies of " + to_string(AFA1.size()) + " SNPs are updated.");

        marker->keep_extracted_index(extract_index);
        vector<double> AFA1o = AFA1;
        //vector<uint32_t> countMarkerso = countMarkers;

        AFA1.resize(extract_index.size());
        countMarkers.resize(extract_index.size());

        #pragma omp parallel for
        for(uint32_t index = 0; index < extract_index.size(); index++){
            uint32_t cur_index = extract_index[index];
            AFA1[index] = AFA1o[cur_index];
           // countMarkers[index] = countMarkerso[cur_index];
        }
        if(filterByMaf)LOGGER << "  " << extract_index.size() << " SNPs remain after MAF filtering." << std::endl;
        bHasPreAF = true;
    }
}

void Geno::out_freq(string filename){
    string name_frq = filename + ".frq";
    LOGGER.i(0, "Saving allele frequencies...");
    std::ofstream o_freq(name_frq.c_str());
    if (!o_freq) { LOGGER.e(0, "cannot open the file [" + name_frq + "] to write"); }
    vector<string> out_contents;
    out_contents.reserve(AFA1.size() + 1);
    out_contents.push_back("CHR\tSNP\tPOS\tA1\tA2\tAF\tNCHROBS");
    for(int i = 0; i != AFA1.size(); i++){
        out_contents.push_back(marker->get_marker(marker->getRawIndex(i)) + "\t" + to_string(AFA1[i])
                               + "\t" + to_string(countMarkers[i]));
    }
    std::copy(out_contents.begin(), out_contents.end(), std::ostream_iterator<string>(o_freq, "\n"));
    o_freq.close();
    LOGGER.i(0, "Allele frequencies of " + to_string(AFA1.size()) + " SNPs have been saved in the file [" + name_frq + "]");
}

bool Geno::getGenoHasInfo(){
    return hasInfo;
}

void Geno::setGRMMode(bool grm, bool dominace){
    this->bGRM = grm;
    this->bGRMDom = dominace;
    if (dominace) {
        genoCodingModel = GenoCodingModel::DOMINANCE;
    } else if (genoCodingModel == GenoCodingModel::DOMINANCE) {
        genoCodingModel = GenoCodingModel::ADDITIVE;
    }
}

void Geno::setGenoCodingModel(gcta_cli::GenoModel model){
    switch (model) {
        case gcta_cli::GenoModel::Additive:    genoCodingModel = GenoCodingModel::ADDITIVE; break;
        case gcta_cli::GenoModel::Dominance:   genoCodingModel = GenoCodingModel::DOMINANCE; break;
        case gcta_cli::GenoModel::NonAdditive: genoCodingModel = GenoCodingModel::NONADDITIVE; break;
    }
}

void Geno::setGenoItemSize(uint32_t &genoSize, uint32_t &missSize){
    genoSize = keepSampleCT;
    missSize = missPtrSize;
}

void Geno::getGenoDouble(uintptr_t *buf, int bufIndex, GenoBufItem* gbuf){
    (this->*getGenoDoubleFunc)(buf, bufIndex, gbuf);
}

// ---------------------------------------------------------------------------
// Float-native genotype decode
// ---------------------------------------------------------------------------

// buildCodingSpecF32: identical logic to buildCodingSpec but stores float.
// All arithmetic is done in double so the rounding matches exactly.
Geno::GenoCodingSpecF32 Geno::buildCodingSpecF32(double mu, double sd) const {
    const GenoCodingSpec spec = buildCodingSpec(mu, sd);
    GenoCodingSpecF32 f;
    f.a0 = static_cast<float>(spec.a0);
    f.a1 = static_cast<float>(spec.a1);
    f.a2 = static_cast<float>(spec.a2);
    f.na = static_cast<float>(spec.na);
    return f;
}

// getGenoFloat dispatch (analogous to getGenoDouble)
void Geno::getGenoFloat(uintptr_t *buf, int bufIndex, float *dest,
                        float &af_out, float &additive_af_out,
                        bool &valid_out, uint32_t extractedMarkerIndex) {
    (this->*getGenoFloatFunc)(buf, bufIndex, dest, af_out, additive_af_out,
                              valid_out, extractedMarkerIndex);
}

// ---------------------------------------------------------------------------
// Native BED helpers (bed_simd): exact replicas of PgenReader's
// CountHardFreqMissExt / CountHardFreqMissExtX arithmetic, driven by the raw
// PLINK1 2-bit counts (genocounts[0..3] == Counts{n0, n1, n2, nmiss}).
// ---------------------------------------------------------------------------
namespace {
inline void bed_fill_snpinfo(const bed_simd::Counts &c, uint32_t keepSize, bool f_std, SNPInfo *snpinfo) {
    const std::array<uint32_t, 4> genocounts = {c.n0, c.n1, c.n2, c.nmiss};
    snpinfo->N = keepSize - genocounts[3];
    snpinfo->AlCount = 2 * snpinfo->N;
    uint32_t dGenoCounts2 = 2 * genocounts[2];
    uint32_t dosage = genocounts[1] + dGenoCounts2;
    snpinfo->af = 1.0 * dosage / snpinfo->AlCount;
    snpinfo->mean = 2.0 * snpinfo->af;
    if (f_std) snpinfo->std = (dosage + dGenoCounts2 - snpinfo->mean * dosage) / (snpinfo->N - 1);
    snpinfo->nMissRate = 1.0 * snpinfo->N / keepSize;
}

inline bool bed_fill_snpinfo_x(const bed_simd::Counts &c, const bed_simd::Counts &c2, uint32_t keepSize,
                               bool dosageComp, bool f_std, SNPInfo *snpinfo) {
    const std::array<uint32_t, 4> genocounts = {c.n0, c.n1, c.n2, c.nmiss};
    const std::array<uint32_t, 4> genocounts2 = {c2.n0, c2.n1, c2.n2, c2.nmiss};
    bool retVal = true;
    if (genocounts2[1] != 0) {
        retVal = false;
    }
    snpinfo->N = keepSize - genocounts[3];
    snpinfo->AlCount = 2 * snpinfo->N - genocounts2[0] - genocounts2[1] - genocounts2[2];
    snpinfo->nMissRate = 1.0 * snpinfo->N / keepSize;
    int dGenoCount1Homo = 2 * genocounts[2];
    double dosage = genocounts[1] + dGenoCount1Homo;
    double dosage_half = dosage - genocounts2[2] - 0.5 * genocounts2[1];

    snpinfo->af = dosage_half / snpinfo->AlCount;
    if (dosageComp) {
        snpinfo->mean = dosage / snpinfo->N;
        if (f_std) snpinfo->std = (dosage + dGenoCount1Homo - dosage * snpinfo->mean) / (snpinfo->N - 1);
    } else {
        snpinfo->mean = dosage_half / snpinfo->N;
        if (f_std) snpinfo->std = (dosage + dGenoCount1Homo - 0.75 * genocounts2[1] - 3 * genocounts2[2] - dosage_half * snpinfo->mean) / (snpinfo->N - 1);
    }
    return retVal;
}

// Prepare gbuf->missing for a BED decode: returns the bitmask pointer the
// kernel should (lazily) fill, or nullptr when there is nothing to write.
inline uintptr_t *bed_prepare_missing(GenoBufItem *gbuf, bool bMakeMiss, uint32_t nmiss,
                                      int missPtrSize, uint32_t keepSampleCT) {
    if (!bMakeMiss) return nullptr;
    if (nmiss == 0) {
        if (!gbuf->lazy_missing) gbuf->missing.assign(static_cast<std::size_t>(missPtrSize), 0);
        return nullptr;
    }
    gbuf->missing.resize(static_cast<std::size_t>(missPtrSize));
    // Kernel writes the first ceil(keep/64) words; clear any vector padding words.
    const std::size_t used = (static_cast<std::size_t>(keepSampleCT) + 63) / 64;
    for (std::size_t w = used; w < gbuf->missing.size(); ++w) gbuf->missing[w] = 0;
    return gbuf->missing.data();
}
}  // namespace

void Geno::bedSnpInfoFromCounts(const uint8_t *raw, const bed_simd::Counts &c, uint8_t sexChromType,
                                SNPInfo *snpinfo) const {
    if (sexChromType != 1) {
        bed_fill_snpinfo(c, keepSampleCT, f_std, snpinfo);
        return;
    }
    // Mirrors CountHardFreqMissExtX's choice of heterogametic counts.
    const uint64_t *het = reinterpret_cast<const uint64_t *>(heterogameticMaskPtr);
    bed_simd::Counts c2;
    if (rawSampleCT == keepSampleCT) {
        c2 = (keepSampleCT == keepHeterogameticSampleCT)
                 ? c
                 : bed_simd::count(raw, het, rawSampleCT, keepHeterogameticSampleCT);
    } else if (keepHeterogameticSampleCT != 0) {
        c2 = bed_simd::count(raw, het, rawSampleCT, keepHeterogameticSampleCT);
    }
    bed_fill_snpinfo_x(c, c2, keepSampleCT, iDC == 1, f_std, snpinfo);
}

bed_simd::Counts Geno::bedSnpInfo(const uint8_t *raw, uint8_t sexChromType, SNPInfo *snpinfo) const {
    const bed_simd::Counts c = bed_simd::count(raw, bedKeepMask(), rawSampleCT, keepSampleCT);
    bedSnpInfoFromCounts(raw, c, sexChromType, snpinfo);
    return c;
}

// BED path: counts via VPOPCNTDQ/VPTERNLOGQ (or AVX2/NEON/scalar), then a
// register-table decode (VPERMPS on Zen 4) -- no 4 KiB per-SNP lookup table.
void Geno::getGenoFloat_bed(uintptr_t *buf, int idx, float *dest,
                             float &af_out, float &additive_af_out,
                             bool &valid_out, uint32_t extractedMarkerIndex) {
    valid_out = false;

    SNPInfo snpinfo;
    const uint8_t *raw = bedRecord(buf, idx);
    uint8_t sexChromType = markerSexChromTypes[curBufferIndex];
    bedSnpInfo(raw, sexChromType, &snpinfo);

    double af = snpinfo.af;
    if (bHasPreAF) {
        af = AFA1[extractedMarkerIndex];
        snpinfo.mean = 2 * af;
        snpinfo.std  = 2 * af * (1.0 - af);
    }
    double maf = std::min(af, 1.0 - af);
    if (maf < min_maf || maf > max_maf) return;
    if (snpinfo.nMissRate < dFilterMiss) return;

    double mu;
    if (bGRM) {
        mu = 2.0 * af;
    } else {
        mu = snpinfo.mean;
    }
    double sd = f_std ? snpinfo.std : mu * (1.0 - af);
    if (sd < 1.0e-50) return;

    af_out          = static_cast<float>(af);
    additive_af_out = static_cast<float>(af);
    valid_out       = true;

    const GenoCodingSpecF32 spec = buildCodingSpecF32(mu, sd);
    const float table[4] = {spec.a0, spec.a1, spec.a2, spec.na};
    bed_simd::decode_f32(raw, bedKeepMask(), rawSampleCT, keepSampleCT, table, dest, nullptr);

    if (sexChromType == 1) {
        double weight;
        bool needWeight;
        setHeterogameticWeight(weight, needWeight);
        if (needWeight) {
            const float fw = static_cast<float>(weight);
            if (bGRM) {
                for (int i = 0; i < keepHeterogameticSampleCT; ++i)
                    dest[keepHeterogameticExtractIndex[i]] *= fw;
            } else {
                // Correct centering under heterogametic weight:
                //   recoded' = recoded * w + (w-1) * rdev * centerValue
                const GenoCodingSpec dspec = buildCodingSpec(mu, sd);
                const float correctWeight = static_cast<float>((weight - 1.0) * dspec.centerValue * dspec.rdev);
                for (int i = 0; i < keepHeterogameticSampleCT; ++i) {
                    uint32_t ci = keepHeterogameticExtractIndex[i];
                    dest[ci] *= fw;
                    dest[ci] += correctWeight;
                }
            }
        }
    }
}

// Single fused pass: raw .bed -> int8 {0,1,2,0} + exact counts (+ optional
// weighted sums / sparse missing indices); the missing bitmask is written
// only when the marker actually has missing calls.
void Geno::getGenoInt8_bed(uintptr_t *buf, int idx, int8_t *dest,
                           float &af_out, float &additive_af_out,
                           double &mu_out, uint32_t &nValidN_out,
                           uintptr_t *miss_out, bool &valid_out,
                           uint32_t extractedMarkerIndex, BedI8Fused *fused) {
    valid_out   = false;
    mu_out      = 0.0;
    nValidN_out = 0;

    SNPInfo snpinfo;
    const uint8_t *raw = bedRecord(buf, idx);
    uint8_t sexChromType = markerSexChromTypes[curBufferIndex];

    bed_simd::I8Opts opts;
    opts.miss_bits = reinterpret_cast<uint64_t *>(miss_out);
    if (fused) {
        opts.wf = fused->vw;
        opts.wd = fused->w2;
    }
    const bed_simd::Counts c = bed_simd::decode_i8(raw, bedKeepMask(), rawSampleCT, keepSampleCT, dest, &opts);
    bedSnpInfoFromCounts(raw, c, sexChromType, &snpinfo);

    if (fused) {
        fused->c0 = c.n0;
        fused->c1 = c.n1;
        fused->c2 = c.n2;
        fused->cmiss = c.nmiss;
        fused->dot_vw = opts.wf_dot;
        fused->miss_vw = opts.wf_miss;
        fused->dot_w2 = opts.wd_dot;
        fused->dot2_w2 = opts.wd_dot2;
        fused->miss_w2 = opts.wd_miss;
        if (fused->miss_list) {
            fused->miss_list->clear();
            if (c.nmiss && miss_out) {
                fused->miss_list->reserve(c.nmiss);
                const int num_words = static_cast<int>((keepSampleCT + 63) / 64);
                for (int w = 0; w < num_words; ++w) {
                    uintptr_t word = miss_out[w];
                    while (word != 0) {
                        fused->miss_list->push_back(w * 64 + std::countr_zero(word));
                        word &= (word - 1);
                    }
                }
            }
        }
    }

    double af = snpinfo.af;
    if (bHasPreAF) {
        af = AFA1[extractedMarkerIndex];
        snpinfo.mean = 2 * af;
        snpinfo.std  = 2 * af * (1.0 - af);
    }
    double maf = std::min(af, 1.0 - af);
    if (maf < min_maf || maf > max_maf) return;
    if (snpinfo.nMissRate < dFilterMiss) return;

    const double mu = bGRM ? (2.0 * af) : snpinfo.mean;
    const double sd = f_std ? snpinfo.std : mu * (1.0 - af);
    if (sd < 1.0e-50) return;

    af_out          = static_cast<float>(af);
    additive_af_out = static_cast<float>(af);
    mu_out          = mu;
    nValidN_out     = snpinfo.N;
    valid_out       = true;
}

void Geno::decodeBedI8KMajor(uintptr_t *buf, const int *markerIdx, int K, int n_rows,
                             int8_t *kmajor, int ldK, const double *mu, double *r_raw) {
    if (n_rows <= 0) return;
    if (K <= 0) {
        std::fill(r_raw, r_raw + n_rows, 0.0);
        return;
    }
    constexpr int kBlk = 64;
    const uint64_t *keep = bedKeepMask();
    const int nTiles = (n_rows + kBlk - 1) / kBlk;
    std::vector<uint32_t> tileRaw(static_cast<std::size_t>(nTiles));
    for (int t = 0; t < nTiles; ++t) {
        tileRaw[t] = keep ? sampleKeepIndex[static_cast<std::size_t>(t) * kBlk]
                          : static_cast<uint32_t>(t * kBlk);
    }
    const uint32_t rawCT = rawSampleCT;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nTiles; ++t) {
        const int i0 = t * kBlk;
        const int s_len = std::min(kBlk, n_rows - i0);
        alignas(64) int8_t col[kBlk][kBlk];   // [marker][sample]
        alignas(64) int8_t blk[kBlk][kBlk];   // [sample][marker]
        double r_acc[kBlk] = {0.0};
        for (int m0 = 0; m0 < K; m0 += kBlk) {
            const int m_len = std::min(kBlk, K - m0);
            for (int dm = 0; dm < m_len; ++dm) {
                bed_simd::decode_i8_range(bedRecord(buf, markerIdx[m0 + dm]), keep, rawCT,
                                          tileRaw[t], static_cast<uint32_t>(s_len), col[dm]);
            }
            for (int dm = 0; dm < m_len; ++dm) {
                const double mu_m = mu[m0 + dm];
                for (int ds = 0; ds < s_len; ++ds) {
                    const int8_t g = col[dm][ds];
                    r_acc[ds] += static_cast<double>(g) * mu_m;
                    blk[ds][dm] = g;
                }
            }
            for (int ds = 0; ds < s_len; ++ds) {
                std::memcpy(kmajor + static_cast<std::size_t>(i0 + ds) * ldK + m0, blk[ds],
                            static_cast<std::size_t>(m_len));
            }
        }
        for (int ds = 0; ds < s_len; ++ds) {
            r_raw[i0 + ds] = r_acc[ds];
        }
    }
}

// PGEN path: decode directly into dest via geno_f32_ptr (no temporary double vector).
void Geno::getGenoFloat_pgen(uintptr_t *buf, int idx, float *dest,
                              float &af_out, float &additive_af_out,
                              bool &valid_out, uint32_t extractedMarkerIndex) {
    valid_out = false;
    GenoBufItem tmp;
    tmp.extractedMarkerIndex = extractedMarkerIndex;
    tmp.geno_f32_ptr = dest;
    getGenoDouble_pgen(buf, idx, &tmp);
    if (!tmp.valid) return;
    valid_out       = true;
    af_out          = static_cast<float>(tmp.af);
    additive_af_out = static_cast<float>(tmp.additive_af);
}

// BGEN path: decode directly into dest via geno_f32_ptr (no temporary double vector).
void Geno::getGenoFloat_bgen(uintptr_t *buf, int idx, float *dest,
                              float &af_out, float &additive_af_out,
                              bool &valid_out, uint32_t extractedMarkerIndex) {
    valid_out = false;
    GenoBufItem tmp;
    tmp.extractedMarkerIndex = extractedMarkerIndex;
    tmp.geno_f32_ptr = dest;
    getGenoDouble_bgen(buf, idx, &tmp);
    if (!tmp.valid) return;
    valid_out       = true;
    af_out          = static_cast<float>(tmp.af);
    additive_af_out = static_cast<float>(tmp.additive_af);
}


void Geno::setHeterogameticWeight(double &weight, bool &needWeight){
    weight = 1.0;
    if(bGRM){
        weight = sqrt(0.5);
        if(iGRMdc == 1){
            weight *= sqrt(2.0);
        }else if(iGRMdc == 0){
            weight *= sqrt(0.5);
        }
    }else{
        if(iDC == 0){
            weight = 0.5;
        }
    }
    if(std::abs(weight - 1.0) > 1e-6){
        needWeight = true;
    }else{
        needWeight = false;
    }
}

Geno::GenoCodingModel Geno::getCodingModel() const {
    return genoCodingModel;
}

double Geno::mapDosageToModel(double dosage, double mu, GenoCodingModel model) const {
    if (model == GenoCodingModel::ADDITIVE) {
        return dosage;
    }

    if (model == GenoCodingModel::DOMINANCE || model == GenoCodingModel::NONADDITIVE) {
        if (dosage < 0.5) {
            return 0.0;
        }
        if (dosage < 1.5) {
            return mu;
        }
        return (2.0 * mu - 2.0);
    }

    return dosage;
}

Geno::GenoCodingSpec Geno::buildCodingSpec(double mu, double sd) const {
    GenoCodingSpec spec;
    const GenoCodingModel model = getCodingModel();
    if (model == GenoCodingModel::ADDITIVE) {
        spec.centerValue = bGenoCenter ? mu : 0.0;
        spec.rdev = bGenoStd ? sqrt(1.0 / sd) : 1.0;

        const double raw0 = 0.0;
        const double raw1 = 1.0;
        const double raw2 = 2.0;
        const double rawNa = mu;

        spec.a0 = (raw0 - spec.centerValue) * spec.rdev;
        spec.a1 = (raw1 - spec.centerValue) * spec.rdev;
        spec.a2 = (raw2 - spec.centerValue) * spec.rdev;
        spec.na = (rawNa - spec.centerValue) * spec.rdev;
        return spec;
    }

    const double psq = 0.5 * mu * mu;
    spec.centerValue = bGenoCenter ? psq : 0.0;
    spec.rdev = bGenoStd ? (1.0 / sd) : 1.0;

    const double raw0 = 0.0;
    const double raw1 = mu;
    const double raw2 = (2.0 * mu - 2.0);
    const double rawNa = psq;

    spec.a0 = (raw0 - spec.centerValue) * spec.rdev;
    spec.a1 = (raw1 - spec.centerValue) * spec.rdev;
    spec.a2 = (raw2 - spec.centerValue) * spec.rdev;
    spec.na = (rawNa - spec.centerValue) * spec.rdev;
    return spec;
}

void Geno::getGenoDouble_pgen(uintptr_t *buf, int idx, GenoBufItem* gbuf){
    SNPInfo snpinfo;
    uintptr_t *cur_buf = buf + static_cast<std::size_t>(idx) * static_cast<std::size_t>(pgenGenoBuf1PtrSize);
    uintptr_t *dosage_present = cur_buf + pgenGenoPtrSize;
    uint16_t *dosage_main = reinterpret_cast<uint16_t*>(cur_buf + pgenGenoPtrSize + pgenDosagePresentPtrSize);
    uint32_t dosage_ct = static_cast<uint32_t>(cur_buf[pgenGenoBuf1PtrSize - 1]);
    uint8_t sexChromType = markerSexChromTypes[curBufferIndex];

    gbuf->valid = false;
    if (dosage_ct > keepSampleCT) {
        LOGGER.e(0, "invalid pgen dosage count " + to_string(dosage_ct)
                    + " (expected <= " + to_string(keepSampleCT) + ").");
    }

    const vector<uint32_t> *curHeterogameticIndex = NULL;
    if(sexChromType == 1){
        curHeterogameticIndex = &keepHeterogameticExtractIndex;
    }

    string err;
    if(!PgenReader::CountHardDosage(cur_buf, dosage_main, dosage_present, curHeterogameticIndex, keepSampleCT, dosage_ct, &snpinfo, err)){
        LOGGER.e(0, err);
    }

    double af = snpinfo.af;
    double std = snpinfo.std;
    if(bHasPreAF){
        af = AFA1[gbuf->extractedMarkerIndex];
        std = 2.0 * af * (1.0 - af);
    }
    double maf = std::min(af, 1.0 - af);
    if(maf >= min_maf && maf <= max_maf && snpinfo.nMissRate >= dFilterMiss){
        gbuf->valid = true;
        gbuf->af = af;
        gbuf->additive_af = af;
        gbuf->nValidN = snpinfo.N;
        gbuf->nValidAllele = snpinfo.AlCount;
        gbuf->mean = 2.0 * af;
        gbuf->sd = std;

        if(bMakeGeno){
            const double mu = gbuf->mean;
            if(std < 1.0e-50){
                gbuf->valid = false;
                return;
            }
            const GenoCodingModel model = getCodingModel();
            const GenoCodingSpec spec = buildCodingSpec(mu, std);

            // PGEN dosage_main uses 14-bit fractional scaling of allele
            // dosage in [0, 2].
            static constexpr double kDosageScale = 1.0 / 16384.0;
            if (gbuf->geno_f32_ptr) {
                float *out_f32 = gbuf->geno_f32_ptr;
                for(uint32_t j = 0; j < keepSampleCT; ++j){
                    const double dos = static_cast<double>(dosage_main[j]) * kDosageScale;
                    const double recoded = mapDosageToModel(dos, mu, model);
                    out_f32[j] = static_cast<float>((recoded - spec.centerValue) * spec.rdev);
                }
                if(sexChromType == 1){
                    double weight;
                    bool needWeight;
                    setHeterogameticWeight(weight, needWeight);
                    if(needWeight){
                        const float fw = static_cast<float>(weight);
                        for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                            out_f32[keepHeterogameticExtractIndex[i]] *= fw;
                        }
                    }
                }
            } else {
                double *out_geno = gbuf->geno_ptr;
                if (!out_geno) {
                    gbuf->geno.resize(keepSampleCT);
                    out_geno = gbuf->geno.data();
                }
                for(uint32_t j = 0; j < keepSampleCT; ++j){
                    const double dos = static_cast<double>(dosage_main[j]) * kDosageScale;
                    const double recoded = mapDosageToModel(dos, mu, model);
                    out_geno[j] = (recoded - spec.centerValue) * spec.rdev;
                }

                if(sexChromType == 1){
                    double weight;
                    bool needWeight;
                    setHeterogameticWeight(weight, needWeight);
                    if(needWeight){
                        for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                            out_geno[keepHeterogameticExtractIndex[i]] *= weight;
                        }
                    }
                }
            }
        }
        if(bMakeMiss){
            gbuf->missing.resize(missPtrSize, 0);
        }
    }
}

void Geno::getGenoDouble_bed(uintptr_t *buf, int idx, GenoBufItem* gbuf){
    SNPInfo snpinfo;
    const uint8_t *raw = bedRecord(buf, idx);
    uint8_t sexChromType = markerSexChromTypes[curBufferIndex];
    const bed_simd::Counts counts = bedSnpInfo(raw, sexChromType, &snpinfo);
    uint32_t curExtractIndex = gbuf->extractedMarkerIndex;
    double af = snpinfo.af;
    if(bHasPreAF){
        af = AFA1[curExtractIndex];
        snpinfo.mean = 2 * af;
        snpinfo.std = 2 * af * (1.0 - af); 
    }
    double maf = std::min(af, 1.0 - af);
    if(maf >= min_maf && maf <= max_maf){
        if(snpinfo.nMissRate >= dFilterMiss){
            gbuf->valid = true;
            gbuf->af = af;
            gbuf->additive_af = af;
            gbuf->nValidN = snpinfo.N;
            gbuf->nValidAllele = snpinfo.AlCount;

            if(bGRM){
                gbuf->mean = 2.0 * af;
            }else{
                gbuf->mean = snpinfo.mean;
            }
            if(f_std){
                gbuf->sd = snpinfo.std;
            }else{
                gbuf->sd = gbuf->mean * (1.0 - af);
            }
            if(bMakeGeno){
                double mu = gbuf->mean;
                double sd = gbuf->sd;
                if(sd < 1.0e-50){
                    gbuf->valid = false;
                    return;
                }
                if (gbuf->stats_only) {
                    return;
                }
                const GenoCodingSpec spec = buildCodingSpec(mu, sd);
                const uint64_t *keep = bedKeepMask();
                uintptr_t *pmiss = bed_prepare_missing(gbuf, bMakeMiss, counts.nmiss, missPtrSize, keepSampleCT);
                uint64_t *pmiss64 = reinterpret_cast<uint64_t *>(pmiss);
                if (gbuf->geno_i8_ptr) {
                    bed_simd::I8Opts opts;
                    opts.miss_bits = pmiss64;
                    bed_simd::decode_i8(raw, keep, rawSampleCT, keepSampleCT, gbuf->geno_i8_ptr, &opts);
                } else if (gbuf->geno_f32_ptr) {
                    float *out_f32 = gbuf->geno_f32_ptr;
                    const GenoCodingSpecF32 spec_f32 = buildCodingSpecF32(mu, sd);
                    const float table[4] = {spec_f32.a0, spec_f32.a1, spec_f32.a2, spec_f32.na};
                    bed_simd::decode_f32(raw, keep, rawSampleCT, keepSampleCT, table, out_f32, pmiss64);
                    if(sexChromType == 1){
                        double weight;
                        bool needWeight;
                        setHeterogameticWeight(weight, needWeight);
                        if(needWeight){
                            const float fw = static_cast<float>(weight);
                            if(bGRM){
                                for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                    out_f32[keepHeterogameticExtractIndex[i]] *= fw;
                                }
                            }else{
                                const float correctWeight = static_cast<float>((weight - 1) * spec.rdev * spec.centerValue);
                                for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                    uint32_t curIndex = keepHeterogameticExtractIndex[i];
                                    out_f32[curIndex] *= fw;
                                    out_f32[curIndex] += correctWeight;
                                }
                            }
                        }
                    }
                } else {
                    const double table[4] = {spec.a0, spec.a1, spec.a2, spec.na};
                    double *out_geno = gbuf->geno_ptr;
                    if (!out_geno) {
                        gbuf->geno.resize(keepSampleCT);
                        out_geno = gbuf->geno.data();
                    }
                    bed_simd::decode_f64(raw, keep, rawSampleCT, keepSampleCT, table, out_geno, pmiss64);
                    if(sexChromType == 1){
                        double weight;
                        bool needWeight;
                        setHeterogameticWeight(weight, needWeight);
                        if(needWeight){
                            if(bGRM){
                                for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                    out_geno[keepHeterogameticExtractIndex[i]] *= weight;
                                }
                            }else{
                                double correctWeight = (weight - 1) * spec.rdev * spec.centerValue;
                                for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                    uint32_t curIndex = keepHeterogameticExtractIndex[i];
                                    out_geno[curIndex] *= weight;
                                    out_geno[curIndex] += correctWeight;
                                }
                            }
                        }
                    }
                }
            }
            return;
        }
    }
    gbuf->valid = false;
}

void calDosage_bgen(uint32_t prob1, uint32_t prob2, uint64_t &dosage, uint32_t &prob1d){
    prob1d = prob1 * 2;
    dosage = prob1d + prob2;
}

void calDosagePhase_bgen(uint32_t prob1, uint32_t prob2, uint64_t &dosage, uint32_t &prob1d){
    dosage = prob1 + prob2;
    prob1d = 2 * prob1 * prob2;
}

void Geno::getGenoDouble_bgen(uintptr_t *buf, int idx, GenoBufItem* gbuf){
    SNPInfo snpinfo;
    uintptr_t *cur_buf = buf + static_cast<std::size_t>(idx) * static_cast<std::size_t>(bgenRawGenoBuf1PtrSize);
    uint8_t *curbuf = (uint8_t*)cur_buf;
    int fileIndex = fileIndexBuf[curBufferIndex];

    int compressFormat = compressFormats[fileIndex];

    uint16_t L16;
    uint32_t L32;
    memcpy(&L16, curbuf, sizeof(L16));
    curbuf += sizeof(L16) + L16;
    memcpy(&L16, curbuf, sizeof(L16));
    curbuf += sizeof(L16) + L16;
    memcpy(&L16, curbuf, sizeof(L16));
    curbuf += sizeof(L16) + L16;
    curbuf += sizeof(uint32_t);
    memcpy(&L16, curbuf, sizeof(L16));
    curbuf += sizeof(L16);
    for(int i = 0; i < L16; i++){
        memcpy(&L32, curbuf, sizeof(L32));
        curbuf += sizeof(L32) + L32;
    }

    uint32_t len_comp, len_decomp;
    memcpy(&len_comp, curbuf, sizeof(len_comp));
    curbuf += sizeof(len_comp);

    if(compressFormat == 0){
        len_decomp = len_comp;
    }else{
        len_comp -= 4;
        memcpy(&len_decomp, curbuf, sizeof(len_decomp));
        curbuf += sizeof(len_decomp);
    }

    string error_promp = to_string(gbuf->extractedMarkerIndex) + "th SNP of [" + geno_files[fileIndex] + "]."; 
    uint8_t *dec_data;
    if(compressFormat != 0){
        dec_data = new uint8_t[len_decomp + 8];
        uint32_t curCompSize = len_comp;
        if(compressFormat == 1){
            uint32_t Ldecomp = len_decomp;
            int z_result = uncompress((Bytef*)dec_data, (uLongf*)&Ldecomp, (Bytef*)curbuf, curCompSize);
            if(z_result != Z_OK || len_decomp != Ldecomp){
                LOGGER.e(0, "decompressing genotype data error in " + error_promp); 
            }
        }else if(compressFormat == 2){
            uint64_t const rSize = ZSTD_getFrameContentSize((void*)curbuf, curCompSize);
            switch(rSize){
                case ZSTD_CONTENTSIZE_ERROR:
                    LOGGER.e(0, "not compressed by zstd in " + error_promp);
                    break;
                case ZSTD_CONTENTSIZE_UNKNOWN:
                    LOGGER.e(0, "original size unknown in " + error_promp);
                    break;
            }
            if(rSize != len_decomp){
                LOGGER.e(0, "size stated in the compressed file is different from " + error_promp);
            }
            size_t const dSize = ZSTD_decompress((void *)dec_data, len_decomp, (void*)curbuf, curCompSize); 
            if(ZSTD_isError(dSize)){
                LOGGER.e(0, "decompressing genotype error: " + string(ZSTD_getErrorName(dSize)) + " in " + error_promp);
            }
        }else{
            LOGGER.e(0, "unknown compress format in " + error_promp);
        }
    }else{
        dec_data = curbuf;
    }

    uint32_t n_sample = *(uint32_t *)dec_data;
    if(n_sample != rawCountSamples[fileIndex]){
        LOGGER.e(0, "inconsistent number of individuals in " + error_promp);
    }
    uint16_t num_alleles = *(uint16_t *)(dec_data + 4);
    if(num_alleles != 2){
        LOGGER.e(0, "multi-allelic SNPs detected in " + error_promp);
    }

    uint8_t min_ploidy = *(uint8_t *)(dec_data + 6);
    uint8_t max_ploidy = *(uint8_t *)(dec_data + 7);
    uint8_t * sample_ploidy = (uint8_t *)(dec_data + 8);
    if(min_ploidy != 2){
        LOGGER.e(0, "multiploidy detected in " + error_promp);
    }

    uint8_t *geno_prob = sample_ploidy + n_sample;
    uint8_t is_phased = *(geno_prob);
    uint8_t bits_prob = *(geno_prob+1);
    uint8_t* X_prob = geno_prob + 2;
    uint32_t len_prob = len_decomp - n_sample - 10;
    void (*calFunc)(uint32_t, uint32_t, uint64_t&, uint32_t &prob1d);
    if(is_phased){
        calFunc = &calDosagePhase_bgen;
    }else{
        calFunc = &calDosage_bgen;
    }

    uint8_t double_bits_prob = bits_prob * 2;
    vector<uint32_t> miss_index;

    uint8_t sexChromType = markerSexChromTypes[curBufferIndex];

    uint64_t mask = (1U << bits_prob) - 1;
    uint64_t dosage_sum = 0, fij_sum = 0, dosage2_sum = 0;
    uint32_t validN = 0;
    uint32_t validAllele = 0;
    vector<uint32_t> dosages(keepSampleCT);
    uint32_t max_dos = mask * 2 + 1;
    bool has_miss = false;

    uint32_t curSampleCT = keepSampleCT;
    vector<uint32_t> *curSampleIndexPtr = &sampleKeepIndex;

    for(int j = 0; j < curSampleCT; j++){
        uint32_t sindex = (*curSampleIndexPtr)[j];
        uint8_t item_ploidy = sample_ploidy[sindex];
        if(item_ploidy > 128){
            miss_index.push_back(sindex);
            has_miss = true;
            dosages[j] = max_dos;
        }else if(item_ploidy == 2){
            uint32_t start_bits = sindex * double_bits_prob;
            uint64_t geno_temp;
            memcpy(&geno_temp, &(X_prob[start_bits/CHAR_BIT]), sizeof(geno_temp));
            geno_temp = geno_temp >> (start_bits % CHAR_BIT);
            uint32_t prob1 = geno_temp & mask;
            uint32_t prob2 = (geno_temp >> bits_prob) & mask;
            uint32_t prob1d;
            uint64_t dosage;
            calFunc(prob1, prob2, dosage, prob1d);
            dosages[j] = dosage;
            dosage_sum += dosage;
            dosage2_sum += dosage * dosage;
            fij_sum += prob1d;
            validN++;
            validAllele += 2;
        }else{
            LOGGER.e(0, "multiploidy detected in " + error_promp);
        }
    }

    double dosage_sum_half = dosage_sum;
    double dosage2_sum_half = dosage2_sum;
    if(sexChromType == 1){
        for(int j = 0; j < keepHeterogameticSampleCT; j++){
            uint32_t sindex = keepHeterogameticIndex[j];
            uint8_t item_ploidy = sample_ploidy[sindex];
            if(item_ploidy == 2){
                uint32_t start_bits = sindex * double_bits_prob;
                uint64_t geno_temp;
                memcpy(&geno_temp, &(X_prob[start_bits/CHAR_BIT]), sizeof(geno_temp));
                geno_temp = geno_temp >> (start_bits % CHAR_BIT);
                uint32_t prob1 = geno_temp & mask;
                uint32_t prob2 = (geno_temp >> bits_prob) & mask;
                uint32_t prob1d;
                uint64_t dosage;
                calFunc(prob1, prob2, dosage, prob1d);
                uint32_t prob1_true = prob1d / 2;
                dosage_sum_half -= prob1_true;
                dosage2_sum_half -= (dosage * dosage - (uint64_t)prob1_true * prob1_true);
                validAllele--;
            }
        }
    }

    if(compressFormat != 0){
        delete[] dec_data;
    }

    double maskd = (double)mask;
    double af = (double)dosage_sum_half / maskd / validAllele;
    double mean;
    double std = 2.0 * af * (1.0 - af);
    double info = 0.0;
    double mask2 = mask * mask;
    if(std < 1e-50){
        info = 1.0;
    }else{
        double dos2_fij_sum = (double)(dosage_sum + fij_sum)/ maskd - (double)dosage2_sum / mask2; 
        info = 1.0 - dos2_fij_sum / (std * validN);
    }
    if(is_phased){
        info = 1;
    }

    if(bHasPreAF){
        af = AFA1[gbuf->extractedMarkerIndex];
        mean = 2.0 * af;
        std = 2.0 * af * (1.0 - af);
    }else{
        if(iDC == 1 && (!bGRM)){
            double dos_double = (double)dosage_sum / maskd;
            mean = dos_double / validN;
            std = ((double)dosage2_sum / mask2 - dos_double * mean)/(validN - 1);
        }else{
            double dos_double = (double)dosage_sum_half / maskd;
            mean = dos_double / validN;
            std = ((double)dosage2_sum_half / mask2 - dos_double * mean)/(validN - 1);
        }
    }

    double maf = std::min(af, 1.0 - af);
    if(maf >= min_maf && maf <= max_maf){
        double nMissRate = 1.0*validN / curSampleCT;
        if(nMissRate >= dFilterMiss && info >= dFilterInfo){
            gbuf->valid = true;
            gbuf->af = af;
            gbuf->additive_af = af;
            gbuf->nValidN = validN;
            gbuf->nValidAllele = validAllele;
            gbuf->info = is_phased ? (std::numeric_limits<double>::quiet_NaN()) : info;
            gbuf->mean = mean;
            gbuf->sd = std;
            if(bMakeGeno){
                double mu = gbuf->mean;
                if(std < 1.0e-50){
                    gbuf->valid = false;
                    return;
                }

                double* dos_lookup = new double[max_dos + 2];
                const GenoCodingModel model = getCodingModel();
                const GenoCodingSpec spec = buildCodingSpec(mu, std);
                for(uint32_t i = 0; i < max_dos; i++){
                    const double tdos = static_cast<double>(i) / mask;
                    const double recoded = mapDosageToModel(tdos, mu, model);
                    dos_lookup[i] = (recoded - spec.centerValue) * spec.rdev;
                }
                dos_lookup[max_dos] = spec.na;

                if (gbuf->geno_f32_ptr) {
                    float *out_f32 = gbuf->geno_f32_ptr;
                    for(int j = 0; j < curSampleCT; j++){
                        out_f32[j] = static_cast<float>(dos_lookup[dosages[j]]);
                    }
                    delete[] dos_lookup;
                    if(sexChromType == 1){
                        double weight;
                        bool needWeight;
                        setHeterogameticWeight(weight, needWeight);
                        if(needWeight){
                            const float fw = static_cast<float>(weight);
                            if(bGRM){
                                for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                    out_f32[keepHeterogameticExtractIndex[i]] *= fw;
                                }
                            }else{
                                const float correctWeight = static_cast<float>((weight - 1) * spec.rdev * spec.centerValue);
                                for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                    uint32_t curIndex = keepHeterogameticExtractIndex[i];
                                    out_f32[curIndex] *= fw;
                                    out_f32[curIndex] += correctWeight;
                                }
                            }
                        }
                    }
                } else {
                    double *out_geno = gbuf->geno_ptr;
                    if (!out_geno) {
                        gbuf->geno.resize(curSampleCT);
                        out_geno = gbuf->geno.data();
                    }
                    for(int j = 0; j < curSampleCT; j++){
                        out_geno[j] = dos_lookup[dosages[j]];
                    }
                    delete[] dos_lookup;
                    if(sexChromType == 1){
                        double weight;
                        bool needWeight;
                        setHeterogameticWeight(weight, needWeight);
                        if(needWeight){
                            if(bGRM){
                                for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                    out_geno[keepHeterogameticExtractIndex[i]] *= weight;
                                }
                            }else{
                                double correctWeight = (weight - 1) * spec.rdev * spec.centerValue;
                                for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                    uint32_t curIndex = keepHeterogameticExtractIndex[i];
                                    out_geno[curIndex] *= weight;
                                    out_geno[curIndex] += correctWeight;
                                }
                            }
                        }
                    }
                }
            }
            if(bMakeMiss){
                gbuf->missing.resize(missPtrSize, 0); 
                const int ptrsize = sizeof(uintptr_t) * CHAR_BIT;
                for(int j = 0; j < (int)miss_index.size(); j++){
                    int cur_index = miss_index[j];
                    gbuf->missing[cur_index / ptrsize] |= (1UL << (cur_index % ptrsize));
                }
            }
            return;
        }
    }
    gbuf->valid = false;
}

void Geno::loopDouble(const vector<uint32_t> &extractIndex, int numMarkerBuf,
                      bool bMakeGeno, bool bGenoCenter, bool bGenoStd, bool bMakeMiss,
                      vector<function<void(uintptr_t *buf, std::span<const uint32_t> exIndex)>> callbacks,
                      bool showLog)
{
    // ── Common pre-logic (was in preGenoDouble) ──────────────────────────
    sampleKeepIndex    = pheno->get_index_keep();
    keepSampleCT       = sampleKeepIndex.size();
    rawSampleCT        = pheno->count_raw();
    numMarkerBlock     = numMarkerBuf;
    keepSexIndex       = pheno->getSexValidRawIndex();
    keepHeterogameticIndex      = pheno->getHeterogameticRawIndex();
    keepHeterogameticExtractIndex = pheno->getHeterogameticExtractIndex();
    keepSexSampleCT    = keepSexIndex.size();
    keepHeterogameticSampleCT   = keepHeterogameticIndex.size();
    this->bMakeGeno    = bMakeGeno;
    this->bGenoCenter  = bGenoCenter;
    this->bGenoStd     = bGenoStd;
    this->bMakeMiss    = bMakeMiss;

    // ── Create format-specific backend (runs format pre-logic) ───────────
    std::unique_ptr<GenoBackend> backend;
    if (genoFormat == "BED") {
        backend = makeBedBackend(*this);
    } else if (genoFormat == "PGEN") {
        backend = makePgenBackend(*this);
    } else if (genoFormat == "BGEN") {
        backend = makeBgenBackend(*this);
    } else {
        LOGGER.e(0, "loopDouble: unknown genotype format: " + genoFormat);
    }

    // ── Build baseIndexLookup (depends on rawCountSNPs set by backend) ───
    baseIndexLookup.clear();
    baseIndexLookup.push_back(0);
    int32_t sumIndex = 0;
    for (int i = 0; i < (int)geno_files.size() - 1; ++i) {
        sumIndex += rawCountSNPs[i];
        baseIndexLookup.push_back(sumIndex);
    }
    missPtrSize = PgenReader::GetSubsetMaskSize(keepSampleCT);

    // ── Single-slot block metadata (index 0 is always current block) ─────
    markerSexChromTypes.assign(1, 0);
    fileIndexBuf.assign(1, 0);
    curBufferIndex = 0;

    LOGGER.ts("LOOP_GENO_TOT");
    LOGGER.ts("LOOP_GENO_PRE");
    uint32_t nFinishedMarker = 0;
    const uint32_t nTMarker  = static_cast<uint32_t>(extractIndex.size());
    int pre_block = 0;

    // ── Adapter: GenoBlock → legacy (uintptr_t*, span<const uint32_t>) ────
    auto blockCallback = [&](GenoBlock &block) {
        markerSexChromTypes[0] = block.sexChromType;
        fileIndexBuf[0]    = block.fileIndex;
        curBufferIndex     = 0;

        for (auto &cb : callbacks) {
            cb(block.buf.data(), block.extractIndex);
        }

        nFinishedMarker += block.numMarkers;
        if (showLog) {
            int cur_block = nFinishedMarker >> 14;
            if (cur_block > pre_block) {
                pre_block = cur_block;
                float time_p = LOGGER.tp("LOOP_GENO_PRE");
                if (time_p > 300) {
                    LOGGER.ts("LOOP_GENO_PRE");
                    float elapse_time      = LOGGER.tp("LOOP_GENO_TOT");
                    float finished_percent = static_cast<float>(nFinishedMarker) / nTMarker;
                    float remain_time      = (1.0f / finished_percent - 1.0f) * elapse_time / 60.0f;
                    LOGGER.info(1, "{:.1f}% Estimated time remaining {:.1f} min",
                                finished_percent * 100.0f, remain_time);
                }
            }
        }
    };

    backend->stream(extractIndex, blockCallback);

    if (showLog) {
        LOGGER.info(1, "100% finished in {:.1f} sec", LOGGER.tp("LOOP_GENO_TOT"));
        LOGGER << nFinishedMarker << " SNPs have been processed." << std::endl;
    }
    // backend destructor runs end-logic (masks freed, files closed).
}

union Geno_prob{
    char byte[4];
    uint32_t value = 0;
};


void Geno::bgen2bed(const vector<uint32_t> &raw_marker_index){
    LOGGER << "Old bgen 2 bed" << std::endl;
    LOGGER.ts("LOOP_BGEN_BED");
    LOGGER.ts("LOOP_BGEN_TOT");
    vector<uint32_t>& index_keep = pheno->get_index_keep();

    /*
    std::ofstream out((options["out"] + "_sample_index.txt").c_str());
    for(auto & item : index_keep){
        out << item << std::endl;
    }
    out.close();
    */
    
    auto buf_size = (num_raw_sample + 31) / 32;
    size_t buf_size_byte = buf_size * 8;

    int num_marker = 1;
    int num_markers = raw_marker_index.size();
    LOGGER << "samples: " << num_raw_sample << ", keep_sample: " << index_keep.size() << std::endl;
    LOGGER << "Markers: " << num_markers << std::endl;

    FILE * h_bgen = fopen(geno_files[0].c_str(), "rb");
    /*
    std::ofstream infos("exmaple.txt");
    infos << "index\traw_index\tpos\tLen_comp\tLen_decomp" << std::endl;
    */
    #pragma omp parallel for schedule(static) ordered
    for(uint32_t index = 0; index < num_markers; index++){
        //LOGGER.i(0, to_string(index) + "NUM_thread: " + to_string(omp_get_max_threads()));
        auto raw_index = raw_marker_index[index];
        uint64_t *buf = new uint64_t[buf_size]();
        uint64_t byte_pos, byte_size;
        this->marker->getStartPosSize(raw_index, byte_pos, byte_size);
        uint32_t len_comp, len_decomp;
        char * snp_data;

        #pragma omp ordered
        {
            fseek(h_bgen, byte_pos, SEEK_SET);
            len_comp = read1Byte<uint32_t>(h_bgen) - 4;
            len_decomp = read1Byte<uint32_t>(h_bgen);
            snp_data = new char[len_comp];
            readBytes(h_bgen, len_comp, snp_data);
            //infos << index << "\t" << raw_index << "\t" << byte_pos << "\t" << len_comp << "\t" << len_decomp << std::endl;
        }
        uLongf dec_size = len_decomp;

        char * dec_data =  new char[len_decomp];
        int z_result = uncompress((Bytef*)dec_data, &dec_size, (Bytef*)snp_data, len_comp);
        delete[] snp_data;
        if(z_result == Z_MEM_ERROR || z_result == Z_BUF_ERROR || dec_size != len_decomp){
            LOGGER.e(0, "decompressing genotype data error in " + to_string(raw_index) + "th SNP."); 
        }

        uint32_t n_sample = *(uint32_t *)dec_data;
        if(n_sample != num_raw_sample){
            LOGGER.e(0, "inconsistent number of samples in " + to_string(raw_index) + "th SNP." );
        }
        uint16_t num_alleles = *(uint16_t *)(dec_data + 4);
        if(num_alleles != 2){
            LOGGER.e(0, "multi-allelic SNPs detected likely because the bgen file is malformed.");
        }

        uint8_t min_ploidy = *(uint8_t *)(dec_data + 6);//2
        uint8_t max_ploidy = *(uint8_t *)(dec_data + 7); //2
        uint8_t * sample_ploidy = (uint8_t *)(dec_data + 8);

        uint8_t *geno_prob = sample_ploidy + n_sample;
        uint8_t is_phased = *(geno_prob);
        uint8_t bits_prob = *(geno_prob+1);
        uint8_t* X_prob = geno_prob + 2;
        uint32_t len_prob = len_decomp - n_sample - 10;
        if(is_phased){
            LOGGER.e(0, "GCTA does not support phased data currently.");
        }

        int byte_per_prob = bits_prob / 8;
        int double_byte_per_prob = byte_per_prob * 2;
        if(bits_prob % 8 != 0){
            LOGGER.e(0, "GCTA does not support probability bits other than in byte units.");
        }

        if(len_prob != double_byte_per_prob * n_sample){
            LOGGER.e(0, "malformed data in " + to_string(raw_index) + "th SNP.");
        }
        /*
        infos << index << "_2\t" << raw_index << "\t" << byte_pos << "\t" << len_comp << "\t" << len_prob << std::endl;
        FILE *obgen = fopen((to_string(index) + ".bin").c_str(), "wb");
        fwrite(snp_data, sizeof(char), len_comp, obgen);
        fwrite(X_prob, sizeof(char), len_prob, obgen);
        fclose(obgen);
        */

        uint32_t base_value = (1 << bits_prob) - 1;

        uint8_t *buf_ptr = (uint8_t *)buf;
        if(!opt_dosage_call){
            uint32_t cut_value = ceil(base_value * opt_hard_call_thresh);
            for(uint32_t i = 0; i < num_keep_sample; i++){
                uint32_t item_byte = i >> 2;
                uint32_t move_byte = (i & 3) << 1;

                uint32_t sindex = index_keep[i];
                uint8_t item_ploidy = sample_ploidy[sindex];

                uint8_t geno_value;
                if(item_ploidy > 128){
                    geno_value = 1;
                }else if(item_ploidy == 2){
                    auto base = sindex * double_byte_per_prob;
                    auto base1 = base + byte_per_prob;
                    Geno_prob prob_item;
                    Geno_prob prob_item1;
                    /*
                       memcpy(prob_item.byte, X_prob + base,  byte_per_prob); 
                       memcpy(prob_item1.byte, X_prob + base1, byte_per_prob); 
                       */
                    for(int i = 0 ; i != byte_per_prob; i++){
                        prob_item.byte[i] = X_prob[base + i];
                        prob_item1.byte[i] = X_prob[base1 + i];
                    }

                    uint32_t t1 = prob_item.value;
                    uint32_t t2 = prob_item1.value;
                    uint32_t t3 = base_value - t1 - t2;
                    if(t1 >= cut_value){
                        geno_value = 0;
                    }else if(t2 >= cut_value){
                        geno_value = 2;
                    }else if(t3 >= cut_value){
                        geno_value = 3;
                    }else{
                        geno_value = 1;
                    }
                }else{
                    LOGGER.e(0, "multi-allelic SNPs detected in the " + to_string(raw_index) + "th SNP.");
                }
                buf_ptr[item_byte] += geno_value << move_byte;
            }
        }else{
            uint32_t A1U = floor(base_value * 1.5);
            uint32_t A1L = ceil(base_value * 0.5);


            for(uint32_t i = 0; i < num_keep_sample; i++){
                uint32_t item_byte = i >> 2;
                uint32_t move_byte = (i & 3) << 1;

                uint32_t sindex = index_keep[i];
                uint8_t item_ploidy = sample_ploidy[sindex];

                uint8_t geno_value;
                if(item_ploidy > 128){
                    //missing
                    geno_value = 1;
                }else if(item_ploidy == 2){
                    auto base = sindex * double_byte_per_prob;
                    auto base1 = base + byte_per_prob;
                    Geno_prob prob_item;
                    Geno_prob prob_item1;
                    /*
                       memcpy(prob_item.byte, X_prob + base,  byte_per_prob); 
                       memcpy(prob_item1.byte, X_prob + base1, byte_per_prob); 
                       */
                    for(int i = 0 ; i != byte_per_prob; i++){
                        prob_item.byte[i] = X_prob[base + i];
                        prob_item1.byte[i] = X_prob[base1 + i];
                    }

                    uint32_t t1 = prob_item.value;
                    uint32_t t2 = prob_item1.value;
                    uint32_t dosageA = 2 * t1 + t2;
                    if(dosageA > A1U){
                        geno_value = 0;
                    }else if(dosageA < A1L){
                        geno_value = 3;
                    }else{
                        geno_value = 2;
                    }
                }else{
                    LOGGER.e(0, " multi-allelic SNPs detected in the " + to_string(raw_index) + "th SNP.");
                }
                buf_ptr[item_byte] += geno_value << move_byte;
            }
 
        }
        //LOGGER.i(0, "MIDDLE: " + to_string(index) + "NUM_thread: " + to_string(omp_get_max_threads()));

        // exactly one 'ordered' directive must appear in the loop body of an enclosing directive
        // #pragma omp ordered
        save_bed(buf, num_marker);
        delete[] buf;
        delete[] dec_data;
        //#pragma omp ordered
        //LOGGER.i(0, "Finished " + to_string(index) + "NUM_thread: " + to_string(omp_get_max_threads()));
        if(index % 10000 == 0){
            float time_p = LOGGER.tp("LOOP_BGEN_BED");
            if(time_p > 300){
                LOGGER.ts("LOOP_BGEN_BED");
                float elapse_time = LOGGER.tp("LOOP_BGEN_TOT");
                float finished_percent = (float) index / num_markers;
                float remain_time = (1.0 / finished_percent - 1) * elapse_time / 60;

                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << finished_percent * 100 << "% Estimated time remaining " << remain_time << " min"; 
                
                LOGGER.i(1, ss.str());
            }
        }

    }
    //infos.close();
    closeOut();
    fclose(h_bgen);
}

// extracted and revised from plink2.0
// GPL v3, license detailed on github
// https://github.com/chrchang/plink-ng

void Geno::save_bed(uint64_t *buf, int num_marker){
    static string err_string = "can't write to [" + out_prefix_ + ".bed].";
    static bool inited = false;
    if(!inited){
        hOut = fopen((out_prefix_ + ".bed").c_str(), "wb");
        if(hOut == NULL){
            LOGGER.e(0, err_string);
        }
        uint8_t header[3] = {0x6c, 0x1b, 0x01};
        if(3 != fwrite(header, sizeof(uint8_t), 3, hOut)){
            LOGGER.e(0, err_string);
        }
        inited = true;
    }
    uint64_t base_buffer = 0;
    for(int i = 0; i < num_marker; i++){
        uint8_t *buffer = (uint8_t *)(buf + base_buffer);
        if(fwrite(buffer, sizeof(uint8_t), num_byte_keep_geno1, hOut) != num_byte_keep_geno1){
            LOGGER.e(0, err_string);
        }
        base_buffer += num_item_1geno;
    }
}

void Geno::closeOut(){
    fclose(hOut);
}


// === Restored functions ===

void Geno::setMAF(double val){
    if(val < 0){
        LOGGER.e(0, "MAF can't be negative: " + to_string(val));
    }
    this->min_maf = val * (1.0 - Constants::SMALL_EPSILON);
    deterFilterMAF();
}

double Geno::getMAF(){
    return this->min_maf;
}

double Geno::getFilterInfo(){
    return this->dFilterInfo;
}

double Geno::getFilterMiss(){
    return this->dFilterMiss;
}

void Geno::deterFilterMAF(){
    if(max_maf < min_maf){
        LOGGER.e(0, "the value specified for --max-maf can't be smaller than that for --min-maf");
    }

    if(std::abs(min_maf) <= 1e-10 && std::abs(max_maf - 0.5) <= 1e-10){
        this->bFilterMAF = false;
    }else{
        this->bFilterMAF = true;
    }
}

void Geno::setMaxMAF(double val){
    if(val <= 0 || val > 0.5){
        LOGGER.e(0, "the value specified for --max-maf can't be negative or larger than 0.5");
    }
    this->max_maf = val * (1.0 + Constants::SMALL_EPSILON);
    deterFilterMAF();
}

void Geno::setFilterInfo(double val){
    this->dFilterInfo = val;
}

void Geno::setFilterMiss(double val){
    this->dFilterMiss = val;
}

void Geno::processFreq(){
    string name_out = out_prefix_ + ".frq";
    int buf_size = 23068672;
    osBuf.resize(buf_size);
    osOut.rdbuf()->pubsetbuf(&osBuf[0], buf_size);
 
    osOut.open(name_out.c_str());
    if (!osOut) { LOGGER.e(0, "cannot open the file [" + name_out + "] to write."); }
    osOut << "CHR\tSNP\tPOS\tA1\tA2\tAF\tNCHROBS";
    if(hasInfo){
        osOut << "\tINFO";
    }
    osOut << "\n";

    LOGGER << "Computing allele frequencies and saving them to [" << name_out << "]..." << std::endl;

    int nMarker = 128;
    vector<uint32_t> extractIndex(marker->count_extract());
    std::iota(extractIndex.begin(), extractIndex.end(), 0);
    
    vector<function<void (uintptr_t *, std::span<const uint32_t>)>> callBacks;
    callBacks.emplace_back([this](uintptr_t* buf, std::span<const uint32_t> idx) {
        freq_func(buf, idx);
    });

    numMarkerOutput = 0;
    loopDouble(extractIndex, nMarker, false, false, false, false, callBacks);

    osOut.flush();
    osOut.close();
    LOGGER << "Saved " << numMarkerOutput << " SNPs." << std::endl;
}

void Geno::processRecodet(){
    string name_out = out_prefix_ + ".xmat";
    int buf_size = 23068672;
    osBuf.resize(buf_size);
    osOut.rdbuf()->pubsetbuf(&osBuf[0], buf_size);
 
    osOut.open(name_out.c_str());
    if (!osOut) { LOGGER.e(0, "cannot open the file [" + name_out + "] to write."); }
    osOut << "CHR\tSNP\tPOS\tA1\tA2\tAF\tNCHROBS";
    if(hasInfo){
        osOut << "\tINFO";
    }

    LOGGER << "Recoding genotypes and saving them to [" << name_out << "]..." << std::endl;
    uint32_t n_sample = pheno->count_keep();
    vector<string> phenoID = pheno->get_id(0, n_sample - 1, "|");

    for(auto & phenItem : phenoID){
        osOut << "\t" << phenItem;
    }
    osOut << "\n";

    int nMarker = 128;
    bool center, std, saveMiss;
    if(recode_method_ == "std"){
        center = true;
        std = true;
        saveMiss = false;
    }else if(recode_method_ == "nomiss"){
        center = false;
        std = false;
        saveMiss = false;
    }else if(recode_method_ == "raw"){
        center = false;
        std = false;
        saveMiss = true;
    }
    bRecodeSaveMiss = saveMiss;

    vector<uint32_t> extractIndex(marker->count_extract());
    std::iota(extractIndex.begin(), extractIndex.end(), 0);
    
    vector<function<void (uintptr_t *, std::span<const uint32_t>)>> callBacks;
    callBacks.emplace_back([this](uintptr_t* buf, std::span<const uint32_t> idx) {
        recode_func(buf, idx);
    });

    numMarkerOutput = 0;
    loopDouble(extractIndex, nMarker, true, center, std, saveMiss, callBacks);

    osOut.flush();
    osOut.close();
    LOGGER << "Saved " << numMarkerOutput << " SNPs." << std::endl;

}

void Geno::freq_func(uintptr_t* genobuf, std::span<const uint32_t> markerIndex){
    int num_marker = markerIndex.size();
    vector<uint8_t> isValids(num_marker);
    vector<double> af(num_marker);
    vector<uint32_t> nValidAllele(num_marker);
    vector<double> info(num_marker);
    /*
    vector<string> outs(num_marker);
    int bufsize = keepSampleCT * 10;
    for(int i = 0; i < num_marker; i++){
        outs[i].reserve(bufsize);
    }
    */
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < num_marker; i++){
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;

        getGenoDouble(genobuf, i, &item);
        isValids[i] = item.valid;
        if(item.valid){
            af[i] = item.additive_af;
            nValidAllele[i] = item.nValidAllele;
            info[i] = item.info;
        }
    }
    //output
    for(int i = 0; i != num_marker; i++){
        if(isValids[i]){
            numMarkerOutput++;
            osOut << marker->getMarkerStrExtract(markerIndex[i]) << "\t"  << af[i]
                <<"\t" << nValidAllele[i];
            if(hasInfo)osOut << "\t" << info[i];
            osOut << "\n";
        }
    }

}

void Geno::recode_func(uintptr_t* genobuf, std::span<const uint32_t> markerIndex){
    int num_marker = markerIndex.size();
    //vector<uint8_t> isValids(num_marker);
    /*
    vector<string> outs(num_marker);
    int bufsize = keepSampleCT * 10;
    for(int i = 0; i < num_marker; i++){
        outs[i].reserve(bufsize);
    }
    */
    #pragma omp parallel for ordered schedule(static,1)
    for(int i = 0; i < num_marker; i++){
        uint32_t cur_marker = markerIndex[i];
        GenoBufItem item;
        item.extractedMarkerIndex = cur_marker;

        getGenoDouble(genobuf, i, &item);

        #pragma omp ordered
        {
            if(item.valid) {
                numMarkerOutput++;
                osOut << marker->getMarkerStrExtract(cur_marker) << "\t"
                    << item.additive_af << "\t" << item.nValidAllele;
                if(hasInfo) osOut << "\t" << item.info;
                for(int j = 0; j < keepSampleCT; j++){
                    osOut << "\t";
                    if(bRecodeSaveMiss && (item.missing[j/64] & (1UL << (j %64)))){
                        osOut << "NA";
                    }else{
                        osOut << item.geno[j];
                    }
                }
                osOut << "\n";
            }
        }
    }
}



// ─────────────────────────────────────────────────────────────────────────────
// make_bed: convert BED/PGEN input to PLINK .bed/.bim/.fam
// Uses a two-pass strategy:
//   Pass 1 (no geno extraction) — determines which markers pass all runtime
//           filters (MAF, missingness, INFO) via getGenoDouble item.valid.
//   Pass 2 (with geno + missing mask) — writes the .bed file.
// The .bim is written between the two passes so it matches exactly the set
// of variants written to .bed.
// ─────────────────────────────────────────────────────────────────────────────
void Geno::processMakeBed() {
    LOGGER.i(0, "Making PLINK binary PED files...");
    const string filename = out_prefix_;

    // ── Pass 1: collect valid extract-list indices ────────────────────────
    const int N = static_cast<int>(marker->count_extract());
    vector<uint32_t> extractIndex(N);
    std::iota(extractIndex.begin(), extractIndex.end(), 0);

    vector<uint32_t> validIdx;
    validIdx.reserve(N);

    loopDouble(extractIndex, Constants::NUM_MARKER_READ,
               /*bMakeGeno*/false, /*bGenoCenter*/false, /*bGenoStd*/false, /*bMakeMiss*/false,
        {[this, &validIdx](uintptr_t *buf, std::span<const uint32_t> exIdx) {
            for (int i = 0; i < static_cast<int>(exIdx.size()); ++i) {
                GenoBufItem item;
                item.extractedMarkerIndex = exIdx[i];
                getGenoDouble(buf, i, &item);
                if (item.valid) validIdx.push_back(exIdx[i]);
            }
        }}, /*showLog*/false);

    marker->keep_extracted_index(validIdx);
    LOGGER.i(0, to_string(validIdx.size()) + " SNPs pass QC filters.");

    // ── Write .fam and .bim (valid markers only) ──────────────────────────
    pheno->save_pheno(filename + ".fam");
    marker->save_marker(filename + ".bim");

    // ── Pass 2: write .bed ────────────────────────────────────────────────
    hOut = fopen((filename + ".bed").c_str(), "wb");
    if (!hOut) LOGGER.e(0, "cannot open the file [" + filename + ".bed] to write.");
    const uint8_t bedHeader[3] = {0x6c, 0x1b, 0x01};
    fwrite(bedHeader, sizeof(uint8_t), 3, hOut);

    const int N2 = static_cast<int>(marker->count_extract());
    extractIndex.resize(N2);
    std::iota(extractIndex.begin(), extractIndex.end(), 0);
    numMarkerOutput = 0;

    loopDouble(extractIndex, Constants::NUM_MARKER_READ,
               /*bMakeGeno*/true, /*bGenoCenter*/false, /*bGenoStd*/false, /*bMakeMiss*/true,
        {[this](uintptr_t *buf, std::span<const uint32_t> exIdx) {
            make_bed_func(buf, exIdx);
        }});

    fclose(hOut);
    hOut = nullptr;
    LOGGER << numMarkerOutput << " SNPs have been saved to [" << filename << ".bed]." << std::endl;
}

// BED 2-bit encoding per sample (LSB-first within each byte, 4 samples/byte):
//   0b00 = 0 → homozygous A1   (GCTA dosage 0)
//   0b01 = 1 → missing
//   0b10 = 2 → heterozygous    (GCTA dosage 1)
//   0b11 = 3 → homozygous A2   (GCTA dosage 2)
static constexpr uint8_t kGenoToBed[3] = {0u, 2u, 3u};

void Geno::make_bed_func(uintptr_t* genobuf, std::span<const uint32_t> markerIndex) {
    const int num_marker = static_cast<int>(markerIndex.size());
    const uint32_t bytesPerMarker = (keepSampleCT + 3) / 4;
    vector<uint8_t> bedBuf(bytesPerMarker);

    for (int i = 0; i < num_marker; i++) {
        GenoBufItem item;
        item.extractedMarkerIndex = markerIndex[i];
        getGenoDouble(genobuf, i, &item);
        if (!item.valid) continue;

        std::fill(bedBuf.begin(), bedBuf.end(), 0u);
        for (uint32_t j = 0; j < keepSampleCT; j++) {
            uint8_t bits;
            if (item.missing[j / 64u] & (static_cast<uintptr_t>(1) << (j % 64u))) {
                bits = 1u;  // 0b01 = missing
            } else {
                const int g = static_cast<int>(std::round(item.geno[j]));
                bits = (g >= 0 && g <= 2) ? kGenoToBed[g] : 1u;
            }
            bedBuf[j / 4u] |= static_cast<uint8_t>(bits << ((j % 4u) * 2u));
        }

        if (fwrite(bedBuf.data(), 1u, bytesPerMarker, hOut) != bytesPerMarker) {
            LOGGER.e(0, "write error to [" + out_prefix_ + ".bed].");
        }
        ++numMarkerOutput;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// sum_geno_x: per-sample genotype sum across all (kept) homogametic chromosome markers.
// Output file columns: FID TAB IID TAB Sex TAB SumGenoHomogametic TAB NValidHomogametic TAB MeanGenoHomogametic
// Sex encoding follows FAM convention: 1=heterogametic, 2=homogametic, 0=unknown.
// ─────────────────────────────────────────────────────────────────────────────
void Geno::processSumGenoHomogametic() {
    LOGGER.i(0, "Computing per-sample genotype sums on homogametic chromosomes...");
    const string outFile = out_prefix_ + ".sum_geno_x.txt";

    const uint32_t nSample = pheno->count_keep();
    sumGenoHomogametic.assign(nSample, 0.0);
    nValidGenoHomogametic.assign(nSample, 0u);

    const int N = static_cast<int>(marker->count_extract());
    vector<uint32_t> extractIndex(N);
    std::iota(extractIndex.begin(), extractIndex.end(), 0);

    loopDouble(extractIndex, Constants::NUM_MARKER_READ,
               /*bMakeGeno*/true, /*bGenoCenter*/false, /*bGenoStd*/false, /*bMakeMiss*/false,
        {[this](uintptr_t *buf, std::span<const uint32_t> exIdx) {
            sum_geno_homogametic_func(buf, exIdx);
        }});

    // ── Write output ──────────────────────────────────────────────────────
    std::ofstream out(outFile.c_str());
    if (!out) LOGGER.e(0, "cannot open the file [" + outFile + "] to write.");
    out << "FID\tIID\tSex\tSumGenoHomogametic\tNValidHomogametic\tMeanGenoHomogametic\n";

    const vector<string> ids = pheno->get_id(0, static_cast<int>(nSample) - 1, "\t");
    for (uint32_t i = 0; i < nSample; i++) {
        const int8_t sex = pheno->get_sex(i);
        const double mean = (nValidGenoHomogametic[i] > 0u)
                            ? sumGenoHomogametic[i] / static_cast<double>(nValidGenoHomogametic[i])
                            : 0.0;
        out << ids[i] << "\t" << static_cast<int>(sex)
            << "\t" << sumGenoHomogametic[i]
            << "\t" << nValidGenoHomogametic[i]
            << "\t" << mean << "\n";
    }
    out.close();
    LOGGER.i(0, "Results saved to [" + outFile + "].");
}

void Geno::sum_geno_homogametic_func(uintptr_t* genobuf, std::span<const uint32_t> markerIndex) {
    const int num_marker = static_cast<int>(markerIndex.size());
    const int nS = static_cast<int>(keepSampleCT);
    const int nThreads = omp_get_max_threads();

    // Thread-local buffers avoid atomic overhead on sumGenoHomogametic[j] (shared per-sample arrays).
    vector<vector<double>>   localSum(nThreads, vector<double>(nS, 0.0));
    vector<vector<uint32_t>> localN  (nThreads, vector<uint32_t>(nS, 0u));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_marker; i++) {
        const int tid = omp_get_thread_num();
        GenoBufItem item;
        item.extractedMarkerIndex = markerIndex[i];
        getGenoDouble(genobuf, i, &item);
        if (!item.valid) continue;

        for (int j = 0; j < nS; j++) {
            localSum[tid][j] += item.geno[j];
            localN  [tid][j]++;
        }
    }

    // Merge thread-local results into the member accumulators.
    for (int t = 0; t < nThreads; t++) {
        for (int j = 0; j < nS; j++) {
            sumGenoHomogametic   [j] += localSum[t][j];
            nValidGenoHomogametic[j] += localN  [t][j];
        }
    }
}
