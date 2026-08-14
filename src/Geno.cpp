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
#include <boost/algorithm/string.hpp>
#include "OptionIO.h"
#include "zlib.h"
#include "zstd.h"
#include <cstring>
#include "cpu.h"
#include <Eigen/Eigen>
#include <algorithm>
#include "third_party/Pgenlib/PgenReader.h"
#include <numeric>

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

map<string, string> Geno::options;
map<string, double> Geno::options_d;
vector<string> Geno::processFunctions;

Geno::Geno(Pheno* pheno, Marker* marker) {
    geno_files.clear();
    int num_geno = 0;
    if(options.find("geno_file") != options.end()){
        genoFormat = "BED";
        geno_files.push_back(options["geno_file"]);
        num_geno++;
        hasInfo = false;
    }

    if(options.find("m_file") != options.end()){
        genoFormat = "BED";
        boost::split(geno_files, options["m_file"], boost::is_any_of("\t "));
        //std::transform(geno_files.begin(), geno_files.end(), geno_files.begin(), [](string r){return r + ".bed";});
        num_geno++;
        hasInfo = false;
    }

    if(options.find("pgen_file") != options.end()){
        genoFormat = "PGEN";
        geno_files.push_back(options["pgen_file"]);
        boost::split(geno_files, options["pgen_file"], boost::is_any_of("\t "));
        num_geno++;
        hasInfo = false;
    }

    if(options.find("mpgen_file") != options.end()){
        genoFormat = "PGEN";
        boost::split(geno_files, options["mpgen_file"], boost::is_any_of("\t "));
        num_geno++;
        hasInfo = false;
    }

    if(options.find("bgen_file") != options.end()){
        genoFormat = "BGEN";
        geno_files.push_back(options["bgen_file"]);
        num_geno++;
        hasInfo = true;
    }

    if(options.find("mbgen_file") != options.end()){
        genoFormat = "BGEN";
        boost::split(geno_files, options["mbgen_file"], boost::is_any_of("\t "));
        num_geno++;
        hasInfo = true;
    }

    if(num_geno == 0){
        LOGGER.e(0, "no genotype file is specified");
    }

    //open and check genotype files

    this->pheno = pheno;
    this->marker = marker;

    //this->sampleKeepIndex = pheno->get_index_keep();

    //register format handlers



    // Resolve the unpacking function once so getGenoDouble() is a direct
    // member-function call on every marker, not a hash-map lookup.
    if      (genoFormat == "BED")  getGenoDoubleFunc = &Geno::getGenoDouble_bed;
    else if (genoFormat == "PGEN") getGenoDoubleFunc = &Geno::getGenoDouble_pgen;
    else if (genoFormat == "BGEN") getGenoDoubleFunc = &Geno::getGenoDouble_bgen;
    else LOGGER.e(0, "Geno: unknown genoFormat '" + genoFormat + "' — cannot resolve getGenoDouble.");

    if      (genoFormat == "BED")  getGenoFloatFunc = &Geno::getGenoFloat_bed;
    else if (genoFormat == "PGEN") getGenoFloatFunc = &Geno::getGenoFloat_pgen;
    else if (genoFormat == "BGEN") getGenoFloatFunc = &Geno::getGenoFloat_bgen;

    string alleleFileName = "";
    if(options.find("update_freq_file") != options.end()){
        alleleFileName = options["update_freq_file"];
    }

    setMAF(options_d["min_maf"]);
    setMaxMAF(options_d["max_maf"]);
    setFilterInfo(options_d["info_score"]);
    setFilterMiss(1.0 - options_d["geno_rate"]);

    string filterprompt = "Threshold to filter variants:";
    bool outFilterPrompt = false;
    if(options_d["min_maf"] != 0.0){
        filterprompt += " MAF > " + to_string(options_d["min_maf"]);
        outFilterPrompt = true;
    }
    if(options_d["max_maf"] != 0.5){
        filterprompt += string(outFilterPrompt ? "," : "") + " MAF < " + to_string(options_d["max_maf"]);
        outFilterPrompt = true;
    }
    if(options_d["info_score"] != 0.0){
        filterprompt += string(outFilterPrompt ? "," : "") + " imputation INFO score > " + to_string(options_d["info_score"]);
        outFilterPrompt = true;
    }

    if(options_d["geno_rate"] != 1.0){
        filterprompt += string(outFilterPrompt ? "," : "") + " missingness rate < " + to_string(options_d["geno_rate"]);
        outFilterPrompt = true;
    }
    if(outFilterPrompt){
        LOGGER << filterprompt << "." << std::endl;
    }
    if(options_d["dos_dc"] == 1.0){
        iGRMdc = 1;
        iDC = 1;
        LOGGER << "Switch to the full dosage compensation mode." << std::endl;
    }else if(options_d["dos_dc"] == 0.0){
        iGRMdc = 0;
        iDC = 0;
        LOGGER << "Switch to the no dosage compensation mode." << std::endl;
    }else{
        // default equal variance mode for grm
        iGRMdc = -1;
        // compensation mode for x
        iDC = 1;
    }

    init_AF(alleleFileName);

    //olds
    //init_AsyncBuffer();

    //num_keep_sample = 0;
    //init_keep();
    //olds;
}

Geno::~Geno(){
}

void Geno::setLocoBfilePrefix(const std::string& bfile_prefix) {
    if (bfile_prefix.empty()) {
        LOGGER.e(0, "[LOCO] empty bfile prefix in manifest.");
    }

    options["geno_file"] = bfile_prefix + ".bed";
    options.erase("m_file");
    options.erase("pgen_file");
    options.erase("mpgen_file");
    options.erase("bgen_file");
    options.erase("mbgen_file");

    std::ifstream f(options["geno_file"].c_str());
    if (!f.good()) {
        LOGGER.e(0, "[LOCO] manifest bfile not found: [" + options["geno_file"] + "]");
    }
}

uint32_t Geno::getTotalMarker(){
    return total_markers;
}

void Geno::setSexMode(){
    std::map<string, vector<string>> t_option;
    t_option["--chr-homogametic"] = {};
    t_option["--filter-sex"] = {}; 
    Pheno::registerOption(t_option);
    Marker::registerOption(t_option);
    Geno::registerOption(t_option);
}


//
//true:  filtered; false: not necessary to filter
bool Geno::filterMAF(){
    if((options_d["min_maf"] != 0.0) || (options_d["max_maf"] != 0.5)){
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
        double min_maf_eps = options_d["min_maf"] * (1.0 - Constants::SMALL_EPSILON);
        double max_maf_eps = options_d["max_maf"] * (1.0 + Constants::SMALL_EPSILON);
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

void Geno::setGenoCodingModel(const string& model_name){
    if (model_name == "additive") {
        genoCodingModel = GenoCodingModel::ADDITIVE;
    } else if (model_name == "dominance") {
        genoCodingModel = GenoCodingModel::DOMINANCE;
    } else if (model_name == "nonadditive") {
        genoCodingModel = GenoCodingModel::NONADDITIVE;
    } else {
        LOGGER.e(0, "Unknown genotype model '" + model_name + "'. Expected additive, nonadditive, or dominance.");
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

// BED path: 256×4 float lookup table decoded by GenoarrLookup256x4bx4.
// The 256-entry table (256 * 4 * 4 B = 4 KB) fits entirely in L1D.
void Geno::getGenoFloat_bed(uintptr_t *buf, int idx, float *dest,
                             float &af_out, float &additive_af_out,
                             bool &valid_out, uint32_t extractedMarkerIndex) {
    valid_out = false;

    SNPInfo snpinfo;
    uintptr_t *cur_buf = buf + idx * bedRawGenoBuf1PtrSize;
    uint8_t sexChromType = markerSexChromTypes[curBufferIndex];
    bool hasNoHET = true;
    if (sexChromType != 1) {
        PgenReader::CountHardFreqMissExt(cur_buf, keepMaskInterPtr, rawSampleCT, keepSampleCT, &snpinfo, f_std);
    } else {
        string errmsg;
        hasNoHET = PgenReader::CountHardFreqMissExtX(cur_buf, keepMaskInterPtr, heterogameticMaskInterPtr,
                                                     rawSampleCT, keepSampleCT, keepHeterogameticSampleCT,
                                                     &snpinfo, errmsg, iDC == 1, f_std);
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

    // Build 256×4 float lookup (QUAD_TABLE256 expands to 1024 floats = 4 KB).
    const GenoCodingSpecF32 spec = buildCodingSpecF32(mu, sd);
    const float gtable256x4[1024] __attribute__((aligned(16))) =
        QUAD_TABLE256(spec.a0, spec.a1, spec.a2, spec.na);

    PgenReader::ExtractFloatExt(cur_buf, keepMaskPtr, rawSampleCT, keepSampleCT,
                                 gtable256x4, dest, /*missOut=*/nullptr);

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
                // Both rdev and centerValue are embedded in spec already, but
                // spec.{a0,a1,a2,na} are already (raw - center)*rdev.
                // We need correctWeight = (w-1) * centerValue * rdev.
                // Recompute via buildCodingSpec to avoid storing center+rdev in F32.
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

// PGEN path: reuse frequency/filter logic from getGenoDouble_pgen, then cast.
void Geno::getGenoFloat_pgen(uintptr_t *buf, int idx, float *dest,
                              float &af_out, float &additive_af_out,
                              bool &valid_out, uint32_t extractedMarkerIndex) {
    valid_out = false;
    // Decode via the existing double path into a temporary GenoBufItem.
    GenoBufItem tmp;
    tmp.extractedMarkerIndex = extractedMarkerIndex;
    getGenoDouble_pgen(buf, idx, &tmp);
    if (!tmp.valid) return;
    valid_out       = true;
    af_out          = static_cast<float>(tmp.af);
    additive_af_out = static_cast<float>(tmp.additive_af);
    for (uint32_t j = 0; j < keepSampleCT; ++j)
        dest[j] = static_cast<float>(tmp.geno[j]);
}

// BGEN path: same strategy as PGEN — delegate to the double path, then cast.
void Geno::getGenoFloat_bgen(uintptr_t *buf, int idx, float *dest,
                              float &af_out, float &additive_af_out,
                              bool &valid_out, uint32_t extractedMarkerIndex) {
    valid_out = false;
    GenoBufItem tmp;
    tmp.extractedMarkerIndex = extractedMarkerIndex;
    getGenoDouble_bgen(buf, idx, &tmp);
    if (!tmp.valid) return;
    valid_out       = true;
    af_out          = static_cast<float>(tmp.af);
    additive_af_out = static_cast<float>(tmp.additive_af);
    for (uint32_t j = 0; j < static_cast<uint32_t>(tmp.geno.size()); ++j)
        dest[j] = static_cast<float>(tmp.geno[j]);
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
    uintptr_t *cur_buf = buf + idx * pgenGenoBuf1PtrSize;
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
            gbuf->geno.resize(keepSampleCT);
            for(uint32_t j = 0; j < keepSampleCT; ++j){
                const double dos = static_cast<double>(dosage_main[j]) * kDosageScale;
                const double recoded = mapDosageToModel(dos, mu, model);
                gbuf->geno[j] = (recoded - spec.centerValue) * spec.rdev;
            }

            if(sexChromType == 1){
                double weight;
                bool needWeight;
                setHeterogameticWeight(weight, needWeight);
                if(needWeight){
                    for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                        gbuf->geno[keepHeterogameticExtractIndex[i]] *= weight;
                    }
                }
            }
        }
        if(bMakeMiss){
            gbuf->missing.resize(missPtrSize, 0);
        }
    }
}

//TODO: do we need to adjust here
void Geno::getGenoDouble_bed(uintptr_t *buf, int idx, GenoBufItem* gbuf){
    SNPInfo snpinfo;
    uintptr_t *cur_buf = buf + idx * bedRawGenoBuf1PtrSize;
    uint8_t sexChromType = markerSexChromTypes[curBufferIndex];
    bool hasNoHET = true;
    if(sexChromType != 1){
        PgenReader::CountHardFreqMissExt(cur_buf, keepMaskInterPtr, rawSampleCT, keepSampleCT, &snpinfo, f_std);
    }else{
        string errmsg;
        hasNoHET = PgenReader::CountHardFreqMissExtX(cur_buf, keepMaskInterPtr, heterogameticMaskInterPtr, rawSampleCT, keepSampleCT, keepHeterogameticSampleCT, &snpinfo, errmsg, iDC==1, f_std);
    }
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
                const GenoCodingSpec spec = buildCodingSpec(mu, sd);
                const double lookup[32] __attribute__ ((aligned (16))) =
                    GET_TABLE16(spec.a0, spec.a1, spec.a2, spec.na);
                gbuf->geno.resize(keepSampleCT);
                uintptr_t * pmiss = NULL;
                if(bMakeMiss){
                    gbuf->missing.resize(missPtrSize); 
                    pmiss = gbuf->missing.data();
                }
                PgenReader::ExtractDoubleExt(cur_buf, keepMaskPtr, rawSampleCT, keepSampleCT, lookup, gbuf->geno.data(), pmiss); 
                if(sexChromType == 1){
                    double weight;
                    bool needWeight;
                    setHeterogameticWeight(weight, needWeight);
                    if(needWeight){
                        if(bGRM){
                            for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                gbuf->geno[keepHeterogameticExtractIndex[i]] *= weight;
                            }
                        }else{
                            double correctWeight = (weight - 1) * spec.rdev * spec.centerValue;
                            for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                uint32_t curIndex = keepHeterogameticExtractIndex[i];
                                gbuf->geno[curIndex] *= weight;
                                gbuf->geno[curIndex] += correctWeight;
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
    uintptr_t *cur_buf = buf + idx * bgenRawGenoBuf1PtrSize;
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

                gbuf->geno.resize(curSampleCT);
                for(int j = 0; j < curSampleCT; j++){
                    gbuf->geno[j] = dos_lookup[dosages[j]];
                }
                delete[] dos_lookup;
                if(sexChromType == 1){
                    double weight;
                    bool needWeight;
                    setHeterogameticWeight(weight, needWeight);
                    if(needWeight){
                        if(bGRM){
                            for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                gbuf->geno[keepHeterogameticExtractIndex[i]] *= weight;
                            }
                        }else{
                            double correctWeight = (weight - 1) * spec.rdev * spec.centerValue;
                            for(int i = 0 ; i < keepHeterogameticSampleCT; i++){
                                uint32_t curIndex = keepHeterogameticExtractIndex[i];
                                gbuf->geno[curIndex] *= weight;
                                gbuf->geno[curIndex] += correctWeight;
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

    // ── Thread pools ──────────────────────────────────────────────────────
    //    io_pool  : 1 thread — file handles and PgenReader are not thread-safe.
    //    cpu_pool : 1 thread — Option A from the coroutine migration plan.
    //      Callbacks use #pragma omp parallel for internally, so OMP expands
    //      to omp_get_max_threads() workers inside each block.  The full
    //      machine is therefore used; cpu_pool only acts as the scheduler
    //      that launches each OMP region.
    //
    //    *** Do NOT raise cpu_pool above 1 without first removing or
    //        nesting-guarding every OMP pragma in GRM.cpp and FastFAM.cpp.
    //        Concurrent OMP regions from multiple cpu_pool threads would
    //        oversubscribe the machine and degrade performance. ***
    exec::static_thread_pool io_pool{1};
    exec::static_thread_pool cpu_pool{1};   // OMP handles intra-block parallelism

    LOGGER.ts("LOOP_GENO_TOT");
    LOGGER.ts("LOOP_GENO_PRE");
    uint32_t nFinishedMarker = 0;
    const uint32_t nTMarker  = static_cast<uint32_t>(extractIndex.size());
    int pre_block = 0;

    // ── Adapter: GenoBlock → legacy (uintptr_t*, vector<uint32_t>) ───────
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
                    float finished_percent = (float)nFinishedMarker / nTMarker;
                    float remain_time      = (1.0f / finished_percent - 1) * elapse_time / 60;
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(1)
                       << finished_percent * 100
                       << "% Estimated time remaining " << remain_time << " min";
                    LOGGER.i(1, ss.str());
                }
            }
        }
    };

    // Backend takes extractIndex by value so block metadata can safely
    // hold spans/slices without caller lifetime coupling.
    auto streamExtractIndex = extractIndex;

    // ── Run coroutine pipeline (blocks until all markers processed) ───────
    stdexec::sync_wait(backend->stream(
        io_pool.get_scheduler(),
        cpu_pool.get_scheduler(),
        std::move(streamExtractIndex),
        std::move(blockCallback)));

    if (showLog) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1)
           << "100% finished in " << LOGGER.tp("LOOP_GENO_TOT") << " sec";
        LOGGER.i(1, ss.str());
        LOGGER << nFinishedMarker << " SNPs have been processed." << std::endl;
    }
    // backend destructor runs end-logic (masks freed, files closed).
}

//function for BGEN
inline void bgen12ExtractVal(uint64_t val, uint8_t bit_prob, uint64_t mask, uint64_t &v1, uint64_t &v2){
    v1 = val & mask;
    v2 = (val >> bit_prob) & mask;
}

// call genotype
void dosageFunc(uint32_t prob1, uint32_t prob2, uint64_t mask, uint32_t cutVal, uint32_t A1U, uint32_t A1L, double &gval, bool &miss){
    gval = (double)(prob1 * 2 + prob2) / mask;
}


void dosageCallFunc(uint32_t prob1, uint32_t prob2, uint64_t mask, uint32_t cutVal, uint32_t A1U, uint32_t A1L, double &gval, bool &miss){
    uint32_t dos = prob1 * 2 + prob2;
    if(dos > A1U){
        gval = 2.0;
    }else if(dos < A1L){
        gval = 0.0;
    }else{
        gval = 1.0;
    }
}

void hardCallFunc(uint32_t prob1, uint32_t prob2, uint64_t mask, uint32_t cutVal, uint32_t A1U, uint32_t A1L, double &gval, bool &miss){
    miss = false;
    if(prob1 >= cutVal){
        gval = 2.0;
    }else if(prob2 >= cutVal){
        gval = 1.0;
    }else if(mask - prob1 - prob2 >= cutVal){
        gval = 0.0;
    }else{
        gval = 0.0;
        miss = true;
    }
};
        /* obsoleted functions hard calling
        uint8_t double_bits_prob = bits_prob * 2;
        uint32_t A1U = floor(mask * 1.5);
        uint32_t A1L = ceil(mask * 0.5);
        uint32_t cutVal = ceil(mask * options_d["hard_call_thresh"]);

        void (*callFunc)(uint32_t prob1, uint32_t prob2, uint64_t mask, uint32_t cutVal, uint32_t A1U, uint32_t A1L, double &gval, bool &miss);
        if(options.find("dosage_call") != options.end()){
            callFunc = dosageCallFunc;
        }else if(options.find("dosage") != options.end()){
            callFunc = dosageFunc;
        }else{
            callFunc = hardCallFunc;
        }

        auto callFuncB = [callFunc, mask, cutVal, A1U, A1L](uint32_t prob1, uint32_t prob2, double &gval, bool&miss){
            callFunc(prob1, prob2, mask, cutVal, A1U, A1L, gval, miss);
        };
    
        //vector<uint32_t> prob1s(n_res), prob2s(n_res);
        vector<uint32_t> miss_index;
        double sum_geno = 0;
        double *base_geno = (gbuf->geno).data() + i * n_res;
        uint64_t *base_miss;
        uint32_t n_valid = 0;
        if(gbuf->saveMiss) base_miss = (gbuf->miss).data() + i * n_bmiss; 

        uint64_t dosage_sum = 0, fij_dosage2_sum = 0;
        bool *lookup_miss = new bool[mask + 1];


        for(int j = 0; j < n_res; j++){
            uint32_t sindex = sampleKeepIndex[j];
            uint8_t item_ploidy = sample_ploidy[sindex];
            if(item_ploidy > 128){
                miss_index.push_back(sindex);
            }else if(item_ploidy == 2){
                uint32_t start_bits = sindex * double_bits_prob;
                uint64_t geno_temp;
                memcpy(&geno_temp, &(X_prob[start_bits/CHAR_BIT]), sizeof(uint64_t));
                geno_temp = geno_temp >> (start_bits % CHAR_BIT);
                uint32_t prob1 = geno_temp & mask;
                uint32_t prob2 = (geno_temp >> bits_prob) & mask;
                uint32_t prob1d = prob1 * 2;
                uint32_t dosage = prob1d + prob2;

                

                dosage_sum += dosage;

                uint32_t fij = dosage + prob1d;
                fij_dosage2_sum += (fij - dosage * dosage);
            }else{
               LOGGER.e(0, "multi-allelic SNPs detected in " + error_promp);
            }
        }

        for(int j = 0; j < n_res; j++){
            uint32_t sindex = sampleKeepIndex[j];
            uint8_t item_ploidy = sample_ploidy[sindex];
            if(item_ploidy > 128){
                miss_index.push_back(sindex);
            }else if(item_ploidy == 2){
                uint32_t start_bits = sindex * double_bits_prob;
                uint64_t geno_temp;
                memcpy(&geno_temp, &(X_prob[start_bits/CHAR_BIT]), sizeof(uint64_t));
                geno_temp = geno_temp >> (start_bits % CHAR_BIT);
                uint32_t prob1 = geno_temp & mask;
                uint32_t prob2 = (geno_temp >> bits_prob) & mask;

                //total_prob += ((prob1 << 1) + prob2);
                //prob1s[i] = prob1;
                //prob2s[i] = prob2;
                double gval;
                bool miss;
                callFuncB(prob1, prob2, gval, miss);
                base_geno[j] = gval;
                if(miss){
                    miss_index.push_back(j);
                    if(gbuf->saveMiss) base_miss[j / 64] |= (1UL << (j % 64)); 
                }else{
                    n_valid++;
                }
           }else{
               LOGGER.e(0, "multi-allelic SNPs detected in " + error_promp);
           }
        }
        delete[] dec_data;
        Eigen::VectorXd Vgeno = Eigen::Map< Eigen::Matrix<double,Eigen::Dynamic,1> > (base_geno, n_res);
        double af = Vgeno.sum() / (2*n_valid);
        (gbuf->validN)[i] = n_valid;
        (gbuf->af)[i] = af;
        */

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

    FILE * h_bgen = fopen(options["bgen_file"].c_str(), "rb");
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
        if(options.find("dosage_call") == options.end()){
            uint32_t cut_value = ceil(base_value * options_d["hard_call_thresh"]);
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
    static string err_string = "can't write to [" + options["out"] + ".bed].";
    static bool inited = false;
    if(!inited){
        hOut = fopen((options["out"] + ".bed").c_str(), "wb");
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


void Geno::addOneFileOption(string key_store, string append_string, string key_name,
                                     map<string, vector<string>> options_in) {
    if(options_in.find(key_name) != options_in.end()){
        if(options_in[key_name].size() == 1){
            options[key_store] = options_in[key_name][0] + append_string;
        }else if(options_in[key_name].size() > 1){
            options[key_store] = options_in[key_name][0] + append_string;
            LOGGER.w(0, "Geno: multiple " + key_name + ", use the first one only" );
        }else{
            LOGGER.e(0, "no " + key_name + " parameter found");
        }
        std::ifstream f(options[key_store].c_str());
        if(!f.good()){
            LOGGER.e(0, key_name + " " + options[key_store] + " not found");
        }
        f.close();
    }
}

int Geno::registerOption(map<string, vector<string>>& options_in) {
    int return_value = 0;
    addOneFileOption("geno_file", ".bed", "--bfile", options_in);
    addOneFileOption("bgen_file", "", "--bgen", options_in);
    addOneFileOption("pgen_file", ".pgen", "--pfile", options_in);
    addOneFileOption("pgen_file", ".pgen", "--bpfile", options_in);

    addMFileListsOption("m_file", ".bed", "--mbfile", options_in, options);
    addMFileListsOption("mbgen_file", ".bgen", "--mbgen", options_in, options);
    addMFileListsOption("mpgen_file", ".pgen", "--mbpfile", options_in, options);
    addMFileListsOption("mpgen_file", ".pgen", "--mpfile", options_in, options);

    options_d["min_maf"] = 0.0;
    options_d["max_maf"] = 0.5;
    if(options_in.find("--maf") != options_in.end()){
        auto option = options_in["--maf"];
        if(option.size() == 1){
            try{
                options_d["min_maf"] = std::stod(option[0]);
            }catch(std::invalid_argument&){
                LOGGER.e(0, "invalid value for --maf");
            }
            if(options_d["min_maf"]<0.0 || options_d["max_maf"]>0.5){
                LOGGER.e(0, "value specified for--maf can't be smaller than 0 or larger than 0.5");
            }else if(options_d["min_maf"] == 0.0){
                options_in["--nofilter"] = {};
            }

        }else{
            LOGGER.e(0, "GCTA does not support multiple values for --maf currently");
        }
        options_in.erase("--maf");
    }

     if(options_in.find("--max-maf") != options_in.end()){
        auto option = options_in["--max-maf"];
        if(option.size() == 1){
            try{
                options_d["max_maf"] = std::stod(option[0]);
           }catch(std::invalid_argument&){
                LOGGER.e(0, "invalid value for --maf");
           }
           if(options_d["max_maf"] < 0.0 || options_d["max_maf"] > 0.5){
               LOGGER.e(0, "the value specified for --max-maf can't be smaller than 0 or larger than 0.5");
           }
        }else{
            LOGGER.e(0, " GCTA does not support multiple values for --maf currently ");
        }
        options_in.erase("--max-maf");
    }

    if(options_d["min_maf"] > options_d["max_maf"]){
        LOGGER.e(0, "value specified for --max-maf can't be smaller than that for --min-maf");
    }


    addOneValOption<double>("geno_rate", "--geno", options_in, options_d, 1.0, 0.0, 1.0);
    if(options_in.find("--geno") != options_in.end()){
        if(options_d["geno_rate"] == 0.0){
            options_in["--nofilter"] = {};
        }
    }

    addOneValOption<double>("info_score", "--info", options_in, options_d, 0.0, 0.0, 1.0);
    addOneValOption<double>("dos_dc", "--dc", options_in, options_d, -1.0, -1.0, 1.0);



    options_d["hard_call_thresh"] = 0.9;
    string flag = "--hard-call-thresh";
    if(options_in.find(flag) != options_in.end()){
        auto option = options_in[flag];
        if(options.size() == 1){
            try{
                options_d["hard_call_thresh"] = std::stod(option[0]);
            }catch(std::invalid_argument&){
                LOGGER.e(0, "invalid value in " + flag);
            }
        }else{
            LOGGER.e(0, " GCTA does not support multiple values for " + flag + " currently.");
        }
        options_in.erase(flag);
    }

    flag = "--dosage-call";
    if(options_in.find(flag) != options_in.end()){
        options["dosage_call"] = "true";
        options_in.erase(flag);
    }

    if(options_in.find("--freq") != options_in.end()){
        processFunctions.push_back("freq");
        if(options_in["--freq"].size() != 0){
            LOGGER.w(0, "--freq should not be followed by any parameter, if you want to calculate the allele frequencies in founders only, "
                    "please specify by --founders option");
        }
        options_in.erase("--freq");

        options["out"] = options_in["--out"][0];

        return_value++;
    }

    if(options_in.find("--freqx") != options_in.end()){
        processFunctions.push_back("freqx");
        if(options_in["--freqx"].size() != 0){
            LOGGER.w(0, "--freq should not be followed by any other parameter, if you want to calculate the allele frequencies in founders only, "
                    "please specify by --founders option");
        }
        options_in.erase("--freqx");

        options["out"] = options_in["--out"][0];

        return_value++;
    }

    if(options_in.find("--make-bed") != options_in.end()){
        if(options.find("bgen_file") == options.end()){
            processFunctions.push_back("make_bed");
        }else{
            processFunctions.push_back("make_bed_bgen");
        } 

        options_in.erase("--make-bed");
        options["out"] = options_in["--out"][0];

        return_value++;
    }

    if(options_in.find("--recodet") != options_in.end()){
        processFunctions.push_back("recodet");
        options["recode_method"] = "nomiss";
        if(options_in["--recodet"].size() > 0){
            string cop = options_in["--recodet"][0];
            vector<string> ops = {"nomiss", "raw", "std"};
            if(std::find(ops.begin(), ops.end(), cop) != ops.end()){
                options["recode_method"] = cop;
            }else{
                LOGGER.e(0, "can't recognize recode method: " + options_in["--recodet"][0]);
            }
        }
        options_in.erase("--recodet");
        options["out"] = options_in["--out"][0];
        return_value++;
    }


    addOneFileOption("update_freq_file", "", "--update-freq", options_in);

    if(options_in.find("--filter-sex") != options_in.end()){
        options["sex"] = "yes";
    }

    if(options_in.find("--sum-geno-x") != options_in.end()){
        processFunctions.push_back("sum_geno_homogametic");
        options["sex"] = "yes";
        std::map<string, vector<string>> t_option;
        t_option["--chr-homogametic"] = {};
        t_option["--filter-sex"] = {};
        Pheno::registerOption(t_option);
        Marker::registerOption(t_option);
        options["out"] = options_in["--out"][0];
        return_value++;
    }



    return return_value;
}

void Geno::processFreq(){
    string name_out = options["out"] + ".frq";
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
    callBacks.push_back(bind(&Geno::freq_func, this, _1, _2));

    numMarkerOutput = 0;
    loopDouble(extractIndex, nMarker, false, false, false, false, callBacks);

    osOut.flush();
    osOut.close();
    LOGGER << "Saved " << numMarkerOutput << " SNPs." << std::endl;
}

void Geno::processRecodet(){
    string name_out = options["out"] + ".xmat";
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
    if(options["recode_method"] == "std"){
        center = true;
        std = true;
        saveMiss = false;
    }else if(options["recode_method"] == "nomiss"){
        center = false;
        std = false;
        saveMiss = false;
    }else if(options["recode_method"] == "raw"){
        center = false;
        std = false;
        saveMiss = true;
    }
    bRecodeSaveMiss = saveMiss;

    vector<uint32_t> extractIndex(marker->count_extract());
    std::iota(extractIndex.begin(), extractIndex.end(), 0);
    
    vector<function<void (uintptr_t *, std::span<const uint32_t>)>> callBacks;
    callBacks.push_back(bind(&Geno::recode_func, this, _1, _2));

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
    const string filename = options["out"];

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
            LOGGER.e(0, "write error to [" + options["out"] + ".bed].");
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
    const string outFile = options["out"] + ".sum_geno_x.txt";

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

void Geno::processMain() {
    for(auto &process_function : processFunctions){
        if(process_function == "freq"){
            Pheno pheno;
            Marker marker;
            Geno geno(&pheno, &marker);
            geno.processFreq();
        }

        if(process_function == "recodet"){
            Pheno pheno;
            Marker marker;
            Geno geno(&pheno, &marker);
            geno.processRecodet();
        }

        if(process_function == "make_bed"){
            Pheno pheno;
            Marker marker;
            Geno geno(&pheno, &marker);
            geno.processMakeBed();
        }

        if(process_function == "make_bed_bgen"){
            Pheno pheno;
            Marker marker;
            Geno geno(&pheno, &marker);
            string filename = options["out"];
            pheno.save_pheno(filename + ".fam");
            marker.save_marker(filename + ".bim");
            LOGGER.i(0, "Converting bgen to PLINK binary PED format [" + filename + ".bed]...");
            geno.bgen2bed(marker.get_extract_index());
            LOGGER.i(0, "Genotype has been saved.");
        }

        if(process_function == "sum_geno_homogametic"){
            Pheno pheno;
            Marker marker;
            Geno geno(&pheno, &marker);
            geno.processSumGenoHomogametic();
        }
    }
}
