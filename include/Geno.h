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

#ifndef GCTA2_GENO_H
#define GCTA2_GENO_H
#include <string>
#include <map>
#include <span>
#include <vector>
#include "Pheno.h"
#include "Marker.h"
#include "Cli.hpp"
#include "Logger.h"
#include "GenoBackend.h"     // GenoBlock, GenoBackend
#include "bed_simd.hpp"      // native PLINK1 .bed kernels
#include <functional>

struct SNPInfo;              // third_party/Pgenlib/PgenReader.h

// Forward declarations for the three concrete backends (needed for friend).
class BedBackend;
class BgenBackend;
class PgenBackend;

using std::function;
using std::string;
using std::map;
using std::vector;

typedef struct GenoBufItem{
    // in
    uint32_t extractedMarkerIndex = 0;   // for allele lookup
    double *geno_ptr = nullptr;          // optional external genotype buffer (length >= keepSampleCT)
    float  *geno_f32_ptr = nullptr;      // optional external float32 genotype buffer (length >= keepSampleCT)
    int8_t *geno_i8_ptr = nullptr;       // optional external int8 raw {0,1,2} genotype buffer (length >= keepSampleCT)
    // BED only: when true and the marker has no missing call among the kept
    // samples (nValidN == keepSampleCT), `missing` is left untouched instead of
    // being zero-filled.  Callers must then gate reads of `missing` on nValidN.
    bool lazy_missing = false;
    // BED only: compute counts / af / mean / sd / validity (including the
    // sd < 1e-50 check) but skip the genotype decode.
    bool stats_only = false;

    // out
    bool valid = false;
  //  int refAllele; //0 as 1, 1 as GCTA
    uint8_t sexChromType = 0;   // 0: non-sex, 1: homogametic, 2: heterogametic
    vector<double> geno;
    vector<uintptr_t> missing;
    double af = 0.0;
    // Real (additive-scale) allele frequency, captured after sample compaction
    // but before genotypes are recoded under the nonadditive/dominance model.
    // Mirrors `af` (which is already computed from raw hard-call/dosage counts
    // and is unaffected by recoding) — use this for reporting/output purposes.
    double additive_af = 0.0;
    double mean = 0.0;
    double sd = 0.0; //std^2
    double info = 0.0;
    uint32_t nValidN = 0;
    uint32_t nValidAllele = 0;
} GenoBufItem;

// Optional fused side-outputs for Geno::getGenoInt8_bed (BED additive MLMA):
// exact genotype counts plus weighted sums computed during the single decode
// pass, so callers never re-scan the int8 column.
struct BedI8Fused {
    // in
    const float*  vw = nullptr;          ///< per-kept-sample float weights (e.g. V^-1 y), or nullptr
    const double* w2 = nullptr;          ///< per-kept-sample double weights (e.g. w_sqrt^2), or nullptr
    std::vector<int>* miss_list = nullptr; ///< receives ascending missing sample indices when cmiss > 0
    // out
    uint32_t c0 = 0, c1 = 0, c2 = 0, cmiss = 0;  ///< exact counts of values 0/1/2/missing
    double dot_vw = 0.0, miss_vw = 0.0;           ///< sum g*vw, sum_{missing} vw
    double dot_w2 = 0.0, dot2_w2 = 0.0, miss_w2 = 0.0; ///< sum g*w2, sum g^2*w2, sum_{missing} w2
};


class Geno {
public:
    Geno(Pheno* pheno,
         Marker* marker,
         const gcta_cli::GenoInputOptions& geno_opts,
         const gcta_cli::MarkerFilterOptions& marker_filter,
         const std::string& out_prefix = "");
    ~Geno();
    // Used internally by bgen2bed for BED output
    void save_bed(uint64_t *buf, int num_marker);
    void closeOut();
    void setMAF(double val);
    void setMaxMAF(double val);
    void setFilterInfo(double val);
    void setFilterMiss(double val);
    double getFilterInfo();
    double getMAF();
    double getFilterMiss();
            
    bool check_bed() = delete;
    void out_freq(string filename);
    bool filterMAF();
    uint32_t getTotalMarker();

    // coroutine-based loop — pre/end now handled by GenoBackend RAII
    void getGenoDouble(uintptr_t *buf, int bufIndex, GenoBufItem* gbuf);

    // Direct float-valued genotype decode: writes n float values into dest[].
    // Sets *af_out and *valid_out from the same allele-frequency / filter logic.
    // Uses single-precision lookup tables (BED) or casts from double (BGEN/PGEN).
    void getGenoFloat(uintptr_t *buf, int bufIndex, float *dest,
                      float &af_out, float &additive_af_out,
                      bool &valid_out, uint32_t extractedMarkerIndex);

    // Direct uncentered int8 genotype decode for BED additive MLMA streaming:
    // writes n int8 values {0, 1, 2} (missing -> 0) into dest[], and populates
    // miss_out bitmask when nValidN_out < keepSampleCT (left untouched
    // otherwise).  Single fused pass (bed_simd): counts, optional weighted
    // sums and sparse missing indices are produced alongside the decode.
    void getGenoInt8_bed(uintptr_t *buf, int idx, int8_t *dest,
                         float &af_out, float &additive_af_out,
                         double &mu_out, uint32_t &nValidN_out,
                         uintptr_t *miss_out, bool &valid_out,
                         uint32_t extractedMarkerIndex,
                         BedI8Fused *fused = nullptr);

    // BED int8 GRM: decode block-local markers markerIdx[0..K) for kept samples
    // [0, n_rows) straight from the raw .bed records into the sample-major
    // (K-contiguous) matrix kmajor (row stride ldK), fused with
    // r_raw[s] = sum_m g(s, m) * mu[m] (accumulated in ascending m), using a
    // cache-blocked 64x64 decode+transpose.  No intermediate marker-major buffer.
    void decodeBedI8KMajor(uintptr_t *buf, const int *markerIdx, int K, int n_rows,
                           int8_t *kmajor, int ldK, const double *mu, double *r_raw);

    bool canUseInt8Mlma() const {
        return genoFormat == "BED" &&
               genoCodingModel == GenoCodingModel::ADDITIVE &&
               iDC == 1;
    }
    int getMissPtrSize() const { return missPtrSize; }
    uint32_t getRawSampleCT() { return pheno ? pheno->count_raw() : keepSampleCT; }

    void loopDouble(const vector<uint32_t> &extractIndex, int numMarkerBuf, bool bMakeGeno, bool bGenoCenter, bool bGenoStd, bool bMakeMiss, vector<function<void (uintptr_t *buf, std::span<const uint32_t> exIndex)>> callbacks = vector<function<void (uintptr_t *buf, std::span<const uint32_t> exIndex)>>(), bool showLog = true);

    bool getGenoHasInfo();
    const string& getGenoFormat() const { return genoFormat; }

    void setGRMMode(bool grm, bool dominace);
    void setGenoCodingModel(gcta_cli::GenoModel model);
    void setGenoItemSize(uint32_t &genoSize, uint32_t &missSize);
 
private:
    Pheno* pheno;
    Marker* marker;
    FILE* hOut = NULL;

    vector<double> AFA1;
    
    //vector<uint32_t> countA1A1;
    //vector<uint32_t> countA1A2;
    //vector<uint32_t> countA2A2;
    vector<uint32_t> countMarkers;
    //vector<double> RDev;
    uint32_t total_markers = 0;

    std::string out_prefix_;
    std::string recode_method_ = "nomiss";
    double opt_min_maf = 0.0;
    double opt_max_maf = 0.5;
    double opt_geno_rate = 1.0;
    double opt_info_score = 0.0;
    double opt_dos_dc = -1.0;
    double opt_hard_call_thresh = 0.9;
    bool opt_dosage_call = false;

    void init_AF(string alleleFileName);
    uint64_t num_item_1geno;
    uint64_t num_raw_sample;
    uint64_t num_keep_sample;
    uint64_t num_byte_keep_geno1;
    uint64_t num_heterogametic_keep_sample;
    uint64_t num_item_geno_buffer;
    uint64_t *keep_mask = NULL;
    uint64_t *keep_heterogametic_mask = NULL;
    bool isHomogameticChrom;
    void bgen2bed(const vector<uint32_t> &raw_marker_index);

    friend class LD;

// seq read start;
    string genoFormat = "";
    vector<string> geno_files;
    vector<int> compressFormats;
    vector<uint32_t> rawCountSamples;
    vector<uint32_t> rawCountSNPs;
    vector<uint32_t> sampleKeepIndex;

    bool bHasPreAF = false;
    int numMarkerBlock;

    double min_maf = 0.0;
    double max_maf = 0.5;
    double dFilterInfo = 0;
    bool bFilterMAF = false;
    double dFilterMiss = 0; // 1 - missingrate
    void deterFilterMAF();

    // Concrete backend classes need access to private members.
    friend class BedBackend;
    friend class BgenBackend;
    friend class PgenBackend;

    // getGenoDouble — resolved once at construction from genoFormat.
    typedef void (Geno::*GetGenoDoubleFunc)(uintptr_t *buf, int idx, GenoBufItem* gbuf);
    GetGenoDoubleFunc getGenoDoubleFunc = nullptr;

    bool hasInfo = false;

    bool bMakeGeno;
    bool bGenoCenter;
    bool bGenoStd;
    bool bMakeMiss;
    bool bGRM = false;
    bool bGRMDom = false;
    int iGRMdc = -1; // 0 no heterogametic dosage comp; 1 full comp; //default value shall be -1, equal variance
    int iDC = 1;
    bool f_std = false;
    void setHeterogameticWeight(double &weight, bool &needWeight); // set the heterogametic weight by bGRM, dc specity

    enum class GenoCodingModel : uint8_t { ADDITIVE = 0, DOMINANCE = 1, NONADDITIVE = 2 };
    struct GenoCodingSpec {
        double centerValue = 0.0;
        double rdev = 1.0;
        double a0 = 0.0;
        double a1 = 0.0;
        double a2 = 0.0;
        double na = 0.0;
    };
    // Single-precision variant: same logical layout but float entries.
    struct GenoCodingSpecF32 {
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float na = 0.0f;
    };
    GenoCodingModel getCodingModel() const;
    double mapDosageToModel(double    dosage, double mu, GenoCodingModel model) const;
    GenoCodingSpec    buildCodingSpec(double mu, double sd) const;
    GenoCodingSpecF32 buildCodingSpecF32(double mu, double sd) const;
    GenoCodingModel genoCodingModel = GenoCodingModel::ADDITIVE;

    int8_t alleModel = 1; // 1: add; 2: Dom; 3: Reces; 4: Het; //currently unused affect a0 a1 a2 na;

    int curBufferIndex;
    vector<int> numMarkersReadBlocks;
    vector<uint8_t> markerSexChromTypes;
    vector<int> fileIndexBuf;
    vector<int32_t> baseIndexLookup;

    // Format-specific getGenoDouble (processing/unpacking — unchanged).
    // pre/read/endGeno* moved to BedBackend / BgenBackend / PgenBackend.
    void getGenoDouble_bed(uintptr_t *buf, int idx, GenoBufItem* gbuf);
    void getGenoDouble_bgen(uintptr_t *buf, int idx, GenoBufItem* gbuf);
    void getGenoDouble_pgen(uintptr_t *buf, int idx, GenoBufItem* gbuf);

    // Float-native decode helpers: write directly into a caller-supplied float*.
    // getGenoFloat_bed: bed_simd register-table decode (VPERMPS on AVX-512).
    // getGenoFloat_pgen / getGenoFloat_bgen: share the frequency/filter logic
    //   of their double counterparts, then cast the dosage array to float.
    void getGenoFloat_bed(uintptr_t *buf, int idx, float *dest,
                          float &af_out, float &additive_af_out, bool &valid_out,
                          uint32_t extractedMarkerIndex);
    void getGenoFloat_pgen(uintptr_t *buf, int idx, float *dest,
                           float &af_out, float &additive_af_out, bool &valid_out,
                           uint32_t extractedMarkerIndex);
    void getGenoFloat_bgen(uintptr_t *buf, int idx, float *dest,
                           float &af_out, float &additive_af_out, bool &valid_out,
                           uint32_t extractedMarkerIndex);

    using GetGenoFloatFunc = void (Geno::*)(uintptr_t*, int, float*, float&, float&, bool&, uint32_t);
    GetGenoFloatFunc getGenoFloatFunc = nullptr;
 
    //BED
    uint64_t bedRecordBytes = 0; // exact bytes per raw .bed variant record (unpadded contiguous stride)
    int maskPtrSize;
    int missPtrSize;
    uint32_t rawSampleCT;
    uint32_t keepSampleCT;

    // 1-bit-per-raw-sample masks (bed_simd::build_keep_mask layout), owned by BedBackend.
    uintptr_t *keepMaskPtr = NULL;
    
    uint32_t keepSexSampleCT;
    uint32_t keepHeterogameticSampleCT;

    vector<uint32_t> keepSexIndex;  // in the raw index of fam

    vector<uint32_t> keepHeterogameticIndex; // in raw index of fam
    vector<uint32_t> keepHeterogameticExtractIndex;
    uintptr_t *heterogameticMaskPtr = NULL;

    // Native BED helpers (bed_simd).
    const uint8_t *bedRecord(const uintptr_t *buf, int idx) const {
        return reinterpret_cast<const uint8_t *>(buf) +
               static_cast<std::size_t>(idx) * static_cast<std::size_t>(bedRecordBytes);
    }
    const uint64_t *bedKeepMask() const {
        return keepSampleCT == rawSampleCT ? nullptr : reinterpret_cast<const uint64_t *>(keepMaskPtr);
    }
    // Counts over the kept samples (+ heterogametic subset on homogametic
    // chromosomes) and the SNPInfo exactly as PgenReader::CountHardFreqMissExt(X).
    bed_simd::Counts bedSnpInfo(const uint8_t *raw, uint8_t sexChromType, SNPInfo *snpinfo) const;
    void bedSnpInfoFromCounts(const uint8_t *raw, const bed_simd::Counts &c, uint8_t sexChromType,
                              SNPInfo *snpinfo) const;

    //BGEN
    int bgenRawGenoBuf1PtrSize;

    //PGEN
    int pgenGenoBuf1PtrSize;
    int pgenGenoPtrSize;
    int pgenDosagePresentPtrSize;
    int pgenDosageMainPtrSize;

    std::ofstream osOut;
    FILE * bOut = NULL;
    vector<char> osBuf;
    uint32_t numMarkerOutput = 0;

    // main funcs
    void processRecodet();
    bool bRecodeSaveMiss = false;
    void recode_func(uintptr_t* genobuf, std::span<const uint32_t> markerIndex);

    void processFreq();
    void freq_func(uintptr_t * genobuf, std::span<const uint32_t> markerIndex);

    // make_bed: convert any supported genotype format to PLINK BED (non-BGEN path).
    // Uses a two-pass approach: pass 1 determines valid markers, pass 2 writes .bed.
    void processMakeBed();
    void make_bed_func(uintptr_t* genobuf, std::span<const uint32_t> markerIndex);

    // sum_geno_x: per-sample genotype sum across homogametic chromosome markers.
    // Used for dosage compensation analysis (--sum-geno-x).
    void processSumGenoHomogametic();
    void sum_geno_homogametic_func(uintptr_t* genobuf, std::span<const uint32_t> markerIndex);
    vector<double>   sumGenoHomogametic;    // per-sample accumulated genotype sum on homogametic chromosomes
    vector<uint32_t> nValidGenoHomogametic; // per-sample count of valid homogametic markers

 };


#endif //GCTA2_GENO_H
