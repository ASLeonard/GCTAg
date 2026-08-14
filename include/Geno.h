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
#include "Logger.h"
#include "GenoBackend.h"     // GenoBlock, GenoBackend, GenoScheduler
#include <functional>
#include "tables.h"

// Forward declarations for the three concrete backends (needed for friend).
class BedBackend;
class BgenBackend;
class PgenBackend;

using std::function;
using std::string;
using std::map;
using std::vector;
using std::placeholders::_1;
using std::placeholders::_2;
using std::bind;

typedef struct GenoBuffer{
    bool center; //center or not //IN
    bool std;  //std or not  //IN
    bool saveMiss; // save missing or not //IN
    bool saveGeno; // save geno or no //IN

    uint32_t n_sample; // number of filtered keep samples //preIN
    uint32_t n_marker; // number of largest buffer size; //preIN
    uint32_t nBLMiss; // missing data 1 marker size (number of uint64_t); //preIN

    uint32_t n_sub_sample = 0; // subset samples, 0 mean all; //IN
    uint32_t indexStartMarker = 0; // which index to start.  //IN

    vector<uint8_t> usedIndex; //true the index pass the filter.
    vector<double> af; // allele frequencies
    vector<float> info; // info scores
    vector<uint32_t> nValidN; // non-missing samples in a marker
    vector<uint32_t> nValidAllele; // non-missing SNPs in a marker;
    vector<double> geno;  // genotype double per marker
    vector<uint64_t> miss; // missing bits per marker, 1 means miss.

    bool bSex = false; // sex mode (homogametic chromosome filter) //IN
    bool hasInfo = false; // determine has info score or not //preIN
    bool bHasPreAF = false; //use Pre AF? //preIN
    vector<double> preAF; //pre AF //preIN
} GenoBuf;

typedef struct GenoBufItem{
    // in
    uint32_t extractedMarkerIndex = 0;   // for allele lookup

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


class Geno {
public:
    Geno(Pheno* pheno, Marker* marker);
    ~Geno();
    // LOCO helper: override active genotype source to a single PLINK prefix.
    static void setLocoBfilePrefix(const std::string& bfile_prefix);
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
    static int registerOption(map<string, vector<string>>& options_in);
    static void processMain();
    bool filterMAF();
    static void setSexMode();
    uint32_t getTotalMarker();

    // coroutine-based loop — pre/end now handled by GenoBackend RAII
    void getGenoDouble(uintptr_t *buf, int bufIndex, GenoBufItem* gbuf);

    // Direct float-valued genotype decode: writes n float values into dest[].
    // Sets *af_out and *valid_out from the same allele-frequency / filter logic.
    // Uses single-precision lookup tables (BED) or casts from double (BGEN/PGEN).
    void getGenoFloat(uintptr_t *buf, int bufIndex, float *dest,
                      float &af_out, float &additive_af_out,
                      bool &valid_out, uint32_t extractedMarkerIndex);

    void loopDouble(const vector<uint32_t> &extractIndex, int numMarkerBuf, bool bMakeGeno, bool bGenoCenter, bool bGenoStd, bool bMakeMiss, vector<function<void (uintptr_t *buf, std::span<const uint32_t> exIndex)>> callbacks = vector<function<void (uintptr_t *buf, std::span<const uint32_t> exIndex)>>(), bool showLog = true);

    bool getGenoHasInfo();

    void setGRMMode(bool grm, bool dominace);
    void setGenoCodingModel(const string& model_name);
    void setGenoItemSize(uint32_t &genoSize, uint32_t &missSize);
 
private:
    Pheno* pheno;
    Marker* marker;
    FILE* hOut = NULL;

    GBitCountTable g_table;
    vector<double> AFA1;
    
    //vector<uint32_t> countA1A1;
    //vector<uint32_t> countA1A2;
    //vector<uint32_t> countA2A2;
    vector<uint32_t> countMarkers;
    //vector<double> RDev;
    uint32_t total_markers;

    
    static map<string, string> options;
    static map<string, double> options_d;
    static vector<string> processFunctions;
    static void addOneFileOption(string key_store, string append_string, string key_name,
                                 map<string, vector<string>> options_in);

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
    // getGenoFloat_bed: uses the 256×4 float lookup table (GenoarrLookup256x4bx4).
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
    int bedRawGenoBuf1PtrSize; // how many 64bit geno of raw sample save 
    int maskPtrSize;
    int missPtrSize;
    uint32_t rawSampleCT;
    uint32_t keepSampleCT;

    uintptr_t *keepMaskPtr = NULL;
    uintptr_t *keepMaskInterPtr = NULL; 
    
    uint32_t keepSexSampleCT;
    uint32_t keepHeterogameticSampleCT;

    vector<uint32_t> keepSexIndex;  // in the raw index of fam
    uintptr_t *sexMaskPtr = NULL;
    uintptr_t *sexMaskInterPtr = NULL;

    vector<uint32_t> keepHeterogameticIndex; // in raw index of fam
    vector<uint32_t> keepHeterogameticExtractIndex;
    uintptr_t *heterogameticMaskPtr = NULL;
    uintptr_t *heterogameticMaskInterPtr = NULL; 
    //uintptr_t *heterogameticMaskExtractPtr = NULL;

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
