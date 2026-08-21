/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   New implementation: generate, read and process GRM.

   Depends on the class of genotype

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GCTA2_GRM_H
#define GCTA2_GRM_H
#include "Geno.h"
#include <string>
#include <span>
#include <vector>
#include <utility>
#include <cmath>
#include "constants.hpp"
#include "mem.hpp"

using std::string;
using std::vector;
using std::pair;

class GRM {
public:
    GRM(Pheno *pheno, Marker *marker);
    GRM();
    ~GRM() {
        posix_mem_free(grm);
        posix_mem_free(N);
        posix_mem_free(cmask_buf);
        if(lookup_GRM_table) delete[] lookup_GRM_table;
        if(geno_buf) posix_mem_free(geno_buf);
        if(mask_buf) posix_mem_free(mask_buf);
        //if(stdGeno) posix_mem_free(stdGeno);
        if(geno) delete geno;
    };

    void calculate_GRM(uintptr_t* genobuf, std::span<const uint32_t> markerIndex);
    void calculate_GRM_blas(uintptr_t* genobuf, std::span<const uint32_t> markerIndex);
    
    void grm_thread(int grm_index_from, int grm_index_to);
    void N_thread(int grm_index_from, int grm_index_to, const uintptr_t* cmask);
    void deduce_GRM();
    vector<uint32_t> divide_parts(uint32_t from, uint32_t to, uint32_t num_parts);
    vector<uint32_t> divide_parts_mem(uint32_t n_sample, uint32_t num_parts);

    static int registerOption(map<string, vector<string>>& options_in);
    static void processMain();
    void processMakeGRM();
    void processMakeGRMX();

    void loop_block(vector<function<void (double *buf, int num_block)>> callbacks
                    = vector<function<void (double *buf, int num_block)>>());
    void cut_rel(float thresh, bool no_grm = false);
    void prune_fam(float thresh, bool isSparse = true, float *val = NULL);
    void unify_grm(string mgrm_file, string out_file);
    void subtract_grm(string mgrm_file, string out_file);
    void merge_grm_streaming(string mgrm_file, string out_file, int row_block_rows);

    inline static int nMarkerBlock = 1024;

private:
    Pheno *pheno = NULL;
    Marker *marker = NULL;
    Geno *geno = NULL;
    std::vector<uint32_t> index_keep;
    uint32_t part;
    uint32_t num_parts;
    uint32_t num_individual;
    uint64_t num_grm;
    std::pair<uint32_t, uint32_t> part_keep_indices; //the index in pheno keep index;
    std::vector<std::pair<int, int>> index_grm_pairs;

    double *grm = NULL;
    uint32_t *N = NULL;
    std::vector<uint32_t> sub_miss; // sample miss in all markers

    //========
    // for byte style calculation
    uint16_t *geno_buf = NULL;  // genotype buffer tranpose
    uint16_t *mask_buf = NULL;  // mask of genotype
    uint64_t *cmask_buf = NULL; // sample count mask //used in both byte or blas
    int cur_num_block = 0;
    int finished_marker = 0;
    const static int num_marker_block = 5;
    const static int num_lookup_table = 8 * 8 * 8 * 8 * 8;

    const static int num_cmask_block = 60;


    double GRM_table[Constants::NUM_MARKER_READ][8];
    //double lookup_GRM_table[Constants::NUM_MARKER_READ / num_marker_block][num_lookup_table];
    double (*lookup_GRM_table)[num_lookup_table] = NULL;

    std::vector<uint16_t> elements = {0, 2, 3, 4, 5, 6, 7};
    uint64_t num_byte_geno;
    uint64_t num_byte_cmask;
    int reg_bit_width;
    int num_grm_handle;
    int num_count_handle;
    int num_block_handle;
    int num_marker_process_block;
    //====
    //end for byte style calculation

    bool bBLAS;
    double *stdGeno = NULL;

    void output_id();

    string o_name;

    static map<string, string> options;
    static map<string, double> options_d;
    static map<string, bool> options_b;
    static std::vector<string> processFunctions;

    string grm_file;
    int num_byte_buffer;
    uint64_t num_subjects;
    std::vector<string> grm_ids;
    static constexpr int num_byte_GRM_read = 100 * 1024 * 1024;
    std::vector<uint64_t> byte_part_grms;

    void outBinFile(FILE *sFile, FILE *dFile);

    bool isDominance = false;
    bool isMtd = false;
    std::vector<double> sd;
    uint32_t numValidMarkers = 0;

    GenoBufItem *gbufitems = NULL;

    // Constants derived from part_keep_indices, cached after construction
    int grm_n = 0;               // part_keep_indices.second + 1
    int grm_m = 0;               // part_keep_indices.second - part_keep_indices.first + 1
    int grm_s_n = 0;             // grm_n - grm_m
    int grm_bytes_std_geno = 0;  // sizeof(double) * grm_n
    int stdGenoLD = 0;           // leading dimension of stdGeno: grm_n rounded up to multiple of 8 (64-byte column alignment)

    // Pre-allocated scratch buffers reused across calculate_GRM_blas calls
    std::vector<int> validIndexBuf;
    std::vector<uintptr_t> sampleMissBuf;

    // Block-tiled GRM: when grm_tile_size > 0 (set via --GRM-tile-size),
    // processMakeGRM iterates over grm_tile_size-row tiles.  For each tile
    // [grm_tile_rs, grm_tile_re) a rectangular grm/N slab of
    // grm_tile_rows × grm_tile_cols elements is allocated, computed and
    // written before moving to the next tile.
    // grm and N are nullptr between tiles.
    // grm_tile_size == 0 means no tiling (single-pass, original behaviour).
    bool grm_tiling_enabled = false;      // true when --GRM-tile-budget is active
    double grm_tile_budget_bytes = 0;   // budget in bytes for the grm+N tile buffer; 0 = disabled
    int  grm_tile_rs    = 0;  // tile row start (global index)
    int  grm_tile_re    = 0;  // tile row end   (global, exclusive)
    int  grm_tile_rows  = 0;  // grm_tile_re - grm_tile_rs
    int  grm_tile_cols  = 0;  // grm_tile_re (rectangular buffer width)
    // When true, calculate_GRM_blas skips accumulating sub_miss / sd /
    // numValidMarkers / finished_marker.  Set after the first tile so that
    // subsequent tiles reuse the values computed on the first pass — these
    // quantities depend only on global genotype state, not on which tile row
    // range is being processed.
    bool grm_skip_global_state = false;

    // Write the current tile's normalised rows to already-open output files.
    // w_grm/w_N/grm_block/sparse_buf are caller-owned scratch (sized once by
    // processMakeGRM to the widest tile, reused across every tile call).
    void flush_grm_tile(FILE *grm_out, FILE *N_out,
                        float thresh, bool isSparse, float mtd_weight,
                        std::vector<float>& w_grm, std::vector<float>& w_N,
                        std::vector<float>& grm_block, std::string& sparse_buf);

    //Just for testing
#ifndef NDEBUG
    FILE * o_geno0;
    FILE * o_mask0;
#endif



};


#endif //GCTA2_GRM_H
