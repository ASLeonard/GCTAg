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
#include "Cli.hpp"
#include "Geno.h"
#include "async_seq_io.hpp"
#include "grm_format.hpp"
#include <memory>
#include <optional>
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
    GRM(Pheno *pheno, Marker *marker, const gcta_cli::GrmMakeOptions& opts, const std::string& out_prefix);
    GRM(const std::string& grm_prefix, const std::string& keep_file, const std::string& remove_file,
        const std::string& out_prefix, bool cutoff_detail = false, bool out_fp16 = false);
    ~GRM() = default;

    void calculate_GRM_blas(uintptr_t* genobuf, std::span<const uint32_t> markerIndex);

    void N_thread(int grm_index_from, int grm_index_to, const uintptr_t* cmask);
    void deduce_GRM();
    vector<uint32_t> divide_parts(uint32_t from, uint32_t to, uint32_t num_parts);
    vector<uint32_t> divide_parts_mem(uint32_t n_sample, uint32_t num_parts);

    static void runMake(const gcta_cli::GrmMakeOptions& opts, const std::string& out_prefix);
    static void runMerge(const gcta_cli::GrmMergeOptions& opts, const std::string& out_prefix);
    static void runPrune(const gcta_cli::GrmPruneOptions& opts, const std::string& out_prefix);
    static void runSparse(const gcta_cli::GrmSparseOptions& opts, const std::string& out_prefix);
    static void runSubtract(const gcta_cli::GrmCombineOptions& opts, const std::string& out_prefix);
    static void runUnify(const gcta_cli::GrmCombineOptions& opts, const std::string& out_prefix);
    void processMakeGRM(bool is_homogametic = false);
    void processMakeGRMX();

    void loop_block(vector<function<void (double *buf, int num_block)>> callbacks
                    = vector<function<void (double *buf, int num_block)>>());
    void cut_rel(float thresh, bool no_grm = false);
    void prune_fam(float thresh, bool isSparse = true, float *val = NULL);
    void unify_grm(string mgrm_file, string out_file, bool out_fp16 = false);
    void subtract_grm(string mgrm_file, string out_file, bool out_fp16 = false);
    void merge_grm(string mgrm_file, string out_file, int row_block_rows, bool out_fp16 = false);

    inline static int nMarkerBlock = 1024;

private:
    Pheno *pheno = nullptr;
    Marker *marker = nullptr;
    std::unique_ptr<Geno> geno;
    std::vector<uint32_t> index_keep;
    uint32_t part = 1;
    uint32_t num_parts = 1;
    uint32_t num_individual = 0;
    uint64_t num_grm = 0;
    std::pair<uint32_t, uint32_t> part_keep_indices{0, 0}; // the index in pheno keep index
    std::vector<std::pair<int, int>> index_grm_pairs;

    // 4096-row strip-packed lower-triangular accumulators: strip s covering
    // global rows [r0, r1) has shape (r1 - r0) x r1 ColMajor (ldc = r1 - r0 <= 4096).
    // Eliminates the 2x virtual/THP RSS blowup on tile 0 and L2 STLB thrashing.
    static constexpr int kGrmStripRows = 4096;
    std::vector<AlignedUniquePtr<double, 64>> grm_strips_;
    std::vector<AlignedUniquePtr<int32_t, 64>> grm_i32_strips_;
    // Unified packed triangular N buffer for [grm_tile_rs, grm_tile_re)
    // (identical layout in both tiled and non-tiled paths).
    AlignedUniquePtr<uint32_t, 64> N;
    bool grm_is_i32_ = false;
    uint32_t grm_tile_i32_markers_ = 0;
    std::vector<double> R_total_;
    double S_total_ = 0.0;
    bool grm_any_missing_seen_ = false;
    bool n_markers_explicit_ = false;
    static constexpr uint32_t kMaxInt32Markers = 500000000u;

    // Native .grm.f16 writer / preset state
    bool write_fp16_ = false;
    bool fast_preset_ = false;
    bool is_homogametic_ = false;
    uint64_t grm_id_hash_ = 0;
    std::vector<float> fp16_diag_;
    double fp16_trace_K_ = 0.0;
    double fp16_trace_K2_ = 0.0;

    std::vector<uint32_t> sub_miss; // sample miss in all markers
    int finished_marker = 0;
    uint64_t num_byte_geno = 0;
    int num_count_handle = 2;

    bool bBLAS = true;
    AlignedUniquePtr<float, 64> stdGeno;
    AlignedUniquePtr<int8_t, 64> stdGeno_i8;   // BED int8: compact columns of markers with missing calls only
    int stdGeno_i8_cols_ = 0;                  // capacity (columns) of stdGeno_i8
    std::vector<int> miss_col_of_;             // valid-marker index -> column in stdGeno_i8 (or -1)
    AlignedUniquePtr<int8_t, 64> stdGeno_i8_kmajor_;
    bool use_int8_grm = false;

    void output_id();

    string o_name;
    std::string keep_file;
    std::string remove_file;
    std::optional<float> sparse_cutoff;
    bool cutoff_detail = false;

    // GRM-reader state (for grm prune / grm sparse)
    string grm_file;
    std::string input_grm_data_path_;
    gcta_grm_io::GrmDiskFormat input_grm_format_ = gcta_grm_io::GrmDiskFormat::F32Bin;
    gcta_grm_io::GrmFp16Header input_f16_hdr_{};
    std::vector<float> input_f16_diag_;
    bool input_has_n_file_ = false;
    uint64_t num_byte_buffer = 0;
    uint64_t num_subjects = 0;
    std::vector<string> grm_ids;
    static constexpr size_t kGrmScanSlotBytes = 64ULL << 20;
    std::vector<uint64_t> byte_part_grms;

    void write_pruned_grm_output(bool out_fp16);

    bool isDominance = false;
    bool isMtd = false;
    std::vector<double> sd;
    uint32_t numValidMarkers = 0;

    std::vector<GenoBufItem> gbufitems;

    // Constants derived from part_keep_indices, cached after construction
    int grm_n = 0;               // part_keep_indices.second + 1
    int grm_m = 0;               // part_keep_indices.second - part_keep_indices.first + 1
    int grm_s_n = 0;             // grm_n - grm_m
    size_t grm_bytes_std_geno = 0;  // sizeof(elem) * grm_n
    int stdGenoLD = 0;           // leading dimension of stdGeno: max(grm_n, index_keep.size()) rounded up to multiple of 64

    // Pre-allocated / lazily-allocated scratch buffers reused across calculate_GRM_blas calls
    std::vector<int> validIndexBuf;
    AlignedRawBuffer<uintptr_t, 64> sampleMissBuf;
    AlignedRawBuffer<float, 64> grm_f32_tile_buf_;
    AlignedRawBuffer<int32_t, 64> grm_i32_tile_;
    AlignedRawBuffer<int32_t, 64> grm_i32_diag_scratch_;
    AlignedRawBuffer<int8_t, 64> i8_reordered_B_;
    std::vector<float> i8_scratch_A_f32_;
    std::vector<float> i8_scratch_B_f32_;
    std::vector<float> i8_scratch_C_f32_;
    std::vector<double> mu_vec_;
    std::vector<double> r_raw_;
    std::vector<double> r_tilde_;
    std::vector<int> marker_miss_offset_;
    std::vector<int> marker_miss_idx_;
    std::vector<int> sample_miss_offset_;
    std::vector<int> sample_miss_m_;

    // Block-tiled GRM: when grm_tiling_enabled is active (set via --tile-budget),
    // processMakeGRM iterates over row tiles [grm_tile_rs, grm_tile_re).
    bool grm_tiling_enabled = false;
    double grm_tile_budget_bytes = 0;
    int  grm_tile_rs    = 0;  // tile row start (global index)
    int  grm_tile_re    = 0;  // tile row end   (global, exclusive)
    int  grm_tile_rows  = 0;  // grm_tile_re - grm_tile_rs
    int  grm_tile_cols  = 0;  // grm_tile_re (widest row length)
    bool grm_skip_global_state = false;
    bool grm_first_accum = true;

    // Tile-1 valid-marker & stats cache reused across Tiles 2..T
    std::vector<uint32_t> tile1_valid_extract_;
    std::vector<double> tile1_valid_mean_;
    std::vector<uint8_t> tile1_valid_has_miss_;
    size_t tile1_cursor_ = 0;
    bool use_tile1_cache_ = false;

    void allocate_grm_strips(bool as_i32);
    void reset_grm_strips();
    void ensure_N_allocated();
    void promote_i32_to_f64();

    // Write the current tile's normalised rows to the single-stream AsyncSeqWriter
    // (or sparse FILE*). When N_fd >= 0, drains grm_writer first and writes N_fd
    // sequentially so two files are never written concurrently.
    void flush_grm_tile(gcta_io::AsyncSeqWriter* grm_writer,
                        int N_fd,
                        FILE *sparse_out,
                        const std::string& grm_path,
                        const std::string& N_path,
                        float thresh,
                        bool isSparse,
                        double mtd_weight,
                        std::vector<float>& const_N_buf,
                        std::string& sparse_buf);
};


#endif //GCTA2_GRM_H
