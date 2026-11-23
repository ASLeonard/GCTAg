/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   bed_simd: native PLINK1 .bed genotype kernels (counting + fused decode)
   with runtime ISA dispatch:

     x86_64 : AVX-512 (F/BW/DQ/VL/VBMI/VBMI2/VPOPCNTDQ + BMI2, e.g. AMD Zen 4)
              -> AVX2 (+BMI2/POPCNT, x86-64-v3) -> scalar.
              Selected once via __builtin_cpu_supports, or statically when the
              translation unit is compiled with -march=znver4 / native.
     arm64  : NEON (Apple Silicon / Graviton) -> scalar.

   Raw PLINK1 2-bit codes (LSB-first, 4 samples per byte) and the GCTA value
   each maps to (identical to PLINK2's PgrPlink1ToPlink2 + PgenReader):

        PLINK1 bits   meaning      int8 value  "code" index used by tables
        00            hom A1        2           2
        01            missing       0           3   (table[3] = missing value)
        10            het           1           1
        11            hom A2        0           0

   Counts.n0/n1/n2/nmiss correspond exactly to PLINK2 genocounts[0..3].

   This header is deliberately freestanding (stdint/stddef only) so that
   src/bed_simd.cpp can be compiled for any target (e.g. a Zen 4 cross-check
   from macOS) without a target C++ standard library.
*/
#ifndef GCTA_BED_SIMD_HPP
#define GCTA_BED_SIMD_HPP

#include <stddef.h>
#include <stdint.h>

namespace bed_simd {

struct Counts {
    uint32_t n0    = 0;  ///< value 0  (PLINK1 '11', hom A2)  == genocounts[0]
    uint32_t n1    = 0;  ///< value 1  (PLINK1 '10', het)     == genocounts[1]
    uint32_t n2    = 0;  ///< value 2  (PLINK1 '00', hom A1)  == genocounts[2]
    uint32_t nmiss = 0;  ///< missing  (PLINK1 '01')          == genocounts[3]
};

/// Optional fused side-outputs of decode_i8().  All output positions / weights
/// are in *output* (kept-sample) index space.
struct I8Opts {
    // --- inputs --------------------------------------------------------
    /// Missing bitmask (bit s set <=> output sample s missing), ceil(n_keep/64)
    /// words.  Written (fully, zero-filled first) ONLY when nmiss > 0; left
    /// untouched otherwise, so the zero-missing fast path never touches it.
    uint64_t* miss_bits    = nullptr;
    /// Sparse missing indices (ascending). Up to miss_idx_cap are written;
    /// miss_idx_ct receives the total number of missing samples.
    uint32_t* miss_idx     = nullptr;
    uint32_t  miss_idx_cap = 0;
    /// Optional per-sample weights (length >= n_keep).
    const float*  wf = nullptr;
    const double* wd = nullptr;
    // --- outputs -------------------------------------------------------
    uint32_t miss_idx_ct = 0;
    double   wf_dot  = 0.0;  ///< sum_s g_s * wf_s       (missing contribute 0)
    double   wf_miss = 0.0;  ///< sum_{s missing} wf_s
    double   wd_dot  = 0.0;  ///< sum_s g_s * wd_s
    double   wd_dot2 = 0.0;  ///< sum_s g_s^2 * wd_s
    double   wd_miss = 0.0;  ///< sum_{s missing} wd_s
};

enum class Isa : int { Scalar = 0, Neon = 1, Avx2 = 2, Avx512 = 3 };

/// Name of the active kernel set ("avx512", "avx2", "neon", "scalar").
const char* isa_name();
Isa  active_isa();
/// Best ISA supported by this CPU/build.
Isa  best_isa();
/// Force a kernel set (testing / benchmarking). Returns false (and leaves the
/// current selection unchanged) when the ISA is not supported here.
/// The environment variable GCTA_BED_ISA=scalar|neon|avx2|avx512 is honoured
/// at first use as well.
bool set_isa(Isa isa);

/// Bytes in one raw .bed variant record for n_raw samples.
constexpr uint32_t record_bytes(uint32_t n_raw) { return (n_raw + 3u) / 4u; }

/// Number of uint64 words to allocate for a 1-bit-per-raw-sample keep mask
/// (rounded up to a 64-byte multiple; trailing bits must be zero).
constexpr size_t keep_mask_words(uint32_t n_raw) { return ((size_t(n_raw) + 511u) / 512u) * 8u; }

/// Build a 1-bit keep mask from strictly increasing raw indices.
/// words must hold keep_mask_words(n_raw) entries. Returns false when an index
/// is out of range or the indices are not strictly increasing.
bool build_keep_mask(const uint32_t* idx, size_t n_idx, uint32_t n_raw, uint64_t* words);

// ---------------------------------------------------------------------------
// Kernels.  `raw` points at one raw .bed variant record (record_bytes(n_raw)
// bytes are read; trailing bits / bytes beyond the record are never read or
// trusted).  `keep` (nullable) is a 1-bit mask over raw samples built by
// build_keep_mask(); keep == nullptr means all n_raw samples (n_keep == n_raw).
// ---------------------------------------------------------------------------

/// Genotype counts over the kept samples (AVX-512: VPOPCNTDQ + VPTERNLOGQ).
Counts count(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep);

/// Single-pass decode to int8 values {0,1,2} (missing -> 0) with simultaneous
/// counts and optional missing bitmask / indices / weighted sums.
/// AVX-512: VPERMB+VPMULTISHIFTQB expansion, VPSHUFB value map, VPCOMPRESSB
/// subsetting.  out must hold n_keep bytes (no slack required).
Counts decode_i8(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                 int8_t* out, I8Opts* opts = nullptr);

/// Register-table float decode: out[s] = table[value(s)] with
/// table = {v0, v1, v2, v_missing}.  miss_bits as in I8Opts (lazy).
/// AVX-512: VPCOMPRESSB on codes + VPMOVZXBD + VPERMPS (16 floats/iter).
Counts decode_f32(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                  const float table[4], float* out, uint64_t* miss_bits = nullptr);

/// Register-table double decode (AVX-512: VPMOVZXBQ + VPERMPD, 8 doubles/iter).
Counts decode_f64(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                  const double table[4], double* out, uint64_t* miss_bits = nullptr);

/// Range decode of kept samples [k, k + n_out) to int8, where r_begin is the
/// raw index of kept sample k (keep == nullptr: r_begin == k and r_begin must
/// be a multiple of 64).  Used for cache-blocked K-major GRM tiles.
void decode_i8_range(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                     uint32_t r_begin, uint32_t n_out, int8_t* out);

/// Int8 decode skipping redundant popcnt summary counts.
void decode_i8_nocount(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                       int8_t* out, I8Opts* opts = nullptr);

/// Decode K valid markers directly into K-major int8 layout for n_out kept samples
/// starting at raw sample index r_begin (must be % 4 == 0 when keep == nullptr):
///   kmajor_out[s * ldK + m] in {0, 1, 2} for s in [0, n_out), m in [0, K).
/// If mu != nullptr && r_out != nullptr, overwrites r_out[s] with
///   sum_{m=0..K-1} (double)kmajor_out[s * ldK + m] * mu[m] (in marker order).
void decode_i8_kmajor_band(const uint8_t* raw_base, size_t rec_bytes,
                           const int* valid_idx, int K,
                           const uint64_t* keep, uint32_t n_raw,
                           uint32_t r_begin, uint32_t n_out,
                           int8_t* kmajor_out, size_t ldK,
                           const double* mu, double* r_out);

}  // namespace bed_simd

#endif  // GCTA_BED_SIMD_HPP
