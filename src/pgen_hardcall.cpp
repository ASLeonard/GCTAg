/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   pgen_hardcall.cpp — Clean-room native PLINK2 .pgen hardcall index parser,
   compressed batch span reader, runtime-dispatched SIMD record decoders
   (AVX-512 / AVX2 / NEON / Scalar) into PLINK1 .bed 2-bit layout, and
   per-thread L2 decode cache.

   Excluded from unity builds so per-function __attribute__((target(...)))
   SIMD kernels stay strictly isolated.
*/

#include "pgen_hardcall.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#  define PGEN_X86 1
#  include <immintrin.h>
#else
#  define PGEN_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  define PGEN_ARM64 1
#  include <arm_neon.h>
#else
#  define PGEN_ARM64 0
#endif

#if PGEN_X86
#  define PGEN_T512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi,bmi2,popcnt")))
#  define PGEN_T256 __attribute__((target("avx2,bmi,bmi2,popcnt,fma")))
#  if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__) && \
      defined(__AVX512VBMI__) && defined(__AVX512VBMI2__) && defined(__AVX512VPOPCNTDQ__) && \
      defined(__BMI__) && defined(__BMI2__) && defined(__POPCNT__)
#    define PGEN_STATIC_AVX512 1
#  else
#    define PGEN_STATIC_AVX512 0
#  endif
#  if defined(__AVX2__) && defined(__BMI__) && defined(__BMI2__) && defined(__POPCNT__) && defined(__FMA__)
#    define PGEN_STATIC_AVX2 1
#  else
#    define PGEN_STATIC_AVX2 0
#  endif
#endif

namespace pgen_hardcall {
namespace {

constexpr uint32_t kPglVblockSize = 65536u;
constexpr uint32_t kPglDifflistGroupSize = 64u;
constexpr uint32_t kPglMaxDifflistLenDivisor = 8u;
constexpr uint32_t kPglMaxVariantCt = 0x7ffffffdu;
constexpr uint32_t kPglMaxSampleCt = 0x7ffffffeu;
constexpr uint32_t kPglMaxBytesPerVariant = 0xfffdffc0u;

inline constexpr std::size_t align_up_64(std::size_t n) noexcept {
    return (n + 63u) & ~std::size_t(63u);
}

inline constexpr uint32_t bed_record_bytes(uint32_t raw_sample_ct) noexcept {
    return (raw_sample_ct + 3u) / 4u;
}

inline uint16_t load_u16_le(const uint8_t* p) noexcept {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint32_t load_u32_le(const uint8_t* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint64_t load_u64_le(const uint8_t* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint32_t load_sub_u32_le(const uint8_t* p, uint32_t byte_ct) noexcept {
    switch (byte_ct) {
        case 1:
            return p[0];
        case 2:
            return load_u16_le(p);
        case 3:
            return static_cast<uint32_t>(p[0]) |
                   (static_cast<uint32_t>(load_u16_le(p + 1)) << 8u);
        default:
            return load_u32_le(p);
    }
}

// Exact clean-room match of PLINK2 BytesToRepresentNzU32(raw_sample_ct):
// 1 + floor(log2(uii)) / 8 for uii >= 1.
inline uint32_t bytes_to_represent_nz_u32(uint32_t uii) noexcept {
    const uint32_t bsr = 31u - static_cast<uint32_t>(__builtin_clz(uii));
    return 1u + (bsr / 8u);
}

inline void zero_trailing_bed_bits(uint32_t raw_sample_ct, uint8_t* dst_bed2) noexcept {
    const uint32_t rem = raw_sample_ct & 3u;
    if (rem != 0u) {
        const uint32_t rb = bed_record_bytes(raw_sample_ct);
        dst_bed2[rb - 1u] &= static_cast<uint8_t>((1u << (rem * 2u)) - 1u);
    }
}

inline void build_onebit_nibble_lut(uint8_t c0, uint8_t c1, uint8_t lut[16]) noexcept {
    for (uint32_t nib = 0; nib < 16u; ++nib) {
        const uint32_t b0 = (nib & 1u) ? c1 : c0;
        const uint32_t b1 = (nib & 2u) ? c1 : c0;
        const uint32_t b2 = (nib & 4u) ? c1 : c0;
        const uint32_t b3 = (nib & 8u) ? c1 : c0;
        lut[nib] = static_cast<uint8_t>(b0 | (b1 << 2u) | (b2 << 4u) | (b3 << 6u));
    }
}

// ===========================================================================
// Scalar kernels
// ===========================================================================
void p2_to_p1_bytes_scalar(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept {
    constexpr uint64_t kM55 = 0x5555555555555555ULL;
    std::size_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        const uint64_t w = load_u64_le(src + i);
        const uint64_t out = ~w ^ ((w >> 1u) & kM55);
        std::memcpy(dst + i, &out, 8u);
    }
    for (; i < n; ++i) {
        const uint8_t b = src[i];
        dst[i] = static_cast<uint8_t>(~b ^ ((b >> 1u) & 0x55u));
    }
}

void invert_hom_p1_bytes_scalar(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept {
    constexpr uint64_t kM55 = 0x5555555555555555ULL;
    std::size_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        const uint64_t w = load_u64_le(src + i);
        const uint64_t eq = ~(w ^ (w >> 1u)) & kM55;
        const uint64_t out = w ^ (eq * 3ULL);
        std::memcpy(dst + i, &out, 8u);
    }
    for (; i < n; ++i) {
        const uint8_t b = src[i];
        const uint8_t eq = static_cast<uint8_t>(~(b ^ (b >> 1u)) & 0x55u);
        dst[i] = static_cast<uint8_t>(b ^ (eq * 3u));
    }
}

inline void expand_onebit_tail_scalar(const uint8_t* bit_ptr,
                                      std::size_t start_bit_byte,
                                      std::size_t bed_rec_bytes,
                                      const uint8_t lut[16],
                                      uint8_t* dst_bed2) noexcept {
    const std::size_t full_pairs = bed_rec_bytes >> 1u;
    for (std::size_t i = start_bit_byte; i < full_pairs; ++i) {
        const uint8_t b = bit_ptr[i];
        dst_bed2[2u * i] = lut[b & 0x0Fu];
        dst_bed2[2u * i + 1u] = lut[b >> 4u];
    }
    if ((bed_rec_bytes & 1u) != 0u && start_bit_byte <= full_pairs) {
        dst_bed2[bed_rec_bytes - 1u] = lut[bit_ptr[full_pairs] & 0x0Fu];
    }
}

void expand_onebit_to_p1_scalar(const uint8_t* bit_ptr,
                                uint32_t raw_sample_ct,
                                uint8_t c0,
                                uint8_t c1,
                                uint8_t* dst_bed2) noexcept {
    uint8_t lut[16];
    build_onebit_nibble_lut(c0, c1, lut);
    expand_onebit_tail_scalar(bit_ptr, 0u, bed_record_bytes(raw_sample_ct), lut, dst_bed2);
}

// ===========================================================================
// ARM64 NEON kernels
// ===========================================================================
#if PGEN_ARM64
void p2_to_p1_bytes_neon(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept {
    const uint8x16_t m55 = vdupq_n_u8(0x55u);
    std::size_t i = 0;
    for (; i + 32u <= n; i += 32u) {
        const uint8x16_t w0 = vld1q_u8(src + i);
        const uint8x16_t w1 = vld1q_u8(src + i + 16u);
        const uint8x16_t r0 = veorq_u8(vmvnq_u8(w0), vandq_u8(vshrq_n_u8(w0, 1), m55));
        const uint8x16_t r1 = veorq_u8(vmvnq_u8(w1), vandq_u8(vshrq_n_u8(w1, 1), m55));
        vst1q_u8(dst + i, r0);
        vst1q_u8(dst + i + 16u, r1);
    }
    if (i + 16u <= n) {
        const uint8x16_t w0 = vld1q_u8(src + i);
        const uint8x16_t r0 = veorq_u8(vmvnq_u8(w0), vandq_u8(vshrq_n_u8(w0, 1), m55));
        vst1q_u8(dst + i, r0);
        i += 16u;
    }
    if (i < n) {
        p2_to_p1_bytes_scalar(src + i, dst + i, n - i);
    }
}

void invert_hom_p1_bytes_neon(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept {
    const uint8x16_t m55 = vdupq_n_u8(0x55u);
    std::size_t i = 0;
    for (; i + 32u <= n; i += 32u) {
        const uint8x16_t w0 = vld1q_u8(src + i);
        const uint8x16_t w1 = vld1q_u8(src + i + 16u);
        const uint8x16_t eq0 = vandq_u8(vmvnq_u8(veorq_u8(w0, vshrq_n_u8(w0, 1))), m55);
        const uint8x16_t eq1 = vandq_u8(vmvnq_u8(veorq_u8(w1, vshrq_n_u8(w1, 1))), m55);
        const uint8x16_t r0 = veorq_u8(w0, vorrq_u8(eq0, vshlq_n_u8(eq0, 1)));
        const uint8x16_t r1 = veorq_u8(w1, vorrq_u8(eq1, vshlq_n_u8(eq1, 1)));
        vst1q_u8(dst + i, r0);
        vst1q_u8(dst + i + 16u, r1);
    }
    if (i + 16u <= n) {
        const uint8x16_t w0 = vld1q_u8(src + i);
        const uint8x16_t eq0 = vandq_u8(vmvnq_u8(veorq_u8(w0, vshrq_n_u8(w0, 1))), m55);
        const uint8x16_t r0 = veorq_u8(w0, vorrq_u8(eq0, vshlq_n_u8(eq0, 1)));
        vst1q_u8(dst + i, r0);
        i += 16u;
    }
    if (i < n) {
        invert_hom_p1_bytes_scalar(src + i, dst + i, n - i);
    }
}

void expand_onebit_to_p1_neon(const uint8_t* bit_ptr,
                              uint32_t raw_sample_ct,
                              uint8_t c0,
                              uint8_t c1,
                              uint8_t* dst_bed2) noexcept {
    alignas(16) uint8_t lut[16];
    build_onebit_nibble_lut(c0, c1, lut);
    const std::size_t rb = bed_record_bytes(raw_sample_ct);
    const uint8x16_t lut_v = vld1q_u8(lut);
    const uint8x16_t m0f = vdupq_n_u8(0x0Fu);

    std::size_t i = 0;
    for (; 2u * (i + 16u) <= rb; i += 16u) {
        const uint8x16_t v = vld1q_u8(bit_ptr + i);
        // vtbl1q_u8 / vqtbl1q_u8 16-entry nibble LUT expansion
        const uint8x16_t ev = vqtbl1q_u8(lut_v, vandq_u8(v, m0f));
        const uint8x16_t od = vqtbl1q_u8(lut_v, vshrq_n_u8(v, 4));
        vst1q_u8(dst_bed2 + 2u * i, vzip1q_u8(ev, od));
        vst1q_u8(dst_bed2 + 2u * i + 16u, vzip2q_u8(ev, od));
    }
    expand_onebit_tail_scalar(bit_ptr, i, rb, lut, dst_bed2);
}
#endif  // PGEN_ARM64

// ===========================================================================
// x86_64 AVX2 & AVX-512 kernels
// ===========================================================================
#if PGEN_X86
PGEN_T256 void p2_to_p1_bytes_avx2(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept {
    const __m256i m55 = _mm256_set1_epi8(0x55);
    const __m256i ones = _mm256_set1_epi32(-1);
    std::size_t i = 0;
    for (; i + 32u <= n; i += 32u) {
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        const __m256i nw = _mm256_xor_si256(w, ones);
        const __m256i sh = _mm256_and_si256(_mm256_srli_epi16(w, 1), m55);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_xor_si256(nw, sh));
    }
    if (i < n) {
        p2_to_p1_bytes_scalar(src + i, dst + i, n - i);
    }
}

PGEN_T256 void invert_hom_p1_bytes_avx2(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept {
    const __m256i m55 = _mm256_set1_epi8(0x55);
    std::size_t i = 0;
    for (; i + 32u <= n; i += 32u) {
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        const __m256i x = _mm256_xor_si256(w, _mm256_srli_epi16(w, 1));
        const __m256i eq = _mm256_andnot_si256(x, m55);
        const __m256i mask = _mm256_or_si256(eq, _mm256_slli_epi16(eq, 1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_xor_si256(w, mask));
    }
    if (i < n) {
        invert_hom_p1_bytes_scalar(src + i, dst + i, n - i);
    }
}

PGEN_T256 void expand_onebit_to_p1_avx2(const uint8_t* bit_ptr,
                                        uint32_t raw_sample_ct,
                                        uint8_t c0,
                                        uint8_t c1,
                                        uint8_t* dst_bed2) noexcept {
    alignas(16) uint8_t lut[16];
    build_onebit_nibble_lut(c0, c1, lut);
    const std::size_t rb = bed_record_bytes(raw_sample_ct);
    const __m256i lut256 = _mm256_broadcastsi128_si256(
        _mm_load_si128(reinterpret_cast<const __m128i*>(lut)));
    const __m256i m0f = _mm256_set1_epi8(0x0F);

    std::size_t i = 0;
    for (; 2u * (i + 32u) <= rb; i += 32u) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bit_ptr + i));
        const __m256i ev = _mm256_shuffle_epi8(lut256, _mm256_and_si256(v, m0f));
        const __m256i od = _mm256_shuffle_epi8(lut256, _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f));
        const __m256i ulo = _mm256_unpacklo_epi8(ev, od);
        const __m256i uhi = _mm256_unpackhi_epi8(ev, od);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bed2 + 2u * i),
                            _mm256_permute2x128_si256(ulo, uhi, 0x20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bed2 + 2u * i + 32u),
                            _mm256_permute2x128_si256(ulo, uhi, 0x31));
    }
    expand_onebit_tail_scalar(bit_ptr, i, rb, lut, dst_bed2);
}

struct InterleavePerm512 {
    uint8_t lo[64];
    uint8_t hi[64];
    constexpr InterleavePerm512() : lo{}, hi{} {
        for (uint32_t k = 0; k < 32u; ++k) {
            lo[2u * k] = static_cast<uint8_t>(k);
            lo[2u * k + 1u] = static_cast<uint8_t>(64u + k);
            hi[2u * k] = static_cast<uint8_t>(32u + k);
            hi[2u * k + 1u] = static_cast<uint8_t>(96u + k);
        }
    }
};
alignas(64) constexpr InterleavePerm512 kPerm512{};

PGEN_T512 void p2_to_p1_bytes_avx512(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept {
    const __m512i m55 = _mm512_set1_epi8(0x55);
    const __m512i ones = _mm512_set1_epi32(-1);
    std::size_t i = 0;
    for (; i + 64u <= n; i += 64u) {
        const __m512i w = _mm512_loadu_si512(src + i);
        const __m512i nw = _mm512_xor_si512(w, ones);
        const __m512i sh = _mm512_and_si512(_mm512_srli_epi16(w, 1), m55);
        _mm512_storeu_si512(dst + i, _mm512_xor_si512(nw, sh));
    }
    if (i < n) {
        p2_to_p1_bytes_avx2(src + i, dst + i, n - i);
    }
}

PGEN_T512 void invert_hom_p1_bytes_avx512(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept {
    const __m512i m55 = _mm512_set1_epi8(0x55);
    std::size_t i = 0;
    for (; i + 64u <= n; i += 64u) {
        const __m512i w = _mm512_loadu_si512(src + i);
        const __m512i x = _mm512_xor_si512(w, _mm512_srli_epi16(w, 1));
        const __m512i eq = _mm512_andnot_si512(x, m55);
        const __m512i mask = _mm512_or_si512(eq, _mm512_slli_epi16(eq, 1));
        _mm512_storeu_si512(dst + i, _mm512_xor_si512(w, mask));
    }
    if (i < n) {
        invert_hom_p1_bytes_avx2(src + i, dst + i, n - i);
    }
}

PGEN_T512 void expand_onebit_to_p1_avx512(const uint8_t* bit_ptr,
                                          uint32_t raw_sample_ct,
                                          uint8_t c0,
                                          uint8_t c1,
                                          uint8_t* dst_bed2) noexcept {
    alignas(16) uint8_t lut[16];
    build_onebit_nibble_lut(c0, c1, lut);
    const std::size_t rb = bed_record_bytes(raw_sample_ct);
    const __m512i lut512 = _mm512_broadcast_i32x4(
        _mm_load_si128(reinterpret_cast<const __m128i*>(lut)));
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    const __m512i perm_lo = _mm512_load_si512(kPerm512.lo);
    const __m512i perm_hi = _mm512_load_si512(kPerm512.hi);

    std::size_t i = 0;
    for (; 2u * (i + 64u) <= rb; i += 64u) {
        const __m512i v = _mm512_loadu_si512(bit_ptr + i);
        const __m512i ev = _mm512_shuffle_epi8(lut512, _mm512_and_si512(v, m0f));
        const __m512i od = _mm512_shuffle_epi8(lut512, _mm512_and_si512(_mm512_srli_epi16(v, 4), m0f));
        _mm512_storeu_si512(dst_bed2 + 2u * i, _mm512_permutex2var_epi8(ev, perm_lo, od));
        _mm512_storeu_si512(dst_bed2 + 2u * i + 64u, _mm512_permutex2var_epi8(ev, perm_hi, od));
    }
    if (2u * (i + 32u) <= rb) {
        const __m256i lut256 = _mm256_broadcastsi128_si256(
            _mm_load_si128(reinterpret_cast<const __m128i*>(lut)));
        const __m256i m0f256 = _mm256_set1_epi8(0x0F);
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bit_ptr + i));
        const __m256i ev = _mm256_shuffle_epi8(lut256, _mm256_and_si256(v, m0f256));
        const __m256i od = _mm256_shuffle_epi8(lut256, _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f256));
        const __m256i ulo = _mm256_unpacklo_epi8(ev, od);
        const __m256i uhi = _mm256_unpackhi_epi8(ev, od);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bed2 + 2u * i),
                            _mm256_permute2x128_si256(ulo, uhi, 0x20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bed2 + 2u * i + 32u),
                            _mm256_permute2x128_si256(ulo, uhi, 0x31));
        i += 32u;
    }
    expand_onebit_tail_scalar(bit_ptr, i, rb, lut, dst_bed2);
}

bool cpu_has_avx512() noexcept {
#if PGEN_STATIC_AVX512
    return true;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vbmi2") &&
           __builtin_cpu_supports("avx512vpopcntdq") && __builtin_cpu_supports("bmi") &&
           __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt");
#endif
}

bool cpu_has_avx2() noexcept {
#if PGEN_STATIC_AVX2
    return true;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi") &&
           __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt") &&
           __builtin_cpu_supports("fma");
#endif
}
#endif  // PGEN_X86

// ===========================================================================
// Runtime ISA dispatch table
// ===========================================================================
struct PgenKernels {
    Isa isa;
    void (*p2_to_p1)(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept;
    void (*invert_hom_p1)(const uint8_t* src, uint8_t* dst, std::size_t n) noexcept;
    void (*expand_onebit)(const uint8_t* bit_ptr, uint32_t raw_sample_ct,
                          uint8_t c0, uint8_t c1, uint8_t* dst_bed2) noexcept;
};

const PgenKernels kScalarKernels = {
    Isa::Scalar,
    &p2_to_p1_bytes_scalar,
    &invert_hom_p1_bytes_scalar,
    &expand_onebit_to_p1_scalar,
};

#if PGEN_ARM64
const PgenKernels kNeonKernels = {
    Isa::Neon,
    &p2_to_p1_bytes_neon,
    &invert_hom_p1_bytes_neon,
    &expand_onebit_to_p1_neon,
};
#endif

#if PGEN_X86
const PgenKernels kAvx2Kernels = {
    Isa::Avx2,
    &p2_to_p1_bytes_avx2,
    &invert_hom_p1_bytes_avx2,
    &expand_onebit_to_p1_avx2,
};

const PgenKernels kAvx512Kernels = {
    Isa::Avx512,
    &p2_to_p1_bytes_avx512,
    &invert_hom_p1_bytes_avx512,
    &expand_onebit_to_p1_avx512,
};
#endif

const PgenKernels* kernels_for(Isa isa) noexcept {
    switch (isa) {
#if PGEN_X86
        case Isa::Avx512:
            return cpu_has_avx512() ? &kAvx512Kernels : nullptr;
        case Isa::Avx2:
            return cpu_has_avx2() ? &kAvx2Kernels : nullptr;
#endif
#if PGEN_ARM64
        case Isa::Neon:
            return &kNeonKernels;
#endif
        case Isa::Scalar:
            return &kScalarKernels;
        default:
            return nullptr;
    }
}

Isa detect_best_isa() noexcept {
#if PGEN_X86
    if (cpu_has_avx512()) return Isa::Avx512;
    if (cpu_has_avx2()) return Isa::Avx2;
#endif
#if PGEN_ARM64
    return Isa::Neon;
#endif
    return Isa::Scalar;
}

bool str_eq_ci(const char* a, const char* b) noexcept {
    while (*a && *b) {
        char ca = *a;
        char cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb - 'A' + 'a');
        if (ca != cb) return false;
        ++a;
        ++b;
    }
    return *a == *b;
}

const PgenKernels* select_initial_kernels() noexcept {
    const PgenKernels* k = kernels_for(detect_best_isa());
    if (const char* e = std::getenv("GCTA_BED_ISA")) {
        const PgenKernels* f = nullptr;
        if (str_eq_ci(e, "scalar")) f = kernels_for(Isa::Scalar);
        else if (str_eq_ci(e, "neon")) f = kernels_for(Isa::Neon);
        else if (str_eq_ci(e, "avx2")) f = kernels_for(Isa::Avx2);
        else if (str_eq_ci(e, "avx512")) f = kernels_for(Isa::Avx512);
        if (f) k = f;
    }
    return k ? k : &kScalarKernels;
}

const PgenKernels* g_pgen_kernels = nullptr;

inline const PgenKernels& active_kernels() noexcept {
    const PgenKernels* k = __atomic_load_n(&g_pgen_kernels, __ATOMIC_ACQUIRE);
    if (__builtin_expect(k == nullptr, 0)) {
        k = select_initial_kernels();
        __atomic_store_n(&g_pgen_kernels, k, __ATOMIC_RELEASE);
    }
    return *k;
}

// ===========================================================================
// VINT31 & Difflist decoder
// ===========================================================================
inline uint32_t read_vint31_checked(const uint8_t*& p, const uint8_t* end) {
    if (p >= end) {
        throw std::runtime_error("PGEN decode error: truncated VINT31.");
    }
    uint32_t b = *p++;
    if (b <= 0x7Fu) {
        return b;
    }
    uint32_t val = b & 0x7Fu;
    for (uint32_t shift = 7u; shift <= 21u; shift += 7u) {
        if (p >= end) {
            throw std::runtime_error("PGEN decode error: truncated multi-byte VINT31.");
        }
        b = *p++;
        val |= (b & 0x7Fu) << shift;
        if (b <= 0x7Fu) {
            return val;
        }
    }
    if (p >= end) {
        throw std::runtime_error("PGEN decode error: truncated 5-byte VINT31.");
    }
    b = *p++;
    if (b > 0x07u) {
        throw std::runtime_error("PGEN decode error: VINT31 overflow (>= 2^31).");
    }
    val |= b << 28u;
    return val;
}

inline void set_bed2_sample(uint8_t* dst_bed2, uint32_t sample_idx, uint8_t p1_code) noexcept {
    const std::size_t byte_pos = sample_idx >> 2u;
    const uint32_t shift = (sample_idx & 3u) * 2u;
    dst_bed2[byte_pos] = static_cast<uint8_t>(
        (dst_bed2[byte_pos] & ~(3u << shift)) | (static_cast<uint32_t>(p1_code) << shift));
}

void parse_and_apply_difflist(const uint8_t*& cur,
                              const uint8_t* end,
                              uint32_t raw_sample_ct,
                              const uint8_t code_map[4],
                              uint8_t* dst_bed2) {
    const uint32_t difflist_len = read_vint31_checked(cur, end);
    if (difflist_len == 0u) {
        return;
    }
    if (difflist_len > raw_sample_ct / kPglMaxDifflistLenDivisor) {
        throw std::runtime_error("PGEN decode error: difflist_len exceeds raw_sample_ct / 8.");
    }

    const uint32_t group_ct = (difflist_len + kPglDifflistGroupSize - 1u) / kPglDifflistGroupSize;
    const uint32_t sample_id_byte_ct = bytes_to_represent_nz_u32(raw_sample_ct);
    const std::size_t difflist_index_bytes =
        static_cast<std::size_t>(group_ct) * (sample_id_byte_ct + 1u) - 1u;
    const std::size_t raregeno_bytes = (static_cast<std::size_t>(difflist_len) + 3u) / 4u;

    if (static_cast<std::size_t>(end - cur) < difflist_index_bytes ||
        static_cast<std::size_t>(end - cur) - difflist_index_bytes < raregeno_bytes) {
        throw std::runtime_error("PGEN decode error: truncated difflist header or raregeno payload.");
    }

    const uint8_t* const group_starts_ptr = cur;
    const uint8_t* const raregeno_ptr = cur + difflist_index_bytes;
    cur = raregeno_ptr + raregeno_bytes;

    uint64_t prev_sample_idx = 0;
    for (uint32_t g = 0; g < group_ct; ++g) {
        const uint32_t elem_base = g * kPglDifflistGroupSize;
        const uint32_t group_elems = std::min<uint32_t>(kPglDifflistGroupSize, difflist_len - elem_base);

        uint64_t sample_idx = load_sub_u32_le(
            group_starts_ptr + static_cast<std::size_t>(g) * sample_id_byte_ct,
            sample_id_byte_ct);
        if (sample_idx >= raw_sample_ct || (g > 0u && sample_idx <= prev_sample_idx)) {
            throw std::runtime_error("PGEN decode error: invalid difflist group start sample index.");
        }

        const uint8_t* rg_group = raregeno_ptr + (elem_base >> 2u);
        uint32_t rg_byte = rg_group[0];
        set_bed2_sample(dst_bed2, static_cast<uint32_t>(sample_idx), code_map[rg_byte & 3u]);

        for (uint32_t e = 1u; e < group_elems; ++e) {
            if ((e & 3u) == 0u) {
                rg_byte = rg_group[e >> 2u];
            } else {
                rg_byte >>= 2u;
            }
            const uint32_t delta = read_vint31_checked(cur, end);
            if (delta == 0u) {
                throw std::runtime_error("PGEN decode error: zero delta in difflist.");
            }
            sample_idx += delta;
            if (sample_idx >= raw_sample_ct) {
                throw std::runtime_error("PGEN decode error: difflist sample index out of range.");
            }
            set_bed2_sample(dst_bed2, static_cast<uint32_t>(sample_idx), code_map[rg_byte & 3u]);
        }
        prev_sample_idx = sample_idx;
    }
}

bool is_native_pgen_disabled_by_env() noexcept {
    const char* env = std::getenv("GCTA_DISABLE_NATIVE_PGEN");
    return (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0);
}

}  // namespace

// ===========================================================================
// Public ISA query / override API
// ===========================================================================
const char* isa_name() {
    switch (active_kernels().isa) {
        case Isa::Avx512:
            return "avx512";
        case Isa::Avx2:
            return "avx2";
        case Isa::Neon:
            return "neon";
        default:
            return "scalar";
    }
}

Isa active_isa() { return active_kernels().isa; }

Isa best_isa() { return detect_best_isa(); }

bool set_isa(Isa isa) {
    const PgenKernels* k = kernels_for(isa);
    if (!k) return false;
    __atomic_store_n(&g_pgen_kernels, k, __ATOMIC_RELEASE);
    return true;
}

// ===========================================================================
// PGEN / PGI Index Loader
// ===========================================================================
PgenFileIndex load_pgen_index(const std::string& path,
                              uint32_t expected_sample_ct,
                              uint32_t expected_variant_ct) {
    if (expected_variant_ct != UINT32_MAX &&
        (expected_variant_ct == 0u || expected_variant_ct > kPglMaxVariantCt)) {
        throw std::runtime_error("load_pgen_index: invalid expected_variant_ct for [" + path + "].");
    }
    if (expected_sample_ct != UINT32_MAX &&
        (expected_sample_ct == 0u || expected_sample_ct > kPglMaxSampleCt)) {
        throw std::runtime_error("load_pgen_index: invalid expected_sample_ct for [" + path + "].");
    }

    gcta_io::PosixUniqueFd pgen_fd = gcta_io::open_posix_readonly_throwing(path);
    const uint64_t pgen_fsize = gcta_io::stat_file_size_throwing(pgen_fd.get(), path);
    if (pgen_fsize < 3u) {
        throw std::runtime_error("[" + path + "] is too small to be a valid .pgen file.");
    }

    uint8_t magic[3] = {0, 0, 0};
    gcta_io::pread_full_throwing(pgen_fd.get(), magic, 3u, 0u, path);
    if (magic[0] != 0x6cu || magic[1] != 0x1bu) {
        throw std::runtime_error("[" + path + "] is not a .pgen file (bad magic number).");
    }

    const uint8_t mode = magic[2];
    PgenFileIndex idx;
    idx.path = path;
    idx.file_mode = mode;

    // Mode 0x00: PLINK1 sample-major .bed (unsupported)
    if (mode == 0x00u) {
        throw std::runtime_error("[" + path + "] is a sample-major PLINK1 .bed file (unsupported).");
    }

    // Mode 0x01: PLINK1 variant-major .bed (e.g. --bpfile)
    if (mode == 0x01u) {
        if (expected_sample_ct == UINT32_MAX) {
            throw std::runtime_error(
                "load_pgen_index: expected_sample_ct is required for PLINK1 .bed mode (0x01) in [" + path + "].");
        }
        const uint32_t rb = bed_record_bytes(expected_sample_ct);
        uint32_t raw_m = expected_variant_ct;
        if (raw_m == UINT32_MAX) {
            const uint64_t payload = pgen_fsize - 3u;
            const uint64_t q = payload / rb;
            if (q == 0u || q > kPglMaxVariantCt || q * rb + 3u != pgen_fsize) {
                throw std::runtime_error("[" + path + "] has unexpected PLINK1 .bed file size.");
            }
            raw_m = static_cast<uint32_t>(q);
        } else {
            const uint64_t expected_fsize = 3ULL + static_cast<uint64_t>(raw_m) * rb;
            if (pgen_fsize != expected_fsize) {
                throw std::runtime_error("[" + path + "] has unexpected PLINK1 .bed file size.");
            }
        }
        idx.raw_variant_ct = raw_m;
        idx.raw_sample_ct = expected_sample_ct;
        idx.bed_rec_bytes = rb;
        idx.is_const_width = true;
        idx.const_fpos_offset = 3u;
        idx.const_vrec_width = rb;
        idx.const_vrtype = kVrtypePlink1;
        idx.total_payload_bytes = static_cast<uint64_t>(raw_m) * rb;
        idx.avg_rec_bytes = static_cast<double>(rb);
        if (is_native_pgen_disabled_by_env()) {
            idx.native_hardcall_ok = false;
            idx.fallback_reason = "disabled_by_env";
        } else {
            idx.native_hardcall_ok = true;
        }
        return idx;
    }

    if ((mode & 0xfeu) == 0x30u) {
        throw std::runtime_error("[" + path + "] is a .pgen.pgi index file rather than a .pgen file.");
    }

    // Determine whether header is inline or in external <path>.pgi
    const bool is_pgi = ((mode & 0xfeu) == 0x20u);
    if (!is_pgi && !(mode >= 0x02u && mode <= 0x04u) && !(mode == 0x10u || mode == 0x11u)) {
        throw std::runtime_error("[" + path + "] uses an unsupported .pgen storage mode.");
    }

    gcta_io::PosixUniqueFd pgi_fd;
    std::string hdr_path = path;
    int hdr_fd = pgen_fd.get();
    uint64_t hdr_fsize = pgen_fsize;

    if (is_pgi) {
        hdr_path = path + ".pgi";
        pgi_fd = gcta_io::open_posix_readonly_throwing(hdr_path);
        hdr_fd = pgi_fd.get();
        hdr_fsize = gcta_io::stat_file_size_throwing(hdr_fd, hdr_path);
        if (hdr_fsize < 12u) {
            throw std::runtime_error("[" + hdr_path + "] is too small to be a valid .pgen.pgi file.");
        }
        uint8_t pgi_magic[3] = {0, 0, 0};
        gcta_io::pread_full_throwing(hdr_fd, pgi_magic, 3u, 0u, hdr_path);
        if (pgi_magic[0] != 0x6cu || pgi_magic[1] != 0x1bu || (pgi_magic[2] & 0xfeu) != 0x30u) {
            throw std::runtime_error("[" + hdr_path + "] is not a valid .pgen.pgi file (bad magic number).");
        }
    } else if (pgen_fsize < 12u) {
        throw std::runtime_error("[" + path + "] is too small to be a valid .pgen file.");
    }

    uint8_t hdr9[9];
    gcta_io::pread_full_throwing(hdr_fd, hdr9, 9u, 3u, hdr_path);
    const uint32_t raw_m = load_u32_le(hdr9);
    const uint32_t raw_n = load_u32_le(hdr9 + 4u);
    const uint8_t header_ctrl = hdr9[8];

    if (raw_m == 0u || raw_m > kPglMaxVariantCt) {
        throw std::runtime_error("Invalid variant count in [" + hdr_path + "].");
    }
    if (raw_n == 0u || raw_n > kPglMaxSampleCt) {
        throw std::runtime_error("Invalid sample count in [" + hdr_path + "].");
    }
    if (expected_variant_ct != UINT32_MAX && raw_m != expected_variant_ct) {
        throw std::runtime_error(
            "Variant count mismatch in [" + path + "]: header has " + std::to_string(raw_m) +
            ", expected " + std::to_string(expected_variant_ct) + ".");
    }
    if (expected_sample_ct != UINT32_MAX && raw_n != expected_sample_ct) {
        throw std::runtime_error(
            "Sample count mismatch in [" + path + "]: header has " + std::to_string(raw_n) +
            ", expected " + std::to_string(expected_sample_ct) + ".");
    }

    const uint32_t rb = bed_record_bytes(raw_n);
    idx.raw_variant_ct = raw_m;
    idx.raw_sample_ct = raw_n;
    idx.bed_rec_bytes = rb;

    const uint32_t nonref_flags_storage = (header_ctrl >> 6u) & 3u;

    // Fixed-width PLINK2 modes: 0x02 (hardcall), 0x03 (dosage), 0x04 (phased dosage)
    if (mode < 0x10u) {
        if ((header_ctrl & 0x3fu) != 0u) {
            throw std::runtime_error(
                "[" + path + "] has fixed-width mode byte with variable-width header_ctrl bits.");
        }
        uint64_t fpos_off = 12u;
        if (nonref_flags_storage == 3u) {
            fpos_off += (static_cast<uint64_t>(raw_m) + 7ULL) / 8ULL;
        }
        uint64_t rec_w = rb;
        uint16_t const_vt = 0u;
        if (mode == 0x03u) {
            const_vt = 0x40u;
            rec_w += static_cast<uint64_t>(raw_n) * 2ULL;
        } else if (mode == 0x04u) {
            const_vt = 0xc0u;
            rec_w += static_cast<uint64_t>(raw_n) * 4ULL;
        }
        if (rec_w > kPglMaxBytesPerVariant) {
            throw std::runtime_error("[" + path + "] variant record width exceeds PGEN limit.");
        }
        const uint64_t expected_fsize = fpos_off + static_cast<uint64_t>(raw_m) * rec_w;
        if (pgen_fsize != expected_fsize) {
            throw std::runtime_error("[" + path + "] has unexpected .pgen file size.");
        }

        idx.is_const_width = true;
        idx.const_fpos_offset = fpos_off;
        idx.const_vrec_width = static_cast<uint32_t>(rec_w);
        idx.const_vrtype = const_vt;
        idx.total_payload_bytes = static_cast<uint64_t>(raw_m) * rec_w;
        idx.avg_rec_bytes = static_cast<double>(rec_w);
        if (mode == 0x02u) {
            if (is_native_pgen_disabled_by_env()) {
                idx.native_hardcall_ok = false;
                idx.fallback_reason = "disabled_by_env";
            } else {
                idx.native_hardcall_ok = true;
            }
        } else {
            idx.native_hardcall_ok = false;
            idx.fallback_reason = (mode == 0x03u) ? "dosage_mode_0x03" : "phased_dosage_mode_0x04";
        }
        return idx;
    }

    // Variable-width PLINK2 modes: 0x10, 0x11, 0x20, 0x21
    const uint32_t vrtype_and_fpos_storage = header_ctrl & 15u;
    const uint32_t alt_allele_ct_byte_ct = (header_ctrl >> 4u) & 3u;
    if (alt_allele_ct_byte_ct > 1u) {
        throw std::runtime_error("[" + hdr_path + "] uses >1-byte allele counts (>254 alt alleles unsupported).");
    }
    if (vrtype_and_fpos_storage == 13u || vrtype_and_fpos_storage == 15u) {
        throw std::runtime_error("[" + hdr_path + "] uses reserved PGEN header_ctrl encoding.");
    }
    if ((vrtype_and_fpos_storage == 12u || vrtype_and_fpos_storage == 14u) && raw_n != 1u) {
        throw std::runtime_error("[" + hdr_path + "] uses single-sample fused dosage header_ctrl with N != 1.");
    }

    const uint32_t vblock_ct = (raw_m + kPglVblockSize - 1u) / kPglVblockSize;

    // Compute exact base header byte size from (raw_m, header_ctrl) so we can read
    // the entire index table in a single pread_full_throwing() syscall.
    auto vblock_meta_bytes = [&](uint32_t cnt) -> uint64_t {
        uint64_t bytes = 0;
        if (vrtype_and_fpos_storage >= 8u) {
            uint32_t log2_bw = 1u;
            if (vrtype_and_fpos_storage == 8u) log2_bw = 0u;
            else if (vrtype_and_fpos_storage == 11u || vrtype_and_fpos_storage == 14u) log2_bw = 2u;
            bytes += 1ULL + ((cnt - 1u) >> (3u - log2_bw));
        } else {
            bytes += (vrtype_and_fpos_storage < 4u) ? ((cnt + 1u) / 2u) : cnt;
            bytes += static_cast<uint64_t>(cnt) * (1ULL + (vrtype_and_fpos_storage & 3u));
        }
        if (alt_allele_ct_byte_ct != 0u) {
            bytes += static_cast<uint64_t>(cnt) * alt_allele_ct_byte_ct;
        }
        if (nonref_flags_storage == 3u) {
            bytes += (cnt + 7u) / 8u;
        }
        return bytes;
    };

    uint64_t hdr_body_bytes = static_cast<uint64_t>(vblock_ct) * sizeof(uint64_t);
    if (vblock_ct > 1u) {
        hdr_body_bytes += static_cast<uint64_t>(vblock_ct - 1u) * vblock_meta_bytes(kPglVblockSize);
    }
    const uint32_t last_vblock_cnt = raw_m - (vblock_ct - 1u) * kPglVblockSize;
    hdr_body_bytes += vblock_meta_bytes(last_vblock_cnt);

    if (12ULL + hdr_body_bytes > hdr_fsize) {
        throw std::runtime_error("[" + hdr_path + "] is truncated inside the PGEN index header.");
    }

    AlignedRawBuffer<uint8_t, 64> hdr_buf(static_cast<std::size_t>(hdr_body_bytes) + 8u);
    gcta_io::pread_full_throwing(hdr_fd, hdr_buf.data(), static_cast<std::size_t>(hdr_body_bytes), 12u, hdr_path);

    const uint8_t* p = hdr_buf.data();
    uint64_t variant_fpos = load_u64_le(p);
    p += static_cast<std::size_t>(vblock_ct) * sizeof(uint64_t);

    if (!is_pgi) {
        if (12ULL + hdr_body_bytes > variant_fpos) {
            throw std::runtime_error("[" + path + "] header overlaps first variant fpos.");
        }
    } else if (variant_fpos < 3u) {
        throw std::runtime_error("[" + hdr_path + "] specifies invalid first variant fpos (< 3).");
    }
    if (variant_fpos > pgen_fsize) {
        throw std::runtime_error("[" + path + "] first variant fpos exceeds file size.");
    }

    // Prepare fused lookup tables when vrtype_and_fpos_storage >= 8
    uint32_t fused_log2_bw = 1u;
    std::array<uint8_t, 16> fused_vrtype{};
    std::array<uint32_t, 16> fused_vrec_len{};
    std::array<bool, 16> fused_valid{};
    if (vrtype_and_fpos_storage >= 8u) {
        fused_vrtype[0] = 5u;
        fused_vrec_len[0] = 0u;
        fused_valid[0] = true;
        fused_vrtype[1] = 0u;
        fused_vrec_len[1] = rb;
        fused_valid[1] = true;

        if (vrtype_and_fpos_storage == 8u) {
            fused_log2_bw = 0u;
        } else if (vrtype_and_fpos_storage == 9u) {
            fused_vrtype[2] = 8u;
            fused_vrec_len[2] = rb + 2u;
            fused_valid[2] = true;
            fused_vrtype[3] = 8u;
            fused_vrec_len[3] = rb + 3u;
            fused_valid[3] = true;
        } else if (vrtype_and_fpos_storage == 10u) {
            fused_vrtype[2] = 0x10u;
            fused_vrec_len[2] = rb + 1u;
            fused_valid[2] = true;
            fused_vrtype[3] = 0x10u;
            fused_vrec_len[3] = rb + 2u;
            fused_valid[3] = true;
        } else if (vrtype_and_fpos_storage == 11u) {
            fused_log2_bw = 2u;
            fused_vrtype[9] = 0x10u;
            fused_vrec_len[9] = rb + 1u;
            fused_valid[9] = true;
            for (uint32_t u = 2u; u < 8u; ++u) {
                fused_vrtype[u] = 8u;
                fused_vrec_len[u] = rb + u;
                fused_valid[u] = true;
                fused_vrtype[u + 8u] = 24u;
                fused_vrec_len[u + 8u] = rb + 1u + u;
                fused_valid[u + 8u] = true;
            }
        } else if (vrtype_and_fpos_storage == 12u) {
            fused_vrtype[2] = 0x45u;
            fused_vrec_len[2] = 2u;
            fused_valid[2] = true;
            fused_vrtype[3] = 0x40u;
            fused_vrec_len[3] = 3u;
            fused_valid[3] = true;
        } else if (vrtype_and_fpos_storage == 14u) {
            fused_log2_bw = 2u;
            fused_vrtype[3] = 0x10u;
            fused_vrec_len[3] = 2u;
            fused_valid[3] = true;
            fused_vrtype[4] = 0x45u;
            fused_vrec_len[4] = 2u;
            fused_valid[4] = true;
            fused_vrtype[5] = 0x40u;
            fused_vrec_len[5] = 3u;
            fused_valid[5] = true;
            fused_vrtype[12] = 0xc5u;
            fused_vrec_len[12] = 4u;
            fused_valid[12] = true;
            fused_vrtype[13] = 0xc0u;
            fused_vrec_len[13] = 5u;
            fused_valid[13] = true;
            fused_vrtype[15] = 0xe0u;
            fused_vrec_len[15] = 6u;
            fused_valid[15] = true;
        }
    }

    idx.var_fpos.resize(static_cast<std::size_t>(raw_m) + 1u);
    idx.vrtypes.resize(raw_m);

    uint8_t or_vt = 0u;
    bool has_non_biallelic = false;

    for (uint32_t vb = 0; vb < vblock_ct; ++vb) {
        const uint32_t v_start = vb * kPglVblockSize;
        const uint32_t cnt = (vb + 1u == vblock_ct) ? last_vblock_cnt : kPglVblockSize;
        uint8_t* vt_out = idx.vrtypes.data() + v_start;
        uint64_t* fp_out = idx.var_fpos.data() + v_start;

        if (vrtype_and_fpos_storage >= 8u) {
            const uint32_t entry_bw = 1u << fused_log2_bw;
            const uint32_t entry_mask = (1u << entry_bw) - 1u;
            const uint32_t entries_per_byte = 8u >> fused_log2_bw;
            const std::size_t byte_ct = 1u + ((cnt - 1u) >> (3u - fused_log2_bw));

            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t code = (p[i / entries_per_byte] >> ((i % entries_per_byte) * entry_bw)) & entry_mask;
                if (!fused_valid[code]) {
                    throw std::runtime_error("[" + hdr_path + "] contains invalid fused vrtype code.");
                }
                const uint8_t vt = fused_vrtype[code];
                vt_out[i] = vt;
                or_vt |= vt;
                fp_out[i] = variant_fpos;
                variant_fpos += fused_vrec_len[code];
            }
            p += byte_ct;
        } else {
            if (vrtype_and_fpos_storage < 4u) {
                const std::size_t vt_bytes = (cnt + 1u) / 2u;
                const uint32_t pairs = cnt / 2u;
                for (uint32_t i = 0; i < pairs; ++i) {
                    const uint8_t b = p[i];
                    const uint8_t lo = b & 0x0Fu;
                    const uint8_t hi = b >> 4u;
                    vt_out[2u * i] = lo;
                    vt_out[2u * i + 1u] = hi;
                    or_vt |= static_cast<uint8_t>(lo | hi);
                }
                if ((cnt & 1u) != 0u) {
                    const uint8_t lo = p[pairs] & 0x0Fu;
                    vt_out[cnt - 1u] = lo;
                    or_vt |= lo;
                }
                p += vt_bytes;
            } else {
                std::memcpy(vt_out, p, cnt);
                for (uint32_t i = 0; i < cnt; ++i) {
                    or_vt |= vt_out[i];
                }
                p += cnt;
            }

            if (PgenFileIndex::is_ld_vrtype(vt_out[0])) {
                throw std::runtime_error(
                    "[" + hdr_path + "] has invalid LD-compressed variant at start of 65536-variant vblock.");
            }

            const uint32_t len_bytes = 1u + (vrtype_and_fpos_storage & 3u);
            if (len_bytes == 1u) {
                for (uint32_t i = 0; i < cnt; ++i) {
                    fp_out[i] = variant_fpos;
                    variant_fpos += p[i];
                }
            } else if (len_bytes == 2u) {
                for (uint32_t i = 0; i < cnt; ++i) {
                    fp_out[i] = variant_fpos;
                    variant_fpos += load_u16_le(p + 2u * static_cast<std::size_t>(i));
                }
            } else if (len_bytes == 3u) {
                for (uint32_t i = 0; i < cnt; ++i) {
                    // hdr_buf has 8 extra tail bytes so load_u32_le is always in-bounds
                    const uint32_t len = load_u32_le(p + 3u * static_cast<std::size_t>(i)) & 0x00ffffffu;
                    fp_out[i] = variant_fpos;
                    variant_fpos += len;
                }
            } else {
                for (uint32_t i = 0; i < cnt; ++i) {
                    const uint32_t len = load_u32_le(p + 4u * static_cast<std::size_t>(i));
                    if (len > kPglMaxBytesPerVariant) {
                        throw std::runtime_error("[" + hdr_path + "] variant record exceeds max byte length.");
                    }
                    fp_out[i] = variant_fpos;
                    variant_fpos += len;
                }
            }
            p += static_cast<std::size_t>(cnt) * len_bytes;
        }

        if (alt_allele_ct_byte_ct != 0u) {
            for (uint32_t i = 0; i < cnt; ++i) {
                const uint8_t cur_allele_ct = p[i];
                if (cur_allele_ct == 0u) {
                    throw std::runtime_error("[" + hdr_path + "] contains variant with allele_ct == 0.");
                }
                if (cur_allele_ct != 2u) {
                    has_non_biallelic = true;
                }
            }
            p += cnt;
        }

        if (nonref_flags_storage == 3u) {
            p += (cnt + 7u) / 8u;
        }
    }

    idx.var_fpos[raw_m] = variant_fpos;
    if (variant_fpos > pgen_fsize) {
        throw std::runtime_error("[" + path + "] is truncated (variant records extend past end of file).");
    }
    if ((mode & 1u) == 0u && variant_fpos != pgen_fsize) {
        // Without extensions (mode 0x10 / 0x20), variant payload ends at EOF.
        throw std::runtime_error("[" + path + "] has unexpected trailing bytes past variant payload.");
    }

    idx.is_const_width = false;
    idx.total_payload_bytes = idx.var_fpos[raw_m] - idx.var_fpos[0];
    idx.avg_rec_bytes = static_cast<double>(idx.total_payload_bytes) / static_cast<double>(raw_m);

    if ((or_vt & 0xe0u) != 0u) {
        idx.native_hardcall_ok = false;
        idx.fallback_reason = "dosage_present";
    } else if ((or_vt & 0x08u) != 0u || has_non_biallelic) {
        idx.native_hardcall_ok = false;
        idx.fallback_reason = "multiallelic_present";
    } else if (is_native_pgen_disabled_by_env()) {
        idx.native_hardcall_ok = false;
        idx.fallback_reason = "disabled_by_env";
    } else {
        idx.native_hardcall_ok = true;
    }

    return idx;
}

// ===========================================================================
// CompressedBatchReader
// ===========================================================================
std::size_t CompressedBatchReader::build_and_read(
    int fd,
    const PgenFileIndex& index,
    std::span<const uint32_t> raw_vindices,
    uint64_t block_seq_id,
    uint8_t* (*resize_fn)(void* ctx, std::size_t total_bytes),
    void* resize_ctx) {
    const std::size_t B = raw_vindices.size();
    if (B == 0u) {
        uint8_t* const out = resize_fn(resize_ctx, sizeof(PgenBlockHeader));
        PgenBlockHeader hdr{};
        hdr.magic = kPgenBlockMagic;
        hdr.block_seq_id = block_seq_id;
        hdr.num_markers = 0u;
        hdr.raw_sample_ct = index.raw_sample_ct;
        hdr.bed_rec_bytes = index.bed_rec_bytes;
        std::memcpy(out, &hdr, sizeof(hdr));
        return sizeof(PgenBlockHeader);
    }

    req_vidx_.clear();
    req_vidx_.reserve(B + 16u);

    bool strictly_increasing = true;
    uint32_t prev_req = 0;
    for (std::size_t i = 0; i < B; ++i) {
        const uint32_t v = raw_vindices[i];
        if (v >= index.raw_variant_ct) {
            throw std::runtime_error(
                "CompressedBatchReader: raw_vidx out of range for [" + index.path + "].");
        }
        if (index.is_ld(v)) {
            const uint32_t base_v = index.ldbase_vidx(v);
            if (req_vidx_.empty() || base_v != req_vidx_.back()) {
                if (!req_vidx_.empty() && base_v < req_vidx_.back()) {
                    strictly_increasing = false;
                }
                req_vidx_.push_back(base_v);
                prev_req = base_v;
            }
        }
        if (req_vidx_.empty() || v != prev_req) {
            if (!req_vidx_.empty() && v < prev_req) {
                strictly_increasing = false;
            }
            req_vidx_.push_back(v);
            prev_req = v;
        }
    }

    if (!strictly_increasing) {
        std::sort(req_vidx_.begin(), req_vidx_.end());
        req_vidx_.erase(std::unique(req_vidx_.begin(), req_vidx_.end()), req_vidx_.end());
    }

    const std::size_t U = req_vidx_.size();
    req_buf_off_.resize(U);
    pending_slices_.clear();

    const std::size_t payload_start =
        align_up_64(sizeof(PgenBlockHeader) + B * sizeof(PgenBlockEntry));
    std::size_t cur_buf_off = payload_start;
    uint64_t total_payload_bytes = 0;

    bool have_slice = false;
    uint64_t slice_file_start = 0;
    uint64_t slice_file_end = 0;
    std::size_t slice_buf_start = 0;

    auto flush_slice = [&]() {
        if (have_slice && slice_file_end > slice_file_start) {
            pending_slices_.push_back({
                slice_file_start,
                static_cast<std::size_t>(slice_file_end - slice_file_start),
                slice_buf_start,
            });
        }
        have_slice = false;
    };

    for (std::size_t u = 0; u < U; ++u) {
        const uint32_t v = req_vidx_[u];
        const uint64_t f_off = index.fpos(v);
        const uint32_t f_len = index.vrec_len(v);

        if (f_len == 0u) {
            req_buf_off_[u] = 0u;
            continue;
        }

        if (!have_slice || f_off != slice_file_end) {
            flush_slice();
            cur_buf_off = align_up_64(cur_buf_off);
            slice_file_start = f_off;
            slice_file_end = f_off;
            slice_buf_start = cur_buf_off;
            have_slice = true;
        }
        req_buf_off_[u] = cur_buf_off;
        cur_buf_off += f_len;
        slice_file_end += f_len;
        total_payload_bytes += f_len;
    }
    flush_slice();

    const std::size_t total_bytes = std::max<std::size_t>(64u, align_up_64(cur_buf_off));
    uint8_t* const out = resize_fn(resize_ctx, total_bytes);

    PgenBlockHeader hdr{};
    hdr.magic = kPgenBlockMagic;
    hdr.block_seq_id = block_seq_id;
    hdr.num_markers = static_cast<uint32_t>(B);
    hdr.raw_sample_ct = index.raw_sample_ct;
    hdr.bed_rec_bytes = index.bed_rec_bytes;
    hdr.reserved = 0u;
    hdr.payload_bytes = total_payload_bytes;
    std::memcpy(out, &hdr, sizeof(hdr));

    auto* const entries = reinterpret_cast<PgenBlockEntry*>(out + sizeof(PgenBlockHeader));
    std::size_t u_hint = 0;

    auto find_req_slot = [&](uint32_t target_v) -> std::size_t {
        if (u_hint < U && req_vidx_[u_hint] == target_v) {
            return u_hint;
        }
        if (u_hint + 1u < U && req_vidx_[u_hint + 1u] == target_v) {
            return ++u_hint;
        }
        const auto it = std::lower_bound(req_vidx_.begin(), req_vidx_.end(), target_v);
        u_hint = static_cast<std::size_t>(it - req_vidx_.begin());
        return u_hint;
    };

    for (std::size_t i = 0; i < B; ++i) {
        const uint32_t v = raw_vindices[i];
        const uint16_t vt = index.vrtype(v);
        const std::size_t u_v = find_req_slot(v);

        PgenBlockEntry e{};
        e.rec_off = req_buf_off_[u_v];
        e.rec_len = index.vrec_len(v);
        e.raw_vidx = v;
        e.vrtype = vt;

        if (PgenFileIndex::is_ld_vrtype(vt)) {
            const uint32_t base_v = index.ldbase_vidx(v);
            std::size_t u_b = u_v;
            if (u_v > 0u && req_vidx_[u_v - 1u] == base_v) {
                u_b = u_v - 1u;
            } else {
                const auto it = std::lower_bound(req_vidx_.begin(), req_vidx_.begin() + u_v, base_v);
                u_b = static_cast<std::size_t>(it - req_vidx_.begin());
            }
            e.base_off = req_buf_off_[u_b];
            e.base_len = index.vrec_len(base_v);
            e.base_vrtype = index.vrtype(base_v);
        }
        entries[i] = e;
    }

    spans_.resize(pending_slices_.size());
    for (std::size_t s = 0; s < pending_slices_.size(); ++s) {
        spans_[s] = gcta_io::FileByteSpan{
            pending_slices_[s].file_off,
            pending_slices_[s].len,
            out + pending_slices_[s].buf_off,
        };
    }

    coalescer_.read_spans(fd, index.path, spans_);
    return total_bytes;
}

std::size_t CompressedBatchReader::read_batch(
    int fd,
    const PgenFileIndex& index,
    std::span<const uint32_t> raw_vindices,
    uint64_t block_seq_id,
    AlignedRawBuffer<uint8_t, 64>& dst_buf) {
    auto resize_u8 = +[](void* ctx, std::size_t total_bytes) -> uint8_t* {
        auto* buf = static_cast<AlignedRawBuffer<uint8_t, 64>*>(ctx);
        if (buf->size() < total_bytes) {
            buf->resize(total_bytes);
        }
        return buf->data();
    };
    return build_and_read(fd, index, raw_vindices, block_seq_id, resize_u8, &dst_buf);
}

std::size_t CompressedBatchReader::read_batch_uptr(
    int fd,
    const PgenFileIndex& index,
    std::span<const uint32_t> raw_vindices,
    uint64_t block_seq_id,
    AlignedRawBuffer<uintptr_t, 64>& dst_buf) {
    auto resize_uptr = +[](void* ctx, std::size_t total_bytes) -> uint8_t* {
        auto* buf = static_cast<AlignedRawBuffer<uintptr_t, 64>*>(ctx);
        const std::size_t elems = (total_bytes + sizeof(uintptr_t) - 1u) / sizeof(uintptr_t);
        if (buf->size() < elems) {
            buf->resize(elems);
        }
        return reinterpret_cast<uint8_t*>(buf->data());
    };
    return build_and_read(fd, index, raw_vindices, block_seq_id, resize_uptr, &dst_buf);
}

// ===========================================================================
// Record Decoders into PLINK1 .bed 2-bit layout
// ===========================================================================
void decode_non_ld_to_bed2(const uint8_t* rec_ptr,
                           std::size_t rec_len,
                           uint16_t vrtype,
                           uint32_t raw_sample_ct,
                           uint8_t* dst_bed2) {
    if (raw_sample_ct == 0u) return;
    const std::size_t rb = bed_record_bytes(raw_sample_ct);

    if (vrtype == kVrtypePlink1) {
        if (rec_len < rb) {
            throw std::runtime_error("PGEN decode error: truncated PLINK1 record.");
        }
        if (dst_bed2 != rec_ptr) {
            std::memcpy(dst_bed2, rec_ptr, rb);
        }
        zero_trailing_bed_bits(raw_sample_ct, dst_bed2);
        return;
    }

    if ((vrtype & ~uint16_t(0x17u)) != 0u) {
        throw std::runtime_error(
            "PGEN decode error: unsupported auxiliary vrtype bits in native hardcall decoder.");
    }

    const uint32_t main_vt = vrtype & 7u;
    const PgenKernels& K = active_kernels();

    switch (main_vt) {
        case 0u: {
            if (rec_len < rb) {
                throw std::runtime_error("PGEN decode error: truncated vrtype 0 record.");
            }
            K.p2_to_p1(rec_ptr, dst_bed2, rb);
            zero_trailing_bed_bits(raw_sample_ct, dst_bed2);
            return;
        }
        case 1u: {
            const std::size_t onebit_hdr_bytes = (static_cast<std::size_t>(raw_sample_ct) + 15u) / 8u;
            if (rec_len < onebit_hdr_bytes) {
                throw std::runtime_error("PGEN decode error: truncated vrtype 1 bitarray.");
            }
            const uint8_t common2_code = rec_ptr[0];
            const uint32_t p2_base = (common2_code >> 2u) & 3u;
            const uint32_t p2_delta = common2_code & 3u;
            if ((common2_code >> 4u) != 0u || p2_delta == 0u || (p2_base + p2_delta) > 3u) {
                throw std::runtime_error("PGEN decode error: invalid vrtype 1 common2_code.");
            }
            const uint8_t c0 = kP2ToP1[p2_base];
            const uint8_t c1 = kP2ToP1[p2_base + p2_delta];
            K.expand_onebit(rec_ptr + 1u, raw_sample_ct, c0, c1, dst_bed2);

            const uint8_t* cur = rec_ptr + onebit_hdr_bytes;
            const uint8_t* const end = rec_ptr + rec_len;
            parse_and_apply_difflist(cur, end, raw_sample_ct, kP2ToP1, dst_bed2);
            zero_trailing_bed_bits(raw_sample_ct, dst_bed2);
            return;
        }
        case 4u:
        case 6u:
        case 7u: {
            const uint8_t base_p1 = kP2ToP1[main_vt & 3u];
            std::memset(dst_bed2, static_cast<int>(base_p1 * 0x55u), rb);
            const uint8_t* cur = rec_ptr;
            const uint8_t* const end = rec_ptr + rec_len;
            parse_and_apply_difflist(cur, end, raw_sample_ct, kP2ToP1, dst_bed2);
            zero_trailing_bed_bits(raw_sample_ct, dst_bed2);
            return;
        }
        case 5u: {
            if ((vrtype & 0x10u) == 0u && rec_len != 0u) {
                throw std::runtime_error("PGEN decode error: non-empty unphased vrtype 5 record.");
            }
            std::memset(dst_bed2, 0xFF, rb);
            zero_trailing_bed_bits(raw_sample_ct, dst_bed2);
            return;
        }
        default:
            throw std::runtime_error(
                "PGEN decode error: LD-compressed vrtype passed to decode_non_ld_to_bed2.");
    }
}

void decode_ld_to_bed2(const uint8_t* rec_ptr,
                       std::size_t rec_len,
                       uint16_t vrtype,
                       uint32_t raw_sample_ct,
                       const uint8_t* ldbase_bed2,
                       uint8_t* dst_bed2) {
    if (raw_sample_ct == 0u) return;
    if (!ldbase_bed2) {
        throw std::runtime_error("PGEN decode error: null ldbase_bed2 in decode_ld_to_bed2.");
    }
    if ((vrtype & ~uint16_t(0x17u)) != 0u) {
        throw std::runtime_error(
            "PGEN decode error: unsupported auxiliary vrtype bits in decode_ld_to_bed2.");
    }

    const uint32_t main_vt = vrtype & 7u;
    const std::size_t rb = bed_record_bytes(raw_sample_ct);
    const uint8_t* cur = rec_ptr;
    const uint8_t* const end = rec_ptr + rec_len;

    if (main_vt == 2u) {
        if (dst_bed2 != ldbase_bed2) {
            std::memcpy(dst_bed2, ldbase_bed2, rb);
        }
        parse_and_apply_difflist(cur, end, raw_sample_ct, kP2ToP1, dst_bed2);
        zero_trailing_bed_bits(raw_sample_ct, dst_bed2);
        return;
    }

    if (main_vt == 3u) {
        active_kernels().invert_hom_p1(ldbase_bed2, dst_bed2, rb);
        parse_and_apply_difflist(cur, end, raw_sample_ct, kP2InvToP1, dst_bed2);
        zero_trailing_bed_bits(raw_sample_ct, dst_bed2);
        return;
    }

    throw std::runtime_error("PGEN decode error: non-LD vrtype passed to decode_ld_to_bed2.");
}

// ===========================================================================
// PgenThreadCache
// ===========================================================================
void PgenThreadCache::ensure_buffers(uint32_t bed_rec_bytes) {
    if (cached_bed_rec_bytes_ == bed_rec_bytes &&
        ldbase_buf_.size() >= bed_rec_bytes &&
        cur_buf_.size() >= bed_rec_bytes) {
        return;
    }
    const std::size_t cap = std::max<std::size_t>(64u, align_up_64(bed_rec_bytes));
    ldbase_buf_.resize(cap);
    cur_buf_.resize(cap);
    cached_bed_rec_bytes_ = bed_rec_bytes;
    ldbase_valid_ = false;
    ext_valid_ = false;
}

const uint8_t* PgenThreadCache::resolve(const void* block_buf, uint32_t idx) {
    if (!block_buf) {
        throw std::runtime_error("PgenThreadCache::resolve: null block_buf.");
    }
    const PgenBlockHeader* const hdr = block_header(block_buf);
    if (hdr->magic != kPgenBlockMagic || idx >= hdr->num_markers) {
        throw std::runtime_error("PgenThreadCache::resolve: invalid PgenBlockHeader or marker index.");
    }

    const PgenBlockEntry* const entries = block_entries(block_buf);
    const PgenBlockEntry& e = entries[idx];
    const auto* const base_bytes = static_cast<const uint8_t*>(block_buf);

    if (e.vrtype == kVrtypePlink1) {
        if (e.rec_len < hdr->bed_rec_bytes) {
            throw std::runtime_error("PgenThreadCache::resolve: truncated PLINK1 record.");
        }
        return base_bytes + e.rec_off;
    }

    ensure_buffers(hdr->bed_rec_bytes);

    if (!PgenFileIndex::is_ld_vrtype(e.vrtype)) {
        if (ldbase_valid_ &&
            ldbase_block_ptr_ == block_buf &&
            ldbase_seq_id_ == hdr->block_seq_id &&
            ldbase_rec_off_ == e.rec_off &&
            ldbase_rec_len_ == e.rec_len &&
            ldbase_vrtype_ == e.vrtype) {
            return ldbase_buf_.data();
        }

        const bool next_uses_as_base =
            (idx + 1u < hdr->num_markers) &&
            PgenFileIndex::is_ld_vrtype(entries[idx + 1u].vrtype) &&
            (entries[idx + 1u].base_off == e.rec_off) &&
            (entries[idx + 1u].base_len == e.rec_len) &&
            (entries[idx + 1u].base_vrtype == e.vrtype);

        const bool cache_stale =
            !ldbase_valid_ ||
            ldbase_block_ptr_ != block_buf ||
            ldbase_seq_id_ != hdr->block_seq_id;

        if (next_uses_as_base || cache_stale) {
            decode_non_ld_to_bed2(base_bytes + e.rec_off, e.rec_len, e.vrtype,
                                  hdr->raw_sample_ct, ldbase_buf_.data());
            ldbase_valid_ = true;
            ldbase_block_ptr_ = block_buf;
            ldbase_seq_id_ = hdr->block_seq_id;
            ldbase_rec_off_ = e.rec_off;
            ldbase_rec_len_ = e.rec_len;
            ldbase_vrtype_ = e.vrtype;
            return ldbase_buf_.data();
        }

        decode_non_ld_to_bed2(base_bytes + e.rec_off, e.rec_len, e.vrtype,
                              hdr->raw_sample_ct, cur_buf_.data());
        return cur_buf_.data();
    }

    // LD-compressed record
    const uint8_t* ldbase_ptr = nullptr;
    if (ldbase_valid_ &&
        ldbase_block_ptr_ == block_buf &&
        ldbase_seq_id_ == hdr->block_seq_id &&
        ldbase_rec_off_ == e.base_off &&
        ldbase_rec_len_ == e.base_len &&
        ldbase_vrtype_ == e.base_vrtype) {
        ldbase_ptr = ldbase_buf_.data();
    } else if (ext_valid_ &&
               ext_block_ptr_ == block_buf &&
               ext_seq_id_ == hdr->block_seq_id &&
               ext_rec_off_ == e.base_off &&
               ext_rec_len_ == e.base_len &&
               ext_vrtype_ == e.base_vrtype) {
        ldbase_ptr = ext_bed2_ptr_;
    } else {
        decode_non_ld_to_bed2(base_bytes + e.base_off, e.base_len, e.base_vrtype,
                              hdr->raw_sample_ct, ldbase_buf_.data());
        ldbase_valid_ = true;
        ldbase_block_ptr_ = block_buf;
        ldbase_seq_id_ = hdr->block_seq_id;
        ldbase_rec_off_ = e.base_off;
        ldbase_rec_len_ = e.base_len;
        ldbase_vrtype_ = e.base_vrtype;
        ldbase_ptr = ldbase_buf_.data();
    }

    decode_ld_to_bed2(base_bytes + e.rec_off, e.rec_len, e.vrtype,
                      hdr->raw_sample_ct, ldbase_ptr, cur_buf_.data());
    return cur_buf_.data();
}

void PgenThreadCache::decode_into(const void* block_buf, uint32_t idx, uint8_t* dst_bed2) {
    if (!block_buf || !dst_bed2) {
        throw std::runtime_error("PgenThreadCache::decode_into: null pointer.");
    }
    const PgenBlockHeader* const hdr = block_header(block_buf);
    if (hdr->magic != kPgenBlockMagic || idx >= hdr->num_markers) {
        throw std::runtime_error("PgenThreadCache::decode_into: invalid PgenBlockHeader or marker index.");
    }

    const std::size_t rb = hdr->bed_rec_bytes;
    ensure_buffers(hdr->bed_rec_bytes);

    const PgenBlockEntry* const entries = block_entries(block_buf);
    const PgenBlockEntry& e = entries[idx];
    const auto* const base_bytes = static_cast<const uint8_t*>(block_buf);

    const bool dst_overlaps_ext =
        ext_valid_ &&
        (reinterpret_cast<uintptr_t>(dst_bed2) < reinterpret_cast<uintptr_t>(ext_bed2_ptr_) + rb) &&
        (reinterpret_cast<uintptr_t>(ext_bed2_ptr_) < reinterpret_cast<uintptr_t>(dst_bed2) + rb);

    if (!PgenFileIndex::is_ld_vrtype(e.vrtype)) {
        if (dst_overlaps_ext) {
            ext_valid_ = false;
        }
        if (ldbase_valid_ &&
            ldbase_block_ptr_ == block_buf &&
            ldbase_seq_id_ == hdr->block_seq_id &&
            ldbase_rec_off_ == e.rec_off &&
            ldbase_rec_len_ == e.rec_len &&
            ldbase_vrtype_ == e.vrtype) {
            std::memcpy(dst_bed2, ldbase_buf_.data(), rb);
        } else {
            decode_non_ld_to_bed2(base_bytes + e.rec_off, e.rec_len, e.vrtype,
                                  hdr->raw_sample_ct, dst_bed2);
        }
        ext_valid_ = true;
        ext_block_ptr_ = block_buf;
        ext_seq_id_ = hdr->block_seq_id;
        ext_rec_off_ = e.rec_off;
        ext_rec_len_ = e.rec_len;
        ext_vrtype_ = e.vrtype;
        ext_bed2_ptr_ = dst_bed2;
        return;
    }

    // LD-compressed entry
    if (ext_valid_ &&
        ext_block_ptr_ == block_buf &&
        ext_seq_id_ == hdr->block_seq_id &&
        ext_rec_off_ == e.base_off &&
        ext_rec_len_ == e.base_len &&
        ext_vrtype_ == e.base_vrtype) {
        if (!dst_overlaps_ext) {
            decode_ld_to_bed2(base_bytes + e.rec_off, e.rec_len, e.vrtype,
                              hdr->raw_sample_ct, ext_bed2_ptr_, dst_bed2);
            return;
        }
        // Caller passed the same scratch buffer that held ext_bed2_ptr_: snapshot
        // into ldbase_buf_ before overwriting dst_bed2 so subsequent LD siblings
        // sharing e.base_off remain valid.
        std::memcpy(ldbase_buf_.data(), ext_bed2_ptr_, rb);
        ldbase_valid_ = true;
        ldbase_block_ptr_ = block_buf;
        ldbase_seq_id_ = hdr->block_seq_id;
        ldbase_rec_off_ = e.base_off;
        ldbase_rec_len_ = e.base_len;
        ldbase_vrtype_ = e.base_vrtype;
        ext_valid_ = false;
    } else {
        if (dst_overlaps_ext) {
            ext_valid_ = false;
        }
        if (!(ldbase_valid_ &&
              ldbase_block_ptr_ == block_buf &&
              ldbase_seq_id_ == hdr->block_seq_id &&
              ldbase_rec_off_ == e.base_off &&
              ldbase_rec_len_ == e.base_len &&
              ldbase_vrtype_ == e.base_vrtype)) {
            decode_non_ld_to_bed2(base_bytes + e.base_off, e.base_len, e.base_vrtype,
                                  hdr->raw_sample_ct, ldbase_buf_.data());
            ldbase_valid_ = true;
            ldbase_block_ptr_ = block_buf;
            ldbase_seq_id_ = hdr->block_seq_id;
            ldbase_rec_off_ = e.base_off;
            ldbase_rec_len_ = e.base_len;
            ldbase_vrtype_ = e.base_vrtype;
        }
    }

    decode_ld_to_bed2(base_bytes + e.rec_off, e.rec_len, e.vrtype,
                      hdr->raw_sample_ct, ldbase_buf_.data(), dst_bed2);
}

}  // namespace pgen_hardcall
