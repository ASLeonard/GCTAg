// -----------------------------------------------------------------------------
// avx512_emu.hpp -- portable scalar emulation of the AVX-512 intrinsics used by
// src/bed_simd_avx512.inc (exact Intel SDM semantics), so the kernel logic can
// be validated bit-for-bit on non-x86 hosts (e.g. Apple Silicon).
//
// Everything lives in namespace avx512_emu.  Because the .inc refers to the
// intrinsics unqualified, import them into the namespace that includes the .inc
// with AVX512_EMU_IMPORT (a list of using-declarations).  Those hide any
// same-named global x86 intrinsics/types (e.g. <emmintrin.h> pulled in by
// libstdc++'s <random> under -msse3, or a real <immintrin.h> under
// -march=znver4), so this header is safe to use on any host / -march.
// Masked loads/stores touch only the selected elements (ASan-clean).
// -----------------------------------------------------------------------------
#ifndef GCTA_AVX512_EMU_HPP
#define GCTA_AVX512_EMU_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

// Intrinsics that x86 headers may define as function-like macros
// (e.g. clang's _mm_srli_si128, _mm512_cmpeq_epi8_mask).
#ifdef _mm256_maskz_loadu_epi8
#  undef _mm256_maskz_loadu_epi8
#endif
#ifdef _mm512_add_epi64
#  undef _mm512_add_epi64
#endif
#ifdef _mm512_add_pd
#  undef _mm512_add_pd
#endif
#ifdef _mm512_and_si512
#  undef _mm512_and_si512
#endif
#ifdef _mm512_broadcast_i32x4
#  undef _mm512_broadcast_i32x4
#endif
#ifdef _mm512_castps512_ps256
#  undef _mm512_castps512_ps256
#endif
#ifdef _mm512_cmpeq_epi8_mask
#  undef _mm512_cmpeq_epi8_mask
#endif
#ifdef _mm512_cvtepu8_epi16
#  undef _mm512_cvtepu8_epi16
#endif
#ifdef _mm512_cvtepu8_epi32
#  undef _mm512_cvtepu8_epi32
#endif
#ifdef _mm512_cvtepu8_epi64
#  undef _mm512_cvtepu8_epi64
#endif
#ifdef _mm512_cvtps_pd
#  undef _mm512_cvtps_pd
#endif
#ifdef _mm512_extractf32x8_ps
#  undef _mm512_extractf32x8_ps
#endif
#ifdef _mm512_extracti32x4_epi32
#  undef _mm512_extracti32x4_epi32
#endif
#ifdef _mm512_fmadd_pd
#  undef _mm512_fmadd_pd
#endif
#ifdef _mm512_load_pd
#  undef _mm512_load_pd
#endif
#ifdef _mm512_load_ps
#  undef _mm512_load_ps
#endif
#ifdef _mm512_load_si512
#  undef _mm512_load_si512
#endif
#ifdef _mm512_mask_add_ps
#  undef _mm512_mask_add_ps
#endif
#ifdef _mm512_mask_storeu_epi8
#  undef _mm512_mask_storeu_epi8
#endif
#ifdef _mm512_mask_storeu_pd
#  undef _mm512_mask_storeu_pd
#endif
#ifdef _mm512_mask_storeu_ps
#  undef _mm512_mask_storeu_ps
#endif
#ifdef _mm512_maskz_compress_epi8
#  undef _mm512_maskz_compress_epi8
#endif
#ifdef _mm512_maskz_loadu_epi8
#  undef _mm512_maskz_loadu_epi8
#endif
#ifdef _mm512_maskz_loadu_pd
#  undef _mm512_maskz_loadu_pd
#endif
#ifdef _mm512_maskz_loadu_ps
#  undef _mm512_maskz_loadu_ps
#endif
#ifdef _mm512_maskz_mov_pd
#  undef _mm512_maskz_mov_pd
#endif
#ifdef _mm512_multishift_epi64_epi8
#  undef _mm512_multishift_epi64_epi8
#endif
#ifdef _mm512_or_si512
#  undef _mm512_or_si512
#endif
#ifdef _mm512_permutexvar_epi8
#  undef _mm512_permutexvar_epi8
#endif
#ifdef _mm512_permutexvar_pd
#  undef _mm512_permutexvar_pd
#endif
#ifdef _mm512_permutexvar_ps
#  undef _mm512_permutexvar_ps
#endif
#ifdef _mm512_popcnt_epi64
#  undef _mm512_popcnt_epi64
#endif
#ifdef _mm512_reduce_add_epi64
#  undef _mm512_reduce_add_epi64
#endif
#ifdef _mm512_reduce_add_pd
#  undef _mm512_reduce_add_pd
#endif
#ifdef _mm512_set1_epi16
#  undef _mm512_set1_epi16
#endif
#ifdef _mm512_set1_epi8
#  undef _mm512_set1_epi8
#endif
#ifdef _mm512_set1_pd
#  undef _mm512_set1_pd
#endif
#ifdef _mm512_setzero_pd
#  undef _mm512_setzero_pd
#endif
#ifdef _mm512_setzero_si512
#  undef _mm512_setzero_si512
#endif
#ifdef _mm512_shuffle_epi8
#  undef _mm512_shuffle_epi8
#endif
#ifdef _mm512_slli_epi16
#  undef _mm512_slli_epi16
#endif
#ifdef _mm512_srli_epi64
#  undef _mm512_srli_epi64
#endif
#ifdef _mm512_ternarylogic_epi64
#  undef _mm512_ternarylogic_epi64
#endif
#ifdef _mm512_testn_epi8_mask
#  undef _mm512_testn_epi8_mask
#endif
#ifdef _mm512_zextsi128_si512
#  undef _mm512_zextsi128_si512
#endif
#ifdef _mm_loadu_si128
#  undef _mm_loadu_si128
#endif
#ifdef _mm_maskz_loadu_epi8
#  undef _mm_maskz_loadu_epi8
#endif
#ifdef _mm_srli_si128
#  undef _mm_srli_si128
#endif
#ifdef _pdep_u64
#  undef _pdep_u64
#endif

namespace avx512_emu {

struct __m128i { alignas(16) uint8_t b[16]; };
struct __m256i { alignas(32) uint8_t b[32]; };
struct __m512i { alignas(64) uint8_t b[64]; };
struct __m256  { alignas(32) float f[8]; };
struct __m512  { alignas(64) float f[16]; };
struct __m512d { alignas(64) double d[8]; };
typedef uint8_t  __mmask8;
typedef uint16_t __mmask16;
typedef uint32_t __mmask32;
typedef uint64_t __mmask64;

namespace detail {
template <class T, class V> inline T lane(const V& v, int i) {
    T x; std::memcpy(&x, reinterpret_cast<const uint8_t*>(&v) + i * sizeof(T), sizeof(T)); return x;
}
template <class T, class V> inline void set_lane(V& v, int i, T x) {
    std::memcpy(reinterpret_cast<uint8_t*>(&v) + i * sizeof(T), &x, sizeof(T));
}
inline void check_align64(const void* p) {
    assert((reinterpret_cast<uintptr_t>(p) & 63u) == 0 && "aligned 512-bit load on unaligned address");
    (void)p;
}
}  // namespace detail

// ---- 128 / 256-bit ----------------------------------------------------------
inline __m128i _mm_loadu_si128(const __m128i* p) { __m128i r; std::memcpy(r.b, p, 16); return r; }
inline __m128i _mm_maskz_loadu_epi8(__mmask16 k, const void* p) {
    __m128i r{}; const uint8_t* s = static_cast<const uint8_t*>(p);
    for (int i = 0; i < 16; ++i) if ((k >> i) & 1u) r.b[i] = s[i];
    return r;
}
inline __m128i _mm_srli_si128(__m128i a, int imm) {
    __m128i r{}; const unsigned n = (unsigned)imm & 0xFFu;
    for (unsigned i = 0; i < 16; ++i) r.b[i] = (i + n < 16) ? a.b[i + n] : 0;
    return r;
}
inline __m256i _mm256_maskz_loadu_epi8(__mmask32 k, const void* p) {
    __m256i r{}; const uint8_t* s = static_cast<const uint8_t*>(p);
    for (int i = 0; i < 32; ++i) if ((k >> i) & 1u) r.b[i] = s[i];
    return r;
}

// ---- 512-bit integer --------------------------------------------------------
inline __m512i _mm512_setzero_si512() { return __m512i{}; }
inline __m512i _mm512_set1_epi8(char v) { __m512i r; std::memset(r.b, (uint8_t)v, 64); return r; }
inline __m512i _mm512_set1_epi16(short v) {
    __m512i r; for (int i = 0; i < 32; ++i) detail::set_lane<uint16_t>(r, i, (uint16_t)v); return r;
}
inline __m512i _mm512_load_si512(const void* p) {
    detail::check_align64(p); __m512i r; std::memcpy(r.b, p, 64); return r;
}
inline __m512i _mm512_maskz_loadu_epi8(__mmask64 k, const void* p) {
    __m512i r{}; const uint8_t* s = static_cast<const uint8_t*>(p);
    for (int i = 0; i < 64; ++i) if ((k >> i) & 1u) r.b[i] = s[i];
    return r;
}
inline void _mm512_mask_storeu_epi8(void* p, __mmask64 k, __m512i a) {
    uint8_t* d = static_cast<uint8_t*>(p);
    for (int i = 0; i < 64; ++i) if ((k >> i) & 1u) d[i] = a.b[i];
}
inline __m512i _mm512_zextsi128_si512(__m128i a) { __m512i r{}; std::memcpy(r.b, a.b, 16); return r; }
inline __m512i _mm512_broadcast_i32x4(__m128i a) {
    __m512i r; for (int l = 0; l < 4; ++l) std::memcpy(r.b + 16 * l, a.b, 16); return r;
}
inline __m128i _mm512_extracti32x4_epi32(__m512i a, int imm) {
    __m128i r; std::memcpy(r.b, a.b + 16 * (imm & 3), 16); return r;
}
inline __m512i _mm512_and_si512(__m512i a, __m512i b) { __m512i r; for (int i = 0; i < 64; ++i) r.b[i] = a.b[i] & b.b[i]; return r; }
inline __m512i _mm512_or_si512(__m512i a, __m512i b)  { __m512i r; for (int i = 0; i < 64; ++i) r.b[i] = a.b[i] | b.b[i]; return r; }

// VPERMB: dst.byte[i] = a.byte[idx.byte[i] & 63]
inline __m512i _mm512_permutexvar_epi8(__m512i idx, __m512i a) {
    __m512i r; for (int i = 0; i < 64; ++i) r.b[i] = a.b[idx.b[i] & 63u]; return r;
}
// VPMULTISHIFTQB: dst.qword[q].byte[j] = 8 bits of b.qword[q] starting at bit
// (a.qword[q].byte[j] & 63), wrapping around (rotate right).
inline __m512i _mm512_multishift_epi64_epi8(__m512i a, __m512i b) {
    __m512i r;
    for (int q = 0; q < 8; ++q) {
        const uint64_t src = detail::lane<uint64_t>(b, q);
        for (int j = 0; j < 8; ++j) {
            const unsigned ctl = a.b[8 * q + j] & 63u;
            uint8_t v = 0;
            for (unsigned k = 0; k < 8; ++k) v |= (uint8_t)(((src >> ((ctl + k) & 63u)) & 1u) << k);
            r.b[8 * q + j] = v;
        }
    }
    return r;
}
// VPSHUFB (per 128-bit lane; index bit 7 zeroes)
inline __m512i _mm512_shuffle_epi8(__m512i a, __m512i b) {
    __m512i r;
    for (int i = 0; i < 64; ++i) {
        const int base = i & ~15;
        r.b[i] = (b.b[i] & 0x80u) ? 0 : a.b[base + (b.b[i] & 15u)];
    }
    return r;
}
inline __mmask64 _mm512_cmpeq_epi8_mask(__m512i a, __m512i b) {
    __mmask64 m = 0; for (int i = 0; i < 64; ++i) m |= (uint64_t)(a.b[i] == b.b[i]) << i; return m;
}
inline __mmask64 _mm512_testn_epi8_mask(__m512i a, __m512i b) {
    __mmask64 m = 0; for (int i = 0; i < 64; ++i) m |= (uint64_t)((a.b[i] & b.b[i]) == 0) << i; return m;
}
// VPCOMPRESSB (zero-masking)
inline __m512i _mm512_maskz_compress_epi8(__mmask64 k, __m512i a) {
    __m512i r{}; int j = 0;
    for (int i = 0; i < 64; ++i) if ((k >> i) & 1u) r.b[j++] = a.b[i];
    return r;
}
inline __m512i _mm512_cvtepu8_epi16(__m256i a) {
    __m512i r; for (int i = 0; i < 32; ++i) detail::set_lane<uint16_t>(r, i, a.b[i]); return r;
}
inline __m512i _mm512_cvtepu8_epi32(__m128i a) {
    __m512i r; for (int i = 0; i < 16; ++i) detail::set_lane<uint32_t>(r, i, a.b[i]); return r;
}
inline __m512i _mm512_cvtepu8_epi64(__m128i a) {
    __m512i r; for (int i = 0; i < 8; ++i) detail::set_lane<uint64_t>(r, i, a.b[i]); return r;
}
inline __m512i _mm512_slli_epi16(__m512i a, unsigned int imm) {
    __m512i r;
    for (int i = 0; i < 32; ++i) {
        const uint16_t x = detail::lane<uint16_t>(a, i);
        detail::set_lane<uint16_t>(r, i, imm > 15 ? (uint16_t)0 : (uint16_t)(x << imm));
    }
    return r;
}
inline __m512i _mm512_srli_epi64(__m512i a, unsigned int imm) {
    __m512i r;
    for (int i = 0; i < 8; ++i) {
        const uint64_t x = detail::lane<uint64_t>(a, i);
        detail::set_lane<uint64_t>(r, i, imm > 63 ? 0ull : (x >> imm));
    }
    return r;
}
// VPTERNLOGQ: result bit = imm[(a<<2)|(b<<1)|c]
inline __m512i _mm512_ternarylogic_epi64(__m512i a, __m512i b, __m512i c, int imm) {
    __m512i r;
    for (int q = 0; q < 8; ++q) {
        const uint64_t A = detail::lane<uint64_t>(a, q), B = detail::lane<uint64_t>(b, q),
                       C = detail::lane<uint64_t>(c, q);
        uint64_t x = 0;
        for (int t = 0; t < 8; ++t) {
            if (!((imm >> t) & 1)) continue;
            x |= ((t & 4) ? A : ~A) & ((t & 2) ? B : ~B) & ((t & 1) ? C : ~C);
        }
        detail::set_lane<uint64_t>(r, q, x);
    }
    return r;
}
inline __m512i _mm512_popcnt_epi64(__m512i a) {
    __m512i r;
    for (int q = 0; q < 8; ++q)
        detail::set_lane<uint64_t>(r, q, (uint64_t)__builtin_popcountll(detail::lane<uint64_t>(a, q)));
    return r;
}
inline __m512i _mm512_add_epi64(__m512i a, __m512i b) {
    __m512i r;
    for (int q = 0; q < 8; ++q)
        detail::set_lane<uint64_t>(r, q, detail::lane<uint64_t>(a, q) + detail::lane<uint64_t>(b, q));
    return r;
}
inline long long _mm512_reduce_add_epi64(__m512i a) {
    uint64_t s = 0; for (int q = 0; q < 8; ++q) s += detail::lane<uint64_t>(a, q); return (long long)s;
}

// ---- 512-bit float / double -------------------------------------------------
inline __m512 _mm512_load_ps(const void* p) { detail::check_align64(p); __m512 r; std::memcpy(r.f, p, 64); return r; }
inline __m512d _mm512_load_pd(const void* p) { detail::check_align64(p); __m512d r; std::memcpy(r.d, p, 64); return r; }
inline __m512 _mm512_maskz_loadu_ps(__mmask16 k, const void* p) {
    __m512 r{}; const float* s = static_cast<const float*>(p);
    for (int i = 0; i < 16; ++i) if ((k >> i) & 1u) std::memcpy(&r.f[i], s + i, 4);
    return r;
}
inline __m512d _mm512_maskz_loadu_pd(__mmask8 k, const void* p) {
    __m512d r{}; const double* s = static_cast<const double*>(p);
    for (int i = 0; i < 8; ++i) if ((k >> i) & 1u) std::memcpy(&r.d[i], s + i, 8);
    return r;
}
inline void _mm512_mask_storeu_ps(void* p, __mmask16 k, __m512 a) {
    float* d = static_cast<float*>(p);
    for (int i = 0; i < 16; ++i) if ((k >> i) & 1u) std::memcpy(d + i, &a.f[i], 4);
}
inline void _mm512_mask_storeu_pd(void* p, __mmask8 k, __m512d a) {
    double* d = static_cast<double*>(p);
    for (int i = 0; i < 8; ++i) if ((k >> i) & 1u) std::memcpy(d + i, &a.d[i], 8);
}
// VPERMPS / VPERMPD (bit-exact copies)
inline __m512 _mm512_permutexvar_ps(__m512i idx, __m512 a) {
    __m512 r;
    for (int i = 0; i < 16; ++i) std::memcpy(&r.f[i], &a.f[detail::lane<uint32_t>(idx, i) & 15u], 4);
    return r;
}
inline __m512d _mm512_permutexvar_pd(__m512i idx, __m512d a) {
    __m512d r;
    for (int i = 0; i < 8; ++i) std::memcpy(&r.d[i], &a.d[detail::lane<uint64_t>(idx, i) & 7u], 8);
    return r;
}
inline __m512 _mm512_mask_add_ps(__m512 src, __mmask16 k, __m512 a, __m512 b) {
    __m512 r; for (int i = 0; i < 16; ++i) r.f[i] = ((k >> i) & 1u) ? a.f[i] + b.f[i] : src.f[i]; return r;
}
inline __m256 _mm512_castps512_ps256(__m512 a) { __m256 r; std::memcpy(r.f, a.f, 32); return r; }
inline __m256 _mm512_extractf32x8_ps(__m512 a, int imm) { __m256 r; std::memcpy(r.f, a.f + 8 * (imm & 1), 32); return r; }
inline __m512d _mm512_cvtps_pd(__m256 a) { __m512d r; for (int i = 0; i < 8; ++i) r.d[i] = (double)a.f[i]; return r; }
inline __m512d _mm512_setzero_pd() { return __m512d{}; }
inline __m512d _mm512_set1_pd(double v) { __m512d r; for (int i = 0; i < 8; ++i) r.d[i] = v; return r; }
inline __m512d _mm512_add_pd(__m512d a, __m512d b) { __m512d r; for (int i = 0; i < 8; ++i) r.d[i] = a.d[i] + b.d[i]; return r; }
inline __m512d _mm512_maskz_mov_pd(__mmask8 k, __m512d a) {
    __m512d r{}; for (int i = 0; i < 8; ++i) if ((k >> i) & 1u) r.d[i] = a.d[i]; return r;
}
inline __m512d _mm512_fmadd_pd(__m512d a, __m512d b, __m512d c) {
    __m512d r; for (int i = 0; i < 8; ++i) r.d[i] = std::fma(a.d[i], b.d[i], c.d[i]); return r;
}
// Tree reduction like clang's _mm512_reduce_add_pd (order not guaranteed to match HW).
inline double _mm512_reduce_add_pd(__m512d a) {
    double t4[4], t2[2];
    for (int i = 0; i < 4; ++i) t4[i] = a.d[i] + a.d[i + 4];
    for (int i = 0; i < 2; ++i) t2[i] = t4[i] + t4[i + 2];
    return t2[0] + t2[1];
}

// ---- BMI2 -------------------------------------------------------------------
inline uint64_t _pdep_u64(uint64_t src, uint64_t mask) {
    uint64_t r = 0;
    for (uint64_t bb = 1; mask; bb <<= 1) {
        const uint64_t low = mask & (~mask + 1);   // lowest set bit of mask
        if (src & bb) r |= low;
        mask &= mask - 1;
    }
    return r;
}

}  // namespace avx512_emu

// Import every emulated type / intrinsic into the current namespace.
#define AVX512_EMU_IMPORT \
    using ::avx512_emu::__m128i; \
    using ::avx512_emu::__m256i; \
    using ::avx512_emu::__m512i; \
    using ::avx512_emu::__m256; \
    using ::avx512_emu::__m512; \
    using ::avx512_emu::__m512d; \
    using ::avx512_emu::__mmask8; \
    using ::avx512_emu::__mmask16; \
    using ::avx512_emu::__mmask32; \
    using ::avx512_emu::__mmask64; \
    using ::avx512_emu::_mm256_maskz_loadu_epi8; \
    using ::avx512_emu::_mm512_add_epi64; \
    using ::avx512_emu::_mm512_add_pd; \
    using ::avx512_emu::_mm512_and_si512; \
    using ::avx512_emu::_mm512_broadcast_i32x4; \
    using ::avx512_emu::_mm512_castps512_ps256; \
    using ::avx512_emu::_mm512_cmpeq_epi8_mask; \
    using ::avx512_emu::_mm512_cvtepu8_epi16; \
    using ::avx512_emu::_mm512_cvtepu8_epi32; \
    using ::avx512_emu::_mm512_cvtepu8_epi64; \
    using ::avx512_emu::_mm512_cvtps_pd; \
    using ::avx512_emu::_mm512_extractf32x8_ps; \
    using ::avx512_emu::_mm512_extracti32x4_epi32; \
    using ::avx512_emu::_mm512_fmadd_pd; \
    using ::avx512_emu::_mm512_load_pd; \
    using ::avx512_emu::_mm512_load_ps; \
    using ::avx512_emu::_mm512_load_si512; \
    using ::avx512_emu::_mm512_mask_add_ps; \
    using ::avx512_emu::_mm512_mask_storeu_epi8; \
    using ::avx512_emu::_mm512_mask_storeu_pd; \
    using ::avx512_emu::_mm512_mask_storeu_ps; \
    using ::avx512_emu::_mm512_maskz_compress_epi8; \
    using ::avx512_emu::_mm512_maskz_loadu_epi8; \
    using ::avx512_emu::_mm512_maskz_loadu_pd; \
    using ::avx512_emu::_mm512_maskz_loadu_ps; \
    using ::avx512_emu::_mm512_maskz_mov_pd; \
    using ::avx512_emu::_mm512_multishift_epi64_epi8; \
    using ::avx512_emu::_mm512_or_si512; \
    using ::avx512_emu::_mm512_permutexvar_epi8; \
    using ::avx512_emu::_mm512_permutexvar_pd; \
    using ::avx512_emu::_mm512_permutexvar_ps; \
    using ::avx512_emu::_mm512_popcnt_epi64; \
    using ::avx512_emu::_mm512_reduce_add_epi64; \
    using ::avx512_emu::_mm512_reduce_add_pd; \
    using ::avx512_emu::_mm512_set1_epi16; \
    using ::avx512_emu::_mm512_set1_epi8; \
    using ::avx512_emu::_mm512_set1_pd; \
    using ::avx512_emu::_mm512_setzero_pd; \
    using ::avx512_emu::_mm512_setzero_si512; \
    using ::avx512_emu::_mm512_shuffle_epi8; \
    using ::avx512_emu::_mm512_slli_epi16; \
    using ::avx512_emu::_mm512_srli_epi64; \
    using ::avx512_emu::_mm512_ternarylogic_epi64; \
    using ::avx512_emu::_mm512_testn_epi8_mask; \
    using ::avx512_emu::_mm512_zextsi128_si512; \
    using ::avx512_emu::_mm_loadu_si128; \
    using ::avx512_emu::_mm_maskz_loadu_epi8; \
    using ::avx512_emu::_mm_srli_si128; \
    using ::avx512_emu::_pdep_u64; \
    static_assert(true, "")

#endif  // GCTA_AVX512_EMU_HPP
