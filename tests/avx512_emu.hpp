// -----------------------------------------------------------------------------
// avx512_emu.hpp -- portable scalar emulation of the AVX-512 intrinsics used by
// src/bed_simd_avx512.inc (exact Intel SDM semantics), so the kernel logic can
// be validated bit-for-bit on non-x86 hosts (e.g. Apple Silicon).
//
// Everything lives in the global namespace (the .inc refers to intrinsics
// unqualified).  Never include this together with <immintrin.h>.
// Masked loads/stores touch only the selected elements (ASan-clean).
// -----------------------------------------------------------------------------
#ifndef GCTA_AVX512_EMU_HPP
#define GCTA_AVX512_EMU_HPP

#if defined(__x86_64__) && defined(__AVX512F__)
#  error "avx512_emu.hpp must not be used together with real AVX-512 intrinsics"
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

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

namespace avx512_emu {
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
}  // namespace avx512_emu

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
    __m512i r; for (int i = 0; i < 32; ++i) avx512_emu::set_lane<uint16_t>(r, i, (uint16_t)v); return r;
}
inline __m512i _mm512_load_si512(const void* p) {
    avx512_emu::check_align64(p); __m512i r; std::memcpy(r.b, p, 64); return r;
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
        const uint64_t src = avx512_emu::lane<uint64_t>(b, q);
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
    __m512i r; for (int i = 0; i < 32; ++i) avx512_emu::set_lane<uint16_t>(r, i, a.b[i]); return r;
}
inline __m512i _mm512_cvtepu8_epi32(__m128i a) {
    __m512i r; for (int i = 0; i < 16; ++i) avx512_emu::set_lane<uint32_t>(r, i, a.b[i]); return r;
}
inline __m512i _mm512_cvtepu8_epi64(__m128i a) {
    __m512i r; for (int i = 0; i < 8; ++i) avx512_emu::set_lane<uint64_t>(r, i, a.b[i]); return r;
}
inline __m512i _mm512_slli_epi16(__m512i a, unsigned int imm) {
    __m512i r;
    for (int i = 0; i < 32; ++i) {
        const uint16_t x = avx512_emu::lane<uint16_t>(a, i);
        avx512_emu::set_lane<uint16_t>(r, i, imm > 15 ? (uint16_t)0 : (uint16_t)(x << imm));
    }
    return r;
}
inline __m512i _mm512_srli_epi64(__m512i a, unsigned int imm) {
    __m512i r;
    for (int i = 0; i < 8; ++i) {
        const uint64_t x = avx512_emu::lane<uint64_t>(a, i);
        avx512_emu::set_lane<uint64_t>(r, i, imm > 63 ? 0ull : (x >> imm));
    }
    return r;
}
// VPTERNLOGQ: result bit = imm[(a<<2)|(b<<1)|c]
inline __m512i _mm512_ternarylogic_epi64(__m512i a, __m512i b, __m512i c, int imm) {
    __m512i r;
    for (int q = 0; q < 8; ++q) {
        const uint64_t A = avx512_emu::lane<uint64_t>(a, q), B = avx512_emu::lane<uint64_t>(b, q),
                       C = avx512_emu::lane<uint64_t>(c, q);
        uint64_t x = 0;
        for (int t = 0; t < 8; ++t) {
            if (!((imm >> t) & 1)) continue;
            x |= ((t & 4) ? A : ~A) & ((t & 2) ? B : ~B) & ((t & 1) ? C : ~C);
        }
        avx512_emu::set_lane<uint64_t>(r, q, x);
    }
    return r;
}
inline __m512i _mm512_popcnt_epi64(__m512i a) {
    __m512i r;
    for (int q = 0; q < 8; ++q)
        avx512_emu::set_lane<uint64_t>(r, q, (uint64_t)__builtin_popcountll(avx512_emu::lane<uint64_t>(a, q)));
    return r;
}
inline __m512i _mm512_add_epi64(__m512i a, __m512i b) {
    __m512i r;
    for (int q = 0; q < 8; ++q)
        avx512_emu::set_lane<uint64_t>(r, q, avx512_emu::lane<uint64_t>(a, q) + avx512_emu::lane<uint64_t>(b, q));
    return r;
}
inline long long _mm512_reduce_add_epi64(__m512i a) {
    uint64_t s = 0; for (int q = 0; q < 8; ++q) s += avx512_emu::lane<uint64_t>(a, q); return (long long)s;
}

// ---- 512-bit float / double -------------------------------------------------
inline __m512 _mm512_load_ps(const void* p) { avx512_emu::check_align64(p); __m512 r; std::memcpy(r.f, p, 64); return r; }
inline __m512d _mm512_load_pd(const void* p) { avx512_emu::check_align64(p); __m512d r; std::memcpy(r.d, p, 64); return r; }
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
    for (int i = 0; i < 16; ++i) std::memcpy(&r.f[i], &a.f[avx512_emu::lane<uint32_t>(idx, i) & 15u], 4);
    return r;
}
inline __m512d _mm512_permutexvar_pd(__m512i idx, __m512d a) {
    __m512d r;
    for (int i = 0; i < 8; ++i) std::memcpy(&r.d[i], &a.d[avx512_emu::lane<uint64_t>(idx, i) & 7u], 8);
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

#endif  // GCTA_AVX512_EMU_HPP
