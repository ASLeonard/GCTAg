/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   bed_simd.cpp -- native PLINK1 .bed kernels with runtime ISA dispatch.
   See include/bed_simd.hpp for the contract.

   This translation unit is freestanding-friendly (no C++ standard library
   headers) and is excluded from unity builds so that per-function
   __attribute__((target(...))) AVX-512 code never leaks into other TUs.
*/
#include "bed_simd.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#  define BED_X86 1
#  include <immintrin.h>
#else
#  define BED_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  define BED_ARM64 1
#  include <arm_neon.h>
#else
#  define BED_ARM64 0
#endif

#if BED_X86
#  define BED_T512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi,bmi2,popcnt")))
#  define BED_T256 __attribute__((target("avx2,bmi,bmi2,popcnt,fma")))
#  if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__) && \
      defined(__AVX512VBMI__) && defined(__AVX512VBMI2__) && defined(__AVX512VPOPCNTDQ__) && \
      defined(__BMI__) && defined(__BMI2__) && defined(__POPCNT__)
#    define BED_STATIC_AVX512 1   // e.g. -march=znver4 / native on Zen 4: no CPUID dispatch needed
#  else
#    define BED_STATIC_AVX512 0
#  endif
#  if defined(__AVX2__) && defined(__BMI__) && defined(__BMI2__) && defined(__POPCNT__) && defined(__FMA__)
#    define BED_STATIC_AVX2 1
#  else
#    define BED_STATIC_AVX2 0
#  endif
#endif

#if __STDC_HOSTED__ && __has_include(<stdlib.h>)
#  include <stdlib.h>
#else
extern "C" char* getenv(const char*);   // freestanding cross-compile (no target libc headers)
#endif

namespace bed_simd {
namespace detail {

#include "bed_simd_internal.hpp"

// PLINK1 code (0..3) -> int8 value.
static constexpr int8_t kCodeVal[4] = {2, 0, 1, 0};

// Bits of a 32-bit keep word spread to the even positions of a 64-bit word.
static inline uint64_t spread32(uint64_t x) {
    x &= 0xFFFFFFFFull;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFull;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x << 2))  & 0x3333333333333333ull;
    x = (x | (x << 1))  & 0x5555555555555555ull;
    return x;
}

// ===========================================================================
// Portable scalar kernels (also the reference model for the SIMD paths).
// ===========================================================================
#define BED_DEFINE_COUNT_WORDS(NAME, ATTR)                                                     \
    ATTR static Counts NAME(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,          \
                            uint32_t n_keep) {                                                 \
        const uint64_t M55 = 0x5555555555555555ull;                                            \
        const uint32_t nw = n_raw >> 5; /* complete 64-bit raw words (32 samples) */           \
        uint64_t n0 = 0, nh = 0, nm = 0;                                                       \
        for (uint32_t i = 0; i < nw; ++i) {                                                    \
            const uint64_t v = load_u64(raw + 8u * i);                                         \
            const uint64_t h = v >> 1;                                                         \
            const uint64_t k = keep ? spread32(keep[i >> 1] >> ((i & 1u) * 32u)) : M55;        \
            n0 += popcnt64(v & h & k);                                                         \
            nh += popcnt64(~v & h & k);                                                        \
            nm += popcnt64(v & ~h & k);                                                        \
        }                                                                                      \
        const uint32_t rem = n_raw - (nw << 5);                                                \
        if (rem) {                                                                             \
            const uint64_t v = load_u64_partial(raw + 8u * nw, (rem + 3u) >> 2);               \
            const uint64_t h = v >> 1;                                                         \
            uint64_t k = keep ? spread32(keep[nw >> 1] >> ((nw & 1u) * 32u)) : M55;            \
            k &= (1ull << (2u * rem)) - 1ull;                                                  \
            n0 += popcnt64(v & h & k);                                                         \
            nh += popcnt64(~v & h & k);                                                        \
            nm += popcnt64(v & ~h & k);                                                        \
        }                                                                                      \
        Counts c;                                                                              \
        c.n0 = (uint32_t)n0; c.n1 = (uint32_t)nh; c.nmiss = (uint32_t)nm;                      \
        c.n2 = (keep ? n_keep : n_raw) - c.n0 - c.n1 - c.nmiss;                                \
        return c;                                                                              \
    }

BED_DEFINE_COUNT_WORDS(count_scalar, )

// Left-pack shuffle indices: for an 8-bit keep mask, the positions of its set
// bits followed by 0x80 (out-of-range -> zero for VTBL / VPSHUFB).
struct PackTable {
    uint8_t v[256][8];
    constexpr PackTable() : v{} {
        for (int m = 0; m < 256; ++m) {
            int k = 0;
            for (int b = 0; b < 8; ++b)
                if (m & (1 << b)) v[m][k++] = (uint8_t)b;
            for (; k < 8; ++k) v[m][k] = 0x80;
        }
    }
};
alignas(64) static constexpr PackTable kPack{};

struct BandChunk64 {
    uint32_t r_begin;
    uint32_t s_len;
};

static inline uint32_t plan_band_chunks(const uint64_t* keep, uint32_t nw,
                                        uint32_t& cursor_r_or_w, uint64_t& first,
                                        uint32_t b_len, BandChunk64 chunks[4]) {
    const uint32_t n_chunks = (b_len + 63u) >> 6;
    if (!keep) {
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            const uint32_t s0 = ci * 64u;
            const uint32_t want = (b_len - s0) < 64u ? (b_len - s0) : 64u;
            chunks[ci] = {cursor_r_or_w + s0, want};
        }
        cursor_r_or_w += b_len;
        return n_chunks;
    }
    uint32_t out_n = 0u;
    for (uint32_t ci = 0; ci < n_chunks; ++ci) {
        const uint32_t s0 = ci * 64u;
        const uint32_t want = (b_len - s0) < 64u ? (b_len - s0) : 64u;
        while (cursor_r_or_w < nw && (keep[cursor_r_or_w] & first) == 0u) {
            ++cursor_r_or_w;
            first = ~0ull;
        }
        if (cursor_r_or_w >= nw) break;
        const uint32_t chunk_r = (cursor_r_or_w << 6) + ctz64(keep[cursor_r_or_w] & first);
        uint32_t got = 0u;
        while (got < want && cursor_r_or_w < nw) {
            const uint64_t kw = keep[cursor_r_or_w] & first;
            if (!kw) {
                ++cursor_r_or_w;
                first = ~0ull;
                continue;
            }
            const uint32_t cnt = popcnt64(kw);
            const uint32_t need = want - got;
            if (cnt <= need) {
                got += cnt;
                ++cursor_r_or_w;
                first = ~0ull;
            } else {
                uint64_t t = kw;
                for (uint32_t u = 0; u < need; ++u) t &= t - 1u;
                first = t;
                got += need;
            }
        }
        chunks[out_n++] = {chunk_r, got};
    }
    return out_n;
}

#define BED_DEFINE_KMAJOR_BAND(NAME, ATTR, CHUNK_DECODE)                                       \
    ATTR static void NAME(const uint8_t* raw_base, size_t rec_bytes, const int* valid_idx,     \
                          int K, const uint64_t* keep, uint32_t n_raw, uint32_t r_begin,       \
                          uint32_t n_out, int8_t* kmajor_out, size_t ldK, const double* mu,    \
                          double* r_out) {                                                     \
        if (K <= 0 || n_out == 0u) return;                                                     \
        const bool want_r = (mu != nullptr && r_out != nullptr);                               \
        if (want_r) {                                                                          \
            for (uint32_t s = 0; s < n_out; ++s) r_out[s] = 0.0;                               \
        }                                                                                      \
        const uint32_t nw = (n_raw + 63u) >> 6;                                                \
        uint32_t cursor = keep ? (r_begin >> 6) : r_begin;                                     \
        uint64_t first = keep ? (~0ull << (r_begin & 63u)) : ~0ull;                            \
        constexpr uint32_t kBand = 256u;                                                       \
        alignas(64) int8_t col[64][64];                                                        \
        alignas(64) int8_t tr[64][64];                                                         \
        for (uint32_t b_off = 0; b_off < n_out; b_off += kBand) {                              \
            const uint32_t rem = n_out - b_off;                                                \
            const uint32_t b_len = rem < kBand ? rem : kBand;                                  \
            BandChunk64 chunks[4];                                                             \
            const uint32_t n_chunks = plan_band_chunks(keep, nw, cursor, first, b_len, chunks);\
            for (int m0 = 0; m0 < K; m0 += 64) {                                               \
                const int m_len = (K - m0) < 64 ? (K - m0) : 64;                               \
                for (uint32_t ci = 0; ci < n_chunks; ++ci) {                                   \
                    const uint32_t s0 = ci * 64u;                                              \
                    const uint32_t s_len = chunks[ci].s_len;                                   \
                    if (s_len == 0u) continue;                                                 \
                    for (int dm = 0; dm < m_len; ++dm) {                                       \
                        const uint8_t* raw = raw_base + (size_t)valid_idx[m0 + dm] * rec_bytes;\
                        CHUNK_DECODE(raw, keep, n_raw, chunks[ci].r_begin, s_len, col[dm]);    \
                    }                                                                          \
                    if (want_r) {                                                              \
                        for (int dm = 0; dm < m_len; ++dm) {                                   \
                            const double mu_m = mu[m0 + dm];                                   \
                            double* rp = r_out + b_off + s0;                                   \
                            const int8_t* gp = col[dm];                                        \
                            for (uint32_t ds = 0; ds < s_len; ++ds) {                          \
                                rp[ds] += (double)gp[ds] * mu_m;                               \
                            }                                                                  \
                        }                                                                      \
                    }                                                                          \
                    for (uint32_t ds = 0; ds < s_len; ++ds) {                                  \
                        for (int dm = 0; dm < m_len; ++dm) tr[ds][dm] = col[dm][ds];           \
                    }                                                                          \
                    for (uint32_t ds = 0; ds < s_len; ++ds) {                                  \
                        __builtin_memcpy(kmajor_out + (size_t)(b_off + s0 + ds) * ldK + m0,    \
                                         tr[ds], (size_t)m_len);                               \
                    }                                                                          \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
    }

namespace scalar {

// Visit output samples [0, n_out) in order: f(out_pos, plink1_code).
// Full mode requires r_begin % 4 == 0.
template <class F>
static inline void for_each_code(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                 uint32_t r_begin, uint32_t n_out, F&& f) {
    if (n_out == 0) return;
    if (!keep) {
        uint32_t pos = 0;
        uint32_t j = r_begin >> 2;
        while (pos + 4u <= n_out) {
            const uint32_t b = raw[j++];
            f(pos, b & 3u); f(pos + 1, (b >> 2) & 3u); f(pos + 2, (b >> 4) & 3u); f(pos + 3, b >> 6);
            pos += 4;
        }
        if (pos < n_out) {
            const uint32_t b = raw[j];
            for (uint32_t t = 0; pos < n_out; ++t, ++pos) f(pos, (b >> (2 * t)) & 3u);
        }
        return;
    }
    const uint32_t nwk = (n_raw + 63u) >> 6;
    uint32_t pos = 0;
    uint32_t w = r_begin >> 6;
    uint64_t kw = w < nwk ? (keep[w] & (~0ull << (r_begin & 63u))) : 0;
    while (pos < n_out && w < nwk) {
        while (kw && pos < n_out) {
            const uint32_t r = (w << 6) + ctz64(kw);
            f(pos++, (uint32_t)(raw[r >> 2] >> ((r & 3u) * 2u)) & 3u);
            kw &= kw - 1;
        }
        if (++w < nwk) kw = keep[w];
    }
}

// Even bits of x (positions 0,2,..,62) compressed into the low 32 bits.
static inline uint64_t unspread32(uint64_t x) {
    x &= 0x5555555555555555ull;
    x = (x | (x >> 1))  & 0x3333333333333333ull;
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FFull;
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFFull;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFFull;
    return x;
}

// Missing-call bitmask of 64 raw samples (16 raw bytes, nb valid).
struct MissWordScalar {
    uint64_t operator()(const uint8_t* p, uint32_t nb) const {
        uint64_t v0, v1 = 0;
        if (nb >= 16u) {
            v0 = load_u64(p);
            v1 = load_u64(p + 8);
        } else {
            v0 = load_u64_partial(p, nb < 8u ? nb : 8u);
            if (nb > 8u) v1 = load_u64_partial(p + 8, nb - 8u);
        }
        return unspread32(v0 & ~(v0 >> 1)) | (unspread32(v1 & ~(v1 >> 1)) << 32);
    }
};

// Word-parallel missing pass (only run when nmiss > 0): 64 raw samples per
// step, missing = lo & ~hi, compressed to kept-sample space by popcount.
template <class MW>
static void miss_pass_t(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out,
                        MissSink* ms, const MW& mw) {
    const uint32_t nbytes = (n_raw + 3u) >> 2;
    const uint32_t nw = (n_raw + 63u) >> 6;
    if (!keep && ms->bits && !ms->idx && !ms->wf && !ms->wd) {
        // Full-sample bitmask only: output words align with raw words, so
        // write every word unconditionally (branch-free; nmiss > 0 known).
        const uint32_t nwo = (n_out + 63u) >> 6;
        uint32_t n = 0;
        for (uint32_t w = 0; w < nwo; ++w) {
            const uint32_t b0 = w << 4;
            const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
            uint64_t m = mw(raw + b0, nb);
            const uint32_t rem = n_out - (w << 6);
            if (rem < 64u) m &= (1ull << rem) - 1ull;
            ms->bits[w] = m;
            n += popcnt64(m);
        }
        ms->started = true;
        ms->n_miss += n;
        return;
    }
    uint32_t pos = 0;
    for (uint32_t w = 0; w < nw && pos < n_out; ++w) {
        const uint32_t b0 = w << 4;
        if (keep && !keep[w]) continue;
        const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
        uint64_t m = mw(raw + b0, nb);
        const uint32_t rem = n_raw - (w << 6);
        if (rem < 64u) m &= (1ull << rem) - 1ull;
        if (!keep) {
            if (m) ms->add(m, pos);
            pos += rem < 64u ? rem : 64u;
            continue;
        }
        const uint64_t kw = keep[w];
        m &= kw;
        if (m) {
            uint64_t cm = 0;
            for (uint64_t t = m; t; t &= t - 1) {
                const uint32_t b = ctz64(t);
                cm |= 1ull << popcnt64(kw & ((1ull << b) - 1ull));
            }
            ms->add(cm, pos);
        }
        pos += popcnt64(kw);
    }
}

static void miss_pass(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out, MissSink* ms) {
    miss_pass_t(raw, keep, n_raw, n_out, ms, MissWordScalar{});
}

static void weight_pass(const int8_t* g, uint32_t n, I8Opts* o) {
    if (o->wf) {
        const float* w = o->wf;
        double acc = 0.0;
        for (uint32_t s = 0; s < n; ++s) acc += (double)g[s] * (double)w[s];
        o->wf_dot = acc;
    }
    if (o->wd) {
        const double* w = o->wd;
        double a1 = 0.0, a2 = 0.0;
        for (uint32_t s = 0; s < n; ++s) {
            const double gd = (double)g[s];
            a1 += gd * w[s];
            a2 += gd * gd * w[s];
        }
        o->wd_dot = a1;
        o->wd_dot2 = a2;
    }
}

static Counts decode_i8(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                        uint32_t r_begin, uint32_t n_out, int8_t* out, I8Opts* o, bool want_counts) {
    for_each_code(raw, keep, n_raw, r_begin, n_out, [&](uint32_t s, uint32_t c) { out[s] = kCodeVal[c]; });
    Counts c;
    if (want_counts) c = count_scalar(raw, keep, n_raw, n_out);
    if (o) {
        MissSink sink;
        sink.init(o, n_out);
        if ((!want_counts || c.nmiss) && sink.active()) miss_pass(raw, keep, n_raw, n_out, &sink);
        weight_pass(out, n_out, o);
        sink.finish(o);
    }
    return c;
}

template <class T>
static Counts decode_tab(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out,
                         const T* t4, T* out, uint64_t* miss_bits, bool want_counts) {
    const T tb[4] = {t4[2], t4[3], t4[1], t4[0]};   // indexed by PLINK1 code
    for_each_code(raw, keep, n_raw, 0, n_out, [&](uint32_t s, uint32_t c) { out[s] = tb[c]; });
    Counts c;
    if (want_counts) c = count_scalar(raw, keep, n_raw, n_out);
    if (miss_bits && (!want_counts || c.nmiss)) {
        MissSink sink;
        sink.init_bits(miss_bits, n_out);
        miss_pass(raw, keep, n_raw, n_out, &sink);
    }
    return c;
}
static Counts decode_f32(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out,
                         const float* t4, float* out, uint64_t* miss_bits, bool want_counts) {
    return decode_tab<float>(raw, keep, n_raw, n_out, t4, out, miss_bits, want_counts);
}
static Counts decode_f64(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out,
                         const double* t4, double* out, uint64_t* miss_bits, bool want_counts) {
    return decode_tab<double>(raw, keep, n_raw, n_out, t4, out, miss_bits, want_counts);
}

static inline void decode_i8_chunk(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                   uint32_t r_begin, uint32_t n_out, int8_t* out) {
    decode_i8(raw, keep, n_raw, r_begin, n_out, out, nullptr, false);
}
BED_DEFINE_KMAJOR_BAND(decode_i8_kmajor_band, , decode_i8_chunk)

}  // namespace scalar

// ===========================================================================
// ARM64 NEON kernels (Apple Silicon dev hosts, Graviton).
// ===========================================================================
#if BED_ARM64
namespace neon {

static inline uint8x16_t spread_keep16(uint64_t kw, uint8x16_t nib) {
    // 64 keep bits -> 128-bit mask with each bit spread to the even position
    // of its 2-bit genotype slot (raw byte j covers keep bits 4j..4j+3).
    const uint8x8_t kb = vcreate_u8(kw);
    const uint8x8_t lo = vand_u8(kb, vdup_n_u8(0x0F));
    const uint8x8_t hi = vshr_n_u8(kb, 4);
    const uint8x8x2_t z = vzip_u8(lo, hi);   // lo0 hi0 lo1 hi1 ...
    return vqtbl1q_u8(nib, vcombine_u8(z.val[0], z.val[1]));
}

static Counts count(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep) {
    static const uint8_t kNib[16] = {0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
                                     0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55};
    const uint8x16_t nib = vld1q_u8(kNib);
    const uint8x16_t m55 = vdupq_n_u8(0x55);
    const uint32_t nblk = n_raw >> 6;   // complete 16-byte (64-sample) blocks
    uint64_t n0 = 0, nh = 0, nm = 0;
    uint32_t i = 0;
    while (i < nblk) {
        // Each block adds <= 8 to a u16 lane (vpadal of two <=4 popcounts): flush every 4096 blocks.
        const uint32_t stop = (nblk - i) > 4096u ? i + 4096u : nblk;
        uint16x8_t s0 = vdupq_n_u16(0), sh = vdupq_n_u16(0), sm = vdupq_n_u16(0);
        for (; i < stop; ++i) {
            const uint8x16_t v = vld1q_u8(raw + 16u * i);
            const uint8x16_t h = vshrq_n_u8(v, 1);
            const uint8x16_t k = keep ? spread_keep16(keep[i], nib) : m55;
            s0 = vpadalq_u8(s0, vcntq_u8(vandq_u8(vandq_u8(v, h), k)));
            sh = vpadalq_u8(sh, vcntq_u8(vandq_u8(vbicq_u8(h, v), k)));
            sm = vpadalq_u8(sm, vcntq_u8(vandq_u8(vbicq_u8(v, h), k)));
        }
        n0 += vaddlvq_u16(s0);
        nh += vaddlvq_u16(sh);
        nm += vaddlvq_u16(sm);
    }
    // tail (< 64 samples): scalar word kernel on the remaining bytes
    const uint32_t done = nblk << 6;
    if (done < n_raw) {
        const uint32_t rem = n_raw - done;
        const uint8_t* rp = raw + (done >> 2);
        const uint64_t* kp = keep ? keep + (done >> 6) : nullptr;
        const Counts t = count_scalar(rp, kp, rem, 0);
        n0 += t.n0; nh += t.n1; nm += t.nmiss;
    }
    Counts c;
    c.n0 = (uint32_t)n0; c.n1 = (uint32_t)nh; c.nmiss = (uint32_t)nm;
    c.n2 = (keep ? n_keep : n_raw) - c.n0 - c.n1 - c.nmiss;
    return c;
}

// Missing-call bitmask of 64 raw samples: lo & ~hi on even bits, then the
// even bits of each byte folded to a nibble and nibble pairs merged (VSHRN).
struct MissWordNeon {
    uint64_t operator()(const uint8_t* p, uint32_t nb) const {
        uint8x16_t v;
        if (nb >= 16u) {
            v = vld1q_u8(p);
        } else {
            uint8_t tmp[16] = {0};
            __builtin_memcpy(tmp, p, nb);
            v = vld1q_u8(tmp);
        }
        const uint8x16_t m = vandq_u8(vbicq_u8(v, vshrq_n_u8(v, 1)), vdupq_n_u8(0x55));
        uint8x16_t x = vandq_u8(vorrq_u8(m, vshrq_n_u8(m, 1)), vdupq_n_u8(0x33));
        x = vandq_u8(vorrq_u8(x, vshrq_n_u8(x, 2)), vdupq_n_u8(0x0F));
        const uint8x8_t ev = vmovn_u16(vreinterpretq_u16_u8(x));
        const uint8x8_t od = vshrn_n_u16(vreinterpretq_u16_u8(x), 8);
        return vget_lane_u64(vreinterpret_u64_u8(vorr_u8(ev, vshl_n_u8(od, 4))), 0);
    }
};
static void miss_pass(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out, MissSink* ms) {
    scalar::miss_pass_t(raw, keep, n_raw, n_out, ms, MissWordNeon{});
}

// 16 raw bytes (64 samples; nb valid bytes) -> 64 PLINK1 codes via VST4.
static inline void codes64(const uint8_t* raw, uint32_t nb, uint8_t* dst) {
    uint8x16_t v;
    if (nb >= 16u) {
        v = vld1q_u8(raw);
    } else {
        uint8_t tmp[16] = {0};
        __builtin_memcpy(tmp, raw, nb);
        v = vld1q_u8(tmp);
    }
    const uint8x16_t m3 = vdupq_n_u8(3);
    uint8x16x4_t r;
    r.val[0] = vandq_u8(v, m3);
    r.val[1] = vandq_u8(vshrq_n_u8(v, 2), m3);
    r.val[2] = vandq_u8(vshrq_n_u8(v, 4), m3);
    r.val[3] = vshrq_n_u8(v, 6);
    vst4q_u8(dst, r);
}

// Full-sample int8 decode: 16 raw bytes -> 64 values via 4x VTBL + VST4.
static void decode_i8_full(const uint8_t* raw, uint32_t n_out, int8_t* out) {
    static const uint8_t kLut[16] = {2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint8x16_t lut = vld1q_u8(kLut);
    const uint8x16_t m3 = vdupq_n_u8(3);
    uint32_t pos = 0;
    for (; pos + 64u <= n_out; pos += 64u) {
        const uint8x16_t v = vld1q_u8(raw + (pos >> 2));
        uint8x16x4_t r;
        r.val[0] = vqtbl1q_u8(lut, vandq_u8(v, m3));
        r.val[1] = vqtbl1q_u8(lut, vandq_u8(vshrq_n_u8(v, 2), m3));
        r.val[2] = vqtbl1q_u8(lut, vandq_u8(vshrq_n_u8(v, 4), m3));
        r.val[3] = vqtbl1q_u8(lut, vshrq_n_u8(v, 6));
        vst4q_u8((uint8_t*)out + pos, r);
    }
    if (pos < n_out) {
        scalar::for_each_code(raw + (pos >> 2), nullptr, n_out - pos, 0, n_out - pos,
                              [&](uint32_t s, uint32_t c) { out[pos + s] = kCodeVal[c]; });
    }
}

// Stage 1: packed codes for up to `want` (+63) output samples into buf
// (buf needs want + 72 bytes).  Subsets are left-packed 8 samples at a time
// with VTBL and a 256-entry shuffle table.
struct Cursor {
    const uint8_t* raw;
    const uint64_t* keep;
    uint32_t nbytes, w, nw;
    uint64_t first;
};
static uint32_t stage_codes(Cursor& cur, uint32_t want, uint8_t* buf) {
    uint32_t cnt = 0;
    while (cnt < want && cur.w < cur.nw) {
        const uint32_t b0 = cur.w << 4;
        const uint32_t nb = (cur.nbytes - b0) < 16u ? (cur.nbytes - b0) : 16u;
        if (!cur.keep) {
            codes64(cur.raw + b0, nb, buf + cnt);
            cnt += 64;
        } else {
            const uint64_t kw = cur.keep[cur.w] & cur.first;
            cur.first = ~0ull;
            if (kw) {
                alignas(16) uint8_t tmp[64];
                codes64(cur.raw + b0, nb, tmp);
                for (int g = 0; g < 8; ++g) {
                    const uint32_t kb = (uint32_t)(kw >> (8 * g)) & 0xFFu;
                    if (!kb) continue;
                    vst1_u8(buf + cnt, vtbl1_u8(vld1_u8(tmp + 8 * g), vld1_u8(kPack.v[kb])));
                    cnt += popcnt64(kb);
                }
            }
        }
        ++cur.w;
    }
    return cnt;
}

static void map_i8(const uint8_t* codes, uint32_t n, int8_t* out) {
    static const uint8_t kLut[16] = {2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint8x16_t lut = vld1q_u8(kLut);
    uint32_t i = 0;
    for (; i + 16u <= n; i += 16u) vst1q_u8((uint8_t*)out + i, vqtbl1q_u8(lut, vld1q_u8(codes + i)));
    for (; i < n; ++i) out[i] = kCodeVal[codes[i]];
}

// out[i] = tb[codes[i]] for 4-byte elements via byte-table VTBL.
static void map_f32(const uint8_t* codes, uint32_t n, const float* tb, float* out) {
    uint8_t tbytes[16];
    __builtin_memcpy(tbytes, tb, 16);
    const uint8x16_t tbl = vld1q_u8(tbytes);
    static const uint8_t kOff[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const uint8x16_t off = vld1q_u8(kOff);
    uint32_t i = 0;
    for (; i + 16u <= n; i += 16u) {
        const uint8x16_t c4 = vshlq_n_u8(vld1q_u8(codes + i), 2);
        const uint8x16x2_t z1 = vzipq_u8(c4, c4);                 // each code x2
        const uint8x16x2_t za = vzipq_u8(z1.val[0], z1.val[0]);   // codes 0..7 x4
        const uint8x16x2_t zb = vzipq_u8(z1.val[1], z1.val[1]);   // codes 8..15 x4
        uint8_t* o = (uint8_t*)(out + i);
        vst1q_u8(o +  0, vqtbl1q_u8(tbl, vaddq_u8(za.val[0], off)));
        vst1q_u8(o + 16, vqtbl1q_u8(tbl, vaddq_u8(za.val[1], off)));
        vst1q_u8(o + 32, vqtbl1q_u8(tbl, vaddq_u8(zb.val[0], off)));
        vst1q_u8(o + 48, vqtbl1q_u8(tbl, vaddq_u8(zb.val[1], off)));
    }
    for (; i < n; ++i) out[i] = tb[codes[i]];
}

// out[i] = tb[codes[i]] for 8-byte elements via 32-byte VTBL2.
static void map_f64(const uint8_t* codes, uint32_t n, const double* tb, double* out) {
    uint8_t tbytes[32];
    __builtin_memcpy(tbytes, tb, 32);
    uint8x16x2_t tbl;
    tbl.val[0] = vld1q_u8(tbytes);
    tbl.val[1] = vld1q_u8(tbytes + 16);
    static const uint8_t kOff[16] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    const uint8x16_t off = vld1q_u8(kOff);
    uint32_t i = 0;
    for (; i + 16u <= n; i += 16u) {
        const uint8x16_t c8 = vshlq_n_u8(vld1q_u8(codes + i), 3);
        const uint8x16x2_t z1 = vzipq_u8(c8, c8);
        uint8_t* o = (uint8_t*)(out + i);
        for (int h = 0; h < 2; ++h) {
            const uint8x16x2_t z2 = vzipq_u8(z1.val[h], z1.val[h]);
            for (int q = 0; q < 2; ++q) {
                const uint8x16x2_t z3 = vzipq_u8(z2.val[q], z2.val[q]);   // 2 codes x8 per vector
                vst1q_u8(o, vqtbl2q_u8(tbl, vaddq_u8(z3.val[0], off)));
                vst1q_u8(o + 16, vqtbl2q_u8(tbl, vaddq_u8(z3.val[1], off)));
                o += 32;
            }
        }
    }
    for (; i < n; ++i) out[i] = tb[codes[i]];
}

static constexpr uint32_t kStage = 512;

template <class MapFn>
static void run_stages(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                       uint32_t r_begin, uint32_t n_out, MapFn&& map) {
    alignas(16) uint8_t buf[kStage + 128];
    Cursor cur;
    cur.raw = raw;
    cur.keep = keep;
    cur.nbytes = (n_raw + 3u) >> 2;
    cur.w = r_begin >> 6;
    cur.nw = (n_raw + 63u) >> 6;
    cur.first = keep ? (~0ull << (r_begin & 63u)) : ~0ull;
    uint32_t pos = 0;
    while (pos < n_out) {
        const uint32_t want = (n_out - pos) < kStage ? (n_out - pos) : kStage;
        uint32_t got = stage_codes(cur, want, buf);
        if (got == 0) break;
        if (got > n_out - pos) got = n_out - pos;
        map(buf, got, pos);
        pos += got;
    }
}

static void weight_pass(const int8_t* g, uint32_t n, I8Opts* o) {
    if (o->wf) {
        const float* w = o->wf;
        float64x2_t a0 = vdupq_n_f64(0.0), a1 = vdupq_n_f64(0.0);
        float64x2_t a2 = vdupq_n_f64(0.0), a3 = vdupq_n_f64(0.0);
        float64x2_t b0 = vdupq_n_f64(0.0), b1 = vdupq_n_f64(0.0);
        float64x2_t b2 = vdupq_n_f64(0.0), b3 = vdupq_n_f64(0.0);
        uint32_t s = 0;
        for (; s + 16u <= n; s += 16u) {
            const int8x16_t gv = vld1q_s8(g + s);
            const int16x8_t lo16 = vmovl_s8(vget_low_s8(gv));
            const int16x8_t hi16 = vmovl_s8(vget_high_s8(gv));
            const float32x4_t p0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vld1q_f32(w + s));
            const float32x4_t p1 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo16)), vld1q_f32(w + s + 4));
            const float32x4_t p2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vld1q_f32(w + s + 8));
            const float32x4_t p3 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi16)), vld1q_f32(w + s + 12));
            a0 = vaddq_f64(a0, vcvt_f64_f32(vget_low_f32(p0)));
            a1 = vaddq_f64(a1, vcvt_high_f64_f32(p0));
            a2 = vaddq_f64(a2, vcvt_f64_f32(vget_low_f32(p1)));
            a3 = vaddq_f64(a3, vcvt_high_f64_f32(p1));
            b0 = vaddq_f64(b0, vcvt_f64_f32(vget_low_f32(p2)));
            b1 = vaddq_f64(b1, vcvt_high_f64_f32(p2));
            b2 = vaddq_f64(b2, vcvt_f64_f32(vget_low_f32(p3)));
            b3 = vaddq_f64(b3, vcvt_high_f64_f32(p3));
        }
        const float64x2_t sum01 = vaddq_f64(vaddq_f64(a0, b0), vaddq_f64(a1, b1));
        const float64x2_t sum23 = vaddq_f64(vaddq_f64(a2, b2), vaddq_f64(a3, b3));
        double acc = vaddvq_f64(vaddq_f64(sum01, sum23));
        for (; s < n; ++s) acc += static_cast<double>(g[s]) * static_cast<double>(w[s]);
        o->wf_dot = acc;
    }
    if (o->wd) {
        const double* w = o->wd;
        float64x2_t a1_0 = vdupq_n_f64(0.0), a1_1 = vdupq_n_f64(0.0);
        float64x2_t a1_2 = vdupq_n_f64(0.0), a1_3 = vdupq_n_f64(0.0);
        float64x2_t a2_0 = vdupq_n_f64(0.0), a2_1 = vdupq_n_f64(0.0);
        float64x2_t a2_2 = vdupq_n_f64(0.0), a2_3 = vdupq_n_f64(0.0);
        uint32_t s = 0;
        for (; s + 8u <= n; s += 8u) {
            const int16x8_t g16 = vmovl_s8(vld1_s8(g + s));
            const int32x4_t g32_0 = vmovl_s16(vget_low_s16(g16));
            const int32x4_t g32_1 = vmovl_s16(vget_high_s16(g16));
            const float64x2_t d0 = vcvtq_f64_s64(vmovl_s32(vget_low_s32(g32_0)));
            const float64x2_t d1 = vcvtq_f64_s64(vmovl_high_s32(g32_0));
            const float64x2_t d2 = vcvtq_f64_s64(vmovl_s32(vget_low_s32(g32_1)));
            const float64x2_t d3 = vcvtq_f64_s64(vmovl_high_s32(g32_1));
            const float64x2_t gw0 = vmulq_f64(d0, vld1q_f64(w + s));
            const float64x2_t gw1 = vmulq_f64(d1, vld1q_f64(w + s + 2));
            const float64x2_t gw2 = vmulq_f64(d2, vld1q_f64(w + s + 4));
            const float64x2_t gw3 = vmulq_f64(d3, vld1q_f64(w + s + 6));
            a1_0 = vaddq_f64(a1_0, gw0);
            a1_1 = vaddq_f64(a1_1, gw1);
            a1_2 = vaddq_f64(a1_2, gw2);
            a1_3 = vaddq_f64(a1_3, gw3);
            a2_0 = vfmaq_f64(a2_0, d0, gw0);
            a2_1 = vfmaq_f64(a2_1, d1, gw1);
            a2_2 = vfmaq_f64(a2_2, d2, gw2);
            a2_3 = vfmaq_f64(a2_3, d3, gw3);
        }
        double s1 = vaddvq_f64(vaddq_f64(vaddq_f64(a1_0, a1_1), vaddq_f64(a1_2, a1_3)));
        double s2 = vaddvq_f64(vaddq_f64(vaddq_f64(a2_0, a2_1), vaddq_f64(a2_2, a2_3)));
        for (; s < n; ++s) {
            const double gd = static_cast<double>(g[s]);
            const double gw = gd * w[s];
            s1 += gw;
            s2 += gd * gw;
        }
        o->wd_dot = s1;
        o->wd_dot2 = s2;
    }
}

static Counts decode_i8(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                        uint32_t r_begin, uint32_t n_out, int8_t* out, I8Opts* o, bool want_counts) {
    if (keep) {
        run_stages(raw, keep, n_raw, r_begin, n_out,
                   [&](const uint8_t* c, uint32_t n, uint32_t p) { map_i8(c, n, out + p); });
    } else {
        decode_i8_full(raw + (r_begin >> 2), n_out, out);
    }
    Counts c;
    if (want_counts) c = count(raw, keep, n_raw, n_out);
    if (o) {
        MissSink sink;
        sink.init(o, n_out);
        if ((!want_counts || c.nmiss) && sink.active()) miss_pass(raw, keep, n_raw, n_out, &sink);
        weight_pass(out, n_out, o);
        sink.finish(o);
    }
    return c;
}

static Counts decode_f32(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out,
                         const float* t4, float* out, uint64_t* miss_bits, bool want_counts) {
    const float tb[4] = {t4[2], t4[3], t4[1], t4[0]};   // indexed by PLINK1 code
    run_stages(raw, keep, n_raw, 0, n_out,
               [&](const uint8_t* c, uint32_t n, uint32_t p) { map_f32(c, n, tb, out + p); });
    Counts c;
    if (want_counts) c = count(raw, keep, n_raw, n_out);
    if (miss_bits && (!want_counts || c.nmiss)) {
        MissSink sink;
        sink.init_bits(miss_bits, n_out);
        miss_pass(raw, keep, n_raw, n_out, &sink);
    }
    return c;
}

static Counts decode_f64(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out,
                         const double* t4, double* out, uint64_t* miss_bits, bool want_counts) {
    const double tb[4] = {t4[2], t4[3], t4[1], t4[0]};
    run_stages(raw, keep, n_raw, 0, n_out,
               [&](const uint8_t* c, uint32_t n, uint32_t p) { map_f64(c, n, tb, out + p); });
    Counts c;
    if (want_counts) c = count(raw, keep, n_raw, n_out);
    if (miss_bits && (!want_counts || c.nmiss)) {
        MissSink sink;
        sink.init_bits(miss_bits, n_out);
        miss_pass(raw, keep, n_raw, n_out, &sink);
    }
    return c;
}

static inline void decode_i8_chunk(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                   uint32_t r_begin, uint32_t n_out, int8_t* out) {
    decode_i8(raw, keep, n_raw, r_begin, n_out, out, nullptr, false);
}
BED_DEFINE_KMAJOR_BAND(decode_i8_kmajor_band, , decode_i8_chunk)

}  // namespace neon
#endif  // BED_ARM64

// ===========================================================================
// x86_64 AVX2 (+BMI2/POPCNT) kernels: x86-64-v3 hosts without AVX-512.
// Two-stage: (1) expand 32 raw samples/iter to PLINK1 code bytes and left-pack
// kept samples with a VPSHUFB table into an L1-resident staging buffer;
// (2) map staged codes to int8 (VPSHUFB) / float (VPERMPS) / double.
// ===========================================================================
#if BED_X86
BED_DEFINE_COUNT_WORDS(count_popcnt, BED_T256)

namespace avx2 {

alignas(32) static const uint8_t kRep32[32] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                               4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
alignas(32) static const uint8_t kLoM[32] = {1, 4, 16, 64, 1, 4, 16, 64, 1, 4, 16, 64, 1, 4, 16, 64,
                                             1, 4, 16, 64, 1, 4, 16, 64, 1, 4, 16, 64, 1, 4, 16, 64};
alignas(32) static const uint8_t kHiM[32] = {2, 8, 32, 128, 2, 8, 32, 128, 2, 8, 32, 128, 2, 8, 32, 128,
                                             2, 8, 32, 128, 2, 8, 32, 128, 2, 8, 32, 128, 2, 8, 32, 128};
alignas(32) static const uint8_t kValLut32[32] = {2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// 32 raw samples starting at raw byte p (nbytes in [1,8]) -> 32 PLINK1 codes.
BED_T256 static inline __m256i codes32(const uint8_t* p, uint32_t nbytes) {
    const uint64_t x = nbytes >= 8 ? load_u64(p) : load_u64_partial(p, nbytes);
    const __m256i r = _mm256_shuffle_epi8(_mm256_set1_epi64x((long long)x),
                                          _mm256_load_si256((const __m256i*)kRep32));
    const __m256i lom = _mm256_load_si256((const __m256i*)kLoM);
    const __m256i him = _mm256_load_si256((const __m256i*)kHiM);
    const __m256i lo = _mm256_cmpeq_epi8(_mm256_and_si256(r, lom), lom);
    const __m256i hi = _mm256_cmpeq_epi8(_mm256_and_si256(r, him), him);
    return _mm256_or_si256(_mm256_and_si256(lo, _mm256_set1_epi8(1)), _mm256_and_si256(hi, _mm256_set1_epi8(2)));
}

// Stage 1: produce up to `want` packed codes into buf (buf needs >= want + 40
// bytes), advancing the raw word cursor.  Returns number produced.
struct Cursor {
    const uint8_t* raw;
    const uint64_t* keep;
    uint32_t nbytes;     // raw record bytes
    uint32_t q;          // next 32-sample raw group
    uint32_t nq;         // number of 32-sample groups
    uint32_t first_skip; // keep bits to clear in first group (subset range start)
};

BED_T256 static uint32_t stage_codes(Cursor& cur, uint32_t want, uint8_t* buf) {
    uint32_t cnt = 0;
    if (!cur.keep) {
        while (cnt < want && cur.q < cur.nq) {
            const uint32_t b0 = cur.q << 3;
            const uint32_t nb = (cur.nbytes - b0) < 8u ? (cur.nbytes - b0) : 8u;
            _mm256_storeu_si256((__m256i*)(buf + cnt), codes32(cur.raw + b0, nb));
            cnt += 32;
            ++cur.q;
        }
        return cnt;   // caller clamps to n_out
    }
    while (cnt < want && cur.q < cur.nq) {
        uint32_t k32 = (uint32_t)(cur.keep[cur.q >> 1] >> ((cur.q & 1u) * 32u));
        if (cur.first_skip) { k32 &= ~0u << cur.first_skip; cur.first_skip = 0; }
        if (k32) {
            const uint32_t b0 = cur.q << 3;
            const uint32_t nb = (cur.nbytes - b0) < 8u ? (cur.nbytes - b0) : 8u;
            const __m256i c = codes32(cur.raw + b0, nb);
            const __m128i l0 = _mm256_castsi256_si128(c);
            const __m128i l1 = _mm256_extracti128_si256(c, 1);
            const __m128i add8 = _mm_set1_epi8(8);
            for (int g = 0; g < 4; ++g) {
                const uint32_t kb = (k32 >> (8 * g)) & 0xFFu;
                if (!kb) continue;
                __m128i ctl = _mm_loadl_epi64((const __m128i*)kPack.v[kb]);
                if (g & 1) ctl = _mm_add_epi8(ctl, add8);
                const __m128i src = (g < 2) ? l0 : l1;
                _mm_storel_epi64((__m128i*)(buf + cnt), _mm_shuffle_epi8(src, ctl));
                cnt += popcnt64(kb);
            }
        }
        ++cur.q;
    }
    return cnt;
}

BED_T256 static void map_i8(const uint8_t* codes, uint32_t n, int8_t* out) {
    const __m256i lut = _mm256_load_si256((const __m256i*)kValLut32);
    uint32_t i = 0;
    for (; i + 32u <= n; i += 32u)
        _mm256_storeu_si256((__m256i*)(out + i),
                            _mm256_shuffle_epi8(lut, _mm256_loadu_si256((const __m256i*)(codes + i))));
    for (; i < n; ++i) out[i] = kCodeVal[codes[i]];
}

BED_T256 static void map_f32(const uint8_t* codes, uint32_t n, const float* tb, float* out) {
    const __m256 tbl = _mm256_setr_ps(tb[0], tb[1], tb[2], tb[3], tb[0], tb[1], tb[2], tb[3]);
    uint32_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        const __m256i ix = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(codes + i)));
        _mm256_storeu_ps(out + i, _mm256_permutevar8x32_ps(tbl, ix));
    }
    for (; i < n; ++i) out[i] = tb[codes[i]];
}

BED_T256 static void map_f64(const uint8_t* codes, uint32_t n, const double* tb, double* out) {
    const __m256d t01 = _mm256_setr_pd(tb[0], tb[1], tb[0], tb[1]);
    const __m256d t23 = _mm256_setr_pd(tb[2], tb[3], tb[2], tb[3]);
    uint32_t i = 0;
    for (; i + 4u <= n; i += 4u) {
        int32_t four;
        __builtin_memcpy(&four, codes + i, 4);
        const __m256i ix = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(four));
        const __m256i sel = _mm256_slli_epi64(ix, 1);                     // code bit0 -> bit1 (vpermilpd)
        const __m256d r01 = _mm256_permutevar_pd(t01, sel);
        const __m256d r23 = _mm256_permutevar_pd(t23, sel);
        const __m256d hi  = _mm256_castsi256_pd(_mm256_slli_epi64(ix, 62)); // code bit1 -> sign
        _mm256_storeu_pd(out + i, _mm256_blendv_pd(r01, r23, hi));
    }
    for (; i < n; ++i) out[i] = tb[codes[i]];
}

BED_T256 static void weight_pass(const int8_t* g, uint32_t n, I8Opts* o) {
    if (o->wf) {
        const float* w = o->wf;
        __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
        uint32_t s = 0;
        for (; s + 8u <= n; s += 8u) {
            const __m256 gf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(g + s))));
            const __m256 p = _mm256_mul_ps(gf, _mm256_loadu_ps(w + s));   // exact: g in {0,1,2}
            a0 = _mm256_add_pd(a0, _mm256_cvtps_pd(_mm256_castps256_ps128(p)));
            a1 = _mm256_add_pd(a1, _mm256_cvtps_pd(_mm256_extractf128_ps(p, 1)));
        }
        alignas(32) double t[4];
        _mm256_store_pd(t, _mm256_add_pd(a0, a1));
        double acc = (t[0] + t[1]) + (t[2] + t[3]);
        for (; s < n; ++s) acc += (double)g[s] * (double)w[s];
        o->wf_dot = acc;
    }
    if (o->wd) {
        const double* w = o->wd;
        __m256d a1 = _mm256_setzero_pd(), a2 = _mm256_setzero_pd();
        uint32_t s = 0;
        for (; s + 4u <= n; s += 4u) {
            int32_t four;
            __builtin_memcpy(&four, g + s, 4);
            const __m256d gd = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(four)));
            const __m256d gw = _mm256_mul_pd(gd, _mm256_loadu_pd(w + s));
            a1 = _mm256_add_pd(a1, gw);
            a2 = _mm256_fmadd_pd(gd, gw, a2);
        }
        alignas(32) double t1[4], t2[4];
        _mm256_store_pd(t1, a1);
        _mm256_store_pd(t2, a2);
        double s1 = (t1[0] + t1[1]) + (t1[2] + t1[3]);
        double s2 = (t2[0] + t2[1]) + (t2[2] + t2[3]);
        for (; s < n; ++s) { const double gd = (double)g[s]; s1 += gd * w[s]; s2 += gd * gd * w[s]; }
        o->wd_dot = s1;
        o->wd_dot2 = s2;
    }
}

static constexpr uint32_t kStage = 512;

// Generic two-stage driver; MapFn(codes, n, out_offset).
template <class MapFn>
BED_T256 static void run_stages(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                uint32_t r_begin, uint32_t n_out, MapFn&& map) {
    alignas(32) uint8_t buf[kStage + 64];
    Cursor cur;
    cur.raw = raw;
    cur.keep = keep;
    cur.nbytes = (n_raw + 3u) >> 2;
    cur.q = r_begin >> 5;
    cur.nq = (n_raw + 31u) >> 5;
    cur.first_skip = keep ? (r_begin & 31u) : 0u;
    uint32_t pos = 0;
    while (pos < n_out) {
        const uint32_t want = (n_out - pos) < kStage ? (n_out - pos) : kStage;
        uint32_t got = stage_codes(cur, want, buf);
        if (got == 0) break;
        if (got > n_out - pos) got = n_out - pos;
        map(buf, got, pos);
        pos += got;
    }
}

BED_T256 static Counts decode_i8(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                 uint32_t r_begin, uint32_t n_out, int8_t* out, I8Opts* o, bool want_counts) {
    run_stages(raw, keep, n_raw, r_begin, n_out,
               [&](const uint8_t* c, uint32_t n, uint32_t p) { map_i8(c, n, out + p); });
    Counts c;
    if (want_counts) c = count_popcnt(raw, keep, n_raw, n_out);
    if (o) {
        MissSink sink;
        sink.init(o, n_out);
        if ((!want_counts || c.nmiss) && sink.active()) scalar::miss_pass(raw, keep, n_raw, n_out, &sink);
        weight_pass(out, n_out, o);
        sink.finish(o);
    }
    return c;
}

BED_T256 static Counts decode_f32(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out,
                                  const float* t4, float* out, uint64_t* miss_bits, bool want_counts) {
    const float tb[4] = {t4[2], t4[3], t4[1], t4[0]};
    run_stages(raw, keep, n_raw, 0, n_out,
               [&](const uint8_t* c, uint32_t n, uint32_t p) { map_f32(c, n, tb, out + p); });
    Counts c;
    if (want_counts) c = count_popcnt(raw, keep, n_raw, n_out);
    if (miss_bits && (!want_counts || c.nmiss)) {
        MissSink sink;
        sink.init_bits(miss_bits, n_out);
        scalar::miss_pass(raw, keep, n_raw, n_out, &sink);
    }
    return c;
}

BED_T256 static Counts decode_f64(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_out,
                                  const double* t4, double* out, uint64_t* miss_bits, bool want_counts) {
    const double tb[4] = {t4[2], t4[3], t4[1], t4[0]};
    run_stages(raw, keep, n_raw, 0, n_out,
               [&](const uint8_t* c, uint32_t n, uint32_t p) { map_f64(c, n, tb, out + p); });
    Counts c;
    if (want_counts) c = count_popcnt(raw, keep, n_raw, n_out);
    if (miss_bits && (!want_counts || c.nmiss)) {
        MissSink sink;
        sink.init_bits(miss_bits, n_out);
        scalar::miss_pass(raw, keep, n_raw, n_out, &sink);
    }
    return c;
}

BED_T256 static inline void decode_i8_chunk(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                            uint32_t r_begin, uint32_t s_len, int8_t* out) {
    if (!keep) {
        const uint32_t rem_raw = n_raw - r_begin;
        const uint32_t n_vis = s_len < rem_raw ? s_len : rem_raw;
        const uint32_t b0 = r_begin >> 2;
        const uint32_t nb = (n_vis + 3u) >> 2;
        alignas(32) uint8_t tmp[64];
        const uint32_t nb0 = nb < 8u ? nb : 8u;
        _mm256_store_si256((__m256i*)tmp, codes32(raw + b0, nb0));
        if (nb > 8u) {
            _mm256_store_si256((__m256i*)(tmp + 32), codes32(raw + b0 + 8u, nb - 8u));
        }
        map_i8(tmp, n_vis, out);
        return;
    }
    decode_i8(raw, keep, n_raw, r_begin, s_len, out, nullptr, false);
}
BED_DEFINE_KMAJOR_BAND(decode_i8_kmajor_band, BED_T256, decode_i8_chunk)

}  // namespace avx2

// AVX-512 kernels (shared with the emulator-based unit test).
#include "bed_simd_avx512.inc"

BED_T512 static Counts avx512_decode_f32_wrap(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                              uint32_t n_out, const float* t4, float* out,
                                              uint64_t* miss_bits, bool want_counts) {
    (void)want_counts;   // AVX-512 fuses counting into the single decode pass
    return avx512::decode_f32(raw, keep, n_raw, n_out, t4, out, miss_bits);
}

BED_T512 static Counts avx512_decode_f64_wrap(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                              uint32_t n_out, const double* t4, double* out,
                                              uint64_t* miss_bits, bool want_counts) {
    (void)want_counts;   // AVX-512 fuses counting into the single decode pass
    return avx512::decode_f64(raw, keep, n_raw, n_out, t4, out, miss_bits);
}

#endif  // BED_X86

// ===========================================================================
// Dispatch
// ===========================================================================
struct Kernels {
    Isa isa;
    Counts (*count)(const uint8_t*, const uint64_t*, uint32_t, uint32_t);
    Counts (*i8)(const uint8_t*, const uint64_t*, uint32_t, uint32_t, uint32_t, int8_t*, I8Opts*, bool);
    Counts (*f32)(const uint8_t*, const uint64_t*, uint32_t, uint32_t, const float*, float*, uint64_t*, bool);
    Counts (*f64)(const uint8_t*, const uint64_t*, uint32_t, uint32_t, const double*, double*, uint64_t*, bool);
    void (*i8_kmajor_band)(const uint8_t*, size_t, const int*, int, const uint64_t*, uint32_t,
                           uint32_t, uint32_t, int8_t*, size_t, const double*, double*);
};

static const Kernels kScalar = {Isa::Scalar, &count_scalar, &scalar::decode_i8, &scalar::decode_f32, &scalar::decode_f64, &scalar::decode_i8_kmajor_band};
#if BED_ARM64
static const Kernels kNeon = {Isa::Neon, &neon::count, &neon::decode_i8, &neon::decode_f32, &neon::decode_f64, &neon::decode_i8_kmajor_band};
#endif
#if BED_X86
static const Kernels kAvx2 = {Isa::Avx2, &count_popcnt, &avx2::decode_i8, &avx2::decode_f32, &avx2::decode_f64, &avx2::decode_i8_kmajor_band};
static const Kernels kAvx512 = {Isa::Avx512, &avx512::count, &avx512::decode_i8, &avx512_decode_f32_wrap, &avx512_decode_f64_wrap, &avx512::decode_i8_kmajor_band};

static bool cpu_has_avx512() {
#if BED_STATIC_AVX512
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
static bool cpu_has_avx2() {
#if BED_STATIC_AVX2
    return true;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi") &&
           __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt") &&
           __builtin_cpu_supports("fma");
#endif
}
#endif

static const Kernels* kernels_for(Isa isa) {
    switch (isa) {
#if BED_X86
        case Isa::Avx512: return cpu_has_avx512() ? &kAvx512 : nullptr;
        case Isa::Avx2:   return cpu_has_avx2() ? &kAvx2 : nullptr;
#endif
#if BED_ARM64
        case Isa::Neon:   return &kNeon;
#endif
        case Isa::Scalar: return &kScalar;
        default:          return nullptr;
    }
}

static Isa detect_best() {
#if BED_X86
    if (cpu_has_avx512()) return Isa::Avx512;
    if (cpu_has_avx2()) return Isa::Avx2;
#endif
#if BED_ARM64
    return Isa::Neon;
#endif
    return Isa::Scalar;
}

static bool str_eq(const char* a, const char* b) {
    while (*a && *b) {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (ca != cb) return false;
        ++a; ++b;
    }
    return *a == *b;
}

static const Kernels* select_initial() {
    const Kernels* k = kernels_for(detect_best());
    if (const char* e = getenv("GCTA_BED_ISA")) {
        const Kernels* f = nullptr;
        if (str_eq(e, "scalar")) f = kernels_for(Isa::Scalar);
        else if (str_eq(e, "neon")) f = kernels_for(Isa::Neon);
        else if (str_eq(e, "avx2")) f = kernels_for(Isa::Avx2);
        else if (str_eq(e, "avx512")) f = kernels_for(Isa::Avx512);
        if (f) k = f;
    }
    return k ? k : &kScalar;
}

// Selected lazily on first use (no static-init-order dependence); set_isa()
// may override (tests / benchmarks).  Benign idempotent race -> atomics.
static const Kernels* g_kernels = nullptr;

static inline const Kernels& K() {
    const Kernels* k = __atomic_load_n(&g_kernels, __ATOMIC_ACQUIRE);
    if (__builtin_expect(k == nullptr, 0)) {
        k = select_initial();
        __atomic_store_n(&g_kernels, k, __ATOMIC_RELEASE);
    }
    return *k;
}

}  // namespace detail

using namespace detail;

const char* isa_name() {
    switch (K().isa) {
        case Isa::Avx512: return "avx512";
        case Isa::Avx2:   return "avx2";
        case Isa::Neon:   return "neon";
        default:          return "scalar";
    }
}
Isa active_isa() { return K().isa; }
Isa best_isa() { return detect_best(); }
bool set_isa(Isa isa) {
    const Kernels* k = kernels_for(isa);
    if (!k) return false;
    __atomic_store_n(&g_kernels, k, __ATOMIC_RELEASE);
    return true;
}

bool build_keep_mask(const uint32_t* idx, size_t n_idx, uint32_t n_raw, uint64_t* words) {
    const size_t nw = keep_mask_words(n_raw);
    for (size_t i = 0; i < nw; ++i) words[i] = 0;
    uint64_t prev = 0;
    for (size_t i = 0; i < n_idx; ++i) {
        const uint32_t r = idx[i];
        if (r >= n_raw) return false;
        if (i && (uint64_t)r + 1 <= prev) return false;
        prev = (uint64_t)r + 1;
        words[r >> 6] |= 1ull << (r & 63u);
    }
    return true;
}

Counts count(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep) {
    return K().count(raw, keep, n_raw, keep ? n_keep : n_raw);
}

Counts decode_i8(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                 int8_t* out, I8Opts* opts) {
    return K().i8(raw, keep, n_raw, 0, keep ? n_keep : n_raw, out, opts, true);
}

Counts decode_f32(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                  const float table[4], float* out, uint64_t* miss_bits) {
    return K().f32(raw, keep, n_raw, keep ? n_keep : n_raw, table, out, miss_bits, true);
}

Counts decode_f64(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                  const double table[4], double* out, uint64_t* miss_bits) {
    return K().f64(raw, keep, n_raw, keep ? n_keep : n_raw, table, out, miss_bits, true);
}

void decode_i8_nocount(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                       int8_t* out, I8Opts* opts) {
    K().i8(raw, keep, n_raw, 0, keep ? n_keep : n_raw, out, opts, false);
}

void decode_f32_nocount(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                        const float table[4], float* out, uint64_t* miss_bits) {
    K().f32(raw, keep, n_raw, keep ? n_keep : n_raw, table, out, miss_bits, false);
}

void decode_f64_nocount(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep,
                        const double table[4], double* out, uint64_t* miss_bits) {
    K().f64(raw, keep, n_raw, keep ? n_keep : n_raw, table, out, miss_bits, false);
}

void decode_i8_range(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                     uint32_t r_begin, uint32_t n_out, int8_t* out) {
    if (n_out == 0) return;
    K().i8(raw, keep, n_raw, r_begin, n_out, out, nullptr, false);
}

void decode_i8_kmajor_band(const uint8_t* raw_base, size_t rec_bytes,
                           const int* valid_idx, int K,
                           const uint64_t* keep, uint32_t n_raw,
                           uint32_t r_begin, uint32_t n_out,
                           int8_t* kmajor_out, size_t ldK,
                           const double* mu, double* r_out) {
    detail::K().i8_kmajor_band(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out,
                               kmajor_out, ldK, mu, r_out);
}

}  // namespace bed_simd
