// -----------------------------------------------------------------------------
// bed_simd_avx512.inc -- AVX-512 (Zen 4 / Ice Lake+) PLINK1 .bed kernels.
//
// Included by src/bed_simd.cpp (inside namespace bed_simd::detail) with
// BED_T512 = __attribute__((target("avx512f,...,avx512vbmi2,avx512vpopcntdq")))
// and <immintrin.h>.  tests/test_bed_simd.cpp also includes it against a
// portable intrinsic emulator (BED_T512 empty) so the exact same kernel logic
// can be validated bit-for-bit on non-AVX-512 hosts (e.g. Apple Silicon).
//
// Requires from the including context: Counts, I8Opts and src/bed_simd_internal.hpp.
// -----------------------------------------------------------------------------

namespace avx512 {

// vpermb indices: qword i of the expanded vector receives raw bytes (2i, 2i+1)
// (16 bits = samples 8i..8i+7) in its low two bytes.
alignas(64) static const uint8_t kExpandIdx[64] = {
     0, 1, 0, 1, 0, 1, 0, 1,   2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5,   6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9,  10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,  14,15,14,15,14,15,14,15};
// vpmultishiftqb controls: output byte p of each qword = bits [2p, 2p+8).
alignas(64) static const uint8_t kExpandCtl[64] = {
    0, 2, 4, 6, 8, 10, 12, 14,  0, 2, 4, 6, 8, 10, 12, 14,
    0, 2, 4, 6, 8, 10, 12, 14,  0, 2, 4, 6, 8, 10, 12, 14,
    0, 2, 4, 6, 8, 10, 12, 14,  0, 2, 4, 6, 8, 10, 12, 14,
    0, 2, 4, 6, 8, 10, 12, 14,  0, 2, 4, 6, 8, 10, 12, 14};
// vpshufb: PLINK1 code -> int8 value {00:2, 01:0(miss), 10:1, 11:0}.
alignas(16) static const uint8_t kValLut[16] = {2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// vpshufb: 4-bit nibble -> bits spread to even positions (pdep(n, 0x55)).
alignas(16) static const uint8_t kSpreadNib[16] = {0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
                                                   0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55};

// 64 raw samples (16 bytes, nbytes in [1,16] valid) -> 64 PLINK1 codes (0..3),
// one per byte, using VPERMB + VPMULTISHIFTQB.  Masked load: never touches
// memory past p + nbytes.
BED_T512 static inline __m512i load_codes64(const uint8_t* p, uint32_t nbytes,
                                            __m512i idx, __m512i ctl, __m512i three) {
    const __mmask16 m = nbytes >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << nbytes) - 1u);
    const __m128i x = _mm_maskz_loadu_epi8(m, p);
    const __m512i v = _mm512_permutexvar_epi8(idx, _mm512_zextsi128_si512(x));
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, v), three);
}

// ---------------------------------------------------------------------------
// Counting: VPSRLQ + 3x VPTERNLOGQ + 3x VPOPCNTQ per 256 samples.
// ---------------------------------------------------------------------------
BED_T512 static Counts count(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw, uint32_t n_keep) {
    const uint32_t nfull = n_raw >> 2;            // complete raw bytes
    const __m512i m55  = _mm512_set1_epi8(0x55);
    const __m512i nib  = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)kSpreadNib));
    const __m512i lo4  = _mm512_set1_epi16(0x000F);
    const __m512i hi4  = _mm512_set1_epi16(0x0F00);
    const uint8_t* kb  = (const uint8_t*)keep;
    __m512i a0 = _mm512_setzero_si512(), ah = _mm512_setzero_si512(), am = _mm512_setzero_si512();
    for (uint32_t off = 0; off < nfull; off += 64) {
        const uint32_t rem = nfull - off;
        const __mmask64 lm = rem >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << rem) - 1);
        const __m512i v = _mm512_maskz_loadu_epi8(lm, raw + off);
        const __m512i h = _mm512_srli_epi64(v, 1);
        __m512i k = m55;
        if (keep) {
            const uint32_t kr = rem >= 64 ? 32u : (rem + 1u) >> 1;
            const __mmask32 km = kr >= 32 ? ~(__mmask32)0 : (((__mmask32)1 << kr) - 1);
            const __m512i k16 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(km, kb + (off >> 1)));
            const __m512i t = _mm512_or_si512(_mm512_and_si512(k16, lo4),
                                              _mm512_and_si512(_mm512_slli_epi16(k16, 4), hi4));
            k = _mm512_shuffle_epi8(nib, t);   // 2-bit spread keep mask (even bits)
        }
        a0 = _mm512_add_epi64(a0, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(v, h, k, 0x80)));  //  lo &  hi
        ah = _mm512_add_epi64(ah, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(v, h, k, 0x08)));  // ~lo &  hi
        am = _mm512_add_epi64(am, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(v, h, k, 0x20)));  //  lo & ~hi
    }
    uint32_t n0 = (uint32_t)_mm512_reduce_add_epi64(a0);
    uint32_t nh = (uint32_t)_mm512_reduce_add_epi64(ah);
    uint32_t nm = (uint32_t)_mm512_reduce_add_epi64(am);
    const uint32_t tail = n_raw & 3u;
    if (tail) {
        const uint32_t b = raw[nfull];
        const uint32_t r0 = nfull << 2;
        for (uint32_t t = 0; t < tail; ++t) {
            const uint32_t r = r0 + t;
            if (keep && !((keep[r >> 6] >> (r & 63)) & 1u)) continue;
            const uint32_t c = (b >> (2 * t)) & 3u;
            n0 += (c == 3u); nh += (c == 2u); nm += (c == 1u);
        }
    }
    Counts out;
    out.n0 = n0; out.n1 = nh; out.nmiss = nm;
    out.n2 = (keep ? n_keep : n_raw) - n0 - nh - nm;
    return out;
}

// ---------------------------------------------------------------------------
// Fused int8 decode (+ counts, lazy missing outputs, optional weighted sums).
// 64 raw samples / iteration; VPCOMPRESSB packs kept samples in-register and a
// masked store writes exactly the produced bytes (no output slack needed).
// ---------------------------------------------------------------------------
template <bool SUBSET, bool WF, bool WD>
BED_T512 static Counts decode_i8_t(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                   uint32_t r_begin, uint32_t n_out, int8_t* out,
                                   MissSink* ms, const float* wf, const double* wd,
                                   double* wf_dot, double* wd_dot, double* wd_dot2) {
    const __m512i idx   = _mm512_load_si512((const void*)kExpandIdx);
    const __m512i ctl   = _mm512_load_si512((const void*)kExpandCtl);
    const __m512i three = _mm512_set1_epi8(3);
    const __m512i k1    = _mm512_set1_epi8(1);
    const __m512i k2    = _mm512_set1_epi8(2);
    const __m512i lut   = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)kValLut));
    const uint32_t nbytes = (n_raw + 3u) >> 2;
    uint64_t c1 = 0, c2 = 0, cm = 0;
    __m512d af0 = _mm512_setzero_pd(), af1 = _mm512_setzero_pd();
    __m512d ag  = _mm512_setzero_pd(), ag2 = _mm512_setzero_pd();
    const __m512d dthree = _mm512_set1_pd(3.0);
    uint32_t pos = 0;
    uint32_t w = r_begin >> 6;
    uint64_t first = SUBSET ? (~0ull << (r_begin & 63u)) : ~0ull;
    while (pos < n_out) {
        const uint32_t b0 = w << 4;
        if (b0 >= nbytes) break;
        uint32_t cnt;
        uint64_t kw = ~0ull;
        if constexpr (SUBSET) {
            kw = keep[w] & first;
            first = ~0ull;
            if (!kw) { ++w; continue; }
            cnt = popcnt64(kw);
            if (cnt > n_out - pos) {
                cnt = n_out - pos;
                kw = _pdep_u64((1ull << cnt) - 1ull, kw);   // lowest cnt kept samples
            }
        } else {
            const uint32_t rem = n_out - pos;
            cnt = rem < 64u ? rem : 64u;
        }
        const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
        __m512i c = load_codes64(raw + b0, nb, idx, ctl, three);
        if constexpr (SUBSET) c = _mm512_maskz_compress_epi8(kw, c);
        const uint64_t valid = cnt >= 64u ? ~0ull : ((1ull << cnt) - 1ull);
        _mm512_mask_storeu_epi8(out + pos, valid, _mm512_shuffle_epi8(lut, c));
        const uint64_t mh = _mm512_cmpeq_epi8_mask(c, k2) & valid;   // het      (value 1)
        const uint64_t m2 = _mm512_testn_epi8_mask(c, c) & valid;    // hom A1   (value 2)
        const uint64_t mm = _mm512_cmpeq_epi8_mask(c, k1) & valid;   // missing
        c1 += popcnt64(mh);
        c2 += popcnt64(m2);
        if (mm) {                                                     // rare on imputed data
            cm += popcnt64(mm);
            if (ms) ms->add(mm, pos);
        }
        if constexpr (WF) {
            const uint64_t m12 = mh | m2;
            for (int s = 0; s < 4; ++s) {
                const __mmask16 a = (__mmask16)(m12 >> (16 * s));
                if (!a) continue;
                const __mmask16 b = (__mmask16)(m2 >> (16 * s));
                __m512 x = _mm512_maskz_loadu_ps(a, wf + pos + 16 * s);
                x = _mm512_mask_add_ps(x, b, x, x);                  // g * w (exact in fp32)
                af0 = _mm512_add_pd(af0, _mm512_cvtps_pd(_mm512_castps512_ps256(x)));
                af1 = _mm512_add_pd(af1, _mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1)));
            }
        }
        if constexpr (WD) {
            const uint64_t m12 = mh | m2;
            for (int s = 0; s < 8; ++s) {
                const __mmask8 a = (__mmask8)(m12 >> (8 * s));
                if (!a) continue;
                const __mmask8 b = (__mmask8)(m2 >> (8 * s));
                const __m512d x = _mm512_maskz_loadu_pd(a, wd + pos + 8 * s);
                const __m512d y = _mm512_maskz_mov_pd(b, x);
                ag  = _mm512_add_pd(ag, _mm512_add_pd(x, y));          // g   * w
                ag2 = _mm512_add_pd(ag2, _mm512_fmadd_pd(y, dthree, x)); // g^2 * w
            }
        }
        pos += cnt;
        ++w;
    }
    if constexpr (WF) *wf_dot = _mm512_reduce_add_pd(_mm512_add_pd(af0, af1));
    if constexpr (WD) { *wd_dot = _mm512_reduce_add_pd(ag); *wd_dot2 = _mm512_reduce_add_pd(ag2); }
    Counts r;
    r.n1 = (uint32_t)c1; r.n2 = (uint32_t)c2; r.nmiss = (uint32_t)cm;
    r.n0 = pos - r.n1 - r.n2 - r.nmiss;
    return r;
}

// ---------------------------------------------------------------------------
// Register-table float / double decode.
// ---------------------------------------------------------------------------
#define BED_F32_SEG(S)                                                                        \
    if (cnt > 16u * (S)) {                                                                    \
        const __m512 v = _mm512_permutexvar_ps(                                               \
            _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(c, (S))), tbl);                    \
        _mm512_mask_storeu_ps(out + pos + 16u * (S), (__mmask16)(valid >> (16 * (S))), v);    \
    }

template <bool SUBSET>
BED_T512 static Counts decode_f32_t(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                    uint32_t n_out, const float* t4, float* out, MissSink* ms) {
    const __m512i idx   = _mm512_load_si512((const void*)kExpandIdx);
    const __m512i ctl   = _mm512_load_si512((const void*)kExpandCtl);
    const __m512i three = _mm512_set1_epi8(3);
    const __m512i k1    = _mm512_set1_epi8(1);
    const __m512i k2    = _mm512_set1_epi8(2);
    // Indexed by PLINK1 code: 00 -> v2, 01 -> v_miss, 10 -> v1, 11 -> v0.
    alignas(64) float tb[16] = {t4[2], t4[3], t4[1], t4[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const __m512 tbl = _mm512_load_ps(tb);
    const uint32_t nbytes = (n_raw + 3u) >> 2;
    uint64_t c1 = 0, c2 = 0, cm = 0;
    uint32_t pos = 0, w = 0;
    while (pos < n_out) {
        const uint32_t b0 = w << 4;
        if (b0 >= nbytes) break;
        uint32_t cnt;
        uint64_t kw = ~0ull;
        if constexpr (SUBSET) {
            kw = keep[w];
            if (!kw) { ++w; continue; }
            cnt = popcnt64(kw);
            if (cnt > n_out - pos) { cnt = n_out - pos; kw = _pdep_u64((1ull << cnt) - 1ull, kw); }
        } else {
            const uint32_t rem = n_out - pos;
            cnt = rem < 64u ? rem : 64u;
        }
        const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
        __m512i c = load_codes64(raw + b0, nb, idx, ctl, three);
        if constexpr (SUBSET) c = _mm512_maskz_compress_epi8(kw, c);
        const uint64_t valid = cnt >= 64u ? ~0ull : ((1ull << cnt) - 1ull);
        BED_F32_SEG(0) BED_F32_SEG(1) BED_F32_SEG(2) BED_F32_SEG(3)
        c1 += popcnt64(_mm512_cmpeq_epi8_mask(c, k2) & valid);
        c2 += popcnt64(_mm512_testn_epi8_mask(c, c) & valid);
        const uint64_t mm = _mm512_cmpeq_epi8_mask(c, k1) & valid;
        if (mm) { cm += popcnt64(mm); if (ms) ms->add(mm, pos); }
        pos += cnt;
        ++w;
    }
    Counts r;
    r.n1 = (uint32_t)c1; r.n2 = (uint32_t)c2; r.nmiss = (uint32_t)cm;
    r.n0 = pos - r.n1 - r.n2 - r.nmiss;
    return r;
}
#undef BED_F32_SEG

#define BED_F64_SEG(S)                                                                        \
    if (cnt > 8u * (S)) {                                                                     \
        __m128i lane = _mm512_extracti32x4_epi32(c, (S) / 2);                                 \
        if ((S) & 1) lane = _mm_srli_si128(lane, 8);                                          \
        const __m512d v = _mm512_permutexvar_pd(_mm512_cvtepu8_epi64(lane), tbl);             \
        _mm512_mask_storeu_pd(out + pos + 8u * (S), (__mmask8)(valid >> (8 * (S))), v);       \
    }

template <bool SUBSET>
BED_T512 static Counts decode_f64_t(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                    uint32_t n_out, const double* t4, double* out, MissSink* ms) {
    const __m512i idx   = _mm512_load_si512((const void*)kExpandIdx);
    const __m512i ctl   = _mm512_load_si512((const void*)kExpandCtl);
    const __m512i three = _mm512_set1_epi8(3);
    const __m512i k1    = _mm512_set1_epi8(1);
    const __m512i k2    = _mm512_set1_epi8(2);
    alignas(64) double tb[8] = {t4[2], t4[3], t4[1], t4[0], 0, 0, 0, 0};
    const __m512d tbl = _mm512_load_pd(tb);
    const uint32_t nbytes = (n_raw + 3u) >> 2;
    uint64_t c1 = 0, c2 = 0, cm = 0;
    uint32_t pos = 0, w = 0;
    while (pos < n_out) {
        const uint32_t b0 = w << 4;
        if (b0 >= nbytes) break;
        uint32_t cnt;
        uint64_t kw = ~0ull;
        if constexpr (SUBSET) {
            kw = keep[w];
            if (!kw) { ++w; continue; }
            cnt = popcnt64(kw);
            if (cnt > n_out - pos) { cnt = n_out - pos; kw = _pdep_u64((1ull << cnt) - 1ull, kw); }
        } else {
            const uint32_t rem = n_out - pos;
            cnt = rem < 64u ? rem : 64u;
        }
        const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
        __m512i c = load_codes64(raw + b0, nb, idx, ctl, three);
        if constexpr (SUBSET) c = _mm512_maskz_compress_epi8(kw, c);
        const uint64_t valid = cnt >= 64u ? ~0ull : ((1ull << cnt) - 1ull);
        BED_F64_SEG(0) BED_F64_SEG(1) BED_F64_SEG(2) BED_F64_SEG(3)
        BED_F64_SEG(4) BED_F64_SEG(5) BED_F64_SEG(6) BED_F64_SEG(7)
        c1 += popcnt64(_mm512_cmpeq_epi8_mask(c, k2) & valid);
        c2 += popcnt64(_mm512_testn_epi8_mask(c, c) & valid);
        const uint64_t mm = _mm512_cmpeq_epi8_mask(c, k1) & valid;
        if (mm) { cm += popcnt64(mm); if (ms) ms->add(mm, pos); }
        pos += cnt;
        ++w;
    }
    Counts r;
    r.n1 = (uint32_t)c1; r.n2 = (uint32_t)c2; r.nmiss = (uint32_t)cm;
    r.n0 = pos - r.n1 - r.n2 - r.nmiss;
    return r;
}
#undef BED_F64_SEG

// ---------------------------------------------------------------------------
// Entry points (non-template, uniform signatures for the dispatch table).
// ---------------------------------------------------------------------------
BED_T512 static Counts decode_i8(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                 uint32_t r_begin, uint32_t n_out, int8_t* out, I8Opts* o,
                                 bool want_counts = true) {
    (void)want_counts;   // fused loop only visits n_out samples anyway
    MissSink sink;
    MissSink* ms = nullptr;
    const float* wf = nullptr;
    const double* wd = nullptr;
    if (o) {
        sink.init(o, n_out);
        ms = sink.active() ? &sink : nullptr;
        wf = o->wf;
        wd = o->wd;
    }
    double f_dot = 0.0, d_dot = 0.0, d_dot2 = 0.0;
    Counts c;
    const int sel = (keep ? 4 : 0) | (wf ? 2 : 0) | (wd ? 1 : 0);
    switch (sel) {
        case 0: c = decode_i8_t<false, false, false>(raw, keep, n_raw, r_begin, n_out, out, ms, wf, wd, &f_dot, &d_dot, &d_dot2); break;
        case 1: c = decode_i8_t<false, false, true >(raw, keep, n_raw, r_begin, n_out, out, ms, wf, wd, &f_dot, &d_dot, &d_dot2); break;
        case 2: c = decode_i8_t<false, true,  false>(raw, keep, n_raw, r_begin, n_out, out, ms, wf, wd, &f_dot, &d_dot, &d_dot2); break;
        case 3: c = decode_i8_t<false, true,  true >(raw, keep, n_raw, r_begin, n_out, out, ms, wf, wd, &f_dot, &d_dot, &d_dot2); break;
        case 4: c = decode_i8_t<true,  false, false>(raw, keep, n_raw, r_begin, n_out, out, ms, wf, wd, &f_dot, &d_dot, &d_dot2); break;
        case 5: c = decode_i8_t<true,  false, true >(raw, keep, n_raw, r_begin, n_out, out, ms, wf, wd, &f_dot, &d_dot, &d_dot2); break;
        case 6: c = decode_i8_t<true,  true,  false>(raw, keep, n_raw, r_begin, n_out, out, ms, wf, wd, &f_dot, &d_dot, &d_dot2); break;
        default: c = decode_i8_t<true, true,  true >(raw, keep, n_raw, r_begin, n_out, out, ms, wf, wd, &f_dot, &d_dot, &d_dot2); break;
    }
    if (o) {
        o->wf_dot = f_dot;
        o->wd_dot = d_dot;
        o->wd_dot2 = d_dot2;
        sink.finish(o);
    }
    return c;
}

BED_T512 static Counts decode_f32(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                  uint32_t n_out, const float* t4, float* out, uint64_t* miss_bits) {
    MissSink sink;
    sink.init_bits(miss_bits, n_out);
    MissSink* ms = miss_bits ? &sink : nullptr;
    return keep ? decode_f32_t<true>(raw, keep, n_raw, n_out, t4, out, ms)
                : decode_f32_t<false>(raw, keep, n_raw, n_out, t4, out, ms);
}

BED_T512 static Counts decode_f64(const uint8_t* raw, const uint64_t* keep, uint32_t n_raw,
                                  uint32_t n_out, const double* t4, double* out, uint64_t* miss_bits) {
    MissSink sink;
    sink.init_bits(miss_bits, n_out);
    MissSink* ms = miss_bits ? &sink : nullptr;
    return keep ? decode_f64_t<true>(raw, keep, n_raw, n_out, t4, out, ms)
                : decode_f64_t<false>(raw, keep, n_raw, n_out, t4, out, ms);
}

// ---------------------------------------------------------------------------
// Fused K-major int8 band decode (+ 64x64 in-register transpose + exact r_out).
// Sub-tiles n_out into 256-sample bands (= 64B raw cache line / marker) and
// 64-marker tiles -> 4 KiB tr64[64][8] transpose buffer in L1D.
// ---------------------------------------------------------------------------
struct KeepSpan64 {
    uint64_t kw;
    uint32_t w;
    uint8_t  pos;
    uint8_t  cnt;
};

struct KeepChunkDesc {
    KeepSpan64 spans[64];
    uint32_t   n_spans;
    uint32_t   s_len;
};

BED_T512 static inline void transpose_8x64_to_qwords(const __m512i r[8],
                                                     uint64_t tr64[64][8],
                                                     int mg) {
    const __m512i a0 = _mm512_unpacklo_epi8(r[0], r[1]);
    const __m512i a1 = _mm512_unpackhi_epi8(r[0], r[1]);
    const __m512i a2 = _mm512_unpacklo_epi8(r[2], r[3]);
    const __m512i a3 = _mm512_unpackhi_epi8(r[2], r[3]);
    const __m512i a4 = _mm512_unpacklo_epi8(r[4], r[5]);
    const __m512i a5 = _mm512_unpackhi_epi8(r[4], r[5]);
    const __m512i a6 = _mm512_unpacklo_epi8(r[6], r[7]);
    const __m512i a7 = _mm512_unpackhi_epi8(r[6], r[7]);

    const __m512i b0 = _mm512_unpacklo_epi16(a0, a2);
    const __m512i b1 = _mm512_unpackhi_epi16(a0, a2);
    const __m512i b2 = _mm512_unpacklo_epi16(a1, a3);
    const __m512i b3 = _mm512_unpackhi_epi16(a1, a3);
    const __m512i b4 = _mm512_unpacklo_epi16(a4, a6);
    const __m512i b5 = _mm512_unpackhi_epi16(a4, a6);
    const __m512i b6 = _mm512_unpacklo_epi16(a5, a7);
    const __m512i b7 = _mm512_unpackhi_epi16(a5, a7);

    alignas(64) uint64_t tmp[8][8];
    _mm512_store_si512((void*)tmp[0], _mm512_unpacklo_epi32(b0, b4));
    _mm512_store_si512((void*)tmp[1], _mm512_unpackhi_epi32(b0, b4));
    _mm512_store_si512((void*)tmp[2], _mm512_unpacklo_epi32(b1, b5));
    _mm512_store_si512((void*)tmp[3], _mm512_unpackhi_epi32(b1, b5));
    _mm512_store_si512((void*)tmp[4], _mm512_unpacklo_epi32(b2, b6));
    _mm512_store_si512((void*)tmp[5], _mm512_unpackhi_epi32(b2, b6));
    _mm512_store_si512((void*)tmp[6], _mm512_unpacklo_epi32(b3, b7));
    _mm512_store_si512((void*)tmp[7], _mm512_unpackhi_epi32(b3, b7));

    for (int l = 0; l < 4; ++l) {
        for (int t = 0; t < 8; ++t) {
            tr64[16 * l + 2 * t + 0][mg] = tmp[t][2 * l + 0];
            tr64[16 * l + 2 * t + 1][mg] = tmp[t][2 * l + 1];
        }
    }
}

template <bool SUBSET, bool WANT_R>
BED_T512 static void decode_i8_kmajor_band_t(const uint8_t* raw_base, size_t rec_bytes,
                                             const int* valid_idx, int K,
                                             const uint64_t* keep, uint32_t n_raw,
                                             uint32_t r_begin, uint32_t n_out,
                                             int8_t* kmajor_out, size_t ldK,
                                             const double* mu, double* r_out) {
    if (K <= 0 || n_out == 0u) return;
    if constexpr (WANT_R) {
        for (uint32_t s = 0; s < n_out; ++s) r_out[s] = 0.0;
    }

    const __m512i idx   = _mm512_load_si512((const void*)kExpandIdx);
    const __m512i ctl   = _mm512_load_si512((const void*)kExpandCtl);
    const __m512i three = _mm512_set1_epi8(3);
    const __m512i val1  = _mm512_set1_epi8(1);
    const __m512i val2  = _mm512_set1_epi8(2);
    const __m512i lut   = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)kValLut));
    const uint32_t nbytes = (n_raw + 3u) >> 2;
    const uint32_t nw = (n_raw + 63u) >> 6;

    uint32_t band_w = SUBSET ? (r_begin >> 6) : 0u;
    uint64_t band_first = SUBSET ? (~0ull << (r_begin & 63u)) : ~0ull;

    constexpr uint32_t kBand = 256u;
    for (uint32_t b_off = 0; b_off < n_out; b_off += kBand) {
        const uint32_t rem_band = n_out - b_off;
        const uint32_t b_len = rem_band < kBand ? rem_band : kBand;
        const uint32_t n_chunks = (b_len + 63u) >> 6;

        KeepChunkDesc chunks[4];
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            const uint32_t s0 = ci * 64u;
            const uint32_t want = (b_len - s0) < 64u ? (b_len - s0) : 64u;
            chunks[ci].n_spans = 0u;
            if constexpr (SUBSET) {
                uint32_t collected = 0u;
                while (collected < want && band_w < nw) {
                    const uint64_t kw = keep[band_w] & band_first;
                    if (!kw) {
                        ++band_w;
                        band_first = ~0ull;
                        continue;
                    }
                    const uint32_t cnt = popcnt64(kw);
                    const uint32_t need = want - collected;
                    if (cnt > need) {
                        const uint64_t sub_kw = _pdep_u64((1ull << need) - 1ull, kw);
                        chunks[ci].spans[chunks[ci].n_spans++] =
                            KeepSpan64{sub_kw, band_w, (uint8_t)collected, (uint8_t)need};
                        band_first &= ~sub_kw;
                        collected += need;
                    } else {
                        chunks[ci].spans[chunks[ci].n_spans++] =
                            KeepSpan64{kw, band_w, (uint8_t)collected, (uint8_t)cnt};
                        ++band_w;
                        band_first = ~0ull;
                        collected += cnt;
                    }
                }
                chunks[ci].s_len = collected;
            } else {
                chunks[ci].s_len = want;
            }
        }

        alignas(64) uint64_t tr64[64][8];
        alignas(64) int8_t rbuf[64];

        for (int m0 = 0; m0 < K; m0 += 64) {
            const int m_len = (K - m0) < 64 ? (K - m0) : 64;
            const __mmask64 write_mask = (m_len == 64) ? ~(__mmask64)0 : (((__mmask64)1 << m_len) - 1u);

            for (uint32_t ci = 0; ci < n_chunks; ++ci) {
                const uint32_t s0 = ci * 64u;
                const uint32_t s_len = chunks[ci].s_len;
                if (s_len == 0u) continue;
                const uint64_t valid = (s_len >= 64u) ? ~0ull : ((1ull << s_len) - 1ull);

                __m512d rv[8];
                if constexpr (WANT_R) {
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t lane0 = (uint32_t)p * 8u;
                        if (lane0 + 8u <= s_len) {
                            rv[p] = _mm512_loadu_pd(r_out + b_off + s0 + lane0);
                        } else if (lane0 < s_len) {
                            const __mmask8 km = (__mmask8)((1u << (s_len - lane0)) - 1u);
                            rv[p] = _mm512_maskz_loadu_pd(km, r_out + b_off + s0 + lane0);
                        } else {
                            rv[p] = _mm512_setzero_pd();
                        }
                    }
                }

                for (int mg = 0; mg * 8 < m_len; ++mg) {
                    const int g_rem = m_len - mg * 8;
                    const int g_len = g_rem < 8 ? g_rem : 8;
                    __m512i r[8];
                    for (int k = 0; k < g_len; ++k) {
                        const int dm = mg * 8 + k;
                        const uint8_t* raw = raw_base + (size_t)valid_idx[m0 + dm] * rec_bytes;
                        __m512i v;
                        if constexpr (SUBSET) {
                            _mm512_store_si512((void*)rbuf, _mm512_setzero_si512());
                            const KeepChunkDesc& cd = chunks[ci];
                            for (uint32_t si = 0; si < cd.n_spans; ++si) {
                                const KeepSpan64& sp = cd.spans[si];
                                const uint32_t b0 = sp.w << 4;
                                const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
                                const __m512i c = _mm512_maskz_compress_epi8(
                                    sp.kw, load_codes64(raw + b0, nb, idx, ctl, three));
                                const __m512i vals = _mm512_shuffle_epi8(lut, c);
                                const uint64_t smask = sp.cnt >= 64u ? ~0ull : ((1ull << sp.cnt) - 1ull);
                                _mm512_mask_storeu_epi8(rbuf + sp.pos, smask, vals);
                            }
                            v = _mm512_load_si512((const void*)rbuf);
                        } else {
                            const uint32_t b0 = (r_begin + b_off + s0) >> 2;
                            const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
                            const __m512i c = load_codes64(raw + b0, nb, idx, ctl, three);
                            v = _mm512_shuffle_epi8(lut, c);
                        }
                        if constexpr (WANT_R) {
                            const uint64_t m1 = _mm512_cmpeq_epi8_mask(v, val1) & valid;
                            const uint64_t m2 = _mm512_cmpeq_epi8_mask(v, val2) & valid;
                            if (m1 | m2) {
                                const double mu_m = mu[m0 + dm];
                                const __m512d vmu1 = _mm512_set1_pd(mu_m);
                                const __m512d vmu2 = _mm512_set1_pd(2.0 * mu_m);
                                for (int p = 0; p < 8; ++p) {
                                    const __mmask8 mk1 = (__mmask8)(m1 >> (8 * p));
                                    const __mmask8 mk2 = (__mmask8)(m2 >> (8 * p));
                                    if (mk1) rv[p] = _mm512_mask_add_pd(rv[p], mk1, rv[p], vmu1);
                                    if (mk2) rv[p] = _mm512_mask_add_pd(rv[p], mk2, rv[p], vmu2);
                                }
                            }
                        }
                        r[k] = v;
                    }
                    for (int k = g_len; k < 8; ++k) {
                        r[k] = _mm512_setzero_si512();
                    }
                    transpose_8x64_to_qwords(r, tr64, mg);
                }

                if constexpr (WANT_R) {
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t lane0 = (uint32_t)p * 8u;
                        if (lane0 + 8u <= s_len) {
                            _mm512_storeu_pd(r_out + b_off + s0 + lane0, rv[p]);
                        } else if (lane0 < s_len) {
                            const __mmask8 km = (__mmask8)((1u << (s_len - lane0)) - 1u);
                            _mm512_mask_storeu_pd(r_out + b_off + s0 + lane0, km, rv[p]);
                        }
                    }
                }

                for (uint32_t s = 0; s < s_len; ++s) {
                    int8_t* dst = kmajor_out + (size_t)(b_off + s0 + s) * ldK + m0;
                    const __m512i row = _mm512_load_si512((const void*)tr64[s]);
                    if (m_len == 64) {
                        _mm512_storeu_si512((void*)dst, row);
                    } else {
                        _mm512_mask_storeu_epi8((void*)dst, write_mask, row);
                    }
                }
            }
        }
    }
}

BED_T512 static void decode_i8_kmajor_band(const uint8_t* raw_base, size_t rec_bytes,
                                           const int* valid_idx, int K,
                                           const uint64_t* keep, uint32_t n_raw,
                                           uint32_t r_begin, uint32_t n_out,
                                           int8_t* kmajor_out, size_t ldK,
                                           const double* mu, double* r_out) {
    const int sel = (keep ? 2 : 0) | ((mu && r_out) ? 1 : 0);
    switch (sel) {
        case 0: decode_i8_kmajor_band_t<false, false>(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, mu, r_out); break;
        case 1: decode_i8_kmajor_band_t<false, true >(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, mu, r_out); break;
        case 2: decode_i8_kmajor_band_t<true,  false>(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, mu, r_out); break;
        default: decode_i8_kmajor_band_t<true, true >(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, mu, r_out); break;
    }
}

template <bool SUBSET, bool WANT_R, bool WANT_DERR>
BED_T512 static void decode_i8_kmajor_band_quant_t(const uint8_t* raw_base, size_t rec_bytes,
                                                   const int* valid_idx, int K,
                                                   const uint64_t* keep, uint32_t n_raw,
                                                   uint32_t r_begin, uint32_t n_out,
                                                   int8_t* kmajor_out, size_t ldK,
                                                   int8_t q, const int8_t* miss_q,
                                                   const double* mu_q, double* r_out,
                                                   const double* miss_err_q2, double* diag_err_out) {
    if (K <= 0 || n_out == 0u) return;
    if constexpr (WANT_R) {
        for (uint32_t s = 0; s < n_out; ++s) r_out[s] = 0.0;
    }
    if constexpr (WANT_DERR) {
        for (uint32_t s = 0; s < n_out; ++s) diag_err_out[s] = 0.0;
    }

    const __m512i idx    = _mm512_load_si512((const void*)kExpandIdx);
    const __m512i ctl    = _mm512_load_si512((const void*)kExpandCtl);
    const __m512i three  = _mm512_set1_epi8(3);
    const __m512i c0_vec = _mm512_setzero_si512();
    const __m512i c1_vec = _mm512_set1_epi8(1);
    const __m512i c2_vec = _mm512_set1_epi8(2);
    const uint32_t nbytes = (n_raw + 3u) >> 2;
    const uint32_t nw = (n_raw + 63u) >> 6;

    uint32_t band_w = SUBSET ? (r_begin >> 6) : 0u;
    uint64_t band_first = SUBSET ? (~0ull << (r_begin & 63u)) : ~0ull;

    constexpr uint32_t kBand = 256u;
    for (uint32_t b_off = 0; b_off < n_out; b_off += kBand) {
        const uint32_t rem_band = n_out - b_off;
        const uint32_t b_len = rem_band < kBand ? rem_band : kBand;
        const uint32_t n_chunks = (b_len + 63u) >> 6;

        KeepChunkDesc chunks[4];
        for (uint32_t ci = 0; ci < n_chunks; ++ci) {
            const uint32_t s0 = ci * 64u;
            const uint32_t want = (b_len - s0) < 64u ? (b_len - s0) : 64u;
            chunks[ci].n_spans = 0u;
            if constexpr (SUBSET) {
                uint32_t collected = 0u;
                while (collected < want && band_w < nw) {
                    const uint64_t kw = keep[band_w] & band_first;
                    if (!kw) {
                        ++band_w;
                        band_first = ~0ull;
                        continue;
                    }
                    const uint32_t cnt = popcnt64(kw);
                    const uint32_t need = want - collected;
                    if (cnt > need) {
                        const uint64_t sub_kw = _pdep_u64((1ull << need) - 1ull, kw);
                        chunks[ci].spans[chunks[ci].n_spans++] =
                            KeepSpan64{sub_kw, band_w, (uint8_t)collected, (uint8_t)need};
                        band_first &= ~sub_kw;
                        collected += need;
                    } else {
                        chunks[ci].spans[chunks[ci].n_spans++] =
                            KeepSpan64{kw, band_w, (uint8_t)collected, (uint8_t)cnt};
                        ++band_w;
                        band_first = ~0ull;
                        collected += cnt;
                    }
                }
                chunks[ci].s_len = collected;
            } else {
                chunks[ci].s_len = want;
            }
        }

        alignas(64) uint64_t tr64[64][8];
        alignas(64) uint8_t cbuf[64];

        for (int m0 = 0; m0 < K; m0 += 64) {
            const int m_len = (K - m0) < 64 ? (K - m0) : 64;
            const __mmask64 write_mask = (m_len == 64) ? ~(__mmask64)0 : (((__mmask64)1 << m_len) - 1u);

            for (uint32_t ci = 0; ci < n_chunks; ++ci) {
                const uint32_t s0 = ci * 64u;
                const uint32_t s_len = chunks[ci].s_len;
                if (s_len == 0u) continue;
                const uint64_t valid = (s_len >= 64u) ? ~0ull : ((1ull << s_len) - 1ull);

                __m512d rv[8];
                if constexpr (WANT_R) {
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t lane0 = (uint32_t)p * 8u;
                        if (lane0 + 8u <= s_len) {
                            rv[p] = _mm512_loadu_pd(r_out + b_off + s0 + lane0);
                        } else if (lane0 < s_len) {
                            const __mmask8 km = (__mmask8)((1u << (s_len - lane0)) - 1u);
                            rv[p] = _mm512_maskz_loadu_pd(km, r_out + b_off + s0 + lane0);
                        } else {
                            rv[p] = _mm512_setzero_pd();
                        }
                    }
                }
                __m512d dv[8];
                if constexpr (WANT_DERR) {
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t lane0 = (uint32_t)p * 8u;
                        if (lane0 + 8u <= s_len) {
                            dv[p] = _mm512_loadu_pd(diag_err_out + b_off + s0 + lane0);
                        } else if (lane0 < s_len) {
                            const __mmask8 km = (__mmask8)((1u << (s_len - lane0)) - 1u);
                            dv[p] = _mm512_maskz_loadu_pd(km, diag_err_out + b_off + s0 + lane0);
                        } else {
                            dv[p] = _mm512_setzero_pd();
                        }
                    }
                }

                for (int mg = 0; mg * 8 < m_len; ++mg) {
                    const int g_rem = m_len - mg * 8;
                    const int g_len = g_rem < 8 ? g_rem : 8;
                    __m512i r[8];
                    for (int k = 0; k < g_len; ++k) {
                        const int dm = mg * 8 + k;
                        const int m = m0 + dm;
                        const uint8_t* raw = raw_base + (size_t)valid_idx[m] * rec_bytes;
                        __m512i c;
                        if constexpr (SUBSET) {
                            _mm512_store_si512((void*)cbuf, three);
                            const KeepChunkDesc& cd = chunks[ci];
                            for (uint32_t si = 0; si < cd.n_spans; ++si) {
                                const KeepSpan64& sp = cd.spans[si];
                                const uint32_t b0 = sp.w << 4;
                                const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
                                const __m512i cs = _mm512_maskz_compress_epi8(
                                    sp.kw, load_codes64(raw + b0, nb, idx, ctl, three));
                                const uint64_t smask = sp.cnt >= 64u ? ~0ull : ((1ull << sp.cnt) - 1ull);
                                _mm512_mask_storeu_epi8(cbuf + sp.pos, smask, cs);
                            }
                            c = _mm512_load_si512((const void*)cbuf);
                        } else {
                            const uint32_t b0 = (r_begin + b_off + s0) >> 2;
                            const uint32_t nb = (nbytes - b0) < 16u ? (nbytes - b0) : 16u;
                            c = load_codes64(raw + b0, nb, idx, ctl, three);
                        }
                        const int8_t mq = miss_q[m];
                        const uint32_t lw = (uint32_t)(uint8_t)(2 * (int)q) |
                                            ((uint32_t)(uint8_t)mq << 8) |
                                            ((uint32_t)(uint8_t)q << 16);
                        const __m512i lut_m = _mm512_set1_epi32((int)lw);
                        const __m512i v = _mm512_shuffle_epi8(lut_m, c);
                        uint64_t m_miss = 0;
                        if constexpr (WANT_R || WANT_DERR) {
                            m_miss = _mm512_cmpeq_epi8_mask(c, c1_vec) & valid;
                        }
                        if constexpr (WANT_R) {
                            const uint64_t m_hom2 = _mm512_cmpeq_epi8_mask(c, c0_vec) & valid;
                            const uint64_t m_het  = _mm512_cmpeq_epi8_mask(c, c2_vec) & valid;
                            if (m_hom2 | m_het | (mq != 0 ? m_miss : 0ull)) {
                                const double mu_m = mu_q[m];
                                const __m512d vq2 = _mm512_set1_pd((double)(2 * (int)q) * mu_m);
                                const __m512d vq1 = _mm512_set1_pd((double)q * mu_m);
                                const __m512d vqm = _mm512_set1_pd((double)mq * mu_m);
                                for (int p = 0; p < 8; ++p) {
                                    const __mmask8 mk2 = (__mmask8)(m_hom2 >> (8 * p));
                                    const __mmask8 mk1 = (__mmask8)(m_het >> (8 * p));
                                    const __mmask8 mkm = (__mmask8)(m_miss >> (8 * p));
                                    if (mk2) rv[p] = _mm512_mask_add_pd(rv[p], mk2, rv[p], vq2);
                                    if (mk1) rv[p] = _mm512_mask_add_pd(rv[p], mk1, rv[p], vq1);
                                    if (mkm && mq != 0) rv[p] = _mm512_mask_add_pd(rv[p], mkm, rv[p], vqm);
                                }
                            }
                        }
                        if constexpr (WANT_DERR) {
                            if (m_miss) {
                                const __m512d vde = _mm512_set1_pd(miss_err_q2[m]);
                                for (int p = 0; p < 8; ++p) {
                                    const __mmask8 mkm = (__mmask8)(m_miss >> (8 * p));
                                    if (mkm) dv[p] = _mm512_mask_add_pd(dv[p], mkm, dv[p], vde);
                                }
                            }
                        }
                        r[k] = v;
                    }
                    for (int k = g_len; k < 8; ++k) {
                        r[k] = _mm512_setzero_si512();
                    }
                    transpose_8x64_to_qwords(r, tr64, mg);
                }

                if constexpr (WANT_R) {
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t lane0 = (uint32_t)p * 8u;
                        if (lane0 + 8u <= s_len) {
                            _mm512_storeu_pd(r_out + b_off + s0 + lane0, rv[p]);
                        } else if (lane0 < s_len) {
                            const __mmask8 km = (__mmask8)((1u << (s_len - lane0)) - 1u);
                            _mm512_mask_storeu_pd(r_out + b_off + s0 + lane0, km, rv[p]);
                        }
                    }
                }
                if constexpr (WANT_DERR) {
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t lane0 = (uint32_t)p * 8u;
                        if (lane0 + 8u <= s_len) {
                            _mm512_storeu_pd(diag_err_out + b_off + s0 + lane0, dv[p]);
                        } else if (lane0 < s_len) {
                            const __mmask8 km = (__mmask8)((1u << (s_len - lane0)) - 1u);
                            _mm512_mask_storeu_pd(diag_err_out + b_off + s0 + lane0, km, dv[p]);
                        }
                    }
                }

                for (uint32_t s = 0; s < s_len; ++s) {
                    int8_t* dst = kmajor_out + (size_t)(b_off + s0 + s) * ldK + m0;
                    const __m512i row = _mm512_load_si512((const void*)tr64[s]);
                    if (m_len == 64) {
                        _mm512_store_si512((void*)dst, row);
                    } else {
                        _mm512_mask_storeu_epi8((void*)dst, write_mask, row);
                    }
                }
            }
        }
    }
}

BED_T512 static void decode_i8_kmajor_band_quant(const uint8_t* raw_base, size_t rec_bytes,
                                                 const int* valid_idx, int K,
                                                 const uint64_t* keep, uint32_t n_raw,
                                                 uint32_t r_begin, uint32_t n_out,
                                                 int8_t* kmajor_out, size_t ldK,
                                                 int8_t q, const int8_t* miss_q,
                                                 const double* mu_q, double* r_out,
                                                 const double* miss_err_q2, double* diag_err_out) {
    const int sel = (keep ? 4 : 0) | ((mu_q && r_out) ? 2 : 0) | ((miss_err_q2 && diag_err_out) ? 1 : 0);
    switch (sel) {
        case 0:  decode_i8_kmajor_band_quant_t<false, false, false>(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, q, miss_q, mu_q, r_out, miss_err_q2, diag_err_out); break;
        case 1:  decode_i8_kmajor_band_quant_t<false, false, true >(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, q, miss_q, mu_q, r_out, miss_err_q2, diag_err_out); break;
        case 2:  decode_i8_kmajor_band_quant_t<false, true,  false>(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, q, miss_q, mu_q, r_out, miss_err_q2, diag_err_out); break;
        case 3:  decode_i8_kmajor_band_quant_t<false, true,  true >(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, q, miss_q, mu_q, r_out, miss_err_q2, diag_err_out); break;
        case 4:  decode_i8_kmajor_band_quant_t<true,  false, false>(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, q, miss_q, mu_q, r_out, miss_err_q2, diag_err_out); break;
        case 5:  decode_i8_kmajor_band_quant_t<true,  false, true >(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, q, miss_q, mu_q, r_out, miss_err_q2, diag_err_out); break;
        case 6:  decode_i8_kmajor_band_quant_t<true,  true,  false>(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, q, miss_q, mu_q, r_out, miss_err_q2, diag_err_out); break;
        default: decode_i8_kmajor_band_quant_t<true,  true,  true >(raw_base, rec_bytes, valid_idx, K, keep, n_raw, r_begin, n_out, kmajor_out, ldK, q, miss_q, mu_q, r_out, miss_err_q2, diag_err_out); break;
    }
}

}  // namespace avx512
