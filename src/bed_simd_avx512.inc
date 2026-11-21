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

}  // namespace avx512
