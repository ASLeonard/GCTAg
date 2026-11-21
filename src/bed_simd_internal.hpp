// Private helpers shared by src/bed_simd.cpp and the AVX-512 emulator test
// (tests/test_bed_simd.cpp).  Must be included inside namespace
// bed_simd::detail after bed_simd.hpp.  Freestanding: builtins only.
#ifndef GCTA_BED_SIMD_INTERNAL_HPP
#define GCTA_BED_SIMD_INTERNAL_HPP

static inline uint32_t popcnt64(uint64_t x) { return (uint32_t)__builtin_popcountll(x); }
static inline uint32_t ctz64(uint64_t x) { return (uint32_t)__builtin_ctzll(x); }

static inline uint64_t load_u64(const uint8_t* p) {
    uint64_t x;
    __builtin_memcpy(&x, p, 8);
    return x;   // little-endian targets only (x86_64, aarch64)
}
static inline uint64_t load_u64_partial(const uint8_t* p, uint32_t nbytes) {
    uint64_t x = 0;
    __builtin_memcpy(&x, p, nbytes);
    return x;
}

// Collects the rare missing-genotype side outputs (lazy zero-fill bitmask,
// sparse indices, weighted missing sums).  Only touched when a chunk actually
// contains a missing call, so imputed (zero-missing) data never pays for it.
struct MissSink {
    uint64_t* bits = nullptr;
    uint32_t nwords = 0;
    bool started = false;
    uint32_t* idx = nullptr;
    uint32_t cap = 0;
    uint32_t n_miss = 0;
    const float* wf = nullptr;
    const double* wd = nullptr;
    double wf_miss = 0.0;
    double wd_miss = 0.0;

    void init(const I8Opts* o, uint32_t n_out) {
        bits = o->miss_bits;
        nwords = (n_out + 63u) >> 6;
        idx = o->miss_idx;
        cap = o->miss_idx_cap;
        wf = o->wf;
        wd = o->wd;
    }
    void init_bits(uint64_t* b, uint32_t n_out) {
        bits = b;
        nwords = (n_out + 63u) >> 6;
    }
    bool active() const { return bits || idx || wf || wd; }
    // m: missing bits for output samples [pos, pos + 64).
    void add(uint64_t m, uint32_t pos) {
        if (bits) {
            if (!started) {
                __builtin_memset(bits, 0, sizeof(uint64_t) * nwords);
                started = true;
            }
            const uint32_t wi = pos >> 6, sh = pos & 63u;
            bits[wi] |= m << sh;
            if (sh) {
                const uint64_t hi = m >> (64u - sh);
                if (hi) bits[wi + 1] |= hi;
            }
        }
        if (idx || wf || wd) {
            uint64_t t = m;
            uint32_t k = n_miss;
            while (t) {
                const uint32_t s = pos + ctz64(t);
                if (idx && k < cap) idx[k] = s;
                ++k;
                if (wf) wf_miss += (double)wf[s];
                if (wd) wd_miss += wd[s];
                t &= t - 1;
            }
        }
        n_miss += popcnt64(m);
    }
    void add1(uint32_t s) { add(1ull, s); }
    void finish(I8Opts* o) const {
        o->miss_idx_ct = n_miss;
        o->wf_miss = wf_miss;
        o->wd_miss = wd_miss;
    }
};

#endif  // GCTA_BED_SIMD_INTERNAL_HPP
