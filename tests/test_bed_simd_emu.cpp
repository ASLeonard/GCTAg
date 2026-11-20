// -----------------------------------------------------------------------------
// test_bed_simd_emu.cpp -- validates the AVX-512 kernels of
// src/bed_simd_avx512.inc bit-for-bit against the scalar reference kernels,
// on any host, by compiling the .inc against tests/avx512_emu.hpp (a portable
// scalar emulation of the intrinsics it uses).
//
// Build (macOS arm64 example):
//   clang++ -std=c++23 -O1 -g -fsanitize=address,undefined -Iinclude
//       tests/test_bed_simd_emu.cpp src/bed_simd.cpp -o test_bed_simd_emu
// -----------------------------------------------------------------------------
#include "bed_simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "avx512_emu.hpp"   // intrinsic emulation (namespace avx512_emu)

#define BED_T512
namespace bed_simd { namespace detail {
#include "../src/bed_simd_internal.hpp"
// using-declarations: hide any global x86 intrinsics of the same names.
AVX512_EMU_IMPORT;
#include "../src/bed_simd_avx512.inc"
}}  // namespace bed_simd::detail
#undef BED_T512

namespace emu = bed_simd::detail::avx512;
using bed_simd::Counts;
using bed_simd::I8Opts;

// ---------------------------------------------------------------------------
static std::mt19937_64 g_rng(0xB5ED5EEDull);
static uint64_t rnd() { return g_rng(); }
static uint32_t rndu(uint32_t n) { return n ? (uint32_t)(rnd() % n) : 0u; }
static double rnd01() { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }

static uint64_t g_cases = 0, g_checks = 0, g_fail = 0;
static std::string g_ctx;

#define CHECK(cond, ...)                                                                   \
    do {                                                                                   \
        ++g_checks;                                                                        \
        if (!(cond)) {                                                                     \
            if (++g_fail <= 40) {                                                          \
                std::fprintf(stderr, "FAIL [%s] %s:%d: ", g_ctx.c_str(), __FILE__, __LINE__); \
                std::fprintf(stderr, __VA_ARGS__);                                         \
                std::fprintf(stderr, "\n");                                                \
            }                                                                              \
        }                                                                                  \
    } while (0)

// Exactly-sized heap allocation: any over-read / over-write hits the ASan redzone.
template <class T> struct Exact {
    T* p = nullptr;
    size_t n = 0;
    explicit Exact(size_t n_) : n(n_) { p = static_cast<T*>(std::malloc((n != 0) ? n * sizeof(T) : 1)); }
    ~Exact() { std::free(p); }
    Exact(const Exact&) = delete;
    Exact& operator=(const Exact&) = delete;
};

// Output buffer with sentinel prefill and guard bytes after it.
static constexpr size_t kGuard = 128;
static constexpr uint8_t kGuardByte = 0xCD;
static constexpr uint8_t kFillByte = 0xA5;
template <class T> struct Guarded {
    std::vector<uint8_t> mem;
    T* p;
    size_t n;
    size_t off;
    explicit Guarded(size_t n_, size_t off_ = 0) : n(n_), off(off_) {
        mem.assign(off + n * sizeof(T) + kGuard, kGuardByte);
        std::memset(mem.data() + off, kFillByte, n * sizeof(T));
        p = reinterpret_cast<T*>(mem.data() + off);
    }
    bool guard_ok() const {
        for (size_t i = 0; i < off; ++i) if (mem[i] != kGuardByte) return false;
        for (size_t i = off + n * sizeof(T); i < mem.size(); ++i) if (mem[i] != kGuardByte) return false;
        return true;
    }
    bool untouched() const {
        for (size_t i = 0; i < n * sizeof(T); ++i) if (mem[off + i] != kFillByte) return false;
        return true;
    }
};

static bool eq(const Counts& a, const Counts& b) {
    return a.n0 == b.n0 && a.n1 == b.n1 && a.n2 == b.n2 && a.nmiss == b.nmiss;
}
#define CHECK_COUNTS(ref, got, what)                                                          \
    CHECK(eq(ref, got), "%s counts ref{%u,%u,%u,m%u} got{%u,%u,%u,m%u}", what, (ref).n0, (ref).n1, \
          (ref).n2, (ref).nmiss, (got).n0, (got).n1, (got).n2, (got).nmiss)

static bool close_rel(double a, double b, double scale) {
    return std::fabs(a - b) <= 1e-9 * std::max(scale, 1e-300) || a == b;
}

// ---------------------------------------------------------------------------
// Raw record generation.
// ---------------------------------------------------------------------------
static void gen_raw(uint8_t* raw, uint32_t n_raw, double miss_rate) {
    const uint32_t nb = bed_simd::record_bytes(n_raw);
    for (uint32_t i = 0; i < nb; ++i) raw[i] = (uint8_t)rnd();   // also random trailing garbage
    if (miss_rate < 0) return;                                    // fully random bytes (~25% missing)
    for (uint32_t s = 0; s < n_raw; ++s) {
        uint32_t c;
        if (rnd01() < miss_rate) c = 1u;
        else { static const uint32_t nm[3] = {0u, 2u, 3u}; c = nm[rndu(3)]; }
        uint8_t& b = raw[s >> 2];
        const uint32_t sh = (s & 3u) * 2u;
        b = (uint8_t)((b & ~(3u << sh)) | (c << sh));
    }
}

// ---------------------------------------------------------------------------
// Keep subsets: returns false for "full" (keep == nullptr).
// ---------------------------------------------------------------------------
struct KeepSpec { std::string name; bool full; std::vector<uint32_t> idx; };

static std::vector<uint32_t> random_subset(uint32_t n_raw, uint32_t k) {
    k = std::min(k, n_raw);
    std::vector<uint32_t> all(n_raw);
    for (uint32_t i = 0; i < n_raw; ++i) all[i] = i;
    for (uint32_t i = 0; i < k; ++i) std::swap(all[i], all[i + rndu(n_raw - i)]);
    all.resize(k);
    std::sort(all.begin(), all.end());
    return all;
}
static std::vector<uint32_t> bernoulli_subset(uint32_t n_raw, double p) {
    std::vector<uint32_t> v;
    for (uint32_t i = 0; i < n_raw; ++i) if (rnd01() < p) v.push_back(i);
    return v;
}

static std::vector<KeepSpec> keep_specs(uint32_t n_raw) {
    std::vector<KeepSpec> ks;
    ks.push_back({"full", true, {}});
    {
        std::vector<uint32_t> all(n_raw);
        for (uint32_t i = 0; i < n_raw; ++i) all[i] = i;
        ks.push_back({"all-mask", false, all});
    }
    ks.push_back({"empty", false, {}});
    for (uint32_t k : {1u, 2u, 3u, 63u, 64u, 65u, 255u, 256u, 257u})
        if (k <= n_raw) ks.push_back({"rand" + std::to_string(k), false, random_subset(n_raw, k)});
    ks.push_back({"p50", false, bernoulli_subset(n_raw, 0.5)});
    ks.push_back({"p99", false, bernoulli_subset(n_raw, 0.99)});
    ks.push_back({"p05", false, bernoulli_subset(n_raw, 0.05)});
    ks.push_back({"first", false, {0u}});
    ks.push_back({"last", false, {n_raw - 1}});
    // Contiguous blocks with non-aligned starts / ends.
    for (int t = 0; t < 3; ++t) {
        const uint32_t s = rndu(n_raw);
        const uint32_t len = 1 + rndu(n_raw - s);
        std::vector<uint32_t> v;
        for (uint32_t i = s; i < s + len; ++i) v.push_back(i);
        ks.push_back({"block" + std::to_string(s) + "+" + std::to_string(len), false, v});
    }
    {   // tail block (ends exactly at n_raw)
        const uint32_t s = rndu(n_raw);
        std::vector<uint32_t> v;
        for (uint32_t i = s; i < n_raw; ++i) v.push_back(i);
        ks.push_back({"tail" + std::to_string(s), false, v});
    }
    {   // sparse word gaps: whole empty 64-sample words in between
        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n_raw; ++i) if (((i >> 6) % 3u) == 1u && rnd01() < 0.7) v.push_back(i);
        ks.push_back({"wordgaps", false, v});
    }
    return ks;
}

// ---------------------------------------------------------------------------
// One case.
// ---------------------------------------------------------------------------
static void run_case(const uint8_t* raw, uint32_t n_raw, const KeepSpec& spec) {
    ++g_cases;
    const uint64_t* km = nullptr;
    Exact<uint64_t> kbuf(bed_simd::keep_mask_words(n_raw));
    if (!spec.full) {
        const bool ok = bed_simd::build_keep_mask(spec.idx.data(), spec.idx.size(), n_raw, kbuf.p);
        CHECK(ok, "build_keep_mask failed");
        km = kbuf.p;
    }
    const uint32_t n_out = spec.full ? n_raw : (uint32_t)spec.idx.size();

    // --- count ------------------------------------------------------------
    const Counts ref_c = bed_simd::count(raw, km, n_raw, n_out);
    {
        const Counts got = emu::count(raw, km, n_raw, n_out);
        CHECK_COUNTS(ref_c, got, "count");
    }
    const uint32_t nmiss = ref_c.nmiss;
    const uint32_t nwords = (n_out + 63u) >> 6;

    // --- decode_i8 (no opts) ------------------------------------------------
    {
        const size_t off = rndu(64);
        Guarded<int8_t> r(n_out, off), g(n_out, off);
        const Counts rc = bed_simd::decode_i8(raw, km, n_raw, n_out, r.p, nullptr);
        const Counts gc = emu::decode_i8(raw, km, n_raw, 0, n_out, g.p, nullptr);
        CHECK_COUNTS(rc, gc, "decode_i8");
        CHECK_COUNTS(ref_c, gc, "decode_i8 vs count");
        CHECK(std::memcmp(r.p, g.p, n_out) == 0, "decode_i8 output mismatch");
        CHECK(g.guard_ok(), "decode_i8 output overrun");
    }

    // --- decode_i8 with opts ----------------------------------------------
    // weights (exactly sized)
    Exact<float> wf(n_out);
    Exact<double> wd(n_out);
    double scale_f = 0, scale_d = 0;
    for (uint32_t s = 0; s < n_out; ++s) {
        wf.p[s] = (float)(rnd01() * 4.0 - 2.0);
        wd.p[s] = rnd01() * 4.0 - 2.0;
        scale_f += 4.0 * std::fabs((double)wf.p[s]);
        scale_d += 4.0 * std::fabs(wd.p[s]);
    }
    for (int cfg = 0; cfg < 16; ++cfg) {
        const bool use_bits = cfg & 1, use_idx = cfg & 2, use_wf = cfg & 4, use_wd = cfg & 8;
        std::vector<uint32_t> caps = {0};
        if (use_idx) caps = {0u, 1u, nmiss / 2u, nmiss, nmiss + 3u};
        for (uint32_t cap : caps) {
            Guarded<int8_t> ro(n_out), go(n_out);
            Guarded<uint64_t> rb(nwords), gb(nwords);
            Guarded<uint32_t> ri(cap), gi(cap);
            I8Opts R, G;
            if (use_bits) { R.miss_bits = rb.p; G.miss_bits = gb.p; }
            if (use_idx) { R.miss_idx = ri.p; G.miss_idx = gi.p; R.miss_idx_cap = G.miss_idx_cap = cap; }
            if (use_wf) { R.wf = wf.p; G.wf = wf.p; }
            if (use_wd) { R.wd = wd.p; G.wd = wd.p; }
            const Counts rc = bed_simd::decode_i8(raw, km, n_raw, n_out, ro.p, &R);
            const Counts gc = emu::decode_i8(raw, km, n_raw, 0, n_out, go.p, &G);
            char what[64];
            std::snprintf(what, sizeof what, "decode_i8 opts cfg=%d cap=%u", cfg, cap);
            CHECK_COUNTS(rc, gc, what);
            CHECK(std::memcmp(ro.p, go.p, n_out) == 0, "%s: output mismatch", what);
            CHECK(go.guard_ok(), "%s: output overrun", what);
            CHECK(gb.guard_ok(), "%s: miss_bits overrun", what);
            CHECK(gi.guard_ok(), "%s: miss_idx overrun", what);
            if (use_bits) {
                if (nmiss == 0) CHECK(gb.untouched(), "%s: miss_bits touched with nmiss==0", what);
                else CHECK(std::memcmp(rb.p, gb.p, nwords * 8) == 0, "%s: miss_bits mismatch", what);
            }
            CHECK(R.miss_idx_ct == G.miss_idx_ct, "%s: miss_idx_ct ref %u got %u", what, R.miss_idx_ct,
                  G.miss_idx_ct);
            if (use_idx) {
                const uint32_t k = std::min(cap, nmiss);
                CHECK(std::memcmp(ri.p, gi.p, k * 4u) == 0, "%s: miss_idx mismatch", what);
                // entries [k, cap) must be untouched
                bool tail_ok = true;
                for (size_t i = k * 4u; i < cap * 4u; ++i) tail_ok &= gi.mem[i] == kFillByte;
                CHECK(tail_ok, "%s: miss_idx written beyond nmiss", what);
            }
            CHECK(close_rel(R.wf_dot, G.wf_dot, scale_f), "%s: wf_dot ref %.17g got %.17g", what, R.wf_dot, G.wf_dot);
            CHECK(close_rel(R.wf_miss, G.wf_miss, scale_f), "%s: wf_miss ref %.17g got %.17g", what, R.wf_miss, G.wf_miss);
            CHECK(close_rel(R.wd_dot, G.wd_dot, scale_d), "%s: wd_dot ref %.17g got %.17g", what, R.wd_dot, G.wd_dot);
            CHECK(close_rel(R.wd_dot2, G.wd_dot2, scale_d), "%s: wd_dot2 ref %.17g got %.17g", what, R.wd_dot2, G.wd_dot2);
            CHECK(close_rel(R.wd_miss, G.wd_miss, scale_d), "%s: wd_miss ref %.17g got %.17g", what, R.wd_miss, G.wd_miss);
        }
    }

    // --- decode_f32 / decode_f64 -------------------------------------------
    for (int with_bits = 0; with_bits < 2; ++with_bits) {
        float tf[4];
        double td[4];
        for (int i = 0; i < 4; ++i) {   // distinct, NaN-free (incl. the missing value)
            tf[i] = (float)(rnd01() * 20.0 - 10.0) + (float)i * 32.0f;
            td[i] = rnd01() * 20.0 - 10.0 + i * 32.0;
        }
        if (with_bits) { tf[3] = -9.0f; td[3] = -9.0; }
        {
            Guarded<float> r(n_out), g(n_out);
            Guarded<uint64_t> rb(nwords), gb(nwords);
            const Counts rc = bed_simd::decode_f32(raw, km, n_raw, n_out, tf, r.p, with_bits ? rb.p : nullptr);
            const Counts gc = emu::decode_f32(raw, km, n_raw, n_out, tf, g.p, with_bits ? gb.p : nullptr);
            CHECK_COUNTS(rc, gc, "decode_f32");
            CHECK(std::memcmp(r.p, g.p, n_out * sizeof(float)) == 0, "decode_f32 output mismatch (bits=%d)", with_bits);
            CHECK(g.guard_ok(), "decode_f32 output overrun");
            CHECK(gb.guard_ok(), "decode_f32 miss_bits overrun");
            if (with_bits) {
                if (nmiss == 0) CHECK(gb.untouched(), "decode_f32 miss_bits touched with nmiss==0");
                else CHECK(std::memcmp(rb.p, gb.p, nwords * 8) == 0, "decode_f32 miss_bits mismatch");
            } else {
                CHECK(gb.untouched(), "decode_f32 miss_bits touched while null");
            }
        }
        {
            Guarded<double> r(n_out), g(n_out);
            Guarded<uint64_t> rb(nwords), gb(nwords);
            const Counts rc = bed_simd::decode_f64(raw, km, n_raw, n_out, td, r.p, with_bits ? rb.p : nullptr);
            const Counts gc = emu::decode_f64(raw, km, n_raw, n_out, td, g.p, with_bits ? gb.p : nullptr);
            CHECK_COUNTS(rc, gc, "decode_f64");
            CHECK(std::memcmp(r.p, g.p, n_out * sizeof(double)) == 0, "decode_f64 output mismatch (bits=%d)", with_bits);
            CHECK(g.guard_ok(), "decode_f64 output overrun");
            if (with_bits) {
                if (nmiss == 0) CHECK(gb.untouched(), "decode_f64 miss_bits touched with nmiss==0");
                else CHECK(std::memcmp(rb.p, gb.p, nwords * 8) == 0, "decode_f64 miss_bits mismatch");
            }
        }
    }

    // --- range mode (decode_i8_range) --------------------------------------
    if (n_out > 0) {
        std::vector<std::pair<uint32_t, uint32_t>> ranges;   // (k, n)
        auto add_k = [&](uint32_t k) {
            if (spec.full) k &= ~63u;
            if (k >= n_out) return;
            const uint32_t rem = n_out - k;
            ranges.push_back({k, rem});
            ranges.push_back({k, 1u});
            ranges.push_back({k, 1u + rndu(rem)});
            if (rem > 64) ranges.push_back({k, 64u});
            if (rem > 65) ranges.push_back({k, 65u});
        };
        add_k(0);
        add_k(rndu(n_out));
        add_k(rndu(n_out));
        add_k(n_out - 1);
        if (n_out > 64) add_k(64);
        if (n_out > 130) add_k(129);
        for (auto [k, n] : ranges) {
            const uint32_t r_begin = spec.full ? k : spec.idx[k];
            Guarded<int8_t> r(n), g(n);
            bed_simd::decode_i8_range(raw, km, n_raw, r_begin, n, r.p);
            emu::decode_i8(raw, km, n_raw, r_begin, n, g.p, nullptr);
            CHECK(std::memcmp(r.p, g.p, n) == 0, "range k=%u r_begin=%u n=%u: output mismatch", k, r_begin, n);
            CHECK(g.guard_ok(), "range k=%u r_begin=%u n=%u: overrun", k, r_begin, n);
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_rng.seed(std::strtoull(argv[1], nullptr, 0));   // optional RNG seed
    if (!bed_simd::set_isa(bed_simd::Isa::Scalar)) {
        std::fprintf(stderr, "cannot select scalar reference kernels\n");
        return 2;
    }
    const uint32_t sizes[] = {1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                              255, 256, 257, 1000, 1023, 4097, 10001};
    const double miss_rates[] = {0.0, 0.001, 0.05, 0.5, -1.0};
    for (uint32_t n_raw : sizes) {
        for (double mr : miss_rates) {
            const uint32_t nb = bed_simd::record_bytes(n_raw);
            Exact<uint8_t> raw(nb);   // record ends exactly at the allocation end
            gen_raw(raw.p, n_raw, mr);
            for (const KeepSpec& ks : keep_specs(n_raw)) {
                char ctx[160];
                std::snprintf(ctx, sizeof ctx, "n_raw=%u miss=%g keep=%s n_keep=%zu", n_raw, mr, ks.name.c_str(),
                              ks.full ? (size_t)n_raw : ks.idx.size());
                g_ctx = ctx;
                run_case(raw.p, n_raw, ks);
            }
        }
    }
    // Large-N accumulator overflow regression (N > 65535 and > 262144):
    // exercises constant 0xFF / 0xAA / 0x55 / 0x00 and random records.
    for (uint32_t n_raw : {70000u, 270000u}) {
        const uint32_t nb = bed_simd::record_bytes(n_raw);
        Exact<uint8_t> raw(nb);
        std::vector<uint32_t> sub;
        for (uint32_t i = 0; i < n_raw; ++i)
            if (i % 7 != 3) sub.push_back(i);
        const KeepSpec kfull{"full", true, {}};
        const KeepSpec ksub{"sub", false, sub};
        for (int pat : {0xFF, 0xAA, 0x55, 0x00, -1}) {
            if (pat >= 0) std::memset(raw.p, pat, nb);
            else gen_raw(raw.p, n_raw, 0.05);
            for (const KeepSpec& ks : {kfull, ksub}) {
                char ctx[160];
                std::snprintf(ctx, sizeof ctx, "large n_raw=%u pat=%d keep=%s", n_raw, pat, ks.name.c_str());
                g_ctx = ctx;
                ++g_cases;
                const uint32_t n_out = ks.full ? n_raw : (uint32_t)ks.idx.size();
                std::vector<uint64_t> mask;
                const uint64_t* km = nullptr;
                if (!ks.full) {
                    mask.resize(bed_simd::keep_mask_words(n_raw));
                    bed_simd::build_keep_mask(ks.idx.data(), ks.idx.size(), n_raw, mask.data());
                    km = mask.data();
                }
                const Counts rc = bed_simd::count(raw.p, km, n_raw, n_out);
                const Counts gc = emu::count(raw.p, km, n_raw, n_out);
                CHECK_COUNTS(rc, gc, "large count");
                if (n_raw == 70000u) {
                    Guarded<int8_t> r(n_out), g(n_out);
                    const Counts di_r = bed_simd::decode_i8(raw.p, km, n_raw, n_out, r.p, nullptr);
                    const Counts di_g = emu::decode_i8(raw.p, km, n_raw, 0, n_out, g.p, nullptr);
                    CHECK_COUNTS(di_r, di_g, "large decode_i8");
                    CHECK(std::memcmp(r.p, g.p, n_out) == 0, "large decode_i8 output mismatch");
                    CHECK(g.guard_ok(), "large decode_i8 overrun");
                }
            }
        }
    }
    std::printf("%s: %llu cases, %llu checks, %llu failures\n", g_fail ? "FAIL" : "PASS",
                (unsigned long long)g_cases, (unsigned long long)g_checks, (unsigned long long)g_fail);
    return g_fail ? 1 : 0;
}
