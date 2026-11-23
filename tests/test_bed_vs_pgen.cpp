// -----------------------------------------------------------------------------
// test_bed_vs_pgen.cpp -- bit-for-bit cross-check of bed_simd (every ISA the
// host supports) against the ORIGINAL PLINK2/Pgenlib pipeline used by GCTA:
//   PgenReader::Load(.bed) + ReadRawFullHard (PgrPlink1ToPlink2 conversion)
//   -> CountHardFreqMissExt / CountHardFreqMissExtX
//   -> ExtractInt8Ext / ExtractFloatExt / ExtractDoubleExt (+ missOut)
// versus bed_simd::count / decode_i8 / decode_f32 / decode_f64 on the raw
// record bytes read directly from the same file.
//
// Usage: test_bed_vs_pgen [tmpdir] [seed]
// Build (macOS arm64 example):
//   P=third_party/plink-ng/2.0/include
//   clang++ -std=c++23 -O1 -g -fsanitize=address,undefined -Iinclude -I. -I$P \
//     -Ithird_party/plink-ng/2.0/simde tests/test_bed_vs_pgen.cpp src/bed_simd.cpp \
//     third_party/Pgenlib/PgenReader.cpp $P/pgenlib_misc.cc $P/pgenlib_read.cc \
//     $P/plink2_base.cc $P/plink2_bits.cc -o test_bed_vs_pgen
// -----------------------------------------------------------------------------
#include "bed_simd.hpp"
#include "third_party/Pgenlib/PgenReader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using bed_simd::Counts;

static std::mt19937_64 g_rng(0x9E3779B97F4A7C15ull);
static uint64_t rnd() { return g_rng(); }
static uint32_t rndu(uint32_t n) { return n ? (uint32_t)(rnd() % n) : 0u; }
static double rnd01() { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }

static uint64_t g_checks = 0, g_fail = 0, g_cases = 0;
static std::string g_ctx;
#define CHECK(cond, ...)                                                                        \
    do {                                                                                        \
        ++g_checks;                                                                             \
        if (!(cond)) {                                                                          \
            if (++g_fail <= 40) {                                                               \
                std::fprintf(stderr, "FAIL [%s] line %d: ", g_ctx.c_str(), __LINE__);           \
                std::fprintf(stderr, __VA_ARGS__);                                              \
                std::fprintf(stderr, "\n");                                                     \
            }                                                                                   \
        }                                                                                       \
    } while (0)

// 64-byte aligned, zeroed, with `pre` zero words before the returned pointer.
struct AWords {
    uintptr_t* base = nullptr;
    uintptr_t* p = nullptr;
    size_t n = 0;
    AWords(size_t n_, size_t pre = 0) : n(n_) {
        const size_t bytes = ((pre + n_ + 8) * sizeof(uintptr_t) + 63) / 64 * 64;
        base = static_cast<uintptr_t*>(std::aligned_alloc(64, bytes));
        std::memset(base, 0, bytes);
        p = base + pre;
    }
    ~AWords() { std::free(base); }
    AWords(const AWords&) = delete;
    AWords& operator=(const AWords&) = delete;
};

// Exactly sized output with guard bytes after it (bed_simd outputs).
template <class T> struct Guarded {
    std::vector<uint8_t> mem;
    T* p;
    size_t n;
    explicit Guarded(size_t n_) : n(n_) {
        mem.assign(n * sizeof(T) + 64 + 64, 0xCD);   // 64 B lead pad for alignment, 64 B guard
        p = reinterpret_cast<T*>(mem.data() + ((64 - (reinterpret_cast<uintptr_t>(mem.data()) & 63)) & 63));
        std::memset(p, 0xA5, n * sizeof(T));
    }
    bool guard_ok() const {
        const uint8_t* e = reinterpret_cast<const uint8_t*>(p + n);
        for (const uint8_t* q = e; q < mem.data() + mem.size(); ++q) if (*q != 0xCD) return false;
        return true;
    }
    bool untouched() const {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n * sizeof(T); ++i) if (b[i] != 0xA5) return false;
        return true;
    }
};

static bool same_double(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof a) == 0;
}
static bool same_info(const SNPInfo& a, const SNPInfo& b) {
    return a.N == b.N && a.AlCount == b.AlCount && same_double(a.af, b.af) && same_double(a.mean, b.mean) &&
           same_double(a.std, b.std) && same_double(a.nMissRate, b.nMissRate);
}

// Replicas of src/Geno.cpp's bed_fill_snpinfo / bed_fill_snpinfo_x (anonymous namespace there).
static void bed_fill_snpinfo(const Counts& c, uint32_t keepSize, bool f_std, SNPInfo* snpinfo) {
    const std::array<uint32_t, 4> genocounts = {c.n0, c.n1, c.n2, c.nmiss};
    snpinfo->N = keepSize - genocounts[3];
    snpinfo->AlCount = 2 * snpinfo->N;
    uint32_t dGenoCounts2 = 2 * genocounts[2];
    uint32_t dosage = genocounts[1] + dGenoCounts2;
    snpinfo->af = 1.0 * dosage / snpinfo->AlCount;
    snpinfo->mean = 2.0 * snpinfo->af;
    if (f_std) snpinfo->std = (dosage + dGenoCounts2 - snpinfo->mean * dosage) / (snpinfo->N - 1);
    snpinfo->nMissRate = 1.0 * snpinfo->N / keepSize;
}
static bool bed_fill_snpinfo_x(const Counts& c, const Counts& c2, uint32_t keepSize, bool dosageComp, bool f_std,
                               SNPInfo* snpinfo) {
    const std::array<uint32_t, 4> genocounts = {c.n0, c.n1, c.n2, c.nmiss};
    const std::array<uint32_t, 4> genocounts2 = {c2.n0, c2.n1, c2.n2, c2.nmiss};
    bool retVal = genocounts2[1] == 0;
    snpinfo->N = keepSize - genocounts[3];
    snpinfo->AlCount = 2 * snpinfo->N - genocounts2[0] - genocounts2[1] - genocounts2[2];
    snpinfo->nMissRate = 1.0 * snpinfo->N / keepSize;
    int dGenoCount1Homo = 2 * genocounts[2];
    double dosage = genocounts[1] + dGenoCount1Homo;
    double dosage_half = dosage - genocounts2[2] - 0.5 * genocounts2[1];
    snpinfo->af = dosage_half / snpinfo->AlCount;
    if (dosageComp) {
        snpinfo->mean = dosage / snpinfo->N;
        if (f_std) snpinfo->std = (dosage + dGenoCount1Homo - dosage * snpinfo->mean) / (snpinfo->N - 1);
    } else {
        snpinfo->mean = dosage_half / snpinfo->N;
        if (f_std)
            snpinfo->std = (dosage + dGenoCount1Homo - 0.75 * genocounts2[1] - 3 * genocounts2[2] -
                            dosage_half * snpinfo->mean) / (snpinfo->N - 1);
    }
    return retVal;
}

static bool eqc(const Counts& c, const std::array<uint32_t, 4>& g) {
    return c.n0 == g[0] && c.n1 == g[1] && c.n2 == g[2] && c.nmiss == g[3];
}

// ---------------------------------------------------------------------------
struct Subset {
    std::string name;
    std::vector<uint32_t> idx;   // strictly increasing raw indices
    bool full;                   // keep == all samples (bed_simd gets keep == nullptr)
};

static std::vector<uint32_t> rand_subset_of(const std::vector<uint32_t>& from, uint32_t k) {
    std::vector<uint32_t> v = from;
    k = std::min<uint32_t>(k, (uint32_t)v.size());
    for (uint32_t i = 0; i < k; ++i) std::swap(v[i], v[i + rndu((uint32_t)v.size() - i)]);
    v.resize(k);
    std::sort(v.begin(), v.end());
    return v;
}

static std::vector<Subset> keep_sets(uint32_t n) {
    std::vector<uint32_t> all(n);
    for (uint32_t i = 0; i < n; ++i) all[i] = i;
    std::vector<Subset> ks;
    ks.push_back({"full", all, true});
    ks.push_back({"all-as-mask", all, false});
    for (uint32_t k : {1u, 2u, 3u, 5u, 17u, 63u, 65u, 100u, 255u, 257u, 1001u})
        if (k < n) ks.push_back({"rand" + std::to_string(k), rand_subset_of(all, k), false});
    for (double p : {0.5, 0.99, 0.05}) {
        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) if (rnd01() < p) v.push_back(i);
        if (!v.empty() && v.size() < n) ks.push_back({"p" + std::to_string((int)(p * 100)), v, false});
    }
    for (int t = 0; t < 2 && n > 1; ++t) {   // contiguous blocks with unaligned starts
        const uint32_t s = 1 + rndu(n - 1);
        const uint32_t len = 1 + rndu(n - s);
        std::vector<uint32_t> v;
        for (uint32_t i = s; i < s + len; ++i) v.push_back(i);
        ks.push_back({"block" + std::to_string(s) + "+" + std::to_string(len), v, false});
    }
    if (n > 1) ks.push_back({"last", {n - 1}, false});
    return ks;
}

// plink2 masks for a subset
struct PgMasks {
    AWords incl, inter;
    PgMasks(uint32_t n_raw, const std::vector<uint32_t>& idx)
        : incl(PgenReader::GetSubsetMaskSize(n_raw)), inter(PgenReader::GetSubsetMaskSize(n_raw), 8) {
        PgenReader::SetSampleSubsets(idx, n_raw, incl.p, inter.p);
    }
};
// bed_simd keep mask (exactly keep_mask_words, 64-B aligned)
struct BedMask {
    uint64_t* p;
    explicit BedMask(uint32_t n_raw, const std::vector<uint32_t>& idx) {
        const size_t w = bed_simd::keep_mask_words(n_raw);
        p = static_cast<uint64_t*>(std::aligned_alloc(64, w * 8));
        if (!bed_simd::build_keep_mask(idx.data(), idx.size(), n_raw, p)) {
            std::fprintf(stderr, "build_keep_mask failed\n");
            std::abort();
        }
    }
    ~BedMask() { std::free(p); }
};

static std::vector<bed_simd::Isa> g_isas;
static const char* isa_str(bed_simd::Isa i) {
    switch (i) {
        case bed_simd::Isa::Scalar: return "scalar";
        case bed_simd::Isa::Neon: return "neon";
        case bed_simd::Isa::Avx2: return "avx2";
        case bed_simd::Isa::Avx512: return "avx512";
    }
    return "?";
}

static void gen_record(uint8_t* rec, uint32_t n_raw, double miss_rate) {
    const uint32_t nb = bed_simd::record_bytes(n_raw);
    for (uint32_t i = 0; i < nb; ++i) rec[i] = (uint8_t)rnd();   // trailing bits = garbage
    if (miss_rate < 0) return;
    for (uint32_t s = 0; s < n_raw; ++s) {
        static const uint32_t nm[3] = {0u, 2u, 3u};
        const uint32_t c = rnd01() < miss_rate ? 1u : nm[rndu(3)];
        const uint32_t sh = (s & 3u) * 2u;
        rec[s >> 2] = (uint8_t)((rec[s >> 2] & ~(3u << sh)) | (c << sh));
    }
}

static void run_file(const std::string& dir, uint32_t n_raw) {
    const uint32_t nvar = 40;
    const uint32_t nb = bed_simd::record_bytes(n_raw);
    std::vector<uint8_t> file(3 + (size_t)nvar * nb);
    file[0] = 0x6c; file[1] = 0x1b; file[2] = 0x01;
    const double rates[5] = {0.0, 0.001, 0.05, 0.5, -1.0};
    for (uint32_t v = 0; v < nvar; ++v) gen_record(&file[3 + (size_t)v * nb], n_raw, rates[v % 5]);
    const std::string path = dir + "/bed_vs_pgen_" + std::to_string(n_raw) + ".bed";
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f || std::fwrite(file.data(), 1, file.size(), f) != file.size()) {
            std::fprintf(stderr, "cannot write %s\n", path.c_str());
            std::exit(2);
        }
        std::fclose(f);
    }
    PgenReader pr;
    uint32_t rawN = n_raw, nv = nvar;
    pr.Load(path, &rawN, &nv, {});
    if (pr.GetRawSampleCt() != n_raw || pr.GetVariantCt() != nvar) {
        std::fprintf(stderr, "PgenReader::Load size mismatch\n");
        std::exit(2);
    }

    const std::vector<Subset> keeps = keep_sets(n_raw);
    AWords genovec(PgenReader::GetGenoBufPtrSize(n_raw) + 8);

    for (uint32_t v = 0; v < nvar; ++v) {
        // raw record bytes, exactly sized (bed_simd input)
        uint8_t* raw = static_cast<uint8_t*>(std::malloc(nb));
        std::memcpy(raw, &file[3 + (size_t)v * nb], nb);

        for (const Subset& ks : keeps) {
            const uint32_t keepN = (uint32_t)ks.idx.size();
            PgMasks pm(n_raw, ks.full ? std::vector<uint32_t>{} : ks.idx);
            BedMask bm(n_raw, ks.idx);
            const uint64_t* km = ks.full ? nullptr : bm.p;
            const uint32_t nwords = (keepN + 63u) / 64u;

            // heterogametic subsets of the keep set
            std::vector<Subset> hets;
            hets.push_back({"het-empty", {}, false});
            hets.push_back({"het=keep", ks.idx, false});
            hets.push_back({"het-rand", rand_subset_of(ks.idx, std::max<uint32_t>(1u, keepN / 3)), false});
            hets.push_back({"het-one", rand_subset_of(ks.idx, 1), false});

            // ---- PLINK2 reference ------------------------------------------
            std::memset(genovec.p, 0, genovec.n * sizeof(uintptr_t));
            pr.ReadRawFullHard(genovec.p, (int)v);

            SNPInfo ref_info{};
            PgenReader::CountHardFreqMissExt(genovec.p, pm.inter.p, n_raw, keepN, &ref_info, true);
            std::array<uint32_t, 4> gc{};
            if (keepN == n_raw) plink2::GenoarrCountFreqsUnsafe(genovec.p, keepN, gc);
            else plink2::GenoarrCountSubsetFreqs(genovec.p, pm.inter.p, n_raw, keepN, gc);

            const float fa[4] = {(float)(rnd01() * 4 - 2), (float)(rnd01() * 4 - 2) + 8.f,
                                 (float)(rnd01() * 4 - 2) + 16.f, -9.f};
            const double da[4] = {rnd01() * 4 - 2, rnd01() * 4 - 2 + 8.0, rnd01() * 4 - 2 + 16.0, -9.0};
            alignas(64) static float ftab[1024];
            {
                const float t[1024] = QUAD_TABLE256(fa[0], fa[1], fa[2], fa[3]);
                std::memcpy(ftab, t, sizeof t);
            }
            alignas(64) const double dtab[32] = GET_TABLE16(da[0], da[1], da[2], da[3]);

            const size_t slack = 256;
            std::vector<int8_t> r_i8(keepN + slack);
            std::vector<float> r_f32(keepN + slack);
            std::vector<double> r_f64(keepN + slack);
            AWords r_miss_i8(PgenReader::GetSubsetMaskSize(keepN) + 8), r_miss_f32(PgenReader::GetSubsetMaskSize(keepN) + 8),
                r_miss_f64(PgenReader::GetSubsetMaskSize(keepN) + 8);
            PgenReader::ExtractInt8Ext(genovec.p, pm.incl.p, n_raw, keepN, r_i8.data(), r_miss_i8.p);
            PgenReader::ExtractFloatExt(genovec.p, pm.incl.p, n_raw, keepN, ftab, r_f32.data(), r_miss_f32.p);
            PgenReader::ExtractDoubleExt(genovec.p, pm.incl.p, n_raw, keepN, dtab, r_f64.data(), r_miss_f64.p);
            auto miss_word = [&](const AWords& m, uint32_t w) {   // only bits < keepN are meaningful
                uint64_t x = (uint64_t)m.p[w];
                if (w == nwords - 1 && (keepN & 63u)) x &= (1ull << (keepN & 63u)) - 1;
                return x;
            };

            // X-chromosome reference per het set
            struct XRef { std::array<uint32_t, 4> gc2{}; SNPInfo info[2]{}; bool ret[2]{}; std::string err[2]; };
            std::vector<XRef> xref(hets.size());
            std::vector<std::unique_ptr<PgMasks>> hpm;
            for (size_t h = 0; h < hets.size(); ++h) {
                hpm.emplace_back(new PgMasks(n_raw, hets[h].idx));
                const uint32_t hN = (uint32_t)hets[h].idx.size();
                if (hN) plink2::GenoarrCountSubsetFreqs(genovec.p, hpm[h]->inter.p, n_raw, hN, xref[h].gc2);
                for (int dc = 0; dc < 2; ++dc)
                    xref[h].ret[dc] = PgenReader::CountHardFreqMissExtX(genovec.p, pm.inter.p, hpm[h]->inter.p, n_raw,
                                                                        keepN, hN, &xref[h].info[dc], xref[h].err[dc],
                                                                        dc == 1, true);
            }

            // ---- bed_simd, every ISA ---------------------------------------
            for (bed_simd::Isa isa : g_isas) {
                bed_simd::set_isa(isa);
                ++g_cases;
                char ctx[200];
                std::snprintf(ctx, sizeof ctx, "isa=%s n_raw=%u var=%u (miss=%g) keep=%s n_keep=%u", isa_str(isa),
                              n_raw, v, rates[v % 5], ks.name.c_str(), keepN);
                g_ctx = ctx;

                const Counts c = bed_simd::count(raw, km, n_raw, keepN);
                CHECK(eqc(c, gc), "count {%u,%u,%u,m%u} vs plink2 {%u,%u,%u,m%u}", c.n0, c.n1, c.n2, c.nmiss, gc[0],
                      gc[1], gc[2], gc[3]);
                SNPInfo info{};
                bed_fill_snpinfo(c, keepN, true, &info);
                CHECK(same_info(info, ref_info), "SNPInfo mismatch vs CountHardFreqMissExt (af %.17g vs %.17g)", info.af,
                      ref_info.af);

                // X chromosome
                for (size_t h = 0; h < hets.size(); ++h) {
                    const uint32_t hN = (uint32_t)hets[h].idx.size();
                    BedMask hm(n_raw, hets[h].idx);
                    if (hN) {
                        const Counts c2 = bed_simd::count(raw, hm.p, n_raw, hN);
                        CHECK(eqc(c2, xref[h].gc2), "%s count mismatch vs plink2", hets[h].name.c_str());
                    }
                    // Geno::bedSnpInfoFromCounts selection logic
                    Counts c2;
                    if (n_raw == keepN) c2 = (keepN == hN) ? c : bed_simd::count(raw, hm.p, n_raw, hN);
                    else if (hN != 0) c2 = bed_simd::count(raw, hm.p, n_raw, hN);
                    for (int dc = 0; dc < 2; ++dc) {
                        SNPInfo xi{};
                        const bool ret = bed_fill_snpinfo_x(c, c2, keepN, dc == 1, true, &xi);
                        CHECK(ret == xref[h].ret[dc], "%s dc=%d retVal %d vs ExtX %d", hets[h].name.c_str(), dc, ret,
                              xref[h].ret[dc]);
                        CHECK(same_info(xi, xref[h].info[dc]), "%s dc=%d SNPInfo mismatch vs CountHardFreqMissExtX",
                              hets[h].name.c_str(), dc);
                    }
                }

                // int8 (+ miss_bits through I8Opts)
                {
                    Guarded<int8_t> o(keepN);
                    Guarded<uint64_t> mb(nwords);
                    bed_simd::I8Opts opts;
                    opts.miss_bits = mb.p;
                    const Counts ci = bed_simd::decode_i8(raw, km, n_raw, keepN, o.p, &opts);
                    CHECK(eqc(ci, gc), "decode_i8 counts mismatch");
                    CHECK(std::memcmp(o.p, r_i8.data(), keepN) == 0, "decode_i8 vs ExtractInt8Ext mismatch");
                    CHECK(o.guard_ok() && mb.guard_ok(), "decode_i8 overrun");
                    if (c.nmiss == 0) CHECK(mb.untouched(), "decode_i8 miss_bits touched with nmiss==0");
                    for (uint32_t w = 0; w < nwords; ++w) {
                        const uint64_t mine = c.nmiss ? mb.p[w] : 0;
                        CHECK(mine == miss_word(r_miss_i8, w), "decode_i8 miss_bits word %u: %016llx vs %016llx", w,
                              (unsigned long long)mine, (unsigned long long)miss_word(r_miss_i8, w));
                    }
                    CHECK(opts.miss_idx_ct == 0 || opts.miss_idx_ct == c.nmiss, "miss_idx_ct");
                }
                // float
                {
                    Guarded<float> o(keepN);
                    Guarded<uint64_t> mb(nwords);
                    const Counts cf = bed_simd::decode_f32(raw, km, n_raw, keepN, fa, o.p, mb.p);
                    CHECK(eqc(cf, gc), "decode_f32 counts mismatch");
                    CHECK(std::memcmp(o.p, r_f32.data(), keepN * sizeof(float)) == 0, "decode_f32 vs ExtractFloatExt mismatch");
                    CHECK(o.guard_ok() && mb.guard_ok(), "decode_f32 overrun");
                    if (c.nmiss == 0) CHECK(mb.untouched(), "decode_f32 miss_bits touched with nmiss==0");
                    for (uint32_t w = 0; w < nwords; ++w) {
                        const uint64_t mine = c.nmiss ? mb.p[w] : 0;
                        CHECK(mine == miss_word(r_miss_f32, w), "decode_f32 miss_bits word %u", w);
                    }
                }
                // double
                {
                    Guarded<double> o(keepN);
                    Guarded<uint64_t> mb(nwords);
                    const Counts cd = bed_simd::decode_f64(raw, km, n_raw, keepN, da, o.p, mb.p);
                    CHECK(eqc(cd, gc), "decode_f64 counts mismatch");
                    CHECK(std::memcmp(o.p, r_f64.data(), keepN * sizeof(double)) == 0, "decode_f64 vs ExtractDoubleExt mismatch");
                    CHECK(o.guard_ok() && mb.guard_ok(), "decode_f64 overrun");
                    if (c.nmiss == 0) CHECK(mb.untouched(), "decode_f64 miss_bits touched with nmiss==0");
                    for (uint32_t w = 0; w < nwords; ++w) {
                        const uint64_t mine = c.nmiss ? mb.p[w] : 0;
                        CHECK(mine == miss_word(r_miss_f64, w), "decode_f64 miss_bits word %u", w);
                    }
                }
            }
        }
        std::free(raw);
    }
    pr.Close();
    std::remove(path.c_str());
}

// Large-N accumulator overflow test (N > 65535 and > 262144, e.g. 16-bit lane
// reduction wrap in NEON/SIMD accumulators) against PgenReader.
static void run_large_n(const std::string& dir, uint32_t n_raw) {
    const uint32_t nvar = 5;   // all-0xFF, all-0xAA, all-0x55, all-0x00, random
    const uint32_t nb = bed_simd::record_bytes(n_raw);
    std::vector<uint8_t> file(3 + (size_t)nvar * nb);
    file[0] = 0x6c; file[1] = 0x1b; file[2] = 0x01;
    const int pats[5] = {0xFF, 0xAA, 0x55, 0x00, -1};
    for (uint32_t v = 0; v < nvar; ++v) {
        uint8_t* rec = &file[3 + (size_t)v * nb];
        if (pats[v] >= 0) std::memset(rec, pats[v], nb);
        else gen_record(rec, n_raw, 0.05);
    }
    const std::string path = dir + "/bed_vs_pgen_large_" + std::to_string(n_raw) + ".bed";
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f || std::fwrite(file.data(), 1, file.size(), f) != file.size()) {
            std::fprintf(stderr, "cannot write %s\n", path.c_str());
            std::exit(2);
        }
        std::fclose(f);
    }
    PgenReader pr;
    uint32_t rawN = n_raw, nv = nvar;
    pr.Load(path, &rawN, &nv, {});
    std::vector<uint32_t> full_idx(n_raw), sub_idx;
    for (uint32_t i = 0; i < n_raw; ++i) {
        full_idx[i] = i;
        if (i % 7 != 3) sub_idx.push_back(i);
    }
    const Subset keeps[2] = {{"full", full_idx, true}, {"sub_6of7", sub_idx, false}};
    AWords genovec(PgenReader::GetGenoBufPtrSize(n_raw) + 8);
    for (uint32_t v = 0; v < nvar; ++v) {
        uint8_t* raw = static_cast<uint8_t*>(std::malloc(nb));
        std::memcpy(raw, &file[3 + (size_t)v * nb], nb);
        for (const Subset& ks : keeps) {
            const uint32_t keepN = (uint32_t)ks.idx.size();
            PgMasks pm(n_raw, ks.full ? std::vector<uint32_t>{} : ks.idx);
            BedMask bm(n_raw, ks.idx);
            const uint64_t* km = ks.full ? nullptr : bm.p;
            pr.ReadRawFullHard(genovec.p, (int)v);
            plink2::ZeroTrailingNyps(n_raw, genovec.p);
            std::array<uint32_t, 4> gc{};
            if (ks.full) plink2::GenoarrCountFreqsUnsafe(genovec.p, keepN, gc);
            else plink2::GenoarrCountSubsetFreqs(genovec.p, pm.inter.p, n_raw, keepN, gc);

            std::vector<int8_t> r_i8(keepN);
            PgenReader::ExtractInt8Ext(genovec.p, pm.incl.p, n_raw, keepN, r_i8.data(), nullptr);

            for (bed_simd::Isa isa : g_isas) {
                bed_simd::set_isa(isa);
                ++g_cases;
                char ctx[160];
                std::snprintf(ctx, sizeof ctx, "isa=%s large n_raw=%u v=%u pat=%d keep=%s",
                              isa_str(isa), n_raw, v, pats[v], ks.name.c_str());
                g_ctx = ctx;
                const Counts c = bed_simd::count(raw, km, n_raw, keepN);
                CHECK(eqc(c, gc), "large count mismatch: (%u,%u,%u,%u) vs (%u,%u,%u,%u)",
                      c.n0, c.n1, c.n2, c.nmiss, gc[0], gc[1], gc[2], gc[3]);
                Guarded<int8_t> o(keepN);
                const Counts ci = bed_simd::decode_i8(raw, km, n_raw, keepN, o.p, nullptr);
                CHECK(eqc(ci, gc), "large decode_i8 counts mismatch");
                CHECK(std::memcmp(o.p, r_i8.data(), keepN) == 0, "large decode_i8 output mismatch");
                CHECK(o.guard_ok(), "large decode_i8 overrun");
            }
        }
        std::free(raw);
    }
    pr.Close();
    std::remove(path.c_str());
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : (std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp");
    if (argc > 2) g_rng.seed(std::strtoull(argv[2], nullptr, 0));
    for (bed_simd::Isa i : {bed_simd::Isa::Scalar, bed_simd::Isa::Neon, bed_simd::Isa::Avx2, bed_simd::Isa::Avx512})
        if (bed_simd::set_isa(i)) g_isas.push_back(i);
    std::printf("ISAs:");
    for (auto i : g_isas) std::printf(" %s", isa_str(i));
    std::printf(" (best=%s)\n", isa_str(bed_simd::best_isa()));
    const uint32_t sizes[] = {1, 3, 4, 5, 63, 64, 65, 255, 256, 257, 1000, 4097};
    for (uint32_t n : sizes) run_file(dir, n);
    run_large_n(dir, 70000);
    run_large_n(dir, 270000);
    std::printf("%s: %llu cases, %llu checks, %llu failures\n", g_fail ? "FAIL" : "PASS",
                (unsigned long long)g_cases, (unsigned long long)g_checks, (unsigned long long)g_fail);
    return g_fail ? 1 : 0;
}
