// Microbenchmark: native bed_simd kernels vs the previous PgenReader-based
// BED path (PgrPlink1ToPlink2InplaceUnsafe + CountHardFreqMissExt +
// Extract{Int8,Float}Ext).  Not part of GCTAg; built with
// -DGCTA_BUILD_BED_TESTS=ON (target bench_bed_simd).
//
//   bench_bed_simd [n_samples=100000] [n_variants=2000] [keep_frac=1.0] [miss_rate=0.0]
#include "bed_simd.hpp"
#include "third_party/Pgenlib/PgenReader.h"
#include "pgenlib_read.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const uint32_t N = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 100000u;
    const uint32_t M = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 2000u;
    const double keep_frac = argc > 3 ? std::atof(argv[3]) : 1.0;
    const double miss = argc > 4 ? std::atof(argv[4]) : 0.0;

    std::mt19937_64 rng(42);
    const uint32_t rec = bed_simd::record_bytes(N);
    const int strideW = PgenReader::GetGenoBufPtrSize(N);
    const size_t stride = (size_t)strideW * 8;
    std::vector<uint64_t> raw((size_t)M * strideW + 8, 0);
    std::uniform_real_distribution<double> U(0, 1);
    for (uint32_t v = 0; v < M; ++v) {
        uint8_t* r = (uint8_t*)raw.data() + v * stride;
        const double p = 0.05 + 0.9 * U(rng);
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t code;
            if (miss > 0 && U(rng) < miss) code = 1;
            else {
                const int g = (U(rng) < p) + (U(rng) < p);
                code = g == 2 ? 0 : (g == 1 ? 2 : 3);
            }
            r[i >> 2] |= (uint8_t)(code << (2 * (i & 3)));
        }
    }
    std::vector<uint32_t> keepIdx;
    for (uint32_t i = 0; i < N; ++i)
        if (keep_frac >= 1.0 || U(rng) < keep_frac) keepIdx.push_back(i);
    const uint32_t K = (uint32_t)keepIdx.size();
    const bool full = (K == N);

    std::vector<uint64_t> keep(bed_simd::keep_mask_words(N));
    bed_simd::build_keep_mask(keepIdx.data(), keepIdx.size(), N, keep.data());
    const uint64_t* kp = full ? nullptr : keep.data();
    const int maskW = PgenReader::GetSubsetMaskSize(N);
    std::vector<uintptr_t> incl(maskW + 8), inter(maskW + 8);
    PgenReader::SetSampleSubsets(keepIdx, N, incl.data(), inter.data() + 4);

    std::vector<int8_t> out_i8(K + 64);
    std::vector<float> out_f(K + 64);
    std::vector<uint64_t> missb((K + 63) / 64 + 8);
    const float tab[4] = {-0.9f, 0.1f, 1.1f, 0.0f};
    const float qt[1024] __attribute__((aligned(16))) = QUAD_TABLE256(-0.9f, 0.1f, 1.1f, 0.0f);

    auto bench = [&](const char* name, auto&& fn) {
        fn();  // warm
        const auto t0 = clk::now();
        int reps = 0;
        double el = 0;
        do {
            fn();
            ++reps;
            el = std::chrono::duration<double>(clk::now() - t0).count();
        } while (el < 0.5);
        const double gs = (double)N * M * reps / el / 1e9;
        std::printf("  %-44s %8.2f Gsample/s  (%.3f ms / pass)\n", name, gs, 1e3 * el / reps);
    };

    std::printf("N=%u M=%u keep=%u (%.1f%%) miss=%.3f isa=%s\n", N, M, K, 100.0 * K / N, miss, bed_simd::isa_name());
    volatile uint64_t sink = 0;
    std::vector<uint64_t> conv(raw.size());
    bench("old: Plink1->2 convert + CountHardFreqMissExt", [&] {
        std::memcpy(conv.data(), raw.data(), raw.size() * 8);
        for (uint32_t v = 0; v < M; ++v) {
            uintptr_t* b = (uintptr_t*)((uint8_t*)conv.data() + v * stride);
            plink2::PgrPlink1ToPlink2InplaceUnsafe(N, b);
            SNPInfo s;
            PgenReader::CountHardFreqMissExt(b, inter.data() + 4, N, K, &s, true);
            sink += s.N;
        }
    });
    bench("new: bed_simd::count", [&] {
        for (uint32_t v = 0; v < M; ++v) {
            const auto c = bed_simd::count((const uint8_t*)raw.data() + v * stride, kp, N, K);
            sink += c.n1;
        }
    });
    std::memcpy(conv.data(), raw.data(), raw.size() * 8);
    for (uint32_t v = 0; v < M; ++v) plink2::PgrPlink1ToPlink2InplaceUnsafe(N, (uintptr_t*)((uint8_t*)conv.data() + v * stride));
    bench("old: ExtractInt8Ext (+miss) [pre-converted]", [&] {
        for (uint32_t v = 0; v < M; ++v)
            PgenReader::ExtractInt8Ext((uintptr_t*)((uint8_t*)conv.data() + v * stride), incl.data(), N, K, out_i8.data(),
                                       (uintptr_t*)missb.data());
    });
    bench("new: decode_i8 fused (counts, lazy miss)", [&] {
        bed_simd::I8Opts o;
        o.miss_bits = missb.data();
        for (uint32_t v = 0; v < M; ++v) {
            const auto c = bed_simd::decode_i8((const uint8_t*)raw.data() + v * stride, kp, N, K, out_i8.data(), &o);
            sink += c.n2;
        }
    });
    bench("old: ExtractFloatExt (4 KiB table) [pre-conv]", [&] {
        for (uint32_t v = 0; v < M; ++v)
            PgenReader::ExtractFloatExt((uintptr_t*)((uint8_t*)conv.data() + v * stride), incl.data(), N, K, qt,
                                        out_f.data(), nullptr);
    });
    bench("new: decode_f32 (register table)", [&] {
        for (uint32_t v = 0; v < M; ++v)
            bed_simd::decode_f32((const uint8_t*)raw.data() + v * stride, kp, N, K, tab, out_f.data(), nullptr);
    });
    (void)rec;
    return (int)(sink & 0);
}
