// -----------------------------------------------------------------------------
// bench_pgen_vs_bed.cpp — Microbenchmark comparing:
//   1. Legacy PgenReader (.pgen): ReadRawFullHard + CountHardFreqMissExt + ExtractInt8Ext
//   2. Native BED (.bed): SpanBatchCoalescer::read_spans + bed_simd::{count, decode_i8}
//   3. Native PGEN Hardcall (.pgen): CompressedBatchReader::read_batch +
//      PgenThreadCache::{resolve, decode_into} + bed_simd::{count, decode_i8}
//
// Generates a realistic LD-structured + MAF-spectrum dataset compressed via
// PLINK2's STPgenWriter (achieving ~2.3x-3.0x compression over raw .bed,
// representative of high-density SNP array / biobank cohorts), verifies exact
// parity, and reports compressed I/O savings and single-thread / OpenMP
// multi-thread Gsample/s throughput.
//
// Usage: bench_pgen_vs_bed [n_samples=50000] [n_variants=2048] [n_threads=omp_max]
// -----------------------------------------------------------------------------

#include "pgen_hardcall.hpp"
#include "posix_span_io.hpp"
#include "bed_simd.hpp"
#include "third_party/Pgenlib/PgenReader.h"
#include "pgenlib_write.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const uint32_t N = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 50000u;
    const uint32_t M = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 2048u;
    const int threads = (argc > 3) ? std::max(1, std::atoi(argv[3])) : omp_get_max_threads();

    std::string tmpdir;
    {
        std::error_code ec;
        const char* env_tmp = std::getenv("TMPDIR");
        std::string cand = (env_tmp && env_tmp[0]) ? std::string(env_tmp) : std::string("/tmp");
        if (!cand.empty() && cand.back() == '/') cand.pop_back();
        tmpdir = cand + "/gcta_bench_pgen_vs_bed_" + std::to_string(::getpid());
        fs::create_directories(tmpdir, ec);
        if (ec) {
            ec.clear();
            tmpdir = "./gcta_bench_pgen_vs_bed_" + std::to_string(::getpid());
            fs::create_directories(tmpdir, ec);
        }
    }
    const std::string bed_path = tmpdir + "/bench.bed";
    const std::string pgen_path = tmpdir + "/bench.pgen";

    const uint32_t rb = bed_simd::record_bytes(N);
    std::mt19937_64 rng(0xBEEFCAFE2026ULL);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Synthesize LD-block + MAF-spectrum genotypes (array + rare + LD blocks -> ~2.3x-2.8x compression)
    std::vector<uint8_t> bed_bytes(3u + static_cast<std::size_t>(M) * rb, 0u);
    bed_bytes[0] = 0x6cu;
    bed_bytes[1] = 0x1bu;
    bed_bytes[2] = 0x01u;

    plink2::STPgenWriter spgw;
    plink2::PreinitSpgw(&spgw);
    uintptr_t alloc_cl = 0;
    uint32_t max_vrec = 0;
    if (plink2::SpgwInitPhase1(pgen_path.c_str(), nullptr, nullptr, M, N, 0,
                               plink2::kPgenWriteBackwardSeek, plink2::kfPgenGlobal0,
                               0, &spgw, &alloc_cl, &max_vrec) != plink2::kPglRetSuccess) {
        std::fprintf(stderr, "SpgwInitPhase1 failed\n");
        return 2;
    }
    AlignedRawBuffer<uint8_t, 64> spgw_mem(alloc_cl * plink2::kCacheline + 128u);
    std::memset(spgw_mem.data(), 0, spgw_mem.size());
    plink2::SpgwInitPhase2(max_vrec, &spgw, spgw_mem.data());

    AlignedRawBuffer<uintptr_t, 64> genovec(plink2::NypCtToAlignedWordCt(N));
    std::vector<uint8_t> cur_p2(N, 0u), anchor_p2(N, 0u);

    for (uint32_t v = 0; v < M; ++v) {
        const uint32_t mod = v % 8u;
        if (mod == 0u) {
            // Common LD anchor (MAF 0.10 .. 0.45)
            const double maf = 0.10 + 0.35 * uni(rng);
            for (uint32_t s = 0; s < N; ++s) {
                const double u = uni(rng);
                if (u < 0.002) anchor_p2[s] = 3u;
                else {
                    const uint32_t g = (uni(rng) < maf ? 1u : 0u) + (uni(rng) < maf ? 1u : 0u);
                    anchor_p2[s] = static_cast<uint8_t>(g);
                }
            }
            cur_p2 = anchor_p2;
        } else if (mod <= 3u) {
            // Strong LD block (~1.5% recombination/mutation divergence -> vrtype 2)
            cur_p2 = anchor_p2;
            const uint32_t n_flips = std::max<uint32_t>(1u, N / 70u);
            for (uint32_t f = 0; f < n_flips; ++f) {
                const uint32_t s = static_cast<uint32_t>(rng() % N);
                cur_p2[s] = static_cast<uint8_t>((cur_p2[s] + 1u) % 3u);
            }
        } else if (mod == 4u) {
            // Flipped-allele LD SNP (~1.5% divergence -> vrtype 3)
            for (uint32_t s = 0; s < N; ++s) {
                const uint8_t g = anchor_p2[s];
                cur_p2[s] = (g == 0u) ? 2u : (g == 2u ? 0u : g);
            }
            const uint32_t n_flips = std::max<uint32_t>(1u, N / 80u);
            for (uint32_t f = 0; f < n_flips; ++f) {
                const uint32_t s = static_cast<uint32_t>(rng() % N);
                cur_p2[s] = 1u;
            }
        } else if (mod == 5u) {
            // Moderate MAF ~0.12 -> mostly hom-ref + het (1-bit + difflist, vrtype 1)
            const double maf = 0.12;
            for (uint32_t s = 0; s < N; ++s) {
                const uint32_t g = (uni(rng) < maf ? 1u : 0u) + (uni(rng) < maf ? 1u : 0u);
                cur_p2[s] = static_cast<uint8_t>(g);
            }
            anchor_p2 = cur_p2;
        } else {
            // Low-frequency / rare variant (MAF ~0.015 -> difflist, vrtype 4)
            std::fill(cur_p2.begin(), cur_p2.end(), 0u);
            const uint32_t carriers = std::max<uint32_t>(1u, N / 35u);
            for (uint32_t c = 0; c < carriers; ++c) {
                const uint32_t s = static_cast<uint32_t>(rng() % N);
                cur_p2[s] = (c % 16u == 0u) ? 2u : 1u;
            }
            anchor_p2 = cur_p2;
        }

        std::memset(genovec.data(), 0, genovec.size() * sizeof(uintptr_t));
        auto* gw_bytes = reinterpret_cast<uint8_t*>(genovec.data());
        uint8_t* bed_rec = bed_bytes.data() + 3u + static_cast<std::size_t>(v) * rb;
        for (uint32_t s = 0; s < N; ++s) {
            const uint8_t p2 = cur_p2[s] & 3u;
            gw_bytes[s >> 2u] |= static_cast<uint8_t>(p2 << ((s & 3u) * 2u));
            bed_rec[s >> 2u] |= static_cast<uint8_t>(pgen_hardcall::kP2ToP1[p2] << ((s & 3u) * 2u));
        }
        if (plink2::SpgwAppendBiallelicGenovec(genovec.data(), &spgw) != plink2::kPglRetSuccess) {
            std::fprintf(stderr, "SpgwAppend failed\n");
            return 2;
        }
    }

    plink2::PglErr err = plink2::SpgwFinish(&spgw);
    plink2::CleanupSpgw(&spgw, &err);
    {
        FILE* f = std::fopen(bed_path.c_str(), "wb");
        if (!f || std::fwrite(bed_bytes.data(), 1, bed_bytes.size(), f) != bed_bytes.size()) {
            std::fprintf(stderr, "Failed writing %s\n", bed_path.c_str());
            return 2;
        }
        std::fclose(f);
    }

    const pgen_hardcall::PgenFileIndex pgen_idx = pgen_hardcall::load_pgen_index(pgen_path, N, M);
    gcta_io::PosixUniqueFd bed_fd = gcta_io::open_posix_readonly_throwing(bed_path);
    gcta_io::PosixUniqueFd pgen_fd = gcta_io::open_posix_readonly_throwing(pgen_path);
    const uint64_t bed_fsize = gcta_io::stat_file_size_throwing(bed_fd.get(), bed_path);
    const uint64_t pgen_fsize = gcta_io::stat_file_size_throwing(pgen_fd.get(), pgen_path);

    // Count vrtype distribution
    uint32_t vt_hist[8] = {};
    for (uint32_t v = 0; v < M; ++v) ++vt_hist[pgen_idx.vrtype(v) & 7u];

    std::printf("================================================================================\n");
    std::printf("PGEN Hardcall vs BED Microbenchmark\n");
    std::printf("  N = %u samples, M = %u variants, threads = %d\n", N, M, threads);
    std::printf("  ISAs: pgen_hardcall=%s, bed_simd=%s\n",
                pgen_hardcall::isa_name(), bed_simd::isa_name());
    std::printf("  File sizes: BED = %.2f MiB (%.1f B/var) | PGEN = %.2f MiB (%.1f B/var, %.2fx smaller)\n",
                static_cast<double>(bed_fsize) / (1024.0 * 1024.0),
                static_cast<double>(rb),
                static_cast<double>(pgen_fsize) / (1024.0 * 1024.0),
                pgen_idx.avg_rec_bytes,
                static_cast<double>(bed_fsize) / static_cast<double>(pgen_fsize));
    std::printf("  vrtype mix: vt0=%u vt1=%u vt2(LD)=%u vt3(invLD)=%u vt4=%u vt5=%u vt6=%u vt7=%u\n",
                vt_hist[0], vt_hist[1], vt_hist[2], vt_hist[3],
                vt_hist[4], vt_hist[5], vt_hist[6], vt_hist[7]);
    std::printf("================================================================================\n");

    std::vector<uint32_t> all_vidx(M);
    std::iota(all_vidx.begin(), all_vidx.end(), 0u);

    pgen_hardcall::CompressedBatchReader pgen_batch_reader;
    AlignedRawBuffer<uint8_t, 64> pgen_blk;
    pgen_batch_reader.read_batch(pgen_fd.get(), pgen_idx, all_vidx, 1ULL, pgen_blk);

    gcta_io::SpanBatchCoalescer bed_coalescer;
    AlignedRawBuffer<uint8_t, 64> bed_blk(static_cast<std::size_t>(M) * rb);
    gcta_io::FileByteSpan bed_span{3ULL, static_cast<std::size_t>(M) * rb, bed_blk.data()};
    bed_coalescer.read_spans(bed_fd.get(), bed_path, std::span<const gcta_io::FileByteSpan>(&bed_span, 1));

    // Quick parity check before timing
    {
        pgen_hardcall::PgenThreadCache tls;
        for (uint32_t v = 0; v < M; ++v) {
            const uint8_t* p_rec = tls.resolve(pgen_blk.data(), v);
            const uint8_t* b_rec = bed_blk.data() + static_cast<std::size_t>(v) * rb;
            if (std::memcmp(p_rec, b_rec, rb) != 0) {
                std::fprintf(stderr, "Parity mismatch at v=%u before benchmark!\n", v);
                return 1;
            }
        }
    }

    auto run_bench = [&](const char* label, auto&& fn) {
        fn(); // warmup
        const auto t0 = Clock::now();
        int reps = 0;
        double elapsed = 0.0;
        do {
            fn();
            ++reps;
            elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        } while (elapsed < 0.35);
        const double ms_per_pass = 1e3 * elapsed / reps;
        const double gsamples_s = (static_cast<double>(N) * M * reps) / elapsed / 1e9;
        std::printf("  %-52s %8.2f Gsample/s  (%7.3f ms/pass)\n", label, gsamples_s, ms_per_pass);
    };

    volatile uint64_t sink = 0;

    // 1. Legacy single-thread PgenReader (.pgen)
    {
        PgenReader pr;
        uint32_t pn = N, pm = M;
        pr.Load(pgen_path, &pn, &pm, {});
        AlignedRawBuffer<uintptr_t, 64> gw(PgenReader::GetGenoBufPtrSize(N) + 8u);
        std::vector<int8_t> i8_out(N + 64u);
        std::vector<uintptr_t> miss_b((N + 63u) / 64u + 8u);

        run_bench("1T Legacy PgenReader (ReadRawHard+Count+ExtractI8)", [&]() {
            uint64_t local_s = 0;
            for (uint32_t v = 0; v < M; ++v) {
                pr.ReadRawFullHard(gw.data(), static_cast<int>(v));
                SNPInfo info{};
                PgenReader::CountHardFreqMissExt(gw.data(), nullptr, N, N, &info, true);
                PgenReader::ExtractInt8Ext(gw.data(), nullptr, N, N, i8_out.data(), miss_b.data());
                local_s += info.N + static_cast<uint8_t>(i8_out[0]);
            }
            sink += local_s;
        });
    }

    // 2. 1-Thread BED vs Native PGEN (resolve + decode_i8 fused)
    {
        std::vector<int8_t> i8_out(N + 64u);
        std::vector<uint64_t> miss_b((N + 63u) / 64u + 8u);

        run_bench("1T Native BED (bed_simd::decode_i8 fused)", [&]() {
            uint64_t local_s = 0;
            bed_simd::I8Opts opts;
            opts.miss_bits = miss_b.data();
            for (uint32_t v = 0; v < M; ++v) {
                const uint8_t* rec = bed_blk.data() + static_cast<std::size_t>(v) * rb;
                const auto c = bed_simd::decode_i8(rec, nullptr, N, N, i8_out.data(), &opts);
                local_s += c.n1 + static_cast<uint8_t>(i8_out[0]);
            }
            sink += local_s;
        });

        uint64_t seq = 10;
        pgen_hardcall::PgenThreadCache tls;
        run_bench("1T Native PGEN (L2 resolve + bed_simd::decode_i8)", [&]() {
            reinterpret_cast<pgen_hardcall::PgenBlockHeader*>(pgen_blk.data())->block_seq_id = ++seq;
            uint64_t local_s = 0;
            bed_simd::I8Opts opts;
            opts.miss_bits = miss_b.data();
            for (uint32_t v = 0; v < M; ++v) {
                const uint8_t* rec = tls.resolve(pgen_blk.data(), v);
                const auto c = bed_simd::decode_i8(rec, nullptr, N, N, i8_out.data(), &opts);
                local_s += c.n1 + static_cast<uint8_t>(i8_out[0]);
            }
            sink += local_s;
        });
    }

    // 3. Multi-threaded OpenMP MLMA pattern (resolve + decode_i8) & GRM Stage 1 pattern (decode_into + count)
    {
        AlignedRawBuffer<uint8_t, 64> unpack_slab(static_cast<std::size_t>(M) * rb);
        uint64_t seq = 1000;

        char label_bed_mlma[96];
        std::snprintf(label_bed_mlma, sizeof(label_bed_mlma),
                      "%dT Native BED OpenMP (decode_i8 fused)", threads);
        run_bench(label_bed_mlma, [&]() {
            uint64_t total_s = 0;
            #pragma omp parallel num_threads(threads) reduction(+:total_s)
            {
                std::vector<int8_t> i8_out(N + 64u);
                std::vector<uint64_t> miss_b((N + 63u) / 64u + 8u);
                bed_simd::I8Opts opts;
                opts.miss_bits = miss_b.data();
                #pragma omp for schedule(static)
                for (uint32_t v = 0; v < M; ++v) {
                    const uint8_t* rec = bed_blk.data() + static_cast<std::size_t>(v) * rb;
                    const auto c = bed_simd::decode_i8(rec, nullptr, N, N, i8_out.data(), &opts);
                    total_s += c.n1 + static_cast<uint8_t>(i8_out[0]);
                }
            }
            sink += total_s;
        });

        char label_pgen_mlma[96];
        std::snprintf(label_pgen_mlma, sizeof(label_pgen_mlma),
                      "%dT Native PGEN OpenMP (L2 resolve + decode_i8)", threads);
        run_bench(label_pgen_mlma, [&]() {
            reinterpret_cast<pgen_hardcall::PgenBlockHeader*>(pgen_blk.data())->block_seq_id = ++seq;
            uint64_t total_s = 0;
            #pragma omp parallel num_threads(threads) reduction(+:total_s)
            {
                thread_local pgen_hardcall::PgenThreadCache tls;
                std::vector<int8_t> i8_out(N + 64u);
                std::vector<uint64_t> miss_b((N + 63u) / 64u + 8u);
                bed_simd::I8Opts opts;
                opts.miss_bits = miss_b.data();
                #pragma omp for schedule(static)
                for (uint32_t v = 0; v < M; ++v) {
                    const uint8_t* rec = tls.resolve(pgen_blk.data(), v);
                    const auto c = bed_simd::decode_i8(rec, nullptr, N, N, i8_out.data(), &opts);
                    total_s += c.n1 + static_cast<uint8_t>(i8_out[0]);
                }
            }
            sink += total_s;
        });

        char label_bed_grm1[96];
        std::snprintf(label_bed_grm1, sizeof(label_bed_grm1),
                      "%dT Native BED GRM-Stage1 (pread + count)", threads);
        run_bench(label_bed_grm1, [&]() {
            bed_coalescer.read_spans(bed_fd.get(), bed_path,
                                     std::span<const gcta_io::FileByteSpan>(&bed_span, 1));
            uint64_t total_s = 0;
            #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:total_s)
            for (uint32_t v = 0; v < M; ++v) {
                const auto c = bed_simd::count(
                    bed_blk.data() + static_cast<std::size_t>(v) * rb, nullptr, N, N);
                total_s += c.n1;
            }
            sink += total_s;
        });

        char label_pgen_grm1[96];
        std::snprintf(label_pgen_grm1, sizeof(label_pgen_grm1),
                      "%dT Native PGEN GRM-Stage1 (read_batch+decode_into+count)", threads);
        run_bench(label_pgen_grm1, [&]() {
            pgen_batch_reader.read_batch(pgen_fd.get(), pgen_idx, all_vidx, ++seq, pgen_blk);
            uint64_t total_s = 0;
            #pragma omp parallel num_threads(threads) reduction(+:total_s)
            {
                thread_local pgen_hardcall::PgenThreadCache tls;
                #pragma omp for schedule(static)
                for (uint32_t v = 0; v < M; ++v) {
                    uint8_t* dst = unpack_slab.data() + static_cast<std::size_t>(v) * rb;
                    tls.decode_into(pgen_blk.data(), v, dst);
                    const auto c = bed_simd::count(dst, nullptr, N, N);
                    total_s += c.n1;
                }
            }
            sink += total_s;
        });
    }

    std::error_code ec;
    fs::remove_all(tmpdir, ec);
    return static_cast<int>(sink & 0u);
}
