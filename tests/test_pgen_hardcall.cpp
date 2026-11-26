// -----------------------------------------------------------------------------
// test_pgen_hardcall.cpp — Comprehensive unit, parity, sanitizer & fuzz suite
// for include/posix_span_io.hpp and include/pgen_hardcall.hpp:
//   * gcta_io::PosixUniqueFd + SpanBatchCoalescer (contiguous coalescing,
//     gap bridging <= max_gap_bytes, >kMaxIov flushes, non-ascending flushes,
//     zero-length spans, EOF / error handling)
//   * All PGEN modes: 0x01 (.bed), 0x02 (fixed hardcall), 0x03/0x04 (fixed
//     dosage fallback), 0x10/0x11 (inline variable-width + extensions),
//     0x20/0x21 (external .pgi index)
//   * All vrtypes: 0, 1 (all 6 common2_code pairs), 2 (LD), 3 (inverted LD),
//     4, 5 (0-byte all-hom-ref), 6, 7, phased hardcalls (| 0x10), fused
//     header_ctrl (8, 9, 10, 11, 12, 14), 1/2/3/4-byte vrec lengths
//   * Multi-vblock M > 65536 (M = 70000 across 65536 boundary)
//   * Non-contiguous --extract subsets dropping LD anchors + multi-threaded
//     OpenMP PgenThreadCache::resolve & PgenThreadCache::decode_into
//   * Dosage / multiallelic / GCTA_DISABLE_NATIVE_PGEN fallback detection
//   * Truncated / malformed header & record bounds-check exceptions
//   * Bit-for-bit parity against PgenReader + bed_simd across every supported
//     ISA (Scalar, NEON / AVX2 / AVX-512)
// -----------------------------------------------------------------------------

#include "pgen_hardcall.hpp"
#include "posix_span_io.hpp"
#include "bed_simd.hpp"
#include "third_party/Pgenlib/PgenReader.h"
#include "pgenlib_write.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

uint64_t g_checks = 0;
uint64_t g_fail = 0;
std::string g_ctx = "init";

#define CHECK(cond, ...)                                                                        \
    do {                                                                                        \
        ++g_checks;                                                                             \
        if (!(cond)) {                                                                          \
            if (++g_fail <= 50) {                                                               \
                std::fprintf(stderr, "FAIL [%s] line %d: ", g_ctx.c_str(), __LINE__);           \
                std::fprintf(stderr, __VA_ARGS__);                                              \
                std::fprintf(stderr, "\n");                                                     \
            }                                                                                   \
        }                                                                                       \
    } while (0)

#define CHECK_THROWS(expr, ...)                                                                 \
    do {                                                                                        \
        ++g_checks;                                                                             \
        bool caught_ = false;                                                                   \
        try {                                                                                   \
            (void)(expr);                                                                       \
        } catch (const std::runtime_error&) {                                                   \
            caught_ = true;                                                                     \
        }                                                                                       \
        if (!caught_) {                                                                         \
            if (++g_fail <= 50) {                                                               \
                std::fprintf(stderr, "FAIL (did not throw) [%s] line %d: ",                     \
                             g_ctx.c_str(), __LINE__);                                          \
                std::fprintf(stderr, __VA_ARGS__);                                              \
                std::fprintf(stderr, "\n");                                                     \
            }                                                                                   \
        }                                                                                       \
    } while (0)

std::mt19937_64 g_rng(0xC0FFEE123456789BULL);
uint64_t rnd64() { return g_rng(); }
uint32_t rndu(uint32_t n) { return n ? static_cast<uint32_t>(rnd64() % n) : 0u; }
double rnd01() { return static_cast<double>(rnd64() >> 11) * (1.0 / 9007199254740992.0); }

void write_bytes_to_file(const std::string& path, const void* data, std::size_t len) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || (len > 0 && std::fwrite(data, 1, len, f) != len)) {
        std::fprintf(stderr, "Failed to write %s\n", path.c_str());
        std::exit(2);
    }
    std::fclose(f);
}

void append_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    for (int s = 0; s < 32; s += 8) buf.push_back(static_cast<uint8_t>((v >> s) & 0xffu));
}

void append_u64_le(std::vector<uint8_t>& buf, uint64_t v) {
    for (int s = 0; s < 64; s += 8) buf.push_back(static_cast<uint8_t>((v >> s) & 0xffULL));
}

void append_sub_u32_le(std::vector<uint8_t>& buf, uint32_t v, uint32_t byte_ct) {
    for (uint32_t b = 0; b < byte_ct; ++b) {
        buf.push_back(static_cast<uint8_t>((v >> (8u * b)) & 0xffu));
    }
}

void append_vint31(std::vector<uint8_t>& buf, uint32_t v) {
    while (v > 0x7fu) {
        buf.push_back(static_cast<uint8_t>((v & 0x7fu) | 0x80u));
        v >>= 7u;
    }
    buf.push_back(static_cast<uint8_t>(v));
}

uint32_t sample_id_bytes(uint32_t raw_sample_ct) {
    return 1u + ((31u - static_cast<uint32_t>(__builtin_clz(raw_sample_ct))) / 8u);
}

// Pack per-sample PLINK2 codes {0,1,2,3} into a 64B-aligned genovec word buffer
void pack_p2_genovec(const std::vector<uint8_t>& p2_codes, uintptr_t* genovec) {
    const uint32_t n = static_cast<uint32_t>(p2_codes.size());
    const std::size_t nwords = plink2::NypCtToAlignedWordCt(n);
    std::memset(genovec, 0, nwords * sizeof(uintptr_t));
    auto* bytes = reinterpret_cast<uint8_t*>(genovec);
    for (uint32_t s = 0; s < n; ++s) {
        bytes[s >> 2u] |= static_cast<uint8_t>((p2_codes[s] & 3u) << ((s & 3u) * 2u));
    }
}

// Convert per-sample PLINK2 codes {0,1,2,3} into expected PLINK1 .bed bytes (trailing bits zeroed)
std::vector<uint8_t> expected_bed2_from_p2(const std::vector<uint8_t>& p2_codes) {
    const uint32_t n = static_cast<uint32_t>(p2_codes.size());
    std::vector<uint8_t> bed((n + 3u) / 4u, 0u);
    for (uint32_t s = 0; s < n; ++s) {
        const uint8_t p1 = pgen_hardcall::kP2ToP1[p2_codes[s] & 3u];
        bed[s >> 2u] |= static_cast<uint8_t>(p1 << ((s & 3u) * 2u));
    }
    return bed;
}

// Build a valid PLINK2 difflist byte stream for strictly increasing (sample_idx, p2_code) pairs
std::vector<uint8_t> encode_difflist(uint32_t raw_sample_ct,
                                     const std::vector<std::pair<uint32_t, uint8_t>>& diffs) {
    std::vector<uint8_t> out;
    const uint32_t L = static_cast<uint32_t>(diffs.size());
    append_vint31(out, L);
    if (L == 0u) return out;

    const uint32_t group_ct = (L + 63u) / 64u;
    const uint32_t sid_bytes = sample_id_bytes(raw_sample_ct);

    std::vector<std::vector<uint8_t>> group_deltas(group_ct);
    for (uint32_t g = 0; g < group_ct; ++g) {
        const uint32_t base = g * 64u;
        const uint32_t cnt = std::min<uint32_t>(64u, L - base);
        append_sub_u32_le(out, diffs[base].first, sid_bytes);
        for (uint32_t e = 1u; e < cnt; ++e) {
            const uint32_t delta = diffs[base + e].first - diffs[base + e - 1u].first;
            append_vint31(group_deltas[g], delta);
        }
    }
    for (uint32_t g = 0; g + 1u < group_ct; ++g) {
        // delta segment byte length minus 63
        out.push_back(static_cast<uint8_t>(group_deltas[g].size() - 63u));
    }
    const std::size_t rg_bytes = (L + 3u) / 4u;
    const std::size_t rg_start = out.size();
    out.resize(rg_start + rg_bytes, 0u);
    for (uint32_t i = 0; i < L; ++i) {
        out[rg_start + (i >> 2u)] |= static_cast<uint8_t>((diffs[i].second & 3u) << ((i & 3u) * 2u));
    }
    for (uint32_t g = 0; g < group_ct; ++g) {
        out.insert(out.end(), group_deltas[g].begin(), group_deltas[g].end());
    }
    return out;
}

// Encode a custom mode 0x10 / 0x11 .pgen file from raw (vrtype, payload, allele_ct) records
std::vector<uint8_t> build_custom_pgen_0x10(
    uint32_t raw_n,
    const std::vector<uint8_t>& vrtypes,
    const std::vector<std::vector<uint8_t>>& payloads,
    uint32_t len_bytes = 2u,
    bool eight_bit_vrtypes = true,
    const std::vector<uint8_t>* allele_cts = nullptr,
    bool with_nonref_flags = false,
    bool mode_0x11_with_pad = false) {
    const uint32_t raw_m = static_cast<uint32_t>(vrtypes.size());
    const uint32_t vblock_ct = (raw_m + 65535u) / 65536u;

    uint8_t header_ctrl = static_cast<uint8_t>((len_bytes - 1u) & 3u);
    if (eight_bit_vrtypes) header_ctrl |= 4u;
    if (allele_cts != nullptr) header_ctrl |= (1u << 4u);
    if (with_nonref_flags) header_ctrl |= (3u << 6u);

    std::vector<uint8_t> meta;
    for (uint32_t vb = 0; vb < vblock_ct; ++vb) {
        const uint32_t v0 = vb * 65536u;
        const uint32_t cnt = std::min<uint32_t>(65536u, raw_m - v0);
        if (eight_bit_vrtypes) {
            meta.insert(meta.end(), vrtypes.begin() + v0, vrtypes.begin() + v0 + cnt);
        } else {
            const uint32_t nb = (cnt + 1u) / 2u;
            for (uint32_t i = 0; i < nb; ++i) {
                const uint8_t lo = vrtypes[v0 + 2u * i] & 0x0fu;
                const uint8_t hi = (2u * i + 1u < cnt) ? (vrtypes[v0 + 2u * i + 1u] & 0x0fu) : 0u;
                meta.push_back(static_cast<uint8_t>(lo | (hi << 4u)));
            }
        }
        for (uint32_t i = 0; i < cnt; ++i) {
            append_sub_u32_le(meta, static_cast<uint32_t>(payloads[v0 + i].size()), len_bytes);
        }
        if (allele_cts != nullptr) {
            meta.insert(meta.end(), allele_cts->begin() + v0, allele_cts->begin() + v0 + cnt);
        }
        if (with_nonref_flags) {
            const uint32_t nf_bytes = (cnt + 7u) / 8u;
            meta.insert(meta.end(), nf_bytes, 0xA5u);
        }
    }

    // Optional mode 0x11 empty header/footer extension flags + extra bytes before var_fpos[0]
    std::vector<uint8_t> ext_hdr;
    if (mode_0x11_with_pad) {
        ext_hdr = {0x00u, 0x00u}; // no header exts, no footer exts (valid for PgenReader too!)
    }

    const uint64_t first_fpos = 12ULL + 8ULL * vblock_ct + meta.size() + ext_hdr.size();
    std::vector<uint64_t> vb_fpos(vblock_ct, 0ULL);
    uint64_t run_fpos = first_fpos;
    for (uint32_t v = 0; v < raw_m; ++v) {
        if ((v % 65536u) == 0u) vb_fpos[v / 65536u] = run_fpos;
        run_fpos += payloads[v].size();
    }

    std::vector<uint8_t> file;
    file.push_back(0x6cu);
    file.push_back(0x1bu);
    file.push_back(mode_0x11_with_pad ? 0x11u : 0x10u);
    append_u32_le(file, raw_m);
    append_u32_le(file, raw_n);
    file.push_back(header_ctrl);
    for (uint32_t vb = 0; vb < vblock_ct; ++vb) append_u64_le(file, vb_fpos[vb]);
    file.insert(file.end(), meta.begin(), meta.end());
    file.insert(file.end(), ext_hdr.begin(), ext_hdr.end());
    for (uint32_t v = 0; v < raw_m; ++v) {
        file.insert(file.end(), payloads[v].begin(), payloads[v].end());
    }
    return file;
}

// Write a genuine PLINK2 .pgen (and optional .pgi) using STPgenWriter
void write_spgw_biallelic(const std::string& path,
                          uint32_t raw_n,
                          const std::vector<std::vector<uint8_t>>& var_p2,
                          plink2::PgenWriteMode write_mode,
                          bool with_hphase = false) {
    const uint32_t raw_m = static_cast<uint32_t>(var_p2.size());
    plink2::STPgenWriter spgw;
    plink2::PreinitSpgw(&spgw);
    uintptr_t alloc_cl = 0;
    uint32_t max_vrec = 0;
    const plink2::PgenGlobalFlags gflags =
        with_hphase ? plink2::kfPgenGlobalHardcallPhasePresent : plink2::kfPgenGlobal0;

    plink2::PglErr err = plink2::SpgwInitPhase1(
        path.c_str(), nullptr, nullptr, raw_m, raw_n, 0,
        write_mode, gflags, 0, &spgw, &alloc_cl, &max_vrec);
    CHECK(err == plink2::kPglRetSuccess, "SpgwInitPhase1 failed (%d)", static_cast<int>(err));

    AlignedRawBuffer<uint8_t, 64> spgw_mem(alloc_cl * plink2::kCacheline + 128u);
    std::memset(spgw_mem.data(), 0, spgw_mem.size());
    plink2::SpgwInitPhase2(max_vrec, &spgw, spgw_mem.data());

    AlignedRawBuffer<uintptr_t, 64> genovec(plink2::NypCtToAlignedWordCt(raw_n) + 8u);
    AlignedRawBuffer<uintptr_t, 64> phasepresent(plink2::BitCtToWordCt(raw_n) + 8u);
    AlignedRawBuffer<uintptr_t, 64> phaseinfo(plink2::BitCtToWordCt(raw_n) + 8u);

    for (uint32_t v = 0; v < raw_m; ++v) {
        pack_p2_genovec(var_p2[v], genovec.data());
        if (with_hphase && (v % 3u == 0u)) {
            std::memset(phasepresent.data(), 0, phasepresent.size() * sizeof(uintptr_t));
            std::memset(phaseinfo.data(), 0, phaseinfo.size() * sizeof(uintptr_t));
            bool any_phased = false;
            for (uint32_t s = 0; s < raw_n; ++s) {
                if (var_p2[v][s] == 1u && ((s + v) & 1u)) {
                    plink2::SetBit(s, phasepresent.data());
                    if ((s ^ v) & 2u) plink2::SetBit(s, phaseinfo.data());
                    any_phased = true;
                }
            }
            if (any_phased) {
                err = plink2::SpgwAppendBiallelicGenovecHphase(
                    genovec.data(),
                    (v % 6u == 0u) ? nullptr : phasepresent.data(),
                    phaseinfo.data(), &spgw);
            } else {
                err = plink2::SpgwAppendBiallelicGenovec(genovec.data(), &spgw);
            }
        } else {
            err = plink2::SpgwAppendBiallelicGenovec(genovec.data(), &spgw);
        }
        CHECK(err == plink2::kPglRetSuccess, "SpgwAppend failed at v=%u", v);
    }

    err = plink2::SpgwFinish(&spgw);
    CHECK(err == plink2::kPglRetSuccess, "SpgwFinish failed (%d)", static_cast<int>(err));
    plink2::CleanupSpgw(&spgw, &err);
}

// Generate realistic structured genotypes that trigger ALL STPgenWriter encodings:
// vrtype 0 (common), vrtype 1 (mostly 2-value), vrtype 2 (LD), vrtype 3 (inv LD),
// vrtype 4 (rare alt), vrtype 5 (monomorphic ref), vrtype 6 (monomorphic/rare ref),
// vrtype 7 (mostly missing).
std::vector<std::vector<uint8_t>> generate_rich_p2_matrix(uint32_t raw_n, uint32_t raw_m) {
    std::vector<std::vector<uint8_t>> mat(raw_m, std::vector<uint8_t>(raw_n, 0u));
    std::vector<uint8_t> anchor(raw_n, 0u);

    for (uint32_t v = 0; v < raw_m; ++v) {
        const uint32_t kind = v % 10u;
        if (kind == 0u) {
            // Common MAF ~0.35 + missing -> vrtype 0 (new LD anchor)
            for (uint32_t s = 0; s < raw_n; ++s) {
                const double u = rnd01();
                anchor[s] = (u < 0.02) ? 3u : (u < 0.42 ? 0u : (u < 0.87 ? 1u : 2u));
            }
            mat[v] = anchor;
        } else if (kind == 1u || kind == 2u) {
            // High-LD copy of anchor with <= 1/32 flips -> vrtype 2
            mat[v] = anchor;
            const uint32_t flips = std::max<uint32_t>(0u, std::min<uint32_t>(raw_n / 32u, 3u));
            for (uint32_t f = 0; f < flips && raw_n > 0u; ++f) {
                mat[v][rndu(raw_n)] = static_cast<uint8_t>(rndu(4u));
            }
        } else if (kind == 3u) {
            // Inverted-LD copy of anchor -> vrtype 3
            for (uint32_t s = 0; s < raw_n; ++s) {
                const uint8_t g = anchor[s];
                mat[v][s] = (g == 0u) ? 2u : (g == 2u ? 0u : g);
            }
            if (raw_n >= 16u) {
                mat[v][rndu(raw_n)] = 3u;
                mat[v][rndu(raw_n)] = 1u;
            }
        } else if (kind == 4u) {
            // Mostly hom-ref & het, tiny hom-alt/missing -> vrtype 1
            for (uint32_t s = 0; s < raw_n; ++s) {
                mat[v][s] = (rnd01() < 0.65) ? 0u : 1u;
            }
            if (raw_n >= 24u) {
                mat[v][rndu(raw_n)] = 2u;
                mat[v][rndu(raw_n)] = 3u;
            }
            anchor = mat[v];
        } else if (kind == 5u) {
            // Sparse rare variant (modal 0) -> vrtype 4
            std::fill(mat[v].begin(), mat[v].end(), 0u);
            const uint32_t k = raw_n / 20u;
            for (uint32_t i = 0; i < k; ++i) mat[v][rndu(raw_n)] = (i & 1u) ? 1u : 2u;
            anchor = mat[v];
        } else if (kind == 6u) {
            // All hom-ref -> vrtype 5
            std::fill(mat[v].begin(), mat[v].end(), 0u);
        } else if (kind == 7u) {
            // Mostly hom-alt (modal 2) -> vrtype 6
            std::fill(mat[v].begin(), mat[v].end(), 2u);
            const uint32_t k = raw_n / 20u;
            for (uint32_t i = 0; i < k; ++i) mat[v][rndu(raw_n)] = (i & 1u) ? 0u : 1u;
            anchor = mat[v];
        } else if (kind == 8u) {
            // Mostly missing (modal 3) -> vrtype 7
            std::fill(mat[v].begin(), mat[v].end(), 3u);
            const uint32_t k = raw_n / 20u;
            for (uint32_t i = 0; i < k; ++i) mat[v][rndu(raw_n)] = static_cast<uint8_t>(i % 3u);
            anchor = mat[v];
        } else {
            // Another LD child off difflist anchor -> vrtype 2 / 3
            mat[v] = anchor;
            if (raw_n >= 16u) mat[v][rndu(raw_n)] = 1u;
        }
    }
    return mat;
}

// Verify full bit-for-bit parity of PgenFileIndex + CompressedBatchReader +
// PgenThreadCache (resolve & decode_into, single-thread & OpenMP) + bed_simd
// against PgenReader on a .pgen/.bed file!
void verify_parity_against_pgenreader(
    const std::string& path,
    uint32_t expected_n,
    uint32_t expected_m,
    const std::vector<std::vector<uint8_t>>* expected_p2_opt,
    const std::vector<pgen_hardcall::Isa>& pgen_isas) {
    const pgen_hardcall::PgenFileIndex idx =
        pgen_hardcall::load_pgen_index(path, expected_n, expected_m);
    CHECK(idx.native_hardcall_ok, "expected native_hardcall_ok=true for %s (reason=%s)",
          path.c_str(), idx.fallback_reason.c_str());
    CHECK(idx.raw_sample_ct == expected_n, "raw_sample_ct mismatch");
    CHECK(idx.raw_variant_ct == expected_m, "raw_variant_ct mismatch");
    CHECK(idx.bed_rec_bytes == bed_simd::record_bytes(expected_n), "bed_rec_bytes mismatch");

    // Reference PgenReader
    PgenReader pr;
    uint32_t pr_n = expected_n;
    uint32_t pr_m = expected_m;
    pr.Load(path, &pr_n, &pr_m, {});

    std::vector<uint32_t> keep_sub;
    for (uint32_t s = 0; s < expected_n; ++s) {
        if ((s * 7u + 3u) % 11u < 7u) keep_sub.push_back(s);
    }
    if (keep_sub.empty()) keep_sub.push_back(0u);
    const uint32_t keep_n = static_cast<uint32_t>(keep_sub.size());
    const bool is_sub = (keep_n < expected_n);

    AlignedRawBuffer<uint64_t, 64> bed_keep(bed_simd::keep_mask_words(expected_n));
    CHECK(bed_simd::build_keep_mask(keep_sub.data(), keep_sub.size(), expected_n, bed_keep.data()),
          "build_keep_mask failed");
    const uint64_t* km_ptr = is_sub ? bed_keep.data() : nullptr;

    const int pg_mask_w = PgenReader::GetSubsetMaskSize(expected_n);
    AlignedRawBuffer<uintptr_t, 64> pg_incl(pg_mask_w + 16), pg_inter(pg_mask_w + 24);
    std::memset(pg_incl.data(), 0, pg_incl.size() * sizeof(uintptr_t));
    std::memset(pg_inter.data(), 0, pg_inter.size() * sizeof(uintptr_t));
    if (is_sub) {
        PgenReader::SetSampleSubsets(keep_sub, expected_n, pg_incl.data(), pg_inter.data() + 8);
    }

    AlignedRawBuffer<uintptr_t, 64> pr_genovec(PgenReader::GetGenoBufPtrSize(expected_n) + 8u);
    std::vector<int8_t> pr_i8(keep_n + 64, 0), my_i8(keep_n + 64, 0);
    std::vector<float> pr_f32(keep_n + 64, 0.f), my_f32(keep_n + 64, 0.f);
    const float ftab4[4] = {-1.25f, 0.5f, 2.25f, -9.0f};
    alignas(64) const float qt1024[1024] = QUAD_TABLE256(-1.25f, 0.5f, 2.25f, -9.0f);

    gcta_io::PosixUniqueFd fd = gcta_io::open_posix_readonly_throwing(path);
    pgen_hardcall::CompressedBatchReader batch_reader;
    AlignedRawBuffer<uint8_t, 64> block_u8;
    AlignedRawBuffer<uintptr_t, 64> block_uptr;

    // Test 3 extraction schedules:
    //  0: full sequential [0 .. M) in blocks of 64
    //  1: sparse non-contiguous --extract (every 3rd + prime jumps, intentionally dropping LD anchors!)
    //  2: reverse / shuffled order within blocks (stressing deduplication & LD base lookup)
    std::vector<std::vector<uint32_t>> schedules(3);
    schedules[0].resize(expected_m);
    std::iota(schedules[0].begin(), schedules[0].end(), 0u);

    for (uint32_t v = 0; v < expected_m; ++v) {
        // Prefer selecting LD-compressed variants while skipping their anchors!
        if (idx.is_ld(v) || (v % 5u == 2u)) {
            schedules[1].push_back(v);
        }
    }
    if (schedules[1].empty()) schedules[1].push_back(0u);

    schedules[2] = schedules[1];
    std::reverse(schedules[2].begin(), schedules[2].end());

    uint64_t seq_id = 1;
    const uint32_t rb = idx.bed_rec_bytes;
    const uint8_t trailing_mask =
        (expected_n & 3u) ? static_cast<uint8_t>((1u << ((expected_n & 3u) * 2u)) - 1u) : 0xFFu;

    for (pgen_hardcall::Isa p_isa : pgen_isas) {
        CHECK(pgen_hardcall::set_isa(p_isa), "set_isa failed");
        for (std::size_t sched_idx = 0; sched_idx < schedules.size(); ++sched_idx) {
            const auto& vlist = schedules[sched_idx];
            pgen_hardcall::PgenThreadCache st_cache;
            std::vector<uint8_t> single_scratch(rb, 0xAAu);

            const std::size_t blk_sz = (sched_idx == 0) ? 53u : 19u;
            for (std::size_t b0 = 0; b0 < vlist.size(); b0 += blk_sz) {
                const std::size_t take = std::min<std::size_t>(blk_sz, vlist.size() - b0);
                std::span<const uint32_t> sub(vlist.data() + b0, take);

                const bool use_uptr = ((seq_id & 1ULL) != 0ULL);
                const void* blk_ptr = nullptr;
                if (use_uptr) {
                    const std::size_t bytes = batch_reader.read_batch_uptr(
                        fd.get(), idx, sub, seq_id, block_uptr);
                    CHECK(bytes >= sizeof(pgen_hardcall::PgenBlockHeader), "read_batch_uptr bytes");
                    blk_ptr = block_uptr.data();
                } else {
                    const std::size_t bytes = batch_reader.read_batch(
                        fd.get(), idx, sub, seq_id, block_u8);
                    CHECK(bytes >= sizeof(pgen_hardcall::PgenBlockHeader), "read_batch bytes");
                    blk_ptr = block_u8.data();
                }
                ++seq_id;

                // 1) OpenMP parallel slab decode_into (GRM / decodeBedI8KMajor pattern)
                AlignedRawBuffer<uint8_t, 64> omp_slab(take * rb);
                std::memset(omp_slab.data(), 0xCC, omp_slab.size());
                #pragma omp parallel num_threads(4)
                {
                    pgen_hardcall::PgenThreadCache tls_cache;
                    #pragma omp for schedule(static)
                    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(take); ++i) {
                        tls_cache.decode_into(
                            blk_ptr,
                            static_cast<uint32_t>(i),
                            omp_slab.data() + static_cast<std::size_t>(i) * rb);
                    }
                }

                // 2) Verify resolve(), single-scratch decode_into(), trailing zero bits, and bed_simd vs PgenReader
                for (std::size_t i = 0; i < take; ++i) {
                    const uint32_t v = sub[i];
                    const uint8_t* res_ptr = st_cache.resolve(blk_ptr, static_cast<uint32_t>(i));
                    if (idx.vrtype(v) == pgen_hardcall::kVrtypePlink1) {
                        const auto* base_b = static_cast<const uint8_t*>(blk_ptr);
                        CHECK(res_ptr == base_b + pgen_hardcall::block_entries(blk_ptr)[i].rec_off,
                              "expected zero-copy resolve() on kVrtypePlink1");
                    } else {
                        CHECK((res_ptr[rb - 1u] & ~trailing_mask) == 0u,
                              "resolve() left nonzero trailing bits at v=%u", v);
                    }

                    st_cache.decode_into(blk_ptr, static_cast<uint32_t>(i), single_scratch.data());
                    CHECK((single_scratch[rb - 1u] & ~trailing_mask) == 0u,
                          "decode_into(scratch) left nonzero trailing bits at v=%u", v);

                    const uint8_t* slab_rec = omp_slab.data() + i * rb;
                    CHECK((slab_rec[rb - 1u] & ~trailing_mask) == 0u,
                          "omp decode_into(slab) left nonzero trailing bits at v=%u", v);
                    CHECK(std::memcmp(single_scratch.data(), slab_rec, rb) == 0,
                          "scratch vs omp_slab mismatch at v=%u", v);

                    if (expected_p2_opt != nullptr) {
                        const std::vector<uint8_t> exp_bed = expected_bed2_from_p2((*expected_p2_opt)[v]);
                        CHECK(std::memcmp(slab_rec, exp_bed.data(), rb) == 0,
                              "decoded bed2 != expected_p2 at v=%u (vrtype=0x%x)", v, idx.vrtype(v));
                    }

                    // Cross-check subsetted + full counts/int8/float32 vs PgenReader on first schedule
                    if (sched_idx <= 1 && (i < 12 || (v % 17u == 0u))) {
                        std::memset(pr_genovec.data(), 0, pr_genovec.size() * sizeof(uintptr_t));
                        pr.ReadRawFullHard(pr_genovec.data(), static_cast<int>(v));
                        plink2::ZeroTrailingNyps(expected_n, pr_genovec.data());

                        std::array<uint32_t, 4> pg_gc{};
                        if (is_sub) {
                            plink2::GenoarrCountSubsetFreqs(
                                pr_genovec.data(), pg_inter.data() + 8, expected_n, keep_n, pg_gc);
                        } else {
                            plink2::GenoarrCountFreqsUnsafe(pr_genovec.data(), expected_n, pg_gc);
                        }
                        PgenReader::ExtractInt8Ext(
                            pr_genovec.data(), is_sub ? pg_incl.data() : nullptr,
                            expected_n, keep_n, pr_i8.data(), nullptr);
                        PgenReader::ExtractFloatExt(
                            pr_genovec.data(), is_sub ? pg_incl.data() : nullptr,
                            expected_n, keep_n, qt1024, pr_f32.data(), nullptr);

                        const bed_simd::Counts c = bed_simd::count(res_ptr, km_ptr, expected_n, keep_n);
                        CHECK(c.n0 == pg_gc[0] && c.n1 == pg_gc[1] &&
                              c.n2 == pg_gc[2] && c.nmiss == pg_gc[3],
                              "bed_simd::count mismatch vs PgenReader at v=%u", v);

                        const bed_simd::Counts ci = bed_simd::decode_i8(
                            res_ptr, km_ptr, expected_n, keep_n, my_i8.data(), nullptr);
                        CHECK(ci.n0 == pg_gc[0] && ci.n1 == pg_gc[1] &&
                              ci.n2 == pg_gc[2] && ci.nmiss == pg_gc[3],
                              "decode_i8 count mismatch at v=%u", v);
                        CHECK(std::memcmp(my_i8.data(), pr_i8.data(), keep_n) == 0,
                              "decode_i8 values mismatch vs PgenReader at v=%u", v);

                        bed_simd::decode_f32(
                            slab_rec, km_ptr, expected_n, keep_n, ftab4, my_f32.data(), nullptr);
                        CHECK(std::memcmp(my_f32.data(), pr_f32.data(), keep_n * sizeof(float)) == 0,
                              "decode_f32 values mismatch vs PgenReader at v=%u", v);
                    }
                }
            }
        }
    }
}

void test_posix_span_io(const std::string& tmpdir) {
    g_ctx = "posix_span_io";
    const std::string path = tmpdir + "/span_io_test.bin";
    const std::size_t file_len = 3u * (1u << 20); // 3 MiB
    std::vector<uint8_t> ref(file_len);
    for (std::size_t i = 0; i < file_len; ++i) {
        ref[i] = static_cast<uint8_t>((i * 1315423911ULL) ^ (i >> 9u));
    }
    write_bytes_to_file(path, ref.data(), ref.size());

    // RAII PosixUniqueFd move / release / reset
    {
        gcta_io::PosixUniqueFd fd1 = gcta_io::open_posix_readonly_throwing(path);
        CHECK(static_cast<bool>(fd1) && fd1.get() >= 0, "open_posix_readonly_throwing");
        CHECK(gcta_io::stat_file_size_throwing(fd1.get(), path) == file_len, "stat_file_size");

        gcta_io::PosixUniqueFd fd2(std::move(fd1));
        CHECK(!fd1 && fd1.get() == -1 && static_cast<bool>(fd2), "PosixUniqueFd move ctor");
        const int raw = fd2.release();
        CHECK(!fd2 && raw >= 0, "PosixUniqueFd release");
        fd1.reset(raw);
        CHECK(static_cast<bool>(fd1), "PosixUniqueFd reset");
    }

    gcta_io::PosixUniqueFd fd = gcta_io::open_posix_readonly_throwing(path);
    gcta_io::SpanBatchCoalescer coalescer;

    // 1) Contiguous + zero-length + small-gap bridged + large-gap flushed + non-ascending
    std::vector<uint8_t> dst(40000, 0);
    std::vector<gcta_io::FileByteSpan> spans = {
        {100, 500, dst.data()},
        {600, 0, dst.data() + 500},            // zero-length span in middle of contiguous run!
        {600, 700, dst.data() + 500},          // coalesces with [100..600) -> [100..1300)!
        {5000, 1200, dst.data() + 1200},       // small forward gap (3700 B <= 64 KiB) bridged!
        {2500000, 900, dst.data() + 2400},     // large gap (> 64 KiB) flushes!
        {50, 300, dst.data() + 3300},          // non-ascending (< 2500900) flushes!
    };
    coalescer.read_spans(fd.get(), path, spans, 65536ULL);
    CHECK(std::memcmp(dst.data(), ref.data() + 100, 1200) == 0, "coalesced span content");
    CHECK(std::memcmp(dst.data() + 1200, ref.data() + 5000, 1200) == 0, "bridged gap span content");
    CHECK(std::memcmp(dst.data() + 2400, ref.data() + 2500000, 900) == 0, "large gap span content");
    CHECK(std::memcmp(dst.data() + 3300, ref.data() + 50, 300) == 0, "non-ascending span content");

    // 2) > kMaxIov (1500 discontiguous destination buffers) with bridged gaps
    const std::size_t many_n = 1500;
    std::vector<std::array<uint8_t, 17>> scatter(many_n);
    std::vector<gcta_io::FileByteSpan> many_spans;
    many_spans.reserve(many_n);
    for (std::size_t i = 0; i < many_n; ++i) {
        many_spans.push_back({i * 31ULL, 17u, scatter[i].data()});
    }
    coalescer.read_spans(fd.get(), path, many_spans);
    bool all_scatter_ok = true;
    for (std::size_t i = 0; i < many_n; ++i) {
        if (std::memcmp(scatter[i].data(), ref.data() + i * 31ULL, 17u) != 0) {
            all_scatter_ok = false;
            break;
        }
    }
    CHECK(all_scatter_ok, ">kMaxIov scatter read_spans content mismatch");

    // 3) Read past EOF throws std::runtime_error
    uint8_t tmp[16];
    CHECK_THROWS(gcta_io::pread_full_throwing(fd.get(), tmp, 16u, file_len - 8u, path),
                 "pread_full_throwing past EOF");
    CHECK_THROWS(gcta_io::open_posix_readonly_throwing(tmpdir + "/nonexistent_xyz.pgen"),
                 "open_posix_readonly_throwing nonexistent");
}

void test_modes_0x01_0x02_0x03_0x04(const std::string& tmpdir,
                                    const std::vector<pgen_hardcall::Isa>& pgen_isas) {
    g_ctx = "modes_fixed_width";
    for (uint32_t raw_n : {1u, 2u, 3u, 4u, 5u, 63u, 64u, 65u, 257u}) {
        const uint32_t raw_m = 25u;
        const uint32_t rb = bed_simd::record_bytes(raw_n);
        const auto var_p2 = generate_rich_p2_matrix(raw_n, raw_m);

        // Mode 0x01 (.bed)
        {
            std::vector<uint8_t> bed_file(3u + static_cast<std::size_t>(raw_m) * rb, 0u);
            bed_file[0] = 0x6cu;
            bed_file[1] = 0x1bu;
            bed_file[2] = 0x01u;
            for (uint32_t v = 0; v < raw_m; ++v) {
                const auto exp_bed = expected_bed2_from_p2(var_p2[v]);
                std::memcpy(&bed_file[3u + static_cast<std::size_t>(v) * rb], exp_bed.data(), rb);
            }
            const std::string p01 = tmpdir + "/m01_" + std::to_string(raw_n) + ".bed";
            write_bytes_to_file(p01, bed_file.data(), bed_file.size());
            verify_parity_against_pgenreader(p01, raw_n, raw_m, &var_p2, pgen_isas);

            // Inferring variant count when expected_variant_ct == UINT32_MAX
            const auto idx_inf = pgen_hardcall::load_pgen_index(p01, raw_n, UINT32_MAX);
            CHECK(idx_inf.raw_variant_ct == raw_m && idx_inf.native_hardcall_ok, "mode 0x01 infer M");
        }

        // Mode 0x02 (fixed-width PLINK2, both with and without explicit nonref_flags!)
        for (bool has_nonref : {false, true}) {
            std::vector<uint8_t> p02;
            p02.push_back(0x6cu);
            p02.push_back(0x1bu);
            p02.push_back(0x02u);
            append_u32_le(p02, raw_m);
            append_u32_le(p02, raw_n);
            p02.push_back(has_nonref ? 0xC0u : 0x00u);
            if (has_nonref) {
                p02.insert(p02.end(), (raw_m + 7u) / 8u, 0x5Au);
            }
            AlignedRawBuffer<uintptr_t, 64> gw(plink2::NypCtToAlignedWordCt(raw_n));
            for (uint32_t v = 0; v < raw_m; ++v) {
                pack_p2_genovec(var_p2[v], gw.data());
                const auto* gb = reinterpret_cast<const uint8_t*>(gw.data());
                p02.insert(p02.end(), gb, gb + rb);
            }
            const std::string path02 =
                tmpdir + "/m02_" + std::to_string(raw_n) + "_" + (has_nonref ? "nr" : "plain") + ".pgen";
            write_bytes_to_file(path02, p02.data(), p02.size());
            verify_parity_against_pgenreader(path02, raw_n, raw_m, &var_p2, pgen_isas);
        }
    }

    // Mode 0x03 & 0x04 fallback detection
    for (uint8_t dmode : {uint8_t(0x03u), uint8_t(0x04u)}) {
        const uint32_t n = 10u, m = 4u;
        const uint32_t rec_w = bed_simd::record_bytes(n) + (dmode == 0x03u ? 2u * n : 4u * n);
        std::vector<uint8_t> buf;
        buf.push_back(0x6cu);
        buf.push_back(0x1bu);
        buf.push_back(dmode);
        append_u32_le(buf, m);
        append_u32_le(buf, n);
        buf.push_back(0u);
        buf.insert(buf.end(), static_cast<std::size_t>(m) * rec_w, 0u);
        const std::string p = tmpdir + "/m0" + std::to_string(dmode) + ".pgen";
        write_bytes_to_file(p, buf.data(), buf.size());
        const auto idx = pgen_hardcall::load_pgen_index(p, n, m);
        CHECK(!idx.native_hardcall_ok && !idx.fallback_reason.empty(),
              "mode 0x%02x must set native_hardcall_ok=false", dmode);
    }
}

void test_spgw_0x10_0x20_and_phased(const std::string& tmpdir,
                                    const std::vector<pgen_hardcall::Isa>& pgen_isas) {
    g_ctx = "spgw_0x10_0x20_phased";
    // Test across key sample sizes, especially 255/256/257 (BytesToRepresentNzU32 boundary!)
    // and SIMD vector boundaries (31, 63, 127, 513, 1025)
    for (uint32_t raw_n : {3u, 15u, 31u, 63u, 127u, 255u, 256u, 257u, 513u, 1025u}) {
        const uint32_t raw_m = 90u;
        const auto var_p2 = generate_rich_p2_matrix(raw_n, raw_m);

        // 1) Mode 0x10 inline header
        const std::string p10 = tmpdir + "/spgw10_" + std::to_string(raw_n) + ".pgen";
        write_spgw_biallelic(p10, raw_n, var_p2, plink2::kPgenWriteBackwardSeek, false);
        verify_parity_against_pgenreader(p10, raw_n, raw_m, &var_p2, pgen_isas);

        // 2) Mode 0x20 external .pgen.pgi index + phased hardcalls (0x10 bit set!)
        const std::string p20 = tmpdir + "/spgw20_phased_" + std::to_string(raw_n) + ".pgen";
        write_spgw_biallelic(p20, raw_n, var_p2, plink2::kPgenWriteSeparateIndex, true);
        const auto idx20 = pgen_hardcall::load_pgen_index(p20);
        CHECK(idx20.file_mode == 0x20u && idx20.native_hardcall_ok,
              "mode 0x20 phased should have native_hardcall_ok=true");
        verify_parity_against_pgenreader(p20, raw_n, raw_m, &var_p2, pgen_isas);
    }
}

void test_all_explicit_vrtypes_and_multigroup_difflists(
    const std::string& tmpdir,
    const std::vector<pgen_hardcall::Isa>& pgen_isas) {
    g_ctx = "explicit_vrtypes_all";
    // N = 1600 lets difflist_len reach up to 200 (> 3 groups of 64!), exercising
    // multi-group difflists, delta segment byte length table, and multi-byte VINT31 deltas.
    const uint32_t raw_n = 1600u;
    const uint32_t rb = bed_simd::record_bytes(raw_n);

    std::vector<uint8_t> vrtypes;
    std::vector<std::vector<uint8_t>> payloads;
    std::vector<std::vector<uint8_t>> expected_p2;

    // 1) All 6 valid common2_code combinations for vrtype 1 (1-bit + multi-group difflist!):
    //    (smaller, larger) = (0,1)->1, (0,2)->2, (0,3)->3, (1,2)->5, (1,3)->6, (2,3)->9
    const std::array<std::pair<uint8_t, uint8_t>, 6> common_pairs = {{
        {0u, 1u}, {0u, 2u}, {0u, 3u}, {1u, 2u}, {1u, 3u}, {2u, 3u}
    }};
    for (auto [p2_a, p2_b] : common_pairs) {
        const uint8_t common2_code = static_cast<uint8_t>((p2_a << 2u) | (p2_b - p2_a));
        std::vector<uint8_t> truth(raw_n, p2_a);
        std::vector<uint8_t> rec((raw_n + 15u) / 8u, 0u);
        rec[0] = common2_code;
        for (uint32_t s = 0; s < raw_n; ++s) {
            if (((s * 13u + p2_a) & 3u) != 0u) {
                truth[s] = p2_b;
                rec[1u + (s >> 3u)] |= static_cast<uint8_t>(1u << (s & 7u));
            }
        }
        // Inject 145 raregeno diffs (3 difflist groups: 64 + 64 + 17, including gap > 128 -> 2B VINT)
        std::vector<std::pair<uint32_t, uint8_t>> diffs;
        uint32_t pos = 3u;
        for (uint32_t d = 0; d < 145u && pos < raw_n; ++d) {
            const uint8_t g = static_cast<uint8_t>((d + p2_b) & 3u);
            truth[pos] = g;
            diffs.push_back({pos, g});
            pos += (d == 70u) ? 150u : 9u; // 150 > 127 tests 2-byte VINT31 delta!
        }
        const auto dl = encode_difflist(raw_n, diffs);
        rec.insert(rec.end(), dl.begin(), dl.end());

        vrtypes.push_back(1u);
        payloads.push_back(std::move(rec));
        expected_p2.push_back(std::move(truth));

        // Immediately attach an LD child (vrtype 2) and an inverted LD child (vrtype 3)
        // with 135 diffs each (> 2 groups of 64) anchored on this vrtype 1 variant!
        for (uint8_t ld_vt : {uint8_t(2u), uint8_t(3u)}) {
            const auto& base_truth = expected_p2[expected_p2.size() - (ld_vt == 2u ? 1u : 2u)];
            std::vector<uint8_t> pre_inv = base_truth;
            std::vector<std::pair<uint32_t, uint8_t>> ld_diffs;
            uint32_t lpos = 5u;
            for (uint32_t d = 0; d < 135u && lpos < raw_n; ++d) {
                const uint8_t ng = static_cast<uint8_t>((d * 3u + 1u) & 3u);
                pre_inv[lpos] = ng;
                ld_diffs.push_back({lpos, ng});
                lpos += 11u;
            }
            std::vector<uint8_t> final_truth = pre_inv;
            if (ld_vt == 3u) {
                for (uint8_t& g : final_truth) {
                    if (g == 0u) g = 2u;
                    else if (g == 2u) g = 0u;
                }
            }
            vrtypes.push_back(ld_vt);
            payloads.push_back(encode_difflist(raw_n, ld_diffs));
            expected_p2.push_back(std::move(final_truth));
        }
    }

    // 2) Explicit vrtype 0, 4, 5, 6, 7 (with 0-diff and 130-diff difflists)
    {
        std::vector<uint8_t> t0(raw_n);
        for (uint32_t s = 0; s < raw_n; ++s) t0[s] = static_cast<uint8_t>(s & 3u);
        AlignedRawBuffer<uintptr_t, 64> gw(plink2::NypCtToAlignedWordCt(raw_n));
        pack_p2_genovec(t0, gw.data());
        const auto* gb = reinterpret_cast<const uint8_t*>(gw.data());
        vrtypes.push_back(0u);
        payloads.emplace_back(gb, gb + rb);
        expected_p2.push_back(t0);
    }
    // vrtype 5 (0 bytes)
    vrtypes.push_back(5u);
    payloads.emplace_back();
    expected_p2.emplace_back(raw_n, 0u);

    for (uint8_t dvt : {uint8_t(4u), uint8_t(6u), uint8_t(7u)}) {
        const uint8_t modal = dvt & 3u;
        // Empty difflist
        vrtypes.push_back(dvt);
        payloads.push_back(encode_difflist(raw_n, {}));
        expected_p2.emplace_back(raw_n, modal);

        // Multi-group difflist (130 entries)
        std::vector<uint8_t> t(raw_n, modal);
        std::vector<std::pair<uint32_t, uint8_t>> diffs;
        for (uint32_t d = 0; d < 130u; ++d) {
            const uint32_t s = d * 12u + 2u;
            const uint8_t g = static_cast<uint8_t>((modal + 1u + (d % 3u)) & 3u);
            t[s] = g;
            diffs.push_back({s, g});
        }
        vrtypes.push_back(dvt);
        payloads.push_back(encode_difflist(raw_n, diffs));
        expected_p2.push_back(std::move(t));
    }

    // Test all combinations of 4-bit vs 8-bit vrtypes, 2/3/4-byte vrec_len, nonref_flags, mode 0x11!
    for (uint32_t len_b : {2u, 3u, 4u}) {
        const bool mode_11 = (len_b == 3u);
        const bool four_bit = (len_b == 2u);
        const auto bytes = build_custom_pgen_0x10(
            raw_n, vrtypes, payloads, len_b, !four_bit, nullptr, len_b == 4u, mode_11);
        const std::string p = tmpdir + "/custom_vt_lb" + std::to_string(len_b) + ".pgen";
        write_bytes_to_file(p, bytes.data(), bytes.size());
        verify_parity_against_pgenreader(
            p, raw_n, static_cast<uint32_t>(vrtypes.size()), &expected_p2, pgen_isas);
    }
}

void test_fused_header_ctrl_modes(const std::string& tmpdir,
                                  const std::vector<pgen_hardcall::Isa>& pgen_isas) {
    g_ctx = "fused_header_ctrl";
    // header_ctrl = 8: 1-bit fused (0 -> vrtype 5 len 0, 1 -> vrtype 0 len rb)
    {
        const uint32_t raw_n = 19u, raw_m = 21u;
        const uint32_t rb = bed_simd::record_bytes(raw_n);
        std::vector<std::vector<uint8_t>> exp_p2(raw_m, std::vector<uint8_t>(raw_n, 0u));
        std::vector<uint8_t> fused_bits((raw_m + 7u) / 8u, 0u);
        std::vector<uint8_t> payload;
        AlignedRawBuffer<uintptr_t, 64> gw(plink2::NypCtToAlignedWordCt(raw_n));

        for (uint32_t v = 0; v < raw_m; ++v) {
            if ((v % 3u) != 0u) {
                fused_bits[v >> 3u] |= static_cast<uint8_t>(1u << (v & 7u));
                for (uint32_t s = 0; s < raw_n; ++s) exp_p2[v][s] = static_cast<uint8_t>((s + v) & 3u);
                pack_p2_genovec(exp_p2[v], gw.data());
                const auto* gb = reinterpret_cast<const uint8_t*>(gw.data());
                payload.insert(payload.end(), gb, gb + rb);
            }
        }
        std::vector<uint8_t> file = {0x6cu, 0x1bu, 0x10u};
        append_u32_le(file, raw_m);
        append_u32_le(file, raw_n);
        file.push_back(8u); // header_ctrl = 8
        append_u64_le(file, 20ULL + fused_bits.size());
        file.insert(file.end(), fused_bits.begin(), fused_bits.end());
        file.insert(file.end(), payload.begin(), payload.end());

        const std::string p = tmpdir + "/fused_hc8.pgen";
        write_bytes_to_file(p, file.data(), file.size());
        verify_parity_against_pgenreader(p, raw_n, raw_m, &exp_p2, pgen_isas);
    }

    // header_ctrl = 10: 2-bit fused phased (0 -> vt 5, 1 -> vt 0, 2 -> vt 16 len rb+1, 3 -> vt 16 len rb+2)
    {
        const uint32_t raw_n = 5u, raw_m = 8u;
        const uint32_t rb = bed_simd::record_bytes(raw_n);
        std::vector<std::vector<uint8_t>> exp_p2(raw_m, std::vector<uint8_t>(raw_n, 0u));
        std::vector<uint8_t> fused_2b((raw_m + 3u) / 4u, 0u);
        std::vector<uint8_t> payload;
        AlignedRawBuffer<uintptr_t, 64> gw(plink2::NypCtToAlignedWordCt(raw_n));

        for (uint32_t v = 0; v < raw_m; ++v) {
            const uint32_t code = v & 3u;
            fused_2b[v >> 2u] |= static_cast<uint8_t>(code << ((v & 3u) * 2u));
            if (code >= 1u) {
                for (uint32_t s = 0; s < raw_n; ++s) exp_p2[v][s] = (s == 0u) ? 1u : 0u;
                pack_p2_genovec(exp_p2[v], gw.data());
                const auto* gb = reinterpret_cast<const uint8_t*>(gw.data());
                payload.insert(payload.end(), gb, gb + rb);
                if (code == 2u) payload.push_back(0x00u); // 1 hphase byte (all hets phased, unswapped)
                if (code == 3u) { payload.push_back(0x01u); payload.push_back(0x00u); }
            }
        }
        std::vector<uint8_t> file = {0x6cu, 0x1bu, 0x10u};
        append_u32_le(file, raw_m);
        append_u32_le(file, raw_n);
        file.push_back(10u); // header_ctrl = 10
        append_u64_le(file, 20ULL + fused_2b.size());
        file.insert(file.end(), fused_2b.begin(), fused_2b.end());
        file.insert(file.end(), payload.begin(), payload.end());

        const std::string p = tmpdir + "/fused_hc10.pgen";
        write_bytes_to_file(p, file.data(), file.size());
        const auto idx = pgen_hardcall::load_pgen_index(p, raw_n, raw_m);
        CHECK(idx.native_hardcall_ok, "fused header_ctrl=10 phased must be native_hardcall_ok");
        verify_parity_against_pgenreader(p, raw_n, raw_m, &exp_p2, pgen_isas);
    }

    // header_ctrl = 9, 11 (multiallelic) and 12, 14 (single-sample dosage) -> native_hardcall_ok == false
    for (uint8_t hc : {uint8_t(9u), uint8_t(11u), uint8_t(12u), uint8_t(14u)}) {
        const uint32_t raw_n = 1u, raw_m = 4u;
        const bool is_4bit = (hc == 11u || hc == 14u);
        std::vector<uint8_t> fused(is_4bit ? 2u : 1u, 0u);
        std::vector<uint8_t> payload;
        if (hc == 9u) {
            fused[0] = (2u << 0u); // entry 0 = code 2 (vt 8, len 1+2=3), entries 1..3 = 0 (vt 5)
            payload.resize(3u, 0u);
        } else if (hc == 11u) {
            fused[0] = 0x02u; // entry 0 = code 2 (vt 8, len 1+2=3)
            payload.resize(3u, 0u);
        } else if (hc == 12u) {
            fused[0] = (2u << 0u); // entry 0 = code 2 (vt 0x45, len 2)
            payload.resize(2u, 0u);
        } else {
            fused[0] = 0x04u; // entry 0 = code 4 (vt 0x45, len 2)
            payload.resize(2u, 0u);
        }
        uint8_t full_hc = hc;
        std::vector<uint8_t> extra_ac;
        if (hc == 9u || hc == 11u) {
            full_hc |= (1u << 4u);
            extra_ac = {3u, 2u, 2u, 2u};
        }
        std::vector<uint8_t> file = {0x6cu, 0x1bu, 0x10u};
        append_u32_le(file, raw_m);
        append_u32_le(file, raw_n);
        file.push_back(full_hc);
        append_u64_le(file, 20ULL + fused.size() + extra_ac.size());
        file.insert(file.end(), fused.begin(), fused.end());
        file.insert(file.end(), extra_ac.begin(), extra_ac.end());
        file.insert(file.end(), payload.begin(), payload.end());

        const std::string p = tmpdir + "/fused_hc" + std::to_string(hc) + ".pgen";
        write_bytes_to_file(p, file.data(), file.size());
        const auto idx = pgen_hardcall::load_pgen_index(p, raw_n, raw_m);
        CHECK(!idx.native_hardcall_ok, "fused hc=%u should fall back", hc);
    }
}

void test_multi_vblock_over_65536(const std::string& tmpdir) {
    g_ctx = "multi_vblock_70000";
    // M = 70000 spans two 65536-variant vblocks!
    const uint32_t raw_n = 73u;
    const uint32_t raw_m = 70000u;
    std::vector<std::vector<uint8_t>> var_p2(raw_m, std::vector<uint8_t>(raw_n, 0u));

    std::vector<uint8_t> anchor(raw_n, 0u);
    for (uint32_t v = 0; v < raw_m; ++v) {
        if ((v % 65536u) == 0u || (v % 8u) == 0u) {
            for (uint32_t s = 0; s < raw_n; ++s) {
                anchor[s] = static_cast<uint8_t>((s * 3u + v) & 3u);
            }
            var_p2[v] = anchor;
        } else if ((v % 8u) <= 3u) {
            var_p2[v] = anchor;
            var_p2[v][v % raw_n] = static_cast<uint8_t>(v & 3u);
        } else if ((v % 8u) == 4u) {
            for (uint32_t s = 0; s < raw_n; ++s) {
                const uint8_t g = anchor[s];
                var_p2[v][s] = (g == 0u) ? 2u : (g == 2u ? 0u : g);
            }
        } else if ((v % 8u) == 5u) {
            std::fill(var_p2[v].begin(), var_p2[v].end(), 0u);
            var_p2[v][(v * 5u) % raw_n] = 1u;
        } else {
            std::fill(var_p2[v].begin(), var_p2[v].end(), 0u);
        }
    }

    const std::string path = tmpdir + "/multi_vblock_70k.pgen";
    write_spgw_biallelic(path, raw_n, var_p2, plink2::kPgenWriteBackwardSeek, false);

    const auto idx = pgen_hardcall::load_pgen_index(path, raw_n, raw_m);
    CHECK(idx.native_hardcall_ok && idx.raw_variant_ct == raw_m, "70k index load");
    CHECK(!idx.is_ld(0u) && !idx.is_ld(65536u), "vblock starts are non-LD");

    // Probe across the 65536 vblock boundary with dropped LD anchors!
    std::vector<uint32_t> probe;
    for (uint32_t v = 65520u; v < 65560u; ++v) {
        if (idx.is_ld(v) || v == 65535u || v == 65536u) probe.push_back(v);
    }
    probe.push_back(0u);
    probe.push_back(69999u);

    gcta_io::PosixUniqueFd fd = gcta_io::open_posix_readonly_throwing(path);
    pgen_hardcall::CompressedBatchReader reader;
    AlignedRawBuffer<uint8_t, 64> blk;
    reader.read_batch(fd.get(), idx, probe, 42ULL, blk);

    pgen_hardcall::PgenThreadCache cache;
    const uint32_t rb = idx.bed_rec_bytes;
    for (std::size_t i = 0; i < probe.size(); ++i) {
        const uint32_t v = probe[i];
        const uint8_t* got = cache.resolve(blk.data(), static_cast<uint32_t>(i));
        const auto exp = expected_bed2_from_p2(var_p2[v]);
        CHECK(std::memcmp(got, exp.data(), rb) == 0,
              "multi-vblock boundary mismatch at v=%u (vrtype=%u)", v, idx.vrtype(v));
    }
}

void test_fallback_detection_and_env(const std::string& tmpdir) {
    g_ctx = "fallback_and_env";
    const uint32_t raw_n = 32u, raw_m = 6u;

    // 1) Variable-width dosage .pgen via STPgenWriter
    {
        const std::string p_dos = tmpdir + "/dosage_var.pgen";
        plink2::STPgenWriter spgw;
        plink2::PreinitSpgw(&spgw);
        uintptr_t alloc_cl = 0;
        uint32_t max_vrec = 0;
        plink2::PglErr err = plink2::SpgwInitPhase1(
            p_dos.c_str(), nullptr, nullptr, raw_m, raw_n, 0,
            plink2::kPgenWriteBackwardSeek, plink2::kfPgenGlobalDosagePresent,
            0, &spgw, &alloc_cl, &max_vrec);
        CHECK(err == plink2::kPglRetSuccess, "dosage SpgwInitPhase1");
        AlignedRawBuffer<uint8_t, 64> mem(alloc_cl * plink2::kCacheline + 128u);
        std::memset(mem.data(), 0, mem.size());
        plink2::SpgwInitPhase2(max_vrec, &spgw, mem.data());

        AlignedRawBuffer<uintptr_t, 64> gw(plink2::NypCtToAlignedWordCt(raw_n));
        AlignedRawBuffer<uintptr_t, 64> dp(plink2::BitCtToWordCt(raw_n) + 2u);
        std::memset(gw.data(), 0, gw.size() * sizeof(uintptr_t));
        std::memset(dp.data(), 0, dp.size() * sizeof(uintptr_t));
        plink2::SetBit(3u, dp.data());
        const uint16_t dval = 12345u;

        for (uint32_t v = 0; v < raw_m; ++v) {
            if (v == 2u) {
                err = plink2::SpgwAppendBiallelicGenovecDosage16(gw.data(), dp.data(), &dval, 1u, &spgw);
            } else {
                err = plink2::SpgwAppendBiallelicGenovec(gw.data(), &spgw);
            }
            CHECK(err == plink2::kPglRetSuccess, "dosage SpgwAppend");
        }
        err = plink2::SpgwFinish(&spgw);
        CHECK(err == plink2::kPglRetSuccess, "dosage SpgwFinish");
        plink2::CleanupSpgw(&spgw, &err);

        const auto idx = pgen_hardcall::load_pgen_index(p_dos, raw_n, raw_m);
        CHECK(!idx.native_hardcall_ok && idx.fallback_reason == "dosage_present",
              "expected dosage_present fallback (got ok=%d reason=%s)",
              idx.native_hardcall_ok, idx.fallback_reason.c_str());
    }

    // 2) Multiallelic .pgen via allele_ct == 3 in header (even when patch_01_ct == patch_10_ct == 0!)
    {
        std::vector<uint8_t> vts = {5u, 5u, 5u};
        std::vector<std::vector<uint8_t>> pls(3);
        std::vector<uint8_t> allele_cts = {2u, 3u, 2u};
        const auto bytes = build_custom_pgen_0x10(raw_n, vts, pls, 1u, false, &allele_cts, false, false);
        const std::string p_ma = tmpdir + "/multiallelic_hdr.pgen";
        write_bytes_to_file(p_ma, bytes.data(), bytes.size());
        const auto idx = pgen_hardcall::load_pgen_index(p_ma, raw_n, 3u);
        CHECK(!idx.native_hardcall_ok && idx.fallback_reason == "multiallelic_present",
              "expected multiallelic_present fallback (got ok=%d reason=%s)",
              idx.native_hardcall_ok, idx.fallback_reason.c_str());
    }

    // 3) GCTA_DISABLE_NATIVE_PGEN env override
    {
        std::vector<uint8_t> vts = {5u, 5u};
        std::vector<std::vector<uint8_t>> pls(2);
        const auto bytes = build_custom_pgen_0x10(raw_n, vts, pls, 1u, false);
        const std::string p_env = tmpdir + "/env_test.pgen";
        write_bytes_to_file(p_env, bytes.data(), bytes.size());

        ::setenv("GCTA_DISABLE_NATIVE_PGEN", "1", 1);
        const auto idx_dis = pgen_hardcall::load_pgen_index(p_env, raw_n, 2u);
        CHECK(!idx_dis.native_hardcall_ok && idx_dis.fallback_reason == "disabled_by_env",
              "GCTA_DISABLE_NATIVE_PGEN=1 must disable native hardcall");

        ::setenv("GCTA_DISABLE_NATIVE_PGEN", "0", 1);
        const auto idx_en = pgen_hardcall::load_pgen_index(p_env, raw_n, 2u);
        CHECK(idx_en.native_hardcall_ok && idx_en.fallback_reason.empty(),
              "GCTA_DISABLE_NATIVE_PGEN=0 must keep native hardcall enabled");
        ::unsetenv("GCTA_DISABLE_NATIVE_PGEN");
    }
}

void test_truncated_and_malformed_errors(const std::string& tmpdir) {
    g_ctx = "malformed_and_truncated";
    // 1) Header-level malformed inputs
    {
        const std::string p_bad_magic = tmpdir + "/bad_magic.pgen";
        const uint8_t bad_m[16] = {0x12, 0x34, 0x10};
        write_bytes_to_file(p_bad_magic, bad_m, sizeof(bad_m));
        CHECK_THROWS(pgen_hardcall::load_pgen_index(p_bad_magic), "bad magic");

        const std::string p_smaj = tmpdir + "/sample_major.bed";
        const uint8_t smaj[8] = {0x6c, 0x1b, 0x00, 0, 0, 0, 0, 0};
        write_bytes_to_file(p_smaj, smaj, sizeof(smaj));
        CHECK_THROWS(pgen_hardcall::load_pgen_index(p_smaj, 4u, 4u), "sample-major bed");

        const std::string p_pgi = tmpdir + "/as_pgen.pgi";
        const uint8_t pgi_m[16] = {0x6c, 0x1b, 0x30};
        write_bytes_to_file(p_pgi, pgi_m, sizeof(pgi_m));
        CHECK_THROWS(pgen_hardcall::load_pgen_index(p_pgi), ".pgi passed as .pgen");

        // LD-compressed variant at start of vblock (v = 0)
        std::vector<uint8_t> vts = {2u};
        std::vector<std::vector<uint8_t>> pls = {{0u}};
        const auto ld0_bytes = build_custom_pgen_0x10(16u, vts, pls);
        const std::string p_ld0 = tmpdir + "/ld_at_v0.pgen";
        write_bytes_to_file(p_ld0, ld0_bytes.data(), ld0_bytes.size());
        CHECK_THROWS(pgen_hardcall::load_pgen_index(p_ld0), "LD at vblock start");

        // Truncated mode 0x10 file
        std::vector<uint8_t> ok_vts = {0u};
        std::vector<std::vector<uint8_t>> ok_pls = {{0, 0, 0, 0}};
        auto trunc_bytes = build_custom_pgen_0x10(16u, ok_vts, ok_pls);
        trunc_bytes.pop_back(); // 1 byte short!
        const std::string p_trunc = tmpdir + "/trunc_payload.pgen";
        write_bytes_to_file(p_trunc, trunc_bytes.data(), trunc_bytes.size());
        CHECK_THROWS(pgen_hardcall::load_pgen_index(p_trunc), "truncated payload");

        // Expected sample/variant count mismatch
        auto valid_bytes = build_custom_pgen_0x10(16u, ok_vts, ok_pls);
        const std::string p_valid = tmpdir + "/valid_16_1.pgen";
        write_bytes_to_file(p_valid, valid_bytes.data(), valid_bytes.size());
        CHECK_THROWS(pgen_hardcall::load_pgen_index(p_valid, 15u, 1u), "sample ct mismatch");
        CHECK_THROWS(pgen_hardcall::load_pgen_index(p_valid, 16u, 2u), "variant ct mismatch");
    }

    // 2) Record-decoder bounds checks (ASan-guarded exact buffers!)
    const uint32_t raw_n = 64u;
    std::vector<uint8_t> dst(bed_simd::record_bytes(raw_n), 0u);
    std::vector<uint8_t> base(bed_simd::record_bytes(raw_n), 0xFFu);

    // Truncated vrtype 0
    {
        std::vector<uint8_t> short_v0(15u, 0u); // needs 16 bytes for N=64
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(
                         short_v0.data(), short_v0.size(), 0u, raw_n, dst.data()),
                     "truncated vrtype 0");
    }
    // Truncated vrtype 1 bitarray & invalid common2_code
    {
        std::vector<uint8_t> short_v1(8u, 1u); // needs (64+15)/8 = 9 bytes before difflist
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(
                         short_v1.data(), short_v1.size(), 1u, raw_n, dst.data()),
                     "truncated vrtype 1 bitarray");

        std::vector<uint8_t> bad_c2(10u, 0u);
        bad_c2[0] = 0x0Fu; // base=3, delta=3 -> base+delta=6 > 3!
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(
                         bad_c2.data(), bad_c2.size(), 1u, raw_n, dst.data()),
                     "invalid vrtype 1 common2_code");
    }
    // Non-empty unphased vrtype 5
    {
        const uint8_t junk = 0u;
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(&junk, 1u, 5u, raw_n, dst.data()),
                     "non-empty unphased vrtype 5");
    }
    // Truncated difflist VINT31 & overflow 5-byte VINT31
    {
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(nullptr, 0u, 4u, raw_n, dst.data()),
                     "empty difflist record");
        const uint8_t cont_vint[2] = {0x80u, 0x80u};
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(cont_vint, 2u, 4u, raw_n, dst.data()),
                     "truncated multi-byte VINT31");
        const uint8_t ovf_vint[5] = {0x80u, 0x80u, 0x80u, 0x80u, 0x08u}; // >= 2^31!
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(ovf_vint, 5u, 4u, raw_n, dst.data()),
                     "overflow 5-byte VINT31");
    }
    // difflist_len > raw_sample_ct / 8
    {
        const uint8_t too_many[1] = {9u}; // 9 > 64 / 8 = 8
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(too_many, 1u, 4u, raw_n, dst.data()),
                     "difflist_len > N/8");
    }
    // Truncated difflist payload & out-of-range sample ID & zero delta
    {
        auto valid_dl = encode_difflist(raw_n, {{2u, 1u}, {10u, 2u}});
        for (std::size_t cut = 1u; cut < valid_dl.size(); ++cut) {
            // Allocate exact-sized buffer so ASan immediately catches any 1-byte overread!
            std::vector<uint8_t> exact(valid_dl.begin(), valid_dl.begin() + cut);
            CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(
                             exact.data(), exact.size(), 4u, raw_n, dst.data()),
                         "truncated difflist at %zu/%zu bytes", cut, valid_dl.size());
            CHECK_THROWS(pgen_hardcall::decode_ld_to_bed2(
                             exact.data(), exact.size(), 2u, raw_n, base.data(), dst.data()),
                         "truncated LD difflist at %zu/%zu bytes", cut, valid_dl.size());
        }

        // Out-of-range sample index (sample_idx == raw_n)
        std::vector<uint8_t> oor_dl = {1u, static_cast<uint8_t>(raw_n), 0x01u};
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(
                         oor_dl.data(), oor_dl.size(), 4u, raw_n, dst.data()),
                     "difflist group start out of range");

        // Zero delta inside group
        std::vector<uint8_t> zero_delta = {2u, 5u, 0x05u, 0u};
        CHECK_THROWS(pgen_hardcall::decode_non_ld_to_bed2(
                         zero_delta.data(), zero_delta.size(), 4u, raw_n, dst.data()),
                     "difflist zero delta");
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string tmpdir;
    if (argc > 1) {
        tmpdir = std::string(argv[1]) + "/gcta_test_pgen_hc_" + std::to_string(::getpid());
        fs::create_directories(tmpdir);
    } else {
        std::error_code ec;
        const char* env_tmp = std::getenv("TMPDIR");
        std::string cand = (env_tmp && env_tmp[0]) ? std::string(env_tmp) : std::string("/tmp");
        if (!cand.empty() && cand.back() == '/') cand.pop_back();
        tmpdir = cand + "/gcta_test_pgen_hc_" + std::to_string(::getpid());
        fs::create_directories(tmpdir, ec);
        if (ec) {
            ec.clear();
            tmpdir = "./gcta_test_pgen_hc_" + std::to_string(::getpid());
            fs::create_directories(tmpdir, ec);
        }
    }

    std::vector<pgen_hardcall::Isa> pgen_isas = {pgen_hardcall::Isa::Scalar};
    for (pgen_hardcall::Isa cand : {pgen_hardcall::Isa::Neon,
                                    pgen_hardcall::Isa::Avx2,
                                    pgen_hardcall::Isa::Avx512}) {
        if (pgen_hardcall::set_isa(cand)) {
            pgen_isas.push_back(cand);
        }
    }
    pgen_hardcall::set_isa(pgen_hardcall::best_isa());

    std::printf("Running test_pgen_hardcall (tmpdir=%s, best_pgen_isa=%s, best_bed_isa=%s)...\n",
                tmpdir.c_str(), pgen_hardcall::isa_name(), bed_simd::isa_name());

    test_posix_span_io(tmpdir);
    test_modes_0x01_0x02_0x03_0x04(tmpdir, pgen_isas);
    test_spgw_0x10_0x20_and_phased(tmpdir, pgen_isas);
    test_all_explicit_vrtypes_and_multigroup_difflists(tmpdir, pgen_isas);
    test_fused_header_ctrl_modes(tmpdir, pgen_isas);
    test_multi_vblock_over_65536(tmpdir);
    test_fallback_detection_and_env(tmpdir);
    test_truncated_and_malformed_errors(tmpdir);

    std::error_code ec;
    fs::remove_all(tmpdir, ec);

    std::printf("test_pgen_hardcall: %llu checks, %llu failures.\n",
                static_cast<unsigned long long>(g_checks),
                static_cast<unsigned long long>(g_fail));
    return (g_fail == 0) ? 0 : 1;
}
