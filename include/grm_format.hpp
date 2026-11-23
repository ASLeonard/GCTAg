#pragma once

#include "Logger.h"
#include "async_seq_io.hpp"

#include <sys/stat.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace gcta_grm_io {

// ---------------------------------------------------------------------------
// Native FP16 GRM on-disk format (<prefix>.grm.f16, v1, little-endian)
//
// Layout:
//   [0 .. 127]                              128-byte GrmFp16Header
//   [128 .. 128 + 4*n)                      exact float32 diag[n]
//   [128 + 4*n .. body_offset)              zero-padding to 4 KiB page boundary
//   [body_offset .. body_offset + tri*2)    packed lower-triangular IEEE fp16
//                                           (row i at element i*(i+1)/2)
//
// Lifecycle: writers initialize magic to "GCTAG16?" (draft), stream the packed
// fp16 body sequentially, then commit exact diag[n] + finalized header
// ("GCTAG16\0" + kGrmF16FlagFinalized) via a single pwrite() at offset 0.
// ---------------------------------------------------------------------------

inline constexpr char kGrmF16MagicFinal[8] = {'G', 'C', 'T', 'A', 'G', '1', '6', '\0'};
inline constexpr char kGrmF16MagicDraft[8] = {'G', 'C', 'T', 'A', 'G', '1', '6', '?'};

inline constexpr uint32_t kGrmF16Version    = 1u;
inline constexpr uint32_t kGrmF16DtypeFp16  = 1u;
inline constexpr uint32_t kGrmF16HeaderSize = 128u;

inline constexpr uint32_t kGrmF16FlagZeroMissing = 1u << 0; // 0x01: no per-pair missingness (no .grm.N.bin needed)
inline constexpr uint32_t kGrmF16FlagHasTraces   = 1u << 1; // 0x02: trace_K and trace_K2 are populated
inline constexpr uint32_t kGrmF16FlagHasDiagF32  = 1u << 2; // 0x04: exact float32 diag[n] present at diag_offset
inline constexpr uint32_t kGrmF16FlagVanRaden    = 1u << 3; // 0x08: built with VanRaden (--alg 1)
inline constexpr uint32_t kGrmF16FlagDominance   = 1u << 4; // 0x10: dominance GRM
inline constexpr uint32_t kGrmF16FlagHomogametic = 1u << 5; // 0x20: homogametic sex-chromosome GRM
inline constexpr uint32_t kGrmF16FlagFinalized   = 1u << 7; // 0x80: cleanly finalized

struct GrmFp16Header {
    char     magic[8]{'G', 'C', 'T', 'A', 'G', '1', '6', '?'};
    uint32_t version        = kGrmF16Version;
    uint32_t flags          = 0;
    uint32_t dtype          = kGrmF16DtypeFp16;
    uint32_t header_bytes   = kGrmF16HeaderSize;
    uint64_t n              = 0;
    uint64_t valid_markers  = 0;
    double   m_snps_mean    = 0.0;
    uint64_t id_hash        = 0;
    double   trace_K        = 0.0;
    double   trace_K2       = 0.0;
    uint64_t diag_offset    = kGrmF16HeaderSize;
    uint64_t body_offset    = 0;
    uint64_t part_row_start = 0;
    uint64_t part_row_end   = 0;
    double   mtd_weight     = 1.0;
    uint8_t  reserved[16]{};
};

static_assert(sizeof(GrmFp16Header) == 128, "GrmFp16Header must be exactly 128 bytes");
static_assert(std::is_trivially_copyable_v<GrmFp16Header>, "GrmFp16Header must be trivially copyable");

inline constexpr uint64_t grm_f16_body_offset(uint64_t n) noexcept {
    const uint64_t raw = static_cast<uint64_t>(kGrmF16HeaderSize) + n * sizeof(float);
    return (raw + 4095ULL) & ~4095ULL;
}

inline constexpr uint64_t grm_f16_packed_bytes(uint64_t n) noexcept {
    return (n * (n + 1ULL) / 2ULL) * sizeof(uint16_t);
}

enum class GrmDiskFormat : uint8_t {
    F32Bin = 0,
    F16    = 1,
};

struct ResolvedGrmInput {
    GrmDiskFormat format = GrmDiskFormat::F32Bin;
    std::string data_path;
    std::string id_path;
    std::string n_path;
    bool has_n_file = false;
};

namespace detail {
inline bool is_regular_file(const std::string& path) noexcept {
    struct stat st{};
    return (::stat(path.c_str(), &st) == 0) && S_ISREG(st.st_mode);
}
} // namespace detail

// Transparent GRM input format resolver: detects <prefix>.grm.f16 vs
// <prefix>.grm.bin. Refuses ambiguity when both exist unless prefer_fp16 is set.
inline ResolvedGrmInput resolve_grm_input(const std::string& prefix, bool prefer_fp16 = false) {
    const std::string f16_path = prefix + ".grm.f16";
    const std::string bin_path = prefix + ".grm.bin";
    const std::string id_path  = prefix + ".grm.id";
    const std::string n_path   = prefix + ".grm.N.bin";

    const bool has_f16 = detail::is_regular_file(f16_path);
    const bool has_bin = detail::is_regular_file(bin_path);
    const bool has_n   = detail::is_regular_file(n_path);

    ResolvedGrmInput out;
    out.id_path    = id_path;
    out.n_path     = n_path;
    out.has_n_file = has_n;

    if (has_f16 && has_bin) {
        if (prefer_fp16) {
            out.format    = GrmDiskFormat::F16;
            out.data_path = f16_path;
            return out;
        }
        LOGGER.e(0, "Ambiguous GRM prefix [" + prefix + "]: both [" + f16_path + "] and [" +
                    bin_path + "] exist. Pass --grm-fp16 (or --fast) to select .grm.f16, "
                    "or remove the stale file.");
    }
    if (has_f16) {
        out.format    = GrmDiskFormat::F16;
        out.data_path = f16_path;
        return out;
    }
    if (has_bin) {
        out.format    = GrmDiskFormat::F32Bin;
        out.data_path = bin_path;
        return out;
    }

    LOGGER.e(0, "Cannot find GRM binary file for prefix [" + prefix + "]: neither [" +
                f16_path + "] nor [" + bin_path + "] exists.");
    return out;
}

// Read and validate a 128-byte GrmFp16Header from an open fd (positioned
// anywhere; reads at offset 0 via pread). Calls LOGGER.e on any corruption,
// draft/incomplete file, unsupported version/dtype, or truncated size.
inline void read_and_validate_grm_f16_header(int fd, const std::string& path, GrmFp16Header& hdr) {
    try {
        gcta_io::read_exact_at_throwing(fd, &hdr, sizeof(GrmFp16Header), 0, path);
    } catch (const std::exception& e) {
        LOGGER.e(0, std::string("Failed to read .grm.f16 header from [") + path + "]: " + e.what());
    }

    if (std::memcmp(hdr.magic, kGrmF16MagicDraft, sizeof(hdr.magic)) == 0) {
        LOGGER.e(0, "[" + path + "] is an incomplete/unfinalized .grm.f16 file (draft magic 'GCTAG16?', "
                    "likely from an interrupted or OOM job). Re-create it with 'GCTAg grm make'.");
    }
    if (std::memcmp(hdr.magic, kGrmF16MagicFinal, sizeof(hdr.magic)) != 0) {
        LOGGER.e(0, "[" + path + "] is not a valid .grm.f16 file (unrecognized magic).");
    }
    if ((hdr.flags & kGrmF16FlagFinalized) == 0) {
        LOGGER.e(0, "[" + path + "] is missing the finalized flag; file may be incomplete.");
    }
    if (hdr.version != kGrmF16Version) {
        LOGGER.e(0, "[" + path + "] has unsupported .grm.f16 version " +
                    std::to_string(hdr.version) + " (expected " + std::to_string(kGrmF16Version) + ").");
    }
    if (hdr.dtype != kGrmF16DtypeFp16) {
        LOGGER.e(0, "[" + path + "] has unsupported .grm.f16 dtype " +
                    std::to_string(hdr.dtype) + " (expected " + std::to_string(kGrmF16DtypeFp16) + ").");
    }
    if (hdr.header_bytes != kGrmF16HeaderSize) {
        LOGGER.e(0, "[" + path + "] has unexpected header_bytes=" +
                    std::to_string(hdr.header_bytes) + " (expected " + std::to_string(kGrmF16HeaderSize) + ").");
    }
    if (hdr.n == 0 || hdr.n > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        LOGGER.e(0, "[" + path + "] has invalid sample count n=" + std::to_string(hdr.n) + ".");
    }
    if (hdr.diag_offset != kGrmF16HeaderSize) {
        LOGGER.e(0, "[" + path + "] has unexpected diag_offset=" + std::to_string(hdr.diag_offset) + ".");
    }
    const uint64_t min_body_off = grm_f16_body_offset(hdr.n);
    if (hdr.body_offset < min_body_off) {
        LOGGER.e(0, "[" + path + "] has invalid body_offset=" + std::to_string(hdr.body_offset) +
                    " (minimum " + std::to_string(min_body_off) + " for n=" + std::to_string(hdr.n) + ").");
    }

    const uint64_t packed_bytes = grm_f16_packed_bytes(hdr.n);
    if (hdr.body_offset > std::numeric_limits<uint64_t>::max() - packed_bytes) {
        LOGGER.e(0, "[" + path + "] size computation overflowed for n=" + std::to_string(hdr.n) + ".");
    }
    const uint64_t expected_min_size = hdr.body_offset + packed_bytes;

    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0 || static_cast<uint64_t>(st.st_size) < expected_min_size) {
        LOGGER.e(0, "unexpected size in [" + path + "]. Expected at least " +
                    std::to_string(expected_min_size) + " bytes for n=" + std::to_string(hdr.n) + ".");
    }
}

} // namespace gcta_grm_io
