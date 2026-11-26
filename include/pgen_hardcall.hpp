/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   pgen_hardcall.hpp — Clean-room native PLINK2 .pgen hardcall index, compressed
   batch span reader, SIMD record decoders (AVX-512 / AVX2 / NEON / Scalar)
   emitting PLINK1 .bed 2-bit records, and per-thread L2 decode cache.
*/

#ifndef GCTA_PGEN_HARDCALL_HPP
#define GCTA_PGEN_HARDCALL_HPP

#include "mem.hpp"
#include "posix_span_io.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pgen_hardcall {

inline constexpr uint16_t kVrtypePlink1 = 0x100u;
inline constexpr uint64_t kPgenBlockMagic = 0x5047454E4843424BULL;

enum class Isa : int { Scalar = 0, Neon = 1, Avx2 = 2, Avx512 = 3 };

const char* isa_name();
Isa active_isa();
Isa best_isa();
bool set_isa(Isa isa);

struct PgenFileIndex {
    std::string path;
    uint32_t raw_variant_ct = 0;
    uint32_t raw_sample_ct = 0;
    uint32_t bed_rec_bytes = 0;
    uint8_t file_mode = 0;
    bool native_hardcall_ok = false;
    std::string fallback_reason;
    bool is_const_width = false;
    uint64_t const_fpos_offset = 0;
    uint32_t const_vrec_width = 0;
    uint16_t const_vrtype = 0;
    uint64_t total_payload_bytes = 0;
    double avg_rec_bytes = 0.0;
    std::vector<uint64_t> var_fpos;
    std::vector<uint8_t> vrtypes;

    [[nodiscard]] uint64_t fpos(uint32_t vidx) const noexcept {
        return is_const_width
                   ? (const_fpos_offset + static_cast<uint64_t>(const_vrec_width) * static_cast<uint64_t>(vidx))
                   : var_fpos[vidx];
    }

    [[nodiscard]] uint32_t vrec_len(uint32_t vidx) const noexcept {
        return is_const_width
                   ? const_vrec_width
                   : static_cast<uint32_t>(var_fpos[vidx + 1] - var_fpos[vidx]);
    }

    [[nodiscard]] uint16_t vrtype(uint32_t vidx) const noexcept {
        return is_const_width ? const_vrtype : static_cast<uint16_t>(vrtypes[vidx]);
    }

    static constexpr bool is_ld_vrtype(uint16_t vt) noexcept {
        return vt != kVrtypePlink1 && ((vt & 6u) == 2u);
    }

    [[nodiscard]] bool is_ld(uint32_t vidx) const noexcept {
        return is_ld_vrtype(vrtype(vidx));
    }

    [[nodiscard]] uint32_t ldbase_vidx(uint32_t vidx) const noexcept {
        if (is_const_width || !is_ld_vrtype(vrtypes[vidx])) {
            return vidx;
        }
        uint32_t b = vidx - 1u;
        while (b > 0u && is_ld_vrtype(vrtypes[b])) {
            --b;
        }
        return b;
    }
};

PgenFileIndex load_pgen_index(const std::string& path,
                              uint32_t expected_sample_ct = UINT32_MAX,
                              uint32_t expected_variant_ct = UINT32_MAX);

struct alignas(64) PgenBlockHeader {
    uint64_t magic = kPgenBlockMagic;
    uint64_t block_seq_id = 0;
    uint32_t num_markers = 0;
    uint32_t raw_sample_ct = 0;
    uint32_t bed_rec_bytes = 0;
    uint32_t reserved = 0;
    uint64_t payload_bytes = 0;
    uint64_t pad[3] = {0, 0, 0};
};
static_assert(sizeof(PgenBlockHeader) == 64);

struct PgenBlockEntry {
    uint64_t rec_off = 0;
    uint32_t rec_len = 0;
    uint32_t raw_vidx = 0;
    uint64_t base_off = 0;
    uint32_t base_len = 0;
    uint16_t vrtype = 0;
    uint16_t base_vrtype = 0;
};
static_assert(sizeof(PgenBlockEntry) == 32);

inline const PgenBlockHeader* block_header(const void* buf) noexcept {
    return static_cast<const PgenBlockHeader*>(buf);
}

inline const PgenBlockEntry* block_entries(const void* buf) noexcept {
    return reinterpret_cast<const PgenBlockEntry*>(
        static_cast<const uint8_t*>(buf) + sizeof(PgenBlockHeader));
}

class CompressedBatchReader {
public:
    CompressedBatchReader() = default;
    CompressedBatchReader(const CompressedBatchReader&) = delete;
    CompressedBatchReader& operator=(const CompressedBatchReader&) = delete;
    CompressedBatchReader(CompressedBatchReader&&) noexcept = default;
    CompressedBatchReader& operator=(CompressedBatchReader&&) noexcept = default;

    std::size_t read_batch(int fd,
                           const PgenFileIndex& index,
                           std::span<const uint32_t> raw_vindices,
                           uint64_t block_seq_id,
                           AlignedRawBuffer<uint8_t, 64>& dst_buf);

    std::size_t read_batch_uptr(int fd,
                                const PgenFileIndex& index,
                                std::span<const uint32_t> raw_vindices,
                                uint64_t block_seq_id,
                                AlignedRawBuffer<uintptr_t, 64>& dst_buf);

private:
    struct PendingSlice {
        uint64_t file_off = 0;
        std::size_t len = 0;
        std::size_t buf_off = 0;
    };

    std::size_t build_and_read(int fd,
                               const PgenFileIndex& index,
                               std::span<const uint32_t> raw_vindices,
                               uint64_t block_seq_id,
                               uint8_t* (*resize_fn)(void* ctx, std::size_t total_bytes),
                               void* resize_ctx);

    std::vector<uint32_t> req_vidx_;
    std::vector<uint64_t> req_buf_off_;
    std::vector<PendingSlice> pending_slices_;
    std::vector<gcta_io::FileByteSpan> spans_;
    gcta_io::SpanBatchCoalescer coalescer_;
};

// PLINK2 2-bit -> PLINK1 .bed 2-bit lookup tables:
//   PLINK2: 0=hom REF, 1=het, 2=hom ALT, 3=missing
//   PLINK1: 3=hom REF(0), 2=het(1), 0=hom ALT(2), 1=missing
inline constexpr uint8_t kP2ToP1[4] = {3u, 2u, 0u, 1u};
inline constexpr uint8_t kP2InvToP1[4] = {0u, 2u, 3u, 1u};

void decode_non_ld_to_bed2(const uint8_t* rec_ptr,
                           std::size_t rec_len,
                           uint16_t vrtype,
                           uint32_t raw_sample_ct,
                           uint8_t* dst_bed2);

void decode_ld_to_bed2(const uint8_t* rec_ptr,
                       std::size_t rec_len,
                       uint16_t vrtype,
                       uint32_t raw_sample_ct,
                       const uint8_t* ldbase_bed2,
                       uint8_t* dst_bed2);

class PgenThreadCache {
public:
    PgenThreadCache() noexcept = default;
    PgenThreadCache(const PgenThreadCache&) = delete;
    PgenThreadCache& operator=(const PgenThreadCache&) = delete;
    PgenThreadCache(PgenThreadCache&&) noexcept = default;
    PgenThreadCache& operator=(PgenThreadCache&&) noexcept = default;

    const uint8_t* resolve(const void* block_buf, uint32_t idx);

    void decode_into(const void* block_buf, uint32_t idx, uint8_t* dst_bed2);

    void clear() noexcept {
        ldbase_valid_ = false;
        ext_valid_ = false;
    }

private:
    void ensure_buffers(uint32_t bed_rec_bytes);

    AlignedRawBuffer<uint8_t, 64> ldbase_buf_;
    AlignedRawBuffer<uint8_t, 64> cur_buf_;
    uint32_t cached_bed_rec_bytes_ = 0;

    bool ldbase_valid_ = false;
    const void* ldbase_block_ptr_ = nullptr;
    uint64_t ldbase_seq_id_ = 0;
    uint64_t ldbase_rec_off_ = 0;
    uint32_t ldbase_rec_len_ = 0;
    uint16_t ldbase_vrtype_ = 0;

    bool ext_valid_ = false;
    const void* ext_block_ptr_ = nullptr;
    uint64_t ext_seq_id_ = 0;
    uint64_t ext_rec_off_ = 0;
    uint32_t ext_rec_len_ = 0;
    uint16_t ext_vrtype_ = 0;
    const uint8_t* ext_bed2_ptr_ = nullptr;
};

}  // namespace pgen_hardcall

#endif  // GCTA_PGEN_HARDCALL_HPP
