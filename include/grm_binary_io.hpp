#pragma once

#include <Eigen/Dense>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <omp.h>

#include "chunked_grm_matvec.hpp"
#include "mem.hpp"
#include "Pheno.h"
#include "async_seq_io.hpp"
#include "grm_format.hpp"

namespace gcta_grm_io {

// Default double-buffered sequential I/O slot size (256 MiB per slot = 512 MiB
// total for 2 slots): matches kMaxAsyncGrmSlotBytes so bottom rows of large
// GRMs keep plenty of rows per chunk across high core counts while keeping RAM bounded.
inline constexpr size_t kDefaultGrmIoSlotBytes = 256ULL << 20;
inline constexpr size_t kMaxAsyncGrmSlotBytes  = 256ULL << 20;

// ---------------------------------------------------------------------------
// IEEE 754 binary16 (FP16) storage & conversion helpers
// ---------------------------------------------------------------------------
#if defined(__FLT16_MANT_DIG__)
using GrmFp16 = _Float16;
inline GrmFp16 f32_to_fp16(float x) noexcept { return static_cast<_Float16>(x); }
inline float   fp16_to_f32(GrmFp16 h) noexcept { return static_cast<float>(h); }
#else
using GrmFp16 = uint16_t;
inline GrmFp16 f32_to_fp16(float x) noexcept {
    const uint32_t u = std::bit_cast<uint32_t>(x);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t  exp  = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t       mant = u & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000u;
        const int shift = 14 - exp;
        const uint32_t round_bit = 1u << (shift - 1);
        const uint32_t sticky    = (mant & (round_bit - 1u)) != 0u;
        const uint32_t lsb       = (mant >> shift) & 1u;
        return static_cast<uint16_t>(sign | ((mant + round_bit - 1u + (sticky | lsb)) >> shift));
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    const uint32_t round_bit = 0x00001000u;
    const uint32_t sticky    = (mant & 0x00000FFFu) != 0u;
    const uint32_t lsb       = (mant >> 13) & 1u;
    uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    h += (mant & round_bit) && (sticky || lsb);
    return static_cast<uint16_t>(h);
}
inline float fp16_to_f32(GrmFp16 h) noexcept {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    uint32_t       mant = static_cast<uint32_t>(h) & 0x03FFu;
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            uint32_t e = 127 - 14;
            while ((mant & 0x0400u) == 0u) { mant <<= 1; --e; }
            mant &= 0x03FFu;
            u = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(u);
}
#endif

static_assert(sizeof(GrmFp16) == 2, "GrmFp16 must be 2 bytes");

// Forward declared so read_grm_binary (below) can call this instead of
// duplicating its logic inline -- see the definition further down.
inline double read_grm_N_mean(const std::string& prefix, int n_grm, bool prefer_fp16 = false);

// Contiguous run-length encoding of a strictly increasing monotonic sample map kp:
// analysis indices [dst_start, dst_start + len) map to contiguous file indices
// [src_start, src_start + len). Replaces scalar gathers with bulk memcpy / SIMD
// conversions when most individuals are retained.
struct KpRun {
    int dst_start;
    int src_start;
    int len;
};

inline std::vector<KpRun> build_kp_runs(const std::vector<int>& kp) {
    std::vector<KpRun> runs;
    const int n = static_cast<int>(kp.size());
    int i = 0;
    while (i < n) {
        const int dst0 = i;
        const int src0 = kp[i];
        int len = 1;
        while (i + len < n && kp[i + len] == src0 + len) {
            ++len;
        }
        runs.push_back({dst0, src0, len});
        i += len;
    }
    return runs;
}

inline size_t find_first_kp_run(const std::vector<KpRun>& runs, int cs) noexcept {
    const auto it = std::upper_bound(
        runs.begin(), runs.end(), cs,
        [](int val, const KpRun& r) { return val < r.dst_start + r.len; });
    return static_cast<size_t>(it - runs.begin());
}

// Row boundaries [0, b1, b2, ..., n] splitting a packed lower-triangle-by-row
// file (row i holds i+1 elements) into num_parts contiguous ranges with
// roughly equal total element count.
inline std::vector<int> triangular_row_partition(int n, int num_parts) {
    std::vector<int> bounds(1, 0);
    const double total = static_cast<double>(n) * (n + 1) / 2.0;
    int rs = 0;
    for (int p = 1; p < num_parts && rs < n; ++p) {
        const double target = total * p / static_cast<double>(num_parts);
        int re = static_cast<int>(std::floor((-1.0 + std::sqrt(1.0 + 8.0 * target)) / 2.0));
        re = std::clamp(re, rs + 1, n);
        bounds.push_back(re);
        rs = re;
    }
    bounds.push_back(n);
    return bounds;
}

// Size in bytes of a GCTA-format packed lower-triangle .grm.bin (or
// .grm.N.bin) for n individuals: n(n+1)/2 float32 entries.
inline size_t grm_packed_bytes(int n) {
    return static_cast<size_t>(n) * (n + 1) / 2 * sizeof(float);
}

// Largest row index r > lo such that r*(r+1)/2 <= target_elems, clamped to
// [lo+1, n]. Same closed form as triangular_row_partition above; used to
// pick row-aligned chunk boundaries so a chunk never splits a row across
// two reads.
inline int row_bound_for_cumulative(size_t target_elems, int lo, int n) {
    if (target_elems == 0) return std::min(lo + 1, n);
    const double t = static_cast<double>(target_elems);
    const int r = static_cast<int>(std::floor((-1.0 + std::sqrt(1.0 + 8.0 * t)) / 2.0));
    return std::clamp(r, lo + 1, n);
}

// Cap per-syscall I/O to 1 GiB (safely below INT_MAX on macOS/Darwin, where
// read()/pread()/write() return EINVAL for counts > INT_MAX).
inline constexpr size_t kMaxIoChunkBytes = gcta_io::kMaxIoChunkBytes;

// Blocking read of exactly `count` bytes into `buf`, retrying on EINTR and
// on short reads (both routine on network filesystems). Errors out via
// LOGGER on unexpected EOF or a hard read error.
inline void read_exact(int fd, void* buf, size_t count, const std::string& path) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxIoChunkBytes, count - done);
        const ssize_t r = ::read(fd, p + done, step);
        if (r < 0) {
            if (errno == EINTR) continue;
            close(fd);
            LOGGER.e(0, "read() failed on [" + path + "]: " + std::string(std::strerror(errno)));
        }
        if (r == 0) {
            close(fd);
            LOGGER.e(0, "unexpected EOF reading [" + path + "].");
        }
        done += static_cast<size_t>(r);
    }
}

// pread-based counterpart to read_exact above: reads exactly `count` bytes
// starting at `offset`, retrying on EINTR and on short reads.
inline void read_exact_at(int fd, void* buf, size_t count, off_t offset, const std::string& path) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxIoChunkBytes, count - done);
        const ssize_t r = ::pread(fd, p + done, step, offset + static_cast<off_t>(done));
        if (r < 0) {
            if (errno == EINTR) continue;
            LOGGER.e(0, "pread() failed on [" + path + "]: " + std::string(std::strerror(errno)));
        }
        if (r == 0)
            LOGGER.e(0, "unexpected EOF reading [" + path + "].");
        done += static_cast<size_t>(r);
    }
}

// Blocking write of exactly `count` bytes from `buf`, retrying on EINTR and
// on short writes. Errors out via LOGGER on a hard write error.
inline void write_exact(int fd, const void* buf, size_t count, const std::string& path) {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxIoChunkBytes, count - done);
        const ssize_t w = ::write(fd, p + done, step);
        if (w < 0) {
            if (errno == EINTR) continue;
            close(fd);
            LOGGER.e(0, "write() failed on [" + path + "]: " + std::string(std::strerror(errno)));
        }
        done += static_cast<size_t>(w);
    }
}

// ---------------------------------------------------------------------------
// Shared POSIX & .grm.f16 header/diagonal/id helpers
// ---------------------------------------------------------------------------

inline int open_readonly_checked(const std::string& path) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(path.c_str(), flags);
    if (fd == -1) {
        LOGGER.e(0, "cannot open [" + path + "]: " + std::string(std::strerror(errno)));
    }
    return fd;
}

// Fast raw-byte comparison of two files in 4 MiB sequential POSIX chunks.
inline bool files_byte_identical(const std::string& path_a, const std::string& path_b) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fda = ::open(path_a.c_str(), flags);
    if (fda < 0) return false;
    const int fdb = ::open(path_b.c_str(), flags);
    if (fdb < 0) {
        ::close(fda);
        return false;
    }
    struct stat sta{}, stb{};
    if (::fstat(fda, &sta) != 0 || ::fstat(fdb, &stb) != 0 || sta.st_size != stb.st_size) {
        ::close(fda);
        ::close(fdb);
        return false;
    }
    const size_t total = static_cast<size_t>(sta.st_size);
    constexpr size_t kBuf = 4ULL << 20;
    const size_t step_cap = std::min(total, kBuf);
    AlignedRawBuffer<char, 64> ba(std::max<size_t>(1, step_cap));
    AlignedRawBuffer<char, 64> bb(std::max<size_t>(1, step_cap));
    size_t done = 0;
    bool same = true;
    try {
        while (done < total) {
            const size_t n = std::min(step_cap, total - done);
            gcta_io::read_exact_throwing(fda, ba.data(), n, path_a);
            gcta_io::read_exact_throwing(fdb, bb.data(), n, path_b);
            if (std::memcmp(ba.data(), bb.data(), n) != 0) {
                same = false;
                break;
            }
            done += n;
        }
    } catch (...) {
        same = false;
    }
    ::close(fda);
    ::close(fdb);
    return same;
}

// Copy a file sequentially in 4 MiB POSIX chunks (no small stdio buffers).
inline void copy_file_posix(const std::string& src_path, const std::string& dst_path) {
    int rflags = O_RDONLY;
    int wflags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_CLOEXEC
    rflags |= O_CLOEXEC;
    wflags |= O_CLOEXEC;
#endif
    const int src_fd = ::open(src_path.c_str(), rflags);
    if (src_fd < 0) {
        LOGGER.e(0, "cannot open [" + src_path + "] to copy: " + std::string(std::strerror(errno)));
    }
    struct stat st{};
    if (::fstat(src_fd, &st) != 0 || st.st_size < 0) {
        ::close(src_fd);
        LOGGER.e(0, "fstat() failed on [" + src_path + "].");
    }
    const int dst_fd = ::open(dst_path.c_str(), wflags, 0644);
    if (dst_fd < 0) {
        ::close(src_fd);
        LOGGER.e(0, "cannot open [" + dst_path + "] for writing: " + std::string(std::strerror(errno)));
    }
    const size_t total = static_cast<size_t>(st.st_size);
    constexpr size_t kBuf = 4ULL << 20;
    AlignedRawBuffer<char, 64> buf(std::max<size_t>(1, std::min(total, kBuf)));
    try {
        size_t done = 0;
        while (done < total) {
            const size_t n = std::min(buf.size(), total - done);
            gcta_io::read_exact_throwing(src_fd, buf.data(), n, src_path);
            gcta_io::write_exact_throwing(dst_fd, buf.data(), n, dst_path);
            done += n;
        }
    } catch (const std::exception& e) {
        ::close(src_fd);
        ::close(dst_fd);
        LOGGER.e(0, std::string("failed while copying [") + src_path + "] to [" + dst_path + "]: " + e.what());
    }
    ::close(src_fd);
    if (::close(dst_fd) != 0) {
        LOGGER.e(0, "close() failed on [" + dst_path + "]: " + std::string(std::strerror(errno)));
    }
}

// Write a .grm.id file sequentially in 4 MiB chunks (zero std::endl flushes).
inline void write_id_file_buffered(const std::string& path, const std::vector<std::string>& ids) {
    int flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(path.c_str(), flags, 0644);
    if (fd < 0) {
        LOGGER.e(0, "cannot open [" + path + "] for writing: " + std::string(std::strerror(errno)));
    }
    constexpr size_t kFlushBytes = 4ULL << 20;
    std::string out_buf;
    out_buf.reserve(std::min<size_t>(kFlushBytes + 4096, ids.size() * 24 + 4096));
    try {
        for (const std::string& id : ids) {
            out_buf.append(id);
            out_buf.push_back('\n');
            if (out_buf.size() >= kFlushBytes) {
                gcta_io::write_exact_throwing(fd, out_buf.data(), out_buf.size(), path);
                out_buf.clear();
            }
        }
        if (!out_buf.empty()) {
            gcta_io::write_exact_throwing(fd, out_buf.data(), out_buf.size(), path);
        }
    } catch (const std::exception& e) {
        ::close(fd);
        LOGGER.e(0, std::string("failed while writing [") + path + "]: " + e.what());
    }
    if (::close(fd) != 0) {
        LOGGER.e(0, "close() failed on [" + path + "]: " + std::string(std::strerror(errno)));
    }
}

// Validate a .grm.f16 header against grm_ids (sample count + id_hash) and read
// the exact float32 diagonal array in a single contiguous pread().
inline void read_grm_f16_header_and_diag(
    int fd,
    const std::string& path,
    const std::vector<std::string>& grm_ids,
    GrmFp16Header& hdr,
    Eigen::VectorXf& diag_out)
{
    read_and_validate_grm_f16_header(fd, path, hdr);
    const uint64_t n_ids = static_cast<uint64_t>(grm_ids.size());
    if (hdr.n != n_ids) {
        LOGGER.e(0, "Sample count mismatch: [" + path + "] header reports n=" +
                    std::to_string(hdr.n) + ", but .grm.id has " + std::to_string(n_ids) + " IDs.");
    }
    if (hdr.id_hash != 0) {
        const uint64_t expected_hash = gcta_io::hash_sample_ids(grm_ids);
        if (hdr.id_hash != expected_hash) {
            LOGGER.e(0, "Sample ID hash mismatch between [" + path + "] (0x" +
                        gcta_io::hash_hex(hdr.id_hash) + ") and .grm.id (0x" +
                        gcta_io::hash_hex(expected_hash) + "). Ensure .grm.f16 and .grm.id belong together.");
        }
    }
    diag_out.resize(static_cast<Eigen::Index>(n_ids));
    if (n_ids > 0) {
        try {
            gcta_io::read_exact_at_throwing(
                fd, diag_out.data(), static_cast<size_t>(n_ids) * sizeof(float),
                static_cast<off_t>(hdr.diag_offset), path);
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("Failed reading diagonal from [") + path + "]: " + e.what());
        }
    }
}

// Write draft header ("GCTAG16?") + zeroed diagonal/padding [0 .. body_offset)
// at offset 0 of a freshly opened AsyncSeqWriter (which starts streaming the
// packed fp16 body at body_offset).
inline void write_grm_f16_draft_prefix(gcta_io::AsyncSeqWriter& writer, uint64_t n) {
    const uint64_t body_off = grm_f16_body_offset(n);
    GrmFp16Header draft{};
    std::memcpy(draft.magic, kGrmF16MagicDraft, sizeof(draft.magic));
    draft.version        = kGrmF16Version;
    draft.flags          = 0;
    draft.dtype          = kGrmF16DtypeFp16;
    draft.header_bytes   = kGrmF16HeaderSize;
    draft.n              = n;
    draft.diag_offset    = kGrmF16HeaderSize;
    draft.body_offset    = body_off;
    draft.part_row_start = 0;
    draft.part_row_end   = n;

    std::vector<char> prefix(static_cast<size_t>(body_off), 0);
    std::memcpy(prefix.data(), &draft, sizeof(GrmFp16Header));
    writer.pwrite_after_drain(0, prefix.data(), prefix.size());
}

// Drain all body writes, commit exact float32 diag[n] at offset 128, then
// atomically commit the finalized 128-byte header ("GCTAG16\0") at offset 0
// and finish/close the AsyncSeqWriter.
inline void finalize_grm_f16_file(
    gcta_io::AsyncSeqWriter& writer,
    const GrmFp16Header& final_hdr,
    const float* diag_f32,
    uint64_t n)
{
    writer.drain();
    if (n > 0 && diag_f32 != nullptr) {
        writer.pwrite_after_drain(
            static_cast<off_t>(final_hdr.diag_offset),
            diag_f32,
            static_cast<size_t>(n) * sizeof(float));
    }
    writer.pwrite_after_drain(0, &final_hdr, sizeof(GrmFp16Header));
    writer.finish();
}

namespace detail {

struct RowChunkPlan {
    int row_lo;
    int row_hi;
    int j_lo;
    int j_hi;
};

// Build row-aligned sequential read chunks for AsyncSeqReader.
// For monotonic subsetting (kp != nullptr && !is_identity), skips leading/middle
// chunks containing no kept rows and stops immediately after kp.back().
inline void build_row_chunk_plan(
    int n_grm,
    const std::vector<int>* kp,
    bool is_identity,
    size_t max_elems_per_chunk,
    off_t body_off,
    size_t elem_bytes,
    std::vector<RowChunkPlan>& plans,
    std::vector<std::pair<off_t, size_t>>& io_chunks)
{
    plans.clear();
    io_chunks.clear();
    const int n_out = is_identity ? n_grm : (kp ? static_cast<int>(kp->size()) : 0);
    if (n_out <= 0 || n_grm <= 0) return;

    const int stop_row = is_identity ? n_grm : ((*kp)[static_cast<size_t>(n_out - 1)] + 1);
    const size_t eff_max_elems = std::max<size_t>(1, max_elems_per_chunk);

    int row_cursor = 0;
    int j_cursor = 0;
    while (row_cursor < stop_row && j_cursor < n_out) {
        const size_t row_base_cursor = static_cast<size_t>(row_cursor) * (row_cursor + 1) / 2;
        const int row_end = row_bound_for_cumulative(row_base_cursor + eff_max_elems, row_cursor, stop_row);
        int j_end = is_identity ? row_end : j_cursor;
        if (!is_identity) {
            while (j_end < n_out && (*kp)[static_cast<size_t>(j_end)] < row_end) ++j_end;
        }
        if (j_end > j_cursor) {
            const size_t row_base_end = static_cast<size_t>(row_end) * (row_end + 1) / 2;
            const size_t elems = row_base_end - row_base_cursor;
            plans.push_back({row_cursor, row_end, j_cursor, j_end});
            io_chunks.emplace_back(
                body_off + static_cast<off_t>(row_base_cursor * elem_bytes),
                elems * elem_bytes);
            j_cursor = j_end;
        }
        row_cursor = row_end;
    }
}

// Validate a classic .grm.bin file size.
inline void validate_f32_bin_size(int fd, const std::string& path, int n_grm) {
    const size_t tri = static_cast<size_t>(n_grm) * (n_grm + 1) / 2;
    const size_t byte_len = tri * sizeof(float);
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0 || static_cast<size_t>(st.st_size) < byte_len) {
        ::close(fd);
        LOGGER.e(0, "unexpected size in [" + path + "]. "
                    "Expected " + std::to_string(tri) + " float32 entries for n=" +
                    std::to_string(n_grm) + ".");
    }
}

} // namespace detail

// Returns:
//   ids          — "FID\tIID" strings in GRM file order
//   G            — n×n matrix (upper-triangle-only when upper_only=true)
//   m_snps       — mean SNP count (from .grm.f16 header or .grm.N.bin; skipped if m_snps < 0)
//   trace_K_out  — optional exact trace(K) accumulated during load (or from .grm.f16 header)
//   trace_K2_out — optional exact trace(K^2) accumulated during load (or from .grm.f16 header)
template <typename Scalar>
inline void read_grm_binary(const std::string& prefix,
                             std::vector<std::string>& ids,
                             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& G,
                             double& m_snps,
                             bool upper_only = false,
                             size_t grm_read_chunk_bytes = kDefaultGrmIoSlotBytes,
                             double* trace_K_out = nullptr,
                             double* trace_K2_out = nullptr,
                             bool ids_preloaded = false,
                             bool prefer_fp16 = false)
{
    LOGGER.ts("grm");
    using std::to_string;

    const ResolvedGrmInput resolved = resolve_grm_input(prefix, prefer_fp16);
    if (!ids_preloaded) {
        ids = Pheno::read_sublist(resolved.id_path);
    }
    const int n = static_cast<int>(ids.size());
    if (n == 0) LOGGER.e(0, "GRM id file [" + resolved.id_path + "] is empty.");

    const int fd = open_readonly_checked(resolved.data_path);
    const bool is_f16 = (resolved.format == GrmDiskFormat::F16);

    GrmFp16Header hdr{};
    Eigen::VectorXf diag_f32;
    off_t body_off = 0;
    size_t elem_bytes = sizeof(float);
    if (is_f16) {
        read_grm_f16_header_and_diag(fd, resolved.data_path, ids, hdr, diag_f32);
        body_off = static_cast<off_t>(hdr.body_offset);
        elem_bytes = sizeof(GrmFp16);
    } else {
        detail::validate_f32_bin_size(fd, resolved.data_path, n);
    }

    G.resize(n, n);

    const bool want_traces = (trace_K_out != nullptr || trace_K2_out != nullptr);
    const bool header_has_traces = is_f16 && ((hdr.flags & kGrmF16FlagHasTraces) != 0);
    const bool accum_traces = want_traces && !header_has_traces;
    double tr_K = header_has_traces ? hdr.trace_K : 0.0;
    double tr_K2 = header_has_traces ? hdr.trace_K2 : 0.0;

    const size_t slot_bytes = std::clamp<size_t>(grm_read_chunk_bytes, 1ULL << 20, kMaxAsyncGrmSlotBytes);
    const size_t max_elems_per_chunk = std::max<size_t>(1, slot_bytes / elem_bytes);

    std::vector<detail::RowChunkPlan> plans;
    std::vector<std::pair<off_t, size_t>> io_chunks;
    detail::build_row_chunk_plan(n, nullptr, /*is_identity=*/true,
                                 max_elems_per_chunk, body_off, elem_bytes,
                                 plans, io_chunks);

    try {
        gcta_io::AsyncSeqReader reader(fd, resolved.data_path, std::move(io_chunks),
                                       /*max_slot_bytes=*/0, /*take_ownership=*/true);
        const int unpack_nt = std::max(1, omp_get_max_threads() - 1);
        const char* raw_ptr = nullptr;
        size_t raw_len = 0;
        size_t plan_idx = 0;
        while (reader.next_slot(raw_ptr, raw_len)) {
            const auto& plan = plans[plan_idx++];
            const int row_cursor = plan.row_lo;
            const int row_end    = plan.row_hi;
            const size_t row_base_cursor = static_cast<size_t>(row_cursor) * (row_cursor + 1) / 2;
            const int rows_in_chunk = row_end - row_cursor;
            const int dyn_chunk = std::clamp(rows_in_chunk / std::max(1, unpack_nt * 4), 1, 16);

            if (is_f16) {
                const auto* chunk_f16 = reinterpret_cast<const GrmFp16*>(raw_ptr);
                if (accum_traces) {
                    #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                    for (int i = row_cursor; i < row_end; ++i) {
                        const size_t local_off = static_cast<size_t>(i) * (i + 1) / 2 - row_base_cursor;
                        const GrmFp16* __restrict__ src_row = chunk_f16 + local_off;
                        Scalar* __restrict__ dst_col = G.col(i).data();
                        double row_off_sq = 0.0;
                        #pragma omp simd reduction(+:row_off_sq)
                        for (int r = 0; r < i; ++r) {
                            const float v = fp16_to_f32(src_row[r]);
                            dst_col[r] = static_cast<Scalar>(v);
                            const double vd = static_cast<double>(v);
                            row_off_sq += vd * vd;
                        }
                        const float diag_v = diag_f32[i];
                        dst_col[i] = static_cast<Scalar>(diag_v);
                        const double diag_d = static_cast<double>(diag_v);
                        tr_K  += diag_d;
                        tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                    }
                } else {
                    #pragma omp parallel for num_threads(unpack_nt) schedule(dynamic, dyn_chunk)
                    for (int i = row_cursor; i < row_end; ++i) {
                        const size_t local_off = static_cast<size_t>(i) * (i + 1) / 2 - row_base_cursor;
                        const GrmFp16* __restrict__ src_row = chunk_f16 + local_off;
                        Scalar* __restrict__ dst_col = G.col(i).data();
                        #pragma omp simd
                        for (int r = 0; r < i; ++r) {
                            dst_col[r] = static_cast<Scalar>(fp16_to_f32(src_row[r]));
                        }
                        dst_col[i] = static_cast<Scalar>(diag_f32[i]);
                    }
                }
            } else {
                const auto* chunk_f32 = reinterpret_cast<const float*>(raw_ptr);
                if (accum_traces) {
                    #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                    for (int i = row_cursor; i < row_end; ++i) {
                        const size_t local_off = static_cast<size_t>(i) * (i + 1) / 2 - row_base_cursor;
                        const float* __restrict__ src_row = chunk_f32 + local_off;
                        Scalar* __restrict__ dst_col = G.col(i).data();
                        double row_off_sq = 0.0;
                        #pragma omp simd reduction(+:row_off_sq)
                        for (int r = 0; r < i; ++r) {
                            const float v = src_row[r];
                            dst_col[r] = static_cast<Scalar>(v);
                            const double vd = static_cast<double>(v);
                            row_off_sq += vd * vd;
                        }
                        const float diag_v = src_row[i];
                        dst_col[i] = static_cast<Scalar>(diag_v);
                        const double diag_d = static_cast<double>(diag_v);
                        tr_K  += diag_d;
                        tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                    }
                } else {
                    #pragma omp parallel for num_threads(unpack_nt) schedule(dynamic, dyn_chunk)
                    for (int i = row_cursor; i < row_end; ++i) {
                        const size_t local_off = static_cast<size_t>(i) * (i + 1) / 2 - row_base_cursor;
                        if constexpr (std::is_same_v<Scalar, float>) {
                            std::memcpy(G.col(i).data(), chunk_f32 + local_off,
                                        static_cast<size_t>(i + 1) * sizeof(float));
                        } else {
                            G.col(i).head(i + 1) =
                                Eigen::Map<const Eigen::VectorXf>(chunk_f32 + local_off, i + 1).template cast<Scalar>();
                        }
                    }
                }
            }
            reader.release_slot();
        }
        reader.stop();
    } catch (const std::exception& e) {
        LOGGER.e(0, std::string("Failed reading GRM [") + resolved.data_path + "]: " + e.what());
    }

    if (trace_K_out)  *trace_K_out  = tr_K;
    if (trace_K2_out) *trace_K2_out = tr_K2;

    LOGGER.i(0, std::string("The ") + (is_f16 ? ".grm.f16" : ".grm.bin") + " fill took " +
                std::to_string(LOGGER.tp("grm")) + " seconds.");
    if (!upper_only) {
        LOGGER.ts("grm_mirror");
        #pragma omp parallel for schedule(guided)
        for (int c = 0; c < n; ++c)
            for (int r = c + 1; r < n; ++r)
                G(r, c) = G(c, r);
        LOGGER.i(0, "The GRM mirror took " + std::to_string(LOGGER.tp("grm_mirror")) + " seconds.");
    }

    if (m_snps < 0.0) {
        LOGGER.i(0, "Skipping SNP count read from .grm.N.bin.");
    } else if (is_f16) {
        m_snps = hdr.m_snps_mean;
    } else {
        m_snps = read_grm_N_mean(prefix, n, prefer_fp16);
    }
}

// Build a mapping from a reference ID list to indices in grm_ids.
// Returns kp[i] = index in grm_ids matching ref_ids[i], or -1 if not found.
inline std::vector<int> match_ids_to_grm(const std::vector<std::string>& ref_ids,
                                          const std::vector<std::string>& grm_ids)
{
    const size_t n_ref = ref_ids.size();
    const size_t n_grm = grm_ids.size();
    std::vector<int> kp(n_ref, -1);
    if (n_ref == 0 || n_grm == 0) return kp;

    // Fast path 1: exact identity (same count & identical order).
    if (n_ref == n_grm) {
        bool all_equal = true;
        for (size_t i = 0; i < n_ref; ++i) {
            if (ref_ids[i] != grm_ids[i]) {
                all_equal = false;
                break;
            }
        }
        if (all_equal) {
            for (size_t i = 0; i < n_ref; ++i) kp[i] = static_cast<int>(i);
            return kp;
        }
    }

    // Fast path 2: strictly increasing ordered subsequence (--keep / --remove).
    if (n_ref <= n_grm) {
        size_t g = 0;
        bool matched_in_order = true;
        for (size_t i = 0; i < n_ref; ++i) {
            while (g < n_grm && grm_ids[g] != ref_ids[i]) {
                ++g;
            }
            if (g >= n_grm) {
                matched_in_order = false;
                break;
            }
            kp[i] = static_cast<int>(g);
            ++g;
        }
        if (matched_in_order) return kp;
        std::fill(kp.begin(), kp.end(), -1);
    }

    std::unordered_map<std::string, int> grm_map;
    grm_map.reserve(n_grm * 2);
    for (int i = 0; i < static_cast<int>(n_grm); ++i)
        grm_map.emplace(grm_ids[i], i);

    for (int i = 0; i < static_cast<int>(n_ref); ++i) {
        if (const auto it = grm_map.find(ref_ids[i]); it != grm_map.end())
            kp[i] = it->second;
    }
    return kp;
}

// Tile reader for --grm-chunked-budget: returns K in ANALYSIS sample order
// (post-kp reindexing), keeping at most a bounded double-buffered row-band
// resident at once (strictly <= band_byte_budget_) rather than the whole file.
// Supports both classic float32 .grm.bin and native .grm.f16 (2x less I/O,
// 2x more rows per band, O(1) exact float32 diagonal from header).
class ChunkedGrmReader {
public:
    ChunkedGrmReader(const std::string& path, std::vector<int> kp, int n_grm,
                     size_t band_byte_budget = (1ull << 30) /* 1GiB */,
                     const std::vector<std::string>* grm_ids_for_hash = nullptr)
        : kp_(std::move(kp)), path_(path), n_grm_(n_grm), band_byte_budget_(band_byte_budget)
    {
        fd_ = open_readonly_checked(path_);

        is_fp16_ = (path_.size() >= 8 && path_.compare(path_.size() - 8, 8, ".grm.f16") == 0);
        if (is_fp16_) {
            read_and_validate_grm_f16_header(fd_, path_, f16_hdr_);
            if (static_cast<int>(f16_hdr_.n) != n_grm_) {
                ::close(fd_);
                fd_ = -1;
                LOGGER.e(0, "Sample count mismatch in [" + path_ + "]: header n=" +
                            std::to_string(f16_hdr_.n) + " vs .grm.id n=" + std::to_string(n_grm_) + ".");
            }
            if (grm_ids_for_hash && f16_hdr_.id_hash != 0) {
                const uint64_t expected_hash = gcta_io::hash_sample_ids(*grm_ids_for_hash);
                if (f16_hdr_.id_hash != expected_hash) {
                    ::close(fd_);
                    fd_ = -1;
                    LOGGER.e(0, "Sample ID hash mismatch in [" + path_ + "].");
                }
            }
            elem_bytes_ = sizeof(GrmFp16);
            body_offset_ = static_cast<off_t>(f16_hdr_.body_offset);
            byte_len_ = static_cast<size_t>(grm_f16_packed_bytes(static_cast<uint64_t>(n_grm_)));
        } else {
            elem_bytes_ = sizeof(float);
            body_offset_ = 0;
            const size_t tri = static_cast<size_t>(n_grm) * (n_grm + 1) / 2;
            byte_len_ = tri * sizeof(float);
            struct stat st{};
            if (::fstat(fd_, &st) != 0 || st.st_size < 0 || static_cast<size_t>(st.st_size) < byte_len_) {
                ::close(fd_);
                fd_ = -1;
                LOGGER.e(0, "unexpected size in [" + path + "].");
            }
        }

        kp_is_identity_ = (static_cast<int>(kp_.size()) == n_grm_);
        bool kp_is_monotonic = true;
        for (int i = 0; i < static_cast<int>(kp_.size()); ++i) {
            if (kp_[i] != i) kp_is_identity_ = false;
            if (i > 0 && kp_[i] <= kp_[i - 1]) kp_is_monotonic = false;
        }

        if (!kp_is_monotonic) {
            ::close(fd_);
            fd_ = -1;
            LOGGER.e(0, "--grm-chunked-budget requires the analysis sample order to match "
                        "(or be an ordered subset of) [" + path + "]'s row order. Re-run "
                        "with sample IDs sorted to match the GRM's native order, or drop "
                        "--grm-chunked-budget to use the dense path.");
        }

        if (is_fp16_) {
            Eigen::VectorXf diag_full(n_grm_);
            if (n_grm_ > 0) {
                try {
                    gcta_io::read_exact_at_throwing(
                        fd_, diag_full.data(), static_cast<size_t>(n_grm_) * sizeof(float),
                        static_cast<off_t>(f16_hdr_.diag_offset), path_);
                } catch (const std::exception& e) {
                    ::close(fd_);
                    fd_ = -1;
                    LOGGER.e(0, std::string("Failed reading diagonal from [") + path_ + "]: " + e.what());
                }
            }
            if (kp_is_identity_) {
                diag_f32_ = std::move(diag_full);
            } else {
                const int n_out = static_cast<int>(kp_.size());
                diag_f32_.resize(n_out);
                for (int i = 0; i < n_out; ++i) {
                    diag_f32_[i] = diag_full[kp_[static_cast<size_t>(i)]];
                }
            }
        }

        if (!kp_is_identity_ && !kp_.empty()) {
            std::vector<KpRun> runs = build_kp_runs(kp_);
            if (runs.size() * 2 < kp_.size()) {
                kp_runs_ = std::move(runs);
            }
        }
    }

    ~ChunkedGrmReader() {
        if (fd_ != -1) {
            ::close(fd_);
        }
    }
    ChunkedGrmReader(const ChunkedGrmReader&)            = delete;
    ChunkedGrmReader& operator=(const ChunkedGrmReader&) = delete;

    bool is_fp16() const noexcept { return is_fp16_; }
    bool kp_is_identity() const noexcept { return kp_is_identity_; }
    const GrmFp16Header& f16_header() const noexcept { return f16_hdr_; }

    // Release cached row-band and tile-scratch buffers immediately (safe to reuse later).
    void release_buffers() const noexcept {
        band_buf_[0].reset();
        band_buf_[1].reset();
        band_lo_ = {0, 0};
        band_hi_ = {0, 0};
        active_buf_ = 0;
        tile_scratch_.resize(0, 0);
        tile_scratch_f32_.resize(0, 0);
        tile_scratch_T_.resize(0, 0);
        tile_scratch_f32_T_.resize(0, 0);
    }

    // Partition analysis rows [0, n) into variable-height row bands [rs, re) so
    // each band has at most max_rows_per_band rows AND its file rows
    // [kp_[rs], kp_[re-1] + 1) fit within band_byte_budget_ / 2 (keeping two
    // double-buffered bands strictly within band_byte_budget_).
    std::vector<int> compute_row_bands(int max_rows_per_band) const {
        const int n = static_cast<int>(kp_.size());
        std::vector<int> bands;
        bands.push_back(0);
        if (n == 0) return bands;

        const int max_rows = std::max(1, max_rows_per_band);
        const size_t min_one_row_bytes = static_cast<size_t>(std::max(1, n_grm_)) * elem_bytes_;
        const size_t per_buf_bytes = std::max(band_byte_budget_ / 2, min_one_row_bytes);
        const size_t per_buf_elems = std::max<size_t>(1, per_buf_bytes / elem_bytes_);

        int rs = 0;
        while (rs < n) {
            const int file_lo = kp_[static_cast<size_t>(rs)];
            const size_t elem_lo = static_cast<size_t>(file_lo) * (file_lo + 1) / 2;
            const int max_file_hi = row_bound_for_cumulative(elem_lo + per_buf_elems, file_lo, n_grm_);
            const int lim = std::min(n, rs + max_rows);

            const auto it = std::lower_bound(kp_.begin() + rs, kp_.begin() + lim, max_file_hi);
            int re = static_cast<int>(it - kp_.begin());
            if (re <= rs) re = rs + 1;
            bands.push_back(re);
            rs = re;
        }
        return bands;
    }

    // Double-buffered async band prefetcher for streaming sweeps.
    // Spawns strictly one std::jthread that prefetches band b+1 into slot (b+1)&1
    // while the caller computes tiles on band b in slot b&1.
    class BandSweepSession final : public gcta_chunked::TileSweep {
    public:
        BandSweepSession(const ChunkedGrmReader& reader, const std::vector<int>& bands)
            : reader_(reader),
              bands_(bands),
              num_bands_(static_cast<int>(bands.size()) - 1)
        {
            io_thread_ = std::jthread([this](std::stop_token st) {
                try {
                    for (int b = 0; b < num_bands_ && !st.stop_requested(); ++b) {
                        const int slot = b & 1;
                        {
                            std::unique_lock<std::mutex> lock(mu_);
                            free_cv_.wait(lock, st, [&]() { return !slot_busy_[static_cast<size_t>(slot)]; });
                            if (st.stop_requested()) return;
                            slot_busy_[static_cast<size_t>(slot)] = true;
                        }
                        const int rs = bands_[static_cast<size_t>(b)];
                        const int re = bands_[static_cast<size_t>(b + 1)];
                        const int fl = reader_.kp_[static_cast<size_t>(rs)];
                        const int fh = reader_.kp_[static_cast<size_t>(re - 1)] + 1;
                        reader_.load_band_exact_into_slot(slot, fl, fh);
                        {
                            std::lock_guard<std::mutex> lock(mu_);
                            slot_ready_band_[static_cast<size_t>(slot)] = b;
                        }
                        ready_cv_.notify_one();
                    }
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(mu_);
                        io_error_ = std::current_exception();
                        failed_ = true;
                    }
                    ready_cv_.notify_all();
                }
            });
        }

        ~BandSweepSession() override {
            io_thread_.request_stop();
            free_cv_.notify_all();
            if (io_thread_.joinable()) {
                io_thread_.join();
            }
        }

        void acquire(int band_idx) override {
            const int slot = band_idx & 1;
            std::unique_lock<std::mutex> lock(mu_);
            ready_cv_.wait(lock, [&]() {
                return slot_ready_band_[static_cast<size_t>(slot)] == band_idx || failed_;
            });
            if (failed_) {
                std::rethrow_exception(io_error_);
            }
            reader_.active_buf_ = slot;
        }

        void release(int band_idx) override {
            const int slot = band_idx & 1;
            {
                std::lock_guard<std::mutex> lock(mu_);
                slot_ready_band_[static_cast<size_t>(slot)] = -1;
                slot_busy_[static_cast<size_t>(slot)] = false;
            }
            free_cv_.notify_one();
        }

    private:
        const ChunkedGrmReader& reader_;
        std::vector<int> bands_;
        int num_bands_ = 0;
        std::mutex mu_;
        std::condition_variable_any ready_cv_;
        std::condition_variable_any free_cv_;
        std::array<bool, 2> slot_busy_{false, false};
        std::array<int, 2> slot_ready_band_{-1, -1};
        bool failed_ = false;
        std::exception_ptr io_error_;
        std::jthread io_thread_;
    };

    std::unique_ptr<gcta_chunked::TileSweep> begin_sweep(const std::vector<int>& bands) const {
        if (bands.size() <= 2) return nullptr;
        return std::make_unique<BandSweepSession>(*this, bands);
    }

    // Exact diagonal of K:
    //   - Native .grm.f16: O(1) directly from header float32 diagonal (0 disk I/O).
    //   - Classic .grm.bin: HPC-safe 1-stream sequential chunked scan via
    //     AsyncSeqReader (2 x 64 MiB), eliminating the 16-thread random 4-byte pread storm.
    Eigen::VectorXd read_diagonal() const {
        const int n = static_cast<int>(kp_.size());
        Eigen::VectorXd diag(n);
        if (n == 0) return diag;

        if (is_fp16_) {
            return diag_f32_.cast<double>();
        }

        std::vector<detail::RowChunkPlan> plans;
        std::vector<std::pair<off_t, size_t>> io_chunks;
        const size_t max_elems = std::max<size_t>(1, kDefaultGrmIoSlotBytes / sizeof(float));
        detail::build_row_chunk_plan(n_grm_, &kp_, kp_is_identity_, max_elems,
                                     /*body_off=*/0, sizeof(float), plans, io_chunks);

        try {
            gcta_io::AsyncSeqReader reader(fd_, path_, std::move(io_chunks),
                                           /*max_slot_bytes=*/0, /*take_ownership=*/false);
            const char* raw_ptr = nullptr;
            size_t raw_len = 0;
            size_t plan_idx = 0;
            while (reader.next_slot(raw_ptr, raw_len)) {
                const auto& plan = plans[plan_idx++];
                const auto* buf = reinterpret_cast<const float*>(raw_ptr);
                const size_t row_base_cursor = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;
                for (int j = plan.j_lo; j < plan.j_hi; ++j) {
                    const int gi = kp_is_identity_ ? j : kp_[static_cast<size_t>(j)];
                    const size_t diag_off = static_cast<size_t>(gi) * (gi + 1) / 2 + static_cast<size_t>(gi) - row_base_cursor;
                    diag[j] = static_cast<double>(buf[diag_off]);
                }
                reader.release_slot();
            }
            reader.stop();
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("Failed reading diagonal of [") + path_ + "]: " + e.what());
        }
        return diag;
    }

    // Fast-path transposed float32 tile: returns K[rs:re, cs:ce]^T of shape
    // (ce-cs) x (re-rs) ColMajor so each GRM row unpacks into a contiguous
    // column (zero 16 KiB power-of-2 stride scatter). For diagonal tiles
    // (rs == cs), the UPPER triangle (r <= c) is populated (with exact float32
    // diagonal entries on .grm.f16).
    Eigen::Ref<const Eigen::MatrixXf> read_tile_f32_T(int rs, int re, int cs, int ce) const {
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_f32_T_.rows() < tile_cols || tile_scratch_f32_T_.cols() < tile_rows) {
            tile_scratch_f32_T_.resize(
                std::max<Eigen::Index>(tile_scratch_f32_T_.rows(), tile_cols),
                std::max<Eigen::Index>(tile_scratch_f32_T_.cols(), tile_rows));
        }
        auto tile_T = tile_scratch_f32_T_.topLeftCorner(tile_cols, tile_rows);
        const bool diagonal_tile = (rs == cs);

        ensure_band(kp_[static_cast<size_t>(rs)], kp_[static_cast<size_t>(re - 1)] + 1);
        const int slot = active_buf_;
        const size_t band_elem_lo = static_cast<size_t>(band_lo_[static_cast<size_t>(slot)]) *
                                    (band_lo_[static_cast<size_t>(slot)] + 1) / 2;

        if (is_fp16_) {
            const auto* buf = reinterpret_cast<const GrmFp16*>(band_buf_[static_cast<size_t>(slot)].data());
            if (kp_is_identity_) {
                #pragma omp parallel for schedule(guided)
                for (int lp = 0; lp < tile_rows; ++lp) {
                    const int gi = rs + lp;
                    const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                    if (lq_end == 0) continue;
                    const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                    const GrmFp16* __restrict__ src = buf + row_base + cs;
                    float* __restrict__ dst = tile_T.col(lp).data();
                    #pragma omp simd
                    for (int lq = 0; lq < lq_end; ++lq) {
                        dst[lq] = fp16_to_f32(src[lq]);
                    }
                    if (diagonal_tile) {
                        dst[lp] = diag_f32_[gi];
                    }
                }
                return tile_T;
            }
            if (!kp_runs_.empty()) {
                const size_t first_run = find_first_kp_run(kp_runs_, cs);
                #pragma omp parallel for schedule(guided)
                for (int lp = 0; lp < tile_rows; ++lp) {
                    const int gi = kp_[static_cast<size_t>(rs + lp)];
                    const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                    if (lq_end == 0) continue;
                    const int dst_limit = cs + lq_end;
                    const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                    const GrmFp16* __restrict__ row_ptr = buf + row_base;
                    float* __restrict__ dst_col = tile_T.col(lp).data();
                    for (size_t ri = first_run; ri < kp_runs_.size(); ++ri) {
                        const KpRun& r = kp_runs_[ri];
                        if (r.dst_start >= dst_limit) break;
                        const int d0 = std::max(r.dst_start, cs);
                        const int d1 = std::min(r.dst_start + r.len, dst_limit);
                        const int cnt = d1 - d0;
                        const int s0 = r.src_start + (d0 - r.dst_start);
                        const GrmFp16* __restrict__ src_seg = row_ptr + s0;
                        float* __restrict__ dst_seg = dst_col + (d0 - cs);
                        #pragma omp simd
                        for (int k = 0; k < cnt; ++k) {
                            dst_seg[k] = fp16_to_f32(src_seg[k]);
                        }
                    }
                    if (diagonal_tile) {
                        dst_col[lp] = diag_f32_[rs + lp];
                    }
                }
                return tile_T;
            }
            #pragma omp parallel for schedule(guided)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = kp_[static_cast<size_t>(rs + lp)];
                const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                if (lq_end == 0) continue;
                const int gj_lo = kp_[static_cast<size_t>(cs)];
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                const GrmFp16* row_span = buf + row_base + gj_lo;
                float* dst_col = tile_T.col(lp).data();
                for (int lq = 0; lq < lq_end; ++lq) {
                    dst_col[lq] = fp16_to_f32(row_span[kp_[static_cast<size_t>(cs + lq)] - gj_lo]);
                }
                if (diagonal_tile) {
                    dst_col[lp] = diag_f32_[rs + lp];
                }
            }
            return tile_T;
        }

        const auto* buf = reinterpret_cast<const float*>(band_buf_[static_cast<size_t>(slot)].data());
        if (kp_is_identity_) {
            #pragma omp parallel for schedule(guided)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = rs + lp;
                const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                if (lq_end == 0) continue;
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                std::memcpy(tile_T.col(lp).data(), buf + row_base + cs,
                            static_cast<size_t>(lq_end) * sizeof(float));
            }
            return tile_T;
        }

        if (!kp_runs_.empty()) {
            const size_t first_run = find_first_kp_run(kp_runs_, cs);
            #pragma omp parallel for schedule(guided)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = kp_[static_cast<size_t>(rs + lp)];
                const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                if (lq_end == 0) continue;
                const int dst_limit = cs + lq_end;
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                const float* row_ptr = buf + row_base;
                float* dst_col = tile_T.col(lp).data();
                for (size_t ri = first_run; ri < kp_runs_.size(); ++ri) {
                    const KpRun& r = kp_runs_[ri];
                    if (r.dst_start >= dst_limit) break;
                    const int d0 = std::max(r.dst_start, cs);
                    const int d1 = std::min(r.dst_start + r.len, dst_limit);
                    const int cnt = d1 - d0;
                    const int s0 = r.src_start + (d0 - r.dst_start);
                    std::memcpy(dst_col + (d0 - cs), row_ptr + s0,
                                static_cast<size_t>(cnt) * sizeof(float));
                }
            }
            return tile_T;
        }

        #pragma omp parallel for schedule(guided)
        for (int lp = 0; lp < tile_rows; ++lp) {
            const int gi = kp_[static_cast<size_t>(rs + lp)];
            const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
            if (lq_end == 0) continue;
            const int gj_lo = kp_[static_cast<size_t>(cs)];
            const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
            const float* row_span = buf + row_base + gj_lo;
            float* dst_col = tile_T.col(lp).data();
            for (int lq = 0; lq < lq_end; ++lq) {
                dst_col[lq] = row_span[kp_[static_cast<size_t>(cs + lq)] - gj_lo];
            }
        }
        return tile_T;
    }

    // Fast-path transposed double tile: returns K[rs:re, cs:ce]^T of shape
    // (ce-cs) x (re-rs) ColMajor. Diagonal tiles have UPPER triangle valid.
    Eigen::Ref<const Eigen::MatrixXd> read_tile_T(int rs, int re, int cs, int ce) const {
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_T_.rows() < tile_cols || tile_scratch_T_.cols() < tile_rows) {
            tile_scratch_T_.resize(
                std::max<Eigen::Index>(tile_scratch_T_.rows(), tile_cols),
                std::max<Eigen::Index>(tile_scratch_T_.cols(), tile_rows));
        }
        auto tile_T = tile_scratch_T_.topLeftCorner(tile_cols, tile_rows);
        const bool diagonal_tile = (rs == cs);

        ensure_band(kp_[static_cast<size_t>(rs)], kp_[static_cast<size_t>(re - 1)] + 1);
        const int slot = active_buf_;
        const size_t band_elem_lo = static_cast<size_t>(band_lo_[static_cast<size_t>(slot)]) *
                                    (band_lo_[static_cast<size_t>(slot)] + 1) / 2;

        if (is_fp16_) {
            const auto* buf = reinterpret_cast<const GrmFp16*>(band_buf_[static_cast<size_t>(slot)].data());
            if (kp_is_identity_) {
                #pragma omp parallel for schedule(guided)
                for (int lp = 0; lp < tile_rows; ++lp) {
                    const int gi = rs + lp;
                    const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                    if (lq_end == 0) continue;
                    const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                    const GrmFp16* __restrict__ src = buf + row_base + cs;
                    double* __restrict__ dst = tile_T.col(lp).data();
                    #pragma omp simd
                    for (int lq = 0; lq < lq_end; ++lq) {
                        dst[lq] = static_cast<double>(fp16_to_f32(src[lq]));
                    }
                    if (diagonal_tile) {
                        dst[lp] = static_cast<double>(diag_f32_[gi]);
                    }
                }
                return tile_T;
            }
            if (!kp_runs_.empty()) {
                const size_t first_run = find_first_kp_run(kp_runs_, cs);
                #pragma omp parallel for schedule(guided)
                for (int lp = 0; lp < tile_rows; ++lp) {
                    const int gi = kp_[static_cast<size_t>(rs + lp)];
                    const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                    if (lq_end == 0) continue;
                    const int dst_limit = cs + lq_end;
                    const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                    const GrmFp16* __restrict__ row_ptr = buf + row_base;
                    double* __restrict__ dst_col = tile_T.col(lp).data();
                    for (size_t ri = first_run; ri < kp_runs_.size(); ++ri) {
                        const KpRun& r = kp_runs_[ri];
                        if (r.dst_start >= dst_limit) break;
                        const int d0 = std::max(r.dst_start, cs);
                        const int d1 = std::min(r.dst_start + r.len, dst_limit);
                        const int cnt = d1 - d0;
                        const int s0 = r.src_start + (d0 - r.dst_start);
                        const GrmFp16* __restrict__ src_seg = row_ptr + s0;
                        double* __restrict__ dst_seg = dst_col + (d0 - cs);
                        #pragma omp simd
                        for (int k = 0; k < cnt; ++k) {
                            dst_seg[k] = static_cast<double>(fp16_to_f32(src_seg[k]));
                        }
                    }
                    if (diagonal_tile) {
                        dst_col[lp] = static_cast<double>(diag_f32_[rs + lp]);
                    }
                }
                return tile_T;
            }
            #pragma omp parallel for schedule(guided)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = kp_[static_cast<size_t>(rs + lp)];
                const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                if (lq_end == 0) continue;
                const int gj_lo = kp_[static_cast<size_t>(cs)];
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                const GrmFp16* row_span = buf + row_base + gj_lo;
                double* dst_col = tile_T.col(lp).data();
                for (int lq = 0; lq < lq_end; ++lq) {
                    dst_col[lq] = static_cast<double>(fp16_to_f32(row_span[kp_[static_cast<size_t>(cs + lq)] - gj_lo]));
                }
                if (diagonal_tile) {
                    dst_col[lp] = static_cast<double>(diag_f32_[rs + lp]);
                }
            }
            return tile_T;
        }

        const auto* buf = reinterpret_cast<const float*>(band_buf_[static_cast<size_t>(slot)].data());
        if (kp_is_identity_) {
            #pragma omp parallel for schedule(guided)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = rs + lp;
                const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                if (lq_end == 0) continue;
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                const float* __restrict__ src = buf + row_base + cs;
                double* __restrict__ dst = tile_T.col(lp).data();
                #pragma omp simd
                for (int lq = 0; lq < lq_end; ++lq) {
                    dst[lq] = static_cast<double>(src[lq]);
                }
            }
            return tile_T;
        }

        if (!kp_runs_.empty()) {
            const size_t first_run = find_first_kp_run(kp_runs_, cs);
            #pragma omp parallel for schedule(guided)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = kp_[static_cast<size_t>(rs + lp)];
                const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                if (lq_end == 0) continue;
                const int dst_limit = cs + lq_end;
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                const float* __restrict__ row_ptr = buf + row_base;
                double* __restrict__ dst_col = tile_T.col(lp).data();
                for (size_t ri = first_run; ri < kp_runs_.size(); ++ri) {
                    const KpRun& r = kp_runs_[ri];
                    if (r.dst_start >= dst_limit) break;
                    const int d0 = std::max(r.dst_start, cs);
                    const int d1 = std::min(r.dst_start + r.len, dst_limit);
                    const int cnt = d1 - d0;
                    const int s0 = r.src_start + (d0 - r.dst_start);
                    const float* __restrict__ src_seg = row_ptr + s0;
                    double* __restrict__ dst_seg = dst_col + (d0 - cs);
                    #pragma omp simd
                    for (int k = 0; k < cnt; ++k) {
                        dst_seg[k] = static_cast<double>(src_seg[k]);
                    }
                }
            }
            return tile_T;
        }

        #pragma omp parallel for schedule(guided)
        for (int lp = 0; lp < tile_rows; ++lp) {
            const int gi = kp_[static_cast<size_t>(rs + lp)];
            const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
            if (lq_end == 0) continue;
            const int gj_lo = kp_[static_cast<size_t>(cs)];
            const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
            const float* row_span = buf + row_base + gj_lo;
            double* dst_col = tile_T.col(lp).data();
            for (int lq = 0; lq < lq_end; ++lq) {
                dst_col[lq] = static_cast<double>(row_span[kp_[static_cast<size_t>(cs + lq)] - gj_lo]);
            }
        }
        return tile_T;
    }

    // Legacy non-transposed double tile (shape (re-rs) x (ce-cs), lower triangle valid on diagonal).
    Eigen::Ref<const Eigen::MatrixXd> read_tile(int rs, int re, int cs, int ce) const {
        const auto tile_T = read_tile_T(rs, re, cs, ce);
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_.rows() < tile_rows || tile_scratch_.cols() < tile_cols) {
            tile_scratch_.resize(std::max<Eigen::Index>(tile_scratch_.rows(), tile_rows),
                                 std::max<Eigen::Index>(tile_scratch_.cols(), tile_cols));
        }
        auto tile = tile_scratch_.topLeftCorner(tile_rows, tile_cols);
        const bool diagonal_tile = (rs == cs);
        constexpr int kB = 64;
        #pragma omp parallel for schedule(guided) if (tile_rows >= 128)
        for (int c0 = 0; c0 < tile_cols; c0 += kB) {
            const int c1 = std::min(tile_cols, c0 + kB);
            const int r_begin = diagonal_tile ? c0 : 0;
            for (int r0 = r_begin; r0 < tile_rows; r0 += kB) {
                const int r1 = std::min(tile_rows, r0 + kB);
                for (int c = c0; c < c1; ++c) {
                    const int r_lo = diagonal_tile ? std::max(r0, c) : r0;
                    double* dst = tile.col(c).data();
                    for (int r = r_lo; r < r1; ++r) {
                        dst[r] = tile_T(c, r);
                    }
                }
            }
        }
        return tile;
    }

    // Legacy non-transposed float32 tile (shape (re-rs) x (ce-cs), lower triangle valid on diagonal).
    Eigen::Ref<const Eigen::MatrixXf> read_tile_f32(int rs, int re, int cs, int ce) const {
        const auto tile_T = read_tile_f32_T(rs, re, cs, ce);
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_f32_.rows() < tile_rows || tile_scratch_f32_.cols() < tile_cols) {
            tile_scratch_f32_.resize(std::max<Eigen::Index>(tile_scratch_f32_.rows(), tile_rows),
                                     std::max<Eigen::Index>(tile_scratch_f32_.cols(), tile_cols));
        }
        auto tile = tile_scratch_f32_.topLeftCorner(tile_rows, tile_cols);
        const bool diagonal_tile = (rs == cs);
        constexpr int kB = 64;
        #pragma omp parallel for schedule(guided) if (tile_rows >= 128)
        for (int c0 = 0; c0 < tile_cols; c0 += kB) {
            const int c1 = std::min(tile_cols, c0 + kB);
            const int r_begin = diagonal_tile ? c0 : 0;
            for (int r0 = r_begin; r0 < tile_rows; r0 += kB) {
                const int r1 = std::min(tile_rows, r0 + kB);
                for (int c = c0; c < c1; ++c) {
                    const int r_lo = diagonal_tile ? std::max(r0, c) : r0;
                    float* dst = tile.col(c).data();
                    for (int r = r_lo; r < r1; ++r) {
                        dst[r] = tile_T(c, r);
                    }
                }
            }
        }
        return tile;
    }

private:
    mutable Eigen::MatrixXd tile_scratch_;
    mutable Eigen::MatrixXf tile_scratch_f32_;
    mutable Eigen::MatrixXd tile_scratch_T_;
    mutable Eigen::MatrixXf tile_scratch_f32_T_;

    mutable std::array<AlignedRawBuffer<char, 64>, 2> band_buf_;
    mutable std::array<int, 2> band_lo_{0, 0};
    mutable std::array<int, 2> band_hi_{0, 0};
    mutable int active_buf_ = 0;

    void load_band_exact_into_slot(int slot, int file_lo, int file_hi) const {
        const size_t sl = static_cast<size_t>(slot);
        if (file_lo >= band_lo_[sl] && file_hi <= band_hi_[sl]) return;
        band_lo_[sl] = 0;
        band_hi_[sl] = 0;
        const size_t elem_lo = static_cast<size_t>(file_lo) * (file_lo + 1) / 2;
        const size_t elem_hi = static_cast<size_t>(file_hi) * (file_hi + 1) / 2;
        const size_t byte_cnt = (elem_hi - elem_lo) * elem_bytes_;
        band_buf_[sl].resize(byte_cnt);
        gcta_io::read_exact_at_throwing(
            fd_, band_buf_[sl].data(), byte_cnt,
            body_offset_ + static_cast<off_t>(elem_lo * elem_bytes_), path_);
        band_lo_[sl] = file_lo;
        band_hi_[sl] = file_hi;
    }

    void ensure_band(int file_lo, int file_hi) const {
        const int slot = active_buf_;
        const size_t sl = static_cast<size_t>(slot);
        if (file_lo >= band_lo_[sl] && file_hi <= band_hi_[sl]) return;

        band_lo_[sl] = 0;
        band_hi_[sl] = 0;
        const size_t elem_lo = static_cast<size_t>(file_lo) * (file_lo + 1) / 2;
        const size_t target_elems = elem_lo + band_byte_budget_ / elem_bytes_;
        const int grown_hi  = row_bound_for_cumulative(target_elems, file_lo, n_grm_);
        const int actual_hi = std::max(file_hi, grown_hi);

        const size_t elem_hi = static_cast<size_t>(actual_hi) * (actual_hi + 1) / 2;
        const size_t byte_cnt = (elem_hi - elem_lo) * elem_bytes_;
        band_buf_[sl].resize(byte_cnt);
        read_exact_at(fd_, band_buf_[sl].data(), byte_cnt,
                      body_offset_ + static_cast<off_t>(elem_lo * elem_bytes_), path_);
        band_lo_[sl] = file_lo;
        band_hi_[sl] = actual_hi;
    }

    std::vector<int> kp_;
    std::vector<KpRun> kp_runs_;
    std::string path_;
    int fd_ = -1;
    size_t byte_len_ = 0;
    int n_grm_ = 0;
    size_t band_byte_budget_ = 0;
    bool kp_is_identity_ = false;
    bool is_fp16_ = false;
    size_t elem_bytes_ = sizeof(float);
    off_t body_offset_ = 0;
    GrmFp16Header f16_hdr_{};
    Eigen::VectorXf diag_f32_;
};

// Mean SNP count for Marchenko-Pastur Woodbury auto-k:
//   - Native .grm.f16: O(1) directly from 128-byte header (zero .grm.N.bin access).
//   - Classic .grm.N.bin: strictly 1-thread stratified sampling (1 contiguous
//     head pread of rows 0..511 + <= 256 evenly spaced single-thread 4-byte
//     preads across [512, n_grm); exact for zero-missing, <0.02% SE for missing;
//     eliminates the 16-thread 498k-RPC Lustre I/O storm).
inline double read_grm_N_mean(const std::string& prefix, int n_grm, bool prefer_fp16) {
    if (n_grm <= 0) return 0.0;

    const std::string f16_path = prefix + ".grm.f16";
    const std::string bin_path = prefix + ".grm.bin";
    const bool has_f16 = detail::is_regular_file(f16_path);
    const bool has_bin = detail::is_regular_file(bin_path);
    if (has_f16 && (!has_bin || prefer_fp16)) {
        const int fd = open_readonly_checked(f16_path);
        GrmFp16Header hdr{};
        read_and_validate_grm_f16_header(fd, f16_path, hdr);
        ::close(fd);
        if (static_cast<int>(hdr.n) != n_grm) {
            LOGGER.w(0, "GRM header [" + f16_path + "] sample count mismatch (" +
                        std::to_string(hdr.n) + " vs " + std::to_string(n_grm) + ").");
            return 0.0;
        }
        return hdr.m_snps_mean;
    }

    const size_t tri = static_cast<size_t>(n_grm) * (n_grm + 1) / 2;
    const size_t byte_len = tri * sizeof(float);
    const std::string n_path = prefix + ".grm.N.bin";

    const int nfd = ::open(n_path.c_str(), O_RDONLY);
    if (nfd == -1) {
        LOGGER.w(0, "GRM N file [" + n_path + "] not found; SNP count "
                    "unavailable (affects --reml-woodbury auto-k).");
        return 0.0;
    }
    struct stat st{};
    if (::fstat(nfd, &st) != 0 || st.st_size < 0 || static_cast<size_t>(st.st_size) < byte_len) {
        ::close(nfd);
        LOGGER.w(0, "GRM N file [" + n_path + "] has unexpected size; "
                    "SNP count unavailable (affects --reml-woodbury auto-k).");
        return 0.0;
    }

    double head_sum = 0.0;
    bool read_failed = false;

    const int c_end = std::min(n_grm, 512);
    if (c_end > 0) {
        const size_t head_elems = static_cast<size_t>(c_end) * (c_end + 1) / 2;
        AlignedRawBuffer<float, 64> head_buf(head_elems);
        try {
            gcta_io::read_exact_at_throwing(nfd, head_buf.data(), head_elems * sizeof(float), 0, n_path);
            for (int i = 0; i < c_end; ++i) {
                const size_t diag_idx = static_cast<size_t>(i) * (i + 1) / 2 + static_cast<size_t>(i);
                head_sum += static_cast<double>(head_buf[diag_idx]);
            }
        } catch (...) {
            read_failed = true;
        }
    }

    double result = 0.0;
    if (!read_failed) {
        if (c_end >= n_grm) {
            result = head_sum / static_cast<double>(n_grm);
        } else {
            const int rem = n_grm - c_end;
            const int k_samples = std::min(rem, 256);
            double tail_sum = 0.0;
            for (int s = 0; s < k_samples; ++s) {
                const int i = (k_samples == 1)
                    ? (n_grm - 1)
                    : (c_end + static_cast<int>((static_cast<int64_t>(s) * (rem - 1)) / (k_samples - 1)));
                const size_t diag_idx = static_cast<size_t>(i) * (i + 1) / 2 + static_cast<size_t>(i);
                float v = 0.0f;
                ssize_t r = 0;
                do {
                    r = ::pread(nfd, &v, sizeof(v), static_cast<off_t>(diag_idx * sizeof(float)));
                } while (r < 0 && errno == EINTR);
                if (r != static_cast<ssize_t>(sizeof(v))) {
                    read_failed = true;
                    break;
                }
                tail_sum += static_cast<double>(v);
            }
            if (!read_failed) {
                const double tail_mean = tail_sum / static_cast<double>(k_samples);
                result = (head_sum + tail_mean * static_cast<double>(rem)) / static_cast<double>(n_grm);
            }
        }
    }
    ::close(nfd);

    if (read_failed) {
        LOGGER.w(0, "pread() failed while reading [" + n_path + "]'s diagonal; "
                    "SNP count unavailable (affects --reml-woodbury auto-k).");
        return 0.0;
    }
    return result;
}

// Splits a --grm-chunked-budget total (GB) into the slice reserved for
// ChunkedGrmReader's double-buffered row-band cache. Because column tiles are
// capped at 4096 (<= 64-128 MB tile scratch), up to 2.0 GB (1.0 GB per buffer)
// can go to row bands while keeping total memory strictly <= total_budget_gb.
inline double chunked_reader_reserved_gb(double total_budget_gb) {
    return std::min(2.0, total_budget_gb * 0.50);
}

struct ChunkedGrmHandle {
    gcta_chunked::TileReader reader;
    gcta_chunked::TileReaderF32 reader_f32;
    std::shared_ptr<const ChunkedGrmReader> file;
    double m_snps = 0.0;
    double trace_K = 0.0;
    double trace_K2 = 0.0;
    bool has_traces = false;
    GrmDiskFormat disk_format = GrmDiskFormat::F32Bin;
    int chunk_rows = 0;
};

inline ChunkedGrmHandle make_chunked_grm_reader(
    const std::string& prefix,
    const std::vector<std::string>& analysis_ids,
    double budget_gb,
    int k_ext = 0,
    const std::string& feature_flag = "--grm-chunked-budget",
    bool need_m_snps = true,
    const std::vector<std::string>* preloaded_grm_ids = nullptr,
    bool prefer_fp16 = false)
{
    const ResolvedGrmInput resolved = resolve_grm_input(prefix, prefer_fp16);
    std::vector<std::string> grm_ids_storage;
    const std::vector<std::string>& grm_ids = preloaded_grm_ids
        ? *preloaded_grm_ids
        : (grm_ids_storage = Pheno::read_sublist(resolved.id_path));
    const int n_grm = static_cast<int>(grm_ids.size());
    const int n = static_cast<int>(analysis_ids.size());

    std::vector<int> kp = match_ids_to_grm(analysis_ids, grm_ids);
    for (int i = 0; i < n; ++i) {
        if (kp[static_cast<size_t>(i)] < 0)
            LOGGER.e(0, feature_flag + ": individual [" + analysis_ids[static_cast<size_t>(i)] +
                        "] not found in GRM [" + resolved.id_path + "].");
    }

    const double reader_reserved_gb = chunked_reader_reserved_gb(budget_gb);
    const int chunk_rows = gcta_chunked::solve_chunk_rows(n, budget_gb, k_ext, reader_reserved_gb);
    if (chunk_rows < 1)
        LOGGER.e(0, feature_flag + ": budget=" + std::to_string(budget_gb) +
                    "GB is too small: reserving " + std::to_string(reader_reserved_gb) +
                    "GB for the reader's row-band cache leaves no room for a single row-chunk "
                    "(n=" + std::to_string(n) +
                    (k_ext > 0 ? ", k_ext=" + std::to_string(k_ext) : "") +
                    "); raise the budget.");
    LOGGER.i(0, feature_flag + ": budget=" + std::to_string(budget_gb) + "GB (" +
                std::to_string(reader_reserved_gb) + "GB reader cache + " +
                std::to_string(budget_gb - reader_reserved_gb) + "GB tiles" +
                (k_ext > 0 ? ", k_ext up to " + std::to_string(k_ext) : "") +
                ") -> " + std::to_string(chunk_rows) + "-col tiles from [" + resolved.data_path +
                "], not loaded densely.");

    ChunkedGrmHandle handle;
    handle.chunk_rows = chunk_rows;
    handle.disk_format = resolved.format;

    auto file = std::make_shared<ChunkedGrmReader>(
        resolved.data_path, std::move(kp), n_grm,
        static_cast<size_t>(reader_reserved_gb * 1e9),
        &grm_ids);
    handle.file = file;

    if (file->is_fp16()) {
        const GrmFp16Header& hdr = file->f16_header();
        if (need_m_snps) {
            handle.m_snps = hdr.m_snps_mean;
        } else {
            handle.m_snps = -1.0;
        }
        if (file->kp_is_identity() && (hdr.flags & kGrmF16FlagHasTraces) != 0) {
            handle.trace_K = hdr.trace_K;
            handle.trace_K2 = hdr.trace_K2;
            handle.has_traces = true;
        }
    } else if (need_m_snps) {
        handle.m_snps = read_grm_N_mean(prefix, n_grm, prefer_fp16);
    } else {
        LOGGER.i(0, "Skipping SNP count read from .grm.N.bin.");
        handle.m_snps = -1.0;
    }

    handle.reader.read_fn = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXd> {
        return file->read_tile(rs, re, cs, ce);
    };
    handle.reader.read_T_fn = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXd> {
        return file->read_tile_T(rs, re, cs, ce);
    };
    handle.reader.read_diagonal_fn = [file]() -> Eigen::VectorXd {
        return file->read_diagonal();
    };
    handle.reader.row_bands_fn = [file](int max_rows) -> std::vector<int> {
        return file->compute_row_bands(max_rows);
    };
    handle.reader.begin_sweep_fn = [file](const std::vector<int>& bands) {
        return file->begin_sweep(bands);
    };
    handle.reader.release_fn = [file]() {
        file->release_buffers();
    };

    handle.reader_f32.read_fn = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXf> {
        return file->read_tile_f32(rs, re, cs, ce);
    };
    handle.reader_f32.read_T_fn = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXf> {
        return file->read_tile_f32_T(rs, re, cs, ce);
    };
    handle.reader_f32.read_diagonal_fn = [file]() -> Eigen::VectorXd {
        return file->read_diagonal();
    };
    handle.reader_f32.row_bands_fn = [file](int max_rows) -> std::vector<int> {
        return file->compute_row_bands(max_rows);
    };
    handle.reader_f32.begin_sweep_fn = [file](const std::vector<int>& bands) {
        return file->begin_sweep(bands);
    };
    handle.reader_f32.release_fn = [file]() {
        file->release_buffers();
    };

    return handle;
}

// ---------------------------------------------------------------------------
// Opt-in / native packed lower-triangular IEEE fp16 in-memory GRM (--grm-fp16)
// ---------------------------------------------------------------------------

class PackedGrmFp16 {
public:
    PackedGrmFp16(int n, AlignedUniquePtr<GrmFp16, 64> packed, Eigen::VectorXf diag_f32,
                  double trace_K, double trace_K2)
        : n_(n),
          packed_(std::move(packed)),
          diag_f32_(std::move(diag_f32)),
          trace_K_(trace_K),
          trace_K2_(trace_K2) {}

    int n() const noexcept { return n_; }
    double trace_K() const noexcept { return trace_K_; }
    double trace_K2() const noexcept { return trace_K2_; }
    const Eigen::VectorXf& diagonal_f32() const noexcept { return diag_f32_; }

    Eigen::VectorXd read_diagonal() const {
        return diag_f32_.cast<double>();
    }

    void release_scratch() const noexcept {
        tile_scratch_f32_.resize(0, 0);
        tile_scratch_f64_.resize(0, 0);
        tile_scratch_f32_T_.resize(0, 0);
        tile_scratch_f64_T_.resize(0, 0);
    }

    // Fast-path transposed float32 tile: shape (ce-cs) x (re-rs) ColMajor so
    // each fp16 row unpacks straight into a contiguous column via SIMD vcvtph2ps
    // (eliminating 16 KiB power-of-2 L1D/STLB stride scatter). Diagonal tiles
    // have their UPPER triangle valid with exact float32 diagonal entries.
    Eigen::Ref<const Eigen::MatrixXf> read_tile_f32_T(int rs, int re, int cs, int ce) const {
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_f32_T_.rows() < tile_cols || tile_scratch_f32_T_.cols() < tile_rows) {
            const Eigen::Index want_rows = std::max<Eigen::Index>(tile_scratch_f32_T_.rows(), static_cast<Eigen::Index>(tile_cols));
            const Eigen::Index padded_rows = ((want_rows & 1023) == 0) ? (want_rows + 16) : want_rows;
            tile_scratch_f32_T_.resize(
                padded_rows,
                std::max<Eigen::Index>(tile_scratch_f32_T_.cols(), static_cast<Eigen::Index>(tile_rows)));
        }
        auto tile_T = tile_scratch_f32_T_.topLeftCorner(tile_cols, tile_rows);
        const bool diagonal_tile = (rs == cs);
        const GrmFp16* base = packed_.get();

        if (diagonal_tile) {
            const int half_pairs = (tile_rows + 1) / 2;
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(tile_rows) * static_cast<size_t>(tile_rows) >= 32768)
            for (int p = 0; p < half_pairs; ++p) {
                const int lps[2] = { p, tile_rows - 1 - p };
                const int cnt = (lps[0] == lps[1]) ? 1 : 2;
                for (int k = 0; k < cnt; ++k) {
                    const int lp = lps[k];
                    const int gi = rs + lp;
                    const GrmFp16* __restrict__ src = base + static_cast<size_t>(gi) * (gi + 1) / 2 + cs;
                    float* __restrict__ dst = tile_T.col(lp).data();
                    #pragma omp simd
                    for (int lq = 0; lq < lp; ++lq) {
                        dst[lq] = fp16_to_f32(src[lq]);
                    }
                    dst[lp] = diag_f32_[gi];
                }
            }
        } else {
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(tile_rows) * static_cast<size_t>(tile_cols) >= 16384)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = rs + lp;
                const GrmFp16* __restrict__ src = base + static_cast<size_t>(gi) * (gi + 1) / 2 + cs;
                float* __restrict__ dst = tile_T.col(lp).data();
                #pragma omp simd
                for (int lq = 0; lq < tile_cols; ++lq) {
                    dst[lq] = fp16_to_f32(src[lq]);
                }
            }
        }
        return tile_T;
    }

    // Fast-path transposed double tile: shape (ce-cs) x (re-rs) ColMajor.
    Eigen::Ref<const Eigen::MatrixXd> read_tile_T(int rs, int re, int cs, int ce) const {
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_f64_T_.rows() < tile_cols || tile_scratch_f64_T_.cols() < tile_rows) {
            const Eigen::Index want_rows = std::max<Eigen::Index>(tile_scratch_f64_T_.rows(), static_cast<Eigen::Index>(tile_cols));
            const Eigen::Index padded_rows = ((want_rows & 511) == 0) ? (want_rows + 8) : want_rows;
            tile_scratch_f64_T_.resize(
                padded_rows,
                std::max<Eigen::Index>(tile_scratch_f64_T_.cols(), static_cast<Eigen::Index>(tile_rows)));
        }
        auto tile_T = tile_scratch_f64_T_.topLeftCorner(tile_cols, tile_rows);
        const bool diagonal_tile = (rs == cs);
        const GrmFp16* base = packed_.get();

        if (diagonal_tile) {
            const int half_pairs = (tile_rows + 1) / 2;
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(tile_rows) * static_cast<size_t>(tile_rows) >= 32768)
            for (int p = 0; p < half_pairs; ++p) {
                const int lps[2] = { p, tile_rows - 1 - p };
                const int cnt = (lps[0] == lps[1]) ? 1 : 2;
                for (int k = 0; k < cnt; ++k) {
                    const int lp = lps[k];
                    const int gi = rs + lp;
                    const GrmFp16* __restrict__ src = base + static_cast<size_t>(gi) * (gi + 1) / 2 + cs;
                    double* __restrict__ dst = tile_T.col(lp).data();
                    #pragma omp simd
                    for (int lq = 0; lq < lp; ++lq) {
                        dst[lq] = static_cast<double>(fp16_to_f32(src[lq]));
                    }
                    dst[lp] = static_cast<double>(diag_f32_[gi]);
                }
            }
        } else {
            #pragma omp parallel for schedule(static) if (static_cast<size_t>(tile_rows) * static_cast<size_t>(tile_cols) >= 16384)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = rs + lp;
                const GrmFp16* __restrict__ src = base + static_cast<size_t>(gi) * (gi + 1) / 2 + cs;
                double* __restrict__ dst = tile_T.col(lp).data();
                #pragma omp simd
                for (int lq = 0; lq < tile_cols; ++lq) {
                    dst[lq] = static_cast<double>(fp16_to_f32(src[lq]));
                }
            }
        }
        return tile_T;
    }

    // Fast-path: direct fused symmetric matvec on packed FP16 storage for narrow X.
    // Avoids all intermediate FP32/BF16 tiles and GEMM packing barriers by streaming
    // strictly-lower FP16 rows once per panel in L2-blocked j-slabs.
    void symm_f32_fused(const Eigen::Ref<const Eigen::MatrixXf>& X,
                        Eigen::MatrixXf& Y,
                        double* trace_K_out = nullptr,
                        double* trace_K2_out = nullptr) const;

    void set_fused_j_block(int jb) noexcept {
        fused_j_block_ = std::max(64, (jb + 15) & ~15);
    }
    int fused_j_block() const noexcept { return fused_j_block_; }

    // Legacy non-transposed float32 tile (shape (re-rs) x (ce-cs)).
    Eigen::Ref<const Eigen::MatrixXf> read_tile_f32(int rs, int re, int cs, int ce) const {
        const auto tile_T = read_tile_f32_T(rs, re, cs, ce);
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_f32_.rows() < tile_rows || tile_scratch_f32_.cols() < tile_cols) {
            tile_scratch_f32_.resize(std::max<Eigen::Index>(tile_scratch_f32_.rows(), tile_rows),
                                     std::max<Eigen::Index>(tile_scratch_f32_.cols(), tile_cols));
        }
        auto tile = tile_scratch_f32_.topLeftCorner(tile_rows, tile_cols);
        const bool diagonal_tile = (rs == cs);
        constexpr int kB = 64;
        #pragma omp parallel for schedule(guided) if (tile_rows >= 128)
        for (int c0 = 0; c0 < tile_cols; c0 += kB) {
            const int c1 = std::min(tile_cols, c0 + kB);
            const int r_begin = diagonal_tile ? c0 : 0;
            for (int r0 = r_begin; r0 < tile_rows; r0 += kB) {
                const int r1 = std::min(tile_rows, r0 + kB);
                for (int c = c0; c < c1; ++c) {
                    const int r_lo = diagonal_tile ? std::max(r0, c) : r0;
                    float* dst = tile.col(c).data();
                    for (int r = r_lo; r < r1; ++r) {
                        dst[r] = tile_T(c, r);
                    }
                }
            }
        }
        return tile;
    }

    // Legacy non-transposed double tile (shape (re-rs) x (ce-cs)).
    Eigen::Ref<const Eigen::MatrixXd> read_tile(int rs, int re, int cs, int ce) const {
        const auto tile_T = read_tile_T(rs, re, cs, ce);
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_f64_.rows() < tile_rows || tile_scratch_f64_.cols() < tile_cols) {
            tile_scratch_f64_.resize(std::max<Eigen::Index>(tile_scratch_f64_.rows(), tile_rows),
                                     std::max<Eigen::Index>(tile_scratch_f64_.cols(), tile_cols));
        }
        auto tile = tile_scratch_f64_.topLeftCorner(tile_rows, tile_cols);
        const bool diagonal_tile = (rs == cs);
        constexpr int kB = 64;
        #pragma omp parallel for schedule(guided) if (tile_rows >= 128)
        for (int c0 = 0; c0 < tile_cols; c0 += kB) {
            const int c1 = std::min(tile_cols, c0 + kB);
            const int r_begin = diagonal_tile ? c0 : 0;
            for (int r0 = r_begin; r0 < tile_rows; r0 += kB) {
                const int r1 = std::min(tile_rows, r0 + kB);
                for (int c = c0; c < c1; ++c) {
                    const int r_lo = diagonal_tile ? std::max(r0, c) : r0;
                    double* dst = tile.col(c).data();
                    for (int r = r_lo; r < r1; ++r) {
                        dst[r] = tile_T(c, r);
                    }
                }
            }
        }
        return tile;
    }

private:
    int n_ = 0;
    AlignedUniquePtr<GrmFp16, 64> packed_;
    Eigen::VectorXf diag_f32_;
    double trace_K_ = 0.0;
    double trace_K2_ = 0.0;
    mutable int fused_j_block_ = 1024;
    mutable Eigen::MatrixXf tile_scratch_f32_;
    mutable Eigen::MatrixXd tile_scratch_f64_;
    mutable Eigen::MatrixXf tile_scratch_f32_T_;
    mutable Eigen::MatrixXd tile_scratch_f64_T_;
};

namespace detail {

inline bool packed_f16_has_avx512() noexcept {
#if GCTA_CPU_x86 && (defined(__GNUC__) || defined(__clang__))
  #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__F16C__) && defined(__FMA__)
    return true;
  #else
    static const bool has =
        __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("f16c") &&
        __builtin_cpu_supports("fma");
    return has;
  #endif
#else
    return false;
#endif
}

template <int P>
inline void packed_f16_kernel_1row_portable(
    const GrmFp16* __restrict__ src,
    int j_len,
    const float* __restrict__ X_j,
    const float* __restrict__ X_i,
    float* __restrict__ y_row,
    float* __restrict__ y_col) noexcept
{
    alignas(64) float yr[P];
    alignas(64) float xi[P];
    for (int d = 0; d < P; ++d) {
        yr[d] = y_row[d];
        xi[d] = X_i[d];
    }
    for (int jj = 0; jj < j_len; ++jj) {
        const float kv = fp16_to_f32(src[jj]);
        const float* __restrict__ xj = X_j + static_cast<size_t>(jj) * P;
        float* __restrict__ yc = y_col + static_cast<size_t>(jj) * P;
        #pragma omp simd
        for (int d = 0; d < P; ++d) {
            yr[d] += kv * xj[d];
            yc[d] += kv * xi[d];
        }
    }
    for (int d = 0; d < P; ++d) {
        y_row[d] = yr[d];
    }
}

template <int P>
inline void packed_f16_kernel_4row_portable(
    const GrmFp16* __restrict__ s0,
    const GrmFp16* __restrict__ s1,
    const GrmFp16* __restrict__ s2,
    const GrmFp16* __restrict__ s3,
    int j_len,
    const float* __restrict__ X_j,
    const float* __restrict__ X_i_base,
    float* __restrict__ y_row_base,
    float* __restrict__ y_col) noexcept
{
    alignas(64) float yr0[P], yr1[P], yr2[P], yr3[P];
    alignas(64) float xi0[P], xi1[P], xi2[P], xi3[P];
    for (int d = 0; d < P; ++d) {
        yr0[d] = y_row_base[0 * P + d];
        yr1[d] = y_row_base[1 * P + d];
        yr2[d] = y_row_base[2 * P + d];
        yr3[d] = y_row_base[3 * P + d];
        xi0[d] = X_i_base[0 * P + d];
        xi1[d] = X_i_base[1 * P + d];
        xi2[d] = X_i_base[2 * P + d];
        xi3[d] = X_i_base[3 * P + d];
    }
    for (int jj = 0; jj < j_len; ++jj) {
        const float k0 = fp16_to_f32(s0[jj]);
        const float k1 = fp16_to_f32(s1[jj]);
        const float k2 = fp16_to_f32(s2[jj]);
        const float k3 = fp16_to_f32(s3[jj]);
        const float* __restrict__ xj = X_j + static_cast<size_t>(jj) * P;
        float* __restrict__ yc = y_col + static_cast<size_t>(jj) * P;
        #pragma omp simd
        for (int d = 0; d < P; ++d) {
            const float xjd = xj[d];
            yr0[d] += k0 * xjd;
            yr1[d] += k1 * xjd;
            yr2[d] += k2 * xjd;
            yr3[d] += k3 * xjd;
            yc[d]  += k0 * xi0[d] + k1 * xi1[d] + k2 * xi2[d] + k3 * xi3[d];
        }
    }
    for (int d = 0; d < P; ++d) {
        y_row_base[0 * P + d] = yr0[d];
        y_row_base[1 * P + d] = yr1[d];
        y_row_base[2 * P + d] = yr2[d];
        y_row_base[3 * P + d] = yr3[d];
    }
}

#if GCTA_CPU_x86 && (defined(__GNUC__) || defined(__clang__))
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__F16C__) && defined(__FMA__)
#define GCTA_PCA_AVX512_ATTR
#else
#define GCTA_PCA_AVX512_ATTR __attribute__((target("avx512f,avx512bw,avx512vl,f16c,fma"), noinline))
#endif

GCTA_PCA_AVX512_ATTR
inline void packed_f16_kernel_1row_avx512_16(
    const GrmFp16* __restrict__ src,
    int j_len,
    const float* __restrict__ X_j,
    const float* __restrict__ X_i,
    float* __restrict__ y_row,
    float* __restrict__ y_col) noexcept
{
    __m512 yr = _mm512_load_ps(y_row);
    const __m512 xi = _mm512_load_ps(X_i);
    int jj = 0;
    alignas(64) float kv[16];
    for (; jj + 16 <= j_len; jj += 16) {
        const __m512 kvec = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + jj)));
        _mm512_store_ps(kv, kvec);
        for (int t = 0; t < 16; ++t) {
            const size_t off = static_cast<size_t>(jj + t) * 16;
            const __m512 kb = _mm512_set1_ps(kv[t]);
            const __m512 xj = _mm512_load_ps(X_j + off);
            __m512 yc = _mm512_load_ps(y_col + off);
            yr = _mm512_fmadd_ps(kb, xj, yr);
            yc = _mm512_fmadd_ps(kb, xi, yc);
            _mm512_store_ps(y_col + off, yc);
        }
    }
    for (; jj < j_len; ++jj) {
        const __m512 kb = _mm512_set1_ps(fp16_to_f32(src[jj]));
        const size_t off = static_cast<size_t>(jj) * 16;
        const __m512 xj = _mm512_load_ps(X_j + off);
        __m512 yc = _mm512_load_ps(y_col + off);
        yr = _mm512_fmadd_ps(kb, xj, yr);
        yc = _mm512_fmadd_ps(kb, xi, yc);
        _mm512_store_ps(y_col + off, yc);
    }
    _mm512_store_ps(y_row, yr);
}

GCTA_PCA_AVX512_ATTR
inline void packed_f16_kernel_1row_avx512_32(
    const GrmFp16* __restrict__ src,
    int j_len,
    const float* __restrict__ X_j,
    const float* __restrict__ X_i,
    float* __restrict__ y_row,
    float* __restrict__ y_col) noexcept
{
    __m512 yr_lo = _mm512_load_ps(y_row);
    __m512 yr_hi = _mm512_load_ps(y_row + 16);
    const __m512 xi_lo = _mm512_load_ps(X_i);
    const __m512 xi_hi = _mm512_load_ps(X_i + 16);
    int jj = 0;
    alignas(64) float kv[16];
    for (; jj + 16 <= j_len; jj += 16) {
        const __m512 kvec = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + jj)));
        _mm512_store_ps(kv, kvec);
        for (int t = 0; t < 16; ++t) {
            const size_t off = static_cast<size_t>(jj + t) * 32;
            const __m512 kb = _mm512_set1_ps(kv[t]);
            const __m512 xj_lo = _mm512_load_ps(X_j + off);
            const __m512 xj_hi = _mm512_load_ps(X_j + off + 16);
            __m512 yc_lo = _mm512_load_ps(y_col + off);
            __m512 yc_hi = _mm512_load_ps(y_col + off + 16);
            yr_lo = _mm512_fmadd_ps(kb, xj_lo, yr_lo);
            yr_hi = _mm512_fmadd_ps(kb, xj_hi, yr_hi);
            yc_lo = _mm512_fmadd_ps(kb, xi_lo, yc_lo);
            yc_hi = _mm512_fmadd_ps(kb, xi_hi, yc_hi);
            _mm512_store_ps(y_col + off, yc_lo);
            _mm512_store_ps(y_col + off + 16, yc_hi);
        }
    }
    for (; jj < j_len; ++jj) {
        const __m512 kb = _mm512_set1_ps(fp16_to_f32(src[jj]));
        const size_t off = static_cast<size_t>(jj) * 32;
        const __m512 xj_lo = _mm512_load_ps(X_j + off);
        const __m512 xj_hi = _mm512_load_ps(X_j + off + 16);
        __m512 yc_lo = _mm512_load_ps(y_col + off);
        __m512 yc_hi = _mm512_load_ps(y_col + off + 16);
        yr_lo = _mm512_fmadd_ps(kb, xj_lo, yr_lo);
        yr_hi = _mm512_fmadd_ps(kb, xj_hi, yr_hi);
        yc_lo = _mm512_fmadd_ps(kb, xi_lo, yc_lo);
        yc_hi = _mm512_fmadd_ps(kb, xi_hi, yc_hi);
        _mm512_store_ps(y_col + off, yc_lo);
        _mm512_store_ps(y_col + off + 16, yc_hi);
    }
    _mm512_store_ps(y_row, yr_lo);
    _mm512_store_ps(y_row + 16, yr_hi);
}

GCTA_PCA_AVX512_ATTR
inline void packed_f16_kernel_4row_avx512_16(
    const GrmFp16* __restrict__ s0,
    const GrmFp16* __restrict__ s1,
    const GrmFp16* __restrict__ s2,
    const GrmFp16* __restrict__ s3,
    int j_len,
    const float* __restrict__ X_j,
    const float* __restrict__ X_i_base,
    float* __restrict__ y_row_base,
    float* __restrict__ y_col) noexcept
{
    __m512 yr0 = _mm512_load_ps(y_row_base + 0 * 16);
    __m512 yr1 = _mm512_load_ps(y_row_base + 1 * 16);
    __m512 yr2 = _mm512_load_ps(y_row_base + 2 * 16);
    __m512 yr3 = _mm512_load_ps(y_row_base + 3 * 16);
    const __m512 xi0 = _mm512_load_ps(X_i_base + 0 * 16);
    const __m512 xi1 = _mm512_load_ps(X_i_base + 1 * 16);
    const __m512 xi2 = _mm512_load_ps(X_i_base + 2 * 16);
    const __m512 xi3 = _mm512_load_ps(X_i_base + 3 * 16);

    int jj = 0;
    alignas(64) float kv0[16], kv1[16], kv2[16], kv3[16];
    for (; jj + 16 <= j_len; jj += 16) {
        _mm512_store_ps(kv0, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s0 + jj))));
        _mm512_store_ps(kv1, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s1 + jj))));
        _mm512_store_ps(kv2, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s2 + jj))));
        _mm512_store_ps(kv3, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s3 + jj))));
        for (int t = 0; t < 16; ++t) {
            const size_t off = static_cast<size_t>(jj + t) * 16;
            const __m512 xj = _mm512_load_ps(X_j + off);
            __m512 yc = _mm512_load_ps(y_col + off);
            const __m512 k0 = _mm512_set1_ps(kv0[t]);
            yr0 = _mm512_fmadd_ps(k0, xj, yr0);
            yc  = _mm512_fmadd_ps(k0, xi0, yc);
            const __m512 k1 = _mm512_set1_ps(kv1[t]);
            yr1 = _mm512_fmadd_ps(k1, xj, yr1);
            yc  = _mm512_fmadd_ps(k1, xi1, yc);
            const __m512 k2 = _mm512_set1_ps(kv2[t]);
            yr2 = _mm512_fmadd_ps(k2, xj, yr2);
            yc  = _mm512_fmadd_ps(k2, xi2, yc);
            const __m512 k3 = _mm512_set1_ps(kv3[t]);
            yr3 = _mm512_fmadd_ps(k3, xj, yr3);
            yc  = _mm512_fmadd_ps(k3, xi3, yc);
            _mm512_store_ps(y_col + off, yc);
        }
    }
    for (; jj < j_len; ++jj) {
        const size_t off = static_cast<size_t>(jj) * 16;
        const __m512 xj = _mm512_load_ps(X_j + off);
        __m512 yc = _mm512_load_ps(y_col + off);
        const __m512 k0 = _mm512_set1_ps(fp16_to_f32(s0[jj]));
        yr0 = _mm512_fmadd_ps(k0, xj, yr0);
        yc  = _mm512_fmadd_ps(k0, xi0, yc);
        const __m512 k1 = _mm512_set1_ps(fp16_to_f32(s1[jj]));
        yr1 = _mm512_fmadd_ps(k1, xj, yr1);
        yc  = _mm512_fmadd_ps(k1, xi1, yc);
        const __m512 k2 = _mm512_set1_ps(fp16_to_f32(s2[jj]));
        yr2 = _mm512_fmadd_ps(k2, xj, yr2);
        yc  = _mm512_fmadd_ps(k2, xi2, yc);
        const __m512 k3 = _mm512_set1_ps(fp16_to_f32(s3[jj]));
        yr3 = _mm512_fmadd_ps(k3, xj, yr3);
        yc  = _mm512_fmadd_ps(k3, xi3, yc);
        _mm512_store_ps(y_col + off, yc);
    }
    _mm512_store_ps(y_row_base + 0 * 16, yr0);
    _mm512_store_ps(y_row_base + 1 * 16, yr1);
    _mm512_store_ps(y_row_base + 2 * 16, yr2);
    _mm512_store_ps(y_row_base + 3 * 16, yr3);
}

GCTA_PCA_AVX512_ATTR
inline void packed_f16_kernel_4row_avx512_32(
    const GrmFp16* __restrict__ s0,
    const GrmFp16* __restrict__ s1,
    const GrmFp16* __restrict__ s2,
    const GrmFp16* __restrict__ s3,
    int j_len,
    const float* __restrict__ X_j,
    const float* __restrict__ X_i_base,
    float* __restrict__ y_row_base,
    float* __restrict__ y_col) noexcept
{
    __m512 yr0_lo = _mm512_load_ps(y_row_base + 0 * 32);
    __m512 yr0_hi = _mm512_load_ps(y_row_base + 0 * 32 + 16);
    __m512 yr1_lo = _mm512_load_ps(y_row_base + 1 * 32);
    __m512 yr1_hi = _mm512_load_ps(y_row_base + 1 * 32 + 16);
    __m512 yr2_lo = _mm512_load_ps(y_row_base + 2 * 32);
    __m512 yr2_hi = _mm512_load_ps(y_row_base + 2 * 32 + 16);
    __m512 yr3_lo = _mm512_load_ps(y_row_base + 3 * 32);
    __m512 yr3_hi = _mm512_load_ps(y_row_base + 3 * 32 + 16);
    const __m512 xi0_lo = _mm512_load_ps(X_i_base + 0 * 32);
    const __m512 xi0_hi = _mm512_load_ps(X_i_base + 0 * 32 + 16);
    const __m512 xi1_lo = _mm512_load_ps(X_i_base + 1 * 32);
    const __m512 xi1_hi = _mm512_load_ps(X_i_base + 1 * 32 + 16);
    const __m512 xi2_lo = _mm512_load_ps(X_i_base + 2 * 32);
    const __m512 xi2_hi = _mm512_load_ps(X_i_base + 2 * 32 + 16);
    const __m512 xi3_lo = _mm512_load_ps(X_i_base + 3 * 32);
    const __m512 xi3_hi = _mm512_load_ps(X_i_base + 3 * 32 + 16);

    int jj = 0;
    alignas(64) float kv0[16], kv1[16], kv2[16], kv3[16];
    for (; jj + 16 <= j_len; jj += 16) {
        _mm512_store_ps(kv0, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s0 + jj))));
        _mm512_store_ps(kv1, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s1 + jj))));
        _mm512_store_ps(kv2, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s2 + jj))));
        _mm512_store_ps(kv3, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s3 + jj))));
        for (int t = 0; t < 16; ++t) {
            const size_t off = static_cast<size_t>(jj + t) * 32;
            const __m512 xj_lo = _mm512_load_ps(X_j + off);
            const __m512 xj_hi = _mm512_load_ps(X_j + off + 16);
            __m512 yc_lo = _mm512_load_ps(y_col + off);
            __m512 yc_hi = _mm512_load_ps(y_col + off + 16);
            const __m512 k0 = _mm512_set1_ps(kv0[t]);
            yr0_lo = _mm512_fmadd_ps(k0, xj_lo, yr0_lo);
            yr0_hi = _mm512_fmadd_ps(k0, xj_hi, yr0_hi);
            yc_lo  = _mm512_fmadd_ps(k0, xi0_lo, yc_lo);
            yc_hi  = _mm512_fmadd_ps(k0, xi0_hi, yc_hi);
            const __m512 k1 = _mm512_set1_ps(kv1[t]);
            yr1_lo = _mm512_fmadd_ps(k1, xj_lo, yr1_lo);
            yr1_hi = _mm512_fmadd_ps(k1, xj_hi, yr1_hi);
            yc_lo  = _mm512_fmadd_ps(k1, xi1_lo, yc_lo);
            yc_hi  = _mm512_fmadd_ps(k1, xi1_hi, yc_hi);
            const __m512 k2 = _mm512_set1_ps(kv2[t]);
            yr2_lo = _mm512_fmadd_ps(k2, xj_lo, yr2_lo);
            yr2_hi = _mm512_fmadd_ps(k2, xj_hi, yr2_hi);
            yc_lo  = _mm512_fmadd_ps(k2, xi2_lo, yc_lo);
            yc_hi  = _mm512_fmadd_ps(k2, xi2_hi, yc_hi);
            const __m512 k3 = _mm512_set1_ps(kv3[t]);
            yr3_lo = _mm512_fmadd_ps(k3, xj_lo, yr3_lo);
            yr3_hi = _mm512_fmadd_ps(k3, xj_hi, yr3_hi);
            yc_lo  = _mm512_fmadd_ps(k3, xi3_lo, yc_lo);
            yc_hi  = _mm512_fmadd_ps(k3, xi3_hi, yc_hi);
            _mm512_store_ps(y_col + off, yc_lo);
            _mm512_store_ps(y_col + off + 16, yc_hi);
        }
    }
    for (; jj < j_len; ++jj) {
        const size_t off = static_cast<size_t>(jj) * 32;
        const __m512 xj_lo = _mm512_load_ps(X_j + off);
        const __m512 xj_hi = _mm512_load_ps(X_j + off + 16);
        __m512 yc_lo = _mm512_load_ps(y_col + off);
        __m512 yc_hi = _mm512_load_ps(y_col + off + 16);
        const __m512 k0 = _mm512_set1_ps(fp16_to_f32(s0[jj]));
        yr0_lo = _mm512_fmadd_ps(k0, xj_lo, yr0_lo);
        yr0_hi = _mm512_fmadd_ps(k0, xj_hi, yr0_hi);
        yc_lo  = _mm512_fmadd_ps(k0, xi0_lo, yc_lo);
        yc_hi  = _mm512_fmadd_ps(k0, xi0_hi, yc_hi);
        const __m512 k1 = _mm512_set1_ps(fp16_to_f32(s1[jj]));
        yr1_lo = _mm512_fmadd_ps(k1, xj_lo, yr1_lo);
        yr1_hi = _mm512_fmadd_ps(k1, xj_hi, yr1_hi);
        yc_lo  = _mm512_fmadd_ps(k1, xi1_lo, yc_lo);
        yc_hi  = _mm512_fmadd_ps(k1, xi1_hi, yc_hi);
        const __m512 k2 = _mm512_set1_ps(fp16_to_f32(s2[jj]));
        yr2_lo = _mm512_fmadd_ps(k2, xj_lo, yr2_lo);
        yr2_hi = _mm512_fmadd_ps(k2, xj_hi, yr2_hi);
        yc_lo  = _mm512_fmadd_ps(k2, xi2_lo, yc_lo);
        yc_hi  = _mm512_fmadd_ps(k2, xi2_hi, yc_hi);
        const __m512 k3 = _mm512_set1_ps(fp16_to_f32(s3[jj]));
        yr3_lo = _mm512_fmadd_ps(k3, xj_lo, yr3_lo);
        yr3_hi = _mm512_fmadd_ps(k3, xj_hi, yr3_hi);
        yc_lo  = _mm512_fmadd_ps(k3, xi3_lo, yc_lo);
        yc_hi  = _mm512_fmadd_ps(k3, xi3_hi, yc_hi);
        _mm512_store_ps(y_col + off, yc_lo);
        _mm512_store_ps(y_col + off + 16, yc_hi);
    }
    _mm512_store_ps(y_row_base + 0 * 32, yr0_lo);
    _mm512_store_ps(y_row_base + 0 * 32 + 16, yr0_hi);
    _mm512_store_ps(y_row_base + 1 * 32, yr1_lo);
    _mm512_store_ps(y_row_base + 1 * 32 + 16, yr1_hi);
    _mm512_store_ps(y_row_base + 2 * 32, yr2_lo);
    _mm512_store_ps(y_row_base + 2 * 32 + 16, yr2_hi);
    _mm512_store_ps(y_row_base + 3 * 32, yr3_lo);
    _mm512_store_ps(y_row_base + 3 * 32 + 16, yr3_hi);
}
#undef GCTA_PCA_AVX512_ATTR
#endif

template <int P>
inline void packed_f16_kernel_1row(
    const GrmFp16* __restrict__ src,
    int j_len,
    const float* __restrict__ X_j,
    const float* __restrict__ X_i,
    float* __restrict__ y_row,
    float* __restrict__ y_col,
    bool use_avx512) noexcept
{
#if GCTA_CPU_x86 && (defined(__GNUC__) || defined(__clang__))
    if (use_avx512) {
        if constexpr (P == 16) {
            packed_f16_kernel_1row_avx512_16(src, j_len, X_j, X_i, y_row, y_col);
            return;
        } else if constexpr (P == 32) {
            packed_f16_kernel_1row_avx512_32(src, j_len, X_j, X_i, y_row, y_col);
            return;
        }
    }
#else
    (void)use_avx512;
#endif
    packed_f16_kernel_1row_portable<P>(src, j_len, X_j, X_i, y_row, y_col);
}

template <int P>
inline void packed_f16_kernel_4row(
    const GrmFp16* __restrict__ s0,
    const GrmFp16* __restrict__ s1,
    const GrmFp16* __restrict__ s2,
    const GrmFp16* __restrict__ s3,
    int j_len,
    const float* __restrict__ X_j,
    const float* __restrict__ X_i_base,
    float* __restrict__ y_row_base,
    float* __restrict__ y_col,
    bool use_avx512) noexcept
{
#if GCTA_CPU_x86 && (defined(__GNUC__) || defined(__clang__))
    if (use_avx512) {
        if constexpr (P == 16) {
            packed_f16_kernel_4row_avx512_16(s0, s1, s2, s3, j_len, X_j, X_i_base, y_row_base, y_col);
            return;
        } else if constexpr (P == 32) {
            packed_f16_kernel_4row_avx512_32(s0, s1, s2, s3, j_len, X_j, X_i_base, y_row_base, y_col);
            return;
        }
    }
#else
    (void)use_avx512;
#endif
    packed_f16_kernel_4row_portable<P>(s0, s1, s2, s3, j_len, X_j, X_i_base, y_row_base, y_col);
}

template <int P>
inline void packed_f16_symm_panel_loop(
    int n,
    int cols,
    int j_block,
    const GrmFp16* base,
    const float* diag_ptr,
    const float* X_ptr,
    Eigen::Index ldX,
    float* Y_ptr,
    Eigen::Index ldY)
{
    AlignedRawBuffer<float, 64> X_rm(static_cast<size_t>(n) * P);
    AlignedRawBuffer<float, 64> Y_row(static_cast<size_t>(n) * P);
    AlignedRawBuffer<float, 64> Y_col(static_cast<size_t>(n) * P);

    const int JB = j_block;
    const int n_slabs = (n + JB - 1) / JB;
    std::vector<std::mutex> slab_locks(static_cast<size_t>(std::max(1, n_slabs)));
    const bool use_avx512 = packed_f16_has_avx512();
    const int use_nt = std::min(std::max(1, omp_get_max_threads()), n);

    for (int p0 = 0; p0 < cols; p0 += P) {
        const int p_len = std::min(P, cols - p0);
        std::vector<int> bounds;

        #pragma omp parallel num_threads(use_nt)
        {
            const int tid = omp_get_thread_num();
            const int nt  = omp_get_num_threads();

            #pragma omp single nowait
            {
                bounds = triangular_row_partition(n, nt);
            }

            // Phase 1: transpose ColMajor X[:, p0..p0+p_len) -> RowMajor X_rm (n x P, padded with 0)
            // and zero Y_col in disjoint row slices.
            const int row_lo = static_cast<int>((static_cast<int64_t>(n) * tid) / nt);
            const int row_hi = static_cast<int>((static_cast<int64_t>(n) * (tid + 1)) / nt);
            float* const x_rm_ptr = X_rm.data();
            float* const y_col_ptr = Y_col.data();
            float* const y_row_ptr = Y_row.data();

            for (int i = row_lo; i < row_hi; ++i) {
                float* __restrict__ xrow = x_rm_ptr + static_cast<size_t>(i) * P;
                for (int d = 0; d < p_len; ++d) {
                    xrow[d] = X_ptr[i + static_cast<Eigen::Index>(p0 + d) * ldX];
                }
                for (int d = p_len; d < P; ++d) {
                    xrow[d] = 0.0f;
                }
            }
            if (row_hi > row_lo) {
                std::memset(y_col_ptr + static_cast<size_t>(row_lo) * P, 0,
                            static_cast<size_t>(row_hi - row_lo) * P * sizeof(float));
            }

            #pragma omp barrier

            // Phase 2: triangular-partitioned single pass over strict-lower packed FP16 rows.
            const int num_parts = static_cast<int>(bounds.size()) - 1;
            const int r_lo = (tid < num_parts) ? bounds[static_cast<size_t>(tid)] : n;
            const int r_hi = (tid < num_parts) ? bounds[static_cast<size_t>(tid + 1)] : n;

            for (int i = r_lo; i < r_hi; ++i) {
                const float dv = diag_ptr[i];
                const float* __restrict__ xi = x_rm_ptr + static_cast<size_t>(i) * P;
                float* __restrict__ yr = y_row_ptr + static_cast<size_t>(i) * P;
                #pragma omp simd
                for (int d = 0; d < P; ++d) {
                    yr[d] = dv * xi[d];
                }
            }

            const int n_slabs_t = (r_hi > 1) ? ((r_hi - 2) / JB + 1) : 0;
            if (n_slabs_t > 0 && r_lo < r_hi) {
                AlignedRawBuffer<float, 64> y_col_loc(static_cast<size_t>(JB) * P);
                const int offset = (tid * 7) % n_slabs_t;
                for (int step = 0; step < n_slabs_t; ++step) {
                    const int s = (step + offset) % n_slabs_t;
                    const int j0 = s * JB;
                    const int j1 = std::min(n, j0 + JB);
                    const int i_start = std::max(r_lo, j0 + 1);
                    if (i_start >= r_hi) continue;
                    const int j_len = j1 - j0;
                    std::memset(y_col_loc.data(), 0, static_cast<size_t>(j_len) * P * sizeof(float));

                    const float* const x_j_ptr = x_rm_ptr + static_cast<size_t>(j0) * P;

                    // (a) Diagonal-straddling rows: j0 < i < min(r_hi, j1)
                    const int i_diag_end = std::min(r_hi, j1);
                    for (int i = i_start; i < i_diag_end; ++i) {
                        const GrmFp16* src_i = base + static_cast<size_t>(i) * (i + 1) / 2 + j0;
                        const int len = i - j0;
                        packed_f16_kernel_1row<P>(
                            src_i, len, x_j_ptr,
                            x_rm_ptr + static_cast<size_t>(i) * P,
                            y_row_ptr + static_cast<size_t>(i) * P,
                            y_col_loc.data(), use_avx512);
                    }

                    // (b) Rectangular rows: i in [max(i_start, j1), r_hi), 4 rows at a time
                    int i = std::max(i_start, j1);
                    for (; i + 3 < r_hi; i += 4) {
                        const GrmFp16* s0 = base + static_cast<size_t>(i) * (i + 1) / 2 + j0;
                        const GrmFp16* s1 = base + static_cast<size_t>(i + 1) * (i + 2) / 2 + j0;
                        const GrmFp16* s2 = base + static_cast<size_t>(i + 2) * (i + 3) / 2 + j0;
                        const GrmFp16* s3 = base + static_cast<size_t>(i + 3) * (i + 4) / 2 + j0;
                        packed_f16_kernel_4row<P>(
                            s0, s1, s2, s3, j_len, x_j_ptr,
                            x_rm_ptr + static_cast<size_t>(i) * P,
                            y_row_ptr + static_cast<size_t>(i) * P,
                            y_col_loc.data(), use_avx512);
                    }
                    for (; i < r_hi; ++i) {
                        const GrmFp16* src_i = base + static_cast<size_t>(i) * (i + 1) / 2 + j0;
                        packed_f16_kernel_1row<P>(
                            src_i, j_len, x_j_ptr,
                            x_rm_ptr + static_cast<size_t>(i) * P,
                            y_row_ptr + static_cast<size_t>(i) * P,
                            y_col_loc.data(), use_avx512);
                    }

                    // Flush thread-local y_col_loc into shared Y_col[j0..j1) under per-slab mutex.
                    {
                        std::lock_guard<std::mutex> lk(slab_locks[static_cast<size_t>(s)]);
                        float* __restrict__ dst = y_col_ptr + static_cast<size_t>(j0) * P;
                        const float* __restrict__ src = y_col_loc.data();
                        const size_t cnt = static_cast<size_t>(j_len) * P;
                        #pragma omp simd
                        for (size_t idx = 0; idx < cnt; ++idx) {
                            dst[idx] += src[idx];
                        }
                    }
                }
            }

            #pragma omp barrier

            // Phase 3: combine Y_row + Y_col and write back ColMajor Y[:, p0..p0+p_len).
            for (int i = row_lo; i < row_hi; ++i) {
                const float* __restrict__ yr = y_row_ptr + static_cast<size_t>(i) * P;
                const float* __restrict__ yc = y_col_ptr + static_cast<size_t>(i) * P;
                for (int d = 0; d < p_len; ++d) {
                    Y_ptr[i + static_cast<Eigen::Index>(p0 + d) * ldY] = yr[d] + yc[d];
                }
            }
        }
    }
}

} // namespace detail

inline void PackedGrmFp16::symm_f32_fused(
    const Eigen::Ref<const Eigen::MatrixXf>& X,
    Eigen::MatrixXf& Y,
    double* trace_K_out,
    double* trace_K2_out) const
{
    if (trace_K_out)  *trace_K_out  = trace_K_;
    if (trace_K2_out) *trace_K2_out = trace_K2_;
    const int n = n_;
    const int cols = static_cast<int>(X.cols());
    if (n == 0 || cols == 0) return;

    const float* X_ptr = X.data();
    const Eigen::Index ldX = X.outerStride();
    float* Y_ptr = Y.data();
    const Eigen::Index ldY = Y.outerStride();
    const GrmFp16* base = packed_.get();
    const float* diag_ptr = diag_f32_.data();

    if (cols <= 16) {
        detail::packed_f16_symm_panel_loop<16>(n, cols, fused_j_block_, base, diag_ptr, X_ptr, ldX, Y_ptr, ldY);
    } else {
        detail::packed_f16_symm_panel_loop<32>(n, cols, fused_j_block_, base, diag_ptr, X_ptr, ldX, Y_ptr, ldY);
    }
}

struct PackedGrmFp16Handle {
    gcta_chunked::TileReader reader;
    gcta_chunked::TileReaderF32 reader_f32;
    std::shared_ptr<PackedGrmFp16> file;
    double m_snps = -1.0;
    double trace_K = 0.0;
    double trace_K2 = 0.0;
    uint64_t id_hash = 0;
    int chunk_rows = 0;
};

inline PackedGrmFp16Handle load_packed_grm_fp16(
    const std::string& grm_prefix,
    const std::vector<std::string>& analysis_ids,
    bool need_m_snps = false,
    const std::string& caller_flag = "--grm-fp16",
    size_t grm_read_chunk_bytes = kDefaultGrmIoSlotBytes,
    const std::vector<std::string>* preloaded_grm_ids = nullptr,
    bool direct_identity_f16 = true)
{
    LOGGER.ts("grm_fp16");
    using std::to_string;

    const ResolvedGrmInput resolved = resolve_grm_input(grm_prefix, /*prefer_fp16=*/true);
    std::vector<std::string> grm_ids_storage;
    const std::vector<std::string>& grm_ids = preloaded_grm_ids
        ? *preloaded_grm_ids
        : (grm_ids_storage = Pheno::read_sublist(resolved.id_path));
    const int n_grm = static_cast<int>(grm_ids.size());
    if (n_grm == 0)
        LOGGER.e(0, "GRM id file [" + resolved.id_path + "] is empty.");

    const int n = static_cast<int>(analysis_ids.size());
    const std::vector<int> kp = match_ids_to_grm(analysis_ids, grm_ids);
    for (int i = 0; i < n; ++i) {
        if (kp[static_cast<size_t>(i)] < 0)
            LOGGER.e(0, caller_flag + ": individual [" + analysis_ids[static_cast<size_t>(i)] +
                        "] not found in GRM [" + grm_prefix + "].");
    }

    bool is_identity = (n_grm == n);
    bool is_monotonic = true;
    for (int i = 0; i < n; ++i) {
        if (kp[static_cast<size_t>(i)] != i) is_identity = false;
        if (i > 0 && kp[static_cast<size_t>(i)] <= kp[static_cast<size_t>(i - 1)]) is_monotonic = false;
    }
    if (!is_monotonic)
        LOGGER.e(0, caller_flag + " requires the analysis sample order to match "
                    "(or be an ordered subset of) [" + resolved.id_path + "]'s row order.");

    std::vector<KpRun> kp_runs;
    if (!is_identity && n > 0) {
        std::vector<KpRun> runs = build_kp_runs(kp);
        if (runs.size() * 2 < static_cast<size_t>(n)) {
            kp_runs = std::move(runs);
        }
    }

    const int fd = open_readonly_checked(resolved.data_path);
    const bool is_native_f16 = (resolved.format == GrmDiskFormat::F16);

    GrmFp16Header hdr{};
    Eigen::VectorXf diag_full;
    off_t body_off = 0;
    size_t elem_bytes = sizeof(float);
    if (is_native_f16) {
        read_grm_f16_header_and_diag(fd, resolved.data_path, grm_ids, hdr, diag_full);
        body_off = static_cast<off_t>(hdr.body_offset);
        elem_bytes = sizeof(GrmFp16);
    } else {
        detail::validate_f32_bin_size(fd, resolved.data_path, n_grm);
    }

    const bool use_header_traces = is_native_f16 && is_identity && ((hdr.flags & kGrmF16FlagHasTraces) != 0);
    const bool use_direct_identity_f16 = direct_identity_f16 && is_native_f16 && is_identity && use_header_traces;

    const size_t tri_n = static_cast<size_t>(n) * (n + 1) / 2;
    const double packed_gb = static_cast<double>(tri_n * sizeof(GrmFp16)) / 1e9;
    LOGGER.i(0, caller_flag + ": loading [" + resolved.data_path + "] into packed fp16 lower triangle (" +
                to_string(packed_gb) + " GB for n=" + to_string(n) +
                (use_direct_identity_f16
                    ? ", native .grm.f16 direct pread"
                    : (is_native_f16 ? ", native .grm.f16 async stream" : ", converting float32 -> fp16")) +
                ") ...");

    // Uninitialized allocation with parallel OpenMP first-touch distributes pages across NUMA nodes.
    AlignedUniquePtr<GrmFp16, 64> packed = make_aligned_uninitialized<GrmFp16, 64>(tri_n);
    if (tri_n > 0 && !packed) {
        ::close(fd);
        LOGGER.e(0, caller_flag + ": failed to allocate " + to_string(packed_gb) +
                    " GB for packed fp16 GRM.");
    }

    Eigen::VectorXf diag_f32(n);
    if (is_native_f16) {
        if (is_identity) {
            diag_f32 = std::move(diag_full);
        } else {
            for (int j = 0; j < n; ++j) {
                diag_f32[j] = diag_full[kp[static_cast<size_t>(j)]];
            }
            diag_full.resize(0);
        }
    }

    double tr_K  = use_header_traces ? hdr.trace_K : 0.0;
    double tr_K2 = use_header_traces ? hdr.trace_K2 : 0.0;

    if (use_direct_identity_f16) {
        struct ScopedFd {
            int fd = -1;
            ~ScopedFd() { if (fd >= 0) ::close(fd); }
        } fd_guard{fd};

        if (tri_n > 0) {
            GrmFp16* const packed_ptr = packed.get();
            const int use_nt = std::min(std::max(1, omp_get_max_threads()), std::max(1, n));
            std::vector<int> touch_bounds;
            #pragma omp parallel num_threads(use_nt)
            {
                const int tid = omp_get_thread_num();
                const int nt  = omp_get_num_threads();
                #pragma omp single
                {
                    touch_bounds = triangular_row_partition(n, nt);
                }
                const int num_parts = static_cast<int>(touch_bounds.size()) - 1;
                const int r_lo = (tid < num_parts) ? touch_bounds[static_cast<size_t>(tid)] : n;
                const int r_hi = (tid < num_parts) ? touch_bounds[static_cast<size_t>(tid + 1)] : n;
                const size_t byte_lo = (static_cast<size_t>(r_lo) * (r_lo + 1) / 2) * sizeof(GrmFp16);
                const size_t byte_hi = (static_cast<size_t>(r_hi) * (r_hi + 1) / 2) * sizeof(GrmFp16);
                if (byte_lo < byte_hi) {
                    volatile unsigned char* const vp = reinterpret_cast<volatile unsigned char*>(packed_ptr);
                    if (tid == 0) vp[byte_lo] = 0;
                    for (size_t pg = (byte_lo + 4095u) & ~size_t(4095u); pg < byte_hi; pg += 4096u) {
                        vp[pg] = 0;
                    }
                }
            }

            const size_t total_bytes = tri_n * sizeof(GrmFp16);
            const size_t direct_chunk = std::clamp<size_t>(grm_read_chunk_bytes, 1ULL << 20, kMaxIoChunkBytes);
            char* const dst_bytes = reinterpret_cast<char*>(packed_ptr);
            try {
                for (size_t done = 0; done < total_bytes; ) {
                    const size_t step = std::min(direct_chunk, total_bytes - done);
                    gcta_io::read_exact_at_throwing(
                        fd, dst_bytes + done, step, body_off + static_cast<off_t>(done), resolved.data_path);
                    done += step;
                }
            } catch (const std::exception& e) {
                LOGGER.e(0, std::string("Failed reading GRM [") + resolved.data_path + "]: " + e.what());
            }
        }
    } else {
        const size_t slot_bytes = std::clamp<size_t>(grm_read_chunk_bytes, 1ULL << 20, kMaxAsyncGrmSlotBytes);
        const size_t max_elems_per_chunk = std::max<size_t>(1, slot_bytes / elem_bytes);

        std::vector<detail::RowChunkPlan> plans;
        std::vector<std::pair<off_t, size_t>> io_chunks;
        detail::build_row_chunk_plan(n_grm, &kp, is_identity, max_elems_per_chunk,
                                     body_off, elem_bytes, plans, io_chunks);

        try {
            gcta_io::AsyncSeqReader reader(fd, resolved.data_path, std::move(io_chunks),
                                           /*max_slot_bytes=*/0, /*take_ownership=*/true);
            const int unpack_nt = std::max(1, omp_get_max_threads() - 1);
            GrmFp16* packed_ptr = packed.get();
            const char* raw_ptr = nullptr;
            size_t raw_len = 0;
            size_t plan_idx = 0;
            while (reader.next_slot(raw_ptr, raw_len)) {
                const auto& plan = plans[plan_idx++];
                const int row_cursor = plan.row_lo;
                const int j_cursor   = plan.j_lo;
                const int j_end      = plan.j_hi;
                const size_t row_base_cursor = static_cast<size_t>(row_cursor) * (row_cursor + 1) / 2;
                const int rows_in_chunk = j_end - j_cursor;
                const int dyn_chunk = std::clamp(rows_in_chunk / std::max(1, unpack_nt * 4), 1, 16);

                if (is_native_f16) {
                    const auto* src_f16 = reinterpret_cast<const GrmFp16*>(raw_ptr);
                    if (is_identity) {
                        if (use_header_traces) {
                            // Flat byte-slice memcpy into packed_ (NUMA parallel first-touch, zero copy per row).
                            char* const dst_base = reinterpret_cast<char*>(packed_ptr + row_base_cursor);
                            const size_t total_bytes = raw_len;
                            if (unpack_nt > 1 && total_bytes >= (256ULL << 10)) {
                                #pragma omp parallel num_threads(unpack_nt)
                                {
                                    const int tid = omp_get_thread_num();
                                    const int nt  = omp_get_num_threads();
                                    const size_t lo = ((total_bytes * static_cast<size_t>(tid)) / static_cast<size_t>(nt)) & ~size_t(63);
                                    const size_t hi = (tid + 1 == nt)
                                        ? total_bytes
                                        : (((total_bytes * static_cast<size_t>(tid + 1)) / static_cast<size_t>(nt)) & ~size_t(63));
                                    if (hi > lo) {
                                        std::memcpy(dst_base + lo, raw_ptr + lo, hi - lo);
                                    }
                                }
                            } else if (total_bytes > 0) {
                                std::memcpy(dst_base, raw_ptr, total_bytes);
                            }
                        } else {
                            #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                            for (int j = j_cursor; j < j_end; ++j) {
                                const size_t dst_off = static_cast<size_t>(j) * (j + 1) / 2;
                                const GrmFp16* __restrict__ src_row = src_f16 + (dst_off - row_base_cursor);
                                GrmFp16* __restrict__ dst_row = packed_ptr + dst_off;
                                std::memcpy(dst_row, src_row, static_cast<size_t>(j + 1) * sizeof(GrmFp16));
                                double row_off_sq = 0.0;
                                #pragma omp simd reduction(+:row_off_sq)
                                for (int i = 0; i < j; ++i) {
                                    const double vd = static_cast<double>(fp16_to_f32(src_row[i]));
                                    row_off_sq += vd * vd;
                                }
                                const double diag_d = static_cast<double>(diag_f32[j]);
                                tr_K  += diag_d;
                                tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                            }
                        }
                    } else if (!kp_runs.empty()) {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const int r_grm = kp[static_cast<size_t>(j)];
                            const GrmFp16* __restrict__ src_row = src_f16 +
                                (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                            GrmFp16* __restrict__ dst_row = packed_ptr +
                                static_cast<size_t>(j) * (j + 1) / 2;
                            double row_off_sq = 0.0;
                            for (const KpRun& r : kp_runs) {
                                if (r.dst_start >= j) break;
                                const int cnt = std::min(r.len, j - r.dst_start);
                                const GrmFp16* __restrict__ src_seg = src_row + r.src_start;
                                GrmFp16* __restrict__ dst_seg = dst_row + r.dst_start;
                                std::memcpy(dst_seg, src_seg, static_cast<size_t>(cnt) * sizeof(GrmFp16));
                                #pragma omp simd reduction(+:row_off_sq)
                                for (int k = 0; k < cnt; ++k) {
                                    const double vd = static_cast<double>(fp16_to_f32(src_seg[k]));
                                    row_off_sq += vd * vd;
                                }
                            }
                            dst_row[j] = src_row[r_grm];
                            const double diag_d = static_cast<double>(diag_f32[j]);
                            tr_K  += diag_d;
                            tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                        }
                    } else {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const int r_grm = kp[static_cast<size_t>(j)];
                            const GrmFp16* __restrict__ src_row = src_f16 +
                                (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                            GrmFp16* __restrict__ dst_row = packed_ptr +
                                static_cast<size_t>(j) * (j + 1) / 2;
                            double row_off_sq = 0.0;
                            for (int i = 0; i < j; ++i) {
                                const GrmFp16 h = src_row[kp[static_cast<size_t>(i)]];
                                dst_row[i] = h;
                                const double vd = static_cast<double>(fp16_to_f32(h));
                                row_off_sq += vd * vd;
                            }
                            dst_row[j] = src_row[r_grm];
                            const double diag_d = static_cast<double>(diag_f32[j]);
                            tr_K  += diag_d;
                            tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                        }
                    }
                } else {
                    const auto* src_f32 = reinterpret_cast<const float*>(raw_ptr);
                    if (is_identity) {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const float* __restrict__ src_row = src_f32 +
                                (static_cast<size_t>(j) * (j + 1) / 2 - row_base_cursor);
                            GrmFp16* __restrict__ dst_row = packed_ptr +
                                static_cast<size_t>(j) * (j + 1) / 2;
                            double row_off_sq = 0.0;
                            #pragma omp simd reduction(+:row_off_sq)
                            for (int i = 0; i < j; ++i) {
                                const float v = src_row[i];
                                dst_row[i] = f32_to_fp16(v);
                                const double vd = static_cast<double>(v);
                                row_off_sq += vd * vd;
                            }
                            const float diag_v = src_row[j];
                            dst_row[j] = f32_to_fp16(diag_v);
                            diag_f32[j] = diag_v;
                            const double diag_d = static_cast<double>(diag_v);
                            tr_K  += diag_d;
                            tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                        }
                    } else if (!kp_runs.empty()) {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const int r_grm = kp[static_cast<size_t>(j)];
                            const float* __restrict__ src_row = src_f32 +
                                (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                            GrmFp16* __restrict__ dst_row = packed_ptr +
                                static_cast<size_t>(j) * (j + 1) / 2;
                            double row_off_sq = 0.0;
                            for (const KpRun& r : kp_runs) {
                                if (r.dst_start >= j) break;
                                const int cnt = std::min(r.len, j - r.dst_start);
                                const float* __restrict__ src_seg = src_row + r.src_start;
                                GrmFp16* __restrict__ dst_seg = dst_row + r.dst_start;
                                #pragma omp simd reduction(+:row_off_sq)
                                for (int k = 0; k < cnt; ++k) {
                                    const float v = src_seg[k];
                                    dst_seg[k] = f32_to_fp16(v);
                                    const double vd = static_cast<double>(v);
                                    row_off_sq += vd * vd;
                                }
                            }
                            const float diag_v = src_row[r_grm];
                            dst_row[j] = f32_to_fp16(diag_v);
                            diag_f32[j] = diag_v;
                            const double diag_d = static_cast<double>(diag_v);
                            tr_K  += diag_d;
                            tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                        }
                    } else {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const int r_grm = kp[static_cast<size_t>(j)];
                            const float* __restrict__ src_row = src_f32 +
                                (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                            GrmFp16* __restrict__ dst_row = packed_ptr +
                                static_cast<size_t>(j) * (j + 1) / 2;
                            double row_off_sq = 0.0;
                            for (int i = 0; i < j; ++i) {
                                const float v = src_row[kp[static_cast<size_t>(i)]];
                                dst_row[i] = f32_to_fp16(v);
                                const double vd = static_cast<double>(v);
                                row_off_sq += vd * vd;
                            }
                            const float diag_v = src_row[r_grm];
                            dst_row[j] = f32_to_fp16(diag_v);
                            diag_f32[j] = diag_v;
                            const double diag_d = static_cast<double>(diag_v);
                            tr_K  += diag_d;
                            tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                        }
                    }
                }
                reader.release_slot();
            }
            reader.stop();
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("Failed reading GRM [") + resolved.data_path + "]: " + e.what());
        }
    }

    LOGGER.i(0, std::string("The ") + (is_native_f16 ? ".grm.f16" : ".grm.bin") +
                " packed fp16 fill took " + to_string(LOGGER.tp("grm_fp16")) + " seconds.");

    PackedGrmFp16Handle handle;
    handle.chunk_rows = std::min(n, 4096);
    handle.trace_K = tr_K;
    handle.trace_K2 = tr_K2;
    handle.id_hash = (is_native_f16 && is_identity) ? hdr.id_hash : 0ULL;
    if (need_m_snps) {
        handle.m_snps = is_native_f16 ? hdr.m_snps_mean : read_grm_N_mean(grm_prefix, n_grm, /*prefer_fp16=*/true);
    } else {
        LOGGER.i(0, "Skipping SNP count read from .grm.N.bin.");
        handle.m_snps = -1.0;
    }

    auto file = std::make_shared<PackedGrmFp16>(
        n, std::move(packed), std::move(diag_f32), tr_K, tr_K2);
    handle.file = file;

    handle.reader.read_fn = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXd> {
        return file->read_tile(rs, re, cs, ce);
    };
    handle.reader.read_T_fn = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXd> {
        return file->read_tile_T(rs, re, cs, ce);
    };
    handle.reader.read_diagonal_fn = [file]() -> Eigen::VectorXd {
        return file->read_diagonal();
    };
    handle.reader.release_fn = [file]() {
        file->release_scratch();
    };
    handle.reader.small_k_block_size = std::min(n, 8192);

    handle.reader_f32.read_fn = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXf> {
        return file->read_tile_f32(rs, re, cs, ce);
    };
    handle.reader_f32.read_T_fn = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXf> {
        return file->read_tile_f32_T(rs, re, cs, ce);
    };
    handle.reader_f32.read_diagonal_fn = [file]() -> Eigen::VectorXd {
        return file->read_diagonal();
    };
    handle.reader_f32.release_fn = [file]() {
        file->release_scratch();
    };
    handle.reader_f32.fused_symm_fn = [file](const Eigen::Ref<const Eigen::MatrixXf>& X,
                                             Eigen::MatrixXf& Y,
                                             double* tr_K_out,
                                             double* tr_K2_out) {
        file->symm_f32_fused(X, Y, tr_K_out, tr_K2_out);
    };
    // Avoid small-k tile starvation in PCA rSVD (k_ext < 256): use 8192-row tiles.
    handle.reader_f32.small_k_block_size = std::min(n, 8192);
    handle.reader_f32.fused_max_k = 128;

    return handle;
}

// Load a binary GRM (.grm.bin or .grm.f16, upper triangle only) and subset/reorder
// it to match `analysis_ids`, writing only the upper triangle of `G_out` so
// lower-triangle pages stay unfaulted. Uses double-buffered single-stream
// AsyncSeqReader (2 x 64 MiB) overlapping I/O with OpenMP unpack & traces.
template <typename Scalar>
inline void load_and_subset_grm_upper(
    const std::string& grm_prefix,
    const std::vector<std::string>& analysis_ids,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& G_out,
    double& m_snps_out,
    const std::string& caller_flag,
    size_t grm_read_chunk_bytes = kDefaultGrmIoSlotBytes,
    double* trace_K_out = nullptr,
    double* trace_K2_out = nullptr,
    const std::vector<std::string>* preloaded_grm_ids = nullptr,
    bool prefer_fp16 = false)
{
    const ResolvedGrmInput resolved = resolve_grm_input(grm_prefix, prefer_fp16);
    std::vector<std::string> grm_ids = preloaded_grm_ids
        ? *preloaded_grm_ids
        : Pheno::read_sublist(resolved.id_path);
    const int n_grm = static_cast<int>(grm_ids.size());
    if (n_grm == 0)
        LOGGER.e(0, "GRM id file [" + resolved.id_path + "] is empty.");

    const int n = static_cast<int>(analysis_ids.size());
    const std::vector<int> kp = match_ids_to_grm(analysis_ids, grm_ids);
    for (int i = 0; i < n; ++i) {
        if (kp[static_cast<size_t>(i)] < 0)
            LOGGER.e(0, caller_flag + ": individual [" + analysis_ids[static_cast<size_t>(i)] +
                        "] not found in GRM [" + grm_prefix + "].");
    }

    bool is_identity = (n_grm == n);
    bool is_monotonic = true;
    for (int i = 0; i < n; ++i) {
        if (kp[static_cast<size_t>(i)] != i) is_identity = false;
        if (i > 0 && kp[static_cast<size_t>(i)] <= kp[static_cast<size_t>(i - 1)]) is_monotonic = false;
    }

    if (is_identity) {
        read_grm_binary(grm_prefix, grm_ids, G_out, m_snps_out, /*upper_only=*/true,
                        grm_read_chunk_bytes, trace_K_out, trace_K2_out,
                        /*ids_preloaded=*/true, prefer_fp16);
        return;
    }

    const bool want_traces = (trace_K_out != nullptr || trace_K2_out != nullptr);

    if (is_monotonic) {
        LOGGER.ts("grm");
        using std::to_string;

        std::vector<KpRun> kp_runs;
        if (n > 0) {
            std::vector<KpRun> runs = build_kp_runs(kp);
            if (runs.size() * 2 < static_cast<size_t>(n)) {
                kp_runs = std::move(runs);
            }
        }

        const int fd = open_readonly_checked(resolved.data_path);
        const bool is_f16 = (resolved.format == GrmDiskFormat::F16);

        GrmFp16Header hdr{};
        Eigen::VectorXf diag_full;
        off_t body_off = 0;
        size_t elem_bytes = sizeof(float);
        if (is_f16) {
            read_grm_f16_header_and_diag(fd, resolved.data_path, grm_ids, hdr, diag_full);
            body_off = static_cast<off_t>(hdr.body_offset);
            elem_bytes = sizeof(GrmFp16);
        } else {
            detail::validate_f32_bin_size(fd, resolved.data_path, n_grm);
        }

        G_out.resize(n, n);

        double tr_K = 0.0;
        double tr_K2 = 0.0;

        const size_t slot_bytes = std::clamp<size_t>(grm_read_chunk_bytes, 1ULL << 20, kMaxAsyncGrmSlotBytes);
        const size_t max_elems_per_chunk = std::max<size_t>(1, slot_bytes / elem_bytes);

        std::vector<detail::RowChunkPlan> plans;
        std::vector<std::pair<off_t, size_t>> io_chunks;
        detail::build_row_chunk_plan(n_grm, &kp, /*is_identity=*/false, max_elems_per_chunk,
                                     body_off, elem_bytes, plans, io_chunks);

        try {
            gcta_io::AsyncSeqReader reader(fd, resolved.data_path, std::move(io_chunks),
                                           /*max_slot_bytes=*/0, /*take_ownership=*/true);
            const int unpack_nt = std::max(1, omp_get_max_threads() - 1);
            const char* raw_ptr = nullptr;
            size_t raw_len = 0;
            size_t plan_idx = 0;
            while (reader.next_slot(raw_ptr, raw_len)) {
                const auto& plan = plans[plan_idx++];
                const int row_cursor = plan.row_lo;
                const int j_cursor   = plan.j_lo;
                const int j_end      = plan.j_hi;
                const size_t row_base_cursor = static_cast<size_t>(row_cursor) * (row_cursor + 1) / 2;
                const int rows_in_chunk = j_end - j_cursor;
                const int dyn_chunk = std::clamp(rows_in_chunk / std::max(1, unpack_nt * 4), 1, 16);

                if (is_f16) {
                    const auto* chunk_f16 = reinterpret_cast<const GrmFp16*>(raw_ptr);
                    if (!kp_runs.empty()) {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const int r_grm = kp[static_cast<size_t>(j)];
                            const GrmFp16* __restrict__ row_ptr = chunk_f16 +
                                (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                            Scalar* __restrict__ col_out = G_out.col(j).data();
                            double row_off_sq = 0.0;
                            for (const KpRun& r : kp_runs) {
                                if (r.dst_start >= j) break;
                                const int cnt = std::min(r.len, j - r.dst_start);
                                const GrmFp16* __restrict__ src_seg = row_ptr + r.src_start;
                                Scalar* __restrict__ dst_seg = col_out + r.dst_start;
                                if (want_traces) {
                                    #pragma omp simd reduction(+:row_off_sq)
                                    for (int k = 0; k < cnt; ++k) {
                                        const float v = fp16_to_f32(src_seg[k]);
                                        dst_seg[k] = static_cast<Scalar>(v);
                                        const double vd = static_cast<double>(v);
                                        row_off_sq += vd * vd;
                                    }
                                } else {
                                    #pragma omp simd
                                    for (int k = 0; k < cnt; ++k) {
                                        dst_seg[k] = static_cast<Scalar>(fp16_to_f32(src_seg[k]));
                                    }
                                }
                            }
                            const float diag_v = diag_full[r_grm];
                            col_out[j] = static_cast<Scalar>(diag_v);
                            if (want_traces) {
                                const double diag_d = static_cast<double>(diag_v);
                                tr_K  += diag_d;
                                tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                            }
                        }
                    } else {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const int r_grm = kp[static_cast<size_t>(j)];
                            const GrmFp16* row_ptr = chunk_f16 +
                                (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                            Scalar* col_out = G_out.col(j).data();
                            double row_off_sq = 0.0;
                            for (int i = 0; i < j; ++i) {
                                const float v = fp16_to_f32(row_ptr[kp[static_cast<size_t>(i)]]);
                                col_out[i] = static_cast<Scalar>(v);
                                if (want_traces) {
                                    const double vd = static_cast<double>(v);
                                    row_off_sq += vd * vd;
                                }
                            }
                            const float diag_v = diag_full[r_grm];
                            col_out[j] = static_cast<Scalar>(diag_v);
                            if (want_traces) {
                                const double diag_d = static_cast<double>(diag_v);
                                tr_K  += diag_d;
                                tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                            }
                        }
                    }
                } else {
                    const auto* chunk_f32 = reinterpret_cast<const float*>(raw_ptr);
                    if (!kp_runs.empty()) {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const int r_grm = kp[static_cast<size_t>(j)];
                            const float* __restrict__ row_ptr = chunk_f32 +
                                (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                            Scalar* __restrict__ col_out = G_out.col(j).data();
                            double row_off_sq = 0.0;
                            for (const KpRun& r : kp_runs) {
                                if (r.dst_start >= j) break;
                                const int cnt = std::min(r.len, j - r.dst_start);
                                const float* __restrict__ src_seg = row_ptr + r.src_start;
                                Scalar* __restrict__ dst_seg = col_out + r.dst_start;
                                if constexpr (std::is_same_v<Scalar, float>) {
                                    if (!want_traces) {
                                        std::memcpy(dst_seg, src_seg, static_cast<size_t>(cnt) * sizeof(float));
                                    } else {
                                        #pragma omp simd reduction(+:row_off_sq)
                                        for (int k = 0; k < cnt; ++k) {
                                            const float v = src_seg[k];
                                            dst_seg[k] = v;
                                            const double vd = static_cast<double>(v);
                                            row_off_sq += vd * vd;
                                        }
                                    }
                                } else if (want_traces) {
                                    #pragma omp simd reduction(+:row_off_sq)
                                    for (int k = 0; k < cnt; ++k) {
                                        const float v = src_seg[k];
                                        dst_seg[k] = static_cast<Scalar>(v);
                                        const double vd = static_cast<double>(v);
                                        row_off_sq += vd * vd;
                                    }
                                } else {
                                    #pragma omp simd
                                    for (int k = 0; k < cnt; ++k) {
                                        dst_seg[k] = static_cast<Scalar>(src_seg[k]);
                                    }
                                }
                            }
                            const float diag_v = row_ptr[r_grm];
                            col_out[j] = static_cast<Scalar>(diag_v);
                            if (want_traces) {
                                const double diag_d = static_cast<double>(diag_v);
                                tr_K  += diag_d;
                                tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                            }
                        }
                    } else {
                        #pragma omp parallel for num_threads(unpack_nt) reduction(+:tr_K, tr_K2) schedule(dynamic, dyn_chunk)
                        for (int j = j_cursor; j < j_end; ++j) {
                            const int r_grm = kp[static_cast<size_t>(j)];
                            const float* row_ptr = chunk_f32 +
                                (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                            Scalar* col_out = G_out.col(j).data();
                            double row_off_sq = 0.0;
                            for (int i = 0; i < j; ++i) {
                                const float v = row_ptr[kp[static_cast<size_t>(i)]];
                                col_out[i] = static_cast<Scalar>(v);
                                if (want_traces) {
                                    const double vd = static_cast<double>(v);
                                    row_off_sq += vd * vd;
                                }
                            }
                            const float diag_v = row_ptr[r_grm];
                            col_out[j] = static_cast<Scalar>(diag_v);
                            if (want_traces) {
                                const double diag_d = static_cast<double>(diag_v);
                                tr_K  += diag_d;
                                tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
                            }
                        }
                    }
                }
                reader.release_slot();
            }
            reader.stop();
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("Failed reading GRM [") + resolved.data_path + "]: " + e.what());
        }

        if (trace_K_out)  *trace_K_out  = tr_K;
        if (trace_K2_out) *trace_K2_out = tr_K2;

        LOGGER.i(0, std::string("The ") + (is_f16 ? ".grm.f16" : ".grm.bin") +
                    " fill took " + std::to_string(LOGGER.tp("grm")) + " seconds.");
        if (m_snps_out < 0.0) {
            LOGGER.i(0, "Skipping SNP count read from .grm.N.bin.");
        } else if (is_f16) {
            m_snps_out = hdr.m_snps_mean;
        } else {
            m_snps_out = read_grm_N_mean(grm_prefix, n_grm, prefer_fp16);
        }
        return;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> G_full;
    read_grm_binary(grm_prefix, grm_ids, G_full, m_snps_out, /*upper_only=*/true,
                    grm_read_chunk_bytes, nullptr, nullptr, /*ids_preloaded=*/true, prefer_fp16);
    G_out.resize(n, n);
    double tr_K = 0.0;
    double tr_K2 = 0.0;
    const int fallback_dyn_chunk = std::clamp(n / std::max(1, omp_get_max_threads() * 4), 1, 16);
    #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, fallback_dyn_chunk)
    for (int j = 0; j < n; ++j) {
        const int src_col = kp[static_cast<size_t>(j)];
        double row_off_sq = 0.0;
        for (int i = 0; i < j; ++i) {
            const int r = kp[static_cast<size_t>(i)], c = src_col;
            const Scalar v = (r <= c) ? G_full(r, c) : G_full(c, r);
            G_out(i, j) = v;
            if (want_traces) {
                const double vd = static_cast<double>(v);
                row_off_sq += vd * vd;
            }
        }
        const Scalar diag_v = G_full(src_col, src_col);
        G_out(j, j) = diag_v;
        if (want_traces) {
            const double diag_d = static_cast<double>(diag_v);
            tr_K  += diag_d;
            tr_K2 += diag_d * diag_d + 2.0 * row_off_sq;
        }
    }
    G_full.resize(0, 0);
    if (trace_K_out)  *trace_K_out  = tr_K;
    if (trace_K2_out) *trace_K2_out = tr_K2;
}

// ---------------------------------------------------------------------------
// HPC-safe O(1)-RAM accumulator-style single-stream GRM merge
//
// Strictly 1 dedicated sequential reader thread cycling inputs 0..K-1 in large
// chunks into 2 bounce slots + OpenMP accumulation into a single double
// accumulator + strictly 1 dedicated sequential writer thread (never writing
// two files concurrently). Supports classic .grm.bin, native .grm.f16 (zero-
// missing scalar-weight fast path with zero .grm.N.bin I/O), and mixed inputs.
// ---------------------------------------------------------------------------
inline void merge_grms(
    const std::vector<std::string>& prefixes,
    const std::string& out_prefix,
    double memory_budget_gb = 0.0,
    bool out_fp16 = false)
{
    if (prefixes.empty())
        LOGGER.e(0, "merge_grms: no input GRM prefixes given.");

    const std::string id0_path = prefixes[0] + ".grm.id";
    const std::vector<std::string> ids = Pheno::read_sublist(id0_path);
    const int n = static_cast<int>(ids.size());
    if (n == 0) LOGGER.e(0, "GRM id file [" + id0_path + "] is empty.");

    // Fast byte-comparison of remaining .grm.id files (avoids parsing K x 500k strings);
    // falls back to parsed comparison only if raw bytes differ (e.g. CRLF vs LF).
    for (size_t f = 1; f < prefixes.size(); ++f) {
        const std::string other_path = prefixes[f] + ".grm.id";
        if (!files_byte_identical(id0_path, other_path)) {
            const std::vector<std::string> other_ids = Pheno::read_sublist(other_path);
            if (other_ids != ids) {
                LOGGER.e(0, "merge_grms: [" + other_path + "] does not match [" + id0_path +
                            "] exactly (same sample order required). Use 'grm unify' for mismatched orderings.");
            }
        }
    }

    const size_t tri = static_cast<size_t>(n) * (n + 1) / 2;
    const int K = static_cast<int>(prefixes.size());

    struct InputGrmStream {
        int val_fd = -1;
        int n_fd = -1;
        std::string val_path;
        std::string n_path;
        bool is_fp16 = false;
        bool zero_missing = false;
        off_t val_body_off = 0;
        double const_n_val = 0.0;
        uint64_t valid_markers = 0;
        Eigen::VectorXf diag_f32;
        GrmFp16Header hdr{};
    };

    std::vector<InputGrmStream> inputs(static_cast<size_t>(K));
    auto close_all_inputs = [&]() {
        for (auto& in : inputs) {
            if (in.val_fd >= 0) { ::close(in.val_fd); in.val_fd = -1; }
            if (in.n_fd   >= 0) { ::close(in.n_fd);   in.n_fd   = -1; }
        }
    };

    bool all_inputs_fp16 = true;
    bool all_zero_missing = true;
    uint64_t sum_valid_markers = 0;
    double sum_const_n = 0.0;

    for (int f = 0; f < K; ++f) {
        auto& in = inputs[static_cast<size_t>(f)];
        const ResolvedGrmInput res = resolve_grm_input(prefixes[static_cast<size_t>(f)], out_fp16);
        in.val_path = res.data_path;
        in.n_path   = res.n_path;
        in.is_fp16  = (res.format == GrmDiskFormat::F16);
        in.val_fd   = open_readonly_checked(in.val_path);

        if (in.is_fp16) {
            read_grm_f16_header_and_diag(in.val_fd, in.val_path, ids, in.hdr, in.diag_f32);
            in.val_body_off = static_cast<off_t>(in.hdr.body_offset);
            in.zero_missing = ((in.hdr.flags & kGrmF16FlagZeroMissing) != 0);
            in.valid_markers = in.hdr.valid_markers;
            if (in.zero_missing) {
                in.const_n_val = (in.hdr.valid_markers > 0)
                    ? static_cast<double>(in.hdr.valid_markers)
                    : in.hdr.m_snps_mean;
                sum_valid_markers += in.hdr.valid_markers;
                sum_const_n += in.const_n_val;
            } else {
                all_zero_missing = false;
                in.n_fd = open_readonly_checked(in.n_path);
                detail::validate_f32_bin_size(in.n_fd, in.n_path, n);
                sum_valid_markers += in.hdr.valid_markers;
            }
        } else {
            all_inputs_fp16 = false;
            all_zero_missing = false;
            detail::validate_f32_bin_size(in.val_fd, in.val_path, n);
            in.n_fd = open_readonly_checked(in.n_path);
            detail::validate_f32_bin_size(in.n_fd, in.n_path, n);
        }
    }

    const bool write_fp16 = out_fp16 || all_inputs_fp16;
    const bool write_n_file = !(write_fp16 && all_zero_missing);

    // Precompute scalar weights when all inputs are zero-missing.
    std::vector<double> const_weights(static_cast<size_t>(K), 0.0);
    if (all_zero_missing && sum_const_n > 0.0) {
        for (int f = 0; f < K; ++f) {
            const_weights[static_cast<size_t>(f)] = inputs[static_cast<size_t>(f)].const_n_val / sum_const_n;
        }
    }

    // O(1) memory budget independent of K:
    //   acc_wsum (8 B/elem) + optional acc_wtN (8 B/elem)
    //   + 2 input slots (up to 4 B val + 4 B N = 8 B/elem, or 2 B/elem if all fp16 zero-miss)
    //   + 2 output slots (2 or 4 B val + 0 or 4 B N)
    const size_t max_in_bpe  = (all_inputs_fp16 && all_zero_missing) ? 2u : (all_zero_missing ? 4u : 8u);
    const size_t out_val_bpe = write_fp16 ? sizeof(GrmFp16) : sizeof(float);
    const size_t out_n_bpe   = write_n_file ? sizeof(float) : 0u;
    const size_t acc_bpe     = all_zero_missing ? sizeof(double) : (2u * sizeof(double));
    const size_t total_bpe   = acc_bpe + 2u * max_in_bpe + 2u * (out_val_bpe + out_n_bpe);

    constexpr size_t kDefaultChunkElems = 16ULL << 20; // 16M elems (~32 MiB fp16 / 64 MiB f32)
    constexpr size_t kMaxChunkElems     = 32ULL << 20; // 32M elems (64 MiB fp16 / 128 MiB f32)
    size_t chunk_elems = kDefaultChunkElems;
    if (memory_budget_gb > 0.0) {
        const size_t budget_bytes = static_cast<size_t>(memory_budget_gb * 1e9);
        chunk_elems = std::clamp<size_t>(budget_bytes / std::max<size_t>(1, total_bpe), 65536, std::min(tri, kMaxChunkElems));
    } else {
        chunk_elems = std::clamp<size_t>(kDefaultChunkElems, 65536, std::max<size_t>(65536, tri));
        chunk_elems = std::min(chunk_elems, tri);
    }
    if (chunk_elems == 0) chunk_elems = 1;

    LOGGER.i(0, "merge_grms: streaming " + std::to_string(K) + " GRMs (n=" + std::to_string(n) +
                ", chunk=" + std::to_string((chunk_elems * out_val_bpe) >> 20) + " MiB output / " +
                std::to_string((chunk_elems * total_bpe) >> 20) + " MiB O(1) total RAM, format=" +
                (write_fp16 ? ".grm.f16" : ".grm.bin") +
                (all_zero_missing ? ", zero-missing scalar weights" : ", per-pair N weights") + ").");

    // Open output files.
    const std::string out_val_path = out_prefix + (write_fp16 ? ".grm.f16" : ".grm.bin");
    const std::string out_n_path   = out_prefix + ".grm.N.bin";
    const uint64_t out_body_off = write_fp16 ? grm_f16_body_offset(static_cast<uint64_t>(n)) : 0ULL;

    int wflags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_CLOEXEC
    wflags |= O_CLOEXEC;
#endif
    const int out_val_fd = ::open(out_val_path.c_str(), wflags, 0644);
    if (out_val_fd < 0) {
        close_all_inputs();
        LOGGER.e(0, "cannot open [" + out_val_path + "] for writing: " + std::string(std::strerror(errno)));
    }
    int out_n_fd = -1;
    if (write_n_file) {
        out_n_fd = ::open(out_n_path.c_str(), wflags, 0644);
        if (out_n_fd < 0) {
            ::close(out_val_fd);
            close_all_inputs();
            LOGGER.e(0, "cannot open [" + out_n_path + "] for writing: " + std::string(std::strerror(errno)));
        }
    }

    if (write_fp16) {
        GrmFp16Header draft{};
        std::memcpy(draft.magic, kGrmF16MagicDraft, sizeof(draft.magic));
        draft.version = kGrmF16Version;
        draft.dtype = kGrmF16DtypeFp16;
        draft.header_bytes = kGrmF16HeaderSize;
        draft.n = static_cast<uint64_t>(n);
        draft.diag_offset = kGrmF16HeaderSize;
        draft.body_offset = out_body_off;
        draft.part_row_start = 0;
        draft.part_row_end = static_cast<uint64_t>(n);
        std::vector<char> prefix(static_cast<size_t>(out_body_off), 0);
        std::memcpy(prefix.data(), &draft, sizeof(GrmFp16Header));
        try {
            gcta_io::write_exact_throwing(out_val_fd, prefix.data(), prefix.size(), out_val_path);
        } catch (const std::exception& e) {
            ::close(out_val_fd);
            if (out_n_fd >= 0) ::close(out_n_fd);
            close_all_inputs();
            LOGGER.e(0, std::string("merge_grms: ") + e.what());
        }
    }

    // Double-buffered single-stream reader state (1 thread cycling inputs 0..K-1 per chunk).
    struct InSlot {
        AlignedRawBuffer<char, 64> val_bytes;
        AlignedRawBuffer<float, 64> n_floats;
        int file_idx = 0;
        size_t elems = 0;
    };
    std::array<InSlot, 2> in_slots;
    for (int s = 0; s < 2; ++s) {
        in_slots[static_cast<size_t>(s)].val_bytes.resize(chunk_elems * (all_inputs_fp16 ? sizeof(GrmFp16) : sizeof(float)));
        if (!all_zero_missing) {
            in_slots[static_cast<size_t>(s)].n_floats.resize(chunk_elems);
        }
    }

    std::mutex in_mu;
    std::condition_variable_any in_ready_cv, in_free_cv;
    std::array<bool, 2> in_busy{false, false};
    std::array<int, 2> in_ring{0, 0};
    int in_head = 0, in_tail = 0, in_count = 0;
    bool in_failed = false;
    std::exception_ptr in_err;

    std::jthread reader_thread([&](std::stop_token st) {
        try {
            size_t seq = 0;
            for (size_t offset = 0; offset < tri && !st.stop_requested(); offset += chunk_elems) {
                const size_t elems = std::min(chunk_elems, tri - offset);
                for (int f = 0; f < K && !st.stop_requested(); ++f) {
                    const int slot = static_cast<int>(seq & 1u);
                    ++seq;
                    {
                        std::unique_lock<std::mutex> lock(in_mu);
                        in_free_cv.wait(lock, st, [&]() { return !in_busy[static_cast<size_t>(slot)]; });
                        if (st.stop_requested()) return;
                        in_busy[static_cast<size_t>(slot)] = true;
                    }
                    const auto& in = inputs[static_cast<size_t>(f)];
                    const size_t v_bpe = in.is_fp16 ? sizeof(GrmFp16) : sizeof(float);
                    gcta_io::read_exact_at_throwing(
                        in.val_fd, in_slots[static_cast<size_t>(slot)].val_bytes.data(),
                        elems * v_bpe,
                        in.val_body_off + static_cast<off_t>(offset * v_bpe),
                        in.val_path);
                    if (!in.zero_missing) {
                        gcta_io::read_exact_at_throwing(
                            in.n_fd, in_slots[static_cast<size_t>(slot)].n_floats.data(),
                            elems * sizeof(float),
                            static_cast<off_t>(offset * sizeof(float)),
                            in.n_path);
                    }
                    {
                        std::lock_guard<std::mutex> lock(in_mu);
                        in_slots[static_cast<size_t>(slot)].file_idx = f;
                        in_slots[static_cast<size_t>(slot)].elems = elems;
                        in_ring[static_cast<size_t>(in_tail)] = slot;
                        in_tail = (in_tail + 1) & 1;
                        ++in_count;
                    }
                    in_ready_cv.notify_one();
                }
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(in_mu);
                in_err = std::current_exception();
                in_failed = true;
            }
            in_ready_cv.notify_all();
        }
    });

    // Double-buffered single-stream sequential writer thread (writes out_val then
    // out_n sequentially on the same single thread -- zero concurrent 2-file writes).
    struct OutSlot {
        AlignedRawBuffer<char, 64> val_bytes;
        AlignedRawBuffer<float, 64> n_floats;
        size_t val_len = 0;
        size_t n_len = 0;
    };
    std::array<OutSlot, 2> out_slots;
    for (int s = 0; s < 2; ++s) {
        out_slots[static_cast<size_t>(s)].val_bytes.resize(chunk_elems * out_val_bpe);
        if (write_n_file) {
            out_slots[static_cast<size_t>(s)].n_floats.resize(chunk_elems);
        }
    }

    std::mutex out_mu;
    std::condition_variable_any out_work_cv, out_space_cv;
    std::array<bool, 2> out_queued{false, false};
    std::array<int, 2> out_ring{0, 0};
    int out_head = 0, out_tail = 0, out_count = 0;
    bool out_closing = false, out_abort = false, out_failed = false;
    std::exception_ptr out_err;

    std::jthread writer_thread([&](std::stop_token st) {
        while (true) {
            int slot = -1;
            {
                std::unique_lock<std::mutex> lock(out_mu);
                out_work_cv.wait(lock, st, [&]() {
                    return out_count > 0 || out_closing || out_abort || out_failed;
                });
                if (out_abort || out_failed) return;
                if (out_count == 0) {
                    if (out_closing || st.stop_requested()) return;
                    continue;
                }
                slot = out_ring[static_cast<size_t>(out_head)];
                out_head = (out_head + 1) & 1;
                --out_count;
            }
            try {
                const auto& os = out_slots[static_cast<size_t>(slot)];
                if (os.val_len > 0) {
                    gcta_io::write_exact_throwing(out_val_fd, os.val_bytes.data(), os.val_len, out_val_path);
                }
                if (write_n_file && os.n_len > 0) {
                    gcta_io::write_exact_throwing(out_n_fd, os.n_floats.data(), os.n_len, out_n_path);
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(out_mu);
                    out_err = std::current_exception();
                    out_failed = true;
                    out_queued[static_cast<size_t>(slot)] = false;
                }
                out_space_cv.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(out_mu);
                out_queued[static_cast<size_t>(slot)] = false;
            }
            out_space_cv.notify_all();
        }
    });

    // O(1) accumulators
    AlignedRawBuffer<double, 64> acc_wsum(chunk_elems);
    AlignedRawBuffer<double, 64> acc_wtN;
    if (!all_zero_missing) acc_wtN.resize(chunk_elems);

    std::vector<float> out_diag_f32;
    if (write_fp16) out_diag_f32.assign(static_cast<size_t>(n), 0.0f);
    double tr_K = 0.0;
    double tr_K2 = 0.0;
    double diag_n_sum = 0.0;
    int next_diag_row = 0;

    std::vector<std::pair<int, size_t>> chunk_diags;
    int next_out_slot = 0;

    auto cleanup_on_error = [&]() {
        reader_thread.request_stop();
        in_free_cv.notify_all();
        if (reader_thread.joinable()) reader_thread.join();
        {
            std::lock_guard<std::mutex> lock(out_mu);
            out_abort = true;
            out_closing = true;
        }
        writer_thread.request_stop();
        out_work_cv.notify_all();
        if (writer_thread.joinable()) writer_thread.join();
        ::close(out_val_fd);
        if (out_n_fd >= 0) ::close(out_n_fd);
        close_all_inputs();
    };

    try {
        for (size_t offset = 0; offset < tri; offset += chunk_elems) {
            const size_t elems = std::min(chunk_elems, tri - offset);
            const size_t chunk_end = offset + elems;

            // Identify diagonal positions falling inside [offset, chunk_end).
            chunk_diags.clear();
            while (next_diag_row < n) {
                const size_t d_pos = static_cast<size_t>(next_diag_row) * (next_diag_row + 1) / 2 +
                                     static_cast<size_t>(next_diag_row);
                if (d_pos >= chunk_end) break;
                chunk_diags.emplace_back(next_diag_row, d_pos - offset);
                ++next_diag_row;
            }

            double* __restrict__ wsum_ptr = acc_wsum.data();
            double* __restrict__ wtn_ptr  = all_zero_missing ? nullptr : acc_wtN.data();
            #pragma omp parallel for schedule(static)
            for (size_t j = 0; j < elems; ++j) {
                wsum_ptr[j] = 0.0;
                if (wtn_ptr) wtn_ptr[j] = 0.0;
            }

            for (int f = 0; f < K; ++f) {
                int slot = -1;
                {
                    std::unique_lock<std::mutex> lock(in_mu);
                    in_ready_cv.wait(lock, [&]() { return in_failed || in_count > 0; });
                    if (in_failed) std::rethrow_exception(in_err);
                    slot = in_ring[static_cast<size_t>(in_head)];
                    in_head = (in_head + 1) & 1;
                    --in_count;
                }
                const auto& is = in_slots[static_cast<size_t>(slot)];
                const auto& in = inputs[static_cast<size_t>(is.file_idx)];

                if (all_zero_missing) {
                    const double wf = const_weights[static_cast<size_t>(is.file_idx)];
                    if (in.is_fp16) {
                        const auto* vp = reinterpret_cast<const GrmFp16*>(is.val_bytes.data());
                        #pragma omp parallel for schedule(static)
                        for (size_t j = 0; j < elems; ++j) {
                            wsum_ptr[j] += wf * static_cast<double>(fp16_to_f32(vp[j]));
                        }
                        // Preserve exact float32 diagonal accuracy across .grm.f16 merges.
                        for (const auto& [r, loc] : chunk_diags) {
                            const double q = static_cast<double>(fp16_to_f32(vp[loc]));
                            const double exact = static_cast<double>(in.diag_f32[r]);
                            wsum_ptr[loc] += wf * (exact - q);
                        }
                    } else {
                        const auto* vp = reinterpret_cast<const float*>(is.val_bytes.data());
                        #pragma omp parallel for schedule(static)
                        for (size_t j = 0; j < elems; ++j) {
                            wsum_ptr[j] += wf * static_cast<double>(vp[j]);
                        }
                    }
                } else if (in.zero_missing) {
                    const double nn = in.const_n_val;
                    const auto* vp = reinterpret_cast<const GrmFp16*>(is.val_bytes.data());
                    #pragma omp parallel for schedule(static)
                    for (size_t j = 0; j < elems; ++j) {
                        wsum_ptr[j] += static_cast<double>(fp16_to_f32(vp[j])) * nn;
                        wtn_ptr[j]  += nn;
                    }
                    for (const auto& [r, loc] : chunk_diags) {
                        const double q = static_cast<double>(fp16_to_f32(vp[loc]));
                        const double exact = static_cast<double>(in.diag_f32[r]);
                        wsum_ptr[loc] += (exact - q) * nn;
                    }
                } else {
                    const float* __restrict__ np = is.n_floats.data();
                    if (in.is_fp16) {
                        const auto* vp = reinterpret_cast<const GrmFp16*>(is.val_bytes.data());
                        #pragma omp parallel for schedule(static)
                        for (size_t j = 0; j < elems; ++j) {
                            const double nn = static_cast<double>(np[j]);
                            wsum_ptr[j] += static_cast<double>(fp16_to_f32(vp[j])) * nn;
                            wtn_ptr[j]  += nn;
                        }
                        for (const auto& [r, loc] : chunk_diags) {
                            const double nn = static_cast<double>(np[loc]);
                            const double q = static_cast<double>(fp16_to_f32(vp[loc]));
                            const double exact = static_cast<double>(in.diag_f32[r]);
                            wsum_ptr[loc] += (exact - q) * nn;
                        }
                    } else {
                        const auto* vp = reinterpret_cast<const float*>(is.val_bytes.data());
                        #pragma omp parallel for schedule(static)
                        for (size_t j = 0; j < elems; ++j) {
                            const double nn = static_cast<double>(np[j]);
                            wsum_ptr[j] += static_cast<double>(vp[j]) * nn;
                            wtn_ptr[j]  += nn;
                        }
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(in_mu);
                    in_busy[static_cast<size_t>(slot)] = false;
                }
                in_free_cv.notify_one();
            }

            // Acquire free output slot.
            int oslot = -1;
            {
                std::unique_lock<std::mutex> lock(out_mu);
                out_space_cv.wait(lock, [&]() {
                    return out_failed || !out_queued[0] || !out_queued[1];
                });
                if (out_failed) std::rethrow_exception(out_err);
                oslot = !out_queued[static_cast<size_t>(next_out_slot)] ? next_out_slot : (1 - next_out_slot);
                next_out_slot = 1 - oslot;
            }

            auto& os = out_slots[static_cast<size_t>(oslot)];
            if (write_fp16) {
                auto* out_v = reinterpret_cast<GrmFp16*>(os.val_bytes.data());
                float* out_n = write_n_file ? os.n_floats.data() : nullptr;
                double chunk_all_sq = 0.0;
                if (all_zero_missing) {
                    #pragma omp parallel for reduction(+:chunk_all_sq) schedule(static)
                    for (size_t j = 0; j < elems; ++j) {
                        const float v = static_cast<float>(wsum_ptr[j]);
                        out_v[j] = f32_to_fp16(v);
                        const double vd = static_cast<double>(v);
                        chunk_all_sq += vd * vd;
                    }
                    double diag_sq = 0.0;
                    for (const auto& [r, loc] : chunk_diags) {
                        const float dv = static_cast<float>(wsum_ptr[loc]);
                        out_diag_f32[static_cast<size_t>(r)] = dv;
                        const double dd = static_cast<double>(dv);
                        tr_K += dd;
                        diag_sq += dd * dd;
                    }
                    tr_K2 += 2.0 * chunk_all_sq - diag_sq;
                } else {
                    #pragma omp parallel for reduction(+:chunk_all_sq) schedule(static)
                    for (size_t j = 0; j < elems; ++j) {
                        const double nn = wtn_ptr[j];
                        const float v = static_cast<float>(nn > 0.0 ? (wsum_ptr[j] / nn) : 0.0);
                        out_v[j] = f32_to_fp16(v);
                        if (out_n) out_n[j] = static_cast<float>(nn);
                        const double vd = static_cast<double>(v);
                        chunk_all_sq += vd * vd;
                    }
                    double diag_sq = 0.0;
                    for (const auto& [r, loc] : chunk_diags) {
                        const double nn = wtn_ptr[loc];
                        const float dv = static_cast<float>(nn > 0.0 ? (wsum_ptr[loc] / nn) : 0.0);
                        out_diag_f32[static_cast<size_t>(r)] = dv;
                        diag_n_sum += nn;
                        const double dd = static_cast<double>(dv);
                        tr_K += dd;
                        diag_sq += dd * dd;
                    }
                    tr_K2 += 2.0 * chunk_all_sq - diag_sq;
                }
                os.val_len = elems * sizeof(GrmFp16);
                os.n_len   = write_n_file ? (elems * sizeof(float)) : 0;
            } else {
                auto* out_v = reinterpret_cast<float*>(os.val_bytes.data());
                float* out_n = os.n_floats.data();
                if (all_zero_missing) {
                    const float const_n_f = static_cast<float>(sum_const_n);
                    #pragma omp parallel for schedule(static)
                    for (size_t j = 0; j < elems; ++j) {
                        out_v[j] = static_cast<float>(wsum_ptr[j]);
                        out_n[j] = const_n_f;
                    }
                } else {
                    #pragma omp parallel for schedule(static)
                    for (size_t j = 0; j < elems; ++j) {
                        const double nn = wtn_ptr[j];
                        out_v[j] = static_cast<float>(nn > 0.0 ? (wsum_ptr[j] / nn) : 0.0);
                        out_n[j] = static_cast<float>(nn);
                    }
                }
                os.val_len = elems * sizeof(float);
                os.n_len   = elems * sizeof(float);
            }

            {
                std::lock_guard<std::mutex> lock(out_mu);
                if (out_failed) std::rethrow_exception(out_err);
                out_queued[static_cast<size_t>(oslot)] = true;
                out_ring[static_cast<size_t>(out_tail)] = oslot;
                out_tail = (out_tail + 1) & 1;
                ++out_count;
            }
            out_work_cv.notify_one();
        }

        if (reader_thread.joinable()) reader_thread.join();
        if (in_failed) std::rethrow_exception(in_err);

        {
            std::lock_guard<std::mutex> lock(out_mu);
            out_closing = true;
        }
        out_work_cv.notify_all();
        if (writer_thread.joinable()) writer_thread.join();
        if (out_failed) std::rethrow_exception(out_err);

        if (write_fp16) {
            GrmFp16Header final_hdr{};
            std::memcpy(final_hdr.magic, kGrmF16MagicFinal, sizeof(final_hdr.magic));
            final_hdr.version = kGrmF16Version;
            final_hdr.dtype = kGrmF16DtypeFp16;
            final_hdr.header_bytes = kGrmF16HeaderSize;
            final_hdr.flags = kGrmF16FlagFinalized | kGrmF16FlagHasTraces | kGrmF16FlagHasDiagF32 |
                              (all_zero_missing ? kGrmF16FlagZeroMissing : 0u);
            final_hdr.n = static_cast<uint64_t>(n);
            final_hdr.m_snps_mean = all_zero_missing
                ? sum_const_n
                : (n > 0 ? diag_n_sum / static_cast<double>(n) : 0.0);
            final_hdr.valid_markers = (sum_valid_markers > 0)
                ? sum_valid_markers
                : (final_hdr.m_snps_mean > 0.0
                       ? static_cast<uint64_t>(std::llround(final_hdr.m_snps_mean))
                       : 0ULL);
            final_hdr.id_hash = gcta_io::hash_sample_ids(ids);
            final_hdr.trace_K = tr_K;
            final_hdr.trace_K2 = tr_K2;
            final_hdr.diag_offset = kGrmF16HeaderSize;
            final_hdr.body_offset = out_body_off;
            final_hdr.part_row_start = 0;
            final_hdr.part_row_end = static_cast<uint64_t>(n);
            final_hdr.mtd_weight = 1.0;

            if (n > 0) {
                gcta_io::pwrite_exact_throwing(
                    out_val_fd, out_diag_f32.data(),
                    static_cast<size_t>(n) * sizeof(float),
                    static_cast<off_t>(final_hdr.diag_offset), out_val_path);
            }
            gcta_io::pwrite_exact_throwing(out_val_fd, &final_hdr, sizeof(GrmFp16Header), 0, out_val_path);
        }
    } catch (const std::exception& e) {
        cleanup_on_error();
        LOGGER.e(0, std::string("merge_grms: ") + e.what());
    }

    if (::close(out_val_fd) != 0) {
        if (out_n_fd >= 0) ::close(out_n_fd);
        close_all_inputs();
        LOGGER.e(0, "close() failed on [" + out_val_path + "]: " + std::string(std::strerror(errno)));
    }
    if (out_n_fd >= 0 && ::close(out_n_fd) != 0) {
        close_all_inputs();
        LOGGER.e(0, "close() failed on [" + out_n_path + "]: " + std::string(std::strerror(errno)));
    }
    close_all_inputs();

    copy_file_posix(id0_path, out_prefix + ".grm.id");

    LOGGER.i(0, "Merged " + std::to_string(K) + " GRMs (" + std::to_string(n) +
                " individuals) into [" + out_val_path +
                (write_n_file ? ", .grm.N.bin" : " (zero-missing; .grm.N.bin omitted)") +
                ", .grm.id], single-stream O(1)-RAM merge.");
}

} // namespace gcta_grm_io
