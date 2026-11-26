/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   Clean-room BED autosomal additive VanRaden INT8 GRM engine.
   Completely decoupled from legacy GRM / Geno / GenoBufItem / loopDouble / BedBackend.
*/

#include "GrmBedInt8.hpp"

#include "Logger.h"
#include "OptionIO.h"
#include "async_seq_io.hpp"
#include "bed_simd.hpp"
#include "constants.hpp"
#include "cpu.h"
#include "grm_binary_io.hpp"
#include "grm_format.hpp"
#include "mem.hpp"
#include "utils.hpp"
#include "third_party/plink-ng/2.0/include/plink2_bits.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fcntl.h>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/mman.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace gcta_grm_bed_detail {

using Clock = std::chrono::steady_clock;

#if defined(IOV_MAX)
constexpr int kMaxIov = (IOV_MAX < 1024) ? IOV_MAX : 1024;
#else
constexpr int kMaxIov = 1024;
#endif

constexpr uint64_t kMaxBatchBytes     = 1ULL << 30;
constexpr uint64_t kMaxGapBytes       = 2ULL << 20;
constexpr uint32_t kMaxInt32Markers   = 500000000u;
constexpr size_t   kConstNFloats      = (64ULL << 20) / sizeof(float);

// Persistent 2 MiB-aligned (for buffers >= 16 MiB) slab backed by direct
// anonymous mmap + MADV_HUGEPAGE on Linux so physical 2 MiB THP pages are
// decoupled from jemalloc thp:never, reused across tiles, and striped evenly
// across NUMA nodes on first touch.
template <typename T>
class HugeAlignedSlab {
    static_assert(std::is_trivially_default_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>,
                  "HugeAlignedSlab requires trivial T");
public:
    static constexpr size_t kHugeAlignThreshold = 16ULL << 20; // 16 MiB
    static constexpr size_t kHugeAlign          = 2ULL << 20;  // 2 MiB

    HugeAlignedSlab() noexcept = default;
    ~HugeAlignedSlab() { reset(); }

    HugeAlignedSlab(const HugeAlignedSlab&) = delete;
    HugeAlignedSlab& operator=(const HugeAlignedSlab&) = delete;

    HugeAlignedSlab(HugeAlignedSlab&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          cap_(std::exchange(other.cap_, 0)),
          align_(std::exchange(other.align_, 64)),
          mmap_bytes_(std::exchange(other.mmap_bytes_, 0)),
          is_mmap_(std::exchange(other.is_mmap_, false)) {}

    HugeAlignedSlab& operator=(HugeAlignedSlab&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
            cap_ = std::exchange(other.cap_, 0);
            align_ = std::exchange(other.align_, 64);
            mmap_bytes_ = std::exchange(other.mmap_bytes_, 0);
            is_mmap_ = std::exchange(other.is_mmap_, false);
        }
        return *this;
    }

    T* data() const noexcept { return ptr_; }
    size_t capacity() const noexcept { return cap_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    void reset() noexcept {
        if (ptr_) {
#ifdef __linux__
            if (is_mmap_) {
                (void)::munmap(static_cast<void*>(ptr_), mmap_bytes_);
            } else {
#if defined(__cpp_sized_deallocation)
                ::operator delete(ptr_, cap_ * sizeof(T), std::align_val_t{align_});
#else
                ::operator delete(ptr_, std::align_val_t{align_});
#endif
            }
#else
#if defined(__cpp_sized_deallocation)
            ::operator delete(ptr_, cap_ * sizeof(T), std::align_val_t{align_});
#else
            ::operator delete(ptr_, std::align_val_t{align_});
#endif
#endif
            ptr_ = nullptr;
        }
        cap_ = 0;
        align_ = 64;
        mmap_bytes_ = 0;
        is_mmap_ = false;
    }

    bool ensure(size_t need_elems, bool parallel_first_touch = true) noexcept {
        if (need_elems == 0) return true;
        if (need_elems <= cap_) return true;
        if (need_elems > std::numeric_limits<size_t>::max() / sizeof(T)) return false;

        reset();

        const size_t bytes = need_elems * sizeof(T);
        const size_t align = (bytes >= kHugeAlignThreshold) ? kHugeAlign : 64ULL;
        void* p = nullptr;
        size_t mapped_bytes = 0;
        bool mapped = false;
#ifdef __linux__
        if (bytes >= kHugeAlignThreshold) {
            if (bytes > std::numeric_limits<size_t>::max() - 4095ULL - kHugeAlign) {
                return false;
            }
            const size_t aligned_bytes = (bytes + 4095ULL) & ~4095ULL;
            void* const raw = ::mmap(
                nullptr, aligned_bytes + kHugeAlign, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (raw == MAP_FAILED) return false;
            const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw);
            const uintptr_t aligned_addr = (raw_addr + (kHugeAlign - 1ULL)) & ~(kHugeAlign - 1ULL);
            const size_t lead = static_cast<size_t>(aligned_addr - raw_addr);
            const size_t tail = kHugeAlign - lead;
            if (lead > 0) (void)::munmap(raw, lead);
            if (tail > 0) (void)::munmap(reinterpret_cast<void*>(aligned_addr + aligned_bytes), tail);
            p = reinterpret_cast<void*>(aligned_addr);
            mapped_bytes = aligned_bytes;
            mapped = true;
#ifdef MADV_HUGEPAGE
            (void)::madvise(p, aligned_bytes, MADV_HUGEPAGE);
#endif
        } else {
            p = ::operator new(bytes, std::align_val_t{align}, std::nothrow);
        }
#else
        p = ::operator new(bytes, std::align_val_t{align}, std::nothrow);
#endif
        if (!p) return false;

#if defined(__cpp_lib_start_lifetime_as)
        ptr_ = std::start_lifetime_as_array<T>(p, need_elems);
#else
        ptr_ = static_cast<T*>(p);
#endif
        cap_ = need_elems;
        align_ = align;
        mmap_bytes_ = mapped_bytes;
        is_mmap_ = mapped;

        if (parallel_first_touch && bytes >= kHugeAlignThreshold) {
            // Touch one element per 4 KiB page inside 2 MiB round-robin chunks so
            // physical 2 MiB THP pages stripe evenly across all NUMA nodes under local membind.
            constexpr size_t kPageElems = std::max<size_t>(1, 4096ULL / sizeof(T));
            constexpr size_t kChunkElems = std::max<size_t>(1, kHugeAlign / sizeof(T));
            const size_t n_chunks = (need_elems + kChunkElems - 1ULL) / kChunkElems;
            #pragma omp parallel for schedule(static, 1)
            for (size_t ch = 0; ch < n_chunks; ++ch) {
                const size_t beg = ch * kChunkElems;
                const size_t end = std::min(need_elems, beg + kChunkElems);
                for (size_t idx = beg; idx < end; idx += kPageElems) {
                    *static_cast<volatile T*>(&ptr_[idx]) = T{};
                }
            }
        }
        return true;
    }

private:
    T* ptr_ = nullptr;
    size_t cap_ = 0;
    size_t align_ = 64;
    size_t mmap_bytes_ = 0;
    bool is_mmap_ = false;
};

inline void write_buffer_with_cache_drop(int fd,
                                         const char* data,
                                         size_t nbytes,
                                         const std::string& path)
{
    if (nbytes == 0) return;
#ifdef __linux__
    constexpr size_t kMinDropBytes = 64ULL << 20;
    constexpr size_t kChunkBytes   = 256ULL << 20;
    if (nbytes >= kMinDropBytes) {
        const off_t base_off = ::lseek(fd, 0, SEEK_CUR);
        if (base_off != static_cast<off_t>(-1)) {
            size_t done = 0;
            off_t prev_off = 0;
            size_t prev_len = 0;
            while (done < nbytes) {
                const size_t c = std::min(kChunkBytes, nbytes - done);
                const off_t cur_off = base_off + static_cast<off_t>(done);
                gcta_io::write_exact_throwing(fd, data + done, c, path);
#ifdef SYNC_FILE_RANGE_WRITE
                (void)::sync_file_range(fd, cur_off, static_cast<off_t>(c), SYNC_FILE_RANGE_WRITE);
#endif
                if (prev_len > 0) {
#if defined(SYNC_FILE_RANGE_WAIT_BEFORE) && defined(SYNC_FILE_RANGE_WRITE) && defined(SYNC_FILE_RANGE_WAIT_AFTER)
                    (void)::sync_file_range(
                        fd, prev_off, static_cast<off_t>(prev_len),
                        SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
#endif
#ifdef POSIX_FADV_DONTNEED
                    (void)::posix_fadvise(fd, prev_off, static_cast<off_t>(prev_len), POSIX_FADV_DONTNEED);
#endif
                }
                prev_off = cur_off;
                prev_len = c;
                done += c;
            }
            if (prev_len > 0) {
#if defined(SYNC_FILE_RANGE_WAIT_BEFORE) && defined(SYNC_FILE_RANGE_WRITE) && defined(SYNC_FILE_RANGE_WAIT_AFTER)
                (void)::sync_file_range(
                    fd, prev_off, static_cast<off_t>(prev_len),
                    SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
#endif
#ifdef POSIX_FADV_DONTNEED
                (void)::posix_fadvise(fd, prev_off, static_cast<off_t>(prev_len), POSIX_FADV_DONTNEED);
#endif
            }
            return;
        }
    }
#endif
    gcta_io::write_exact_throwing(fd, data, nbytes, path);
}

inline void write_const_N_with_cache_drop(int n_fd,
                                          size_t n_elems,
                                          float n_val,
                                          const std::string& n_path,
                                          std::vector<float>& const_N_buf)
{
    if (n_fd < 0 || n_elems == 0) return;
    const size_t buf_floats = std::min(n_elems, kConstNFloats);
    if (const_N_buf.size() < buf_floats || (buf_floats > 0 && const_N_buf[0] != n_val)) {
        const_N_buf.assign(buf_floats, n_val);
    }
    const size_t total_bytes = n_elems * sizeof(float);
    const size_t cap_bytes = buf_floats * sizeof(float);

#ifdef __linux__
    constexpr size_t kMinDropBytes = 64ULL << 20;
    if (total_bytes >= kMinDropBytes) {
        const off_t base_off = ::lseek(n_fd, 0, SEEK_CUR);
        if (base_off != static_cast<off_t>(-1)) {
            size_t done = 0;
            off_t prev_off = 0;
            size_t prev_len = 0;
            while (done < total_bytes) {
                const size_t c = std::min(cap_bytes, total_bytes - done);
                const off_t cur_off = base_off + static_cast<off_t>(done);
                gcta_io::write_exact_throwing(n_fd, const_N_buf.data(), c, n_path);
#ifdef SYNC_FILE_RANGE_WRITE
                (void)::sync_file_range(n_fd, cur_off, static_cast<off_t>(c), SYNC_FILE_RANGE_WRITE);
#endif
                if (prev_len > 0) {
#if defined(SYNC_FILE_RANGE_WAIT_BEFORE) && defined(SYNC_FILE_RANGE_WRITE) && defined(SYNC_FILE_RANGE_WAIT_AFTER)
                    (void)::sync_file_range(
                        n_fd, prev_off, static_cast<off_t>(prev_len),
                        SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
#endif
#ifdef POSIX_FADV_DONTNEED
                    (void)::posix_fadvise(n_fd, prev_off, static_cast<off_t>(prev_len), POSIX_FADV_DONTNEED);
#endif
                }
                prev_off = cur_off;
                prev_len = c;
                done += c;
            }
            if (prev_len > 0) {
#if defined(SYNC_FILE_RANGE_WAIT_BEFORE) && defined(SYNC_FILE_RANGE_WRITE) && defined(SYNC_FILE_RANGE_WAIT_AFTER)
                (void)::sync_file_range(
                    n_fd, prev_off, static_cast<off_t>(prev_len),
                    SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
#endif
#ifdef POSIX_FADV_DONTNEED
                (void)::posix_fadvise(n_fd, prev_off, static_cast<off_t>(prev_len), POSIX_FADV_DONTNEED);
#endif
            }
            return;
        }
    }
#endif
    size_t rem_bytes = total_bytes;
    while (rem_bytes > 0) {
        const size_t c = std::min(rem_bytes, cap_bytes);
        gcta_io::write_exact_throwing(n_fd, const_N_buf.data(), c, n_path);
        rem_bytes -= c;
    }
}

// Background sequential flusher owning a persistent packed-output slab so tile t
// streams to disk asynchronously while tile t+1 computes in i32_slab_.
class StagedTileFlusher {
public:
    StagedTileFlusher() = default;
    StagedTileFlusher(const StagedTileFlusher&) = delete;
    StagedTileFlusher& operator=(const StagedTileFlusher&) = delete;

    ~StagedTileFlusher() noexcept {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closing_ = true;
        }
        work_cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    HugeAlignedSlab<char>& stage_slab() noexcept { return stage_slab_; }

    void wait_idle() {
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [&]() { return !busy_; });
        if (io_error_) {
            std::exception_ptr err = std::exchange(io_error_, nullptr);
            std::rethrow_exception(err);
        }
    }

    void wait_and_release() {
        wait_idle();
        stage_slab_.reset();
        std::vector<float>().swap(const_N_buf_);
    }

    void submit(int grm_fd,
                std::string grm_path,
                size_t grm_bytes,
                int n_fd,
                std::string n_path,
                size_t n_elems,
                float n_val)
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!worker_.joinable()) {
            worker_ = std::jthread([this](std::stop_token st) { worker_loop(st); });
        }
        grm_fd_ = grm_fd;
        grm_path_ = std::move(grm_path);
        grm_bytes_ = grm_bytes;
        n_fd_ = n_fd;
        n_path_ = std::move(n_path);
        n_elems_ = n_elems;
        n_val_ = n_val;
        busy_ = true;
        work_cv_.notify_one();
    }

private:
    void worker_loop(std::stop_token st) {
        while (true) {
            int grm_fd = -1;
            std::string grm_path;
            size_t grm_bytes = 0;
            int n_fd = -1;
            std::string n_path;
            size_t n_elems = 0;
            float n_val = 0.0f;
            {
                std::unique_lock<std::mutex> lock(mu_);
                work_cv_.wait(lock, st, [&]() { return busy_ || closing_; });
                if (!busy_) {
                    if (closing_ || st.stop_requested()) return;
                    continue;
                }
                grm_fd = grm_fd_;
                grm_path = std::move(grm_path_);
                grm_bytes = grm_bytes_;
                n_fd = n_fd_;
                n_path = std::move(n_path_);
                n_elems = n_elems_;
                n_val = n_val_;
            }

            std::exception_ptr err;
            try {
                write_buffer_with_cache_drop(grm_fd, stage_slab_.data(), grm_bytes, grm_path);
                if (n_fd >= 0 && n_elems > 0) {
                    write_const_N_with_cache_drop(n_fd, n_elems, n_val, n_path, const_N_buf_);
                }
            } catch (...) {
                err = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                if (err && !io_error_) {
                    io_error_ = std::move(err);
                }
                busy_ = false;
            }
            done_cv_.notify_all();
        }
    }

    HugeAlignedSlab<char> stage_slab_;
    std::vector<float> const_N_buf_;
    std::mutex mu_;
    std::condition_variable_any work_cv_;
    std::condition_variable_any done_cv_;
    bool busy_ = false;
    bool closing_ = false;
    std::exception_ptr io_error_;

    int grm_fd_ = -1;
    std::string grm_path_;
    size_t grm_bytes_ = 0;
    int n_fd_ = -1;
    std::string n_path_;
    size_t n_elems_ = 0;
    float n_val_ = 0.0f;

    std::jthread worker_;
};

class UniqueFd {
public:
    UniqueFd() = default;
    explicit UniqueFd(int fd) noexcept : fd_(fd) {}
    ~UniqueFd() { reset(); }
    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    UniqueFd(UniqueFd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    UniqueFd& operator=(UniqueFd&& o) noexcept {
        if (this != &o) {
            reset(o.fd_);
            o.fd_ = -1;
        }
        return *this;
    }
    int get() const noexcept { return fd_; }
    explicit operator bool() const noexcept { return fd_ >= 0; }
    void reset(int new_fd = -1) noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = new_fd;
    }
private:
    int fd_ = -1;
};

inline std::string errno_str() {
    return std::string(std::strerror(errno));
}

inline UniqueFd open_bed_read(const std::string& path) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        LOGGER.e(0, "GrmBedInt8: can't open [" + path + "]: " + errno_str());
    }
    return UniqueFd(fd);
}

inline UniqueFd open_posix_write_trunc(const std::string& path) noexcept {
    int flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    return UniqueFd(::open(path.c_str(), flags, 0644));
}

inline void pread_full_throwing(int fd, void* dst, size_t len, uint64_t off, const std::string& fname) {
    auto* p = static_cast<unsigned char*>(dst);
    while (len > 0) {
        const ssize_t r = ::pread(fd, p, len, static_cast<off_t>(off));
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("read error on [" + fname + "]: " + errno_str());
        }
        if (r == 0) {
            throw std::runtime_error("unexpected end of file in [" + fname + "].");
        }
        p += r;
        len -= static_cast<size_t>(r);
        off += static_cast<uint64_t>(r);
    }
}

inline void preadv_full_throwing(int fd, struct iovec* iov, int cnt, uint64_t off, const std::string& fname) {
    if (cnt <= 0) return;
    if (cnt == 1) {
        pread_full_throwing(fd, iov[0].iov_base, iov[0].iov_len, off, fname);
        return;
    }
#if defined(__linux__) || defined(__APPLE__)
    int idx = 0;
    uint64_t pos = off;
    while (idx < cnt) {
        const int rem_cnt = cnt - idx;
        if (rem_cnt == 1) {
            pread_full_throwing(fd, iov[idx].iov_base, iov[idx].iov_len, pos, fname);
            return;
        }
        ssize_t r;
        do {
            r = ::preadv(fd, iov + idx, rem_cnt, static_cast<off_t>(pos));
        } while (r < 0 && errno == EINTR);
        if (r < 0) {
            throw std::runtime_error("read error on [" + fname + "]: " + errno_str());
        }
        if (r == 0) {
            throw std::runtime_error("unexpected end of file in [" + fname + "].");
        }
        size_t done = static_cast<size_t>(r);
        pos += static_cast<uint64_t>(done);
        while (idx < cnt && done >= iov[idx].iov_len) {
            done -= iov[idx].iov_len;
            ++idx;
        }
        if (idx < cnt && done > 0) {
            iov[idx].iov_base = static_cast<unsigned char*>(iov[idx].iov_base) + done;
            iov[idx].iov_len -= done;
        }
    }
#else
    uint64_t pos = off;
    for (int i = 0; i < cnt; ++i) {
        pread_full_throwing(fd, iov[i].iov_base, iov[i].iov_len, pos, fname);
        pos += static_cast<uint64_t>(iov[i].iov_len);
    }
#endif
}

struct BedMarkerLoc {
    uint32_t file_idx = 0;
    uint32_t local_rec = 0;
};

struct BedBlock {
    AlignedRawBuffer<uint8_t, 64> buf;
    uint32_t num_markers = 0;
};

// Persistent 2-slot double-buffered async sequential BED reader.
// Buffers are allocated ONCE and reused across all GRM row tiles (<=1 reader thread).
class BedBlockReader {
public:
    BedBlockReader(std::vector<std::string> files,
                   std::vector<uint64_t> file_snp_counts,
                   uint32_t rec_bytes,
                   size_t block_marker_cap,
                   size_t total_marker_hint)
        : files_(std::move(files)),
          file_snp_counts_(std::move(file_snp_counts)),
          rec_bytes_(rec_bytes)
    {
        const size_t eff_markers = std::min(block_marker_cap, std::max<size_t>(1, total_marker_hint));
        const size_t slot_bytes = std::max<size_t>(64, eff_markers * static_cast<size_t>(rec_bytes_));
        slots_[0].buf.resize(slot_bytes);
        slots_[1].buf.resize(slot_bytes);
        iov_.reserve(64);
    }

    BedBlockReader(const BedBlockReader&) = delete;
    BedBlockReader& operator=(const BedBlockReader&) = delete;

    void set_drop_read_cache(bool v) noexcept { drop_read_cache_ = v; }

    template <typename Fn>
    void stream(std::span<const BedMarkerLoc> locs,
                uint32_t block_markers,
                bool track_wait,
                double& io_wait_sec_out,
                Fn&& on_block)
    {
        if (locs.empty()) return;

        std::mutex mu;
        std::condition_variable_any ready_cv;
        std::condition_variable_any free_cv;
        std::deque<int> ready;
        std::array<bool, 2> in_use{false, false};
        bool done = false;
        std::exception_ptr io_err;

        std::jthread io_thread([&](std::stop_token st) {
            UniqueFd cur_fd;
            int cur_file = -1;
            try {
                size_t cursor = 0;
                const size_t total = locs.size();
                while (cursor < total && !st.stop_requested()) {
                    const uint32_t file_idx = locs[cursor].file_idx;
                    const size_t remain = total - cursor;
                    const size_t max_take = std::min<size_t>(block_markers, remain);
                    size_t take = 1;
                    while (take < max_take && locs[cursor + take].file_idx == file_idx) {
                        ++take;
                    }

                    int slot = -1;
                    {
                        std::unique_lock<std::mutex> lock(mu);
                        free_cv.wait(lock, st, [&]() { return !in_use[0] || !in_use[1]; });
                        if (st.stop_requested()) break;
                        slot = !in_use[0] ? 0 : 1;
                        in_use[static_cast<size_t>(slot)] = true;
                    }

                    BedBlock& blk = slots_[static_cast<size_t>(slot)];
                    blk.num_markers = static_cast<uint32_t>(take);
                    const size_t need_bytes = take * static_cast<size_t>(rec_bytes_);
                    if (blk.buf.size() < need_bytes) {
                        blk.buf.resize(need_bytes);
                    }

                    const std::string& fname = files_[file_idx];
                    if (cur_file != static_cast<int>(file_idx)) {
                        cur_fd.reset();
                        int flags = O_RDONLY;
#ifdef O_CLOEXEC
                        flags |= O_CLOEXEC;
#endif
                        const int raw_fd = ::open(fname.c_str(), flags);
                        if (raw_fd < 0) {
                            throw std::runtime_error("BedBlockReader: can't open [" + fname + "]: " + errno_str());
                        }
                        cur_fd = UniqueFd(raw_fd);
                        cur_file = static_cast<int>(file_idx);
                    }

                    read_batch(cur_fd.get(), fname, file_snp_counts_[file_idx],
                               locs.subspan(cursor, take), blk.buf.data());
                    cursor += take;

                    {
                        std::lock_guard<std::mutex> lock(mu);
                        ready.push_back(slot);
                    }
                    ready_cv.notify_one();
                }

                {
                    std::lock_guard<std::mutex> lock(mu);
                    done = true;
                }
                ready_cv.notify_all();
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mu);
                    io_err = std::current_exception();
                    done = true;
                }
                ready_cv.notify_all();
            }
        });

        try {
            while (true) {
                int slot = -1;
                {
                    const auto t_wait0 = track_wait ? Clock::now() : Clock::time_point{};
                    std::unique_lock<std::mutex> lock(mu);
                    ready_cv.wait(lock, [&]() { return io_err || done || !ready.empty(); });
                    if (track_wait) {
                        io_wait_sec_out += std::chrono::duration<double>(Clock::now() - t_wait0).count();
                    }
                    if (io_err) break;
                    if (ready.empty()) {
                        if (done) break;
                        continue;
                    }
                    slot = ready.front();
                    ready.pop_front();
                }

                on_block(slots_[static_cast<size_t>(slot)]);

                {
                    std::lock_guard<std::mutex> lock(mu);
                    in_use[static_cast<size_t>(slot)] = false;
                }
                free_cv.notify_one();
            }
        } catch (...) {
            io_thread.request_stop();
            free_cv.notify_all();
            if (io_thread.joinable()) io_thread.join();
            throw;
        }

        if (io_thread.joinable()) io_thread.join();
        if (io_err) std::rethrow_exception(io_err);
    }

private:
    void read_batch(int fd,
                    const std::string& fname,
                    uint64_t n_snp_file,
                    std::span<const BedMarkerLoc> span_locs,
                    uint8_t* base)
    {
        iov_.clear();
        uint64_t batch_start = 0;
        uint64_t next_rec = 0;
        uint64_t batch_bytes = 0;
        const uint64_t rec_b = rec_bytes_;

        auto flush = [&]() {
            if (iov_.empty()) return;
            const uint64_t file_off = 3ULL + batch_start * rec_b;
            if (iov_.size() == 1) {
                pread_full_throwing(fd, iov_[0].iov_base, iov_[0].iov_len, file_off, fname);
            } else {
                preadv_full_throwing(fd, iov_.data(), static_cast<int>(iov_.size()), file_off, fname);
            }
#if defined(__linux__) && defined(POSIX_FADV_DONTNEED)
            if (drop_read_cache_ && batch_bytes >= 8192ULL) {
                const uint64_t drop_beg = (file_off + 4095ULL) & ~4095ULL;
                const uint64_t drop_end = (file_off + batch_bytes) & ~4095ULL;
                if (drop_end > drop_beg) {
                    (void)::posix_fadvise(
                        fd,
                        static_cast<off_t>(drop_beg),
                        static_cast<off_t>(drop_end - drop_beg),
                        POSIX_FADV_DONTNEED);
                }
            }
#endif
            iov_.clear();
            batch_bytes = 0;
        };

        for (size_t i = 0; i < span_locs.size(); ++i) {
            const uint64_t rec = span_locs[i].local_rec;
            if (rec >= n_snp_file) {
                throw std::runtime_error("BedBlockReader: variant index out of range for [" + fname + "].");
            }
            uint8_t* const dst = base + i * rec_b;
            bool start_new = iov_.empty();
            bool bridged_gap = false;
            if (!start_new) {
                if (rec < next_rec) {
                    start_new = true;
                } else if (rec > next_rec) {
                    const uint64_t gap_bytes = (rec - next_rec) * rec_b;
                    if (gap_bytes <= kMaxGapBytes &&
                        iov_.size() + 2 <= static_cast<size_t>(kMaxIov) &&
                        batch_bytes + gap_bytes + rec_b <= kMaxBatchBytes) {
                        if (discard_.size() < kMaxGapBytes) {
                            discard_.resize(static_cast<size_t>(kMaxGapBytes));
                        }
                        iov_.push_back({discard_.data(), static_cast<size_t>(gap_bytes)});
                        batch_bytes += gap_bytes;
                        next_rec = rec;
                        bridged_gap = true;
                    } else {
                        start_new = true;
                    }
                }
            }
            if (!start_new && !bridged_gap) {
                const bool can_coalesce =
                    !iov_.empty() &&
                    iov_.back().iov_base != discard_.data() &&
                    static_cast<uint8_t*>(iov_.back().iov_base) + iov_.back().iov_len == dst;
                if (batch_bytes + rec_b > kMaxBatchBytes ||
                    (!can_coalesce && iov_.size() + 1 > static_cast<size_t>(kMaxIov))) {
                    start_new = true;
                }
            }
            if (start_new) {
                flush();
                batch_start = rec;
                next_rec = rec;
            }
            if (!iov_.empty() &&
                iov_.back().iov_base != discard_.data() &&
                static_cast<uint8_t*>(iov_.back().iov_base) + iov_.back().iov_len == dst) {
                iov_.back().iov_len += static_cast<size_t>(rec_b);
            } else {
                iov_.push_back({dst, static_cast<size_t>(rec_b)});
            }
            batch_bytes += rec_b;
            next_rec = rec + 1;
        }
        flush();
    }

    std::vector<std::string> files_;
    std::vector<uint64_t> file_snp_counts_;
    uint32_t rec_bytes_ = 0;
    std::array<BedBlock, 2> slots_;
    AlignedRawBuffer<uint8_t, 64> discard_;
    std::vector<struct iovec> iov_;
    bool drop_read_cache_ = false;
};

inline std::vector<uint32_t> divide_parts_mem(uint32_t n_sample, uint32_t num_parts) {
    const int parts = static_cast<int>(num_parts);
    std::vector<double> n_part(parts);
    std::vector<double> n_each_part(parts);
    n_part[0] = 1.0;
    n_each_part[0] = 1.0;
    for (int i = 1; i < parts; ++i) {
        const double prev = n_part[i - 1];
        const double cur = (std::sqrt(36.0 * prev * prev + 100.0) - 6.0 * prev) / 10.0;
        n_each_part[i] = cur;
        n_part[i] = prev + cur;
    }
    const double x1 = static_cast<double>(n_sample) / n_part[parts - 1];
    std::vector<uint32_t> div_parts(parts);
    uint32_t total_num = 0;
    for (int i = 0; i < parts - 1; ++i) {
        const uint32_t cur_n = static_cast<uint32_t>(std::round(n_each_part[i] * x1));
        total_num += cur_n;
        div_parts[i] = total_num - 1u;
    }
    div_parts[parts - 1] = n_sample - 1u;
    return div_parts;
}

inline std::vector<uint32_t> divide_parts(uint32_t from, uint32_t to, uint32_t num_parts) {
    std::vector<uint64_t> cum;
    cum.reserve(to - from + 1);
    for (uint64_t idx = from; idx <= to; ++idx) {
        cum.push_back((idx + 2ULL + from) * (idx + 1ULL - from) / 2ULL);
    }
    const uint64_t step = cum[to - from] / num_parts;
    std::vector<uint32_t> parts;
    parts.reserve(num_parts);
    for (uint32_t i = 1; i <= num_parts; ++i) {
        auto it = std::upper_bound(cum.begin(), cum.end(), step * i);
        if (it != cum.begin()) {
            --it;
            if (*it < step * i) ++it;
        }
        parts.push_back(static_cast<uint32_t>(it - cum.begin()) + from);
    }
    parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
    return parts;
}

inline std::pair<int, int> lower_tri_col_range(int cols, int tid, int nthreads) noexcept {
    if (cols <= 0 || nthreads <= 0) return {0, 0};
    if (nthreads == 1) return {0, cols};
    const double W = 0.5 * static_cast<double>(cols) * (static_cast<double>(cols) + 1.0);
    auto col_at = [&](int k) -> int {
        if (k <= 0) return 0;
        if (k >= nthreads) return cols;
        const double rem = W * (static_cast<double>(nthreads - k) / static_cast<double>(nthreads));
        const double tail = std::floor((-1.0 + std::sqrt(1.0 + 8.0 * rem)) * 0.5);
        return std::clamp(cols - static_cast<int>(tail), 0, cols);
    };
    return {col_at(tid), col_at(tid + 1)};
}

inline int solve_tile_budget_end(int rs, int full_re, double budget_elems) {
    const double drs = rs;
    const double disc = 1.0 + 4.0 * (2.0 * budget_elems + drs * (drs + 1.0));
    const int re = static_cast<int>(std::floor((-1.0 + std::sqrt(disc)) / 2.0));
    return std::min(re, full_re);
}

inline bool tile_budget_covers_whole(int full_rs, int full_re, double budget_elems) {
    const double drs = full_rs;
    const double dre = full_re;
    const double whole_elems = (dre * (dre + 1.0) - drs * (drs + 1.0)) / 2.0;
    return whole_elems <= budget_elems;
}

// 64-byte aligned leading dimension with 4K / 32K L1D set-aliasing avoidance.
inline int pad_strip_ld_i32(int min_ld) noexcept {
    int ld = std::max(16, ((min_ld + 15) / 16) * 16);
    if ((ld & 1023) == 0) ld += 16; // avoid 4 KiB / 32 KiB L1D stride aliasing (4 B * 1024 = 4 KiB)
    return ld;
}

inline int pad_strip_ld_f64(int min_ld) noexcept {
    int ld = std::max(8, ((min_ld + 7) / 8) * 8);
    if ((ld & 511) == 0) ld += 8; // avoid 4 KiB / 32 KiB L1D stride aliasing (8 B * 512 = 4 KiB)
    return ld;
}

inline int pad_kmajor_ldk(int k) noexcept {
    int ldk = std::max(64, ((k + 63) / 64) * 64);
    if ((ldk & 4095) == 0) ldk += 64; // avoid 4 KiB / 32 KiB L1D stride aliasing
    return ldk;
}

struct EngineConfig {
    int strip_rows = 4096;
    gcta_cli::GrmStripLayout layout = gcta_cli::GrmStripLayout::RowMajor;
    gcta_cli::GrmDiagMode diag_mode = gcta_cli::GrmDiagMode::Fused;
    int diag_block = 1024;
    int gemm_col_chunk = 0;
    size_t io_slot_bytes = gcta_grm_io::kMaxAsyncGrmSlotBytes;
    int n_marker_block = 1024;
    bool profile = false;
};

struct ProfileStats {
    double init_sec = 0.0;
    double io_wait_sec = 0.0;
    double count_decode_sec = 0.0;
    double miss_sec = 0.0;
    double gemm_sec = 0.0;
    double flush_sec = 0.0;
};

class BedInt8GrmRunner {
public:
    BedInt8GrmRunner(Pheno& pheno,
                     Marker& marker,
                     const gcta_cli::GrmMakeOptions& opts,
                     const std::string& out_prefix)
        : pheno_(pheno),
          marker_(marker),
          opts_(opts),
          out_prefix_(out_prefix) {}

    void run() {
        const auto t_init0 = Clock::now();
        init();
        if (cfg_.profile) {
            prof_.init_sec += std::chrono::duration<double>(Clock::now() - t_init0).count();
        }

        const bool is_sparse = opts_.sparse_cutoff.has_value();
        const float sparse_thresh = static_cast<float>(opts_.sparse_cutoff.value_or(-99.0));
        const bool write_fp16 = opts_.grm_fp16 && !is_sparse;
        const std::string grm_path = is_sparse
            ? (out_prefix_ + ".grm.sp")
            : (out_prefix_ + (write_fp16 ? ".grm.f16" : ".grm.bin"));
        const std::string N_path = out_prefix_ + ".grm.N.bin";
        const uint64_t n_out = num_individual_;
        const uint64_t body_off = write_fp16 ? gcta_grm_io::grm_f16_body_offset(n_out) : 0ULL;

        const size_t out_elem_bytes = (!is_sparse && write_fp16)
            ? sizeof(gcta_grm_io::GrmFp16)
            : sizeof(float);
        const double initial_bpe = allow_i32_mode_ ? 4.0 : 8.0;
        const double staged_bpe = 4.0 + static_cast<double>(out_elem_bytes);
        tiling_enabled_ = tile_budget_bytes_ > 0.0;
        if (tiling_enabled_) {
            const double budget_elems = tile_budget_bytes_ / initial_bpe;
            if (tile_budget_covers_whole(part_rs_, part_re_, budget_elems)) {
                LOGGER.info(0, "--GRM-tile-budget ({:.3f} GB) covers the whole GRM in one tile ({:.0f} B/elem); using the dense path instead.",
                            tile_budget_bytes_ / (1024.0 * 1024.0 * 1024.0), initial_bpe);
                tiling_enabled_ = false;
            }
        }
        if (tiling_enabled_ && total_bed_bytes_ >= (32ULL << 30)) {
            reader_->set_drop_read_cache(true);
        }

        planned_i32_slab_elems_ = 0;
        planned_stage_bytes_ = 0;
        staged_flush_enabled_ = false;
        if (tiling_enabled_) {
            if (is_sparse) {
                LOGGER.i(0, "Computing sparse GRM with a cutoff " + std::to_string(sparse_thresh) + "...");
            } else {
                LOGGER.i(0, write_fp16 ? "Computing tiled native FP16 GRM..." : "Computing tiled GRM...");
            }
            staged_flush_enabled_ = !is_sparse && allow_i32_mode_ &&
                                    (static_cast<double>(part_re_) * staged_bpe <= tile_budget_bytes_);
            const double tile_plan_bpe = staged_flush_enabled_ ? staged_bpe : initial_bpe;
            const double budget_elems = tile_budget_bytes_ / tile_plan_bpe;
            std::vector<std::pair<int, int>> est_ranges;
            std::vector<int> tmp_lds;
            std::vector<size_t> tmp_offs;
            int sim_rs = part_rs_;
            while (sim_rs < part_re_) {
                const int sim_re = solve_tile_budget_end(sim_rs, part_re_, budget_elems);
                if (sim_re <= sim_rs) {
                    const uint64_t need_bytes = static_cast<uint64_t>(sim_rs + 1) * static_cast<uint64_t>(tile_plan_bpe);
                    LOGGER.error(0, "--GRM-tile-budget is too small: the next row (sample offset {}) alone needs {:.3f} GB ({:.0f} B/elem), but the budget is only {:.3f} GB. Increase --GRM-tile-budget.",
                                 sim_rs,
                                 need_bytes / (1024.0 * 1024.0 * 1024.0),
                                 tile_plan_bpe,
                                 tile_budget_bytes_ / (1024.0 * 1024.0 * 1024.0));
                }
                if (allow_i32_mode_) {
                    planned_i32_slab_elems_ = std::max(
                        planned_i32_slab_elems_,
                        compute_tile_strip_layout(sim_rs, sim_re, /*as_i32=*/true, tmp_lds, tmp_offs));
                }
                if (staged_flush_enabled_) {
                    const uint64_t sim_tri =
                        (static_cast<uint64_t>(sim_re) * (static_cast<uint64_t>(sim_re) + 1ULL) -
                         static_cast<uint64_t>(sim_rs) * (static_cast<uint64_t>(sim_rs) + 1ULL)) / 2ULL;
                    planned_stage_bytes_ = std::max(
                        planned_stage_bytes_,
                        static_cast<size_t>(sim_tri) * out_elem_bytes);
                }
                est_ranges.emplace_back(sim_rs, sim_re);
                sim_rs = sim_re;
            }
            if (staged_flush_enabled_) {
                LOGGER.info(0, "Block-tiled GRM (strip-packed + overlapped {} staging, budget {:.3f} GB, {:.0f} B/elem): {} tile(s), rows/tile ranging {}-{}.",
                            write_fp16 ? "fp16" : "f32",
                            tile_budget_bytes_ / (1024.0 * 1024.0 * 1024.0),
                            tile_plan_bpe,
                            est_ranges.size(),
                            est_ranges.front().second - est_ranges.front().first,
                            est_ranges.back().second - est_ranges.back().first);
            } else {
                LOGGER.info(0, "Block-tiled GRM (strip-packed, budget {:.3f} GB, initial {:.0f} B/elem): {} tile(s), rows/tile ranging {}-{}.",
                            tile_budget_bytes_ / (1024.0 * 1024.0 * 1024.0),
                            tile_plan_bpe,
                            est_ranges.size(),
                            est_ranges.front().second - est_ranges.front().first,
                            est_ranges.back().second - est_ranges.back().first);
            }
        } else {
            LOGGER << "Computing dense GRM..." << std::endl;
        }

        UniqueFile sparse_out;
        std::unique_ptr<gcta_io::AsyncSeqWriter> grm_writer;
        UniqueFd N_fd;
        StagedTileFlusher staged_flusher;
        std::vector<float> flush_const_N;
        std::string flush_sparse_buf;

        double mtd_weight = 1.0;
        bool logged_missing_rebudget = false;
        int rs = part_rs_;
        skip_global_state_ = false;

        while (rs < part_re_) {
            int re = part_re_;
            bool known_tile_missing = false;
            if (tiling_enabled_) {
                const bool effective_missing = any_missing_seen_ && !opts_.impute_mean;
                if (effective_missing && staged_flush_enabled_) {
                    staged_flusher.wait_and_release();
                    staged_flush_enabled_ = false;
                }
                const double cur_bpe = effective_missing
                    ? 12.0
                    : (staged_flush_enabled_ ? staged_bpe : initial_bpe);
                if (effective_missing && !logged_missing_rebudget && rs > part_rs_) {
                    logged_missing_rebudget = true;
                    LOGGER.info(0, "  Missing genotypes detected; adapting remaining tile budget to 12 B/elem (double GRM + N).");
                }
                re = solve_tile_budget_end(rs, part_re_, tile_budget_bytes_ / cur_bpe);
                if (re <= rs) {
                    const uint64_t need_bytes = static_cast<uint64_t>(rs + 1) * static_cast<uint64_t>(cur_bpe);
                    LOGGER.error(0, "--GRM-tile-budget is too small: the next row (sample offset {}) alone needs {:.3f} GB ({:.0f} B/elem), but the budget is only {:.3f} GB. Increase --GRM-tile-budget.",
                                 rs,
                                 need_bytes / (1024.0 * 1024.0 * 1024.0),
                                 cur_bpe,
                                 tile_budget_bytes_ / (1024.0 * 1024.0 * 1024.0));
                }
                if (skip_global_state_ && effective_missing) {
                    for (int s = 0; s < re; ++s) {
                        if (sub_miss_[static_cast<size_t>(s)] > 0) {
                            known_tile_missing = true;
                            break;
                        }
                    }
                }
            }

            tile_rs_ = rs;
            tile_re_ = re;
            tile_rows_ = re - rs;
            tile_cols_ = re;

            const bool start_i32 = allow_i32_mode_ && !known_tile_missing;
            clear_strip_views();
            tile_has_N_ = false;
            N_pairs_ready_ = false;

            if (start_i32) {
                allocate_strips(/*as_i32=*/true);
                grm_is_i32_ = true;
                tile_i32_markers_ = 0;
                R_total_.assign(static_cast<size_t>(tile_cols_), 0.0);
                if (opts_.impute_mean) {
                    diag_err_total_.assign(static_cast<size_t>(tile_cols_), 0.0);
                } else {
                    diag_err_total_.clear();
                }
                S_total_ = 0.0;
            } else {
                allocate_strips(/*as_i32=*/false);
                grm_is_i32_ = false;
            }

            if (tiling_enabled_) {
                const double drs = rs;
                const double dre = re;
                const double tile_elems = (dre * (dre + 1.0) - drs * (drs + 1.0)) * 0.5;
                const double est_bpe = start_i32 ? 4.0 : (known_tile_missing ? 12.0 : 8.0);
                const double tile_gb = tile_elems * est_bpe / (1024.0 * 1024.0 * 1024.0);
                if (staged_flush_enabled_ && start_i32) {
                    const double stage_gb = tile_elems * static_cast<double>(out_elem_bytes) / (1024.0 * 1024.0 * 1024.0);
                    LOGGER.info(0, "  Tile rows {}-{} ({:.2f} GB strip-packed int32 GRM + {:.2f} GB {} stage)",
                                rs, re - 1, tile_gb, stage_gb, write_fp16 ? "fp16" : "f32");
                } else {
                    LOGGER.info(0, "  Tile rows {}-{} ({:.2f} GB strip-packed {})",
                                rs, re - 1, tile_gb,
                                start_i32 ? "int32 GRM" : (known_tile_missing ? "double GRM+N" : "double GRM"));
                }
            }

            first_accum_ = true;
            const bool is_first_tile = !skip_global_state_;

            if (is_first_tile) {
                use_tile1_cache_ = false;
                if (tiling_enabled_) {
                    tile1_valid_locs_.reserve(all_locs_.size());
                    tile1_valid_mu_.reserve(all_locs_.size());
                    tile1_valid_has_miss_.reserve(all_locs_.size());
                }
                reader_->stream(all_locs_, static_cast<uint32_t>(cfg_.n_marker_block),
                                cfg_.profile, prof_.io_wait_sec,
                                [&](const BedBlock& blk) { process_block(blk); });
            } else if (num_valid_markers_ > 0 && !tile1_valid_locs_.empty()) {
                tile1_cursor_ = 0;
                use_tile1_cache_ = true;
                reader_->stream(tile1_valid_locs_, static_cast<uint32_t>(cfg_.n_marker_block),
                                cfg_.profile, prof_.io_wait_sec,
                                [&](const BedBlock& blk) { process_block(blk); });
                use_tile1_cache_ = false;
            }

            skip_global_state_ = true;

            if (is_first_tile) {
                if (sd_sum_ > 0.0 && num_valid_markers_ > 0) {
                    mtd_weight = static_cast<double>(num_valid_markers_) / sd_sum_;
                }
                if (!tiling_enabled_) {
                    LOGGER << "  Used " << num_valid_markers_ << " valid SNPs." << std::endl;
                    LOGGER.i(0, "The GRM computation is completed.");
                    if (is_sparse) {
                        LOGGER.i(0, "Saving sparse GRM with a cutoff " + std::to_string(sparse_thresh) + "...");
                    } else if (write_fp16) {
                        LOGGER.i(0, "Saving native FP16 GRM...");
                    } else {
                        LOGGER.i(0, "Saving GRM...");
                    }
                }
                if (is_sparse) {
                    sparse_out = open_file(grm_path, "wb");
                    if (!sparse_out) LOGGER.error(0, "can't open {} to write", grm_path);
                    setvbuf(sparse_out.get(), nullptr, _IOFBF, 4 << 20);
                } else {
                    const uint64_t part_tri =
                        (static_cast<uint64_t>(part_re_) * (static_cast<uint64_t>(part_re_) + 1ULL) -
                         static_cast<uint64_t>(part_rs_) * (static_cast<uint64_t>(part_rs_) + 1ULL)) / 2ULL;
                    const size_t elem_b = write_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
                    const size_t total_grm_bytes = static_cast<size_t>(part_tri) * elem_b;
                    const size_t slot_cap = std::clamp<size_t>(
                        std::min<size_t>(cfg_.io_slot_bytes, std::max<size_t>(4ULL << 20, total_grm_bytes)),
                        1ULL << 20, cfg_.io_slot_bytes);
                    try {
                        grm_writer = std::make_unique<gcta_io::AsyncSeqWriter>(
                            grm_path, slot_cap, static_cast<off_t>(body_off));
                        if (write_fp16) {
                            gcta_grm_io::write_grm_f16_draft_prefix(*grm_writer, n_out);
                            fp16_diag_.assign(static_cast<size_t>(n_out), 0.0f);
                            fp16_trace_K_ = 0.0;
                            fp16_trace_K2_ = 0.0;
                        }
                    } catch (const std::exception& e) {
                        LOGGER.e(0, std::string("GrmBedInt8 output open failed: ") + e.what());
                    }
                    if (!write_fp16 || (any_missing_seen_ && !opts_.impute_mean)) {
                        N_fd = open_posix_write_trunc(N_path);
                        if (!N_fd) LOGGER.error(0, "can't open {} to write", N_path);
                    }
                }
            }

            const auto t_flush0 = Clock::now();
            const bool can_stage = staged_flush_enabled_ && !is_sparse && !tile_has_N_ &&
                                   !(first_accum_ || num_valid_markers_ == 0);
            if (can_stage) {
                try {
                    staged_flusher.wait_idle();
                    const size_t tile_bytes = pack_tile_to_stage(
                        staged_flusher.stage_slab(), planned_stage_bytes_, write_fp16, mtd_weight);
                    grm_writer->drain();
                    const uint64_t dre = static_cast<uint64_t>(tile_re_);
                    const uint64_t drs = static_cast<uint64_t>(tile_rs_);
                    const size_t tile_elems = static_cast<size_t>((dre * (dre + 1ULL) - drs * (drs + 1ULL)) / 2ULL);
                    staged_flusher.submit(
                        grm_writer->fd(),
                        grm_path,
                        tile_bytes,
                        N_fd.get(),
                        N_path,
                        N_fd ? tile_elems : 0ULL,
                        static_cast<float>(num_valid_markers_));
                } catch (const std::exception& e) {
                    LOGGER.e(0, std::string("GrmBedInt8 staged flush failed: ") + e.what());
                }
            } else {
                try {
                    staged_flusher.wait_and_release();
                } catch (const std::exception& e) {
                    LOGGER.e(0, std::string("GrmBedInt8 staged flush failed: ") + e.what());
                }
                flush_tile(grm_writer.get(), N_fd.get(), sparse_out.get(), grm_path, N_path,
                           sparse_thresh, is_sparse, write_fp16, mtd_weight,
                           flush_const_N, flush_sparse_buf);
            }
            if (cfg_.profile) {
                prof_.flush_sec += std::chrono::duration<double>(Clock::now() - t_flush0).count();
            }

            clear_strip_views();
            tile_has_N_ = false;
            rs = re;
        }
        release_slabs();

        const auto t_fin0 = Clock::now();
        try {
            staged_flusher.wait_and_release();
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("GrmBedInt8 staged flush failed: ") + e.what());
        }
        if (!is_sparse) {
            try {
                if (write_fp16) {
                    const bool effective_missing = any_missing_seen_ && !opts_.impute_mean;
                    gcta_grm_io::GrmFp16Header final_hdr{};
                    std::memcpy(final_hdr.magic, gcta_grm_io::kGrmF16MagicFinal, sizeof(final_hdr.magic));
                    final_hdr.version = gcta_grm_io::kGrmF16Version;
                    final_hdr.dtype = gcta_grm_io::kGrmF16DtypeFp16;
                    final_hdr.header_bytes = gcta_grm_io::kGrmF16HeaderSize;
                    final_hdr.flags = gcta_grm_io::kGrmF16FlagFinalized |
                                      gcta_grm_io::kGrmF16FlagHasTraces |
                                      gcta_grm_io::kGrmF16FlagHasDiagF32 |
                                      (!effective_missing ? gcta_grm_io::kGrmF16FlagZeroMissing : 0u) |
                                      gcta_grm_io::kGrmF16FlagVanRaden;
                    final_hdr.n = n_out;
                    final_hdr.valid_markers = num_valid_markers_;
                    if (!effective_missing) {
                        final_hdr.m_snps_mean = static_cast<double>(num_valid_markers_);
                    } else {
                        double miss_sum = 0.0;
                        for (uint64_t i = 0; i < n_out; ++i) {
                            miss_sum += static_cast<double>(sub_miss_[static_cast<size_t>(i)]);
                        }
                        final_hdr.m_snps_mean = (n_out > 0)
                            ? (static_cast<double>(num_valid_markers_) - miss_sum / static_cast<double>(n_out))
                            : 0.0;
                    }
                    final_hdr.id_hash = grm_id_hash_;
                    final_hdr.trace_K = fp16_trace_K_;
                    final_hdr.trace_K2 = fp16_trace_K2_;
                    final_hdr.diag_offset = gcta_grm_io::kGrmF16HeaderSize;
                    final_hdr.body_offset = body_off;
                    final_hdr.part_row_start = 0;
                    final_hdr.part_row_end = n_out;
                    final_hdr.mtd_weight = mtd_weight;
                    gcta_grm_io::finalize_grm_f16_file(*grm_writer, final_hdr, fp16_diag_.data(), n_out);
                } else {
                    grm_writer->finish();
                }
            } catch (const std::exception& e) {
                LOGGER.e(0, std::string("Failed finalizing [") + grm_path + "]: " + e.what());
            }
        }
        grm_writer.reset();
        N_fd.reset();
        sparse_out.reset();
        if (cfg_.profile) {
            prof_.flush_sec += std::chrono::duration<double>(Clock::now() - t_fin0).count();
        }

        if (tiling_enabled_) {
            LOGGER << "  Used " << num_valid_markers_ << " valid SNPs." << std::endl;
        }
        if (is_sparse) {
            LOGGER.i(0, "GRM has been saved in the file [" + out_prefix_ + ".grm.sp]");
        } else if (write_fp16) {
            const std::string fp16_note = !any_missing_seen_
                ? " (zero missing genotypes; .grm.N.bin omitted, m_snps=" + std::to_string(num_valid_markers_) + " in header)"
                : (opts_.impute_mean
                       ? " (missing genotypes mean-imputed at q=" + std::to_string(quant_q_) +
                             "; .grm.N.bin omitted, m_snps=" + std::to_string(num_valid_markers_) + " in header)"
                       : " (missingness present; per-pair counts in [" + N_path + "])");
            LOGGER.i(0, "Native FP16 GRM has been saved in the file [" + grm_path + "]" + fp16_note);
        } else {
            LOGGER.i(0, "GRM has been saved in the file [" + out_prefix_ + ".grm.bin]");
            LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + out_prefix_ + ".grm.N.bin]");
        }

        if (cfg_.profile) {
            LOGGER.info(0,
                "GRM profile (s): init={:.3f} io_wait={:.3f} count+decode={:.3f} miss={:.3f} gemm={:.3f} flush={:.3f}",
                prof_.init_sec, prof_.io_wait_sec, prof_.count_decode_sec,
                prof_.miss_sec, prof_.gemm_sec, prof_.flush_sec);
        }
    }

private:
    void init() {
        raw_ct_ = pheno_.count_raw();
        index_keep_ = pheno_.get_index_keep();
        keep_ct_ = static_cast<uint32_t>(index_keep_.size());
        rec_bytes_ = bed_simd::record_bytes(raw_ct_);

        if (keep_ct_ < raw_ct_) {
            const size_t mask_words = bed_simd::keep_mask_words(raw_ct_);
            keep_mask_ = make_aligned_uninitialized<uint64_t, 64>(mask_words);
            if (!keep_mask_) LOGGER.e(0, "GrmBedInt8: can't allocate keep mask buffer.");
            if (!bed_simd::build_keep_mask(index_keep_.data(), index_keep_.size(), raw_ct_, keep_mask_.get())) {
                LOGGER.e(0, "GrmBedInt8: --keep/--remove indices are invalid or not strictly increasing.");
            }
            keep_mask_ptr_ = keep_mask_.get();
        } else {
            keep_mask_ptr_ = nullptr;
        }

        if (!opts_.geno.bfile.empty()) {
            const std::string f = opts_.geno.bfile + ".bed";
            requireFileReadable("--bfile", f);
            bed_files_.push_back(f);
        } else if (!opts_.geno.mbfile.empty()) {
            bed_files_ = expandFileList(opts_.geno.mbfile, ".bed");
        } else {
            LOGGER.e(0, "GrmBedInt8: no --bfile/--mbfile specified.");
        }

        std::vector<uint64_t> file_snp_counts;
        std::vector<uint64_t> file_raw_starts;
        file_snp_counts.reserve(bed_files_.size());
        file_raw_starts.reserve(bed_files_.size());
        uint64_t raw_cursor = 0;
        for (size_t i = 0; i < bed_files_.size(); ++i) {
            const MarkerParam mp = marker_.getMarkerParams(static_cast<int>(i));
            file_raw_starts.push_back(raw_cursor);
            raw_cursor += mp.rawCountSNP;
            file_snp_counts.push_back(mp.rawCountSNP);

            const std::string& fname = bed_files_[i];
            UniqueFd fd = open_bed_read(fname);
            unsigned char magic[3] = {0, 0, 0};
            pread_full_throwing(fd.get(), magic, 3, 0, fname);
            if (magic[0] != 0x6c || magic[1] != 0x1b) {
                LOGGER.e(0, "[" + fname + "] is not a PLINK 1 binary genotype (.bed) file (bad magic number).");
            }
            if (magic[2] != 0x01) {
                LOGGER.e(0, "[" + fname + "] is in individual-major mode; only variant-major (SNP-major) .bed files are supported.");
            }
            struct stat st{};
            if (::fstat(fd.get(), &st) != 0) {
                LOGGER.e(0, "GrmBedInt8: can't stat [" + fname + "]: " + errno_str());
            }
            const uint64_t expected = 3ULL + static_cast<uint64_t>(rec_bytes_) * static_cast<uint64_t>(mp.rawCountSNP);
            if (st.st_size < 0 || static_cast<uint64_t>(st.st_size) != expected) {
                LOGGER.e(0, "Unexpected PLINK 1 .bed file size for [" + fname + "] (expected " +
                                std::to_string(expected) + " bytes for " + std::to_string(raw_ct_) +
                                " samples x " + std::to_string(mp.rawCountSNP) + " variants, found " +
                                std::to_string(static_cast<long long>(st.st_size)) + ").");
            }
            total_bed_bytes_ += static_cast<uint64_t>(st.st_size);
        }

        // MAF / missingness thresholds matching Geno
        const double opt_min_maf = opts_.marker_filter.maf.value_or(0.0);
        const double opt_max_maf = opts_.marker_filter.max_maf.value_or(0.5);
        const double opt_geno_rate = opts_.marker_filter.geno.value_or(1.0);
        if (opt_min_maf < 0.0) {
            LOGGER.e(0, "MAF can't be negative: " + std::to_string(opt_min_maf));
        }
        if (opt_max_maf <= 0.0 || opt_max_maf > 0.5) {
            LOGGER.e(0, "the value specified for --max-maf can't be negative or larger than 0.5");
        }
        min_maf_ = opt_min_maf * (1.0 - Constants::SMALL_EPSILON);
        max_maf_ = opt_max_maf * (1.0 + Constants::SMALL_EPSILON);
        if (max_maf_ < min_maf_) {
            LOGGER.e(0, "the value specified for --max-maf can't be smaller than that for --min-maf");
        }
        min_obs_rate_ = 1.0 - opt_geno_rate;

        if (opt_min_maf != 0.0 || opt_max_maf != 0.5 || opt_geno_rate != 1.0) {
            std::string prompt = "Threshold to filter variants:";
            bool first = true;
            if (opt_min_maf != 0.0) {
                prompt += " MAF > " + std::to_string(opt_min_maf);
                first = false;
            }
            if (opt_max_maf != 0.5) {
                prompt += std::string(first ? "" : ",") + " MAF < " + std::to_string(opt_max_maf);
                first = false;
            }
            if (opt_geno_rate != 1.0) {
                prompt += std::string(first ? "" : ",") + " missingness rate < " + std::to_string(opt_geno_rate);
            }
            LOGGER << prompt << "." << std::endl;
        }

        // Partitioning (--part TOTAL INDEX)
        uint32_t num_parts = opts_.part ? static_cast<uint32_t>(opts_.part->first) : 1u;
        const uint32_t part = opts_.part ? static_cast<uint32_t>(opts_.part->second) : 1u;
        const std::vector<uint32_t> parts = divide_parts_mem(keep_ct_, num_parts);
        if (parts.size() != num_parts) {
            LOGGER.w(0, "cannot divide into " + std::to_string(num_parts) + ". Use " + std::to_string(parts.size()) + " instead.");
            num_parts = static_cast<uint32_t>(parts.size());
        }
        uint32_t part_first = 0, part_second = 0;
        if (part > num_parts) {
            LOGGER.e(0, "cannot calculate when" + std::to_string(part) + " is larger than " + std::to_string(num_parts));
        } else if (part == 1) {
            part_first = 0;
            part_second = parts[0];
        } else {
            part_first = parts[part - 2] + 1;
            part_second = parts[part - 1];
        }
        part_rs_ = static_cast<int>(part_first);
        part_re_ = static_cast<int>(part_second) + 1;
        num_individual_ = static_cast<uint32_t>(part_re_ - part_rs_);
        num_grm_ = ((static_cast<uint64_t>(part_first) + part_second + 2ULL) * num_individual_) / 2ULL;
        grm_n_ = part_re_;
        tile_budget_bytes_ = opts_.tile_budget_gb
            ? (*opts_.tile_budget_gb * 1024.0 * 1024.0 * 1024.0)
            : 0.0;

        // Marker schedule (precomputed once, zero per-block scratch copying)
        const std::vector<uint32_t> auto_ext = marker_.get_extract_index_autosome();
        const std::vector<uint32_t>& raw_extract = marker_.get_extract_index();
        all_locs_.reserve(auto_ext.size());
        for (const uint32_t ext_i : auto_ext) {
            const uint32_t raw_i = raw_extract[ext_i];
            const uint32_t f_idx = static_cast<uint32_t>(marker_.getMIndex(raw_i));
            all_locs_.push_back(BedMarkerLoc{f_idx, static_cast<uint32_t>(raw_i - file_raw_starts[f_idx])});
        }
        allow_i32_mode_ = (all_locs_.size() <= kMaxInt32Markers);
        if (opts_.impute_mean) {
            if (all_locs_.size() > kMaxInt32Markers) {
                LOGGER.error(0, "--impute-mean requires at most {} autosomal markers (found {}).",
                             kMaxInt32Markers, all_locs_.size());
            }
            if (all_locs_.empty()) {
                quant_q_ = 1;
            } else {
                // Cap worst-case dot product M * (2*q)^2 <= 2.0e9 (< INT32_MAX and < float32 2^31 rounding boundary).
                const double max_q_d = std::floor(std::sqrt(2000000000.0 / static_cast<double>(all_locs_.size())) * 0.5);
                quant_q_ = std::clamp(static_cast<int>(max_q_d), 1, 63);
            }
        } else {
            quant_q_ = 1;
        }

        // Engine config defaults (RowMajor strips, fused diagonal, unchunked GEMM, 256 MiB I/O slots)
        const double dense_i32_bytes = static_cast<double>(num_grm_) * 4.0;
        const bool tight_mem_budget = is_tight_mem_budget();
        if (opts_.strip_rows.has_value()) {
            cfg_.strip_rows = *opts_.strip_rows;
        } else if (!tight_mem_budget && grm_n_ >= 98304 && omp_get_max_threads() >= 64) {
            cfg_.strip_rows = 8192;
        } else {
            cfg_.strip_rows = 4096;
        }
        cfg_.layout = opts_.strip_layout.value_or(gcta_cli::GrmStripLayout::RowMajor);
        cfg_.diag_mode = opts_.diag_mode.value_or(gcta_cli::GrmDiagMode::Fused);
        cfg_.diag_block = opts_.diag_block.value_or(1024);
        cfg_.gemm_col_chunk = opts_.gemm_col_chunk.value_or(0);
        cfg_.io_slot_bytes = static_cast<size_t>(opts_.io_slot_mb.value_or(256)) << 20;
        cfg_.profile = opts_.profile;

        if (opts_.n_markers.has_value()) {
            cfg_.n_marker_block = *opts_.n_markers;
        } else {
            const double eff_grm_bytes = (tile_budget_bytes_ > 0.0)
                ? std::min(dense_i32_bytes, tile_budget_bytes_)
                : dense_i32_bytes;
            const uint64_t kmajor_floor = tight_mem_budget
                ? std::max<uint64_t>(
                      256ULL << 20,
                      std::min<uint64_t>(2ULL << 30, static_cast<uint64_t>(tile_budget_bytes_ * 0.10)))
                : (2ULL << 30);
            const uint64_t kmajor_target = std::clamp<uint64_t>(
                static_cast<uint64_t>(eff_grm_bytes * 0.05),
                kmajor_floor,
                12ULL << 30);
            const uint64_t bed_slot_target = kmajor_target / 4ULL;
            const uint64_t k_by_kmajor = kmajor_target / std::max<uint64_t>(1ULL, static_cast<uint64_t>(grm_n_));
            const uint64_t k_by_bed = bed_slot_target / std::max<uint64_t>(1ULL, static_cast<uint64_t>(rec_bytes_));
            const uint64_t raw_k = std::min(k_by_kmajor, k_by_bed);
            cfg_.n_marker_block = static_cast<int>(std::clamp<uint64_t>((raw_k / 64ULL) * 64ULL, 1024ULL, 12288ULL));
        }

        if (opts_.fast) {
            std::string msg = "Using --fast preset:";
            if (!opts_.alg_explicit) msg += " --alg vanraden (INT8 VNNI)";
            if (opts_.grm_fp16) msg += " --grm-fp16 (native .grm.f16, exact float32 diagonal, zero-missing .grm.N.bin omitted)";
            LOGGER.i(0, msg + ".");
        }
        if (opts_.impute_mean) {
            LOGGER.i(0, "Mean-imputing missing genotypes in INT8 VanRaden GRM (quantization scale q=" +
                        std::to_string(quant_q_) + ", staying in 4 B/elem int32 mode).");
        }
        LOGGER.i(0, "Computing the genetic relationship matrix (GRM) v2  using " +
                    std::to_string(cfg_.n_marker_block) + " markers per block ...");
        LOGGER.i(0, "Subset " + std::to_string(part) + "/" + std::to_string(num_parts) +
                    ", no. subject " + std::to_string(part_first + 1) + "-" + std::to_string(part_second + 1));
        LOGGER.i(1, std::to_string(num_individual_) + " samples, " + std::to_string(marker_.count_extract()) +
                    " markers, " + std::to_string(num_grm_) + " GRM elements");

        if (opts_.strip_rows || opts_.strip_layout || opts_.diag_mode ||
            opts_.gemm_col_chunk || opts_.diag_block || opts_.io_slot_mb) {
            LOGGER.info(0, "GRM dev tuning: strip_rows={} layout={} diag_mode={} diag_block={} col_chunk={} io_slot_mb={}",
                        cfg_.strip_rows,
                        (cfg_.layout == gcta_cli::GrmStripLayout::RowMajor ? "rowmajor" : "colmajor"),
                        (cfg_.diag_mode == gcta_cli::GrmDiagMode::Fused ? "fused" : "syrk"),
                        cfg_.diag_block,
                        cfg_.gemm_col_chunk,
                        cfg_.io_slot_bytes >> 20);
        }

        // Write .grm.id and hash sample IDs
        const std::vector<std::string> out_id = pheno_.get_id(static_cast<int>(part_first), static_cast<int>(part_second));
        grm_id_hash_ = gcta_io::hash_sample_ids(out_id);
        const std::string o_grm_id = out_prefix_ + ".grm.id";
        gcta_grm_io::write_id_file_buffered(o_grm_id, out_id);
        LOGGER.i(0, "IDs for the GRM file have been saved in the file [" + o_grm_id + "]");

        // Preallocate reusable compute buffers ONCE
        const size_t mb = static_cast<size_t>(cfg_.n_marker_block);
        raw_mu_.resize(mb);
        raw_sd_.resize(mb);
        raw_nmiss_.resize(mb);
        raw_valid_.resize(mb);
        valid_buf_idx_.resize(mb);
        valid_mu_.resize(mb);
        valid_has_miss_.resize(mb);
        if (opts_.impute_mean) {
            valid_miss_q_.resize(mb);
            valid_mu_q_.resize(mb);
            valid_miss_err_q2_.resize(mb);
            diag_err_blk_.resize(static_cast<size_t>(grm_n_));
        }
        miss_col_of_.resize(mb);
        miss_markers_.reserve(mb);
        marker_miss_offset_.resize(mb + 1);

        const size_t eff_block_cap = std::min<size_t>(mb, std::max<size_t>(1, all_locs_.size()));
        const int max_ldK = pad_kmajor_ldk(static_cast<int>(eff_block_cap));
        const size_t kmajor_elems = static_cast<size_t>(max_ldK) * static_cast<size_t>(grm_n_);
        if (!kmajor_buf_.ensure(kmajor_elems, /*parallel_first_touch=*/true)) {
            LOGGER.e(0, "can't allocate enough memory for the transposed int8 genotype buffer.");
        }

        miss_col_ld_ = ((static_cast<size_t>(keep_ct_) + 63ULL) / 64ULL) * 64ULL;
        miss_words_full_ = (static_cast<size_t>(keep_ct_) + 63ULL) / 64ULL;
        sub_miss_.assign(static_cast<size_t>(keep_ct_) + 64ULL, 0);
        r_raw_.resize(static_cast<size_t>(grm_n_));
        r_tilde_.resize(static_cast<size_t>(grm_n_));
        sample_miss_offset_.resize(static_cast<size_t>(grm_n_) + 1);

        reader_ = std::make_unique<BedBlockReader>(
            std::move(bed_files_), std::move(file_snp_counts),
            rec_bytes_, mb, all_locs_.size());
    }

    bool is_tight_mem_budget() const noexcept {
        return tile_budget_bytes_ > 0.0 &&
               tile_budget_bytes_ < (48.0 * static_cast<double>(1ULL << 30));
    }

    void clear_strip_views() noexcept {
        strips_i32_.clear();
        strips_f64_.clear();
        strip_ld_.clear();
    }

    void release_slabs() noexcept {
        clear_strip_views();
        i32_slab_.reset();
        f64_slab_.reset();
        N_slab_.reset();
        tile_has_N_ = false;
    }

    size_t compute_tile_strip_layout(int rs,
                                     int re,
                                     bool as_i32,
                                     std::vector<int>& lds_out,
                                     std::vector<size_t>& offs_out) const {
        const int H = cfg_.strip_rows;
        const int tile_rows = re - rs;
        const int num_strips = (tile_rows + H - 1) / H;
        const size_t align_elems = as_i32 ? (64ULL / sizeof(int32_t)) : (64ULL / sizeof(double));
        const bool is_rowmajor = (cfg_.layout == gcta_cli::GrmStripLayout::RowMajor);

        lds_out.resize(static_cast<size_t>(num_strips));
        offs_out.resize(static_cast<size_t>(num_strips));
        size_t total_elems = 0;
        for (int s = 0; s < num_strips; ++s) {
            const int r0 = s * H;
            const int rows_s = std::min(H, tile_rows - r0);
            const int cols_s = rs + r0 + rows_s;
            const int ld = as_i32
                ? (is_rowmajor ? pad_strip_ld_i32(cols_s) : pad_strip_ld_i32(rows_s))
                : (is_rowmajor ? pad_strip_ld_f64(cols_s) : pad_strip_ld_f64(rows_s));
            const size_t elems = is_rowmajor
                ? (static_cast<size_t>(rows_s) * static_cast<size_t>(ld))
                : (static_cast<size_t>(cols_s) * static_cast<size_t>(ld));
            const size_t aligned_elems = ((elems + align_elems - 1ULL) / align_elems) * align_elems;
            lds_out[static_cast<size_t>(s)] = ld;
            offs_out[static_cast<size_t>(s)] = total_elems;
            total_elems += aligned_elems;
        }
        return total_elems;
    }

    void allocate_strips(bool as_i32) {
        const auto t_alloc0 = cfg_.profile ? Clock::now() : Clock::time_point{};
        clear_strip_views();
        std::vector<size_t> offs;
        const size_t need_elems = compute_tile_strip_layout(tile_rs_, tile_re_, as_i32, strip_ld_, offs);
        const size_t num_strips = strip_ld_.size();

        if (as_i32) {
            f64_slab_.reset();
            N_slab_.reset();
            tile_has_N_ = false;
            const size_t alloc_elems = std::max(need_elems, planned_i32_slab_elems_);
            if (!i32_slab_.ensure(alloc_elems, /*parallel_first_touch=*/true)) {
                release_slabs();
                LOGGER.error(0, "can't allocate enough memory for strip-packed int32 GRM ({:.2f} GiB across {} strip(s)).",
                             static_cast<double>(alloc_elems) * sizeof(int32_t) / static_cast<double>(1ULL << 30),
                             num_strips);
            }
            strips_i32_.resize(num_strips);
            for (size_t s = 0; s < num_strips; ++s) {
                strips_i32_[s] = i32_slab_.data() + offs[s];
            }
        } else {
            i32_slab_.reset();
            if (!f64_slab_.ensure(need_elems, /*parallel_first_touch=*/true)) {
                release_slabs();
                LOGGER.error(0, "can't allocate enough memory for strip-packed double GRM ({:.2f} GiB across {} strip(s)).",
                             static_cast<double>(need_elems) * sizeof(double) / static_cast<double>(1ULL << 30),
                             num_strips);
            }
            strips_f64_.resize(num_strips);
            for (size_t s = 0; s < num_strips; ++s) {
                strips_f64_[s] = f64_slab_.data() + offs[s];
            }
        }
        if (cfg_.profile) {
            prof_.init_sec += std::chrono::duration<double>(Clock::now() - t_alloc0).count();
        }
    }

    void ensure_N_allocated() {
        if (!N_pairs_ready_) {
            N_pairs_.clear();
            const int nth = omp_get_max_threads();
            const std::vector<uint32_t> tp = divide_parts(
                static_cast<uint32_t>(tile_rs_), static_cast<uint32_t>(tile_re_ - 1),
                static_cast<uint32_t>(nth));
            N_pairs_.emplace_back(tile_rs_, static_cast<int>(tp[0]));
            for (size_t i = 1; i < tp.size(); ++i) {
                N_pairs_.emplace_back(static_cast<int>(tp[i - 1] + 1), static_cast<int>(tp[i]));
            }
            N_pairs_ready_ = true;
        }
        if (tile_has_N_) return;
        const uint64_t tile_tri =
            (static_cast<uint64_t>(tile_re_) + static_cast<uint64_t>(tile_rs_) + 1ULL) *
            static_cast<uint64_t>(tile_rows_) / 2ULL;
        const uint64_t fill_N = (tile_tri + 1ULL) & ~1ULL;
        if (!N_slab_.ensure(static_cast<size_t>(fill_N), /*parallel_first_touch=*/false)) {
            LOGGER.error(0, "can't allocate enough memory to store packed N: {:.2f}GB required.",
                         fill_N * sizeof(uint32_t) / (1024.0 * 1024.0 * 1024.0));
        }
        tile_has_N_ = true;
        uint32_t* const N_ptr = std::assume_aligned<64>(N_slab_.data());
        const uint64_t rs_u = static_cast<uint64_t>(tile_rs_);
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < static_cast<int>(N_pairs_.size()); ++idx) {
            const auto [p_from, p_to] = N_pairs_[static_cast<size_t>(idx)];
            const uint64_t off_start = (static_cast<uint64_t>(p_from) + 1ULL + rs_u)
                                     * static_cast<uint64_t>(p_from - tile_rs_) / 2ULL;
            const uint64_t off_end = (static_cast<uint64_t>(p_to) + 2ULL + rs_u)
                                   * static_cast<uint64_t>(p_to + 1 - tile_rs_) / 2ULL;
            std::memset(N_ptr + off_start, 0, static_cast<size_t>(off_end - off_start) * sizeof(uint32_t));
        }
        if (fill_N > tile_tri) {
            std::memset(N_ptr + tile_tri, 0, static_cast<size_t>(fill_N - tile_tri) * sizeof(uint32_t));
        }
    }

    void promote_i32_to_f64() {
        if (!grm_is_i32_) return;
        if (tile_budget_bytes_ > 0.0) {
            const double drs = tile_rs_;
            const double dre = tile_re_;
            const double tile_elems = (dre * (dre + 1.0) - drs * (drs + 1.0)) * 0.5;
            if (tile_elems * 12.0 > tile_budget_bytes_) {
                LOGGER.warn(0, "Missing genotypes promoted strip-packed int32 GRM to double+N (~{:.2f} GB > --GRM-tile-budget {:.2f} GB); lower --GRM-tile-budget if memory is tight (or pass --impute-mean to stay at 4 B/elem INT8).",
                            tile_elems * 12.0 / (1024.0 * 1024.0 * 1024.0),
                            tile_budget_bytes_ / (1024.0 * 1024.0 * 1024.0));
            }
        }
        const int H = cfg_.strip_rows;
        const int num_strips = (tile_rows_ + H - 1) / H;
        if (first_accum_) {
            allocate_strips(/*as_i32=*/false);
        } else {
            std::vector<int> dst_lds;
            std::vector<size_t> dst_offs;
            const size_t f64_elems = compute_tile_strip_layout(tile_rs_, tile_re_, /*as_i32=*/false, dst_lds, dst_offs);
            if (!f64_slab_.ensure(f64_elems, /*parallel_first_touch=*/true)) {
                LOGGER.error(0, "can't allocate enough memory to promote strip-packed int32 GRM to double ({:.2f} GiB).",
                             static_cast<double>(f64_elems) * sizeof(double) / static_cast<double>(1ULL << 30));
            }
            strips_f64_.resize(static_cast<size_t>(num_strips));
            const double* const R_ptr = R_total_.data();
            const double S_tot = S_total_;
            const bool is_rowmajor = (cfg_.layout == gcta_cli::GrmStripLayout::RowMajor);

            for (int s = 0; s < num_strips; ++s) {
                const int r0 = s * H;
                const int rows_s = std::min(H, tile_rows_ - r0);
                const int global_r0 = tile_rs_ + r0;
                const int src_ld = strip_ld_[static_cast<size_t>(s)];
                const int dst_ld = dst_lds[static_cast<size_t>(s)];
                strips_f64_[static_cast<size_t>(s)] = f64_slab_.data() + dst_offs[static_cast<size_t>(s)];

                const int32_t* const src_base = std::assume_aligned<64>(strips_i32_[static_cast<size_t>(s)]);
                double* const dst_base = std::assume_aligned<64>(strips_f64_[static_cast<size_t>(s)]);

                if (is_rowmajor) {
                    #pragma omp parallel for schedule(dynamic, 16)
                    for (int r = 0; r < rows_s; ++r) {
                        const int global_r = global_r0 + r;
                        const int32_t* __restrict__ src_row = src_base + static_cast<size_t>(r) * src_ld;
                        double* __restrict__ dst_row = dst_base + static_cast<size_t>(r) * dst_ld;
                        const double R_i = R_ptr[global_r];
                        const int len = global_r + 1;
                        #pragma omp simd
                        for (int c = 0; c < len; ++c) {
                            dst_row[c] = (static_cast<double>(src_row[c]) - R_i) + (S_tot - R_ptr[c]);
                        }
                    }
                } else {
                    const double* const R_row = R_ptr + global_r0;
                    if (global_r0 > 0) {
                        #pragma omp parallel for schedule(static)
                        for (int c = 0; c < global_r0; ++c) {
                            const int32_t* __restrict__ src_col = src_base + static_cast<size_t>(c) * src_ld;
                            double* __restrict__ dst_col = dst_base + static_cast<size_t>(c) * dst_ld;
                            const double r_j_plus_S = S_tot - R_ptr[c];
                            #pragma omp simd
                            for (int r = 0; r < rows_s; ++r) {
                                dst_col[r] = static_cast<double>(src_col[r]) - R_row[r] + r_j_plus_S;
                            }
                        }
                    }
                    #pragma omp parallel
                    {
                        const int tid = omp_get_thread_num();
                        const int nth = omp_get_num_threads();
                        const auto [c_beg, c_end] = lower_tri_col_range(rows_s, tid, nth);
                        for (int c_local = c_beg; c_local < c_end; ++c_local) {
                            const int c = global_r0 + c_local;
                            const int32_t* __restrict__ src_col = src_base + static_cast<size_t>(c) * src_ld;
                            double* __restrict__ dst_col = dst_base + static_cast<size_t>(c) * dst_ld;
                            const double r_j_plus_S = S_tot - R_ptr[c];
                            #pragma omp simd
                            for (int r = c_local; r < rows_s; ++r) {
                                dst_col[r] = static_cast<double>(src_col[r]) - R_row[r] + r_j_plus_S;
                            }
                        }
                    }
                }
            }
            strip_ld_ = std::move(dst_lds);
            strips_i32_.clear();
            i32_slab_.reset();
        }
        grm_is_i32_ = false;
    }

    void N_thread(int grm_index_from, int grm_index_to, const uintptr_t* cur_cmask) {
        uint32_t* const N_base = std::assume_aligned<64>(N_slab_.data());
        const uint64_t rs_u = static_cast<uint64_t>(tile_rs_);
        const uint64_t start_pos = (static_cast<uint64_t>(grm_index_from) + 1ULL + rs_u)
                                 * static_cast<uint64_t>(grm_index_from - tile_rs_) / 2ULL;

        uint32_t* po_N = N_base + start_pos;
        const uintptr_t* p_cmask1 = cur_cmask + grm_index_from;
        for (int index_pair1 = grm_index_from; index_pair1 <= grm_index_to; ++index_pair1) {
            const uintptr_t* p_cmask2 = cur_cmask;
            const uintptr_t cmask1 = *p_cmask1++;
            if (cmask1) {
                const int count = index_pair1 + 1;
                int index_pair2 = 0;
                for (; index_pair2 + 3 < count; index_pair2 += 4) {
                    const uintptr_t c0 = p_cmask2[0];
                    const uintptr_t c1 = p_cmask2[1];
                    const uintptr_t c2 = p_cmask2[2];
                    const uintptr_t c3 = p_cmask2[3];
                    po_N[0] += static_cast<uint32_t>(std::popcount(cmask1 & c0));
                    po_N[1] += static_cast<uint32_t>(std::popcount(cmask1 & c1));
                    po_N[2] += static_cast<uint32_t>(std::popcount(cmask1 & c2));
                    po_N[3] += static_cast<uint32_t>(std::popcount(cmask1 & c3));
                    p_cmask2 += 4;
                    po_N += 4;
                }
                for (; index_pair2 < count; ++index_pair2) {
                    const uintptr_t cmask = cmask1 & (*p_cmask2++);
                    if (cmask) *po_N += static_cast<uint32_t>(std::popcount(cmask));
                    po_N++;
                }
            } else {
                po_N += index_pair1 + 1;
            }
        }
    }

    void process_block(const BedBlock& blk) {
        const auto t_cd0 = Clock::now();
        const int num_marker = static_cast<int>(blk.num_markers);
        const uint8_t* const raw_base = blk.buf.data();
        const uint32_t rb = rec_bytes_;
        const uint32_t raw_ct = raw_ct_;
        const uint32_t keep_ct = keep_ct_;
        const uint64_t* const keep_mask = keep_mask_ptr_;
        const int decode_rows = tile_cols_;

        int K = 0;
        double S_mu2 = 0.0;
        bool any_missing_full = false;
        int8_t* const kmajor_base = std::assume_aligned<64>(kmajor_buf_.data());
        int ldK = 64;

        // Stage 1 + 2 fused in ONE OpenMP parallel region per block:
        // count/filter -> compact SoA -> K-major decode
        #pragma omp parallel
        {
            if (!use_tile1_cache_) {
                #pragma omp for schedule(static)
                for (int i = 0; i < num_marker; ++i) {
                    const bed_simd::Counts c = bed_simd::count(raw_base + static_cast<size_t>(i) * rb,
                                                               keep_mask, raw_ct, keep_ct);
                    const uint32_t n_obs = keep_ct - c.nmiss;
                    if (n_obs == 0) {
                        raw_valid_[static_cast<size_t>(i)] = 0;
                        continue;
                    }
                    const uint32_t dosage = c.n1 + 2u * c.n2;
                    const double af = static_cast<double>(dosage) / static_cast<double>(2u * n_obs);
                    const double maf = std::min(af, 1.0 - af);
                    const double obs_rate = static_cast<double>(n_obs) / static_cast<double>(keep_ct);
                    const double mu = 2.0 * af;
                    const double sd = mu * (1.0 - af);
                    if (maf >= min_maf_ && maf <= max_maf_ && obs_rate >= min_obs_rate_ && sd >= 1.0e-50) {
                        raw_valid_[static_cast<size_t>(i)] = 1;
                        raw_mu_[static_cast<size_t>(i)] = mu;
                        raw_sd_[static_cast<size_t>(i)] = sd;
                        raw_nmiss_[static_cast<size_t>(i)] = c.nmiss;
                    } else {
                        raw_valid_[static_cast<size_t>(i)] = 0;
                    }
                }
            }

            #pragma omp single
            {
                const bool do_impute = opts_.impute_mean;
                const double q_d = static_cast<double>(quant_q_);
                const int max_q2 = 2 * quant_q_;
                if (!use_tile1_cache_) {
                    for (int i = 0; i < num_marker; ++i) {
                        if (raw_valid_[static_cast<size_t>(i)]) {
                            const double mu_m = raw_mu_[static_cast<size_t>(i)];
                            const uint8_t has_miss = (raw_nmiss_[static_cast<size_t>(i)] > 0) ? 1u : 0u;
                            valid_buf_idx_[static_cast<size_t>(K)] = i;
                            valid_mu_[static_cast<size_t>(K)] = mu_m;
                            valid_has_miss_[static_cast<size_t>(K)] = has_miss;
                            if (do_impute) {
                                const double mu_q = q_d * mu_m;
                                const int8_t mq = static_cast<int8_t>(std::clamp<int>(
                                    static_cast<int>(std::llround(mu_q)), 0, max_q2));
                                const double err = static_cast<double>(mq) - mu_q;
                                valid_miss_q_[static_cast<size_t>(K)] = mq;
                                valid_mu_q_[static_cast<size_t>(K)] = mu_q;
                                valid_miss_err_q2_[static_cast<size_t>(K)] = err * err;
                                S_mu2 += mu_q * mu_q;
                            } else {
                                S_mu2 += mu_m * mu_m;
                            }
                            sd_sum_ += raw_sd_[static_cast<size_t>(i)];
                            if (has_miss) any_missing_full = true;
                            if (tiling_enabled_) {
                                tile1_valid_locs_.push_back(all_locs_[static_cast<size_t>(processed_all_ + i)]);
                                tile1_valid_mu_.push_back(mu_m);
                                tile1_valid_has_miss_.push_back(has_miss);
                            }
                            ++K;
                        }
                    }
                    processed_all_ += static_cast<size_t>(num_marker);
                    if (!skip_global_state_) {
                        num_valid_markers_ += static_cast<uint32_t>(K);
                    }
                    if (any_missing_full) {
                        any_missing_seen_ = true;
                    }
                } else {
                    K = num_marker;
                    for (int m = 0; m < K; ++m) {
                        const double mu_m = tile1_valid_mu_[tile1_cursor_ + static_cast<size_t>(m)];
                        const uint8_t has_miss = tile1_valid_has_miss_[tile1_cursor_ + static_cast<size_t>(m)];
                        valid_buf_idx_[static_cast<size_t>(m)] = m;
                        valid_mu_[static_cast<size_t>(m)] = mu_m;
                        valid_has_miss_[static_cast<size_t>(m)] = has_miss;
                        if (do_impute) {
                            const double mu_q = q_d * mu_m;
                            const int8_t mq = static_cast<int8_t>(std::clamp<int>(
                                static_cast<int>(std::llround(mu_q)), 0, max_q2));
                            const double err = static_cast<double>(mq) - mu_q;
                            valid_miss_q_[static_cast<size_t>(m)] = mq;
                            valid_mu_q_[static_cast<size_t>(m)] = mu_q;
                            valid_miss_err_q2_[static_cast<size_t>(m)] = err * err;
                            S_mu2 += mu_q * mu_q;
                        } else {
                            S_mu2 += mu_m * mu_m;
                        }
                        if (has_miss) any_missing_full = true;
                    }
                    tile1_cursor_ += static_cast<size_t>(K);
                }
                ldK = pad_kmajor_ldk(K);
            }

            if (K > 0 && decode_rows > 0) {
                const int n_chunks64 = (decode_rows + 63) / 64;
                const int tid = omp_get_thread_num();
                const int nth = omp_get_num_threads();
                const int c_beg = (n_chunks64 * tid) / nth;
                const int c_end = (n_chunks64 * (tid + 1)) / nth;
                const int s_beg = c_beg * 64;
                const int s_end = std::min(decode_rows, c_end * 64);
                if (s_beg < s_end) {
                    const uint32_t start_raw = keep_mask
                        ? index_keep_[static_cast<size_t>(s_beg)]
                        : static_cast<uint32_t>(s_beg);
                    if (opts_.impute_mean) {
                        bed_simd::decode_i8_kmajor_band_quant(
                            raw_base, rb, valid_buf_idx_.data(), K,
                            keep_mask, raw_ct, start_raw, static_cast<uint32_t>(s_end - s_beg),
                            kmajor_base + static_cast<size_t>(s_beg) * ldK,
                            static_cast<size_t>(ldK),
                            static_cast<int8_t>(quant_q_), valid_miss_q_.data(),
                            valid_mu_q_.data(), r_raw_.data() + s_beg,
                            valid_miss_err_q2_.data(), diag_err_blk_.data() + s_beg);
                    } else {
                        bed_simd::decode_i8_kmajor_band(
                            raw_base, rb, valid_buf_idx_.data(), K,
                            keep_mask, raw_ct, start_raw, static_cast<uint32_t>(s_end - s_beg),
                            kmajor_base + static_cast<size_t>(s_beg) * ldK,
                            static_cast<size_t>(ldK),
                            valid_mu_.data(), r_raw_.data() + s_beg);
                    }
                }
            }
        }

        const auto t_cd1 = Clock::now();
        if (cfg_.profile) {
            prof_.count_decode_sec += std::chrono::duration<double>(t_cd1 - t_cd0).count();
        }
        if (K == 0) return;

        // Missingness extraction (ONLY when block has missing genotypes and not mean-imputing)
        bool tile_has_missing = false;
        const int H = cfg_.strip_rows;
        const int num_strips = (tile_rows_ + H - 1) / H;

        if (any_missing_full && !opts_.impute_mean) {
            const auto t_miss0 = Clock::now();
            miss_markers_.clear();
            for (int m = 0; m < K; ++m) {
                if (valid_has_miss_[static_cast<size_t>(m)]) {
                    miss_col_of_[static_cast<size_t>(m)] = static_cast<int>(miss_markers_.size());
                    miss_markers_.push_back(m);
                } else {
                    miss_col_of_[static_cast<size_t>(m)] = -1;
                }
            }
            const int n_miss_cols = static_cast<int>(miss_markers_.size());
            if (n_miss_cols > miss_cols_cap_) {
                miss_cols_i8_ = make_aligned_uninitialized<int8_t, 64>(
                    static_cast<size_t>(n_miss_cols) * miss_col_ld_);
                if (!miss_cols_i8_) {
                    LOGGER.e(0, "can't allocate enough memory for int8 missing-genotype columns.");
                }
                miss_bits_buf_.resize(static_cast<size_t>(n_miss_cols) * miss_words_full_);
                miss_cols_cap_ = n_miss_cols;
            }
            int8_t* const miss_cols = miss_cols_i8_.get();
            uint64_t* const miss_bits_base = miss_bits_buf_.data();

            // Decode each missing marker ONCE (no redundant count pass)
            #pragma omp parallel for schedule(dynamic, 4)
            for (int km = 0; km < n_miss_cols; ++km) {
                const int bidx = valid_buf_idx_[static_cast<size_t>(miss_markers_[static_cast<size_t>(km)])];
                bed_simd::I8Opts i8o;
                i8o.miss_bits = miss_bits_base + static_cast<size_t>(km) * miss_words_full_;
                bed_simd::decode_i8_nocount(raw_base + static_cast<size_t>(bidx) * rb,
                                            keep_mask, raw_ct, keep_ct,
                                            miss_cols + static_cast<size_t>(km) * miss_col_ld_, &i8o);
            }

            std::memcpy(r_tilde_.data(), r_raw_.data(), static_cast<size_t>(decode_rows) * sizeof(double));
            marker_miss_idx_.clear();
            const int num_words = (decode_rows + 63) / 64;
            const uint64_t last_word_mask = (decode_rows % 64 == 0)
                ? ~0ULL
                : ((1ULL << (decode_rows % 64)) - 1ULL);
            marker_miss_offset_[0] = 0;
            for (int m = 0; m < K; ++m) {
                const int km = miss_col_of_[static_cast<size_t>(m)];
                if (km >= 0) {
                    const double mu2 = valid_mu_[static_cast<size_t>(m)] * valid_mu_[static_cast<size_t>(m)];
                    const uint64_t* mw = miss_bits_base + static_cast<size_t>(km) * miss_words_full_;
                    for (int w = 0; w < num_words; ++w) {
                        uint64_t word = mw[w];
                        if (w == num_words - 1) word &= last_word_mask;
                        while (word != 0) {
                            const int bit = std::countr_zero(word);
                            const int s = w * 64 + bit;
                            r_tilde_[static_cast<size_t>(s)] += mu2;
                            marker_miss_idx_.push_back(s);
                            word &= (word - 1);
                        }
                    }
                }
                marker_miss_offset_[static_cast<size_t>(m + 1)] = static_cast<int>(marker_miss_idx_.size());
            }
            tile_has_missing = !marker_miss_idx_.empty();

            if (!tile_has_missing && !skip_global_state_) {
                const int num_words_all = (grm_n_ + 63) / 64;
                const uint64_t last_mask_all = (grm_n_ % 64 == 0)
                    ? ~0ULL
                    : ((1ULL << (grm_n_ % 64)) - 1ULL);
                #pragma omp parallel for schedule(static) if (num_words_all >= 64)
                for (int w = 0; w < num_words_all; ++w) {
                    const int base_s = w * 64;
                    for (int km = 0; km < n_miss_cols; ++km) {
                        uint64_t word = miss_bits_base[static_cast<size_t>(km) * miss_words_full_ + w];
                        if (w == num_words_all - 1) word &= last_mask_all;
                        while (word != 0) {
                            const int bit = std::countr_zero(word);
                            sub_miss_[static_cast<size_t>(base_s + bit)]++;
                            word &= (word - 1);
                        }
                    }
                }
            }

            if (tile_has_missing) {
                std::fill(sample_miss_offset_.begin(), sample_miss_offset_.begin() + decode_rows + 1, 0);
                for (int s : marker_miss_idx_) {
                    sample_miss_offset_[static_cast<size_t>(s + 1)]++;
                }
                for (int s = 0; s < decode_rows; ++s) {
                    sample_miss_offset_[static_cast<size_t>(s + 1)] += sample_miss_offset_[static_cast<size_t>(s)];
                }
                sample_miss_m_.resize(static_cast<size_t>(sample_miss_offset_[static_cast<size_t>(decode_rows)]));
                std::vector<int> write_pos(sample_miss_offset_.begin(),
                                           sample_miss_offset_.begin() + decode_rows);
                for (int m = 0; m < K; ++m) {
                    const int mb = marker_miss_offset_[static_cast<size_t>(m)];
                    const int me = marker_miss_offset_[static_cast<size_t>(m + 1)];
                    for (int idx = mb; idx < me; ++idx) {
                        const int s = marker_miss_idx_[static_cast<size_t>(idx)];
                        sample_miss_m_[static_cast<size_t>(write_pos[static_cast<size_t>(s)]++)] = m;
                    }
                }
            }

            if (cfg_.profile) {
                prof_.miss_sec += std::chrono::duration<double>(Clock::now() - t_miss0).count();
            }
        }

        if (grm_is_i32_ &&
            (tile_has_missing ||
             static_cast<uint64_t>(tile_i32_markers_) + static_cast<uint64_t>(K) > kMaxInt32Markers)) {
            promote_i32_to_f64();
        }

        const auto t_gemm0 = Clock::now();
        const bool first_accum = first_accum_;
        const bool is_rowmajor = (cfg_.layout == gcta_cli::GrmStripLayout::RowMajor);

        if (grm_is_i32_) {
            const int32_t beta = first_accum ? 0 : 1;
            int max_cols = std::max(H, static_cast<int>((2000000000ULL / static_cast<uint64_t>(std::max(1, ldK))) & ~63ULL));
            if (cfg_.gemm_col_chunk > 0) {
                max_cols = std::min(max_cols, cfg_.gemm_col_chunk);
            }

            for (int s = 0; s < num_strips; ++s) {
                const int r0 = s * H;
                const int rows = std::min(H, tile_rows_ - r0);
                const int global_r0 = tile_rs_ + r0;
                const int8_t* const A_strip = kmajor_base + static_cast<size_t>(global_r0) * ldK;
                int32_t* const strip_base = std::assume_aligned<64>(strips_i32_[static_cast<size_t>(s)]);
                const int s_ld = strip_ld_[static_cast<size_t>(s)];

                if (cfg_.diag_mode == gcta_cli::GrmDiagMode::Fused) {
                    const int total_cols = global_r0 + rows;
                    for (int c0 = 0; c0 < total_cols; c0 += max_cols) {
                        const int cols = std::min(max_cols, total_cols - c0);
                        const int8_t* const B_strip = kmajor_base + static_cast<size_t>(c0) * ldK;
                        if (is_rowmajor) {
                            gcta_gemm_u8s8s32_kmajor_nt_rowmajor_beta(
                                rows, cols, K, A_strip, ldK, B_strip,
                                strip_base + c0, s_ld, beta,
                                scratch_A_f32_, scratch_B_f32_, scratch_C_f32_);
                        } else {
                            gcta_gemm_u8s8s32_kmajor_nt_beta(
                                rows, cols, K, A_strip, ldK, B_strip,
                                strip_base + static_cast<size_t>(c0) * s_ld, s_ld, beta,
                                scratch_A_f32_, scratch_B_f32_, scratch_C_f32_);
                        }
                    }
                } else {
                    for (int c0 = 0; c0 < global_r0; c0 += max_cols) {
                        const int cols = std::min(max_cols, global_r0 - c0);
                        const int8_t* const B_strip = kmajor_base + static_cast<size_t>(c0) * ldK;
                        if (is_rowmajor) {
                            gcta_gemm_u8s8s32_kmajor_nt_rowmajor_beta(
                                rows, cols, K, A_strip, ldK, B_strip,
                                strip_base + c0, s_ld, beta,
                                scratch_A_f32_, scratch_B_f32_, scratch_C_f32_);
                        } else {
                            gcta_gemm_u8s8s32_kmajor_nt_beta(
                                rows, cols, K, A_strip, ldK, B_strip,
                                strip_base + static_cast<size_t>(c0) * s_ld, s_ld, beta,
                                scratch_A_f32_, scratch_B_f32_, scratch_C_f32_);
                        }
                    }
                    if (is_rowmajor) {
                        gcta_syrk_u8s8s32_lower_kmajor_nt_rowmajor_beta(
                            rows, K, A_strip, ldK,
                            strip_base + global_r0, s_ld, beta,
                            scratch_A_f32_, scratch_B_f32_, scratch_C_f32_, cfg_.diag_block);
                    } else {
                        gcta_syrk_u8s8s32_lower_kmajor_nt_beta(
                            rows, K, A_strip, ldK,
                            strip_base + static_cast<size_t>(global_r0) * s_ld, s_ld, beta,
                            scratch_A_f32_, scratch_C_f32_, cfg_.diag_block);
                    }
                }
            }

            S_total_ += S_mu2;
            double* __restrict__ R_tot = R_total_.data();
            const double* __restrict__ r_cur = r_raw_.data();
            if (opts_.impute_mean) {
                double* __restrict__ de_tot = diag_err_total_.data();
                const double* __restrict__ de_cur = diag_err_blk_.data();
                #pragma omp parallel for schedule(static) if (decode_rows >= 4096)
                for (int i = 0; i < decode_rows; ++i) {
                    R_tot[i] += r_cur[i];
                    de_tot[i] += de_cur[i];
                }
            } else {
                #pragma omp parallel for schedule(static) if (decode_rows >= 4096)
                for (int i = 0; i < decode_rows; ++i) {
                    R_tot[i] += r_cur[i];
                }
            }
            tile_i32_markers_ += static_cast<uint32_t>(K);
        } else {
            accumulate_f64_strips(K, decode_rows, S_mu2, tile_has_missing, first_accum, ldK, kmajor_base);
        }
        first_accum_ = false;
        if (cfg_.profile) {
            prof_.gemm_sec += std::chrono::duration<double>(Clock::now() - t_gemm0).count();
        }

        if (!tile_has_missing) return;

        // Accumulate packed triangular N matrix for missing pairs
        const auto t_n0 = Clock::now();
        ensure_N_allocated();
        const int marker_per_N = static_cast<int>(sizeof(uintptr_t) * CHAR_BIT);
        const int num_N_block = (K + marker_per_N - 1) / marker_per_N;
        const int active_samples = (!skip_global_state_) ? grm_n_ : decode_rows;
        const int num_N_sample_block = (active_samples + marker_per_N - 1) / marker_per_N;
        const size_t block_stride = static_cast<size_t>(num_N_sample_block) * static_cast<size_t>(marker_per_N);
        const size_t need_miss_words = block_stride * static_cast<size_t>(num_N_block);
        if (sample_miss_buf_.size() < need_miss_words) {
            sample_miss_buf_.resize(need_miss_words);
        }
        const uint64_t* const miss_bits_base = miss_bits_buf_.data();

        #pragma omp parallel
        {
            std::array<uintptr_t, 64> transpose_input{};
            alignas(32) plink2::VecW transpose_scratch[plink2::kPglBitTransposeBufvecs];

            #pragma omp for schedule(static)
            for (int j = 0; j < num_N_sample_block; ++j) {
                const int base_miss_index = j * marker_per_N;
                for (int i = 0; i < num_N_block; ++i) {
                    const int base_m = marker_per_N * i;
                    const int last_m = std::min(marker_per_N * (i + 1), K);
                    uintptr_t* block_buf = sample_miss_buf_.data() + static_cast<size_t>(i) * block_stride;
                    for (int k = base_m; k < last_m; ++k) {
                        const int km = miss_col_of_[static_cast<size_t>(k)];
                        transpose_input[static_cast<size_t>(k - base_m)] =
                            (km >= 0) ? static_cast<uintptr_t>(miss_bits_base[static_cast<size_t>(km) * miss_words_full_ + static_cast<size_t>(j)]) : 0UL;
                    }
                    for (int k = last_m; k < marker_per_N * (i + 1); ++k) {
                        transpose_input[static_cast<size_t>(k - base_m)] = 0UL;
                    }
                    plink2::TransposeBitblock(
                        transpose_input.data(), 1, 1,
                        static_cast<uint32_t>(marker_per_N), static_cast<uint32_t>(marker_per_N),
                        &block_buf[base_miss_index], transpose_scratch);
                    if (!skip_global_state_) {
                        for (int k = base_miss_index; k < base_miss_index + marker_per_N; ++k) {
                            sub_miss_[static_cast<size_t>(k)] += static_cast<uint32_t>(std::popcount(block_buf[k]));
                        }
                    }
                }
            }

            #pragma omp for schedule(static)
            for (int idx = 0; idx < static_cast<int>(N_pairs_.size()); ++idx) {
                const auto [p_from, p_to] = N_pairs_[static_cast<size_t>(idx)];
                for (int i = 0; i < num_N_block; ++i) {
                    N_thread(p_from, p_to, sample_miss_buf_.data() + static_cast<size_t>(i) * block_stride);
                }
            }
        }

        if (cfg_.profile) {
            prof_.miss_sec += std::chrono::duration<double>(Clock::now() - t_n0).count();
        }
    }

    void accumulate_f64_strips(int K, int decode_rows, double S_mu2,
                               bool block_has_missing, bool first_accum,
                               int ldK, const int8_t* kmajor_base) {
        const int H = cfg_.strip_rows;
        const int num_strips = (tile_rows_ + H - 1) / H;
        const double* const r_row_vec = block_has_missing ? r_tilde_.data() : r_raw_.data();
        const bool is_rowmajor = (cfg_.layout == gcta_cli::GrmStripLayout::RowMajor);

        if (is_rowmajor) {
            const bool tight_mem_budget = is_tight_mem_budget();
            const size_t max_tile_elems = tight_mem_budget ? (16ULL << 20) : (64ULL << 20);
            const int max_c_cap = tight_mem_budget ? 4096 : 16384;
            const int max_c_blk = std::max(64, std::min(max_c_cap, static_cast<int>(max_tile_elems / static_cast<size_t>(std::max(1, std::min(H, tile_rows_))))));
            const size_t need_tile = static_cast<size_t>(std::min(H, tile_rows_)) * static_cast<size_t>(max_c_blk);
            if (i32_tile_.size() < need_tile) i32_tile_.resize(need_tile);

            const int n_miss_cols = static_cast<int>(miss_markers_.size());
            std::vector<int> col_miss_b, col_miss_e, active_col_miss_kms;
            if (block_has_missing) {
                col_miss_b.resize(static_cast<size_t>(n_miss_cols));
                col_miss_e.resize(static_cast<size_t>(n_miss_cols));
                active_col_miss_kms.reserve(static_cast<size_t>(n_miss_cols));
            }
            const int8_t* const miss_cols_base = block_has_missing ? miss_cols_i8_.get() : nullptr;
            const size_t rflag_stride = ((static_cast<size_t>(n_miss_cols) + 63ULL) / 64ULL) * 64ULL;
            std::vector<uint8_t> rflag_all(
                block_has_missing ? static_cast<size_t>(omp_get_max_threads()) * rflag_stride : 0, 0);

            auto prepare_col_missing = [&](int col_beg, int col_end) {
                if (!block_has_missing) return;
                #pragma omp parallel for schedule(static) if (n_miss_cols >= 256)
                for (int km = 0; km < n_miss_cols; ++km) {
                    const int m = miss_markers_[static_cast<size_t>(km)];
                    const int mb = marker_miss_offset_[static_cast<size_t>(m)];
                    const int me = marker_miss_offset_[static_cast<size_t>(m + 1)];
                    if (mb == me) {
                        col_miss_b[static_cast<size_t>(km)] = col_miss_e[static_cast<size_t>(km)] = mb;
                        continue;
                    }
                    const int b = static_cast<int>(std::lower_bound(
                        marker_miss_idx_.begin() + mb, marker_miss_idx_.begin() + me, col_beg) - marker_miss_idx_.begin());
                    const int e = static_cast<int>(std::lower_bound(
                        marker_miss_idx_.begin() + b, marker_miss_idx_.begin() + me, col_end) - marker_miss_idx_.begin());
                    col_miss_b[static_cast<size_t>(km)] = b;
                    col_miss_e[static_cast<size_t>(km)] = e;
                }
                active_col_miss_kms.clear();
                for (int km = 0; km < n_miss_cols; ++km) {
                    if (col_miss_b[static_cast<size_t>(km)] < col_miss_e[static_cast<size_t>(km)]) {
                        active_col_miss_kms.push_back(km);
                    }
                }
            };

            auto accum_row_slice = [&](int i_global, int col_beg, int col_len,
                                       const int32_t* __restrict__ src_row,
                                       double* __restrict__ dst_row,
                                       uint8_t* row_miss_flag) {
                const double r_i = r_row_vec[static_cast<size_t>(i_global)];
                const double* __restrict__ r_col = r_raw_.data() + col_beg;
                if (first_accum) {
                    #pragma omp simd
                    for (int c = 0; c < col_len; ++c) {
                        dst_row[c] = (static_cast<double>(src_row[c]) - r_i) + (S_mu2 - r_col[c]);
                    }
                } else {
                    #pragma omp simd
                    for (int c = 0; c < col_len; ++c) {
                        dst_row[c] += (static_cast<double>(src_row[c]) - r_i) + (S_mu2 - r_col[c]);
                    }
                }
                if (!block_has_missing) return;

                const int rm_b = sample_miss_offset_[static_cast<size_t>(i_global)];
                const int rm_e = sample_miss_offset_[static_cast<size_t>(i_global + 1)];
                for (int idx = rm_b; idx < rm_e; ++idx) {
                    const int m = sample_miss_m_[static_cast<size_t>(idx)];
                    row_miss_flag[static_cast<size_t>(miss_col_of_[static_cast<size_t>(m)])] = 1;
                }
                const int col_end = col_beg + col_len;
                const int8_t* __restrict__ row_i_km = kmajor_base + static_cast<size_t>(i_global) * ldK;
                for (int km : active_col_miss_kms) {
                    const int m = miss_markers_[static_cast<size_t>(km)];
                    const double mu_m = valid_mu_[static_cast<size_t>(m)];
                    const double neg_mu2 = -mu_m * mu_m;
                    if (row_miss_flag[static_cast<size_t>(km)]) {
                        for (int idx = col_miss_b[static_cast<size_t>(km)]; idx < col_miss_e[static_cast<size_t>(km)]; ++idx) {
                            const int j = marker_miss_idx_[static_cast<size_t>(idx)];
                            if (j < col_end) {
                                dst_row[j - col_beg] += neg_mu2;
                                dst_row[j - col_beg] -= neg_mu2;
                            }
                        }
                    } else {
                        const double corr = static_cast<double>(row_i_km[m]) * mu_m + neg_mu2;
                        for (int idx = col_miss_b[static_cast<size_t>(km)]; idx < col_miss_e[static_cast<size_t>(km)]; ++idx) {
                            const int j = marker_miss_idx_[static_cast<size_t>(idx)];
                            if (j < col_end) {
                                dst_row[j - col_beg] += corr;
                            }
                        }
                    }
                }
                for (int idx = rm_b; idx < rm_e; ++idx) {
                    const int m = sample_miss_m_[static_cast<size_t>(idx)];
                    const int km = miss_col_of_[static_cast<size_t>(m)];
                    row_miss_flag[static_cast<size_t>(km)] = 0;
                    const double mu_m = valid_mu_[static_cast<size_t>(m)];
                    const int8_t* __restrict__ col_m = miss_cols_base + static_cast<size_t>(km) * miss_col_ld_ + col_beg;
                    #pragma omp simd
                    for (int c = 0; c < col_len; ++c) {
                        dst_row[c] += static_cast<double>(col_m[c]) * mu_m;
                    }
                }
            };

            for (int s = 0; s < num_strips; ++s) {
                const int r0 = s * H;
                const int rows = std::min(H, tile_rows_ - r0);
                const int global_r0 = tile_rs_ + r0;
                const int8_t* const A_strip = kmajor_base + static_cast<size_t>(global_r0) * ldK;
                double* const strip_base = std::assume_aligned<64>(strips_f64_[static_cast<size_t>(s)]);
                const int s_ld = strip_ld_[static_cast<size_t>(s)];

                for (int c0 = 0; c0 < global_r0; c0 += max_c_blk) {
                    const int cols = std::min(max_c_blk, global_r0 - c0);
                    gcta_gemm_u8s8s32_kmajor_nt_rowmajor_beta(
                        rows, cols, K, A_strip, ldK,
                        kmajor_base + static_cast<size_t>(c0) * ldK,
                        i32_tile_.data(), cols, 0,
                        scratch_A_f32_, scratch_B_f32_, scratch_C_f32_);
                    prepare_col_missing(c0, c0 + cols);
                    #pragma omp parallel
                    {
                        uint8_t* const rflag = block_has_missing
                            ? (rflag_all.data() + static_cast<size_t>(omp_get_thread_num()) * rflag_stride)
                            : nullptr;
                        #pragma omp for schedule(static)
                        for (int r = 0; r < rows; ++r) {
                            accum_row_slice(global_r0 + r, c0, cols,
                                            i32_tile_.data() + static_cast<size_t>(r) * cols,
                                            strip_base + static_cast<size_t>(r) * s_ld + c0,
                                            rflag);
                        }
                    }
                }

                // Diagonal strip sub-tiled so scratch stays bounded
                const int diag_blk = std::min(rows, max_c_blk);
                for (int d0 = 0; d0 < rows; d0 += diag_blk) {
                    const int db = std::min(diag_blk, rows - d0);
                    const int8_t* const A_d = kmajor_base + static_cast<size_t>(global_r0 + d0) * ldK;
                    gcta_syrk_u8s8s32_lower_kmajor_nt_rowmajor_beta(
                        db, K, A_d, ldK, i32_tile_.data(), db, 0,
                        scratch_A_f32_, scratch_B_f32_, scratch_C_f32_, cfg_.diag_block);
                    prepare_col_missing(global_r0 + d0, global_r0 + d0 + db);
                    #pragma omp parallel
                    {
                        uint8_t* const rflag = block_has_missing
                            ? (rflag_all.data() + static_cast<size_t>(omp_get_thread_num()) * rflag_stride)
                            : nullptr;
                        #pragma omp for schedule(guided)
                        for (int dr = 0; dr < db; ++dr) {
                            const int r_local = d0 + dr;
                            accum_row_slice(global_r0 + r_local, global_r0 + d0, dr + 1,
                                            i32_tile_.data() + static_cast<size_t>(dr) * db,
                                            strip_base + static_cast<size_t>(r_local) * s_ld + global_r0 + d0,
                                            rflag);
                        }
                    }
                    const int rem = rows - (d0 + db);
                    if (rem > 0) {
                        gcta_gemm_u8s8s32_kmajor_nt_rowmajor_beta(
                            rem, db, K,
                            kmajor_base + static_cast<size_t>(global_r0 + d0 + db) * ldK, ldK,
                            A_d, i32_tile_.data(), db, 0,
                            scratch_A_f32_, scratch_B_f32_, scratch_C_f32_);
                        #pragma omp parallel
                        {
                            uint8_t* const rflag = block_has_missing
                                ? (rflag_all.data() + static_cast<size_t>(omp_get_thread_num()) * rflag_stride)
                                : nullptr;
                            #pragma omp for schedule(static)
                            for (int rr = 0; rr < rem; ++rr) {
                                const int r_local = d0 + db + rr;
                                accum_row_slice(global_r0 + r_local, global_r0 + d0, db,
                                                i32_tile_.data() + static_cast<size_t>(rr) * db,
                                                strip_base + static_cast<size_t>(r_local) * s_ld + global_r0 + d0,
                                                rflag);
                            }
                        }
                    }
                }
            }
        } else {
            // ColMajor promoted f64 accumulation
            const int max_tile_dim = std::min(H, std::max(1, grm_n_));
            const size_t need_tile = static_cast<size_t>(max_tile_dim) * static_cast<size_t>(max_tile_dim);
            if (i32_tile_.size() < need_tile) i32_tile_.resize(need_tile);

            std::vector<int> row_miss_b, row_miss_e, active_row_miss_markers;
            if (block_has_missing) {
                row_miss_b.resize(static_cast<size_t>(K));
                row_miss_e.resize(static_cast<size_t>(K));
                active_row_miss_markers.reserve(static_cast<size_t>(K));
            }
            const int8_t* const miss_cols_base = block_has_missing ? miss_cols_i8_.get() : nullptr;

            auto prepare_row_missing = [&](int global_r0, int rows) {
                if (!block_has_missing) return;
                #pragma omp parallel for schedule(static) if (K >= 256)
                for (int m = 0; m < K; ++m) {
                    const int mb = marker_miss_offset_[static_cast<size_t>(m)];
                    const int me = marker_miss_offset_[static_cast<size_t>(m + 1)];
                    if (mb == me) {
                        row_miss_b[static_cast<size_t>(m)] = row_miss_e[static_cast<size_t>(m)] = mb;
                        continue;
                    }
                    const int b = static_cast<int>(std::lower_bound(
                        marker_miss_idx_.begin() + mb, marker_miss_idx_.begin() + me, global_r0) - marker_miss_idx_.begin());
                    const int e = static_cast<int>(std::lower_bound(
                        marker_miss_idx_.begin() + b, marker_miss_idx_.begin() + me, global_r0 + rows) - marker_miss_idx_.begin());
                    row_miss_b[static_cast<size_t>(m)] = b;
                    row_miss_e[static_cast<size_t>(m)] = e;
                }
                active_row_miss_markers.clear();
                for (int m = 0; m < K; ++m) {
                    if (row_miss_b[static_cast<size_t>(m)] < row_miss_e[static_cast<size_t>(m)]) {
                        active_row_miss_markers.push_back(m);
                    }
                }
            };

            auto accum_col_with_centering = [&](int j, int row_start, int len,
                                                const int32_t* __restrict__ src_col,
                                                double* __restrict__ dst_col) {
                const double r_j_plus_S = S_mu2 - r_raw_[static_cast<size_t>(j)];
                const double* __restrict__ r_row_slice = r_row_vec + row_start;
                if (first_accum) {
                    #pragma omp simd
                    for (int r = 0; r < len; ++r) {
                        dst_col[r] = static_cast<double>(src_col[r]) - r_row_slice[r] + r_j_plus_S;
                    }
                } else {
                    #pragma omp simd
                    for (int r = 0; r < len; ++r) {
                        dst_col[r] += static_cast<double>(src_col[r]) - r_row_slice[r] + r_j_plus_S;
                    }
                }
                if (!block_has_missing) return;

                const int cm_b = sample_miss_offset_[static_cast<size_t>(j)];
                const int cm_e = sample_miss_offset_[static_cast<size_t>(j + 1)];
                for (int idx = cm_b; idx < cm_e; ++idx) {
                    const int m = sample_miss_m_[static_cast<size_t>(idx)];
                    const double mu_m = valid_mu_[static_cast<size_t>(m)];
                    const double neg_mu2 = -mu_m * mu_m;
                    const int8_t* __restrict__ col_m = miss_cols_base + static_cast<size_t>(miss_col_of_[static_cast<size_t>(m)]) * miss_col_ld_ + row_start;
                    #pragma omp simd
                    for (int r = 0; r < len; ++r) {
                        dst_col[r] += static_cast<double>(col_m[r]) * mu_m + neg_mu2;
                    }
                    for (int r_idx = row_miss_b[static_cast<size_t>(m)]; r_idx < row_miss_e[static_cast<size_t>(m)]; ++r_idx) {
                        const int s = marker_miss_idx_[static_cast<size_t>(r_idx)];
                        if (s >= row_start) {
                            dst_col[s - row_start] -= neg_mu2;
                        }
                    }
                }

                const int8_t* __restrict__ row_j_kmajor = kmajor_base + static_cast<size_t>(j) * ldK;
                for (int m : active_row_miss_markers) {
                    const int8_t g_jm = row_j_kmajor[m];
                    if (g_jm != 0) {
                        const double corr = valid_mu_[static_cast<size_t>(m)] * static_cast<double>(g_jm);
                        for (int r_idx = row_miss_b[static_cast<size_t>(m)]; r_idx < row_miss_e[static_cast<size_t>(m)]; ++r_idx) {
                            const int s = marker_miss_idx_[static_cast<size_t>(r_idx)];
                            if (s >= row_start) {
                                dst_col[s - row_start] += corr;
                            }
                        }
                    }
                }
            };

            for (int s = 0; s < num_strips; ++s) {
                const int r0 = s * H;
                const int rows = std::min(H, tile_rows_ - r0);
                const int global_r0 = tile_rs_ + r0;
                const int8_t* const A_strip = kmajor_base + static_cast<size_t>(global_r0) * ldK;
                double* const strip_base = std::assume_aligned<64>(strips_f64_[static_cast<size_t>(s)]);
                const int s_ld = strip_ld_[static_cast<size_t>(s)];

                prepare_row_missing(global_r0, rows);

                for (int c0 = 0; c0 < global_r0; c0 += H) {
                    const int cols = std::min(H, global_r0 - c0);
                    gcta_gemm_u8s8s32_kmajor_nt_beta(
                        rows, cols, K, A_strip, ldK,
                        kmajor_base + static_cast<size_t>(c0) * ldK,
                        i32_tile_.data(), rows, 0,
                        scratch_A_f32_, scratch_B_f32_, scratch_C_f32_);

                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        accum_col_with_centering(c0 + c, global_r0, rows,
                                                 i32_tile_.data() + static_cast<size_t>(c) * rows,
                                                 strip_base + static_cast<size_t>(c0 + c) * s_ld);
                    }
                }

                gcta_syrk_u8s8s32_lower_kmajor_nt_beta(
                    rows, K, A_strip, ldK,
                    i32_tile_.data(), rows, 0,
                    scratch_A_f32_, scratch_C_f32_, cfg_.diag_block);

                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    const int nth = omp_get_num_threads();
                    const auto [c_beg, c_end] = lower_tri_col_range(rows, tid, nth);
                    for (int c = c_beg; c < c_end; ++c) {
                        const int j = global_r0 + c;
                        accum_col_with_centering(j, j, rows - c,
                                                 i32_tile_.data() + c + static_cast<size_t>(c) * rows,
                                                 strip_base + c + static_cast<size_t>(global_r0 + c) * s_ld);
                    }
                }
            }
        }
        (void)decode_rows;
    }

    size_t pack_tile_to_stage(HugeAlignedSlab<char>& stage_slab,
                              size_t planned_stage_bytes,
                              bool write_fp16,
                              double mtd_weight)
    {
        const uint64_t tile_e0 = static_cast<uint64_t>(tile_rs_) * (static_cast<uint64_t>(tile_rs_) + 1ULL) / 2ULL;
        const uint64_t tile_e1 = static_cast<uint64_t>(tile_re_) * (static_cast<uint64_t>(tile_re_) + 1ULL) / 2ULL;
        const size_t tile_elems = static_cast<size_t>(tile_e1 - tile_e0);
        const size_t out_elem_bytes = write_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
        const size_t tile_bytes = tile_elems * out_elem_bytes;

        const size_t ensure_bytes = std::max(tile_bytes, planned_stage_bytes);
        if (!stage_slab.ensure(ensure_bytes, /*parallel_first_touch=*/false)) {
            LOGGER.error(0, "can't allocate enough memory for staged GRM output buffer ({:.2f} GiB).",
                         static_cast<double>(ensure_bytes) / static_cast<double>(1ULL << 30));
        }

        const double q_scale = opts_.impute_mean
            ? (static_cast<double>(quant_q_) * static_cast<double>(quant_q_))
            : 1.0;
        const double inv_scale = (mtd_weight / static_cast<double>(num_valid_markers_)) / q_scale;
        const double* const diag_err_ptr = (opts_.impute_mean && grm_is_i32_ && !diag_err_total_.empty())
            ? diag_err_total_.data() : nullptr;
        const double* const R_ptr = grm_is_i32_ ? R_total_.data() : nullptr;
        const double S_tot = S_total_;
        const int H = cfg_.strip_rows;
        const bool is_rowmajor = (cfg_.layout == gcta_cli::GrmStripLayout::RowMajor);

        gcta_grm_io::GrmFp16* const out_f16 = write_fp16
            ? reinterpret_cast<gcta_grm_io::GrmFp16*>(stage_slab.data()) : nullptr;
        float* const out_f32 = !write_fp16
            ? reinterpret_cast<float*>(stage_slab.data()) : nullptr;

        double tr_K_tile = 0.0;
        double tr_K2_tile = 0.0;

        if (is_rowmajor) {
            #pragma omp parallel for schedule(dynamic, 32) reduction(+:tr_K_tile, tr_K2_tile)
            for (int pair1 = tile_rs_; pair1 < tile_re_; ++pair1) {
                const int strip_idx = (pair1 - tile_rs_) / H;
                const int local_r = (pair1 - tile_rs_) % H;
                const int s_ld = strip_ld_[static_cast<size_t>(strip_idx)];
                const size_t dst_off = static_cast<size_t>(
                    static_cast<uint64_t>(pair1) * (static_cast<uint64_t>(pair1) + 1ULL) / 2ULL - tile_e0);

                if (grm_is_i32_) {
                    const int32_t* __restrict__ src =
                        strips_i32_[static_cast<size_t>(strip_idx)] + static_cast<size_t>(local_r) * s_ld;
                    const double R_i = R_ptr[pair1];
                    if (out_f16) {
                        gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + dst_off;
                        double off_sq = 0.0;
                        #pragma omp simd reduction(+:off_sq)
                        for (int c = 0; c < pair1; ++c) {
                            const float v = static_cast<float>(((static_cast<double>(src[c]) - R_i) + (S_tot - R_ptr[c])) * inv_scale);
                            dst[c] = gcta_grm_io::f32_to_fp16(v);
                            const double vd = static_cast<double>(v);
                            off_sq += vd * vd;
                        }
                        double d_num = (static_cast<double>(src[pair1]) - R_i) + (S_tot - R_ptr[pair1]);
                        if (diag_err_ptr) {
                            d_num -= diag_err_ptr[pair1];
                            if (d_num < 0.0) d_num = 0.0;
                        }
                        const float dv = static_cast<float>(d_num * inv_scale);
                        dst[pair1] = gcta_grm_io::f32_to_fp16(dv);
                        fp16_diag_[static_cast<size_t>(pair1 - part_rs_)] = dv;
                        const double dd = static_cast<double>(dv);
                        tr_K_tile += dd;
                        tr_K2_tile += dd * dd + 2.0 * off_sq;
                    } else {
                        float* __restrict__ dst = out_f32 + dst_off;
                        #pragma omp simd
                        for (int c = 0; c < pair1; ++c) {
                            dst[c] = static_cast<float>(((static_cast<double>(src[c]) - R_i) + (S_tot - R_ptr[c])) * inv_scale);
                        }
                        double d_num = (static_cast<double>(src[pair1]) - R_i) + (S_tot - R_ptr[pair1]);
                        if (diag_err_ptr) {
                            d_num -= diag_err_ptr[pair1];
                            if (d_num < 0.0) d_num = 0.0;
                        }
                        dst[pair1] = static_cast<float>(d_num * inv_scale);
                    }
                } else {
                    const double* __restrict__ src =
                        strips_f64_[static_cast<size_t>(strip_idx)] + static_cast<size_t>(local_r) * s_ld;
                    if (out_f16) {
                        gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + dst_off;
                        double off_sq = 0.0;
                        #pragma omp simd reduction(+:off_sq)
                        for (int c = 0; c < pair1; ++c) {
                            const float v = static_cast<float>(src[c] * inv_scale);
                            dst[c] = gcta_grm_io::f32_to_fp16(v);
                            const double vd = static_cast<double>(v);
                            off_sq += vd * vd;
                        }
                        const float dv = static_cast<float>(src[pair1] * inv_scale);
                        dst[pair1] = gcta_grm_io::f32_to_fp16(dv);
                        fp16_diag_[static_cast<size_t>(pair1 - part_rs_)] = dv;
                        const double dd = static_cast<double>(dv);
                        tr_K_tile += dd;
                        tr_K2_tile += dd * dd + 2.0 * off_sq;
                    } else {
                        float* __restrict__ dst = out_f32 + dst_off;
                        #pragma omp simd
                        for (int c = 0; c <= pair1; ++c) {
                            dst[c] = static_cast<float>(src[c] * inv_scale);
                        }
                    }
                }
            }
        } else {
            const int num_strips = (tile_rows_ + H - 1) / H;
            std::vector<std::pair<int, int>> active_tiles;
            constexpr int kBlk = 64;
            const bool i32_mode = grm_is_i32_;

            for (int s = 0; s < num_strips; ++s) {
                const int strip_local_r0 = s * H;
                const int strip_h = std::min(H, tile_rows_ - strip_local_r0);
                const int strip_global_r0 = tile_rs_ + strip_local_r0;
                const int strip_global_r1 = strip_global_r0 + strip_h;
                const int s_ld = strip_ld_[static_cast<size_t>(s)];

                const double* const strip_f64 = (!grm_is_i32_ && !strips_f64_.empty())
                    ? std::assume_aligned<64>(strips_f64_[static_cast<size_t>(s)]) : nullptr;
                const int32_t* const strip_i32 = (grm_is_i32_ && !strips_i32_.empty())
                    ? std::assume_aligned<64>(strips_i32_[static_cast<size_t>(s)]) : nullptr;

                active_tiles.clear();
                for (int r0 = 0; r0 < strip_h; r0 += kBlk) {
                    const int r_len = std::min(kBlk, strip_h - r0);
                    const int max_c = std::min(strip_global_r1, strip_global_r0 + r0 + r_len);
                    for (int c0 = 0; c0 < max_c; c0 += kBlk) {
                        active_tiles.emplace_back(r0, c0);
                    }
                }
                const int n_active = static_cast<int>(active_tiles.size());

                #pragma omp parallel for reduction(+:tr_K_tile, tr_K2_tile) schedule(static)
                for (int ti = 0; ti < n_active; ++ti) {
                    const auto [r0, c0] = active_tiles[static_cast<size_t>(ti)];
                    const int r_len = std::min(kBlk, strip_h - r0);
                    const int c_len = std::min(kBlk, strip_global_r1 - c0);
                    const int global_r0 = strip_global_r0 + r0;
                    alignas(64) float tmp[kBlk][kBlk];
                    if (i32_mode) {
                        const double* __restrict__ R_row = R_ptr + global_r0;
                        for (int dc = 0; dc < c_len; ++dc) {
                            const int c = c0 + dc;
                            const int dr_start = std::max(0, c - global_r0);
                            if (dr_start >= r_len) break;
                            const int32_t* __restrict__ col_src = strip_i32 + static_cast<size_t>(c) * s_ld + r0;
                            const double r_j_plus_S = S_tot - R_ptr[c];
                            for (int dr = dr_start; dr < r_len; ++dr) {
                                tmp[dr][dc] = static_cast<float>(
                                    (static_cast<double>(col_src[dr]) - R_row[dr] + r_j_plus_S) * inv_scale);
                            }
                            if (diag_err_ptr && c >= global_r0 && c < global_r0 + r_len) {
                                const int dr_diag = c - global_r0;
                                double d_num = static_cast<double>(col_src[dr_diag]) - R_row[dr_diag] + r_j_plus_S - diag_err_ptr[c];
                                if (d_num < 0.0) d_num = 0.0;
                                tmp[dr_diag][dc] = static_cast<float>(d_num * inv_scale);
                            }
                        }
                    } else {
                        for (int dc = 0; dc < c_len; ++dc) {
                            const int c = c0 + dc;
                            const int dr_start = std::max(0, c - global_r0);
                            if (dr_start >= r_len) break;
                            const double* __restrict__ col_src = strip_f64 + static_cast<size_t>(c) * s_ld + r0;
                            for (int dr = dr_start; dr < r_len; ++dr) {
                                tmp[dr][dc] = static_cast<float>(col_src[dr] * inv_scale);
                            }
                        }
                    }
                    if (out_f16) {
                        for (int dr = 0; dr < r_len; ++dr) {
                            const int pair1 = global_r0 + dr;
                            const int valid_c = std::min(c_len, pair1 + 1 - c0);
                            if (valid_c <= 0) continue;
                            const size_t row_off = static_cast<size_t>(
                                static_cast<uint64_t>(pair1) * (static_cast<uint64_t>(pair1) + 1ULL) / 2ULL - tile_e0);
                            gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + row_off + c0;
                            const float* __restrict__ src = tmp[dr];
                            const bool has_diag = (c0 + valid_c - 1 == pair1);
                            const int off_end = has_diag ? (valid_c - 1) : valid_c;
                            double off_sq = 0.0;
                            #pragma omp simd reduction(+:off_sq)
                            for (int dc = 0; dc < off_end; ++dc) {
                                const float v = src[dc];
                                dst[dc] = gcta_grm_io::f32_to_fp16(v);
                                const double vd = static_cast<double>(v);
                                off_sq += vd * vd;
                            }
                            tr_K2_tile += 2.0 * off_sq;
                            if (has_diag) {
                                const float dv = src[off_end];
                                dst[off_end] = gcta_grm_io::f32_to_fp16(dv);
                                fp16_diag_[static_cast<size_t>(pair1 - part_rs_)] = dv;
                                const double dd = static_cast<double>(dv);
                                tr_K_tile += dd;
                                tr_K2_tile += dd * dd;
                            }
                        }
                    } else {
                        for (int dr = 0; dr < r_len; ++dr) {
                            const int pair1 = global_r0 + dr;
                            const int valid_c = std::min(c_len, pair1 + 1 - c0);
                            if (valid_c <= 0) continue;
                            const size_t row_off = static_cast<size_t>(
                                static_cast<uint64_t>(pair1) * (static_cast<uint64_t>(pair1) + 1ULL) / 2ULL - tile_e0);
                            std::memcpy(out_f32 + row_off + c0, tmp[dr], static_cast<size_t>(valid_c) * sizeof(float));
                        }
                    }
                }
            }
        }

        if (out_f16) {
            fp16_trace_K_ += tr_K_tile;
            fp16_trace_K2_ += tr_K2_tile;
        }
        return tile_bytes;
    }

    void flush_tile(gcta_io::AsyncSeqWriter* grm_writer,
                    int N_fd,
                    FILE* sparse_out,
                    const std::string& grm_path,
                    const std::string& N_path,
                    float thresh,
                    bool is_sparse,
                    bool write_fp16,
                    double mtd_weight,
                    std::vector<float>& const_N_buf,
                    std::string& sparse_buf) {
        const int tile_rs = tile_rs_;
        const int tile_re = tile_re_;
        const int tile_rows = tile_rows_;

        // Edge case: 0 valid markers accumulated
        if (first_accum_ || num_valid_markers_ == 0) {
            if (!is_sparse) {
                const uint64_t dre = static_cast<uint64_t>(tile_re);
                const uint64_t drs = static_cast<uint64_t>(tile_rs);
                const size_t total_elems = static_cast<size_t>((dre * (dre + 1ULL) - drs * (drs + 1ULL)) / 2ULL);
                const size_t elem_bytes = write_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
                const size_t total_grm_bytes = total_elems * elem_bytes;
                try {
                    size_t rem = total_grm_bytes;
                    while (rem > 0) {
                        AlignedRawBuffer<char, 64>& slot = grm_writer->acquire_slot();
                        const size_t c = std::min(rem, slot.size());
                        std::memset(slot.data(), 0, c);
                        grm_writer->submit_slot(c);
                        rem -= c;
                    }
                    if (N_fd >= 0) {
                        grm_writer->drain();
                        const size_t total_n_bytes = total_elems * sizeof(float);
                        const size_t buf_floats = std::min(total_elems, kConstNFloats);
                        if (const_N_buf.size() < buf_floats) const_N_buf.resize(buf_floats);
                        std::fill(const_N_buf.begin(), const_N_buf.begin() + static_cast<std::ptrdiff_t>(buf_floats), 0.0f);
                        size_t n_rem = total_n_bytes;
                        const size_t chunk_cap = buf_floats * sizeof(float);
                        while (n_rem > 0) {
                            const size_t c = std::min(n_rem, chunk_cap);
                            gcta_io::write_exact_throwing(N_fd, const_N_buf.data(), c, N_path);
                            n_rem -= c;
                        }
                    }
                } catch (const std::exception& e) {
                    LOGGER.e(0, std::string("GrmBedInt8 flush zero-marker write failed: ") + e.what());
                }
            } else if (0.0f >= thresh) {
                for (int pair1 = tile_rs; pair1 < tile_re; ++pair1) {
                    sparse_buf.clear();
                    for (int i = 0; i <= pair1; ++i) {
                        std::format_to(std::back_inserter(sparse_buf), "{}\t{}\t0\n", pair1, i);
                    }
                    if (!sparse_buf.empty() && fwrite(sparse_buf.data(), 1, sparse_buf.size(), sparse_out) != sparse_buf.size()) {
                        LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                    }
                }
            }
            return;
        }

        const bool zero_missing = !tile_has_N_;
        const double q_scale = opts_.impute_mean
            ? (static_cast<double>(quant_q_) * static_cast<double>(quant_q_))
            : 1.0;
        const double inv_scale = (mtd_weight / static_cast<double>(num_valid_markers_)) / q_scale;
        const double* const diag_err_ptr = (opts_.impute_mean && grm_is_i32_ && !diag_err_total_.empty())
            ? diag_err_total_.data() : nullptr;
        const size_t out_elem_bytes = (write_fp16 && !is_sparse) ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
        const size_t target_elems = std::max<size_t>(1, cfg_.io_slot_bytes / out_elem_bytes);
        const int flush_nt = is_sparse ? omp_get_max_threads() : std::max(1, omp_get_max_threads() - 1);

        const uint32_t* const N_base = tile_has_N_ ? std::assume_aligned<64>(N_slab_.data()) : nullptr;
        const uint32_t* const sub_miss_ptr = sub_miss_.data();
        const double* const R_ptr = grm_is_i32_ ? R_total_.data() : nullptr;
        const double S_tot = S_total_;
        const uint32_t valid_m = num_valid_markers_;
        const int H = cfg_.strip_rows;
        const bool is_rowmajor = (cfg_.layout == gcta_cli::GrmStripLayout::RowMajor);

        const auto get_N_row = [&](int pair1) -> const uint32_t* {
            const size_t start_pos = static_cast<size_t>(pair1 + 1 + tile_rs) *
                                     static_cast<size_t>(pair1 - tile_rs) / 2ULL;
            return N_base + start_pos;
        };

        auto emit_sparse_block = [&](int block_start, int block_rows, const float* out_f32, const std::vector<size_t>& row_off) {
            const int n_thr = (block_rows >= 64) ? omp_get_max_threads() : 1;
            if (n_thr > 1) {
                std::vector<std::string> thr_bufs(static_cast<size_t>(n_thr));
                #pragma omp parallel num_threads(n_thr)
                {
                    const int tid = omp_get_thread_num();
                    const int r_beg = (block_rows * tid) / n_thr;
                    const int r_end = (block_rows * (tid + 1)) / n_thr;
                    std::string& local_buf = thr_bufs[static_cast<size_t>(tid)];
                    for (int local_r = r_beg; local_r < r_end; ++local_r) {
                        const int pair1 = block_start + local_r;
                        const float* row = out_f32 + row_off[static_cast<size_t>(local_r)];
                        for (int i = 0; i <= pair1; ++i) {
                            if (row[i] >= thresh) {
                                std::format_to(std::back_inserter(local_buf),
                                               "{}\t{}\t{:.{}g}\n",
                                               pair1, i, row[i],
                                               std::numeric_limits<float>::digits10 + 2);
                            }
                        }
                    }
                }
                for (const auto& tb : thr_bufs) {
                    if (!tb.empty() && fwrite(tb.data(), 1, tb.size(), sparse_out) != tb.size()) {
                        LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                    }
                }
            } else {
                sparse_buf.clear();
                for (int local_r = 0; local_r < block_rows; ++local_r) {
                    const int pair1 = block_start + local_r;
                    const float* row = out_f32 + row_off[static_cast<size_t>(local_r)];
                    for (int i = 0; i <= pair1; ++i) {
                        if (row[i] >= thresh) {
                            std::format_to(std::back_inserter(sparse_buf),
                                           "{}\t{}\t{:.{}g}\n",
                                           pair1, i, row[i],
                                           std::numeric_limits<float>::digits10 + 2);
                        }
                    }
                }
                if (!sparse_buf.empty() && fwrite(sparse_buf.data(), 1, sparse_buf.size(), sparse_out) != sparse_buf.size()) {
                    LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                }
            }
        };

        AlignedRawBuffer<float, 64> sparse_packed_grm;

        try {
            if (is_rowmajor) {
                // Linear RowMajor pack across arbitrary strip boundaries up to slot capacity (zero 64x64 transpose!)
                std::vector<size_t> row_off;
                int r_cur = tile_rs;
                while (r_cur < tile_re) {
                    const size_t e0 = static_cast<size_t>(r_cur) * (r_cur + 1) / 2ULL;
                    const int r_end = gcta_grm_io::row_bound_for_cumulative(e0 + target_elems, r_cur, tile_re);
                    const int block_rows = r_end - r_cur;
                    row_off.resize(static_cast<size_t>(block_rows) + 1);
                    size_t total_packed = 0;
                    for (int r = 0; r < block_rows; ++r) {
                        row_off[static_cast<size_t>(r)] = total_packed;
                        total_packed += static_cast<size_t>(r_cur + r + 1);
                    }
                    row_off[static_cast<size_t>(block_rows)] = total_packed;

                    const size_t chunk_bytes = total_packed * out_elem_bytes;
                    char* out_raw = nullptr;
                    if (!is_sparse) {
                        AlignedRawBuffer<char, 64>& slot = grm_writer->acquire_slot();
                        if (slot.size() < chunk_bytes) slot.resize(chunk_bytes);
                        out_raw = slot.data();
                    } else {
                        if (sparse_packed_grm.size() < total_packed) sparse_packed_grm.resize(total_packed);
                        out_raw = reinterpret_cast<char*>(sparse_packed_grm.data());
                    }

                    float* const out_f32 = (!write_fp16 || is_sparse) ? reinterpret_cast<float*>(out_raw) : nullptr;
                    gcta_grm_io::GrmFp16* const out_f16 = (write_fp16 && !is_sparse)
                        ? reinterpret_cast<gcta_grm_io::GrmFp16*>(out_raw) : nullptr;

                    double tr_K_chunk = 0.0;
                    double tr_K2_chunk = 0.0;

                    #pragma omp parallel for num_threads(flush_nt) schedule(dynamic, 32) reduction(+:tr_K_chunk, tr_K2_chunk)
                    for (int pair1 = r_cur; pair1 < r_end; ++pair1) {
                        const int strip_idx = (pair1 - tile_rs) / H;
                        const int local_r = (pair1 - tile_rs) % H;
                        const int s_ld = strip_ld_[static_cast<size_t>(strip_idx)];
                        const size_t dst_off = row_off[static_cast<size_t>(pair1 - r_cur)];

                        if (zero_missing) {
                            if (grm_is_i32_) {
                                const int32_t* __restrict__ src = strips_i32_[static_cast<size_t>(strip_idx)] + static_cast<size_t>(local_r) * s_ld;
                                const double R_i = R_ptr[pair1];
                                if (out_f16) {
                                    gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + dst_off;
                                    double off_sq = 0.0;
                                    #pragma omp simd reduction(+:off_sq)
                                    for (int c = 0; c < pair1; ++c) {
                                        const float v = static_cast<float>(((static_cast<double>(src[c]) - R_i) + (S_tot - R_ptr[c])) * inv_scale);
                                        dst[c] = gcta_grm_io::f32_to_fp16(v);
                                        const double vd = static_cast<double>(v);
                                        off_sq += vd * vd;
                                    }
                                    double d_num = (static_cast<double>(src[pair1]) - R_i) + (S_tot - R_ptr[pair1]);
                                    if (diag_err_ptr) {
                                        d_num -= diag_err_ptr[pair1];
                                        if (d_num < 0.0) d_num = 0.0;
                                    }
                                    const float dv = static_cast<float>(d_num * inv_scale);
                                    dst[pair1] = gcta_grm_io::f32_to_fp16(dv);
                                    fp16_diag_[static_cast<size_t>(pair1 - part_rs_)] = dv;
                                    const double dd = static_cast<double>(dv);
                                    tr_K_chunk += dd;
                                    tr_K2_chunk += dd * dd + 2.0 * off_sq;
                                } else {
                                    float* __restrict__ dst = out_f32 + dst_off;
                                    #pragma omp simd
                                    for (int c = 0; c < pair1; ++c) {
                                        dst[c] = static_cast<float>(((static_cast<double>(src[c]) - R_i) + (S_tot - R_ptr[c])) * inv_scale);
                                    }
                                    double d_num = (static_cast<double>(src[pair1]) - R_i) + (S_tot - R_ptr[pair1]);
                                    if (diag_err_ptr) {
                                        d_num -= diag_err_ptr[pair1];
                                        if (d_num < 0.0) d_num = 0.0;
                                    }
                                    dst[pair1] = static_cast<float>(d_num * inv_scale);
                                }
                            } else {
                                const double* __restrict__ src = strips_f64_[static_cast<size_t>(strip_idx)] + static_cast<size_t>(local_r) * s_ld;
                                if (out_f16) {
                                    gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + dst_off;
                                    double off_sq = 0.0;
                                    #pragma omp simd reduction(+:off_sq)
                                    for (int c = 0; c < pair1; ++c) {
                                        const float v = static_cast<float>(src[c] * inv_scale);
                                        dst[c] = gcta_grm_io::f32_to_fp16(v);
                                        const double vd = static_cast<double>(v);
                                        off_sq += vd * vd;
                                    }
                                    const float dv = static_cast<float>(src[pair1] * inv_scale);
                                    dst[pair1] = gcta_grm_io::f32_to_fp16(dv);
                                    fp16_diag_[static_cast<size_t>(pair1 - part_rs_)] = dv;
                                    const double dd = static_cast<double>(dv);
                                    tr_K_chunk += dd;
                                    tr_K2_chunk += dd * dd + 2.0 * off_sq;
                                } else {
                                    float* __restrict__ dst = out_f32 + dst_off;
                                    #pragma omp simd
                                    for (int c = 0; c <= pair1; ++c) {
                                        dst[c] = static_cast<float>(src[c] * inv_scale);
                                    }
                                }
                            }
                        } else {
                            const double* __restrict__ src = strips_f64_[static_cast<size_t>(strip_idx)] + static_cast<size_t>(local_r) * s_ld;
                            const uint32_t sub_miss1 = valid_m - sub_miss_ptr[pair1];
                            const uint32_t* __restrict__ N_row = get_N_row(pair1);
                            if (out_f16) {
                                gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + dst_off;
                                double off_sq = 0.0;
                                for (int c = 0; c < pair1; ++c) {
                                    const uint32_t sub_N = N_row[c] + sub_miss1 - sub_miss_ptr[c];
                                    const float v = sub_N ? static_cast<float>((src[c] / static_cast<double>(sub_N)) * mtd_weight) : 0.0f;
                                    dst[c] = gcta_grm_io::f32_to_fp16(v);
                                    const double vd = static_cast<double>(v);
                                    off_sq += vd * vd;
                                }
                                const uint32_t diag_N = N_row[pair1] + sub_miss1 - sub_miss_ptr[pair1];
                                const float dv = diag_N ? static_cast<float>((src[pair1] / static_cast<double>(diag_N)) * mtd_weight) : 0.0f;
                                dst[pair1] = gcta_grm_io::f32_to_fp16(dv);
                                fp16_diag_[static_cast<size_t>(pair1 - part_rs_)] = dv;
                                const double dd = static_cast<double>(dv);
                                tr_K_chunk += dd;
                                tr_K2_chunk += dd * dd + 2.0 * off_sq;
                            } else {
                                float* __restrict__ dst = out_f32 + dst_off;
                                for (int c = 0; c <= pair1; ++c) {
                                    const uint32_t sub_N = N_row[c] + sub_miss1 - sub_miss_ptr[c];
                                    dst[c] = sub_N ? static_cast<float>((src[c] / static_cast<double>(sub_N)) * mtd_weight) : 0.0f;
                                }
                            }
                        }
                    }

                    if (out_f16) {
                        fp16_trace_K_ += tr_K_chunk;
                        fp16_trace_K2_ += tr_K2_chunk;
                    }
                    if (!is_sparse) {
                        grm_writer->submit_slot(chunk_bytes);
                    } else {
                        emit_sparse_block(r_cur, block_rows, out_f32, row_off);
                    }
                    r_cur = r_end;
                }
            } else {
                // ColMajor blocked 64x64 / 32x32 strip transpose flush
                std::vector<size_t> row_off(static_cast<size_t>(std::max(1, std::min(H, tile_rows))) + 1);
                std::vector<std::pair<int, int>> active_tiles;
                const int num_strips = (tile_rows + H - 1) / H;

                for (int s = 0; s < num_strips; ++s) {
                    const int strip_local_r0 = s * H;
                    const int strip_h = std::min(H, tile_rows - strip_local_r0);
                    const int strip_global_r0 = tile_rs + strip_local_r0;
                    const int strip_global_r1 = strip_global_r0 + strip_h;
                    const int s_ld = strip_ld_[static_cast<size_t>(s)];

                    const size_t raw_step = target_elems / std::max<size_t>(1, static_cast<size_t>(strip_global_r1));
                    const int sub_step_rows = static_cast<int>(
                        std::clamp<size_t>((raw_step / 64ULL) * 64ULL, 64ULL, static_cast<size_t>(std::max(64, strip_h))));

                    const double* const strip_f64 = (!grm_is_i32_ && !strips_f64_.empty())
                        ? std::assume_aligned<64>(strips_f64_[static_cast<size_t>(s)]) : nullptr;
                    const int32_t* const strip_i32 = (grm_is_i32_ && !strips_i32_.empty())
                        ? std::assume_aligned<64>(strips_i32_[static_cast<size_t>(s)]) : nullptr;

                    for (int block_start = strip_global_r0; block_start < strip_global_r1; block_start += sub_step_rows) {
                        const int block_end = std::min(block_start + sub_step_rows, strip_global_r1);
                        const int block_rows = block_end - block_start;
                        const int strip_row_r0 = block_start - strip_global_r0;

                        size_t total_packed = 0;
                        for (int r = 0; r < block_rows; ++r) {
                            row_off[static_cast<size_t>(r)] = total_packed;
                            total_packed += static_cast<size_t>(block_start + r + 1);
                        }
                        row_off[static_cast<size_t>(block_rows)] = total_packed;

                        const size_t strip_bytes = total_packed * out_elem_bytes;
                        char* out_raw = nullptr;
                        if (!is_sparse) {
                            AlignedRawBuffer<char, 64>& slot = grm_writer->acquire_slot();
                            if (slot.size() < strip_bytes) slot.resize(strip_bytes);
                            out_raw = slot.data();
                        } else {
                            if (sparse_packed_grm.size() < total_packed) sparse_packed_grm.resize(total_packed);
                            out_raw = reinterpret_cast<char*>(sparse_packed_grm.data());
                        }

                        float* const out_f32 = (!write_fp16 || is_sparse) ? reinterpret_cast<float*>(out_raw) : nullptr;
                        gcta_grm_io::GrmFp16* const out_f16 = (write_fp16 && !is_sparse)
                            ? reinterpret_cast<gcta_grm_io::GrmFp16*>(out_raw) : nullptr;

                        double tr_K_strip = 0.0;
                        double tr_K2_strip = 0.0;
                        const int pack_blk = zero_missing ? 64 : 32;
                        active_tiles.clear();
                        for (int r0 = 0; r0 < block_rows; r0 += pack_blk) {
                            const int r_len = std::min(pack_blk, block_rows - r0);
                            const int max_c = std::min(block_end, block_start + r0 + r_len);
                            for (int c0 = 0; c0 < max_c; c0 += pack_blk) {
                                active_tiles.emplace_back(r0, c0);
                            }
                        }
                        const int n_active = static_cast<int>(active_tiles.size());

                        if (zero_missing) {
                            constexpr int kBlk = 64;
                            const bool i32_mode = grm_is_i32_;
                            #pragma omp parallel for num_threads(flush_nt) reduction(+:tr_K_strip, tr_K2_strip) schedule(static)
                            for (int ti = 0; ti < n_active; ++ti) {
                                const auto [r0, c0] = active_tiles[static_cast<size_t>(ti)];
                                const int r_len = std::min(kBlk, block_rows - r0);
                                const int c_len = std::min(kBlk, block_end - c0);
                                const int global_r0 = block_start + r0;
                                const int local_r0 = strip_row_r0 + r0;
                                alignas(64) float tmp[kBlk][kBlk];
                                if (i32_mode) {
                                    const double* __restrict__ R_row = R_ptr + global_r0;
                                    for (int dc = 0; dc < c_len; ++dc) {
                                        const int c = c0 + dc;
                                        const int dr_start = std::max(0, c - global_r0);
                                        if (dr_start >= r_len) break;
                                        const int32_t* __restrict__ col_src = strip_i32 + static_cast<size_t>(c) * s_ld + local_r0;
                                        const double r_j_plus_S = S_tot - R_ptr[c];
                                        for (int dr = dr_start; dr < r_len; ++dr) {
                                            tmp[dr][dc] = static_cast<float>(
                                                (static_cast<double>(col_src[dr]) - R_row[dr] + r_j_plus_S) * inv_scale);
                                        }
                                        if (diag_err_ptr && c >= global_r0 && c < global_r0 + r_len) {
                                            const int dr_diag = c - global_r0;
                                            double d_num = static_cast<double>(col_src[dr_diag]) - R_row[dr_diag] + r_j_plus_S - diag_err_ptr[c];
                                            if (d_num < 0.0) d_num = 0.0;
                                            tmp[dr_diag][dc] = static_cast<float>(d_num * inv_scale);
                                        }
                                    }
                                } else {
                                    for (int dc = 0; dc < c_len; ++dc) {
                                        const int c = c0 + dc;
                                        const int dr_start = std::max(0, c - global_r0);
                                        if (dr_start >= r_len) break;
                                        const double* __restrict__ col_src = strip_f64 + static_cast<size_t>(c) * s_ld + local_r0;
                                        for (int dr = dr_start; dr < r_len; ++dr) {
                                            tmp[dr][dc] = static_cast<float>(col_src[dr] * inv_scale);
                                        }
                                    }
                                }
                                if (out_f16) {
                                    for (int dr = 0; dr < r_len; ++dr) {
                                        const int pair1 = global_r0 + dr;
                                        const int valid_c = std::min(c_len, pair1 + 1 - c0);
                                        if (valid_c <= 0) continue;
                                        gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + row_off[static_cast<size_t>(r0 + dr)] + c0;
                                        const float* __restrict__ src = tmp[dr];
                                        const bool has_diag = (c0 + valid_c - 1 == pair1);
                                        const int off_end = has_diag ? (valid_c - 1) : valid_c;
                                        double off_sq = 0.0;
                                        #pragma omp simd reduction(+:off_sq)
                                        for (int dc = 0; dc < off_end; ++dc) {
                                            const float v = src[dc];
                                            dst[dc] = gcta_grm_io::f32_to_fp16(v);
                                            const double vd = static_cast<double>(v);
                                            off_sq += vd * vd;
                                        }
                                        tr_K2_strip += 2.0 * off_sq;
                                        if (has_diag) {
                                            const float dv = src[off_end];
                                            dst[off_end] = gcta_grm_io::f32_to_fp16(dv);
                                            fp16_diag_[static_cast<size_t>(pair1 - part_rs_)] = dv;
                                            const double dd = static_cast<double>(dv);
                                            tr_K_strip += dd;
                                            tr_K2_strip += dd * dd;
                                        }
                                    }
                                } else {
                                    for (int dr = 0; dr < r_len; ++dr) {
                                        const int pair1 = global_r0 + dr;
                                        const int valid_c = std::min(c_len, pair1 + 1 - c0);
                                        if (valid_c <= 0) continue;
                                        std::memcpy(out_f32 + row_off[static_cast<size_t>(r0 + dr)] + c0,
                                                    tmp[dr], static_cast<size_t>(valid_c) * sizeof(float));
                                    }
                                }
                            }
                        } else {
                            constexpr int kBlk = 32;
                            #pragma omp parallel for num_threads(flush_nt) reduction(+:tr_K_strip, tr_K2_strip) schedule(static)
                            for (int ti = 0; ti < n_active; ++ti) {
                                const auto [r0, c0] = active_tiles[static_cast<size_t>(ti)];
                                const int r_len = std::min(kBlk, block_rows - r0);
                                const int c_len = std::min(kBlk, block_end - c0);
                                const int global_r0 = block_start + r0;
                                const int local_r0 = strip_row_r0 + r0;
                                alignas(64) double tmp_d[kBlk][kBlk];
                                for (int dc = 0; dc < c_len; ++dc) {
                                    const int c = c0 + dc;
                                    const int dr_start = std::max(0, c - global_r0);
                                    if (dr_start >= r_len) break;
                                    const double* __restrict__ col_src = strip_f64 + static_cast<size_t>(c) * s_ld + local_r0;
                                    for (int dr = dr_start; dr < r_len; ++dr) {
                                        tmp_d[dr][dc] = col_src[dr];
                                    }
                                }
                                for (int dr = 0; dr < r_len; ++dr) {
                                    const int pair1 = global_r0 + dr;
                                    const int valid_c = std::min(c_len, pair1 + 1 - c0);
                                    if (valid_c <= 0) continue;
                                    const uint32_t sub_miss1 = valid_m - sub_miss_ptr[pair1];
                                    const uint32_t* __restrict__ N_row = get_N_row(pair1) + c0;
                                    const uint32_t* __restrict__ sm_c = sub_miss_ptr + c0;
                                    const size_t dst_off = row_off[static_cast<size_t>(r0 + dr)] + static_cast<size_t>(c0);
                                    const double* __restrict__ td_row = tmp_d[dr];
                                    if (out_f16) {
                                        gcta_grm_io::GrmFp16* __restrict__ g_dst = out_f16 + dst_off;
                                        const bool has_diag = (c0 + valid_c - 1 == pair1);
                                        const int off_end = has_diag ? (valid_c - 1) : valid_c;
                                        double off_sq = 0.0;
                                        for (int dc = 0; dc < off_end; ++dc) {
                                            const uint32_t sub_N = N_row[dc] + sub_miss1 - sm_c[dc];
                                            const float v = sub_N ? static_cast<float>((td_row[dc] / static_cast<double>(sub_N)) * mtd_weight) : 0.0f;
                                            g_dst[dc] = gcta_grm_io::f32_to_fp16(v);
                                            const double vd = static_cast<double>(v);
                                            off_sq += vd * vd;
                                        }
                                        tr_K2_strip += 2.0 * off_sq;
                                        if (has_diag) {
                                            const uint32_t sub_N = N_row[off_end] + sub_miss1 - sm_c[off_end];
                                            const float dv = sub_N ? static_cast<float>((td_row[off_end] / static_cast<double>(sub_N)) * mtd_weight) : 0.0f;
                                            g_dst[off_end] = gcta_grm_io::f32_to_fp16(dv);
                                            fp16_diag_[static_cast<size_t>(pair1 - part_rs_)] = dv;
                                            const double dd = static_cast<double>(dv);
                                            tr_K_strip += dd;
                                            tr_K2_strip += dd * dd;
                                        }
                                    } else {
                                        float* __restrict__ g_dst = out_f32 + dst_off;
                                        for (int dc = 0; dc < valid_c; ++dc) {
                                            const uint32_t sub_N = N_row[dc] + sub_miss1 - sm_c[dc];
                                            g_dst[dc] = sub_N ? static_cast<float>((td_row[dc] / static_cast<double>(sub_N)) * mtd_weight) : 0.0f;
                                        }
                                    }
                                }
                            }
                        }

                        if (out_f16) {
                            fp16_trace_K_ += tr_K_strip;
                            fp16_trace_K2_ += tr_K2_strip;
                        }
                        if (!is_sparse) {
                            grm_writer->submit_slot(strip_bytes);
                        } else {
                            emit_sparse_block(block_start, block_rows, out_f32, row_off);
                        }
                    }
                }
            }

            // Write .grm.N.bin sequentially AFTER draining grm_writer (strictly 1 active write stream)
            if (!is_sparse && N_fd >= 0) {
                grm_writer->drain();
                const uint64_t dre = static_cast<uint64_t>(tile_re);
                const uint64_t drs = static_cast<uint64_t>(tile_rs);
                const size_t tile_total_elems = static_cast<size_t>((dre * (dre + 1ULL) - drs * (drs + 1ULL)) / 2ULL);

                if (zero_missing) {
                    const float n_val = static_cast<float>(valid_m);
                    const size_t buf_floats = std::min(tile_total_elems, kConstNFloats);
                    if (const_N_buf.size() < buf_floats || (buf_floats > 0 && const_N_buf[0] != n_val)) {
                        const_N_buf.assign(buf_floats, n_val);
                    }
                    size_t rem_bytes = tile_total_elems * sizeof(float);
                    const size_t cap_bytes = buf_floats * sizeof(float);
                    while (rem_bytes > 0) {
                        const size_t c = std::min(rem_bytes, cap_bytes);
                        gcta_io::write_exact_throwing(N_fd, const_N_buf.data(), c, N_path);
                        rem_bytes -= c;
                    }
                } else {
                    const size_t total_n_bytes = tile_total_elems * sizeof(float);
                    const size_t n_slot_cap = std::clamp<size_t>(
                        std::min<size_t>(cfg_.io_slot_bytes, std::max<size_t>(4ULL << 20, total_n_bytes)),
                        1ULL << 20, cfg_.io_slot_bytes);
                    gcta_io::AsyncSeqWriter n_writer(N_fd, N_path, n_slot_cap, /*take_ownership=*/false);
                    const size_t max_n_elems = std::max<size_t>(1, n_slot_cap / sizeof(float));
                    int r_cur = tile_rs;
                    while (r_cur < tile_re) {
                        const size_t e0 = static_cast<size_t>(r_cur) * (r_cur + 1) / 2ULL;
                        const int r_end = gcta_grm_io::row_bound_for_cumulative(e0 + max_n_elems, r_cur, tile_re);
                        const size_t e1 = static_cast<size_t>(r_end) * (r_end + 1) / 2ULL;
                        const size_t chunk_bytes = (e1 - e0) * sizeof(float);

                        AlignedRawBuffer<char, 64>& slot = n_writer.acquire_slot();
                        if (slot.size() < chunk_bytes) slot.resize(chunk_bytes);
                        auto* dst_base = reinterpret_cast<float*>(slot.data());

                        #pragma omp parallel for schedule(dynamic, 64)
                        for (int pair1 = r_cur; pair1 < r_end; ++pair1) {
                            const uint32_t sub_miss1 = valid_m - sub_miss_ptr[pair1];
                            const uint32_t* __restrict__ N_row = get_N_row(pair1);
                            const size_t dst_off = static_cast<size_t>(pair1) * (pair1 + 1) / 2ULL - e0;
                            float* __restrict__ n_dst = dst_base + dst_off;
                            #pragma omp simd
                            for (int c = 0; c <= pair1; ++c) {
                                n_dst[c] = static_cast<float>(N_row[c] + sub_miss1 - sub_miss_ptr[c]);
                            }
                        }
                        n_writer.submit_slot(chunk_bytes);
                        r_cur = r_end;
                    }
                    n_writer.finish();
                }
            }
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("GrmBedInt8 flush_tile failed: ") + e.what());
        }
    }

    Pheno& pheno_;
    Marker& marker_;
    const gcta_cli::GrmMakeOptions& opts_;
    const std::string& out_prefix_;

    EngineConfig cfg_;
    ProfileStats prof_;

    uint32_t raw_ct_ = 0;
    uint32_t keep_ct_ = 0;
    uint32_t rec_bytes_ = 0;
    std::vector<uint32_t> index_keep_;
    AlignedUniquePtr<uint64_t, 64> keep_mask_;
    const uint64_t* keep_mask_ptr_ = nullptr;

    std::vector<std::string> bed_files_;
    uint64_t total_bed_bytes_ = 0;
    std::vector<BedMarkerLoc> all_locs_;
    std::unique_ptr<BedBlockReader> reader_;

    double min_maf_ = 0.0;
    double max_maf_ = 0.5;
    double min_obs_rate_ = 0.0;

    int part_rs_ = 0;
    int part_re_ = 0;
    uint32_t num_individual_ = 0;
    uint64_t num_grm_ = 0;
    int grm_n_ = 0;
    double tile_budget_bytes_ = 0.0;
    bool tiling_enabled_ = false;
    bool allow_i32_mode_ = true;

    int tile_rs_ = 0;
    int tile_re_ = 0;
    int tile_rows_ = 0;
    int tile_cols_ = 0;
    bool grm_is_i32_ = false;
    bool first_accum_ = true;
    bool skip_global_state_ = false;
    uint32_t tile_i32_markers_ = 0;
    uint32_t num_valid_markers_ = 0;
    double sd_sum_ = 0.0;
    bool any_missing_seen_ = false;
    size_t processed_all_ = 0;

    // Strip accumulators (persistent 2 MiB-aligned slabs reused across tiles)
    HugeAlignedSlab<int32_t> i32_slab_;
    HugeAlignedSlab<double> f64_slab_;
    HugeAlignedSlab<uint32_t> N_slab_;
    bool tile_has_N_ = false;
    size_t planned_i32_slab_elems_ = 0;
    size_t planned_stage_bytes_ = 0;
    bool staged_flush_enabled_ = false;
    std::vector<int32_t*> strips_i32_;
    std::vector<double*> strips_f64_;
    std::vector<int> strip_ld_;
    std::vector<std::pair<int, int>> N_pairs_;
    bool N_pairs_ready_ = false;

    // SoA per-block scratch
    int quant_q_ = 1;
    std::vector<double> raw_mu_;
    std::vector<double> raw_sd_;
    std::vector<uint32_t> raw_nmiss_;
    std::vector<uint8_t> raw_valid_;
    std::vector<int> valid_buf_idx_;
    std::vector<double> valid_mu_;
    std::vector<uint8_t> valid_has_miss_;
    std::vector<int8_t> valid_miss_q_;
    std::vector<double> valid_mu_q_;
    std::vector<double> valid_miss_err_q2_;
    HugeAlignedSlab<int8_t> kmajor_buf_;

    // Centering + missingness scratch
    std::vector<double> r_raw_;
    std::vector<double> r_tilde_;
    std::vector<double> R_total_;
    std::vector<double> diag_err_blk_;
    std::vector<double> diag_err_total_;
    double S_total_ = 0.0;
    std::vector<uint32_t> sub_miss_;
    size_t miss_col_ld_ = 64;
    size_t miss_words_full_ = 1;
    int miss_cols_cap_ = 0;
    AlignedUniquePtr<int8_t, 64> miss_cols_i8_;
    std::vector<uint64_t> miss_bits_buf_;
    std::vector<int> miss_col_of_;
    std::vector<int> miss_markers_;
    std::vector<int> marker_miss_offset_;
    std::vector<int> marker_miss_idx_;
    std::vector<int> sample_miss_offset_;
    std::vector<int> sample_miss_m_;
    AlignedRawBuffer<uintptr_t, 64> sample_miss_buf_;
    AlignedRawBuffer<int32_t, 64> i32_tile_;

    // Fallback BLAS float32 scratch (unused when GCTA_HAS_AOCL_S8_GEMM)
    std::vector<float> scratch_A_f32_;
    std::vector<float> scratch_B_f32_;
    std::vector<float> scratch_C_f32_;

    // Tile-1 valid marker cache for Tiles 2..T
    std::vector<BedMarkerLoc> tile1_valid_locs_;
    std::vector<double> tile1_valid_mu_;
    std::vector<uint8_t> tile1_valid_has_miss_;
    size_t tile1_cursor_ = 0;
    bool use_tile1_cache_ = false;

    // .grm.f16 tracking
    uint64_t grm_id_hash_ = 0;
    std::vector<float> fp16_diag_;
    double fp16_trace_K_ = 0.0;
    double fp16_trace_K2_ = 0.0;
};

} // namespace gcta_grm_bed_detail

namespace gcta_grm {

void run_bed_int8_grm(Pheno& pheno,
                      Marker& marker,
                      const gcta_cli::GrmMakeOptions& opts,
                      const std::string& out_prefix) {
    gcta_grm_bed_detail::BedInt8GrmRunner runner(pheno, marker, opts, out_prefix);
    runner.run();
}

} // namespace gcta_grm
