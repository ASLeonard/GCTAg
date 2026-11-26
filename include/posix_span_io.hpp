/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   posix_span_io.hpp — Shared POSIX pread/preadv span coalescer & RAII fd layer.
   Used by BedBackend, GrmBedInt8, and pgen_hardcall to coalesce contiguous
   ascending file+destination spans into single syscalls and bridge forward byte
   gaps (<= max_gap_bytes) into a lazily allocated 64B-aligned discard buffer.
*/

#ifndef GCTA_POSIX_SPAN_IO_HPP
#define GCTA_POSIX_SPAN_IO_HPP

#include "mem.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

static_assert(sizeof(off_t) >= 8, "posix_span_io requires 64-bit file offsets");

namespace gcta_io {

class PosixUniqueFd {
public:
    PosixUniqueFd() noexcept = default;
    explicit PosixUniqueFd(int fd) noexcept : fd_(fd) {}
    ~PosixUniqueFd() { reset(); }

    PosixUniqueFd(const PosixUniqueFd&) = delete;
    PosixUniqueFd& operator=(const PosixUniqueFd&) = delete;

    PosixUniqueFd(PosixUniqueFd&& other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }

    PosixUniqueFd& operator=(PosixUniqueFd&& other) noexcept {
        if (this != &other) {
            reset(other.fd_);
            other.fd_ = -1;
        }
        return *this;
    }

    [[nodiscard]] int get() const noexcept { return fd_; }

    int release() noexcept {
        const int f = fd_;
        fd_ = -1;
        return f;
    }

    explicit operator bool() const noexcept { return fd_ >= 0; }

    void reset(int new_fd = -1) noexcept {
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = new_fd;
    }

private:
    int fd_ = -1;
};

inline std::string posix_errno_str() {
    return std::string(std::strerror(errno));
}

inline PosixUniqueFd open_posix_readonly_throwing(const std::string& path) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        throw std::runtime_error("can't open [" + path + "]: " + posix_errno_str());
    }
    return PosixUniqueFd(fd);
}

inline uint64_t stat_file_size_throwing(int fd, const std::string& path) {
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        throw std::runtime_error("can't stat [" + path + "]: " + posix_errno_str());
    }
    if (st.st_size < 0) {
        throw std::runtime_error("invalid file size on [" + path + "].");
    }
    return static_cast<uint64_t>(st.st_size);
}

inline void pread_full_throwing(int fd, void* dst, std::size_t len, uint64_t off, const std::string& path) {
    auto* p = static_cast<uint8_t*>(dst);
    constexpr std::size_t kMaxSinglePread = 1ULL << 30; // < INT_MAX (macOS) and 0x7ffff000 (Linux)
    while (len > 0) {
        const std::size_t req = std::min<std::size_t>(len, kMaxSinglePread);
        const ssize_t r = ::pread(fd, p, req, static_cast<off_t>(off));
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("read error on [" + path + "]: " + posix_errno_str());
        }
        if (r == 0) {
            throw std::runtime_error("unexpected end of file in [" + path + "].");
        }
        p += r;
        len -= static_cast<std::size_t>(r);
        off += static_cast<uint64_t>(r);
    }
}

inline void preadv_full_throwing(int fd, struct iovec* iov, int cnt, uint64_t off, const std::string& path) {
    if (cnt <= 0) return;
    if (cnt == 1) {
        pread_full_throwing(fd, iov[0].iov_base, iov[0].iov_len, off, path);
        return;
    }
#if defined(__linux__) || defined(__APPLE__)
    int idx = 0;
    uint64_t pos = off;
    while (idx < cnt) {
        while (idx < cnt && iov[idx].iov_len == 0) {
            ++idx;
        }
        if (idx >= cnt) return;
        const int rem_cnt = cnt - idx;
        if (rem_cnt == 1) {
            pread_full_throwing(fd, iov[idx].iov_base, iov[idx].iov_len, pos, path);
            return;
        }
        ssize_t r;
        do {
            r = ::preadv(fd, iov + idx, rem_cnt, static_cast<off_t>(pos));
        } while (r < 0 && errno == EINTR);
        if (r < 0) {
            throw std::runtime_error("read error on [" + path + "]: " + posix_errno_str());
        }
        if (r == 0) {
            throw std::runtime_error("unexpected end of file in [" + path + "].");
        }
        std::size_t done = static_cast<std::size_t>(r);
        pos += static_cast<uint64_t>(done);
        while (idx < cnt && done >= iov[idx].iov_len) {
            done -= iov[idx].iov_len;
            ++idx;
        }
        if (idx < cnt && done > 0) {
            iov[idx].iov_base = static_cast<uint8_t*>(iov[idx].iov_base) + done;
            iov[idx].iov_len -= done;
        }
    }
#else
    uint64_t pos = off;
    for (int i = 0; i < cnt; ++i) {
        pread_full_throwing(fd, iov[i].iov_base, iov[i].iov_len, pos, path);
        pos += static_cast<uint64_t>(iov[i].iov_len);
    }
#endif
}

struct FileByteSpan {
    uint64_t file_off = 0;
    std::size_t len = 0;
    uint8_t* dst = nullptr;
};

class SpanBatchCoalescer {
public:
    static constexpr uint64_t kDefaultMaxGapBytes = 2ULL << 20;
    static constexpr uint64_t kDefaultMaxBatchBytes = 1ULL << 30;

#if defined(IOV_MAX)
    static constexpr int kMaxIov = (IOV_MAX < 1024) ? IOV_MAX : 1024;
#else
    static constexpr int kMaxIov = 1024;
#endif

    SpanBatchCoalescer() { iov_.reserve(64); }
    SpanBatchCoalescer(const SpanBatchCoalescer&) = delete;
    SpanBatchCoalescer& operator=(const SpanBatchCoalescer&) = delete;
    SpanBatchCoalescer(SpanBatchCoalescer&&) noexcept = default;
    SpanBatchCoalescer& operator=(SpanBatchCoalescer&&) noexcept = default;

    void read_spans(int fd,
                    const std::string& path,
                    std::span<const FileByteSpan> spans,
                    uint64_t max_gap_bytes = kDefaultMaxGapBytes) {
        iov_.clear();
        uint64_t batch_start_off = 0;
        uint64_t next_file_off = 0;
        uint64_t batch_bytes = 0;

        const uint64_t eff_max_gap = std::min<uint64_t>(max_gap_bytes, kDefaultMaxBatchBytes);

        auto flush = [&]() {
            if (iov_.empty()) return;
            if (iov_.size() == 1) {
                pread_full_throwing(fd, iov_[0].iov_base, iov_[0].iov_len, batch_start_off, path);
            } else {
                preadv_full_throwing(fd, iov_.data(), static_cast<int>(iov_.size()), batch_start_off, path);
            }
            iov_.clear();
            batch_bytes = 0;
        };

        for (const FileByteSpan& s : spans) {
            if (s.len == 0) continue;
            if (s.len > kDefaultMaxBatchBytes) {
                flush();
                pread_full_throwing(fd, s.dst, s.len, s.file_off, path);
                continue;
            }

            bool start_new = iov_.empty();
            bool bridged_gap = false;
            if (!start_new) {
                if (s.file_off < next_file_off) {
                    start_new = true;
                } else if (s.file_off > next_file_off) {
                    const uint64_t gap_bytes = s.file_off - next_file_off;
                    if (gap_bytes <= eff_max_gap &&
                        iov_.size() + 2 <= static_cast<std::size_t>(kMaxIov) &&
                        batch_bytes + gap_bytes + s.len <= kDefaultMaxBatchBytes) {
                        if (discard_.size() < static_cast<std::size_t>(eff_max_gap)) {
                            // Allocated lazily on the first gap in this read_spans() call,
                            // before any iovec in iov_ references discard_.data().
                            discard_.resize(static_cast<std::size_t>(eff_max_gap));
                        }
                        iov_.push_back({discard_.data(), static_cast<std::size_t>(gap_bytes)});
                        batch_bytes += gap_bytes;
                        next_file_off = s.file_off;
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
                    static_cast<uint8_t*>(iov_.back().iov_base) + iov_.back().iov_len == s.dst;
                if (batch_bytes + s.len > kDefaultMaxBatchBytes ||
                    (!can_coalesce && iov_.size() + 1 > static_cast<std::size_t>(kMaxIov))) {
                    start_new = true;
                }
            }

            if (start_new) {
                flush();
                batch_start_off = s.file_off;
                next_file_off = s.file_off;
            }

            if (!iov_.empty() &&
                iov_.back().iov_base != discard_.data() &&
                static_cast<uint8_t*>(iov_.back().iov_base) + iov_.back().iov_len == s.dst) {
                iov_.back().iov_len += s.len;
            } else {
                iov_.push_back({s.dst, s.len});
            }
            batch_bytes += s.len;
            next_file_off = s.file_off + s.len;
        }
        flush();
    }

private:
    AlignedRawBuffer<uint8_t, 64> discard_;
    std::vector<struct iovec> iov_;
};

}  // namespace gcta_io

#endif  // GCTA_POSIX_SPAN_IO_HPP
