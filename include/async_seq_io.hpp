#pragma once

#include "mem.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gcta_io {

// Cap per-syscall I/O to 1 GiB (safely below INT_MAX on macOS/Darwin and
// 0x7ffff000 on Linux, where read()/pread()/write()/pwrite() cap or fail).
inline constexpr size_t kMaxIoChunkBytes = 1ULL << 30;

// Blocking read of exactly `count` bytes from `fd` into `buf`, retrying on
// EINTR and short reads. Throws std::runtime_error on EOF or hard error
// (safe inside worker threads; never calls LOGGER.e).
inline void read_exact_throwing(int fd, void* buf, size_t count, const std::string& path) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxIoChunkBytes, count - done);
        const ssize_t r = ::read(fd, p + done, step);
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("read() failed on [" + path + "]: " + std::string(std::strerror(errno)));
        }
        if (r == 0) {
            throw std::runtime_error("unexpected EOF reading [" + path + "] (expected " +
                                     std::to_string(count) + " bytes, got " + std::to_string(done) + ").");
        }
        done += static_cast<size_t>(r);
    }
}

// pread-based counterpart to read_exact_throwing: reads exactly `count` bytes
// starting at `offset`, retrying on EINTR and short reads.
inline void read_exact_at_throwing(int fd, void* buf, size_t count, off_t offset, const std::string& path) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxIoChunkBytes, count - done);
        const ssize_t r = ::pread(fd, p + done, step, offset + static_cast<off_t>(done));
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("pread() failed on [" + path + "]: " + std::string(std::strerror(errno)));
        }
        if (r == 0) {
            throw std::runtime_error("unexpected EOF reading [" + path + "] at offset " +
                                     std::to_string(static_cast<long long>(offset + static_cast<off_t>(done))) + ".");
        }
        done += static_cast<size_t>(r);
    }
}

// Blocking write of exactly `count` bytes from `buf` to `fd`, retrying on
// EINTR and short writes. Throws std::runtime_error on hard error.
inline void write_exact_throwing(int fd, const void* buf, size_t count, const std::string& path) {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxIoChunkBytes, count - done);
        const ssize_t w = ::write(fd, p + done, step);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("write() failed on [" + path + "]: " + std::string(std::strerror(errno)));
        }
        if (w == 0) {
            throw std::runtime_error("write() made zero progress on [" + path + "].");
        }
        done += static_cast<size_t>(w);
    }
}

// pwrite-based counterpart to write_exact_throwing: writes exactly `count`
// bytes at `offset`, retrying on EINTR and short writes.
inline void pwrite_exact_throwing(int fd, const void* buf, size_t count, off_t offset, const std::string& path) {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxIoChunkBytes, count - done);
        const ssize_t w = ::pwrite(fd, p + done, step, offset + static_cast<off_t>(done));
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("pwrite() failed on [" + path + "]: " + std::string(std::strerror(errno)));
        }
        if (w == 0) {
            throw std::runtime_error("pwrite() made zero progress on [" + path + "].");
        }
        done += static_cast<size_t>(w);
    }
}

// Deterministic 64-bit FNV-1a hash over arbitrary bytes (never returns 0).
inline uint64_t hash_bytes(const void* data, size_t len) noexcept {
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t h = 1469598103934665603ULL;
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= kPrime;
    }
    return h != 0 ? h : 1ULL;
}

// Deterministic 64-bit FNV-1a hash over ordered sample IDs ("FID\tIID"), each
// terminated by '\n' (byte-identical to MLMA_stream.cpp's v2 id_hash). Never 0.
inline uint64_t hash_sample_ids(const std::vector<std::string>& ids) noexcept {
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t h = 1469598103934665603ULL;
    for (const std::string& s : ids) {
        for (const unsigned char c : s) {
            h ^= c;
            h *= kPrime;
        }
        h ^= static_cast<unsigned char>('\n');
        h *= kPrime;
    }
    return h != 0 ? h : 1ULL;
}

inline std::string hash_hex(uint64_t h) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// AsyncSeqWriter
//
// HPC-safe double-buffered sequential POSIX writer: strictly 1 dedicated
// std::jthread writes 64-byte-aligned slots in FIFO order while the caller
// prepares/packs the next slot. Zero mmap, zero posix_fadvise, zero mbind.
// ---------------------------------------------------------------------------
class AsyncSeqWriter {
public:
    AsyncSeqWriter(const std::string& path, size_t slot_bytes, off_t initial_offset = 0)
        : path_(path), owns_fd_(true)
    {
        int flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        fd_ = ::open(path_.c_str(), flags, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open [" + path_ + "] for writing: " + std::string(std::strerror(errno)));
        }
        if (initial_offset > 0) {
            if (::lseek(fd_, initial_offset, SEEK_SET) == static_cast<off_t>(-1)) {
                const int err = errno;
                ::close(fd_);
                fd_ = -1;
                throw std::runtime_error("lseek() failed on [" + path_ + "]: " + std::string(std::strerror(err)));
            }
        }
        init_slots_and_start(slot_bytes);
    }

    AsyncSeqWriter(int fd, std::string path, size_t slot_bytes, bool take_ownership = false)
        : fd_(fd), path_(std::move(path)), owns_fd_(take_ownership)
    {
        if (fd_ < 0) {
            throw std::runtime_error("AsyncSeqWriter: invalid file descriptor for [" + path_ + "].");
        }
        init_slots_and_start(slot_bytes);
    }

    AsyncSeqWriter(const AsyncSeqWriter&) = delete;
    AsyncSeqWriter& operator=(const AsyncSeqWriter&) = delete;

    ~AsyncSeqWriter() noexcept {
        if (!finished_) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                aborting_ = true;
                closing_ = true;
            }
            writer_.request_stop();
            work_cv_.notify_all();
            if (writer_.joinable()) {
                writer_.join();
            }
            if (owns_fd_ && fd_ >= 0) {
                ::close(fd_);
            }
            fd_ = -1;
            finished_ = true;
        }
    }

    // Acquire the next free aligned slot buffer (blocks only if both slots are
    // currently queued/in-flight). Caller may resize the returned buffer if a
    // single chunk exceeds the initial slot_bytes.
    AlignedRawBuffer<char, 64>& acquire_slot() {
        if (finished_) {
            throw std::logic_error("AsyncSeqWriter::acquire_slot called after finish().");
        }
        if (held_slot_ != -1) {
            throw std::logic_error("AsyncSeqWriter::acquire_slot called while a slot is already acquired.");
        }
        std::unique_lock<std::mutex> lock(mu_);
        space_cv_.wait(lock, [&]() {
            return failed_ || !slot_queued_[0] || !slot_queued_[1];
        });
        if (failed_) {
            std::rethrow_exception(io_error_);
        }
        const int cand = !slot_queued_[next_slot_] ? next_slot_ : (1 - next_slot_);
        next_slot_ = 1 - cand;
        held_slot_ = cand;
        return slots_[static_cast<size_t>(cand)];
    }

    char* acquire_slot(size_t& cap_out) {
        AlignedRawBuffer<char, 64>& buf = acquire_slot();
        cap_out = buf.size();
        return buf.data();
    }

    // Hand the currently acquired slot (containing `nbytes` bytes) to the
    // sequential writer thread.
    void submit_slot(size_t nbytes) {
        if (finished_) {
            throw std::logic_error("AsyncSeqWriter::submit_slot called after finish().");
        }
        if (held_slot_ < 0) {
            throw std::logic_error("AsyncSeqWriter::submit_slot called without an acquired slot.");
        }
        const int slot = held_slot_;
        if (nbytes > slots_[static_cast<size_t>(slot)].size()) {
            throw std::out_of_range("AsyncSeqWriter::submit_slot nbytes exceeds slot buffer size.");
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            held_slot_ = -1;
            if (failed_) {
                std::rethrow_exception(io_error_);
            }
            if (nbytes == 0) {
                return;
            }
            slot_queued_[static_cast<size_t>(slot)] = true;
            slot_len_[static_cast<size_t>(slot)] = nbytes;
            ring_[static_cast<size_t>(ring_tail_)] = slot;
            ring_tail_ = (ring_tail_ + 1) & 1;
            ++ring_count_;
        }
        work_cv_.notify_one();
    }

    // Wait until all submitted slots have been written to disk; rethrows any
    // writer I/O error.
    void drain() {
        std::unique_lock<std::mutex> lock(mu_);
        space_cv_.wait(lock, [&]() {
            return failed_ || (ring_count_ == 0 && !writer_busy_);
        });
        if (failed_) {
            std::rethrow_exception(io_error_);
        }
    }

    // Drain all pending sequential writes, then pwrite() a fixed region (e.g.
    // finalized header + diagonal array at offset 0) without altering the
    // sequential file offset.
    void pwrite_after_drain(off_t offset, const void* buf, size_t count) {
        if (finished_) {
            throw std::logic_error("AsyncSeqWriter::pwrite_after_drain called after finish().");
        }
        drain();
        if (count > 0) {
            pwrite_exact_throwing(fd_, buf, count, offset, path_);
        }
    }

    // Drain all queued slots, join writer thread, close fd (if owned), and
    // rethrow any I/O error.
    void finish() {
        if (finished_) return;
        {
            std::lock_guard<std::mutex> lock(mu_);
            closing_ = true;
        }
        work_cv_.notify_all();
        if (writer_.joinable()) {
            writer_.join();
        }
        int close_err = 0;
        if (owns_fd_ && fd_ >= 0) {
            if (::close(fd_) != 0) {
                close_err = errno;
            }
        }
        fd_ = -1;
        finished_ = true;
        held_slot_ = -1;
        if (failed_) {
            std::rethrow_exception(io_error_);
        }
        if (close_err != 0) {
            throw std::runtime_error("close() failed on [" + path_ + "]: " + std::string(std::strerror(close_err)));
        }
    }

    int fd() const noexcept { return fd_; }
    const std::string& path() const noexcept { return path_; }
    size_t slot_capacity() const noexcept { return slots_[0].size(); }

private:
    void init_slots_and_start(size_t slot_bytes) {
        const size_t cap = std::max<size_t>(1, slot_bytes);
        try {
            slots_[0].resize(cap);
            slots_[1].resize(cap);
            writer_ = std::jthread([this](std::stop_token st) { writer_loop(st); });
        } catch (...) {
            if (owns_fd_ && fd_ >= 0) {
                ::close(fd_);
            }
            fd_ = -1;
            finished_ = true;
            throw;
        }
    }

    void writer_loop(std::stop_token st) {
        while (true) {
            int slot = -1;
            size_t nbytes = 0;
            {
                std::unique_lock<std::mutex> lock(mu_);
                work_cv_.wait(lock, st, [&]() {
                    return ring_count_ > 0 || closing_ || aborting_ || failed_;
                });
                if (aborting_ || failed_) return;
                if (ring_count_ == 0) {
                    if (closing_ || st.stop_requested()) return;
                    continue;
                }
                slot = ring_[static_cast<size_t>(ring_head_)];
                ring_head_ = (ring_head_ + 1) & 1;
                --ring_count_;
                nbytes = slot_len_[static_cast<size_t>(slot)];
                writer_busy_ = true;
            }
            try {
                if (nbytes > 0) {
                    write_exact_throwing(fd_, slots_[static_cast<size_t>(slot)].data(), nbytes, path_);
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    io_error_ = std::current_exception();
                    failed_ = true;
                    writer_busy_ = false;
                    slot_queued_[static_cast<size_t>(slot)] = false;
                }
                space_cv_.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                writer_busy_ = false;
                slot_queued_[static_cast<size_t>(slot)] = false;
            }
            space_cv_.notify_all();
        }
    }

    int fd_ = -1;
    std::string path_;
    bool owns_fd_ = false;

    std::array<AlignedRawBuffer<char, 64>, 2> slots_;
    std::array<bool, 2> slot_queued_{false, false};
    std::array<size_t, 2> slot_len_{0, 0};
    std::array<int, 2> ring_{0, 0};
    int ring_head_ = 0;
    int ring_tail_ = 0;
    int ring_count_ = 0;
    int next_slot_ = 0;
    int held_slot_ = -1;

    bool writer_busy_ = false;
    bool closing_ = false;
    bool aborting_ = false;
    bool finished_ = false;
    bool failed_ = false;
    std::exception_ptr io_error_;

    std::mutex mu_;
    std::condition_variable_any work_cv_;
    std::condition_variable_any space_cv_;
    std::jthread writer_;
};

// ---------------------------------------------------------------------------
// AsyncSeqReader
//
// HPC-safe double-buffered sequential POSIX reader: strictly 1 dedicated
// std::jthread reads contiguous chunks via pread() into 2 aligned slots while
// the caller unpacks/processes the previous slot. Zero mmap, zero posix_fadvise.
// ---------------------------------------------------------------------------
class AsyncSeqReader {
public:
    AsyncSeqReader(int fd, std::string path, off_t start_off, uint64_t total_bytes,
                   size_t slot_bytes, bool take_ownership = false)
        : fd_(fd), path_(std::move(path)), owns_fd_(take_ownership)
    {
        if (fd_ < 0) {
            throw std::runtime_error("AsyncSeqReader: invalid file descriptor for [" + path_ + "].");
        }
        const size_t step = std::max<size_t>(1, slot_bytes);
        uint64_t done = 0;
        while (done < total_bytes) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(step, total_bytes - done));
            chunks_.emplace_back(start_off + static_cast<off_t>(done), n);
            done += n;
        }
        init_slots_and_start(std::min<size_t>(step, static_cast<size_t>(std::min<uint64_t>(step, total_bytes))));
    }

    AsyncSeqReader(int fd, std::string path, std::vector<std::pair<off_t, size_t>> chunks,
                   size_t max_slot_bytes = 0, bool take_ownership = false)
        : fd_(fd), path_(std::move(path)), owns_fd_(take_ownership), chunks_(std::move(chunks))
    {
        if (fd_ < 0) {
            throw std::runtime_error("AsyncSeqReader: invalid file descriptor for [" + path_ + "].");
        }
        size_t cap = max_slot_bytes;
        if (cap == 0) {
            for (const auto& c : chunks_) {
                if (c.second > cap) cap = c.second;
            }
        }
        init_slots_and_start(cap);
    }

    AsyncSeqReader(const AsyncSeqReader&) = delete;
    AsyncSeqReader& operator=(const AsyncSeqReader&) = delete;

    ~AsyncSeqReader() noexcept {
        stop();
    }

    // Retrieve the next ready slot. Returns false cleanly at EOF; rethrows any
    // reader I/O exception. Auto-releases any previously held slot if the
    // caller has not already called release_slot().
    bool next_slot(const char*& data, size_t& len) {
        if (held_slot_ != -1) {
            release_slot();
        }
        std::unique_lock<std::mutex> lock(mu_);
        ready_cv_.wait(lock, [&]() {
            return failed_ || ring_count_ > 0 || eof_;
        });
        if (failed_) {
            std::rethrow_exception(io_error_);
        }
        if (ring_count_ == 0) {
            data = nullptr;
            len = 0;
            return false;
        }
        const int slot = ring_[static_cast<size_t>(ring_head_)];
        ring_head_ = (ring_head_ + 1) & 1;
        --ring_count_;
        held_slot_ = slot;
        data = slots_[static_cast<size_t>(slot)].data();
        len = slot_len_[static_cast<size_t>(slot)];
        return true;
    }

    // Release the currently held slot so the background reader can reuse it.
    void release_slot() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (held_slot_ < 0) return;
            slot_busy_[static_cast<size_t>(held_slot_)] = false;
            held_slot_ = -1;
        }
        free_cv_.notify_one();
    }

    // Stop and join the reader thread and close fd (if owned). Idempotent.
    void stop() noexcept {
        if (stopped_) return;
        stopped_ = true;
        reader_.request_stop();
        free_cv_.notify_all();
        if (reader_.joinable()) {
            reader_.join();
        }
        held_slot_ = -1;
        if (owns_fd_ && fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = -1;
    }

    int fd() const noexcept { return fd_; }
    const std::string& path() const noexcept { return path_; }

private:
    void init_slots_and_start(size_t max_slot_bytes) {
        if (chunks_.empty()) {
            eof_ = true;
            return;
        }
        const size_t cap = std::max<size_t>(1, max_slot_bytes);
        try {
            slots_[0].resize(cap);
            slots_[1].resize(cap);
            reader_ = std::jthread([this](std::stop_token st) { reader_loop(st); });
        } catch (...) {
            if (owns_fd_ && fd_ >= 0) {
                ::close(fd_);
            }
            fd_ = -1;
            stopped_ = true;
            throw;
        }
    }

    void reader_loop(std::stop_token st) {
        try {
            for (size_t idx = 0; idx < chunks_.size() && !st.stop_requested(); ++idx) {
                const int slot = static_cast<int>(idx & 1u);
                {
                    std::unique_lock<std::mutex> lock(mu_);
                    free_cv_.wait(lock, st, [&]() { return !slot_busy_[static_cast<size_t>(slot)]; });
                    if (st.stop_requested()) return;
                    slot_busy_[static_cast<size_t>(slot)] = true;
                }
                const auto [off, nbytes] = chunks_[idx];
                if (nbytes > slots_[static_cast<size_t>(slot)].size()) {
                    slots_[static_cast<size_t>(slot)].resize(nbytes);
                }
                if (nbytes > 0) {
                    read_exact_at_throwing(fd_, slots_[static_cast<size_t>(slot)].data(), nbytes, off, path_);
                }
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    slot_len_[static_cast<size_t>(slot)] = nbytes;
                    ring_[static_cast<size_t>(ring_tail_)] = slot;
                    ring_tail_ = (ring_tail_ + 1) & 1;
                    ++ring_count_;
                }
                ready_cv_.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                eof_ = true;
            }
            ready_cv_.notify_all();
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                io_error_ = std::current_exception();
                failed_ = true;
            }
            ready_cv_.notify_all();
        }
    }

    int fd_ = -1;
    std::string path_;
    bool owns_fd_ = false;
    std::vector<std::pair<off_t, size_t>> chunks_;

    std::array<AlignedRawBuffer<char, 64>, 2> slots_;
    std::array<bool, 2> slot_busy_{false, false};
    std::array<size_t, 2> slot_len_{0, 0};
    std::array<int, 2> ring_{0, 0};
    int ring_head_ = 0;
    int ring_tail_ = 0;
    int ring_count_ = 0;
    int held_slot_ = -1;

    bool eof_ = false;
    bool failed_ = false;
    bool stopped_ = false;
    std::exception_ptr io_error_;

    std::mutex mu_;
    std::condition_variable_any ready_cv_;
    std::condition_variable_any free_cv_;
    std::jthread reader_;
};

} // namespace gcta_io
