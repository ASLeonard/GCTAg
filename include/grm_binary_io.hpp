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
#include <vector>
#include <omp.h>

#include "chunked_grm_matvec.hpp"
#include "mem.hpp"

namespace gcta_grm_io {

// Forward declared so read_grm_binary (below) can call this instead of
// duplicating its logic inline -- see the definition further down.
inline double read_grm_N_mean(const std::string& prefix, int n_grm);

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
inline constexpr size_t kMaxIoChunkBytes = 1ULL << 30;

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

// Returns:
//   ids          — "FID\tIID" strings in GRM file order
//   G            — n×n matrix (upper-triangle-only when upper_only=true)
//   m_snps       — mean SNP count from .grm.N.bin diagonal (skipped if m_snps < 0)
//   trace_K_out  — optional exact trace(K) accumulated during load
//   trace_K2_out — optional exact trace(K^2) accumulated during load
template <typename Scalar>
inline void read_grm_binary(const std::string& prefix,
                             std::vector<std::string>& ids,
                             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& G,
                             double& m_snps,
                             bool upper_only = false,
                             size_t grm_read_chunk_bytes = (1ull << 30) /* 1GiB */,
                             double* trace_K_out = nullptr,
                             double* trace_K2_out = nullptr)
{
    LOGGER.ts("grm");
    using std::to_string;

    ids = Pheno::read_sublist(prefix + ".grm.id");
    const int n = static_cast<int>(ids.size());
    if (n == 0) LOGGER.e(0, "GRM id file [" + prefix + ".grm.id] is empty.");

    const size_t tri      = static_cast<size_t>(n) * (n + 1) / 2;
    const size_t byte_len = tri * sizeof(float);

    const std::string bin_path = prefix + ".grm.bin";
    const int fd = ::open(bin_path.c_str(), O_RDONLY);
    if (fd == -1)
        LOGGER.e(0, "cannot open [" + bin_path + "].");

    {
        struct stat st{};
        if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < byte_len) {
            ::close(fd);
            LOGGER.e(0, "unexpected size in [" + bin_path + "]. "
                        "Expected " + to_string(tri) + " float32 entries for n=" +
                        to_string(n) + ".");
        }
    }
    gcta_fadvise_sequential(fd);

    G.resize(n, n);
    gcta_numa_interleave(G.data(), static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Scalar));

    const bool want_traces = (trace_K_out != nullptr || trace_K2_out != nullptr);
    double tr_K = 0.0;
    double tr_K2 = 0.0;

    const size_t max_elems_per_chunk = std::max<size_t>(1, grm_read_chunk_bytes / sizeof(float));
    AlignedRawBuffer<float, 64> chunk_buf(max_elems_per_chunk);
    int row_cursor = 0;
    while (row_cursor < n) {
        const size_t row_base_cursor = static_cast<size_t>(row_cursor) * (row_cursor + 1) / 2;
        const size_t target = row_base_cursor + max_elems_per_chunk;
        const int row_end = row_bound_for_cumulative(target, row_cursor, n);
        const size_t row_base_end = static_cast<size_t>(row_end) * (row_end + 1) / 2;
        const size_t elems_this_chunk = row_base_end - row_base_cursor;

        if (elems_this_chunk > chunk_buf.size())
            chunk_buf.resize(elems_this_chunk);
        read_exact(fd, chunk_buf.data(), elems_this_chunk * sizeof(float), bin_path);

        if (want_traces) {
            #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
            for (int i = row_cursor; i < row_end; ++i) {
                const size_t local_off = static_cast<size_t>(i) * (i + 1) / 2 - row_base_cursor;
                const float* __restrict__ src_row = chunk_buf.data() + local_off;
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
            #pragma omp parallel for schedule(dynamic, 64)
            for (int i = row_cursor; i < row_end; ++i) {
                const size_t local_off = static_cast<size_t>(i) * (i + 1) / 2 - row_base_cursor;
                if constexpr (std::is_same_v<Scalar, float>) {
                    std::memcpy(G.col(i).data(), chunk_buf.data() + local_off,
                                static_cast<size_t>(i + 1) * sizeof(float));
                } else {
                    G.col(i).head(i + 1) =
                        Eigen::Map<const Eigen::VectorXf>(chunk_buf.data() + local_off, i + 1).template cast<Scalar>();
                }
            }
        }
        gcta_fadvise_dontneed(fd,
                              static_cast<off_t>(row_base_cursor * sizeof(float)),
                              static_cast<off_t>(elems_this_chunk * sizeof(float)));
        row_cursor = row_end;
    }
    gcta_fadvise_dontneed(fd);
    ::close(fd);
    if (trace_K_out)  *trace_K_out  = tr_K;
    if (trace_K2_out) *trace_K2_out = tr_K2;

    LOGGER.i(0, "The .grm.bin fill took " + std::to_string(LOGGER.tp("grm")) + " seconds.");
    if (!upper_only) {
        LOGGER.ts("grm_mirror");
        #pragma omp parallel for schedule(guided)
        for (int c = 0; c < n; ++c)
            for (int r = c + 1; r < n; ++r)
                G(r, c) = G(c, r);
        LOGGER.i(0, "The .grm.bin mirror took " + std::to_string(LOGGER.tp("grm_mirror")) + " seconds.");
    }

    if (m_snps < 0.0) {
        LOGGER.i(0, "Skipping SNP count read from .grm.N.bin.");
    } else {
        m_snps = read_grm_N_mean(prefix, n);
    }
}

// Build a mapping from a reference ID list to indices in grm_ids.
// Returns kp[i] = index in grm_ids matching ref_ids[i], or -1 if not found.
inline std::vector<int> match_ids_to_grm(const std::vector<std::string>& ref_ids,
                                          const std::vector<std::string>& grm_ids)
{
    std::unordered_map<std::string, int> grm_map;
    grm_map.reserve(grm_ids.size() * 2);
    for (int i = 0; i < static_cast<int>(grm_ids.size()); ++i)
        grm_map.emplace(grm_ids[i], i);

    std::vector<int> kp(ref_ids.size(), -1);
    for (int i = 0; i < static_cast<int>(ref_ids.size()); ++i) {
        if (const auto it = grm_map.find(ref_ids[i]); it != grm_map.end())
            kp[i] = it->second;
    }
    return kp;
}

// Tile reader for --grm-chunked-budget: returns K in ANALYSIS sample order
// (post-kp reindexing), keeping at most a bounded double-buffered row-band
// resident at once (strictly <= band_byte_budget_) rather than the whole file.
// Supports contiguous transposed tiles (`read_tile_T` / `read_tile_f32_T`,
// eliminating 16 KiB power-of-2 stride scatter), direct diagonal pread
// (`read_diagonal`), run-length contiguous spans for monotonic kp subsetting,
// and double-buffered `std::jthread` band prefetch (`begin_sweep`).
class ChunkedGrmReader {
public:
    ChunkedGrmReader(const std::string& path, std::vector<int> kp, int n_grm,
                      size_t band_byte_budget = (1ull << 30) /* 1GiB */)
        : kp_(std::move(kp)), path_(path), n_grm_(n_grm), band_byte_budget_(band_byte_budget)
    {
        const size_t tri = static_cast<size_t>(n_grm) * (n_grm + 1) / 2;
        byte_len_ = tri * sizeof(float);

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ == -1)
            LOGGER.e(0, "cannot open [" + path + "].");

        struct stat st{};
        if (::fstat(fd_, &st) != 0 || static_cast<size_t>(st.st_size) < byte_len_) {
            ::close(fd_);
            LOGGER.e(0, "unexpected size in [" + path + "].");
        }
        gcta_fadvise_sequential(fd_);

        kp_is_identity_ = true;
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

        if (!kp_is_identity_ && !kp_.empty()) {
            std::vector<KpRun> runs = build_kp_runs(kp_);
            if (runs.size() * 2 < kp_.size()) {
                kp_runs_ = std::move(runs);
            }
        }
    }

    ~ChunkedGrmReader() {
        if (fd_ != -1) {
            gcta_fadvise_dontneed(fd_);
            ::close(fd_);
        }
    }
    ChunkedGrmReader(const ChunkedGrmReader&)            = delete;
    ChunkedGrmReader& operator=(const ChunkedGrmReader&) = delete;

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
        const size_t min_one_row_bytes = static_cast<size_t>(std::max(1, n_grm_)) * sizeof(float);
        const size_t per_buf_bytes = std::max(band_byte_budget_ / 2, min_one_row_bytes);
        const size_t per_buf_elems = std::max<size_t>(1, per_buf_bytes / sizeof(float));

        int rs = 0;
        while (rs < n) {
            const int file_lo = kp_[rs];
            const size_t elem_lo = static_cast<size_t>(file_lo) * (file_lo + 1) / 2;
            const int max_file_hi = row_bound_for_cumulative(elem_lo + per_buf_elems, file_lo, n_grm_);
            const int lim = std::min(n, rs + max_rows);

            // Largest re in (rs, lim] with kp_[re - 1] < max_file_hi
            const auto it = std::lower_bound(kp_.begin() + rs, kp_.begin() + lim, max_file_hi);
            int re = static_cast<int>(it - kp_.begin());
            if (re <= rs) re = rs + 1;
            bands.push_back(re);
            rs = re;
        }
        return bands;
    }

    // Double-buffered async band prefetcher for streaming sweeps.
    // Spawns one std::jthread (restoring initial process affinity so it never
    // competes with pinned OpenMP worker threads) that prefetches band b+1 into
    // slot (b+1)&1 while the caller computes tiles on band b in slot b&1.
    class BandSweepSession final : public gcta_chunked::TileSweep {
    public:
        BandSweepSession(const ChunkedGrmReader& reader, const std::vector<int>& bands)
            : reader_(reader),
              bands_(bands),
              num_bands_(static_cast<int>(bands.size()) - 1)
        {
            io_thread_ = std::jthread([this](std::stop_token st) {
                gcta_restore_initial_affinity();
                try {
                    for (int b = 0; b < num_bands_ && !st.stop_requested(); ++b) {
                        const int slot = b & 1;
                        {
                            std::unique_lock<std::mutex> lock(mu_);
                            free_cv_.wait(lock, st, [&]() { return !slot_busy_[slot]; });
                            if (st.stop_requested()) return;
                            slot_busy_[slot] = true;
                        }
                        const int rs = bands_[b];
                        const int re = bands_[b + 1];
                        const int fl = reader_.kp_[rs];
                        const int fh = reader_.kp_[re - 1] + 1;
                        reader_.load_band_exact_into_slot(slot, fl, fh);
                        {
                            std::lock_guard<std::mutex> lock(mu_);
                            slot_ready_band_[slot] = b;
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
                return slot_ready_band_[slot] == band_idx || failed_;
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
                slot_ready_band_[slot] = -1;
                slot_busy_[slot] = false;
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

    // Direct diagonal pread without scanning off-diagonal entries of .grm.bin.
    // Coalesces leading rows (< 2048) into one contiguous read and bounds
    // parallel pread threads (<= 16) with MADV/POSIX_FADV_RANDOM for the rest.
    Eigen::VectorXd read_diagonal() const {
        const int n = static_cast<int>(kp_.size());
        Eigen::VectorXd diag(n);
        if (n == 0) return diag;

        constexpr int kCoalesceMaxRow = 2048;
        int split = 0;
        while (split < n && kp_[split] < kCoalesceMaxRow) ++split;

        if (split > 0) {
            const int max_row = kp_[split - 1];
            const size_t elems = static_cast<size_t>(max_row + 1) * (max_row + 2) / 2;
            AlignedRawBuffer<float, 64> head_buf(elems);
            read_exact_at(fd_, head_buf.data(), elems * sizeof(float), 0, path_);
            for (int i = 0; i < split; ++i) {
                const int gi = kp_[i];
                const size_t diag_idx = static_cast<size_t>(gi) * (gi + 1) / 2 + gi;
                diag[i] = static_cast<double>(head_buf[diag_idx]);
            }
        }

        if (split < n) {
            gcta_fadvise_random(fd_);
            const int n_io_threads = std::min(omp_get_max_threads(), 16);
            bool failed = false;
            #pragma omp parallel for schedule(static) num_threads(n_io_threads)
            for (int i = split; i < n; ++i) {
                const int gi = kp_[i];
                const size_t diag_idx = static_cast<size_t>(gi) * (gi + 1) / 2 + gi;
                float v = 0.0f;
                ssize_t r = 0;
                do {
                    r = ::pread(fd_, &v, sizeof(v), static_cast<off_t>(diag_idx * sizeof(float)));
                } while (r < 0 && errno == EINTR);
                if (r == static_cast<ssize_t>(sizeof(v))) {
                    diag[i] = static_cast<double>(v);
                } else {
                    #pragma omp atomic write
                    failed = true;
                }
            }
            gcta_fadvise_sequential(fd_);
            if (failed)
                LOGGER.e(0, "pread() failed while reading diagonal of [" + path_ + "].");
        }
        return diag;
    }

    // Fast-path transposed float32 tile: returns K[rs:re, cs:ce]^T of shape
    // (ce-cs) x (re-rs) ColMajor so each GRM row unpacks into a contiguous
    // column (zero 16 KiB power-of-2 stride scatter). For diagonal tiles
    // (rs == cs), the UPPER triangle (r <= c) is populated.
    Eigen::Ref<const Eigen::MatrixXf> read_tile_f32_T(int rs, int re, int cs, int ce) const {
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_f32_T_.rows() < tile_cols || tile_scratch_f32_T_.cols() < tile_rows) {
            tile_scratch_f32_T_.resize(
                std::max<Eigen::Index>(tile_scratch_f32_T_.rows(), tile_cols),
                std::max<Eigen::Index>(tile_scratch_f32_T_.cols(), tile_rows));
        }
        auto tile_T = tile_scratch_f32_T_.topLeftCorner(tile_cols, tile_rows);
        const bool diagonal_tile = (rs == cs);

        ensure_band(kp_[rs], kp_[re - 1] + 1);
        const int slot = active_buf_;
        const float* buf = band_buf_[slot].data();
        const size_t band_elem_lo = static_cast<size_t>(band_lo_[slot]) * (band_lo_[slot] + 1) / 2;

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
                const int gi = kp_[rs + lp];
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
            const int gi = kp_[rs + lp];
            const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
            if (lq_end == 0) continue;
            const int gj_lo = kp_[cs];
            const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
            const float* row_span = buf + row_base + gj_lo;
            float* dst_col = tile_T.col(lp).data();
            for (int lq = 0; lq < lq_end; ++lq) {
                dst_col[lq] = row_span[kp_[cs + lq] - gj_lo];
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

        ensure_band(kp_[rs], kp_[re - 1] + 1);
        const int slot = active_buf_;
        const float* buf = band_buf_[slot].data();
        const size_t band_elem_lo = static_cast<size_t>(band_lo_[slot]) * (band_lo_[slot] + 1) / 2;

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
                const int gi = kp_[rs + lp];
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
            const int gi = kp_[rs + lp];
            const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
            if (lq_end == 0) continue;
            const int gj_lo = kp_[cs];
            const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
            const float* row_span = buf + row_base + gj_lo;
            double* dst_col = tile_T.col(lp).data();
            for (int lq = 0; lq < lq_end; ++lq) {
                dst_col[lq] = static_cast<double>(row_span[kp_[cs + lq] - gj_lo]);
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

    mutable std::array<AlignedRawBuffer<float, 64>, 2> band_buf_;
    mutable std::array<int, 2> band_lo_{0, 0};
    mutable std::array<int, 2> band_hi_{0, 0};
    mutable int active_buf_ = 0;

    void load_band_exact_into_slot(int slot, int file_lo, int file_hi) const {
        if (file_lo >= band_lo_[slot] && file_hi <= band_hi_[slot]) return;
        band_lo_[slot] = 0;
        band_hi_[slot] = 0;
        const size_t elem_lo = static_cast<size_t>(file_lo) * (file_lo + 1) / 2;
        const size_t elem_hi = static_cast<size_t>(file_hi) * (file_hi + 1) / 2;
        const size_t count = elem_hi - elem_lo;
        band_buf_[slot].resize(count);
        read_exact_at(fd_, band_buf_[slot].data(), count * sizeof(float),
                      static_cast<off_t>(elem_lo * sizeof(float)), path_);
        band_lo_[slot] = file_lo;
        band_hi_[slot] = file_hi;
    }

    void ensure_band(int file_lo, int file_hi) const {
        const int slot = active_buf_;
        if (file_lo >= band_lo_[slot] && file_hi <= band_hi_[slot]) return;

        band_lo_[slot] = 0;
        band_hi_[slot] = 0;
        const size_t elem_lo = static_cast<size_t>(file_lo) * (file_lo + 1) / 2;
        const size_t target_elems = elem_lo + band_byte_budget_ / sizeof(float);
        const int grown_hi  = row_bound_for_cumulative(target_elems, file_lo, n_grm_);
        const int actual_hi = std::max(file_hi, grown_hi);

        const size_t elem_hi = static_cast<size_t>(actual_hi) * (actual_hi + 1) / 2;
        const size_t count = elem_hi - elem_lo;
        band_buf_[slot].resize(count);
        read_exact_at(fd_, band_buf_[slot].data(), count * sizeof(float),
                      static_cast<off_t>(elem_lo * sizeof(float)), path_);
        band_lo_[slot] = file_lo;
        band_hi_[slot] = actual_hi;
    }

    std::vector<int> kp_;
    std::vector<KpRun> kp_runs_;
    std::string path_;
    int fd_ = -1;
    size_t byte_len_ = 0;
    int n_grm_ = 0;
    size_t band_byte_budget_ = 0;
    bool kp_is_identity_ = false;
};

// .grm.N.bin diagonal only (mean SNP count): coalesces the leading <=2048 rows
// into one contiguous pread (<= 8.4 MB), and reads remaining diagonals via
// bounded-thread random pread (<= 16 threads) to avoid Lustre RPC lock storms.
inline double read_grm_N_mean(const std::string& prefix, int n_grm) {
    if (n_grm <= 0) return 0.0;
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
    if (::fstat(nfd, &st) != 0 || static_cast<size_t>(st.st_size) < byte_len) {
        ::close(nfd);
        LOGGER.w(0, "GRM N file [" + n_path + "] has unexpected size; "
                    "SNP count unavailable (affects --reml-woodbury auto-k).");
        return 0.0;
    }

    double sum = 0.0;
    bool read_failed = false;

    const int c_end = std::min(n_grm, 2048);
    if (c_end > 0) {
        const size_t head_elems = static_cast<size_t>(c_end) * (c_end + 1) / 2;
        AlignedRawBuffer<float, 64> head_buf(head_elems);
        const size_t head_bytes = head_elems * sizeof(float);
        size_t done = 0;
        char* dst = reinterpret_cast<char*>(head_buf.data());
        while (done < head_bytes) {
            const ssize_t r = ::pread(nfd, dst + done, head_bytes - done, static_cast<off_t>(done));
            if (r < 0) {
                if (errno == EINTR) continue;
                read_failed = true;
                break;
            }
            if (r == 0) {
                read_failed = true;
                break;
            }
            done += static_cast<size_t>(r);
        }
        if (!read_failed) {
            for (int i = 0; i < c_end; ++i) {
                const size_t diag_idx = static_cast<size_t>(i) * (i + 1) / 2 + i;
                sum += static_cast<double>(head_buf[diag_idx]);
            }
        }
    }

    if (!read_failed && c_end < n_grm) {
        gcta_fadvise_random(nfd);
        const int n_io_threads = std::min(omp_get_max_threads(), 16);
        #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(n_io_threads)
        for (int i = c_end; i < n_grm; ++i) {
            const size_t diag_idx = static_cast<size_t>(i) * (i + 1) / 2 + i;
            float v = 0.0f;
            ssize_t r = 0;
            do {
                r = ::pread(nfd, &v, sizeof(v),
                            static_cast<off_t>(diag_idx * sizeof(float)));
            } while (r < 0 && errno == EINTR);
            if (r == static_cast<ssize_t>(sizeof(v))) {
                sum += static_cast<double>(v);
            } else {
                #pragma omp atomic write
                read_failed = true;
            }
        }
    }
    gcta_fadvise_dontneed(nfd);
    ::close(nfd);

    if (read_failed) {
        LOGGER.w(0, "pread() failed while reading [" + n_path + "]'s diagonal; "
                    "SNP count unavailable (affects --reml-woodbury auto-k).");
        return 0.0;
    }
    return sum / n_grm;
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
    int chunk_rows = 0;
};

inline ChunkedGrmHandle make_chunked_grm_reader(
    const std::string& prefix,
    const std::vector<std::string>& analysis_ids,
    double budget_gb,
    int k_ext = 0,
    const std::string& feature_flag = "--grm-chunked-budget",
    bool need_m_snps = true)
{
    const std::vector<std::string> grm_ids = Pheno::read_sublist(prefix + ".grm.id");
    const int n_grm = static_cast<int>(grm_ids.size());
    const int n = static_cast<int>(analysis_ids.size());

    std::vector<int> kp = match_ids_to_grm(analysis_ids, grm_ids);
    for (int i = 0; i < n; ++i) {
        if (kp[i] < 0)
            LOGGER.e(0, feature_flag + ": individual [" + analysis_ids[i] +
                        "] not found in GRM [" + prefix + ".grm.id].");
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
                ") -> " + std::to_string(chunk_rows) + "-col tiles from [" + prefix +
                ".grm.bin], not loaded densely.");

    ChunkedGrmHandle handle;
    handle.chunk_rows = chunk_rows;
    if (need_m_snps) {
        handle.m_snps = read_grm_N_mean(prefix, n_grm);
    } else {
        LOGGER.i(0, "Skipping SNP count read from .grm.N.bin.");
        handle.m_snps = -1.0;
    }
    auto file = std::make_shared<ChunkedGrmReader>(prefix + ".grm.bin", std::move(kp), n_grm,
                                                    static_cast<size_t>(reader_reserved_gb * 1e9));
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
// Opt-in packed lower-triangular IEEE fp16 in-memory GRM (--grm-fp16)
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
            tile_scratch_f32_T_.resize(
                std::max<Eigen::Index>(tile_scratch_f32_T_.rows(), tile_cols),
                std::max<Eigen::Index>(tile_scratch_f32_T_.cols(), tile_rows));
        }
        auto tile_T = tile_scratch_f32_T_.topLeftCorner(tile_cols, tile_rows);
        const bool diagonal_tile = (rs == cs);
        const GrmFp16* base = packed_.get();

        #pragma omp parallel for schedule(guided)
        for (int lp = 0; lp < tile_rows; ++lp) {
            const int gi = rs + lp;
            const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
            if (lq_end == 0) continue;
            const GrmFp16* __restrict__ src = base + static_cast<size_t>(gi) * (gi + 1) / 2 + cs;
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

    // Fast-path transposed double tile: shape (ce-cs) x (re-rs) ColMajor.
    Eigen::Ref<const Eigen::MatrixXd> read_tile_T(int rs, int re, int cs, int ce) const {
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_f64_T_.rows() < tile_cols || tile_scratch_f64_T_.cols() < tile_rows) {
            tile_scratch_f64_T_.resize(
                std::max<Eigen::Index>(tile_scratch_f64_T_.rows(), tile_cols),
                std::max<Eigen::Index>(tile_scratch_f64_T_.cols(), tile_rows));
        }
        auto tile_T = tile_scratch_f64_T_.topLeftCorner(tile_cols, tile_rows);
        const bool diagonal_tile = (rs == cs);
        const GrmFp16* base = packed_.get();

        #pragma omp parallel for schedule(guided)
        for (int lp = 0; lp < tile_rows; ++lp) {
            const int gi = rs + lp;
            const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
            if (lq_end == 0) continue;
            const GrmFp16* __restrict__ src = base + static_cast<size_t>(gi) * (gi + 1) / 2 + cs;
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
    mutable Eigen::MatrixXf tile_scratch_f32_;
    mutable Eigen::MatrixXd tile_scratch_f64_;
    mutable Eigen::MatrixXf tile_scratch_f32_T_;
    mutable Eigen::MatrixXd tile_scratch_f64_T_;
};

struct PackedGrmFp16Handle {
    gcta_chunked::TileReader reader;
    gcta_chunked::TileReaderF32 reader_f32;
    std::shared_ptr<PackedGrmFp16> file;
    double m_snps = -1.0;
    double trace_K = 0.0;
    double trace_K2 = 0.0;
    int chunk_rows = 0;
};

inline PackedGrmFp16Handle load_packed_grm_fp16(
    const std::string& grm_prefix,
    const std::vector<std::string>& analysis_ids,
    bool need_m_snps = false,
    const std::string& caller_flag = "--grm-fp16",
    size_t grm_read_chunk_bytes = (1ull << 30) /* 1GiB */)
{
    LOGGER.ts("grm_fp16");
    using std::to_string;

    const std::vector<std::string> grm_ids = Pheno::read_sublist(grm_prefix + ".grm.id");
    const int n_grm = static_cast<int>(grm_ids.size());
    if (n_grm == 0)
        LOGGER.e(0, "GRM id file [" + grm_prefix + ".grm.id] is empty.");

    const int n = static_cast<int>(analysis_ids.size());
    const std::vector<int> kp = match_ids_to_grm(analysis_ids, grm_ids);
    for (int i = 0; i < n; ++i) {
        if (kp[i] < 0)
            LOGGER.e(0, caller_flag + ": individual [" + analysis_ids[i] +
                        "] not found in GRM [" + grm_prefix + "].");
    }

    bool is_identity = (n_grm == n);
    bool is_monotonic = true;
    for (int i = 0; i < n; ++i) {
        if (kp[i] != i) is_identity = false;
        if (i > 0 && kp[i] <= kp[i - 1]) is_monotonic = false;
    }
    if (!is_monotonic)
        LOGGER.e(0, caller_flag + " requires the analysis sample order to match "
                    "(or be an ordered subset of) [" + grm_prefix + ".grm.id]'s row order.");

    std::vector<KpRun> kp_runs;
    if (!is_identity && n > 0) {
        std::vector<KpRun> runs = build_kp_runs(kp);
        if (runs.size() * 2 < static_cast<size_t>(n)) {
            kp_runs = std::move(runs);
        }
    }

    const size_t tri_grm  = static_cast<size_t>(n_grm) * (n_grm + 1) / 2;
    const size_t byte_len = tri_grm * sizeof(float);
    const std::string bin_path = grm_prefix + ".grm.bin";
    const int fd = ::open(bin_path.c_str(), O_RDONLY);
    if (fd == -1)
        LOGGER.e(0, "cannot open [" + bin_path + "].");
    {
        struct stat st{};
        if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < byte_len) {
            ::close(fd);
            LOGGER.e(0, "unexpected size in [" + bin_path + "]. "
                        "Expected " + to_string(tri_grm) + " float32 entries for n=" +
                        to_string(n_grm) + ".");
        }
    }
    gcta_fadvise_sequential(fd);

    const size_t tri_n = static_cast<size_t>(n) * (n + 1) / 2;
    const double packed_gb = static_cast<double>(tri_n * sizeof(GrmFp16)) / 1e9;
    LOGGER.i(0, caller_flag + ": loading [" + bin_path + "] into packed fp16 lower triangle (" +
                to_string(packed_gb) + " GB for n=" + to_string(n) +
                ", 4x smaller than dense float32) ...");

    // Uninitialized NUMA-interleaved allocation: eliminates single-threaded Core-0
    // zero-fill of 250 GB-1 TB and spreads pages evenly across all NUMA nodes.
    AlignedUniquePtr<GrmFp16, 64> packed = make_numa_interleaved_uninitialized<GrmFp16, 64>(tri_n);
    if (tri_n > 0 && !packed) {
        ::close(fd);
        LOGGER.e(0, caller_flag + ": failed to allocate " + to_string(packed_gb) +
                    " GB for packed fp16 GRM.");
    }
    Eigen::VectorXf diag_f32(n);
    double tr_K  = 0.0;
    double tr_K2 = 0.0;

    const size_t max_elems_per_chunk = std::max<size_t>(1, grm_read_chunk_bytes / sizeof(float));
    AlignedRawBuffer<float, 64> chunk_buf(max_elems_per_chunk);
    int row_cursor = 0;
    int j_cursor = 0;
    while (row_cursor < n_grm && j_cursor < n) {
        const size_t row_base_cursor = static_cast<size_t>(row_cursor) * (row_cursor + 1) / 2;
        const size_t target = row_base_cursor + max_elems_per_chunk;
        const int row_end = row_bound_for_cumulative(target, row_cursor, n_grm);
        const size_t row_base_end = static_cast<size_t>(row_end) * (row_end + 1) / 2;
        const size_t elems_this_chunk = row_base_end - row_base_cursor;

        int j_end = j_cursor;
        if (is_identity) {
            j_end = row_end;
        } else {
            while (j_end < n && kp[j_end] < row_end) ++j_end;
        }

        if (elems_this_chunk > chunk_buf.size())
            chunk_buf.resize(elems_this_chunk);
        read_exact(fd, chunk_buf.data(), elems_this_chunk * sizeof(float), bin_path);

        if (j_end > j_cursor) {
            GrmFp16* packed_ptr = packed.get();
            if (is_identity) {
                #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
                for (int j = j_cursor; j < j_end; ++j) {
                    const float* __restrict__ src_row = chunk_buf.data() +
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
                #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
                for (int j = j_cursor; j < j_end; ++j) {
                    const int r_grm = kp[j];
                    const float* __restrict__ src_row = chunk_buf.data() +
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
                #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
                for (int j = j_cursor; j < j_end; ++j) {
                    const int r_grm = kp[j];
                    const float* __restrict__ src_row = chunk_buf.data() +
                        (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                    GrmFp16* __restrict__ dst_row = packed_ptr +
                        static_cast<size_t>(j) * (j + 1) / 2;
                    double row_off_sq = 0.0;
                    for (int i = 0; i < j; ++i) {
                        const float v = src_row[kp[i]];
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
            j_cursor = j_end;
        }
        gcta_fadvise_dontneed(fd,
                              static_cast<off_t>(row_base_cursor * sizeof(float)),
                              static_cast<off_t>(elems_this_chunk * sizeof(float)));
        row_cursor = row_end;
    }
    gcta_fadvise_dontneed(fd);
    ::close(fd);
    chunk_buf.reset();

    LOGGER.i(0, "The .grm.bin packed fp16 fill took " + to_string(LOGGER.tp("grm_fp16")) + " seconds.");

    PackedGrmFp16Handle handle;
    handle.chunk_rows = std::min(n, 4096);
    handle.trace_K = tr_K;
    handle.trace_K2 = tr_K2;
    if (need_m_snps) {
        handle.m_snps = read_grm_N_mean(grm_prefix, n_grm);
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
    // Avoid small-k tile starvation in PCA rSVD (k_ext < 256): use 8192-row tiles.
    handle.reader_f32.small_k_block_size = std::min(n, 8192);

    return handle;
}

// Load a binary GRM (upper triangle only) and subset/reorder it to match
// `analysis_ids`, writing only the upper triangle of `G_out` so lower-triangle
// pages stay unfaulted. Optionally accumulates exact trace(K) and trace(K^2)
// during load to eliminate a separate O(N^2) DRAM sweep.
template <typename Scalar>
inline void load_and_subset_grm_upper(
    const std::string& grm_prefix,
    const std::vector<std::string>& analysis_ids,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& G_out,
    double& m_snps_out,
    const std::string& caller_flag,
    size_t grm_read_chunk_bytes = (1ull << 30) /* 1GiB */,
    double* trace_K_out = nullptr,
    double* trace_K2_out = nullptr)
{
    std::vector<std::string> grm_ids = Pheno::read_sublist(grm_prefix + ".grm.id");
    const int n_grm = static_cast<int>(grm_ids.size());
    if (n_grm == 0)
        LOGGER.e(0, "GRM id file [" + grm_prefix + ".grm.id] is empty.");

    const int n = static_cast<int>(analysis_ids.size());
    const std::vector<int> kp = match_ids_to_grm(analysis_ids, grm_ids);
    for (int i = 0; i < n; ++i) {
        if (kp[i] < 0)
            LOGGER.e(0, caller_flag + ": individual [" + analysis_ids[i] +
                        "] not found in GRM [" + grm_prefix + "].");
    }

    bool is_identity = (n_grm == n);
    bool is_monotonic = true;
    for (int i = 0; i < n; ++i) {
        if (kp[i] != i) is_identity = false;
        if (i > 0 && kp[i] <= kp[i - 1]) is_monotonic = false;
    }

    if (is_identity) {
        read_grm_binary(grm_prefix, grm_ids, G_out, m_snps_out, /*upper_only=*/true,
                        grm_read_chunk_bytes, trace_K_out, trace_K2_out);
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

        const size_t tri      = static_cast<size_t>(n_grm) * (n_grm + 1) / 2;
        const size_t byte_len = tri * sizeof(float);
        const std::string bin_path = grm_prefix + ".grm.bin";
        const int fd = ::open(bin_path.c_str(), O_RDONLY);
        if (fd == -1)
            LOGGER.e(0, "cannot open [" + bin_path + "].");
        {
            struct stat st{};
            if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < byte_len) {
                ::close(fd);
                LOGGER.e(0, "unexpected size in [" + bin_path + "]. "
                            "Expected " + to_string(tri) + " float32 entries for n=" +
                            to_string(n_grm) + ".");
            }
        }
        gcta_fadvise_sequential(fd);

        G_out.resize(n, n);
        gcta_numa_interleave(G_out.data(), static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Scalar));

        double tr_K = 0.0;
        double tr_K2 = 0.0;

        const size_t max_elems_per_chunk = std::max<size_t>(1, grm_read_chunk_bytes / sizeof(float));
        AlignedRawBuffer<float, 64> chunk_buf(max_elems_per_chunk);
        int row_cursor = 0;
        int j_cursor = 0;
        while (row_cursor < n_grm && j_cursor < n) {
            const size_t row_base_cursor = static_cast<size_t>(row_cursor) * (row_cursor + 1) / 2;
            const size_t target = row_base_cursor + max_elems_per_chunk;
            const int row_end = row_bound_for_cumulative(target, row_cursor, n_grm);
            const size_t row_base_end = static_cast<size_t>(row_end) * (row_end + 1) / 2;
            const size_t elems_this_chunk = row_base_end - row_base_cursor;

            int j_end = j_cursor;
            while (j_end < n && kp[j_end] < row_end) ++j_end;

            if (elems_this_chunk > chunk_buf.size())
                chunk_buf.resize(elems_this_chunk);
            read_exact(fd, chunk_buf.data(), elems_this_chunk * sizeof(float), bin_path);

            if (j_end > j_cursor) {
                if (!kp_runs.empty()) {
                    #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
                    for (int j = j_cursor; j < j_end; ++j) {
                        const int r_grm = kp[j];
                        const float* __restrict__ row_ptr = chunk_buf.data() +
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
                    #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
                    for (int j = j_cursor; j < j_end; ++j) {
                        const int r_grm = kp[j];
                        const float* row_ptr = chunk_buf.data() +
                            (static_cast<size_t>(r_grm) * (r_grm + 1) / 2 - row_base_cursor);
                        Scalar* col_out = G_out.col(j).data();
                        double row_off_sq = 0.0;
                        for (int i = 0; i < j; ++i) {
                            const float v = row_ptr[kp[i]];
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
                j_cursor = j_end;
            }
            gcta_fadvise_dontneed(fd,
                                  static_cast<off_t>(row_base_cursor * sizeof(float)),
                                  static_cast<off_t>(elems_this_chunk * sizeof(float)));
            row_cursor = row_end;
        }
        gcta_fadvise_dontneed(fd);
        ::close(fd);
        if (trace_K_out)  *trace_K_out  = tr_K;
        if (trace_K2_out) *trace_K2_out = tr_K2;

        LOGGER.i(0, "The .grm.bin fill took " + std::to_string(LOGGER.tp("grm")) + " seconds.");
        if (m_snps_out < 0.0) {
            LOGGER.i(0, "Skipping SNP count read from .grm.N.bin.");
        } else {
            m_snps_out = read_grm_N_mean(grm_prefix, n_grm);
        }
        return;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> G_full;
    read_grm_binary(grm_prefix, grm_ids, G_full, m_snps_out, /*upper_only=*/true, grm_read_chunk_bytes);
    G_out.resize(n, n);
    gcta_numa_interleave(G_out.data(), static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Scalar));
    double tr_K = 0.0;
    double tr_K2 = 0.0;
    #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
    for (int j = 0; j < n; ++j) {
        const int src_col = kp[j];
        double row_off_sq = 0.0;
        for (int i = 0; i < j; ++i) {
            const int r = kp[i], c = src_col;
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

inline void merge_grms(
    const std::vector<std::string>& prefixes,
    const std::string& out_prefix,
    double memory_budget_gb = 0.0)
{
    if (prefixes.empty())
        LOGGER.e(0, "merge_grms: no input GRM prefixes given.");

    const std::vector<std::string> ids = Pheno::read_sublist(prefixes[0] + ".grm.id");
    const int n = static_cast<int>(ids.size());
    if (n == 0) LOGGER.e(0, "GRM id file [" + prefixes[0] + ".grm.id] is empty.");

    for (size_t f = 1; f < prefixes.size(); ++f) {
        const std::vector<std::string> other_ids = Pheno::read_sublist(prefixes[f] + ".grm.id");
        if (other_ids != ids)
            LOGGER.e(0, "merge_grms: [" + prefixes[f] + ".grm.id] does not match "
                        "[" + prefixes[0] + ".grm.id] exactly (same sample order required). "
                        "Use a kp-based merge (ChunkedGrmReader) for mismatched orderings.");
    }

    const size_t tri      = static_cast<size_t>(n) * (n + 1) / 2;
    const size_t byte_len = tri * sizeof(float);
    const int    K        = static_cast<int>(prefixes.size());
    const size_t num_buffers = static_cast<size_t>(2 * K + 2);

    size_t chunk_elems = 0;
    if (memory_budget_gb > 0.0) {
        const double budget_bytes = memory_budget_gb * 1e9;
        chunk_elems = static_cast<size_t>(budget_bytes / (num_buffers * sizeof(float)));
        chunk_elems = std::clamp<size_t>(chunk_elems, 65536, tri);
        LOGGER.i(0, "merge_grms: using " +
                    std::to_string((chunk_elems * sizeof(float)) >> 20) +
                    " MB per stream buffer (" +
                    std::to_string((num_buffers * chunk_elems * sizeof(float)) >> 20) +
                    " MB total buffer memory, budget=" + std::to_string(memory_budget_gb) + "GB).");
    } else {
        constexpr size_t default_target_elems = 4 * 1024 * 1024;
        constexpr size_t max_total_bytes = 512ull << 20;
        const size_t max_elems_from_cap = max_total_bytes / (num_buffers * sizeof(float));
        chunk_elems = std::min(default_target_elems, max_elems_from_cap);
        chunk_elems = std::clamp<size_t>(chunk_elems, 65536, tri);
        LOGGER.i(0, "merge_grms: using " +
                    std::to_string((chunk_elems * sizeof(float)) >> 20) +
                    " MB per stream buffer (" +
                    std::to_string((num_buffers * chunk_elems * sizeof(float)) >> 20) +
                    " MB total buffer memory). Set `--merge-grms <GB>` to customize.");
    }

    struct OpenGrmFile {
        int fd = -1;
        std::string path;
    };
    auto open_checked = [&](const std::string& path) -> OpenGrmFile {
        OpenGrmFile of;
        of.path = path;
        of.fd = ::open(path.c_str(), O_RDONLY);
        if (of.fd == -1) LOGGER.e(0, "cannot open [" + path + "].");
        struct stat st{};
        if (::fstat(of.fd, &st) != 0 || static_cast<size_t>(st.st_size) < byte_len) {
            ::close(of.fd);
            LOGGER.e(0, "unexpected size in [" + path + "].");
        }
        gcta_fadvise_sequential(of.fd);
        return of;
    };

    std::vector<OpenGrmFile> val_files(K), n_files(K);
    for (int f = 0; f < K; ++f) {
        val_files[f] = open_checked(prefixes[f] + ".grm.bin");
        n_files[f]   = open_checked(prefixes[f] + ".grm.N.bin");
    }

    const std::string out_bin_path = out_prefix + ".grm.bin";
    const std::string out_n_path   = out_prefix + ".grm.N.bin";
    const int out_bin_fd = ::open(out_bin_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_bin_fd == -1) LOGGER.e(0, "cannot open [" + out_bin_path + "] for writing: " + std::string(std::strerror(errno)));
    const int out_n_fd   = ::open(out_n_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_n_fd == -1) {
        ::close(out_bin_fd);
        LOGGER.e(0, "cannot open [" + out_n_path + "] for writing: " + std::string(std::strerror(errno)));
    }

    AlignedRawBuffer<float, 64> out_val_buf(chunk_elems), out_n_buf(chunk_elems);
    std::vector<AlignedRawBuffer<float, 64>> val_block_bufs(K), n_block_bufs(K);
    for (int f = 0; f < K; ++f) {
        val_block_bufs[f].resize(chunk_elems);
        n_block_bufs[f].resize(chunk_elems);
    }

    for (size_t offset = 0; offset < tri; offset += chunk_elems) {
        const size_t elems_this_chunk = std::min(chunk_elems, tri - offset);
        const size_t bytes_this_chunk = elems_this_chunk * sizeof(float);

        const int max_io_threads = std::min(omp_get_max_threads(), 8);
        std::string io_error;
        bool io_failed = false;
        #pragma omp parallel for schedule(dynamic, 1) num_threads(max_io_threads)
        for (int f = 0; f < K; ++f) {
            try {
                read_exact(val_files[f].fd, val_block_bufs[f].data(),
                           bytes_this_chunk, val_files[f].path);
                read_exact(n_files[f].fd, n_block_bufs[f].data(),
                           bytes_this_chunk, n_files[f].path);
                gcta_fadvise_dontneed(val_files[f].fd,
                                      static_cast<off_t>(offset * sizeof(float)),
                                      static_cast<off_t>(bytes_this_chunk));
                gcta_fadvise_dontneed(n_files[f].fd,
                                      static_cast<off_t>(offset * sizeof(float)),
                                      static_cast<off_t>(bytes_this_chunk));
            } catch (const std::exception& e) {
                #pragma omp critical
                {
                    if (!io_failed) { io_failed = true; io_error = e.what(); }
                }
            }
        }
        if (io_failed)
            LOGGER.e(0, "merge_grms: " + io_error);

        constexpr size_t TILE_SIZE = 4096;
        #pragma omp parallel for schedule(static)
        for (size_t t_start = 0; t_start < elems_this_chunk; t_start += TILE_SIZE) {
            const size_t t_end = std::min(t_start + TILE_SIZE, elems_this_chunk);
            const size_t tile_len = t_end - t_start;

            alignas(64) double tile_wsum[TILE_SIZE];
            alignas(64) double tile_wtN[TILE_SIZE];
            std::fill_n(tile_wsum, tile_len, 0.0);
            std::fill_n(tile_wtN, tile_len, 0.0);

            for (int f = 0; f < K; ++f) {
                const float* __restrict__ v  = val_block_bufs[f].data() + t_start;
                const float* __restrict__ nn = n_block_bufs[f].data() + t_start;
                #pragma omp simd
                for (size_t j = 0; j < tile_len; ++j) {
                    tile_wsum[j] += static_cast<double>(v[j]) * static_cast<double>(nn[j]);
                    tile_wtN[j]  += static_cast<double>(nn[j]);
                }
            }

            float* __restrict__ out_v = out_val_buf.data() + t_start;
            float* __restrict__ out_n = out_n_buf.data() + t_start;
            #pragma omp simd
            for (size_t j = 0; j < tile_len; ++j) {
                out_v[j] = static_cast<float>(tile_wtN[j] > 0.0 ? tile_wsum[j] / tile_wtN[j] : 0.0);
                out_n[j] = static_cast<float>(tile_wtN[j]);
            }
        }

        std::string write_error;
        bool write_failed = false;
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                try {
                    write_exact(out_bin_fd, out_val_buf.data(), bytes_this_chunk, out_bin_path);
                } catch (const std::exception& e) {
                    #pragma omp critical
                    { if (!write_failed) { write_failed = true; write_error = e.what(); } }
                }
            }
            #pragma omp section
            {
                try {
                    write_exact(out_n_fd, out_n_buf.data(), bytes_this_chunk, out_n_path);
                } catch (const std::exception& e) {
                    #pragma omp critical
                    { if (!write_failed) { write_failed = true; write_error = e.what(); } }
                }
            }
        }
        if (write_failed)
            LOGGER.e(0, "merge_grms: " + write_error);
    }

    gcta_fadvise_dontneed(out_bin_fd);
    gcta_fadvise_dontneed(out_n_fd);
    ::close(out_bin_fd);
    ::close(out_n_fd);

    for (auto& f : val_files) { gcta_fadvise_dontneed(f.fd); ::close(f.fd); }
    for (auto& f : n_files)   { gcta_fadvise_dontneed(f.fd); ::close(f.fd); }

    {
        const std::string id_src_path = prefixes[0] + ".grm.id";
        const std::string id_dst_path = out_prefix + ".grm.id";
        std::ifstream src(id_src_path, std::ios::binary);
        if (!src) LOGGER.e(0, "cannot open [" + id_src_path + "] to copy.");
        std::ofstream dst(id_dst_path, std::ios::binary);
        if (!dst) LOGGER.e(0, "cannot open [" + id_dst_path + "] for writing.");
        dst << src.rdbuf();
        if (!dst) LOGGER.e(0, "failed while copying [" + id_src_path + "] to [" + id_dst_path + "].");
    }

    LOGGER.i(0, "Merged " + std::to_string(K) + " GRMs (" + std::to_string(n) +
                " individuals) into [" + out_prefix + ".grm.bin/.grm.N.bin/.grm.id], "
                "N-weighted average, fully streamed — no dense matrix held for any input or the output.");
}

} // namespace gcta_grm_io
