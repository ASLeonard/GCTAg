#pragma once

#include <Eigen/Dense>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "chunked_grm_matvec.hpp"

namespace gcta_grm_io {

// Forward declared so read_grm_binary (below) can call this instead of
// duplicating its logic inline -- see the definition further down for the
// implementation and why it deliberately does NOT use the whole-file warm-up
// pattern used elsewhere in this file.
inline double read_grm_N_mean(const std::string& prefix, int n_grm);

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
// .grm.N.bin) for n individuals: n(n+1)/2 float32 entries. Exposed so
// callers sizing a memory budget that ChunkedGrmReader's fixed-size buffer
// shares space with (see that class's constructor) can reserve this amount
// up front, e.g. via solve_chunk_rows's reserved_gb parameter in
// chunked_grm_matvec.hpp, rather than duplicating this formula themselves.
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

// Blocking read of exactly `count` bytes into `buf`, retrying on EINTR and
// on short reads (both routine on network filesystems). Errors out via
// LOGGER on unexpected EOF or a hard read error.
inline void read_exact(int fd, void* buf, size_t count, const std::string& path) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < count) {
        const ssize_t r = ::read(fd, p + done, count - done);
        if (r < 0) {
            if (errno == EINTR) continue;
            LOGGER.e(0, "read() failed on [" + path + "]: " + std::string(std::strerror(errno)));
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
        const ssize_t w = ::write(fd, p + done, count - done);
        if (w < 0) {
            if (errno == EINTR) continue;
            LOGGER.e(0, "write() failed on [" + path + "]: " + std::string(std::strerror(errno)));
        }
        done += static_cast<size_t>(w);
    }
}


// Returns:
//   ids      — "FID\tIID" strings in GRM file order
//   G        — full symmetric n×n matrix (double precision)
//   m_snps   — SNP count from element (0,0) of .grm.N.bin; 0 if N file missing
//
// Implementation notes:
//   - .grm.bin is read sequentially in row-aligned chunks (grm_read_chunk_
//     bytes each, default 1GiB) via a single blocking read() stream, not
//     mmap'd. This was a deliberate change from an earlier mmap-based
//     version: even with each thread faulting in its own monotonically
//     increasing row range, N threads doing so *concurrently* against the
//     same mapping still presents the kernel/filesystem with N interleaved
//     access offsets, which defeated page-cache readahead and, on this
//     project's Lustre-backed cluster storage, fragmented reads across
//     whichever OSTs happened to back each thread's current extent --
//     regressions from ~2s to 900+s were observed at n=75000. A single
//     sequential reader avoids this by construction, independent of
//     filesystem/OS specifics -- no platform-specific calls needed.
//   - Only the read itself is single-threaded; float→double conversion
//     and the scatter into G are parallelized per chunk, once that
//     chunk's bytes are already in memory (see the loop below).
//   - The chunk buffer is the only extra transient allocation (bounded by
//     grm_read_chunk_bytes, independent of n) -- negligible next to G's
//     own n²·8 bytes and to what every downstream consumer of G already
//     needs at peak.
inline void read_grm_binary(const std::string& prefix,
                             std::vector<std::string>& ids,
                             Eigen::MatrixXd& G,
                             double& m_snps,
                             size_t grm_read_chunk_bytes = (1ull << 30) /* 1GiB */)
{
    LOGGER.ts("grm");
    using std::to_string;

    ids = Pheno::read_sublist(prefix + ".grm.id");
    const int n = static_cast<int>(ids.size());
    if (n == 0) LOGGER.e(0, "GRM id file [" + prefix + ".grm.id] is empty.");

    const size_t tri      = static_cast<size_t>(n) * (n + 1) / 2;
    const size_t byte_len = tri * sizeof(float);

    // ---- read .grm.bin in row-aligned chunks, fill G -------------------
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

    // Scatter the lower-triangle float32 entries into G (double precision)
    // one row-aligned chunk at a time: a single blocking read() pulls the
    // next chunk in fully (read_exact, above -- one coherent sequential
    // stream, no concurrent-offset I/O), then the cast+scatter of that
    // chunk's rows into G is parallelized (schedule(dynamic, 64) balances
    // the triangular per-row workload). Column-contiguous write per row
    // (G.col(i).head(i+1)) for the same reason as before: it matches the
    // chunk buffer's own contiguous row-major-packed layout, keeping the
    // cast vectorizable.
    G.resize(n, n);
    const size_t max_elems_per_chunk = std::max<size_t>(1, grm_read_chunk_bytes / sizeof(float));
    std::vector<float> chunk_buf(max_elems_per_chunk);
    int row_cursor = 0;
    while (row_cursor < n) {
        const size_t row_base_cursor = static_cast<size_t>(row_cursor) * (row_cursor + 1) / 2;
        const size_t target = row_base_cursor + max_elems_per_chunk;
        const int row_end = row_bound_for_cumulative(target, row_cursor, n);
        const size_t row_base_end = static_cast<size_t>(row_end) * (row_end + 1) / 2;
        const size_t elems_this_chunk = row_base_end - row_base_cursor;

        if (elems_this_chunk > chunk_buf.size())
            chunk_buf.resize(elems_this_chunk); // only ever needed if a single row exceeds the chunk budget
        read_exact(fd, chunk_buf.data(), elems_this_chunk * sizeof(float), bin_path);

        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = row_cursor; i < row_end; ++i) {
            const size_t local_off = static_cast<size_t>(i) * (i + 1) / 2 - row_base_cursor;
            G.col(i).head(i + 1) =
                Eigen::Map<const Eigen::VectorXf>(chunk_buf.data() + local_off, i + 1).cast<double>();
        }
        row_cursor = row_end;
    }
    ::close(fd);
    LOGGER.i(0, "The .grm.bin fill took " + std::to_string(LOGGER.tp("grm")) + " seconds.");
    LOGGER.ts("grm_mirror");
    // Mirror upper -> lower, column-contiguous on the write side (fixed
    // column c, rows c+1..n-1) and scattered only on the read side (G(c,r)
    // across row c). A blocked-transpose version of this (tiling into
    // BSxBS blocks to make the read side contiguous too) was tried and
    // hung/regressed badly at n=75000 -- root cause not yet identified
    // (suspect concurrent first-touch page faults on the still-unwritten
    // lower triangle across many threads, but not confirmed). Reverted to
    // this simpler version, which is validated (ran cleanly across the
    // K10-K25 Woodbury sweep). Do not reintroduce blocking without
    // isolating and timing it separately first.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int c = 0; c < n; ++c)
        for (int r = c + 1; r < n; ++r)
            G(r, c) = G(c, r);
    LOGGER.i(0, "The .grm.bin mirror took " + std::to_string(LOGGER.tp("grm_mirror")) + " seconds.");

    if (m_snps < 0.0) {
        LOGGER.i(0,"Skipping SNP count read from .grm.N.bin.");
    }
    else {
        // ------------------------------------------------------------------ //
        // Read SNP count (.grm.N.bin) — same lower-triangle layout as        //
        // .grm.bin. We only need the diagonal (each individual's own         //
        // non-missing SNP count), so extract it without materialising the    //
        // full n×n N matrix. read_grm_N_mean (below) does exactly this --    //
        // call it rather than duplicating the extraction logic here.         //
        // ------------------------------------------------------------------ //
        m_snps = read_grm_N_mean(prefix, n);
    }
}

// Build a mapping from a reference ID list to indices in grm_ids.
// Returns kp[i] = index in grm_ids matching ref_ids[i], or -1 if not found.
//
// Uses unordered_map (O(1) average lookup) rather than std::map (O(log n))
// and reserves capacity upfront to avoid rehashing.
inline std::vector<int> match_ids_to_grm(const std::vector<std::string>& ref_ids,
                                          const std::vector<std::string>& grm_ids)
{
    std::unordered_map<std::string, int> grm_map;
    grm_map.reserve(grm_ids.size() * 2); // factor of 2 keeps load factor ≤ 0.5
    for (int i = 0; i < static_cast<int>(grm_ids.size()); ++i)
        grm_map.emplace(grm_ids[i], i);

    std::vector<int> kp(ref_ids.size(), -1);
    for (int i = 0; i < static_cast<int>(ref_ids.size()); ++i) {
        if (const auto it = grm_map.find(ref_ids[i]); it != grm_map.end())
            kp[i] = it->second;
    }
    return kp;
}

// Tile reader for --reml-svd-chunked: returns K in ANALYSIS sample order
// (post-kp reindexing), without ever materializing a dense matrix — not
// even transiently, and not just the n x n analysis-subsetted one, unlike
// read_grm_binary() above.
class ChunkedGrmReader {
public:
    // kp[i] = row index (in whatever file this wraps) for analysis
    // individual i (see match_ids_to_grm). Every entry must be >= 0 —
    // validate before constructing, since a -1 here means "individual not
    // in this file" and silently reading garbage at that index is far worse
    // than refusing to start.
    //
    // `path` is the full file path, not a prefix — .grm.bin and .grm.N.bin
    // share an identical packed-lower-triangular float32 layout, so this
    // class serves either; callers needing both (e.g. a weighted merge)
    // construct two instances against the same kp.
    ChunkedGrmReader(const std::string& path, std::vector<int> kp, int n_grm)
        : kp_(std::move(kp))
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

        kp_is_identity_ = true;
        kp_is_monotonic_ = true;
        for (int i = 0; i < static_cast<int>(kp_.size()); ++i) {
            if (kp_[i] != i) kp_is_identity_ = false;
            if (i > 0 && kp_[i] <= kp_[i - 1]) kp_is_monotonic_ = false;
        }

        // Read the whole packed file with one portable, single-threaded
        // sequential read_exact() call into an owned buffer, rather than
        // mmap + madvise + a manual warm-up sweep (the prior version of
        // this constructor). Two things motivated dropping mmap here, not
        // just tuning it further:
        //  1. Every mmap-based path in this file has needed a workaround
        //     for the same underlying issue on this project's Lustre-backed
        //     cluster storage -- read_grm_binary's original dense fill
        //     (concurrent per-thread page faults), merge_grms's
        //     schedule(dynamic) mixing loop, and this class's own read_tile
        //     scatter, all independently regressed the same way. The dense
        //     loader's read()-based chunked design has been robust in every
        //     case it's been tried, including at the largest scales tested
        //     so far; mmap's page-fault-driven access, even single-threaded
        //     and even with an explicit warm-up sweep, has not been.
        //  2. This makes the class's memory cost an explicit, fixed heap
        //     allocation (byte_len_ bytes, known at construction) instead
        //     of page-cache-resident-but-technically-reclaimable pages --
        //     a plain heap buffer is simpler to reason about for memory
        //     budgeting than "resident right now, but the kernel is free
        //     to evict it under pressure and re-fault it later."
        // read_exact already handles retrying on short reads (routine on
        // network filesystems) and errors out via LOGGER on real failure,
        // so this one call is the whole read -- no chunk loop needed here
        // (unlike read_grm_binary's fill, there's no per-chunk float->double
        // cast or scatter to interleave; this class stores the packed
        // float32 data verbatim and defers the cast to read_tile/
        // matvec_blocked, at tile/row granularity, same as before).
        data_.resize(tri);
        read_exact(fd_, data_.data(), byte_len_, path);
        ::close(fd_);
        fd_ = -1;
        fbuf_ = data_.data();

        if (!kp_is_monotonic_)
            LOGGER.w(0, "--svd-chunked-budget: the analysis sample order does not match [" +
                        path + "]'s order (individuals were reordered, not just subsetted). "
                        "GRM tile reads degrade to scattered per-entry access in this case, "
                        "which costs RAM/cache locality (not filesystem I/O, since the whole "
                        "file is already loaded at this point) but can still be noticeably "
                        "slower than the monotonic case.");
    }

    ~ChunkedGrmReader() {
        if (fd_ != -1) ::close(fd_);
    }
    ChunkedGrmReader(const ChunkedGrmReader&)            = delete;
    ChunkedGrmReader& operator=(const ChunkedGrmReader&) = delete;

    // K_analysis[rs:re, cs:ce]. Every consumer (chunked_symmetric_matvec,
    // chunked_diagonal, chunked_trace_K_squared) only ever reads a
    // diagonal-block tile (rs==cs) through selfadjointView<Lower>() or
    // .diagonal() — so only the tile's lower triangle (lq <= lp) needs to
    // be valid there; the monotonic fast path below relies on that and
    // leaves the upper triangle uninitialized for diagonal tiles. For a
    // genuinely scrambled (non-monotonic) kp there's no cheaper option, so
    // the fallback below still fills the whole tile per entry.
    Eigen::MatrixXd read_tile(int rs, int re, int cs, int ce) const {
        const int tile_rows = re - rs, tile_cols = ce - cs;
        Eigen::MatrixXd tile(tile_rows, tile_cols);

        if (!kp_is_monotonic_) {
            // Genuinely scrambled kp: no exploitable locality, every entry
            // can live on a different page of a possibly huge file.
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = kp_[rs + lp];
                for (int lq = 0; lq < tile_cols; ++lq) {
                    const int gj = kp_[cs + lq];
                    tile(lp, lq) = read_raw(gi, gj);
                }
            }
            return tile;
        }

        if (kp_is_identity_) {
            // No reindexing at all: gj - gj_lo == lq exactly, so the source
            // span maps onto the destination row with no gaps. Skip kp_[]
            // (and the per-entry gj/subtraction arithmetic below) entirely
            // and let Eigen vectorize the float->double widen as one cast
            // instead of a hand-rolled scalar gather loop.
            const bool diagonal_tile = (rs == cs);
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = rs + lp;
                const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                if (lq_end == 0) continue;
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2;
                const float* row_span = fbuf_ + row_base + cs;
                tile.row(lp).head(lq_end) =
                    Eigen::Map<const Eigen::RowVectorXf>(row_span, lq_end).cast<double>();
            }
            return tile;
        }

        // Monotonic kp (an order-preserving subset, not full identity): kp
        // is strictly increasing, so every gj needed by row lp satisfies
        // gj <= gi (true for off-diagonal tiles because the whole column
        // block precedes the row block; true for the diagonal tile's lower
        // triangle because lq <= lp there) — so file_row == gi is constant
        // across the row and the needed file_col's form one increasing run.
        // Bulk-read that run once instead of touching the mmap per entry;
        // this is what turns a per-entry page-fault storm on a huge file
        // into one sequential read per output row.
        const bool diagonal_tile = (rs == cs);
        for (int lp = 0; lp < tile_rows; ++lp) {
            const int gi = kp_[rs + lp];
            const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;  // upper triangle unused for diagonal tiles
            if (lq_end == 0) continue;
            const int gj_lo = kp_[cs];
            const int gj_hi = kp_[cs + lq_end - 1];
            const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2;
            const float* row_span = fbuf_ + row_base + gj_lo;
            for (int lq = 0; lq < lq_end; ++lq) {
                const int gj = kp_[cs + lq];
                tile(lp, lq) = static_cast<double>(row_span[gj - gj_lo]);
            }
        }
        return tile;
    }

    // Single-entry accessor for scattered (not tile-shaped) access patterns
    // — e.g. GRM merging, which looks up one (analysis_row, analysis_col)
    // pair at a time rather than processing contiguous blocks. Same index
    // math as read_tile, just without constructing an Eigen::MatrixXd for
    // one value.
    double read_entry(int analysis_row, int analysis_col) const {
        return read_raw(kp_[analysis_row], kp_[analysis_col]);
    }

    // Fused y = Kx for the packed lower-triangular mmap, specialized for the
    // single-vector case that dominates chunked Lanczos. This bypasses the
    // tile -> MatrixXd materialization path entirely.
    //
    // partials_scratch_/y_scratch_ are mutable, lazily sized on first call,
    // and reused across every subsequent call -- this runs once per Lanczos
    // matvec (hundreds of times per eigendecomposition), and was previously
    // a fresh n x num_threads allocation (+ a fresh per-thread n-length
    // Zero() vector) on every single call.
    Eigen::VectorXd matvec_blocked(const Eigen::Ref<const Eigen::VectorXd>& x,
                                   int block_size) const {
        const int n = static_cast<int>(kp_.size());
        if (x.size() != n)
            throw std::invalid_argument("ChunkedGrmReader::matvec_blocked: x has wrong length.");

#ifdef _OPENMP
        const int num_threads = omp_get_max_threads();
        if (num_threads > 1) {
            if (partials_scratch_.rows() != n || partials_scratch_.cols() != num_threads)
                partials_scratch_.resize(n, num_threads);
            partials_scratch_.setZero();
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                auto y_local = partials_scratch_.col(tid);  // view into scratch, not a fresh alloc
                #pragma omp for schedule(static, block_size > 0 ? block_size : 1)
                for (int r = 0; r < n; ++r) {
                    const double xr = x[r];
                    const int gi = kp_[r];
                    double acc = 0.0;

                    if (kp_is_identity_) {
                        const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2;
                        const float* row = fbuf_ + row_base;
                        for (int c = 0; c < r; ++c) {
                            const double a = static_cast<double>(row[c]);
                            acc += a * x[c];
                            y_local[c] += a * xr;
                        }
                        acc += static_cast<double>(row[r]) * xr;
                    } else {
                        for (int c = 0; c < r; ++c) {
                            const double a = read_raw(gi, kp_[c]);
                            acc += a * x[c];
                            y_local[c] += a * xr;
                        }
                        acc += read_raw(gi, gi) * xr;
                    }

                    y_local[r] += acc;
                }
            }
            return partials_scratch_.rowwise().sum();
        }
#endif

        // Single-threaded fallback (rare in practice -- any real SLURM
        // allocation runs with cpus-per-task > 1). Left as a plain local: it
        // already gets RVO/guaranteed-move on return, so persistent scratch
        // here would trade that for a mandatory copy on every call instead.
        Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
        for (int r = 0; r < n; ++r) {
            const double xr = x[r];
            const int gi = kp_[r];
            double acc = 0.0;

            if (kp_is_identity_) {
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2;
                const float* row = fbuf_ + row_base;
                for (int c = 0; c < r; ++c) {
                    const double a = static_cast<double>(row[c]);
                    acc += a * x[c];
                    y[c] += a * xr;
                }
                acc += static_cast<double>(row[r]) * xr;
            } else {
                for (int c = 0; c < r; ++c) {
                    const double a = read_raw(gi, kp_[c]);
                    acc += a * x[c];
                    y[c] += a * xr;
                }
                acc += read_raw(gi, gi) * xr;
            }

            y[r] += acc;
        }

        return y;
    }

private:
    mutable Eigen::MatrixXd partials_scratch_;  // n x num_threads, reused across matvec_blocked calls

    double read_raw(int gi, int gj) const {
        const int file_row = std::max(gi, gj);
        const int file_col = std::min(gi, gj);
        const size_t idx = static_cast<size_t>(file_row) * (file_row + 1) / 2
                          + static_cast<size_t>(file_col);
        return static_cast<double>(fbuf_[idx]);
    }

    std::vector<int> kp_;
    int fd_ = -1;
    size_t byte_len_ = 0;
    std::vector<float> data_;      // owned storage for fbuf_ (see constructor)
    const float* fbuf_ = nullptr;  // = data_.data(), cached for existing call sites
    bool kp_is_identity_ = false;
    bool kp_is_monotonic_ = false;
};

// .grm.N.bin diagonal only (mean SNP count) — same file, same packed layout,
// same lower-triangle-by-row indexing as .grm.bin, but this touches only n
// scattered entries out of a possible n(n+1)/2. read_grm_binary (above) now
// calls this directly rather than duplicating the extraction.
//
// Deliberately NOT the whole-file warm-up pattern used by read_grm_binary's
// dense fill or ChunkedGrmReader's constructor: those warm the whole file
// because they go on to use (or may use) most or all of it, so paying for
// one coherent sequential pass once is strictly better than letting a
// scattered/concurrent access pattern re-derive that same data cold. Here,
// the diagonal genuinely is all we want -- n elements out of possibly
// billions -- so warming the whole file would mean fetching gigabytes to
// read a few hundred KB of actually-needed data. With no sequential
// alternative being defeated (there's no "right order" to visit n
// unrelated, non-adjacent offsets in), concurrency here is a straightforward
// win rather than the regression it caused elsewhere in this file: each
// pread() targets an independent, already-known offset, so running many in
// flight hides per-request latency instead of interleaving otherwise-
// sequential streams. pread() (not mmap+page-fault) makes that concurrency
// explicit and, unlike a page fault, gives us a return value to check --
// a failed read here fails loudly instead of segfaulting.
inline double read_grm_N_mean(const std::string& prefix, int n_grm) {
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
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n_grm; ++i) {
        const size_t diag_idx = static_cast<size_t>(i) * (i + 1) / 2 + i;
        float v = 0.0f;
        const ssize_t r = ::pread(nfd, &v, sizeof(v),
                                   static_cast<off_t>(diag_idx * sizeof(float)));
        if (r == static_cast<ssize_t>(sizeof(v))) {
            sum += static_cast<double>(v);
        } else {
            #pragma omp atomic write
            read_failed = true;
        }
    }
    ::close(nfd);

    if (read_failed) {
        LOGGER.w(0, "pread() failed while reading [" + n_path + "]'s diagonal; "
                    "SNP count unavailable (affects --reml-woodbury auto-k).");
        return 0.0;
    }
    return sum / n_grm;
}


struct ChunkedGrmHandle {
    gcta_chunked::TileReader reader;
    std::shared_ptr<const ChunkedGrmReader> file;
    double m_snps = 0.0;
};

// Build the chunked reader (+ m_snps, read the same way read_grm_binary()
// reads it) for RemlCtx::grm_tile_reader. analysis_ids must be in the exact
// row/column order ctx.y/ctx.X/ctx.A would be built in — same requirement
// as load_pca_warm_start's alignment in MLMA_stream.cpp. Fails loudly (not
// a fallback) on any individual missing from the GRM: a silent misalignment
// here corrupts every downstream REML result without any obvious symptom.
inline ChunkedGrmHandle make_chunked_grm_reader(
    const std::string& prefix,
    const std::vector<std::string>& analysis_ids)
{
    const std::vector<std::string> grm_ids = Pheno::read_sublist(prefix + ".grm.id");
    const int n_grm = static_cast<int>(grm_ids.size());

    std::vector<int> kp = match_ids_to_grm(analysis_ids, grm_ids);
    for (int i = 0; i < static_cast<int>(analysis_ids.size()); ++i) {
        if (kp[i] < 0)
            LOGGER.e(0, "--reml-svd-chunked: individual [" + analysis_ids[i] +
                        "] not found in GRM [" + prefix + ".grm.id].");
    }

    ChunkedGrmHandle handle;
    handle.m_snps = read_grm_N_mean(prefix, n_grm);
    auto file = std::make_shared<ChunkedGrmReader>(prefix + ".grm.bin", std::move(kp), n_grm);
    handle.file = file;
    handle.reader = [file](int rs, int re, int cs, int ce) -> Eigen::MatrixXd {
        return file->read_tile(rs, re, cs, ce);
    };
    return handle;
}

// Merge K GRMs that share the exact same sample order (no subsetting, no
// reindexing) into one N-weighted-average GRM, entirely streamed: never
// holds a dense n x n matrix for any input file, nor for the output.
//
// Each row-block is read via one sequential read() per input file (2K reads
// per block total: .grm.bin and .grm.N.bin for each of the K inputs), not
// mmap. An earlier mmap-based version parallelized the per-row mixing loop
// with schedule(dynamic, 64) directly against the mmap'd files -- which
// means up to 2K files were faulted in via non-monotonic, interleaved
// offsets (threads grab row-chunks in whatever order they finish, not in
// row order), the same failure mode diagnosed for read_grm_binary's
// original dense fill (see that function's comment) and for
// ChunkedGrmReader's constructor. Reading each block into a plain buffer
// first, single-threaded and strictly in order, then parallelizing only the
// in-RAM mixing math, avoids that: the row_block_rows loop itself is
// already the "coherent sequential stream" every filesystem's readahead is
// built around, so there's no reason to let the compute-side parallelism
// leak back into how the files are read.
//
// row_block_rows bounds the only thing that isn't O(1): the per-block
// buffers (input read buffers and the output merge buffer), which hold one
// row-block's worth of values before each write — same role as
// --reml-svd-chunk-size elsewhere, smaller for tighter RSS.
//
// If your GRMs might have different sample orderings, this isn't the right
// tool — use ChunkedGrmReader-based per-entry lookups instead (each source
// file gets its own kp).
inline int solve_merge_chunk_rows(int n, int K, double budget_gb) {
    const double budget_bytes = budget_gb * 1e9;
    const double bytes_per_row = 4.0 * static_cast<double>(n) * (2.0 * K + 2.0);
    const int chunk_rows = static_cast<int>(budget_bytes / bytes_per_row);
    return std::min(std::max(chunk_rows, 0), n);
}

// row_block_rows: caller-chosen block size, unchanged default (4000) for
// backward compatibility with existing callers. memory_budget_gb: if > 0,
// OVERRIDES row_block_rows with solve_merge_chunk_rows(n, K, budget) --
// prefer this over hand-picking row_block_rows, since the right block size
// depends on K (number of inputs), which the caller may not know until
// prefixes.size() is available, and because a fixed row_block_rows chosen
// without K in mind is exactly what caused this function's memory to scale
// unboundedly with K (see solve_merge_chunk_rows's comment above).
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

    // Validate every input shares the exact same sample order — cheap (a
    // handful of small text-file reads), and the alternative (silently
    // merging misaligned GRMs) produces a confidently wrong result with no
    // symptom, which is a far worse failure mode than refusing to start.
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

    // Sizing chunk_elems: the packed lower triangle has tri = n(n+1)/2 floats.
    // Every element is independent in the N-weighted merge, so chunking by a
    // fixed number of float elements (rather than a fixed number of rows)
    // keeps memory strictly constant across iterations, avoids vector reallocations,
    // and eliminates the quadratic growth of row-based buffers at large n.
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
        // Default: 16 MiB (4M floats) per stream buffer, capped at 512 MiB total across all 2K+2 buffers.
        constexpr size_t default_target_elems = 4 * 1024 * 1024;
        constexpr size_t max_total_bytes = 512ull << 20;
        const size_t max_elems_from_cap = max_total_bytes / (num_buffers * sizeof(float));
        chunk_elems = std::min(default_target_elems, max_elems_from_cap);
        chunk_elems = std::clamp<size_t>(chunk_elems, 65536, tri);
        LOGGER.i(0, "merge_grms: using " +
                    std::to_string((chunk_elems * sizeof(float)) >> 20) +
                    " MB per stream buffer (" +
                    std::to_string((num_buffers * chunk_elems * sizeof(float)) >> 20) +
                    " MB total buffer memory). Set `--merge-grm-streaming <GB>` to customize.");
    }

    struct OpenGrmFile {
        int fd = -1;
        std::string path;  // kept for read_exact's error messages
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

    // Allocate all stream buffers once up front. No reallocations occur in the streaming loop.
    std::vector<float> out_val_buf(chunk_elems), out_n_buf(chunk_elems);
    std::vector<std::vector<float>> val_block_bufs(K), n_block_bufs(K);
    for (int f = 0; f < K; ++f) {
        val_block_bufs[f].resize(chunk_elems);
        n_block_bufs[f].resize(chunk_elems);
    }

    for (size_t offset = 0; offset < tri; offset += chunk_elems) {
        const size_t elems_this_chunk = std::min(chunk_elems, tri - offset);
        const size_t bytes_this_chunk = elems_this_chunk * sizeof(float);

        // Pull this block from input files in parallel across all 2K input streams
        // (.grm.bin and .grm.N.bin for each input). Every stream has an independent
        // file descriptor and advances sequentially block by block.
        const int max_io_threads = std::min(omp_get_max_threads(), 8);
        #pragma omp parallel for schedule(dynamic, 1) num_threads(max_io_threads)
        for (int f = 0; f < K; ++f) {
            read_exact(val_files[f].fd, val_block_bufs[f].data(),
                    bytes_this_chunk, val_files[f].path);
            read_exact(n_files[f].fd, n_block_bufs[f].data(),
                    bytes_this_chunk, n_files[f].path);
        }

        // Compute the N-weighted average across all elements in this chunk.
        // Tiled in L1-cache-sized blocks (4096 floats = 32KB per stack tile)
        // with static OpenMP scheduling for uniform load balancing.
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

        // Write the two independent output files in parallel.
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                write_exact(out_bin_fd, out_val_buf.data(), bytes_this_chunk, out_bin_path);
            }
            #pragma omp section
            {
                write_exact(out_n_fd, out_n_buf.data(), bytes_this_chunk, out_n_path);
            }
        }
    }

    ::close(out_bin_fd);
    ::close(out_n_fd);

    for (auto& f : val_files) ::close(f.fd);
    for (auto& f : n_files)   ::close(f.fd);

    // .grm.id is identical across inputs (already validated) — copy it once.
    // dst << src.rdbuf() has no built-in error signalling: an unopenable
    // source or a write failure (disk full, permissions) both produce a
    // silently truncated or empty output .grm.id with no exception and no
    // nonzero exit -- exactly the "confidently wrong result with no
    // symptom" failure mode this function already refuses to risk for the
    // sample-order check above. Check explicitly rather than trust the
    // stream state implicitly.
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
