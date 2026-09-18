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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

// pread-based counterpart to read_exact above: reads exactly `count` bytes
// starting at `offset`, retrying on EINTR and on short reads. pread takes
// the offset as a call argument rather than mutating shared file-position
// state, so unlike read()+lseek it's safe to call from any thread without
// coordinating with others sharing the same fd -- ChunkedGrmReader only
// ever calls this from one thread at a time regardless (see read_tile's
// reentrancy note), but the safety property is what makes that fd share-
// without-locking legitimate in the first place.
inline void read_exact_at(int fd, void* buf, size_t count, off_t offset, const std::string& path) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < count) {
        const ssize_t r = ::pread(fd, p + done, count - done, offset + static_cast<off_t>(done));
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
//   G        — full symmetric n×n matrix (double precision), unless
//              upper_only is set (see below), in which case only the upper
//              triangle (row <= col) holds valid data -- the lower triangle
//              is whatever the allocator handed back (kernel-zero pages for
//              a fresh, sufficiently large Eigen::MatrixXd; not guaranteed
//              zero in general -- see upper_only note).
//   m_snps   — SNP count from element (0,0) of .grm.N.bin; 0 if N file missing
//
// upper_only (default false): skip the upper->lower mirror pass after the
// triangle fill. Saves ~n^2/2 double writes and, for a freshly-constructed
// G (resize() on an empty matrix, no prior .setZero()/full touch), avoids
// physically faulting in the lower-triangle pages at all -- confirmed via
// RSS sampling (~n^2/2 * 8 bytes reduction at n=150k on the REML load
// path). This last part depends on G having received no prior full-matrix
// touch; if G is reused across calls or zero-initialised by the caller
// first, the memory saving does not apply (the mirror-pass write-bandwidth
// saving still does). Only pass upper_only=true when every downstream
// consumer of G has been updated to read it as upper-triangle-only (dense
// products via selfadjointView<Eigen::Upper>(), symmetric reads via row
// instead of column tail on the lower half). Default false preserves the
// original full-matrix contract for any caller not yet audited.
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
//     whichever OSTs happened to back each thread's current extent.
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
                             bool upper_only = false,
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
    if (!upper_only) {
        LOGGER.ts("grm_mirror");
        #pragma omp parallel for schedule(dynamic, 64)
        for (int c = 0; c < n; ++c)
            for (int r = c + 1; r < n; ++r)
                G(r, c) = G(c, r);
        LOGGER.i(0, "The .grm.bin mirror took " + std::to_string(LOGGER.tp("grm_mirror")) + " seconds.");
    }

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
// (post-kp reindexing), keeping only a bounded row-band resident at once
// (see ensure_band below) rather than the whole file. Requires kp to be at
// least an order-preserving subset of the file's row order (identity or
// monotonic) -- see the constructor check for why scrambled kp isn't
// supported: a row-band cache only pays off when each band's bytes are
// read once and reused by every tile touching those rows, which needs
// exactly the monotonicity this class requires.
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
    //
    // band_byte_budget caps how much of the file ensure_band() keeps
    // resident at once (default 1GiB). Callers that size --grm-chunked-budget
    // against a fixed reservation for this reader (see solve_chunk_rows's
    // reserved_gb) must pass that same number here — the two are meant to
    // be the same budget split two ways, not independently chosen.
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

        kp_is_identity_ = true;
        bool kp_is_monotonic = true;
        for (int i = 0; i < static_cast<int>(kp_.size()); ++i) {
            if (kp_[i] != i) kp_is_identity_ = false;
            if (i > 0 && kp_[i] <= kp_[i - 1]) kp_is_monotonic = false;
        }

        // A scrambled kp has no row-range locality: the band cache below
        // only pays off because each row-band's full span is read once and
        // reused by every tile touching those rows. Without monotonicity
        // that reuse doesn't happen, and this degrades to a pread per
        // output entry -- millions of syscalls for a single mid-size tile,
        // far worse than the dense path. Rejected outright, no fallback
        // (an earlier version of this class had a scrambled-kp fallback
        // path; removed together with kp_is_monotonic_/read_raw once this
        // became a hard error instead of a warning).
        if (!kp_is_monotonic)
            LOGGER.e(0, "--reml-svd-chunked requires the analysis sample order to match "
                        "(or be an ordered subset of) [" + path + "]'s row order. Re-run "
                        "with sample IDs sorted to match the GRM's native order, or drop "
                        "--reml-svd-chunked to use the dense path.");

        // fd_ stays open for this object's lifetime; ensure_band() preads
        // from it on demand, band by band. No whole-file buffer is
        // allocated here (unlike the earlier version of this constructor,
        // which read the entire file up front -- see chat history for why
        // that was dropped in favor of a bounded band cache).
    }

    ~ChunkedGrmReader() {
        if (fd_ != -1) ::close(fd_);
    }
    ChunkedGrmReader(const ChunkedGrmReader&)            = delete;
    ChunkedGrmReader& operator=(const ChunkedGrmReader&) = delete;

    // K_analysis[rs:re, cs:ce]. Every consumer (chunked_symmetric_matvec,
    // chunked_diagonal, chunked_trace_K_squared) only ever reads a
    // diagonal-block tile (rs==cs) through selfadjointView<Lower>() or
    // .diagonal() -- so only the tile's lower triangle (lq <= lp) needs to
    // be valid there; both paths below leave the upper triangle
    // uninitialized for diagonal tiles.
    //
    // NOT reentrant: this call, and the ensure_band refill it may trigger,
    // mutate band_buf_/band_lo_/band_hi_ and tile_scratch_ with no locking.
    // Only the #pragma omp parallel for loop body inside one call may run
    // concurrently. Two overlapping read_tile calls on the same reader --
    // e.g. a caller that parallelizes the outer row-band loop instead of
    // just the inner per-row loop -- will race on which band is resident
    // and silently produce wrong numbers, not a crash.
    //
    // Also assumes rs is non-decreasing across a sweep (each call's row
    // range starts at or after the previous call's, as chunked_grm_matvec's
    // row-owner traversal does): ensure_band only grows the band forward,
    // so a call requesting an earlier row than the current band's start
    // pays a fresh read from that point -- still correct, just not the
    // sequential-pass behavior this cache is for.
    Eigen::Ref<const Eigen::MatrixXd> read_tile(int rs, int re, int cs, int ce) const {
        const int tile_rows = re - rs, tile_cols = ce - cs;
        if (tile_scratch_.rows() < tile_rows || tile_scratch_.cols() < tile_cols) {
            tile_scratch_.resize(std::max<Eigen::Index>(tile_scratch_.rows(), tile_rows),
                                  std::max<Eigen::Index>(tile_scratch_.cols(), tile_cols));
        }
        auto tile = tile_scratch_.topLeftCorner(tile_rows, tile_cols);
        const bool diagonal_tile = (rs == cs);

        // kp_ is strictly increasing (enforced at construction), so
        // kp_[rs]/kp_[re-1] are this tile's min/max file row. Single-
        // threaded pread, outside the parallel loop below.
        ensure_band(kp_[rs], kp_[re - 1] + 1);

        const float* buf = band_buf_.data();
        const size_t band_elem_lo = static_cast<size_t>(band_lo_) * (band_lo_ + 1) / 2;

        if (kp_is_identity_) {
            // No reindexing at all: gj - gj_lo == lq exactly, so the source
            // span maps onto the destination row with no gaps. Skip kp_[]
            // (and the per-entry gj/subtraction arithmetic below) entirely
            // and let Eigen vectorize the float->double widen as one cast
            // instead of a hand-rolled scalar gather loop.
            #pragma omp parallel for schedule(dynamic, 64)
            for (int lp = 0; lp < tile_rows; ++lp) {
                const int gi = rs + lp;
                const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;
                if (lq_end == 0) continue;
                const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
                const float* row_span = buf + row_base + cs;
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
        // across the row. Still a scalar per-entry gather (kp_ may have
        // gaps within [cs, cs+lq_end), so the needed file columns aren't
        // necessarily contiguous even though they're bounded and increasing)
        // — bounded to one row's own span (now within the resident band,
        // not the whole file) rather than a filesystem read.
        #pragma omp parallel for schedule(dynamic, 64)
        for (int lp = 0; lp < tile_rows; ++lp) {
            const int gi = kp_[rs + lp];
            const int lq_end = diagonal_tile ? (lp + 1) : tile_cols;  // upper triangle unused for diagonal tiles
            if (lq_end == 0) continue;
            const int gj_lo = kp_[cs];
            const size_t row_base = static_cast<size_t>(gi) * (gi + 1) / 2 - band_elem_lo;
            const float* row_span = buf + row_base + gj_lo;
            for (int lq = 0; lq < lq_end; ++lq) {
                const int gj = kp_[cs + lq];
                tile(lp, lq) = static_cast<double>(row_span[gj - gj_lo]);
            }
        }
        return tile;
    }

private:
    mutable Eigen::MatrixXd tile_scratch_;

    // [band_lo_, band_hi_) = file-row range currently resident in
    // band_buf_. The packed layout has row_base(i+1) == row_base(i) + (i+1),
    // so any contiguous run of file rows is one contiguous byte range,
    // loadable with a single sequential pread.
    //
    // ensure_band grows band_hi_ past what the immediate tile needs, up to
    // band_byte_budget_ total, via row_bound_for_cumulative -- the same
    // closed-form triangular-row math read_grm_binary/merge_grms already
    // use for their own chunk boundaries -- rather than a fixed row count,
    // since a fixed row count is a very different byte size near the top
    // of the matrix than the bottom. Growing past the immediate need also
    // means later tiles sharing this row range (different column tiles,
    // same row-band, as in chunked_symmetric_matvec's inner j loop) hit
    // the resident buffer instead of re-reading.
    mutable std::vector<float> band_buf_;
    mutable int band_lo_ = 0, band_hi_ = 0;

    void ensure_band(int file_lo, int file_hi) const {
        if (file_lo >= band_lo_ && file_hi <= band_hi_) return;   // already resident

        const size_t elem_lo = static_cast<size_t>(file_lo) * (file_lo + 1) / 2;
        const size_t target_elems = elem_lo + band_byte_budget_ / sizeof(float);
        const int grown_hi  = row_bound_for_cumulative(target_elems, file_lo, n_grm_);
        const int actual_hi = std::max(file_hi, grown_hi);  // never smaller than this tile's own need

        const size_t elem_hi = static_cast<size_t>(actual_hi) * (actual_hi + 1) / 2;
        band_buf_.resize(elem_hi - elem_lo);
        read_exact_at(fd_, band_buf_.data(), (elem_hi - elem_lo) * sizeof(float),
                      static_cast<off_t>(elem_lo * sizeof(float)), path_);
        band_lo_ = file_lo;
        band_hi_ = actual_hi;
    }

    std::vector<int> kp_;
    std::string path_;   // kept for ensure_band's read_exact_at error messages
    int fd_ = -1;
    size_t byte_len_ = 0;
    int n_grm_ = 0;
    size_t band_byte_budget_ = 0;
    bool kp_is_identity_ = false;
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


// Splits a --grm-chunked-budget-style total (GB) into the slice reserved for
// ChunkedGrmReader's row-band cache. Kept as one function, rather than each
// call site independently picking a fraction, so the band_byte_budget passed
// to make_chunked_grm_reader and the reserved_gb passed to
// gcta_chunked::solve_chunk_rows can't silently drift apart -- see
// chunked_grm_matvec.hpp's reserved_gb doc for why that drift matters
// (previously reserved_gb was left at its 0.0 default at one call site and
// set to the full packed-file size, a stale pre-band-cache assumption, at
// the other -- both call sites should route through this instead).
// Capped at 512MB so a large total budget doesn't hand the reader far more
// than one row-band ever needs; 10% so a small budget still leaves the
// reader a usable slice.
inline double chunked_reader_reserved_gb(double total_budget_gb) {
    return std::min(0.5, total_budget_gb * 0.10);
}

struct ChunkedGrmHandle {
    gcta_chunked::TileReader reader;
    std::shared_ptr<const ChunkedGrmReader> file;
    double m_snps = 0.0;
    int chunk_rows = 0;   // block_size for chunked_symmetric_matvec/diagonal/trace_K_squared,
                           // solved once here so every caller uses the same number instead of
                           // each re-deriving it (and each independently reserving budget for
                           // the row-band cache above) from scratch.
};

// Build the chunked reader (+ m_snps, read the same way read_grm_binary()
// reads it) for RemlCtx::grm_tile_reader. analysis_ids must be in the exact
// row/column order ctx.y/ctx.X/ctx.A would be built in — same requirement
// as load_pca_warm_start's alignment in MLMA_stream.cpp. Fails loudly (not
// a fallback) on any individual missing from the GRM: a silent misalignment
// here corrupts every downstream REML result without any obvious symptom.
//
// budget_gb/k_ext/feature_flag drive the SAME chunk-sizing math previously
// duplicated (and, in two of three copies, silently stale) across
// pca_stream.cpp, MLMA_stream.cpp, and RemlEngine.cpp's
// setup_chunked_grm_stream: the reader's own row-band cache reservation and
// the resulting chunk_rows are now solved once, here, and returned on the
// handle. feature_flag only affects wording in the error/log messages
// below (which CLI flag to blame) — it does not change the arithmetic, so
// callers whose downstream feature isn't known yet at construction time can
// pass the generic default and the chunk_rows value is still correct for
// whichever feature ends up consuming it, as long as that feature also
// uses k_ext=0 (true of every current caller of this default).
inline ChunkedGrmHandle make_chunked_grm_reader(
    const std::string& prefix,
    const std::vector<std::string>& analysis_ids,
    double budget_gb,
    int k_ext = 0,
    const std::string& feature_flag = "--grm-chunked-budget")
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
                ") -> " + std::to_string(chunk_rows) + "-row chunks from [" + prefix +
                ".grm.bin], not loaded densely.");

    ChunkedGrmHandle handle;
    handle.chunk_rows = chunk_rows;
    handle.m_snps = read_grm_N_mean(prefix, n_grm);
    auto file = std::make_shared<ChunkedGrmReader>(prefix + ".grm.bin", std::move(kp), n_grm,
                                                    static_cast<size_t>(reader_reserved_gb * 1e9));
    handle.file = file;
    handle.reader = [file](int rs, int re, int cs, int ce) -> Eigen::Ref<const Eigen::MatrixXd> {
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
                    " MB total buffer memory). Set `--merge-grms <GB>` to customize.");
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
        // read_exact throws (via LOGGER.e) on failure; letting that escape a
        // parallel region is undefined behaviour (likely std::terminate()
        // instead of the intended error message). Catch per-thread, defer
        // to a single LOGGER.e() call after the region ends, same pattern
        // as read_grm_N_mean's parallel diagonal read elsewhere in this file.
        std::string io_error;
        bool io_failed = false;
        #pragma omp parallel for schedule(dynamic, 1) num_threads(max_io_threads)
        for (int f = 0; f < K; ++f) {
            try {
                read_exact(val_files[f].fd, val_block_bufs[f].data(),
                        bytes_this_chunk, val_files[f].path);
                read_exact(n_files[f].fd, n_block_bufs[f].data(),
                        bytes_this_chunk, n_files[f].path);
            } catch (const std::exception& e) {
                #pragma omp critical
                {
                    if (!io_failed) { io_failed = true; io_error = e.what(); }
                }
            }
        }
        if (io_failed)
            LOGGER.e(0, "merge_grms: " + io_error);

        // Compute the N-weighted average across all elements in this chunk.
        // Tiled in L1-cache-sized blocks (4096 floats = 32KB per stack tile)
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