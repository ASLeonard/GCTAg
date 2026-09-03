#pragma once

#include <Eigen/Dense>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
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

// Read a GCTA GRM binary file set (prefix.grm.id, prefix.grm.bin, prefix.grm.N.bin).
// Returns:
//   ids      — "FID\tIID" strings in GRM file order
//   G        — full symmetric n×n matrix (double precision)
//   m_snps   — SNP count from element (0,0) of .grm.N.bin; 0 if N file missing
//
// Implementation notes:
//   - .grm.bin is mmap'd with MADV_SEQUENTIAL for OS read-ahead; avoids a
//     userspace heap allocation of up to several GB at large n.
//   - float→double conversion happens inline during the scatter into G,
//     not via a separate full-size staging buffer (that would roughly
//     double this function's transient RSS for no benefit -- see the
//     comment at the scatter loop below).
//   - Only the lower triangle of G is filled in the scatter loop; the upper
//     triangle is then materialised via selfadjointView<Lower> to avoid
//     cache-hostile scattered writes into non-sequential columns.
inline void read_grm_binary(const std::string& prefix,
                             std::vector<std::string>& ids,
                             Eigen::MatrixXd& G,
                             double& m_snps)
{
    using std::to_string;

    ids = Pheno::read_sublist(prefix + ".grm.id");
    const int n = static_cast<int>(ids.size());
    if (n == 0) LOGGER.e(0, "GRM id file [" + prefix + ".grm.id] is empty.");

    const size_t tri      = static_cast<size_t>(n) * (n + 1) / 2;
    const size_t byte_len = tri * sizeof(float);

    // ---- (unchanged) mmap .grm.bin, fill G -----------------------------
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

    void* raw = ::mmap(nullptr, byte_len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (raw == MAP_FAILED)
        LOGGER.e(0, "mmap failed for [" + bin_path + "].");

    ::madvise(raw, byte_len, MADV_SEQUENTIAL | MADV_WILLNEED);
    const float* fbuf = static_cast<const float*>(raw);

    // Cast directly into G during the scatter below rather than through a
    // separate n(n+1)/2-double dbuf staging buffer first. At n=100k that
    // buffer was ~40GB held alongside G's own 80GB (~120GB peak in this
    // function alone) for no real benefit: the scatter loop already read
    // dbuf sequentially (dbuf[idx], ++idx) -- the exact same access pattern
    // it gets reading fbuf directly below -- so dbuf's own "sequential
    // source" property wasn't enabling anything the fused version doesn't
    // already have. This does give up dbuf's dedicated AVX2-vectorized
    // cast pass (the cast is now interleaved with G's write, which the
    // comment below explains can't be made fully sequential given Eigen's
    // column-major layout), but that's a single scalar conversion per
    // element, negligible next to removing 40GB of transient allocation
    // and the write+read traffic of filling and draining it.
    //
    // Row start offsets are computed directly (i*(i+1)/2) rather than via
    // a running idx++ so rows are independent -- at n=150000 this loop is
    // ~1.1e10 scattered (stride-n) double stores; serially that's minutes
    // of wall time on one core's worth of memory bandwidth alone, identical
    // regardless of any Woodbury/Hutch rank chosen downstream, and was
    // silently dominating whole-run wall clock. schedule(dynamic) balances
    // row lengths (row i does i+1 elements), matching GRM.cpp's convention
    // for the same kind of triangular workload.
    G.resize(n, n);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        const size_t row_off = static_cast<size_t>(i) * (i + 1) / 2;
        for (int j = 0; j <= i; ++j)
            G(i, j) = static_cast<double>(fbuf[row_off + j]);
    }
    ::munmap(raw, byte_len);
    // Mirror lower -> upper explicitly (column-contiguous writes, matching
    // the scatter loop's write-side-over-read-side priority above) rather
    // than `G = G.selfadjointView<Eigen::Lower>()`, which is plain
    // single-threaded Eigen and just as serial/O(n^2) as the fill loop was.
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < c; ++r)
            G(r, c) = G(c, r);

    // ------------------------------------------------------------------ //
    // Read SNP count (.grm.N.bin) — same lower-triangle layout as        //
    // .grm.bin. We only need the diagonal (each individual's own         //
    // non-missing SNP count), so extract it without materialising the    //
    // full n×n N matrix.                                                 //
    // ------------------------------------------------------------------ //
    m_snps = 0.0;
    {
        const std::string n_path = prefix + ".grm.N.bin";
        const int nfd = ::open(n_path.c_str(), O_RDONLY);
        if (nfd == -1) {
            LOGGER.w(0, "GRM N file [" + n_path + "] not found; SNP count "
                        "unavailable (affects --reml-woodbury auto-k).");
        } else {
            struct stat st{};
            if (::fstat(nfd, &st) != 0 || static_cast<size_t>(st.st_size) < byte_len) {
                ::close(nfd);
                LOGGER.w(0, "GRM N file [" + n_path + "] has unexpected size; "
                            "SNP count unavailable (affects --reml-woodbury auto-k).");
            } else {
                void* nraw = ::mmap(nullptr, byte_len, PROT_READ, MAP_PRIVATE, nfd, 0);
                ::close(nfd);
                if (nraw == MAP_FAILED) {
                    LOGGER.w(0, "mmap failed for [" + n_path + "]; SNP count unavailable.");
                } else {
                    const float* nbuf = static_cast<const float*>(nraw);
                    // Diagonal entries sit at triangular indices i*(i+1)/2 + i
                    // for row i (0-based, lower-triangle-by-row layout).
                    double sum = 0.0;
                    for (int i = 0; i < n; ++i) {
                        const size_t diag_idx = static_cast<size_t>(i) * (i + 1) / 2 + i;
                        sum += static_cast<double>(nbuf[diag_idx]);
                    }
                    ::munmap(nraw, byte_len);
                    m_snps = sum / n;
                }
            }
        }
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
// (post-kp reindexing) directly from the mmap'd .grm.bin file, without ever
// materializing a dense matrix — not even transiently, and not just the
// n x n analysis-subsetted one, unlike read_grm_binary() above.
//
// Why mmap rather than manual pread(): kp (analysis index -> GRM-file row
// index) is an arbitrary permutation in general — match_ids_to_grm doesn't
// promise anything about ordering. When kp turns out to be monotonic
// (identity, or an order-preserving subset — by far the common case, since
// analysis sample order usually tracks the GRM's own .grm.id order),
// read_tile below exploits that to bulk-read one contiguous run per output
// row and the constructor advises the kernel to read ahead accordingly. A
// genuinely scrambled kp degrades to one mmap touch per scalar entry —
// still correct, but each touch can land on a different page of a
// possibly huge file, which is real, unavoidable, and can dominate runtime;
// the constructor warns loudly in that case.
class ChunkedGrmMmap {
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
    ChunkedGrmMmap(const std::string& path, std::vector<int> kp, int n_grm)
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

        void* raw = ::mmap(nullptr, byte_len_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (raw == MAP_FAILED) {
            ::close(fd_);
            LOGGER.e(0, "mmap failed for [" + path + "].");
        }
        // Identity/order-preserving-subset kp (the common case: analysis
        // sample order tracks the GRM's .grm.id order, possibly with some
        // individuals dropped) makes read_tile's per-row access a single
        // increasing run of file offsets (see read_tile below) — tell the
        // kernel to read ahead for that case. A genuinely scrambled kp
        // (sample order shuffled relative to the GRM) has no such locality;
        // keep the no-readahead hint there so we don't pollute the page
        // cache with pages that won't be reused.
        ::madvise(raw, byte_len_, kp_is_monotonic_ ? (MADV_SEQUENTIAL | MADV_WILLNEED) : MADV_RANDOM);
        fbuf_ = static_cast<const float*>(raw);

        if (!kp_is_monotonic_)
            LOGGER.w(0, "--svd-chunked-budget: the analysis sample order does not match [" +
                        path + "]'s order (individuals were reordered, not just subsetted). "
                        "GRM tile reads degrade to scattered per-entry access in this case and "
                        "can be dramatically slower than the reported chunk count suggests.");
    }

    ~ChunkedGrmMmap() {
        if (fbuf_) ::munmap(const_cast<float*>(fbuf_), byte_len_);
        if (fd_ != -1) ::close(fd_);
    }
    ChunkedGrmMmap(const ChunkedGrmMmap&)            = delete;
    ChunkedGrmMmap& operator=(const ChunkedGrmMmap&) = delete;

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
            throw std::invalid_argument("ChunkedGrmMmap::matvec_blocked: x has wrong length.");

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
    const float* fbuf_ = nullptr;
    bool kp_is_identity_ = false;
    bool kp_is_monotonic_ = false;
};

// .grm.N.bin diagonal only (mean SNP count) — same file, same packed layout,
// same lower-triangle-by-row indexing as .grm.bin, but this touches only n
// scattered diagonal entries via mmap+MADV_RANDOM, same "RSS follows touched
// pages, not file size" argument as ChunkedGrmMmap. Kept as its own function
// rather than factored out of read_grm_binary()'s existing N-file handling
// above, so that function's contract for its current callers doesn't change.
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
    void* nraw = ::mmap(nullptr, byte_len, PROT_READ, MAP_PRIVATE, nfd, 0);
    ::close(nfd);
    if (nraw == MAP_FAILED) {
        LOGGER.w(0, "mmap failed for [" + n_path + "]; SNP count unavailable.");
        return 0.0;
    }
    ::madvise(nraw, byte_len, MADV_RANDOM);
    const float* nbuf = static_cast<const float*>(nraw);
    double sum = 0.0;
    for (int i = 0; i < n_grm; ++i) {
        const size_t diag_idx = static_cast<size_t>(i) * (i + 1) / 2 + i;
        sum += static_cast<double>(nbuf[diag_idx]);
    }
    ::munmap(nraw, byte_len);
    return sum / n_grm;
}

struct ChunkedGrmHandle {
    gcta_chunked::TileReader reader;
    std::shared_ptr<const ChunkedGrmMmap> file;
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
    auto file = std::make_shared<ChunkedGrmMmap>(prefix + ".grm.bin", std::move(kp), n_grm);
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
// Unlike ChunkedGrmMmap (built for arbitrary per-file reindexing via a kp
// map), every file here is walked in the identical, predictable row-major
// order — so MADV_SEQUENTIAL is the right hint (lets the kernel prefetch
// well ahead of us) and there's no per-entry index math needed at all, just
// direct pointer arithmetic into each mapped file. This is also what
// sidesteps the "many small read() syscalls" cost a naive row-by-row
// fread() loop would pay: there are no read() calls here at all, mmap +
// readahead does that work in the background.
//
// row_block_rows bounds the only thing that isn't O(1): the output buffer,
// which holds one row-block's worth of merged values before each write —
// same role as --reml-svd-chunk-size elsewhere, smaller for tighter RSS.
//
// If your GRMs might have different sample orderings, this isn't the right
// tool — use ChunkedGrmMmap-based per-entry lookups instead (each source
// file gets its own kp).
inline void merge_grms_streaming(
    const std::vector<std::string>& prefixes,
    const std::string& out_prefix,
    int row_block_rows = 4000)
{
    if (prefixes.empty())
        LOGGER.e(0, "merge_grms_streaming: no input GRM prefixes given.");
    if (row_block_rows <= 0)
        LOGGER.e(0, "merge_grms_streaming: row_block_rows must be positive.");

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
            LOGGER.e(0, "merge_grms_streaming: [" + prefixes[f] + ".grm.id] does not match "
                        "[" + prefixes[0] + ".grm.id] exactly (same sample order required). "
                        "Use a kp-based merge (ChunkedGrmMmap) for mismatched orderings.");
    }

    const size_t tri      = static_cast<size_t>(n) * (n + 1) / 2;
    const size_t byte_len = tri * sizeof(float);
    const int    K        = static_cast<int>(prefixes.size());

    struct MappedFile {
        int fd = -1;
        const float* buf = nullptr;
    };
    auto open_mapped = [&](const std::string& path) -> MappedFile {
        MappedFile mf;
        mf.fd = ::open(path.c_str(), O_RDONLY);
        if (mf.fd == -1) LOGGER.e(0, "cannot open [" + path + "].");
        struct stat st{};
        if (::fstat(mf.fd, &st) != 0 || static_cast<size_t>(st.st_size) < byte_len) {
            ::close(mf.fd);
            LOGGER.e(0, "unexpected size in [" + path + "].");
        }
        void* raw = ::mmap(nullptr, byte_len, PROT_READ, MAP_PRIVATE, mf.fd, 0);
        if (raw == MAP_FAILED) { ::close(mf.fd); LOGGER.e(0, "mmap failed for [" + path + "]."); }
        ::madvise(raw, byte_len, MADV_SEQUENTIAL | MADV_WILLNEED);
        mf.buf = static_cast<const float*>(raw);
        return mf;
    };

    std::vector<MappedFile> val_files(K), n_files(K);
    for (int f = 0; f < K; ++f) {
        val_files[f] = open_mapped(prefixes[f] + ".grm.bin");
        n_files[f]   = open_mapped(prefixes[f] + ".grm.N.bin");
    }

    const std::string out_bin_path = out_prefix + ".grm.bin";
    const std::string out_n_path   = out_prefix + ".grm.N.bin";
    std::ofstream out_bin(out_bin_path, std::ios::binary);
    std::ofstream out_n(out_n_path, std::ios::binary);
    if (!out_bin) LOGGER.e(0, "cannot open [" + out_bin_path + "] for writing.");
    if (!out_n)   LOGGER.e(0, "cannot open [" + out_n_path + "] for writing.");

    std::vector<float> out_val_buf, out_n_buf;

    for (int rs = 0; rs < n; rs += row_block_rows) {
        const int re = std::min(rs + row_block_rows, n);
        const size_t block_base = static_cast<size_t>(rs) * (rs + 1) / 2;
        size_t block_elems = 0;
        for (int i = rs; i < re; ++i) block_elems += static_cast<size_t>(i + 1);
        out_val_buf.resize(block_elems);
        out_n_buf.resize(block_elems);

        // Rows are independent (each writes its own non-overlapping range
        // of the block buffer, computed from the packed-triangular offset
        // formula), so this parallelizes cleanly — each thread keeps its
        // own reusable n-length scratch rather than reallocating per row.
        #pragma omp parallel
        {
            std::vector<double> wsum(n), wtN(n);
            #pragma omp for schedule(dynamic, 64)
            for (int i = rs; i < re; ++i) {
                const int row_len = i + 1;
                const size_t row_start = static_cast<size_t>(i) * (i + 1) / 2;
                std::fill_n(wsum.data(), row_len, 0.0);
                std::fill_n(wtN.data(), row_len, 0.0);
                for (int f = 0; f < K; ++f) {
                    const float* v  = val_files[f].buf + row_start;
                    const float* nn = n_files[f].buf + row_start;
                    for (int j = 0; j < row_len; ++j) {
                        wsum[j] += static_cast<double>(v[j]) * static_cast<double>(nn[j]);
                        wtN[j]  += static_cast<double>(nn[j]);
                    }
                }
                const size_t out_offset = row_start - block_base;
                for (int j = 0; j < row_len; ++j) {
                    out_val_buf[out_offset + j] = static_cast<float>(wtN[j] > 0.0 ? wsum[j] / wtN[j] : 0.0);
                    out_n_buf[out_offset + j]   = static_cast<float>(wtN[j]);
                }
            }
        }

        out_bin.write(reinterpret_cast<const char*>(out_val_buf.data()), block_elems * sizeof(float));
        out_n.write(reinterpret_cast<const char*>(out_n_buf.data()), block_elems * sizeof(float));
        if (!out_bin || !out_n)
            LOGGER.e(0, "write failed while writing merged GRM to [" + out_prefix + "].");
    }

    out_bin.close();
    out_n.close();

    for (auto& mf : val_files) { ::munmap(const_cast<float*>(mf.buf), byte_len); ::close(mf.fd); }
    for (auto& mf : n_files)   { ::munmap(const_cast<float*>(mf.buf), byte_len); ::close(mf.fd); }

    // .grm.id is identical across inputs (already validated) — copy it once.
    {
        std::ifstream src(prefixes[0] + ".grm.id", std::ios::binary);
        std::ofstream dst(out_prefix + ".grm.id", std::ios::binary);
        dst << src.rdbuf();
    }

    LOGGER.i(0, "Merged " + std::to_string(K) + " GRMs (" + std::to_string(n) +
                " individuals) into [" + out_prefix + ".grm.bin/.grm.N.bin/.grm.id], "
                "N-weighted average, fully streamed — no dense matrix held for any input or the output.");
}

} // namespace gcta_grm_io