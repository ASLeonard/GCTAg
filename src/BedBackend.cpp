/*
   BedBackend — double-buffered native PLINK1 .bed reader.

   Replaces the PgenReader-based BED path:
     * Direct POSIX pread/preadv of raw variant records into a 64-byte aligned
       contiguous block buffer (stride == recBytes_, zero per-record padding,
       zero user-space copies): contiguous ascending marker spans coalesce into
       a single ::pread() syscall; non-contiguous extract sets bridge gaps up
       to 2 MiB into the same preadv() via a lazily allocated discard buffer.
     * No PgrPlink1ToPlink2InplaceUnsafe pass and no fread_buf -> block.buf
       copy on the I/O thread: Geno decodes raw PLINK1 2-bit codes directly
       via bed_simd (AVX-512 / AVX2 / NEON / scalar, all using unaligned loads).
     * Validates the 3-byte magic (0x6c 0x1b 0x01) and the exact file size
       (3 + ceil(N/4) * M) of every --bfile / --mbfile member up front.
   PGEN / BGEN keep using third_party/Pgenlib via their own backends.
*/

#include "GenoBackendFactory.h"
#include "Geno.h"                               // full definition for friend access
#include "Logger.h"
#include "mem.hpp"
#include "bed_simd.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

static_assert(sizeof(uintptr_t) == sizeof(uint64_t), "BED kernels assume 64-bit words");
static_assert(sizeof(off_t) >= 8, "BedBackend requires 64-bit file offsets");

namespace {

#if defined(IOV_MAX)
constexpr int kMaxIov = (IOV_MAX < 1024) ? IOV_MAX : 1024;
#else
constexpr int kMaxIov = 1024;
#endif

// Cap each preadv() batch well below INT_MAX (macOS EINVAL) and 0x7ffff000
// (Linux readv/preadv transfer limit).
constexpr uint64_t kMaxBatchBytes = 1ull << 30;

// Merge byte gaps up to 2 MiB into one preadv() call by reading the gap into
// a lazily allocated discard buffer: keeps sequential readahead alive on
// Lustre/GPFS even for sparse --extract subsets without per-record caps.
constexpr uint64_t kMaxGapBytes = 2ull << 20;

std::string errno_str() { return std::string(std::strerror(errno)); }

// Move-only RAII wrapper for a POSIX file descriptor so validation errors or
// mid-stream exceptions never leak fds and --mbfile never holds more than one
// open fd at a time.
class UniqueFd {
public:
    UniqueFd() = default;
    explicit UniqueFd(int fd) : fd_(fd) {}
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
    int get() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }
    void reset(int new_fd = -1) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = new_fd;
    }

private:
    int fd_ = -1;
};

UniqueFd open_bed_fd(const std::string& fname) {
    const int fd = ::open(fname.c_str(), O_RDONLY
#ifdef O_CLOEXEC
                                             | O_CLOEXEC
#endif
    );
    if (fd < 0) {
        LOGGER.e(0, "BedBackend: can't open [" + fname + "]: " + errno_str());
    }
    return UniqueFd(fd);
}

// pread() the full [off, off + len) range, retrying on EINTR / short reads.
void pread_full(int fd, void* dst, size_t len, uint64_t off, const std::string& fname) {
    auto* p = static_cast<unsigned char*>(dst);
    while (len > 0) {
        const ssize_t r = ::pread(fd, p, len, static_cast<off_t>(off));
        if (r < 0) {
            if (errno == EINTR) continue;
            LOGGER.e(0, "BedBackend: read error on [" + fname + "]: " + errno_str());
        }
        if (r == 0) {
            LOGGER.e(0, "BedBackend: unexpected end of file in [" + fname + "].");
        }
        p += r;
        len -= static_cast<size_t>(r);
        off += static_cast<uint64_t>(r);
    }
}

// Scatter-read a contiguous file range into iov[0..cnt). Uses plain pread()
// when cnt == 1, and on a short preadv() advances across completed iovecs and
// retries preadv() on the remaining slice instead of degrading to per-iovec
// pread() calls.
void preadv_full(int fd, struct iovec* iov, int cnt, uint64_t off, const std::string& fname) {
    if (cnt <= 0) return;
    if (cnt == 1) {
        pread_full(fd, iov[0].iov_base, iov[0].iov_len, off, fname);
        return;
    }
#if defined(__linux__) || defined(__APPLE__)
    int idx = 0;
    uint64_t pos = off;
    while (idx < cnt) {
        const int rem_cnt = cnt - idx;
        if (rem_cnt == 1) {
            pread_full(fd, iov[idx].iov_base, iov[idx].iov_len, pos, fname);
            return;
        }
        ssize_t r;
        do {
            r = ::preadv(fd, iov + idx, rem_cnt, static_cast<off_t>(pos));
        } while (r < 0 && errno == EINTR);
        if (r < 0) {
            LOGGER.e(0, "BedBackend: read error on [" + fname + "]: " + errno_str());
        }
        if (r == 0) {
            LOGGER.e(0, "BedBackend: unexpected end of file in [" + fname + "].");
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
        pread_full(fd, iov[i].iov_base, iov[i].iov_len, pos, fname);
        pos += iov[i].iov_len;
    }
#endif
}

// Local equivalent of PgenReader::SetSampleSubsets (1-bit raw-sample mask).
void build_mask_or_die(const std::vector<uint32_t>& idx, uint32_t rawCT, uintptr_t* words, const char* what) {
    if (!bed_simd::build_keep_mask(idx.data(), idx.size(), rawCT, reinterpret_cast<uint64_t*>(words))) {
        LOGGER.e(0, std::string("BedBackend: invalid ") + what +
                        " sample subset (indices must be strictly increasing and < number of samples).");
    }
}

}  // namespace

// --------------------------------------------------------------------------
// BedBackend
// --------------------------------------------------------------------------
class BedBackend : public GenoBackend {
public:
    explicit BedBackend(Geno &geno);
    ~BedBackend() override;

protected:
    std::size_t initialBlockCapacity() const override;
    std::unique_ptr<Producer> makeProducer(std::span<const uint32_t> extractIndex) override;

private:
    class BedProducer;

    Geno &g_;
    uint64_t recBytes_ = 0;

    // Owned 1-bit raw-sample masks (lifetime = this backend). Geno's raw
    // pointers (g_.keepMaskPtr / g_.heterogameticMaskPtr) point into these.
    AlignedUniquePtr<uintptr_t, 64> keepMask_;
    AlignedUniquePtr<uintptr_t, 64> heterogameticMask_;
};

class BedBackend::BedProducer final : public GenoBackend::Producer {
public:
    BedProducer(Geno &geno, uint64_t recBytes,
                std::span<const uint32_t> extractIndex)
        : g_(geno),
          extractIndex_(extractIndex),
          rawMarkerIndex_(g_.marker->get_extract_index()),
          recBytes_(recBytes),
          strideBytes_(recBytes),
          markerBlock_(g_.numMarkerBlock) {
        const std::size_t maxBlock = static_cast<std::size_t>(std::max(1, markerBlock_));
        rawRefScratch_.reserve(maxBlock);
        lagIndexScratch_.reserve(maxBlock);
        iov_.reserve(64);
    }

    bool nextBlock(GenoBlock &block) override {
        const uint32_t numMarker = static_cast<uint32_t>(extractIndex_.size());
        if (finishedMarker_ >= numMarker) {
            return false;
        }

        const uint32_t remain = numMarker - finishedMarker_;
        const uint32_t probeSize = std::min<uint32_t>(
            static_cast<uint32_t>(std::max(1, markerBlock_)), remain);
        rawRefScratch_.resize(probeSize);
        for (uint32_t i = 0; i < probeSize; ++i) {
            rawRefScratch_[i] = rawMarkerIndex_[extractIndex_[finishedMarker_ + i]];
        }

        uint32_t nextSize = g_.marker->getNextSize(
            rawRefScratch_, 0, probeSize, fileIndex_, chrEnds_, sexChromType_);
        if (nextSize == 0) {
            return false;
        }

        block.numMarkers   = nextSize;
        block.sexChromType = sexChromType_;
        block.fileIndex    = fileIndex_;
        const std::size_t requiredBytes =
            static_cast<std::size_t>(nextSize) * static_cast<std::size_t>(recBytes_);
        const std::size_t requiredElems =
            ((requiredBytes + 63) / 64) * (64 / sizeof(uintptr_t));
        if (block.buf.size() < requiredElems) {
            block.buf.resize(requiredElems);
        }
        block.extractIndex = extractIndex_.subspan(finishedMarker_, nextSize);

        const std::string &fname = g_.geno_files[static_cast<std::size_t>(fileIndex_)];
        if (curFileIndex_ != fileIndex_) {
            curFd_.reset();
            curFd_ = open_bed_fd(fname);
            curFileIndex_ = fileIndex_;
        }
        const int fd = curFd_.get();
        const uint64_t nSnpFile = g_.rawCountSNPs[static_cast<std::size_t>(fileIndex_)];
        const int64_t baseIndex = g_.baseIndexLookup[static_cast<std::size_t>(fileIndex_)];

        lagIndexScratch_.resize(nextSize);
        for (uint32_t i = 0; i < nextSize; ++i) {
            const int64_t lag = static_cast<int64_t>(rawRefScratch_[i]) - baseIndex;
            if (lag < 0 || static_cast<uint64_t>(lag) >= nSnpFile) {
                LOGGER.e(0, "BedBackend: variant index out of range for [" + fname + "].");
            }
            lagIndexScratch_[i] = static_cast<uint64_t>(lag);
        }

        auto *base = reinterpret_cast<unsigned char *>(block.buf.data());
        // Coalesce contiguous ascending record spans into a single iovec (issued
        // via plain ::pread() when iov_.size() == 1); bridge small byte gaps up
        // to kMaxGapBytes into the same preadv() via a lazily allocated discard
        // buffer.
        iov_.clear();
        uint64_t batchStart = 0;   // record index of first iovec
        uint64_t nextRec = 0;      // record index following the last iovec
        uint64_t batchBytes = 0;   // total bytes queued in iov_
        auto flush = [&]() {
            if (iov_.empty()) return;
            const uint64_t fileOff = 3ull + batchStart * recBytes_;
            if (iov_.size() == 1) {
                pread_full(fd, iov_[0].iov_base, iov_[0].iov_len, fileOff, fname);
            } else {
                preadv_full(fd, iov_.data(), static_cast<int>(iov_.size()), fileOff, fname);
            }
            iov_.clear();
            batchBytes = 0;
        };
        for (uint32_t i = 0; i < nextSize; ++i) {
            const uint64_t rec = lagIndexScratch_[i];
            unsigned char *const dst = base + static_cast<size_t>(i) * strideBytes_;
            bool startNew = iov_.empty();
            bool bridgedGap = false;
            if (!startNew) {
                if (rec < nextRec) {
                    startNew = true;   // not ascending: new batch
                } else if (rec > nextRec) {
                    const uint64_t gap = rec - nextRec;
                    const uint64_t gapBytes = gap * recBytes_;
                    if (gapBytes <= kMaxGapBytes &&
                        iov_.size() + 2 <= static_cast<size_t>(kMaxIov) &&
                        batchBytes + gapBytes + recBytes_ <= kMaxBatchBytes) {
                        if (discard_.size() < kMaxGapBytes) {
                            discard_.resize(static_cast<size_t>(kMaxGapBytes));
                        }
                        iov_.push_back({discard_.data(), static_cast<size_t>(gapBytes)});
                        batchBytes += gapBytes;
                        nextRec = rec;
                        bridgedGap = true;
                    } else {
                        startNew = true;
                    }
                }
            }
            if (!startNew && !bridgedGap) {
                const bool canCoalesce =
                    !iov_.empty() &&
                    iov_.back().iov_base != discard_.data() &&
                    static_cast<unsigned char *>(iov_.back().iov_base) + iov_.back().iov_len == dst;
                if (batchBytes + recBytes_ > kMaxBatchBytes ||
                    (!canCoalesce && iov_.size() + 1 > static_cast<size_t>(kMaxIov))) {
                    startNew = true;
                }
            }
            if (startNew) {
                flush();
                batchStart = rec;
                nextRec = rec;
            }
            if (!iov_.empty() &&
                iov_.back().iov_base != discard_.data() &&
                static_cast<unsigned char *>(iov_.back().iov_base) + iov_.back().iov_len == dst) {
                iov_.back().iov_len += static_cast<size_t>(recBytes_);
            } else {
                iov_.push_back({dst, static_cast<size_t>(recBytes_)});
            }
            batchBytes += recBytes_;
            nextRec = rec + 1;
        }
        flush();

        finishedMarker_ += nextSize;
        return true;
    }

private:
    Geno &g_;
    std::span<const uint32_t> extractIndex_;
    const std::vector<uint32_t> &rawMarkerIndex_;
    uint64_t recBytes_ = 0;
    uint64_t strideBytes_ = 0;
    int markerBlock_ = 0;

    UniqueFd curFd_;
    int curFileIndex_ = -1;
    AlignedRawBuffer<unsigned char, 64> discard_;
    std::vector<struct iovec> iov_;
    std::vector<uint32_t> rawRefScratch_;
    std::vector<uint64_t> lagIndexScratch_;
    uint32_t finishedMarker_ = 0;
    int fileIndex_ = 0;
    bool chrEnds_ = false;
    uint8_t sexChromType_ = 0;
};

// --------------------------------------------------------------------------
BedBackend::BedBackend(Geno &geno) : g_(geno)
{
    g_.hasInfo = false;

    g_.compressFormats.clear();
    g_.rawCountSamples.clear();
    g_.rawCountSNPs.clear();

    const uint32_t rawCT = g_.rawSampleCT;
    recBytes_ = bed_simd::record_bytes(rawCT);

    for (std::size_t i = 0; i < g_.geno_files.size(); ++i) {
        MarkerParam p = g_.marker->getMarkerParams(static_cast<int>(i));
        g_.compressFormats.push_back(p.compressFormat);
        g_.rawCountSamples.push_back(rawCT);
        g_.rawCountSNPs.push_back(p.rawCountSNP);

        // Open, validate magic + exact size, and close immediately via RAII so
        // --mbfile can hold more files than RLIMIT_NOFILE without leaking fds.
        const std::string &fname = g_.geno_files[i];
        UniqueFd fd = open_bed_fd(fname);

        unsigned char magic[3] = {0, 0, 0};
        pread_full(fd.get(), magic, 3, 0, fname);
        if (magic[0] != 0x6c || magic[1] != 0x1b) {
            LOGGER.e(0, "[" + fname + "] is not a PLINK 1 binary genotype (.bed) file (bad magic number).");
        }
        if (magic[2] != 0x01) {
            LOGGER.e(0, "[" + fname + "] is in individual-major mode; only variant-major (SNP-major) .bed files are supported.");
        }
        struct stat st;
        if (::fstat(fd.get(), &st) != 0) {
            LOGGER.e(0, "BedBackend: can't stat [" + fname + "]: " + errno_str());
        }
        const uint64_t expected = 3ull + recBytes_ * static_cast<uint64_t>(p.rawCountSNP);
        if (st.st_size < 0 || static_cast<uint64_t>(st.st_size) != expected) {
            LOGGER.e(0, "Unexpected PLINK 1 .bed file size for [" + fname + "] (expected " +
                            std::to_string(expected) + " bytes for " + std::to_string(rawCT) +
                            " samples x " + std::to_string(p.rawCountSNP) + " variants, found " +
                            std::to_string(static_cast<long long>(st.st_size)) + ").");
        }
    }

    // Contiguous unpadded record stride (all bed_simd kernels use unaligned loads).
    g_.bedRecordBytes = recBytes_;
    g_.maskPtrSize    = static_cast<int>(bed_simd::keep_mask_words(rawCT));

    const std::size_t nMask = static_cast<std::size_t>(g_.maskPtrSize);
    keepMask_          = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    heterogameticMask_ = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    if (!keepMask_ || !heterogameticMask_) {
        LOGGER.e(0, "BedBackend: can't allocate mask buffers.");
    }
    g_.keepMaskPtr          = keepMask_.get();
    g_.heterogameticMaskPtr = heterogameticMask_.get();

    build_mask_or_die(g_.sampleKeepIndex, rawCT, g_.keepMaskPtr, "--keep/--remove");
    build_mask_or_die(g_.keepHeterogameticIndex, rawCT, g_.heterogameticMaskPtr, "heterogametic");
}

// --------------------------------------------------------------------------
BedBackend::~BedBackend()
{
    // AlignedUniquePtr members free the masks; null Geno's raw pointers so any
    // accidental post-destruction use fails fast.
    g_.keepMaskPtr          = nullptr;
    g_.heterogameticMaskPtr = nullptr;
}

std::size_t BedBackend::initialBlockCapacity() const {
    const std::size_t maxBlockMarkers =
        static_cast<std::size_t>(std::max(1, g_.numMarkerBlock));
    const std::size_t totalBytes = static_cast<std::size_t>(recBytes_) * maxBlockMarkers;
    return ((totalBytes + 63) / 64) * (64 / sizeof(uintptr_t));
}

std::unique_ptr<GenoBackend::Producer> BedBackend::makeProducer(std::span<const uint32_t> extractIndex) {
    return std::make_unique<BedProducer>(g_, recBytes_, extractIndex);
}

// --------------------------------------------------------------------------
// Factory
// --------------------------------------------------------------------------
std::unique_ptr<GenoBackend> makeBedBackend(Geno &geno)
{
    return std::make_unique<BedBackend>(geno);
}
