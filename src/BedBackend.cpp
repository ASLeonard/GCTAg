/*
   BedBackend — double-buffered native PLINK1 .bed reader.

   Replaces the PgenReader-based BED path:
     * Direct POSIX pread/preadv of raw variant records into a 64-byte aligned
       contiguous block buffer (stride == recBytes_, zero per-record padding,
       zero user-space copies) via shared gcta_io::SpanBatchCoalescer:
       contiguous ascending marker spans coalesce into a single ::pread()
       syscall; non-contiguous extract sets bridge gaps up to 2 MiB into the
       same preadv() via a lazily allocated discard buffer.
     * No PgrPlink1ToPlink2InplaceUnsafe pass and no fread_buf -> block.buf
       copy on the I/O thread: Geno decodes raw PLINK1 2-bit codes directly
       via bed_simd (AVX-512 / AVX2 / NEON / scalar, all using unaligned loads).
     * Validates the 3-byte magic (0x6c 0x1b 0x01) and the exact file size
       (3 + ceil(N/4) * M) of every --bfile / --mbfile member up front.
*/

#include "GenoBackendFactory.h"
#include "Geno.h"                               // full definition for friend access
#include "Logger.h"
#include "mem.hpp"
#include "bed_simd.hpp"
#include "posix_span_io.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <vector>

static_assert(sizeof(uintptr_t) == sizeof(uint64_t), "BED kernels assume 64-bit words");

namespace {

// Local equivalent of PgenReader::SetSampleSubsets (1-bit raw-sample mask).
void bed_build_mask_or_die(const std::vector<uint32_t>& idx, uint32_t rawCT, uintptr_t* words, const char* what) {
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
        spans_.reserve(maxBlock);
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
            try {
                curFd_ = gcta_io::open_posix_readonly_throwing(fname);
            } catch (const std::exception &e) {
                LOGGER.e(0, std::string("BedBackend: ") + e.what());
            }
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

        auto *base = reinterpret_cast<uint8_t *>(block.buf.data());
        spans_.resize(nextSize);
        for (uint32_t i = 0; i < nextSize; ++i) {
            const uint64_t rec = lagIndexScratch_[i];
            spans_[i] = gcta_io::FileByteSpan{
                3ULL + rec * recBytes_,
                static_cast<std::size_t>(recBytes_),
                base + static_cast<std::size_t>(i) * static_cast<std::size_t>(strideBytes_)};
        }
        try {
            coalescer_.read_spans(fd, fname, spans_);
        } catch (const std::exception &e) {
            LOGGER.e(0, std::string("BedBackend: ") + e.what());
        }

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

    gcta_io::PosixUniqueFd curFd_;
    int curFileIndex_ = -1;
    gcta_io::SpanBatchCoalescer coalescer_;
    std::vector<gcta_io::FileByteSpan> spans_;
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
        gcta_io::PosixUniqueFd fd;
        uint64_t fileSize = 0;
        unsigned char magic[3] = {0, 0, 0};
        try {
            fd = gcta_io::open_posix_readonly_throwing(fname);
            fileSize = gcta_io::stat_file_size_throwing(fd.get(), fname);
            if (fileSize < 3ULL) {
                const uint64_t expected = 3ull + recBytes_ * static_cast<uint64_t>(p.rawCountSNP);
                LOGGER.e(0, "Unexpected PLINK 1 .bed file size for [" + fname + "] (expected " +
                                std::to_string(expected) + " bytes for " + std::to_string(rawCT) +
                                " samples x " + std::to_string(p.rawCountSNP) + " variants, found " +
                                std::to_string(fileSize) + ").");
            }
            gcta_io::pread_full_throwing(fd.get(), magic, 3, 0, fname);
        } catch (const std::exception &e) {
            LOGGER.e(0, std::string("BedBackend: ") + e.what());
        }
        if (magic[0] != 0x6c || magic[1] != 0x1b) {
            LOGGER.e(0, "[" + fname + "] is not a PLINK 1 binary genotype (.bed) file (bad magic number).");
        }
        if (magic[2] != 0x01) {
            LOGGER.e(0, "[" + fname + "] is in individual-major mode; only variant-major (SNP-major) .bed files are supported.");
        }
        const uint64_t expected = 3ull + recBytes_ * static_cast<uint64_t>(p.rawCountSNP);
        if (fileSize != expected) {
            LOGGER.e(0, "Unexpected PLINK 1 .bed file size for [" + fname + "] (expected " +
                            std::to_string(expected) + " bytes for " + std::to_string(rawCT) +
                            " samples x " + std::to_string(p.rawCountSNP) + " variants, found " +
                            std::to_string(fileSize) + ").");
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

    bed_build_mask_or_die(g_.sampleKeepIndex, rawCT, g_.keepMaskPtr, "--keep/--remove");
    bed_build_mask_or_die(g_.keepHeterogameticIndex, rawCT, g_.heterogameticMaskPtr, "heterogametic");
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
