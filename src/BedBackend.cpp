/*
   BedBackend — double-buffered genotype I/O for PLINK BED files.

   Replaces:
     preGenoDouble_bed()   → constructor
     readGeno_bed()        → BedProducer::nextBlock()
     endGenoDouble_bed()   → destructor (RAII AlignedUniquePtr masks)
*/

#include "GenoBackendFactory.h"
#include "Geno.h"                               // full definition for friend access
#include "Logger.h"
#include "mem.hpp"
#include "third_party/Pgenlib/PgenReader.h"

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

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

    // Owned mask buffers (lifetime = this backend).
    // The raw pointers in Geno (g_.keepMaskPtr etc.) point into these.
    AlignedUniquePtr<uintptr_t, 64> keepMask_;
    AlignedUniquePtr<uintptr_t, 64> keepMaskInter_;
    AlignedUniquePtr<uintptr_t, 64> sexMask_;
    AlignedUniquePtr<uintptr_t, 64> sexMaskInter_;
    AlignedUniquePtr<uintptr_t, 64> heterogameticMask_;
    AlignedUniquePtr<uintptr_t, 64> heterogameticMaskInter_;
};

class BedBackend::BedProducer final : public GenoBackend::Producer {
public:
    BedProducer(Geno &geno, std::span<const uint32_t> extractIndex)
        : g_(geno),
          extractIndex_(extractIndex),
          rawMarkerIndex_(g_.marker->get_extract_index()),
          stride_(g_.bedRawGenoBuf1PtrSize),
          markerBlock_(g_.numMarkerBlock) {
        const std::size_t maxBlock = static_cast<std::size_t>(std::max(1, markerBlock_));
        rawRefScratch_.reserve(maxBlock);
        lagIndexScratch_.reserve(maxBlock);
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
        const std::size_t requiredElems = static_cast<std::size_t>(stride_) * nextSize;
        if (block.buf.size() < requiredElems) {
            block.buf.resize(requiredElems);
        }
        block.extractIndex = extractIndex_.subspan(finishedMarker_, nextSize);

        if (preFileIndex_ != fileIndex_) {
            reader_.Load(g_.geno_files[fileIndex_],
                         &g_.rawCountSamples[fileIndex_],
                         &g_.rawCountSNPs[fileIndex_],
                         g_.sampleKeepIndex);
            preFileIndex_ = fileIndex_;
        }

        lagIndexScratch_.resize(nextSize);
        const int baseIndex = g_.baseIndexLookup[fileIndex_];
        bool contiguousLag = (nextSize > 0);
        for (uint32_t i = 0; i < nextSize; ++i) {
            lagIndexScratch_[i] = static_cast<int>(rawRefScratch_[i]) - baseIndex;
            if (i > 0 && contiguousLag) {
                contiguousLag = (lagIndexScratch_[i] == lagIndexScratch_[i - 1] + 1);
            }
        }
        if (contiguousLag) {
            reader_.ReadRawFullHardRange(
                block.buf.data(), lagIndexScratch_.front(), static_cast<int>(nextSize), stride_);
        } else {
            reader_.ReadRawFullHardBatch(
                block.buf.data(), lagIndexScratch_.data(), static_cast<int>(nextSize), stride_);
        }

        finishedMarker_ += nextSize;
        return true;
    }

private:
    Geno &g_;
    std::span<const uint32_t> extractIndex_;
    const std::vector<uint32_t> &rawMarkerIndex_;
    int stride_ = 0;
    int markerBlock_ = 0;

    std::vector<uint32_t> rawRefScratch_;
    std::vector<int> lagIndexScratch_;
    uint32_t finishedMarker_ = 0;
    int preFileIndex_ = -1;
    int fileIndex_ = 0;
    bool chrEnds_ = false;
    uint8_t sexChromType_ = 0;
    PgenReader reader_;
};

// --------------------------------------------------------------------------
BedBackend::BedBackend(Geno &geno) : g_(geno)
{
    // ── Mirrors preGenoDouble_bed() ──────────────────────────────────────
    g_.hasInfo        = false;

    g_.compressFormats.clear();
    g_.rawCountSamples.clear();
    g_.rawCountSNPs.clear();

    for (std::size_t i = 0; i < g_.geno_files.size(); ++i) {
        MarkerParam p = g_.marker->getMarkerParams(static_cast<int>(i));
        g_.compressFormats.push_back(p.compressFormat);
        g_.rawCountSamples.push_back(g_.rawSampleCT);
        g_.rawCountSNPs.push_back(p.rawCountSNP);
    }

    uint32_t rawCT    = g_.rawSampleCT;
    g_.bedRawGenoBuf1PtrSize = PgenReader::GetGenoBufPtrSize(rawCT);
    g_.maskPtrSize           = static_cast<int>(PgenReader::GetSubsetMaskSize(rawCT));

    std::size_t nMask = static_cast<std::size_t>(g_.maskPtrSize);
    keepMask_               = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    keepMaskInter_          = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    sexMask_                = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    sexMaskInter_           = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    heterogameticMask_      = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    heterogameticMaskInter_ = make_aligned_uninitialized<uintptr_t, 64>(nMask);

    if (!keepMask_ || !keepMaskInter_ ||
        !sexMask_  || !sexMaskInter_  ||
        !heterogameticMask_ || !heterogameticMaskInter_) {
        LOGGER.e(0, "BedBackend: can't allocate mask buffers.");
    }

    // Expose raw pointers so that Geno's getGenoDouble_bed can use them.
    g_.keepMaskPtr      = keepMask_.get();
    g_.keepMaskInterPtr = keepMaskInter_.get();
    g_.sexMaskPtr       = sexMask_.get();
    g_.sexMaskInterPtr  = sexMaskInter_.get();
    g_.heterogameticMaskPtr      = heterogameticMask_.get();
    g_.heterogameticMaskInterPtr = heterogameticMaskInter_.get();

    PgenReader::SetSampleSubsets(g_.sampleKeepIndex, rawCT,
                                 g_.keepMaskPtr, g_.keepMaskInterPtr);
    PgenReader::SetSampleSubsets(g_.keepSexIndex,  rawCT,
                                 g_.sexMaskPtr,  g_.sexMaskInterPtr);
    PgenReader::SetSampleSubsets(g_.keepHeterogameticIndex, rawCT,
                                 g_.heterogameticMaskPtr, g_.heterogameticMaskInterPtr);
}

// --------------------------------------------------------------------------
BedBackend::~BedBackend()
{
    // ── Mirrors endGenoDouble_bed() ──────────────────────────────────────
    // AlignedUniquePtr members free the memory; null out Geno's raw pointers
    // so that any accidental post-destruction use fails fast.
    g_.keepMaskPtr      = nullptr;
    g_.keepMaskInterPtr = nullptr;
    g_.sexMaskPtr       = nullptr;
    g_.sexMaskInterPtr  = nullptr;
    g_.heterogameticMaskPtr      = nullptr;
    g_.heterogameticMaskInterPtr = nullptr;
}

std::size_t BedBackend::initialBlockCapacity() const {
    const std::size_t maxBlockMarkers =
        static_cast<std::size_t>(std::max(1, g_.numMarkerBlock));
    return static_cast<std::size_t>(g_.bedRawGenoBuf1PtrSize) * maxBlockMarkers;
}

std::unique_ptr<GenoBackend::Producer> BedBackend::makeProducer(std::span<const uint32_t> extractIndex) {
    return std::make_unique<BedProducer>(g_, extractIndex);
}

// --------------------------------------------------------------------------
// Factory
// --------------------------------------------------------------------------
std::unique_ptr<GenoBackend> makeBedBackend(Geno &geno)
{
    return std::make_unique<BedBackend>(geno);
}
