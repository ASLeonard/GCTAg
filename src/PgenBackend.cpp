/*
   PgenBackend — double-buffered genotype I/O for PLINK PGEN files.
*/

#include "GenoBackendFactory.h"
#include "Geno.h"
#include "Logger.h"
#include "third_party/Pgenlib/PgenReader.h"

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

// --------------------------------------------------------------------------
class PgenBackend : public GenoBackend {
public:
    explicit PgenBackend(Geno &geno);
    ~PgenBackend() override = default;

protected:
    std::size_t initialBlockCapacity() const override;
    std::unique_ptr<Producer> makeProducer(std::span<const uint32_t> extractIndex) override;

private:
    class PgenProducer;

    Geno &g_;
};

class PgenBackend::PgenProducer final : public GenoBackend::Producer {
public:
    PgenProducer(Geno &geno, std::span<const uint32_t> extractIndex)
        : g_(geno),
          extractIndex_(extractIndex),
          stride_(g_.pgenGenoBuf1PtrSize),
          markerBlock_(g_.numMarkerBlock) {
        const std::vector<uint32_t> &rawMarkerIndex = g_.marker->get_extract_index();
        rawIndices_.resize(extractIndex_.size());
        std::transform(extractIndex_.begin(), extractIndex_.end(),
                       rawIndices_.begin(),
                       [&rawMarkerIndex](std::size_t pos) {
                           return rawMarkerIndex[pos];
                       });
    }

    bool nextBlock(GenoBlock &block) override {
        const uint32_t numMarker = static_cast<uint32_t>(extractIndex_.size());
        if (finishedMarker_ >= numMarker) {
            return false;
        }

        uint32_t nextSize = g_.marker->getNextSize(
            rawIndices_, finishedMarker_, markerBlock_,
            fileIndex_, chrEnds_, sexChromType_);
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

        for (uint32_t i = 0; i < nextSize; ++i) {
            int rawIndex  = static_cast<int>(rawIndices_[finishedMarker_ + i]);
            int lag_index = rawIndex - g_.baseIndexLookup[fileIndex_];
            int al_idx    = 1;
            uintptr_t* slot =
                block.buf.data() + static_cast<std::size_t>(i) * stride_;
            DosageBuf dbuf = DosageBuf::from(
                slot,
                g_.pgenGenoPtrSize,
                g_.pgenDosagePresentPtrSize,
                g_.pgenDosageMainPtrSize);
            reader_.ReadDosage(dbuf, lag_index, al_idx);
            // Legacy getGenoDouble_pgen() still reads dosage_ct from the last
            // uintptr_t of each flat marker slot.
            slot[stride_ - 1] = static_cast<uintptr_t>(dbuf.dosage_ct);
        }

        finishedMarker_ += nextSize;
        return true;
    }

private:
    Geno &g_;
    std::span<const uint32_t> extractIndex_;
    std::vector<uint32_t> rawIndices_;
    int stride_ = 0;
    int markerBlock_ = 0;

    uint32_t finishedMarker_ = 0;
    int preFileIndex_ = -1;
    int fileIndex_ = 0;
    bool chrEnds_ = false;
    uint8_t sexChromType_ = 0;
    PgenReader reader_;
};

// --------------------------------------------------------------------------
PgenBackend::PgenBackend(Geno &geno) : g_(geno)
{
    // ── Mirrors preGenoDouble_pgen() ─────────────────────────────────────
    g_.hasInfo = false;

    g_.compressFormats.clear();
    g_.rawCountSamples.clear();
    g_.rawCountSNPs.clear();

    for (std::size_t i = 0; i < g_.geno_files.size(); ++i) {
        MarkerParam p = g_.marker->getMarkerParams(static_cast<int>(i));
        g_.compressFormats.push_back(p.compressFormat);
        g_.rawCountSamples.push_back(g_.rawSampleCT);
        g_.rawCountSNPs.push_back(p.rawCountSNP);
    }

    // Compute PGEN-specific buffer strides.
    const uint32_t keepCT = g_.keepSampleCT;
    g_.pgenGenoPtrSize =
        static_cast<int>((PgenReader::GetGenoBufPtrSize(keepCT) + 63) / 64 * 64);
    g_.pgenDosageMainPtrSize =
        static_cast<int>((PgenReader::GetDosageMainSize(keepCT) + 63) / 64 * 64);
    g_.pgenDosagePresentPtrSize =
        static_cast<int>((PgenReader::GetDosagePresentSize(keepCT) + 63) / 64 * 64);

    g_.pgenGenoBuf1PtrSize =
        static_cast<int>(
            (static_cast<std::size_t>(g_.pgenGenoPtrSize)
             + g_.pgenDosageMainPtrSize
             + g_.pgenDosagePresentPtrSize + 1 + 63) / 64 * 64);
}

std::size_t PgenBackend::initialBlockCapacity() const {
    const std::size_t maxBlockMarkers =
        static_cast<std::size_t>(std::max(1, g_.numMarkerBlock));
    return static_cast<std::size_t>(g_.pgenGenoBuf1PtrSize) * maxBlockMarkers;
}

std::unique_ptr<GenoBackend::Producer> PgenBackend::makeProducer(std::span<const uint32_t> extractIndex) {
    return std::make_unique<PgenProducer>(g_, extractIndex);
}

// --------------------------------------------------------------------------
std::unique_ptr<GenoBackend> makePgenBackend(Geno &geno)
{
    return std::make_unique<PgenBackend>(geno);
}