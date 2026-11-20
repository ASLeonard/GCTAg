/*
   BgenBackend — double-buffered genotype I/O for Oxford BGEN files.
*/

#include "GenoBackendFactory.h"
#include "Geno.h"
#include "Logger.h"
#include "utils.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

// --------------------------------------------------------------------------
class BgenBackend : public GenoBackend {
public:
    explicit BgenBackend(Geno &geno);
    ~BgenBackend() override = default;

protected:
    std::size_t initialBlockCapacity() const override;
    std::unique_ptr<Producer> makeProducer(std::span<const uint32_t> extractIndex) override;

private:
    class BgenProducer;

    Geno &g_;
    std::vector<UniqueFile> fileHandles_; ///< one per geno_file
};

class BgenBackend::BgenProducer final : public GenoBackend::Producer {
public:
    BgenProducer(Geno &geno,
                 const std::vector<UniqueFile> &fileHandles,
                 std::span<const uint32_t> extractIndex)
        : g_(geno),
          fileHandles_(fileHandles),
          extractIndex_(extractIndex),
          stride_(g_.bgenRawGenoBuf1PtrSize),
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

        FILE *bgenFile = fileHandles_[fileIndex_].get();
        for (uint32_t i = 0; i < nextSize; ++i) {
            int rawIndex = static_cast<int>(rawIndices_[finishedMarker_ + i]);

            uint64_t pos = 0, size = 0;
            g_.marker->getStartPosSize(rawIndex, pos, size);
            std::fseek(bgenFile, static_cast<long>(pos), SEEK_SET);

            uintptr_t *dest = block.buf.data()
                              + static_cast<std::size_t>(i) * stride_;
            if (std::fread(dest, sizeof(char), size, bgenFile) != size) {
                int lag_index = rawIndex - g_.baseIndexLookup[fileIndex_];
                LOGGER.error(0, "can't read {}th SNP in [{}].",
                             lag_index, g_.geno_files[fileIndex_]);
            }
        }

        finishedMarker_ += nextSize;
        return true;
    }

private:
    Geno &g_;
    const std::vector<UniqueFile> &fileHandles_;
    std::span<const uint32_t> extractIndex_;
    std::vector<uint32_t> rawIndices_;
    int stride_ = 0;
    int markerBlock_ = 0;

    uint32_t finishedMarker_ = 0;
    int fileIndex_ = 0;
    bool chrEnds_ = false;
    uint8_t sexChromType_ = 0;
};

// --------------------------------------------------------------------------
BgenBackend::BgenBackend(Geno &geno) : g_(geno)
{
    // ── Mirrors preGenoDouble_bgen() ─────────────────────────────────────
    g_.hasInfo = true;

    g_.compressFormats.clear();
    g_.rawCountSamples.clear();
    g_.rawCountSNPs.clear();

    for (std::size_t i = 0; i < g_.geno_files.size(); ++i) {
        MarkerParam p = g_.marker->getMarkerParams(static_cast<int>(i));
        if (p.rawCountSample != g_.pheno->count_raw()) {
            LOGGER.e(0, "inconsistent sample sizes between the .bgen file ["
                        + g_.geno_files[i]
                        + "] and the .sample file (specified by --sample).");
        }
        g_.compressFormats.push_back(p.compressFormat);
        g_.rawCountSamples.push_back(p.rawCountSample);
        g_.rawCountSNPs.push_back(p.rawCountSNP);
    }

    g_.bgenRawGenoBuf1PtrSize = g_.marker->getMaxGenoMarkerUptrSize();

    // Open file handles.
    fileHandles_.reserve(g_.geno_files.size());
    for (const auto &path : g_.geno_files) {
        UniqueFile f = open_file(path, "rb");
        if (!f) {
            LOGGER.error(0, "BgenBackend: can't open [{}] to read.", path);
        }
        fileHandles_.push_back(std::move(f));
    }
}

std::size_t BgenBackend::initialBlockCapacity() const {
    const std::size_t maxBlockMarkers =
        static_cast<std::size_t>(std::max(1, g_.numMarkerBlock));
    return static_cast<std::size_t>(g_.bgenRawGenoBuf1PtrSize) * maxBlockMarkers;
}

std::unique_ptr<GenoBackend::Producer> BgenBackend::makeProducer(std::span<const uint32_t> extractIndex) {
    return std::make_unique<BgenProducer>(g_, fileHandles_, extractIndex);
}

// --------------------------------------------------------------------------
std::unique_ptr<GenoBackend> makeBgenBackend(Geno &geno)
{
    return std::make_unique<BgenBackend>(geno);
}
