/*
   PgenBackend — double-buffered genotype I/O for PLINK2 PGEN files.

   Two paths selected by Geno::isNativeHardcall():
     1. NativePgenHardcallBackend (biallelic hardcalls, including phased hardcalls):
        - Streams raw compressed .pgen records + external ldbase anchors into
          compact double-buffered blocks via pgen_hardcall::CompressedBatchReader
          (holding at most one open POSIX fd at a time for --mpfile safety).
        - Shares the exact 1-bit raw-sample masks and bed_simd decode pipeline
          of BedBackend (getGenoDouble_bed, getGenoFloat_bed, getGenoInt8_bed,
          decodeBedI8KMajor), decoding compressed records on demand in L2-resident
          per-thread PgenThreadCache buffers.
     2. PgenBackend (dosage, multiallelic, or GCTA_DISABLE_NATIVE_PGEN=1 fallback):
        - Streams via PgenReader::ReadDosage and Geno::getGenoDouble_pgen.
*/

#include "GenoBackendFactory.h"
#include "Geno.h"
#include "Logger.h"
#include "bed_simd.hpp"
#include "pgen_hardcall.hpp"
#include "posix_span_io.hpp"
#include "third_party/Pgenlib/PgenReader.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

// Globally monotonic sequence counter across all NativePgenHardcallBackend
// streams so thread_local PgenThreadCache instances never observe colliding
// block_seq_id values across multi-pass loopDouble runs (e.g. filterMAF + MLMA).
std::atomic<uint64_t> g_pgenBlockSeqCounter{1};

void pgen_build_mask_or_die(const std::vector<uint32_t>& idx, uint32_t rawCT, uintptr_t* words, const char* what) {
    if (!bed_simd::build_keep_mask(idx.data(), idx.size(), rawCT, reinterpret_cast<uint64_t*>(words))) {
        LOGGER.e(0, std::string("PgenBackend: invalid ") + what +
                        " sample subset (indices must be strictly increasing and < number of samples).");
    }
}

}  // namespace

// --------------------------------------------------------------------------
// NativePgenHardcallBackend — compressed PGEN -> L2 thread cache -> bed_simd
// --------------------------------------------------------------------------
class NativePgenHardcallBackend final : public GenoBackend {
public:
    explicit NativePgenHardcallBackend(Geno &geno);
    ~NativePgenHardcallBackend() override;

protected:
    std::size_t initialBlockCapacity() const override;
    std::unique_ptr<Producer> makeProducer(std::span<const uint32_t> extractIndex) override;

private:
    class NativePgenProducer;

    Geno &g_;
    uint64_t recBytes_ = 0;
    double maxAvgRecBytes_ = 0.0;

    AlignedUniquePtr<uintptr_t, 64> keepMask_;
    AlignedUniquePtr<uintptr_t, 64> heterogameticMask_;

    Geno::GetGenoDoubleFunc prevGetGenoDoubleFunc_ = nullptr;
    Geno::GetGenoFloatFunc  prevGetGenoFloatFunc_  = nullptr;
};

class NativePgenHardcallBackend::NativePgenProducer final : public GenoBackend::Producer {
public:
    NativePgenProducer(Geno &geno, std::span<const uint32_t> extractIndex)
        : g_(geno),
          extractIndex_(extractIndex),
          rawMarkerIndex_(g_.marker->get_extract_index()),
          markerBlock_(g_.numMarkerBlock) {
        const std::size_t maxBlock = static_cast<std::size_t>(std::max(1, markerBlock_));
        rawRefScratch_.reserve(maxBlock);
        localVidxScratch_.reserve(maxBlock);
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
        block.extractIndex = extractIndex_.subspan(finishedMarker_, nextSize);

        const std::size_t fIdx = static_cast<std::size_t>(fileIndex_);
        const std::string &fname = g_.geno_files[fIdx];
        if (curFileIndex_ != fileIndex_) {
            curFd_.reset();
            try {
                curFd_ = gcta_io::open_posix_readonly_throwing(fname);
            } catch (const std::exception &e) {
                LOGGER.e(0, std::string("PgenBackend: ") + e.what());
            }
            curFileIndex_ = fileIndex_;
        }

        const uint64_t nSnpFile = g_.rawCountSNPs[fIdx];
        const int64_t baseIndex = g_.baseIndexLookup[fIdx];
        localVidxScratch_.resize(nextSize);
        for (uint32_t i = 0; i < nextSize; ++i) {
            const int64_t lag = static_cast<int64_t>(rawRefScratch_[i]) - baseIndex;
            if (lag < 0 || static_cast<uint64_t>(lag) >= nSnpFile) {
                LOGGER.e(0, "PgenBackend: variant index out of range for [" + fname + "].");
            }
            localVidxScratch_[i] = static_cast<uint32_t>(lag);
        }

        const uint64_t seqId = g_pgenBlockSeqCounter.fetch_add(1, std::memory_order_relaxed);
        try {
            batchReader_.read_batch_uptr(
                curFd_.get(),
                g_.pgenIndices[fIdx],
                std::span<const uint32_t>(localVidxScratch_.data(), nextSize),
                seqId,
                block.buf);
        } catch (const std::exception &e) {
            LOGGER.e(0, std::string("PgenBackend: ") + e.what());
        }

        finishedMarker_ += nextSize;
        return true;
    }

private:
    Geno &g_;
    std::span<const uint32_t> extractIndex_;
    const std::vector<uint32_t> &rawMarkerIndex_;
    int markerBlock_ = 0;

    gcta_io::PosixUniqueFd curFd_;
    int curFileIndex_ = -1;
    pgen_hardcall::CompressedBatchReader batchReader_;
    std::vector<uint32_t> rawRefScratch_;
    std::vector<uint32_t> localVidxScratch_;
    uint32_t finishedMarker_ = 0;
    int fileIndex_ = 0;
    bool chrEnds_ = false;
    uint8_t sexChromType_ = 0;
};

NativePgenHardcallBackend::NativePgenHardcallBackend(Geno &geno) : g_(geno)
{
    g_.hasInfo = false;

    g_.compressFormats.clear();
    g_.rawCountSamples.clear();
    g_.rawCountSNPs.clear();

    g_.ensurePgenIndicesLoaded();
    if (g_.pgenIndices.size() != g_.geno_files.size()) {
        LOGGER.e(0, "PgenBackend: internal PGEN file index count mismatch.");
    }

    const uint32_t rawCT = g_.rawSampleCT;
    recBytes_ = bed_simd::record_bytes(rawCT);
    maxAvgRecBytes_ = 0.0;

    for (std::size_t i = 0; i < g_.geno_files.size(); ++i) {
        MarkerParam p = g_.marker->getMarkerParams(static_cast<int>(i));
        const auto &idx = g_.pgenIndices[i];
        if (idx.raw_sample_ct != rawCT) {
            LOGGER.e(0, "PgenBackend: sample count mismatch in [" + g_.geno_files[i] +
                            "] (expected " + std::to_string(rawCT) +
                            ", found " + std::to_string(idx.raw_sample_ct) + ").");
        }
        if (idx.raw_variant_ct != p.rawCountSNP) {
            LOGGER.e(0, "PgenBackend: variant count mismatch in [" + g_.geno_files[i] +
                            "] (expected " + std::to_string(p.rawCountSNP) +
                            ", found " + std::to_string(idx.raw_variant_ct) + ").");
        }
        g_.compressFormats.push_back(p.compressFormat);
        g_.rawCountSamples.push_back(rawCT);
        g_.rawCountSNPs.push_back(p.rawCountSNP);
        maxAvgRecBytes_ = std::max(maxAvgRecBytes_, idx.avg_rec_bytes);
    }

    g_.bedRecordBytes = recBytes_;
    g_.maskPtrSize    = static_cast<int>(bed_simd::keep_mask_words(rawCT));

    const std::size_t nMask = static_cast<std::size_t>(g_.maskPtrSize);
    keepMask_          = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    heterogameticMask_ = make_aligned_uninitialized<uintptr_t, 64>(nMask);
    if (!keepMask_ || !heterogameticMask_) {
        LOGGER.e(0, "PgenBackend: can't allocate mask buffers.");
    }
    g_.keepMaskPtr          = keepMask_.get();
    g_.heterogameticMaskPtr = heterogameticMask_.get();

    pgen_build_mask_or_die(g_.sampleKeepIndex, rawCT, g_.keepMaskPtr, "--keep/--remove");
    pgen_build_mask_or_die(g_.keepHeterogameticIndex, rawCT, g_.heterogameticMaskPtr, "heterogametic");

    prevGetGenoDoubleFunc_ = g_.getGenoDoubleFunc;
    prevGetGenoFloatFunc_  = g_.getGenoFloatFunc;
    g_.pgenNativeHardcall  = true;
    g_.getGenoDoubleFunc   = &Geno::getGenoDouble_bed;
    g_.getGenoFloatFunc    = &Geno::getGenoFloat_bed;
}

NativePgenHardcallBackend::~NativePgenHardcallBackend()
{
    g_.pgenNativeHardcall   = false;
    g_.getGenoDoubleFunc    = prevGetGenoDoubleFunc_;
    g_.getGenoFloatFunc     = prevGetGenoFloatFunc_;
    g_.pgenKMajorUnpackBuf.reset();
    g_.keepMaskPtr          = nullptr;
    g_.heterogameticMaskPtr = nullptr;
}

std::size_t NativePgenHardcallBackend::initialBlockCapacity() const {
    const std::size_t maxBlock =
        static_cast<std::size_t>(std::max(1, g_.numMarkerBlock));
    const std::size_t payloadBytes = static_cast<std::size_t>(
        std::ceil(static_cast<double>(maxBlock) * maxAvgRecBytes_ * 1.25));
    const std::size_t totalBytes =
        sizeof(pgen_hardcall::PgenBlockHeader) +
        maxBlock * sizeof(pgen_hardcall::PgenBlockEntry) +
        payloadBytes;
    return ((totalBytes + 63) / 64) * (64 / sizeof(uintptr_t));
}

std::unique_ptr<GenoBackend::Producer> NativePgenHardcallBackend::makeProducer(std::span<const uint32_t> extractIndex) {
    return std::make_unique<NativePgenProducer>(g_, extractIndex);
}

// --------------------------------------------------------------------------
// PgenBackend — fallback PgenReader dosage / multiallelic backend
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
            try {
                reader_.Load(g_.geno_files[fileIndex_],
                             &g_.rawCountSamples[fileIndex_],
                             &g_.rawCountSNPs[fileIndex_],
                             g_.sampleKeepIndex);
            } catch (const std::exception &e) {
                LOGGER.e(0, std::string("PgenBackend: ") + e.what());
            }
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
            try {
                reader_.ReadDosage(dbuf, lag_index, al_idx);
            } catch (const std::exception &e) {
                LOGGER.e(0, std::string("PgenBackend: ") + e.what());
            }
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
    geno.ensurePgenIndicesLoaded();
    if (geno.isNativeHardcall()) {
        return std::make_unique<NativePgenHardcallBackend>(geno);
    }
    return std::make_unique<PgenBackend>(geno);
}
