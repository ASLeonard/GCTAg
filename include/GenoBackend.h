/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   Double-buffered genotype I/O backend interface.

   Each backend:
     - runs pre-logic (mask allocation, file opening) in its constructor
     - runs end-logic (mask deallocation, file closing) in its destructor
     - provides a synchronous Producer that fills each GenoBlock on a
       dedicated std::jthread while compute callbacks execute on the
       calling thread (preserving OpenMP/BLAS thread affinity).
*/
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "mem.hpp"

// --------------------------------------------------------------------------
// GenoBlock
//
// One block of raw genotype data filled by a backend Producer.
// The layout of buf[] is format-specific (BED / BGEN / PGEN); callers that
// need to interpret it call Geno::getGenoDouble(buf.data(), i, &item).
// --------------------------------------------------------------------------
struct GenoBlock {
    AlignedRawBuffer<uintptr_t, 64> buf{/*interleave=*/true}; ///< 64-byte aligned uninitialized packed data
    std::span<const uint32_t> extractIndex; ///< marker extract-indices view for this block
    uint32_t               numMarkers   = 0;
    uint8_t                sexChromType = 0; ///< 0=non-sex, 1=homogametic, 2=heterogametic
    int                    fileIndex    = 0; ///< which geno_file[] this block came from
};

// --------------------------------------------------------------------------
// GenoBackend — base class for BED / BGEN / PGEN backends.
//
// Owns a 2-slot double-buffered std::jthread pipeline: stream() runs the
// format Producer on an I/O thread while invoking callback on the caller thread.
// --------------------------------------------------------------------------
class GenoBackend {
public:
    using Callback = std::function<void(GenoBlock &)>;

    GenoBackend()          = default;
    virtual ~GenoBackend() = default;

    GenoBackend(const GenoBackend &)            = delete;
    GenoBackend &operator=(const GenoBackend &) = delete;

    void stream(std::span<const uint32_t> extractIndex, Callback callback);

protected:
    struct Producer {
        virtual ~Producer() = default;
        // Populate `block` with the next batch of markers; return false at EOF.
        virtual bool nextBlock(GenoBlock &block) = 0;
    };

    virtual std::size_t initialBlockCapacity() const { return 0; }
    virtual std::unique_ptr<Producer> makeProducer(std::span<const uint32_t> extractIndex) = 0;

private:
    std::array<GenoBlock, 2> blocks_;
    std::atomic<bool>        streamActive_{false};
};
