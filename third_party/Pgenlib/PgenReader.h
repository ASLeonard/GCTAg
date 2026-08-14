/* A reader library for plink2 PGEN format
 * This code assembled and revised from plink2.
 * This library removes lots of dependencies of plink2 codes, keep clean and minimal
 * The PGEN format seems to be a draft, thus use the library they provided would be a safe way
 * Use plink2 for QC on genotype in BED or PGEN format.
 * The library don't provide any validity function, only support PGEN generated from plink2
 * Limits:  only support bialleric currently, some bugs in multialleric data
 *
 * Assembled and coded by Zhili Zheng <zhilizheng@outlook.com>
 * Bug report to Zhili Zheng
 * Copyright (c) 2019 Zhili Zheng
 * Please refer to plink2 for orginal license statement and authorship
 * https://github.com/chrchang/plink-ng
 *
// This library is free software: you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation; either version 3 of the License, or (at your
// option) any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef PGENREADER
#define PGENREADER
#include "pgenlib_ffi_support.h"
#include "pgenlib_read.h"
#include <stdexcept>
#include <string>
#include <vector>

using std::string;
using std::vector;

struct RefcountedWptrStruct {
  uintptr_t ref_ct;
  uintptr_t p[];
};

typedef struct RefcountedWptrStruct RefcountedWptr;

typedef struct SNPInfo{
    double af;
    double nMissRate;
    double std;
    double mean;
    uint32_t N;
    uint32_t AlCount;
} SNPInfo;

// Re-export plink2's PAIR_TABLE16 under the name used by downstream code (e.g. Geno.cpp).
#define GET_TABLE16(a, b, c, d) PAIR_TABLE16(a, b, c, d)

// Structured view over a flat dosage buffer.
//
// The flat layout (used by PgenBackend's block.buf) is:
//   [genovec:        genoPtrSize   uintptr_t]
//   [dosage_present: presentSize   uintptr_t]
//   [dosage_main:    mainSize      uintptr_t (reinterpreted as uint16_t*)]
//   [dosage_ct:      1             uint32_t  (written by ReadDosage)]
//
// Use DosageBuf::from() to construct a view into a stride-aligned slot of
// such a buffer instead of repeating the pointer arithmetic at each call site.
struct DosageBuf {
    uintptr_t* genovec;
    uintptr_t* dosage_present;
    uint16_t*  dosage_main;
    uint32_t   dosage_ct;      // filled by ReadDosage

    // Construct a DosageBuf view into a flat pre-allocated buffer slot.
    // base        -- pointer to the start of this marker's slot
    // genoPtrSize -- PgenReader::GetGenoBufPtrSize(keepCT),    padded to 64-byte boundary
    // presentSize -- PgenReader::GetDosagePresentSize(keepCT), padded to 64-byte boundary
    // mainSize    -- PgenReader::GetDosageMainSize(keepCT),    padded to 64-byte boundary
    static DosageBuf from(uintptr_t* base,
                          std::size_t genoPtrSize,
                          std::size_t presentSize,
                          std::size_t mainSize) {
        (void)mainSize; // size retained for documentation; dosage_ct sits after it
        return DosageBuf{
            base,
            base + genoPtrSize,
            reinterpret_cast<uint16_t*>(base + genoPtrSize + presentSize),
            0   // dosage_ct written by ReadDosage
        };
    }
};

class PgenReader {
    public:
        PgenReader();

        #if __cplusplus >= 201103L
        PgenReader(const PgenReader&) = delete;
        PgenReader& operator=(const PgenReader&) = delete;
        #endif

        void Load(string filename, uint32_t* raw_sample_ct,
                uint32_t* raw_maker_ct, const vector<uint32_t> &sample_subset_0based);

        uint32_t GetRawSampleCt() const;

        uint32_t GetSubsetSize() const;

        uint32_t GetVariantCt() const;

        uint32_t GetAlleleCt(uint32_t variant_idx) const;

        uint32_t GetMaxAlleleCt() const;

        uintptr_t GetGenoBufSizeUptr() const;

        uintptr_t GetGenoBufFullSizeUptr() const;

        bool HardcallPhasePresent() const;

        bool IsPhasePresent() const;

        bool HasMultiAllelic() const;

        void ReadIntHardcalls(vector<int32_t> &buf, int variant_idx, int allele_idx);

        void ReadHardcalls(vector<double> &buf, int variant_idx, int allele_idx);

        void Read(vector<double> &buf, int variant_idx, int allele_idx);

        void ReadRawHard(uintptr_t *buf, int variant_idx, int allele_idx);

        void ReadRawFullHard(uintptr_t *buf, int variant_idx, int allele_idx);

        void ReadRawFullHard(uintptr_t *buf, int variant_idx);

        // Batch full-hard reads into a strided output buffer.
        // This reuses one subset-index setup for the whole batch to reduce
        // per-variant call overhead in tight streaming loops.
        void ReadRawFullHardBatch(uintptr_t *buf,   
                      const int *variant_indices,
                      int variant_count,
                      int stride_uptr);

        // Fast path for contiguous variant indices [start_variant_idx, start_variant_idx + variant_count).
        void ReadRawFullHardRange(uintptr_t *buf,
                  int start_variant_idx,
                  int variant_count,
                  int stride_uptr);

        void ReadDosage(uintptr_t *buf, int variant_idx, int allele_idx);
        void ReadDosage(DosageBuf &buf, int variant_idx, int allele_idx);

        void CountHardFreqMiss(uintptr_t *buf, SNPInfo *snpinfo);
        static bool CountHardFreqMissExt(uintptr_t *buf, const uintptr_t *subset_iter_vec, uint32_t rawSampleSize, uint32_t keepSize, SNPInfo *snpinfo, bool f_std);
        static bool CountHardFreqMissExtX(uintptr_t *buf, const uintptr_t *subset_iter_vec, const uintptr_t *subset_iter_vec2, 
                uint32_t rawSampleSize, uint32_t keepSize, uint32_t keepSize2, SNPInfo *snpinfo, string &errmsg, bool dosageComp, bool f_std);

        void ExtractGeno(const uintptr_t *in, uintptr_t *out);
        static void ExtractGenoExt(const uintptr_t *in, const uintptr_t * subsets, uint32_t rawSampleSize, uint32_t keepSize, uintptr_t *out);
        static void ExtractDoubleExt(uintptr_t *in, const uintptr_t *subsets, uint32_t rawSampleSize, uint32_t keepSize, const double *gtable, double *gOut, uintptr_t *missOut);
        static void ExtractFloatExt(uintptr_t *in, const uintptr_t *subsets, uint32_t rawSampleSize, uint32_t keepSize, const float *gtable, float *gOut, uintptr_t *missOut);

        static void SetSampleSubsets(const vector<uint32_t> &sample_subset_0based, uint32_t rawSampleSize, uintptr_t *subset_incl_vec, uintptr_t *subset_inter_vec);
        static int GetGenoBufPtrSize(uint32_t sample_ct);
        static int GetSubsetMaskSize(uint32_t sample_ct);
        static int GetDosagePresentSize(uint32_t sample_ct);
        static int GetDosageMainSize(uint32_t sample_ct);

        static bool CountHardDosage(uintptr_t *buf, uint16_t *dosage_buf, const uintptr_t *dosage_present, const vector<uint32_t> *maskp, uint32_t sampleCT, uint32_t dosageCT, SNPInfo *snpinfo, string &err);

        void Close();

        ~PgenReader();

    private:
        plink2::PgenFileInfo* _info_ptr;
        RefcountedWptr* _allele_idx_offsetsp;
        RefcountedWptr* _nonref_flagsp;
        plink2::PgenReader* _state_ptr;
        uintptr_t* _subset_include_vec;
        uintptr_t* _subset_include_interleaved_vec;
        uint32_t* _subset_cumulative_popcounts;
        uint32_t _subset_size;
        uint32_t _raw_sample_ct;
        
        uintptr_t _genovecbuf_uptr_size;
        uintptr_t _genovec_rawbuf_uptr_size;
        
        uintptr_t genovec_size64;
        uintptr_t dosage_present_size64;
        uintptr_t dosage_main_size64;

        plink2::PgenVariant _pgv;

        plink2::VecW* _transpose_batch_buf;
        // Reserved for future multi-sample transpose use; not currently active.

        void SetSampleSubsetInternal(const vector<uint32_t> &sample_subset_0based);

        void ReadAllelesPhasedInternal(int variant_idx);
};
#endif //PGENREADER