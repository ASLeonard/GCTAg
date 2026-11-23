/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   New implementation: SNPs information in plink format.

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GCTA2_MARKER_H
#define GCTA2_MARKER_H
#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <cstdint>
#include "Cli.hpp"
using std::string;
using std::to_string;

// Zero-copy view for arena-backed Marker fields that binds directly to
// std::string_view while remaining implicitly convertible to std::string
// for legacy callers binding const std::string&.
struct MarkerStrView {
    std::string_view sv;
    constexpr operator std::string_view() const noexcept { return sv; }
    operator std::string() const { return std::string(sv); }
    constexpr const char* data() const noexcept { return sv.data(); }
    constexpr std::size_t size() const noexcept { return sv.size(); }
    constexpr bool empty() const noexcept { return sv.empty(); }
};

// Flat character arena storing N strings in two contiguous allocations
// (raw bytes + 64-bit offsets), eliminating per-SNP std::string heap mallocs.
class CompactStringArena {
public:
    CompactStringArena() { offsets_.push_back(0); }

    void reserve(std::size_t count, std::size_t total_bytes) {
        offsets_.reserve(count + 1);
        data_.reserve(total_bytes);
    }

    void push_back(std::string_view sv) {
        data_.insert(data_.end(), sv.data(), sv.data() + sv.size());
        offsets_.push_back(static_cast<uint64_t>(data_.size()));
    }

    void push_back_upper(std::string_view sv) {
        const std::size_t old_sz = data_.size();
        data_.resize(old_sz + sv.size());
        char* dst = data_.data() + old_sz;
        for (std::size_t i = 0; i < sv.size(); ++i) {
            const char c = sv[i];
            dst[i] = (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
        }
        offsets_.push_back(static_cast<uint64_t>(data_.size()));
    }

    std::size_t size() const noexcept { return offsets_.size() - 1; }
    std::size_t bytes() const noexcept { return data_.size(); }
    bool empty() const noexcept { return offsets_.size() <= 1; }

    std::string_view view(std::size_t i) const noexcept {
        const uint64_t beg = offsets_[i];
        return std::string_view(data_.data() + beg, static_cast<std::size_t>(offsets_[i + 1] - beg));
    }

private:
    std::vector<char> data_;
    std::vector<uint64_t> offsets_;
};

// 1-byte-per-SNP allele storage for single-char ASCII alleles (A/C/G/T/...)
// with a sorted sparse overflow arena for multi-character indels/SVs.
class CompactAlleleArena {
public:
    void reserve(std::size_t count) { single_.reserve(count); }

    void push_back_upper(std::string_view sv) {
        if (sv.size() == 1) {
            const unsigned char raw = static_cast<unsigned char>(sv[0]);
            if (raw >= 1 && raw < 128) {
                const char u = (raw >= 'a' && raw <= 'z')
                    ? static_cast<char>(raw - ('a' - 'A'))
                    : static_cast<char>(raw);
                single_.push_back(static_cast<uint8_t>(u));
                return;
            }
        }
        const uint32_t row = static_cast<uint32_t>(single_.size());
        single_.push_back(0);
        overflow_rows_.push_back(row);
        overflow_arena_.push_back_upper(sv);
    }

    std::size_t size() const noexcept { return single_.size(); }

    std::string_view view(std::size_t i) const noexcept {
        const uint8_t c = single_[i];
        if (c != 0) {
            return std::string_view(&kAsciiTable[c], 1);
        }
        const auto it = std::lower_bound(overflow_rows_.begin(), overflow_rows_.end(),
                                         static_cast<uint32_t>(i));
        return overflow_arena_.view(static_cast<std::size_t>(it - overflow_rows_.begin()));
    }

private:
    static constexpr std::array<char, 128> kAsciiTable = []() {
        std::array<char, 128> a{};
        for (int i = 0; i < 128; ++i) a[static_cast<std::size_t>(i)] = static_cast<char>(i);
        return a;
    }();
    std::vector<uint8_t> single_;
    std::vector<uint32_t> overflow_rows_;
    CompactStringArena overflow_arena_;
};

struct MarkerInfo{
    std::string chr;
    std::string name;
    uint32_t pd;
    std::vector<string> alleles;
};

struct MarkerParam{
    uint32_t rawCountSNP;
    uint32_t rawCountSample; //if unknown, 0;
    int compressFormat; // -1, don't know;
    uint64_t posGenoDataStart;
};

class Marker {

public:
    Marker(const gcta_cli::GenoInputOptions& geno,
           const gcta_cli::SampleFilterOptions& sample_filter,
           const gcta_cli::MarkerFilterOptions& marker_filter,
           bool default_autosome = false,
           bool force_chr_homogametic = false);
    uint32_t count_raw(int part = -1);
    uint32_t count_extract();
    bool isInExtract(uint32_t index);
    uint32_t getRawIndex(uint32_t extractedIndex);
    std::vector<uint32_t>& get_extract_index(); // return the raw index for all kept markers
    std::vector<uint32_t> get_extract_index_autosome(); // return the extract index for non-sex markers //not raw index
    std::vector<uint32_t> get_extract_index_X();
    int getMIndex(uint32_t raw_index);
    //uint64_t getStartPosSize(uint32_t raw_index);
    void getStartPosSize(uint32_t raw_index, uint64_t &pos, uint64_t &size);
    std::string get_marker(int rawindex);
    std::string getMarkerStrExtract(int extractindex);
    static MarkerInfo extractBgenMarkerInfo(FILE *h_bgen, uint64_t &pos);
    
    void extract_marker(std::vector<string> markers, bool isExtract);
    void reset_exclude();
    void keep_raw_index(const std::vector<uint32_t>& keep_index);
    void keep_extracted_index(const std::vector<uint32_t>& keep_index);
    void matchSNPListFile(string filename, int num_min_fields, const std::vector<int>& field_return, std::vector<string> &fields);
    void save_marker(string filename);
    std::vector<uint32_t> getNextWindowIndex(uint32_t cur_marker_index, uint32_t window, bool& chr_ends, bool& isHomogameticChrom, bool retRaw = true);
    std::vector<uint32_t> getNextSizeIndex(uint32_t cur_marker_index, uint32_t num, bool& chr_ends, bool& isHomogameticChrom, bool retRaw = false);
    uint32_t getNextSize(const std::vector<uint32_t> &rawRef, uint32_t curExtractIndex, uint32_t num, int &fileIndex, bool &chr_ends, uint8_t &sexChromType);
    uint32_t getNextWindowSize(uint32_t cur_marker_index, uint32_t window);

    MarkerParam getMarkerParams(int part_num);
    uint64_t getMaxGenoMarkerUptrSize();
    std::vector<std::pair<string, std::vector<uint32_t>>> read_gene(string gfile);

   #ifdef BGEN_SUPPORT
    static MarkerParam getBgenMarkerParam(FILE *h_bgen, std::string &outputs);
   #endif

    const std::string& getRawChr(uint32_t i)  const { return chr_dict_[chr_id_[i]]; }
    uint32_t           getRawBp(uint32_t i)    const { return pd[i]; }
    MarkerStrView      getRawName(uint32_t i)  const { return MarkerStrView{name_.view(i)}; }
    MarkerStrView      getRawA1(uint32_t i)    const { return MarkerStrView{a1_.view(i)}; }  ///< effect allele
    MarkerStrView      getRawA2(uint32_t i)    const { return MarkerStrView{a2_.view(i)}; }  ///< reference allele

    std::string_view   getRawChrView(uint32_t i)  const { return chr_dict_[chr_id_[i]]; }
    std::string_view   getRawNameView(uint32_t i) const { return name_.view(i); }
    std::string_view   getRawA1View(uint32_t i)   const { return a1_.view(i); }
    std::string_view   getRawA2View(uint32_t i)   const { return a2_.view(i); }

private:
    bool isHomogameticChr(const std::string& chr_label) const;
    bool isHeterogameticChr(const std::string& chr_label) const;
    static bool isPositiveIntegerChrLabel(const std::string& chr_label);
    bool isAllowedChrLabel(const std::string& chr_label) const;
    bool isAllowedChrLabelView(std::string_view chr_label) const;
    static std::string normalizeChrLabel(const std::string& chr_label);
    static void normalizeChrLabelInto(std::string_view chr_label, std::string& out);
    void loadSexChromosomeFile(const std::string& file_path);
    uint16_t internChr(std::string_view chr_label);

    std::vector<uint16_t> chr_id_;
    std::vector<std::string> chr_dict_;
    std::vector<uint8_t> chr_dict_sex_type_;  ///< 0=autosome/other, 1=homogametic, 2=heterogametic
    std::vector<uint8_t> chr_dict_allowed_;   ///< 1 if passes chr filter
    std::unordered_map<std::string, uint16_t> chr_to_id_;
    CompactStringArena name_;
    std::vector<float> gd;
    std::vector<uint32_t> pd;
    CompactAlleleArena a1_;
    CompactAlleleArena a2_;
    std::vector<uint64_t> byte_start;
    std::vector<uint32_t> raw_limits;

    std::vector<uint32_t> index_extract;
    std::vector<uint32_t> index_exclude;
    uint32_t num_marker = 0;
    uint32_t num_extract = 0;
    uint32_t num_exclude = 0;

    void read_bim(string bim_file);
    void read_mbim(const std::vector<string>& mfiles);

    void read_pvar(string pvar_file);
    void read_mpvar(const std::vector<string>& mfiles);

    int chr_filter_mode = 0;
    std::set<string> homogametic_chrs;
    std::set<string> heterogametic_chrs;
    std::vector<string> read_snplist(string snplist_file);

    std::set<string> allowed_chrs;
    std::vector<MarkerParam> markerParams;

    uint64_t maxGeno1ByteSize = 0;

    std::vector<uint64_t> byte_size;
   #ifdef BGEN_SUPPORT
    //bgen
    void read_bgen(string bgen_file);
    void read_mbgen(const std::vector<string>& mfiles);
    void read_bgen_index(string bgen_file);
   #endif
};


#endif //GCTA2_MARKER_H
