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
#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <cstdint>
using std::string;
using std::to_string;

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
    Marker();
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
    static int registerOption(std::map<string, std::vector<string>>& options_in);
    static void processMain();
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

    const std::string& getRawChr(uint32_t i)  const { return chr[i]; }
    uint32_t           getRawBp(uint32_t i)    const { return pd[i]; }
    const std::string& getRawName(uint32_t i)  const { return name[i]; }
    const std::string& getRawA1(uint32_t i)    const { return a1[i]; }  ///< effect allele
    const std::string& getRawA2(uint32_t i)    const { return a2[i]; }  ///< reference allele

private:
    static bool isHomogameticChr(const std::string& chr_label);
    static bool isHeterogameticChr(const std::string& chr_label);
    static bool isPositiveIntegerChrLabel(const std::string& chr_label);
    static bool isAllowedChrLabel(const std::string& chr_label);
    static std::string normalizeChrLabel(const std::string& chr_label);
    static void resetSexChromosomeSets();
    static void loadSexChromosomeFile(const std::string& file_path);

    std::vector<string> chr;
    std::vector<string> name;
    std::vector<float> gd;
    std::vector<uint32_t> pd;
    std::vector<string> a1;
    std::vector<string> a2;
    std::vector<uint64_t> byte_start;
    std::vector<uint32_t> raw_limits;

    

    std::vector<uint32_t> index_extract;
    std::vector<uint32_t> index_exclude;
    uint32_t num_marker;
    uint32_t num_extract;
    uint32_t num_exclude;

    void read_bim(string bim_file);
    void read_mbim(string bim_file);


    void read_pvar(string pvar_file);
    void read_mpvar(string mpvar_file);

    static std::map<string, std::string> options;
    static std::map<string, int> options_i;
    static std::set<string> homogametic_chrs;
    static std::set<string> heterogametic_chrs;
    static void addOneFileOption(string key_store, std::string append_string, std::string key_name,
                                 std::map<string, std::vector<string>> options_in);
    std::vector<string> read_snplist(string snplist_file);

    static std::set<string> allowed_chrs;
    std::vector<MarkerParam> markerParams;

    uint64_t maxGeno1ByteSize = 0;

    std::vector<uint64_t> byte_size;
   #ifdef BGEN_SUPPORT
    //bgen


    void read_bgen(string bgen_file);
    void read_mbgen(string mbgen_file);
    void read_bgen_index(string bgen_file);
   #endif
};


#endif //GCTA2_MARKER_H
