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

#include <sstream>
#include "Marker.h"
#include "Logger.h"
#include "constants.hpp"
#include <iterator>
#include <algorithm>
#include <numeric>
#include <boost/algorithm/string.hpp>
#include "utils.hpp"
#include "OptionIO.h"
#include <memory>
#include <utility>
#include <cctype>
#include <limits>

#ifdef BGEN_SUPPORT
#include <sqlite3.h>
#endif

using std::to_string;
using std::unique_ptr;

void Marker::normalizeChrLabelInto(std::string_view chr_label, std::string& out) {
    size_t b = 0;
    size_t e = chr_label.size();
    while (b < e && std::isspace(static_cast<unsigned char>(chr_label[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(chr_label[e - 1]))) --e;
    out.resize(e - b);
    for (size_t i = 0; i < e - b; ++i) {
        out[i] = gcta_text_io::ascii_toupper(chr_label[b + i]);
    }
}

std::string Marker::normalizeChrLabel(const std::string& chr_label) {
    std::string norm;
    normalizeChrLabelInto(chr_label, norm);
    return norm;
}

bool Marker::isHomogameticChr(const std::string& chr_label) const {
    if (homogametic_chrs.empty()) return false;
    std::string norm;
    normalizeChrLabelInto(chr_label, norm);
    return homogametic_chrs.count(norm) > 0;
}

bool Marker::isHeterogameticChr(const std::string& chr_label) const {
    if (heterogametic_chrs.empty()) return false;
    std::string norm;
    normalizeChrLabelInto(chr_label, norm);
    return heterogametic_chrs.count(norm) > 0;
}

bool Marker::isPositiveIntegerChrLabel(const std::string& chr_label) {
    const std::string norm = normalizeChrLabel(chr_label);
    if (norm.empty()) return false;
    for (char c : norm) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    int v = 0;
    return gcta_text_io::parse_int_sv(norm, v) && v >= 1;
}

bool Marker::isAllowedChrLabelView(std::string_view chr_label) const {
    if (chr_filter_mode == 2) {
        return true;
    }
    if (chr_filter_mode == 1) {
        if (homogametic_chrs.empty() && heterogametic_chrs.empty()) {
            return true;
        }
        std::string norm;
        normalizeChrLabelInto(chr_label, norm);
        return homogametic_chrs.count(norm) == 0 && heterogametic_chrs.count(norm) == 0;
    }
    if (allowed_chrs.empty()) {
        return true;
    }
    std::string norm;
    normalizeChrLabelInto(chr_label, norm);
    return allowed_chrs.count(norm) > 0;
}

bool Marker::isAllowedChrLabel(const std::string& chr_label) const {
    return isAllowedChrLabelView(chr_label);
}

void Marker::loadSexChromosomeFile(const std::string& file_path) {
    std::ifstream fin(file_path.c_str());
    if (!fin.good()) {
        LOGGER.e(0, "can't open sex chromosome file [" + file_path + "]");
    }

    std::set<std::string> file_homog;
    std::set<std::string> file_hetero;

    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        boost::trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::vector<std::string> toks;
        boost::split(toks, line, boost::is_any_of("\t "), boost::token_compress_on);
        if (toks.empty()) continue;
        if (toks.size() < 3) {
            LOGGER.e(0, "invalid --sex-chr-file format. Require at least 3 columns per non-comment line: <HOMOGAMETIC|HETEROGAMETIC> <PLINK sex code 1|2> <chromosome labels...>.");
        }

        const std::string key = normalizeChrLabel(toks[0]);
        int plink_sex_code = -1;
        try {
            plink_sex_code = std::stoi(toks[1]);
        } catch (std::invalid_argument&) {
            LOGGER.e(0, "invalid PLINK sex code [" + toks[1] + "] in --sex-chr-file. Expected 1 or 2.");
        } catch (std::out_of_range&) {
            LOGGER.e(0, "PLINK sex code [" + toks[1] + "] is out of range in --sex-chr-file.");
        }
        if (plink_sex_code != 1 && plink_sex_code != 2) {
            LOGGER.e(0, "invalid PLINK sex code [" + toks[1] + "] in --sex-chr-file. Expected 1 or 2.");
        }

        if (key == "HOMOGAMETIC" || key == "HOMO" || key == "XX") {
            if (plink_sex_code != 2) {
                LOGGER.e(0, "invalid --sex-chr-file mapping for HOMOGAMETIC. Fixed PLINK coding requires HOMOGAMETIC=2.");
            }
            for (size_t i = 2; i < toks.size(); ++i) {
                const std::string v = normalizeChrLabel(toks[i]);
                if (!v.empty()) file_homog.insert(v);
            }
        } else if (key == "HETEROGAMETIC" || key == "HETERO" || key == "XY") {
            if (plink_sex_code != 1) {
                LOGGER.e(0, "invalid --sex-chr-file mapping for HETEROGAMETIC. Fixed PLINK coding requires HETEROGAMETIC=1.");
            }
            for (size_t i = 2; i < toks.size(); ++i) {
                const std::string v = normalizeChrLabel(toks[i]);
                if (!v.empty()) file_hetero.insert(v);
            }
        } else {
            LOGGER.e(0, "invalid --sex-chr-file line prefix [" + toks[0] + "]. Use HOMOGAMETIC/HETEROGAMETIC.");
        }
    }

    if (file_homog.empty() && file_hetero.empty()) {
        LOGGER.e(0, "invalid --sex-chr-file format. Provide at least one 'HOMOGAMETIC <PLINK sex code 1|2> <labels...>' or 'HETEROGAMETIC <PLINK sex code 1|2> <labels...>' line.");
    }

    for (const auto& v : file_homog) {
        if (file_hetero.count(v)) {
            LOGGER.e(0, "sex chromosome label [" + v + "] is listed for both homogametic and heterogametic sets.");
        }
    }

    homogametic_chrs = std::move(file_homog);
    heterogametic_chrs = std::move(file_hetero);
}

Marker::Marker(const gcta_cli::GenoInputOptions& geno,
               const gcta_cli::SampleFilterOptions& sample_filter,
               const gcta_cli::MarkerFilterOptions& marker_filter,
               bool default_autosome,
               bool force_chr_homogametic) {
    if (!sample_filter.sex_chr_file.empty()) {
        loadSexChromosomeFile(sample_filter.sex_chr_file);
    }

    if (!marker_filter.homogametic_chr.empty()) {
        homogametic_chrs.clear();
        for (const auto& s : marker_filter.homogametic_chr) {
            homogametic_chrs.insert(normalizeChrLabel(s));
        }
    }

    if (!marker_filter.heterogametic_chr.empty()) {
        heterogametic_chrs.clear();
        for (const auto& s : marker_filter.heterogametic_chr) {
            heterogametic_chrs.insert(normalizeChrLabel(s));
        }
    }

    for (const auto& s : homogametic_chrs) {
        if (heterogametic_chrs.count(s)) {
            LOGGER.e(0, "chromosome label [" + s + "] cannot be both homogametic and heterogametic.");
        }
    }

    if (force_chr_homogametic || marker_filter.chr_homogametic) {
        if (homogametic_chrs.empty()) {
            LOGGER.e(0, "--chr-homogametic requires homogametic chromosome labels to be provided via --sex-chr-file or --homogametic-chr.");
        }
        chr_filter_mode = 0;
        allowed_chrs = homogametic_chrs;
    } else if (!marker_filter.chr.empty()) {
        chr_filter_mode = 0;
        auto parse_int_exact = [](const std::string& s, int& out) -> bool {
            size_t pos = 0;
            try {
                out = std::stoi(s, &pos);
                return pos == s.size();
            } catch (...) {
                return false;
            }
        };
        const auto& chr_args = marker_filter.chr;
        if (chr_args.size() == 1) {
            int c = 0;
            if (parse_int_exact(chr_args[0], c)) {
                if (c < 1) {
                    LOGGER.e(0, "the value specified for --chr must be >= 1.");
                }
                allowed_chrs.insert(std::to_string(c));
            } else {
                allowed_chrs.insert(normalizeChrLabel(chr_args[0]));
            }
        } else if (chr_args.size() == 2) {
            int chr_start = 0;
            int chr_end = 0;
            if (parse_int_exact(chr_args[0], chr_start) && parse_int_exact(chr_args[1], chr_end)) {
                if (chr_start < 1 || chr_end < 1) {
                    LOGGER.e(0, "the numeric value(s) specified for --chr must be >= 1.");
                }
                if (chr_start > chr_end) {
                    LOGGER.e(0, "--chr numeric range is invalid: start is larger than end.");
                }
                for (int c = chr_start; c <= chr_end; ++c) {
                    allowed_chrs.insert(std::to_string(c));
                }
            } else {
                allowed_chrs.insert(normalizeChrLabel(chr_args[0]));
                allowed_chrs.insert(normalizeChrLabel(chr_args[1]));
            }
        } else {
            LOGGER.e(0, "multiple --chr flags are not supported currently");
        }
    } else if (marker_filter.autosome || default_autosome) {
        chr_filter_mode = 1;
    } else if (marker_filter.autosome_sex) {
        chr_filter_mode = 2;
    }

    bool has_marker = false;

    const string bim_file = !geno.bim.empty() ? geno.bim
        : (!geno.bfile.empty() ? geno.bfile + ".bim"
        : (!geno.bpfile.empty() ? geno.bpfile + ".bim" : ""));
    if (!bim_file.empty()) {
        requireFileReadable(geno.bim.empty() ? "--bfile/--bpfile" : "--bim", bim_file);
        read_bim(bim_file);
        has_marker = true;
        raw_limits.push_back(num_marker);
    }

    if (!geno.mbfile.empty()) {
        has_marker = true;
        read_mbim(expandFileList(geno.mbfile, ".bim"));
    }

    if (!geno.mbpfile.empty()) {
        has_marker = true;
        read_mbim(expandFileList(geno.mbpfile, ".bim"));
    }

    if (!geno.pfile.empty()) {
        const string pvar_file = geno.pfile + ".pvar";
        requireFileReadable("--pfile", pvar_file);
        has_marker = true;
        read_pvar(pvar_file);
        raw_limits.push_back(num_marker);
    }

    if (!geno.mpfile.empty()) {
        has_marker = true;
        read_mpvar(expandFileList(geno.mpfile, ".pvar"));
    }

#ifdef BGEN_SUPPORT
    if (!geno.bgen.empty()) {
        requireFileReadable("--bgen", geno.bgen);
        has_marker = true;
        read_bgen_index(geno.bgen);
        raw_limits.push_back(num_marker);
        index_extract.resize(num_marker);
        std::iota(index_extract.begin(), index_extract.end(), 0);
    }

    if (!geno.mbgen.empty()) {
        has_marker = true;
        read_mbgen(expandFileList(geno.mbgen, ".bgen"));
    }
#endif

    if (!has_marker) {
        LOGGER.e(0, "no marker found.");
    }

    if (!marker_filter.extract.empty()) {
        requireFileReadable("--extract", marker_filter.extract);
        std::vector<string> extractlist = read_snplist(marker_filter.extract);
        LOGGER << "Get " << extractlist.size() << " SNPs from list [" << marker_filter.extract << "]." << std::endl;
        extract_marker(extractlist, true);
    }

    if (!marker_filter.exclude.empty()) {
        requireFileReadable("--exclude", marker_filter.exclude);
        std::vector<string> excludelist = read_snplist(marker_filter.exclude);
        LOGGER << "Get " << excludelist.size() << " SNPs from list [" << marker_filter.exclude << "]." << std::endl;
        extract_marker(excludelist, false);
    }
    reset_exclude();

    if (index_extract.empty()) {
        LOGGER.e(0, "no SNP remained.");
    }
}

void Marker::matchSNPListFile(string filename, int num_min_fields, const std::vector<int>& field_return, std::vector<string> &fields){
    std::vector<string> temp_fields;
    std::ifstream allele_file(filename.c_str());
    if(allele_file.fail()){
        LOGGER.e(0, "can't open [" + filename + "] to read]");
    }

    int max_fields = num_min_fields;
    int require_fields = (field_return.size() > 0) ? (field_return[field_return.size() - 1] + 1): 0;
    require_fields = (max_fields > require_fields) ? max_fields : require_fields;

    std::vector<string> marker_name, ref_allele;
    std::string line;
    std::vector<uint32_t> bad_lines;
    int num_line = 0;
    std::vector<string> line_elements;
    while(std::getline(allele_file, line)){
        num_line++;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        line_elements.clear();
        size_t start = 0, pos;
        while ((pos = line.find_first_of("\t ", start)) != std::string::npos) {
            line_elements.emplace_back(line, start, pos - start);
            start = pos + 1;
        }
        line_elements.emplace_back(line, start);
        uint16_t num_elements = line_elements.size();
        if(num_elements < require_fields){
            bad_lines.push_back(num_line);
        }
        if(num_elements >= 1){
            marker_name.push_back(line_elements[0]);
        }

        if(num_elements >= 2){
            ref_allele.push_back(line_elements[1]);
        }
        for(auto index : field_return){
            temp_fields.push_back(line_elements[index]);
        }

    }
    allele_file.close();

    // match snp name
    std::vector<uint32_t> marker_index, ref_index;
    vector_commonIndex_sorted1(name, marker_name, marker_index, ref_index);

    // match alleles
    std::vector<uint32_t> index_remained;
    index_remained.reserve(marker_index.size());
    std::vector<uint32_t> ref_index_remained;
    ref_index_remained.reserve(marker_index.size());
    if(ref_allele.size()){
        for(int i = 0; i < marker_index.size(); i++){
            uint32_t cur_marker_index = marker_index[i];
            uint32_t cur_ref_index = ref_index[i];
            std::string cur_ref = ref_allele[cur_ref_index];
            std::transform(cur_ref.begin(), cur_ref.end(), cur_ref.begin(), toupper);
            if(a1[cur_marker_index] == cur_ref || a2[cur_marker_index] == cur_ref){
                index_remained.push_back(cur_marker_index);
                ref_index_remained.push_back(cur_ref_index);
            }
        }
    }else{
        index_remained = marker_index;
    }

    // match original kept list
    std::vector<uint32_t> extract_index, export_index;
    vector_commonIndex_sorted1(index_extract, index_remained, extract_index, export_index);

    std::vector<uint32_t> index_common;
    index_common.resize(extract_index.size());
    std::transform(extract_index.begin(), extract_index.end(), index_common.begin(), [this](size_t pos){return (this->index_extract)[pos];});

    int rm_snps = index_extract.size() - index_common.size();
    if(rm_snps){
        LOGGER.w(0, to_string(rm_snps) + " SNPs are removed due to mismatched SNP IDs or alleles.");
    }

    keep_raw_index(index_common);

    int field_num = field_return.size();
    if(field_num){
        fields.reserve(export_index.size() * field_num);
        for(auto i : export_index){
            uint32_t t_index = ref_index_remained[i];
            uint32_t t_index_from = t_index * field_num;
            for(int j = t_index_from; j < t_index_from + field_num; j++){
                fields.push_back(temp_fields[j]);
            }
        }
    }
}

void Marker::read_mbim(const std::vector<string>& mfiles){
    raw_limits.clear();
    for(const auto& mfile : mfiles){
        read_bim(mfile);
        raw_limits.push_back(num_marker);
    }
}

void Marker::read_mpvar(const std::vector<string>& mfiles){
    raw_limits.clear();
    for(const auto& mfile : mfiles){
        read_pvar(mfile);
        raw_limits.push_back(num_marker);
    }
}

#ifdef BGEN_SUPPORT
void Marker::read_mbgen(const std::vector<string>& mfiles){
    raw_limits.clear();
    for(const auto& mfile : mfiles){
        read_bgen_index(mfile);
        raw_limits.push_back(num_marker);
    }

    index_extract.resize(num_marker);
    std::iota(index_extract.begin(), index_extract.end(), 0); 
}
#endif


//cur_marker_index:  is the index point to position of index_extract
std::vector<uint32_t> Marker::getNextWindowIndex(uint32_t cur_marker_index, uint32_t window, bool& chr_ends, bool& isHomogameticChrom, bool retRaw){
    std::vector<uint32_t> indices;
    if(cur_marker_index >= index_extract.size()){
        chr_ends = true;
        return indices;
    }
    uint32_t cur_index = index_extract[cur_marker_index];

    uint32_t cur_pd = pd[cur_index];
    std::string cur_chr = chr[cur_index];
    uint32_t final_pd = window + cur_pd;

    chr_ends = false;
    isHomogameticChrom = isHomogameticChr(cur_chr);

    for(uint32_t marker_index = cur_marker_index; marker_index < index_extract.size(); marker_index++){
        uint32_t temp_index = index_extract[marker_index];
        if(chr[temp_index] != cur_chr){
            chr_ends = true;
            break;
        }
        if(pd[temp_index] <= final_pd){
            indices.push_back(retRaw ? temp_index : marker_index);
        }else{
            break;
        }
    }
    return indices;
}

uint32_t Marker::getNextSize(const std::vector<uint32_t> &rawRef, uint32_t curExtractIndex, uint32_t num, int &fileIndex, bool &chr_ends, uint8_t &sexChromType){
    if(curExtractIndex >= rawRef.size()){
        chr_ends = true;
        return 0;
    }
    uint32_t cur_index = rawRef[curExtractIndex];
    std::string cur_chr = chr[cur_index];


    chr_ends = false;
    
    if(isHomogameticChr(cur_chr)){
        sexChromType = 1;
    }else if(isHeterogameticChr(cur_chr)){
        sexChromType = 2;
    }else{
        sexChromType = 0;
    }

    fileIndex = getMIndex(cur_index);
    uint32_t rawIndexLimits = raw_limits[fileIndex];

    uint32_t retSize = 0; 
    for(uint32_t marker_index = curExtractIndex; marker_index < curExtractIndex + num; marker_index++){
        if(marker_index >= rawRef.size()){
            chr_ends = true;
            break;
        }
        uint32_t temp_index = rawRef[marker_index];
        if(chr[temp_index] != cur_chr){
            chr_ends = true;
            break;
        }
        if(temp_index >= rawIndexLimits){
            break;
        }
        retSize++;
    }
    return retSize;
}



std::vector<uint32_t> Marker::getNextSizeIndex(uint32_t cur_marker_index, uint32_t num, bool& chr_ends, bool& isHomogameticChrom, bool retRaw){
    std::vector<uint32_t> indices;
    indices.reserve(num);
    if(cur_marker_index >= index_extract.size()){
        chr_ends = true;
        return indices;
    }
    uint32_t cur_index = index_extract[cur_marker_index];
    std::string cur_chr = chr[cur_index];

    chr_ends = false;
    isHomogameticChrom = isHomogameticChr(cur_chr);

    for(uint32_t marker_index = cur_marker_index; marker_index < cur_marker_index + num; marker_index++){
        if(marker_index >= index_extract.size()){
            chr_ends = true;
            break;
        }
        uint32_t temp_index = index_extract[marker_index];
        if(chr[temp_index] != cur_chr){
            chr_ends = true;
            break;
        }
        indices.push_back(retRaw ? temp_index : marker_index);
    }
    return indices;
}



uint32_t Marker::getNextWindowSize(uint32_t cur_marker_index, uint32_t window){
    if(cur_marker_index >= index_extract.size()){
        return 0;
    }
    uint32_t cur_index = index_extract[cur_marker_index];

    uint32_t cur_pd = pd[cur_index];
    std::string cur_chr = chr[cur_index];
    uint32_t final_pd = window + cur_pd;

    uint32_t count = 0;

    for(uint32_t marker_index = cur_marker_index; marker_index < index_extract.size(); marker_index++){
        uint32_t temp_index = index_extract[marker_index];
        if(chr[temp_index] != cur_chr){
            break;
        }
        if(pd[temp_index] <= final_pd){
            count++;
        }else{
            break;
        }
    }
    return count;
}


int Marker::getMIndex(uint32_t raw_index){
    for(int i = 0; i < raw_limits.size(); i++){
        if(raw_index < raw_limits[i]){
            return i;
        }
    }
    LOGGER.e(0, "the SNP index is too large in" + to_string(raw_index));
    return 0;
}

void Marker::save_marker(string filename){
    LOGGER.i(0, "Saving SNP information to [" + filename + "]...");
    std::ofstream out(filename.c_str());
    for(auto & i : index_extract){
        const float gd_val = (i < gd.size()) ? gd[i] : 0.0f;
        out << chr[i] << "\t" << name[i] << "\t" << gd_val
            << "\t" << pd[i] << "\t" << a1[i] << "\t"
            << a2[i] << "\n";
    }
    LOGGER.i(0, to_string(index_extract.size()) + " SNPs saved.");

}

void Marker::read_pvar(string pvar_file){
    LOGGER.i(0, "Reading PLINK2 PVAR file from [" + pvar_file + "]...");
    gcta_text_io::FastLineReader reader;
    if (!reader.open(pvar_file)) {
        LOGGER.e(0, "can't read [" + pvar_file + "].");
    }

    std::string_view line;
    std::string preLine;
    int curReadLine = 0;
    bool found_data = false;
    while (reader.next_line(line)) {
        curReadLine++;
        const size_t first_non_ws = line.find_first_not_of("\r\t ");
        if (first_non_ws != std::string_view::npos && line[first_non_ws] == '#') {
            preLine.assign(line);
            continue;
        }
        if (first_non_ws == std::string_view::npos) {
            LOGGER.e(0, "blank header line or no valid text data.");
        }
        found_data = true;
        break;
    }
    if (!found_data) {
        LOGGER.e(0, "blank header line or no valid text data.");
    }

    const int nHeader = curReadLine - 1;
    std::vector<std::string_view> toks;
    toks.reserve(16);
    gcta_text_io::split_ws_exact(line, toks);
    const int ncol = static_cast<int>(toks.size());

    std::vector<std::string_view> head_toks;
    if (!preLine.empty()) {
        gcta_text_io::split_ws_exact(preLine, head_toks);
        if (static_cast<int>(head_toks.size()) != ncol) {
            LOGGER.e(0, "data are inconsistent with the headers.");
        }
    }
    if (ncol < 5) {
        LOGGER.e(0, "the number of fields in [" + pvar_file + "] is less than 5.");
    }

    int iChr = -1, iPOS = -1, iID = -1, iA1 = -1, iA2 = -1;
    if (head_toks.empty()) {
        if (ncol == 6) {
            iChr = 0;
            iID  = 1;
            iPOS = 3;
            iA1  = 4;
            iA2  = 5;
        } else if (ncol == 5) {
            iChr = 0;
            iID  = 1;
            iPOS = 2;
            iA1  = 3;
            iA2  = 4;
        } else {
            LOGGER.e(0, "The .bim file only has " + to_string(ncol) + " columns, which is not valid.");
        }
    } else {
        if (head_toks[0] != "#CHROM") {
            LOGGER.e(0, "invalid PVAR file. It should start with #CHROM.");
        }
        iChr = 0;
        int validHeadCT = 1;
        for (int c = 0; c < ncol; ++c) {
            if (head_toks[c] == "POS" && iPOS < 0)      { iPOS = c; validHeadCT++; }
            else if (head_toks[c] == "ID" && iID < 0)   { iID  = c; validHeadCT++; }
            else if (head_toks[c] == "REF" && iA2 < 0)  { iA2  = c; validHeadCT++; }
            else if (head_toks[c] == "ALT" && iA1 < 0)  { iA1  = c; validHeadCT++; }
        }
        if (validHeadCT != 5) {
            LOGGER.e(0, "can't find all the essential columns in the PVAR file.");
        }
    }

    const uint32_t oriSize = static_cast<uint32_t>(chr.size());
    if (reader.file_size() > 0) {
        const size_t avg_line = std::max<size_t>(line.size() + 1, 10);
        const size_t est_rows = static_cast<size_t>(reader.file_size() / avg_line) + 64;
        const size_t target = static_cast<size_t>(oriSize) + est_rows;
        chr.reserve(target);
        name.reserve(target);
        pd.reserve(target);
        a1.reserve(target);
        a2.reserve(target);
        index_extract.reserve(index_extract.size() + est_rows);
    }

    uint32_t nrows = 0;
    uint32_t nValidSNP = 0;
    int line_number = nHeader;
    std::string last_chr;
    bool has_last_chr = false;
    bool last_chr_allowed = false;

    auto process_row = [&]() {
        line_number++;
        if (static_cast<int>(toks.size()) < ncol) {
            LOGGER.e(0, "the file [" + pvar_file + "] contains different number of elements in line " + to_string(line_number) + ".");
        } else if (static_cast<int>(toks.size()) > ncol) {
            LOGGER.w(0, "the file [" + pvar_file + "] contains extra number of elements in line " + to_string(line_number) + ".");
        }
        const std::string_view sv_chr = toks[static_cast<size_t>(iChr)];
        const std::string_view sv_id  = toks[static_cast<size_t>(iID)];
        const std::string_view sv_pos = toks[static_cast<size_t>(iPOS)];
        const std::string_view sv_a1  = toks[static_cast<size_t>(iA1)];
        const std::string_view sv_a2  = toks[static_cast<size_t>(iA2)];

        int64_t pos_val = 0;
        if (!gcta_text_io::parse_int64_sv(sv_pos, pos_val) ||
            pos_val < std::numeric_limits<int>::min() ||
            pos_val > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            LOGGER.e(0, "line " + to_string(line_number) + " contains invalid distance value.");
        }

        const uint32_t curRow = oriSize + nrows;
        chr.emplace_back(sv_chr);
        name.emplace_back(sv_id);
        pd.push_back(static_cast<uint32_t>(pos_val));
        a1.emplace_back(sv_a1);
        gcta_text_io::uppercase_inplace(a1.back());
        a2.emplace_back(sv_a2);
        gcta_text_io::uppercase_inplace(a2.back());

        bool allowed;
        if (has_last_chr && sv_chr == last_chr) {
            allowed = last_chr_allowed;
        } else {
            last_chr.assign(sv_chr);
            has_last_chr = true;
            last_chr_allowed = isAllowedChrLabelView(sv_chr);
            allowed = last_chr_allowed;
        }
        if (allowed) {
            index_extract.push_back(curRow);
            nValidSNP++;
        }
        nrows++;
    };

    process_row();
    while (reader.next_line(line)) {
        gcta_text_io::split_ws_exact(line, toks);
        process_row();
    }

    num_marker = static_cast<uint32_t>(name.size());
    num_extract = static_cast<uint32_t>(index_extract.size());

    LOGGER.i(0, to_string(num_marker) + " SNPs to be included from PVAR file(s).");
    if (nrows != nValidSNP) {
        LOGGER.i(0, to_string(num_extract) + " SNPs to be included on valid chromosomes");
    }

    MarkerParam markerParam;
    markerParam.rawCountSNP = nrows;
    markerParam.rawCountSample = 0; // dummy
    markerParam.compressFormat = 0;
    markerParam.posGenoDataStart = 3; // just dummy, pgen don't start with 3
    markerParams.push_back(markerParam);
}

void Marker::read_bim(string bim_file) {
    LOGGER.i(0, "Reading PLINK BIM file from [" + bim_file + "]...");
    gcta_text_io::FastLineReader reader;
    if (!reader.open(bim_file)) {
        LOGGER.e(0, "cannot open the file [" + bim_file + "] to read");
    }

    const uint32_t oriSize = static_cast<uint32_t>(name.size());
    uint32_t line_number = oriSize;
    const uint32_t start_line_number = oriSize + 1;
    size_t last_length = 0;
    bool reserved = false;

    std::vector<std::string_view> toks;
    toks.reserve(8);
    std::string last_chr;
    bool has_last_chr = false;
    bool last_chr_allowed = false;

    std::string_view line;
    while (reader.next_line(line)) {
        if (!reserved) {
            reserved = true;
            if (reader.file_size() > 0) {
                const size_t avg_line = std::max<size_t>(line.size() + 1, 10);
                const size_t est_rows = static_cast<size_t>(reader.file_size() / avg_line) + 64;
                const size_t target = static_cast<size_t>(oriSize) + est_rows;
                chr.reserve(target);
                name.reserve(target);
                pd.reserve(target);
                a1.reserve(target);
                a2.reserve(target);
                index_extract.reserve(index_extract.size() + est_rows);
                if (!gd.empty()) gd.reserve(target);
            }
        }
        line_number++;
        gcta_text_io::split_ws_exact(line, toks);
        if (toks.size() < static_cast<size_t>(Constants::NUM_BIM_COL)) {
            LOGGER.e(0, "the bim file [" + bim_file + "], line " + to_string(line_number)
                   + " has elements less than " + to_string(Constants::NUM_BIM_COL));
        }
        if (line_number > start_line_number && toks.size() != last_length) {
            LOGGER.w(0, "the bim file [" + bim_file + "], line " + to_string(line_number) +
                    " have different elements, take care");
        }

        float gd_val = 0.0f;
        int64_t pd_val = 0;
        if (!gcta_text_io::parse_float_sv(toks[2], gd_val) ||
            !gcta_text_io::parse_int64_sv(toks[3], pd_val) ||
            pd_val < std::numeric_limits<int>::min() ||
            pd_val > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            LOGGER.e(0, "Line " + to_string(line_number) + " of [" + bim_file +
                        "] contains illegal distance value, please check");
        }

        const std::string_view sv_chr = toks[0];
        chr.emplace_back(sv_chr);
        name.emplace_back(toks[1]);
        if (!gd.empty() || gd_val != 0.0f) {
            if (gd.empty()) {
                gd.assign(static_cast<size_t>(line_number - 1), 0.0f);
            }
            gd.push_back(gd_val);
        }
        pd.push_back(static_cast<uint32_t>(pd_val));
        a1.emplace_back(toks[4]);
        gcta_text_io::uppercase_inplace(a1.back());
        a2.emplace_back(toks[5]);
        gcta_text_io::uppercase_inplace(a2.back());

        last_length = toks.size();
        bool allowed;
        if (has_last_chr && sv_chr == last_chr) {
            allowed = last_chr_allowed;
        } else {
            last_chr.assign(sv_chr);
            has_last_chr = true;
            last_chr_allowed = isAllowedChrLabelView(sv_chr);
            allowed = last_chr_allowed;
        }
        if (allowed) {
            index_extract.push_back(line_number - 1);
        }
    }
    num_marker = static_cast<uint32_t>(name.size());
    num_extract = static_cast<uint32_t>(index_extract.size());
    LOGGER.i(0, to_string(num_marker) + " SNPs to be included from BIM file(s).");
    if (num_marker != num_extract) {
        LOGGER.i(0, to_string(num_extract) + " SNPs to be included on valid chromosomes");
    }
    MarkerParam markerParam;
    markerParam.rawCountSNP = num_marker - start_line_number + 1;
    markerParam.rawCountSample = 0;
    markerParam.compressFormat = 0;
    markerParam.posGenoDataStart = 3;
    markerParams.push_back(markerParam);
}

std::vector<std::pair<std::string, std::vector<uint32_t>>> Marker::read_gene(string gfile){
    std::ifstream genein(gfile.c_str());
    if(!genein){
        LOGGER.e(0, "cannot open the file [" + gfile + "] to read");
    }

    std::string line;
    std::getline(genein, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<string> line_elements;
    {
        size_t start = 0, pos;
        while ((pos = line.find_first_of("\t ", start)) != std::string::npos) {
            line_elements.emplace_back(line, start, pos - start);
            start = pos + 1;
        }
        line_elements.emplace_back(line, start);
    }
    int nElements;
    if(line_elements[0] == "gene" && line_elements[1] == "chr" && 
            line_elements[2] == "start" && line_elements[3] == "end" && line_elements[4] == "strand"){
        nElements = line_elements.size();
    }else{
        LOGGER.e(0, "invalid gene list file.");
    }

    int line_number  = 1;
    std::string chr_item;
    std::string curChrItem;
    bool bFind = false;
    uint32_t chrStartIndex = 0;
    uint32_t chrEndIndex = 0;
    std::vector<uint32_t> curIndices;

    std::vector<std::pair<std::string, std::vector<uint32_t>>> gene;
    std::vector<string> gene_line_elements;
    while(std::getline(genein, line)){
        line_number++;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        gene_line_elements.clear();
        {
            size_t start = 0, pos;
            while ((pos = line.find_first_of("\t ", start)) != std::string::npos) {
                gene_line_elements.emplace_back(line, start, pos - start);
                start = pos + 1;
            }
            gene_line_elements.emplace_back(line, start);
        }
        std::vector<string> &line_elements = gene_line_elements;
        if(line_elements.size() != nElements) {
            LOGGER.e(0, "  line " + to_string(line_number)
                   + " has elements not equal to " + to_string(nElements));
        }
        // replace std::strings
        chr_item = line_elements[1];
        
        std::string curGeneName = line_elements[0];
        int curStart, curEnd; 
        try{
            curStart = std::stoi(line_elements[2]);
            curEnd = std::stoi(line_elements[3]);
        }catch(std::invalid_argument&){
           LOGGER.e(0, "  line " + to_string(line_number) + " of [" + gfile +
                    "] contains invalid position value, please check");
        }

        if(chr_item != curChrItem){
            bool isSetStart = false;
            bool isSetEnd = false;
            for(uint32_t marker_index = 0; marker_index < index_extract.size(); marker_index++){
                uint32_t temp_index = index_extract[marker_index];
                //LOGGER << temp_index << "\t" << chr_item << "\t" << chr[temp_index] << std::endl;
                if(chr[temp_index] == chr_item){
                    if((!isSetStart)){
                        chrStartIndex = marker_index;
                        isSetStart = true;
                    }
                }else{
                    if(isSetStart){
                        chrEndIndex = marker_index - 1;
                        isSetEnd = true;
                        break;
                    }
                }
            }
            if(isSetStart){
                if(!isSetEnd) chrEndIndex = index_extract.size() - 1;
            }else{
                //LOGGER << "gene " << curGeneName << " can not find, chr: " <<  int(chr_item) << std::endl;
                //curChrItem = chr_item;
                continue;
                //LOGGER.e(0, " line " + to_string(line_number) + ", chromosome information can't be found in the genotype file");
            }
            curChrItem = chr_item;
        }

        std::vector<uint32_t> indicies;
        for(uint32_t i = chrStartIndex; i <= chrEndIndex; i++){
            uint32_t temp_index = index_extract[i];
            uint32_t cur_pd = pd[temp_index];
            if(cur_pd >= curStart && cur_pd <= curEnd){
                indicies.push_back(i);
            }
        }
        if(indicies.size() == 0) continue;
        std::pair<string, std::vector<uint32_t>> pair_gene = std::make_pair(curGeneName+"\t" + to_string(curStart) + "\t" + to_string(curEnd), indicies);

        gene.emplace_back(pair_gene);
    }
    return gene;
}


void Marker::getStartPosSize(uint32_t raw_index, uint64_t &pos, uint64_t &size){
    pos = (raw_index < byte_start.size()) ? byte_start[raw_index] : 1;
    size = (raw_index < byte_size.size()) ? byte_size[raw_index] : 0;
}

#ifdef BGEN_SUPPORT
MarkerParam Marker::getBgenMarkerParam(FILE *h_bgen, std::string &outputs){
    // check file formats
    uint32_t start_byte = read1Byte<uint32_t>(h_bgen);
    uint32_t start_data_block = start_byte + 4;

    uint32_t len_header = read1Byte<uint32_t>(h_bgen);

    uint32_t n_variants_total = read1Byte<uint32_t>(h_bgen);

    auto n_sample = read1Byte<uint32_t>(h_bgen);
    outputs = outputs + to_string(n_sample) + " samples, " + to_string(n_variants_total) + " SNPs.\n";

    char magic[5];
    readBytes<char>(h_bgen, 4, magic);
    if(strncmp(magic, "bgen", 4) != 0){
        LOGGER.e(0, "bad magic number in the bgen file.");
    }

    int32_t skip_byte = (int32_t)len_header - 20;
    if(skip_byte > 0){
        fseek(h_bgen, skip_byte, SEEK_CUR); 
    }else if(skip_byte < 0){
        LOGGER.e(0, "unusual header length - might be an invalid bgen file.");
    }

    uint32_t flags = read1Byte<uint32_t>(h_bgen);
    int compress_block = flags & 3; // first 2 bit
    int layout = (flags >> 2) & 15; //2-5 bit
    int has_sample = (flags >> 31) & 1;

    outputs += "BGEN version ";
    switch(layout){
        case 1:
            outputs += "1.1, ";
            break;
        case 2:
            outputs += "1.2, ";
            break;
        default:
            outputs += "unkown, ";
    }
    switch(compress_block){
        case 0:
            outputs += "no compress";
            break;
        case 1:
            outputs += "compressed by zlib";
            break;
        case 2:
            outputs += "compressed by zstd";
            break;
        default:
            outputs += "unknown format";
    }
    outputs += ".";

    if(layout != 2){
        LOGGER.e(0, "GCTA only supports bgen version 1.2 and 1.3. Use QCTOOLv2 to convert files with < v1.2 to a newer version.");
    }

    if(has_sample){
        LOGGER.w(0, "GCTA reads sample information from '--sample' input, but ignores the built-in sample data.");
    }

    MarkerParam param;
    param.rawCountSNP = n_variants_total;
    param.rawCountSample = n_sample;
    param.compressFormat = compress_block;
    param.posGenoDataStart = start_data_block;
    return param;
}

#define RCSTR(TEMP) std::string(reinterpret_cast<const char*>(TEMP))

void Marker::read_bgen_index(string bgen_file){
    sqlite3 *db;
    int rc;
    std::string index_fname = bgen_file + ".bgi";
    std::string query_file = "file:" + index_fname + "?nolock=1";
    LOGGER.i(0, "Loading bgen index from [" + index_fname + "]...");
    rc = sqlite3_open_v2(query_file.c_str(), &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, NULL);

    //string prompt_index = "'gcta64 --bgen test.bgen --bgen-index --out test.bgen.bgi' or 'bgenix -g test.bgen -index'";
    std::string prompt_index = "'bgenix -g test.bgen -index'";
    if(rc){
        LOGGER.e(0, "can't open the index file: " + std::string(sqlite3_errmsg(db)) + 
                "\nIndex the bgen file by " + prompt_index + ".");
    }

    sqlite3_stmt *stmt;

    //check the bgen file
    const char * sql_meta = "SELECT * FROM Metadata";
    rc = sqlite3_prepare_v2(db, sql_meta, -1, &stmt, NULL);
    if(rc != SQLITE_OK){
        LOGGER.e(0, "bad index file: " + std::string(sqlite3_errmsg(db)) +
                "\nTry to regenerate the index by " + prompt_index + ".");
    }
    std::string filename;
    uint64_t file_size;
    const char * first1000bytes;
    
    rc = sqlite3_step(stmt);
    if(rc == SQLITE_ROW){
        filename = RCSTR(sqlite3_column_text(stmt, 0));
        file_size = sqlite3_column_int64(stmt, 1);
        first1000bytes = reinterpret_cast< const char* >(sqlite3_column_blob(stmt, 3));
    }

    //check 1000 bytes and file size;
    int num_firstNbytes = file_size < 1000 ? file_size : 1000;

    FILE * h_bgen = fopen(bgen_file.c_str(), "rb");
    if(h_bgen == NULL){
        LOGGER.e(0, "can't read the bgen file [" + bgen_file + "], " + std::string(strerror(errno)));
    }

    char * geno_first1000bytes = new char[num_firstNbytes];
    if(num_firstNbytes != fread(geno_first1000bytes, sizeof(char), num_firstNbytes, h_bgen)){
        LOGGER.e(0, "can't read the bgen file [" + bgen_file + "], " + std::string(strerror(errno)));
    }

    if(strncmp(first1000bytes, geno_first1000bytes, num_firstNbytes) != 0){
        LOGGER.e(0, "bad index file, the first 1000 bytes aren't consistent."
                "\nTry to regenerate the index by " + prompt_index + ".");
    }

    fseek(h_bgen, 0L, SEEK_END);
    if(file_size != ftell(h_bgen)){
        LOGGER.e(0, "bad index file, the file size isn't consistent."
                "\nTry to regenerate the index by " + prompt_index + ".");
    }
    rewind(h_bgen);
    rc = sqlite3_reset(stmt);

    // check variants;
    const char * sql_allvar = "SELECT count(*) FROM Variant";
    const char * sql_var1 = "SELECT * FROM Variant";
    rc = sqlite3_prepare_v2(db, sql_allvar, -1, &stmt, NULL);
    if(rc != SQLITE_OK){
        LOGGER.e(0, "bad index file: " + std::string(sqlite3_errmsg(db)) +
                "\nTry to regenerate the index by " + prompt_index + ".");
    }
    rc = sqlite3_step(stmt);

    int n_variants_total_index;
    if(rc == SQLITE_ROW){
        n_variants_total_index = sqlite3_column_int(stmt, 0);
    }
    rc = sqlite3_reset(stmt);

    std::string outputs;
    MarkerParam markerParam = getBgenMarkerParam(h_bgen, outputs);
    LOGGER << outputs << std::endl;
    if(markerParam.rawCountSNP != n_variants_total_index){
        LOGGER.e(0, "bad index file, the indexed SNPs are different from those in the bgen file."
                "\nTry to regenerate the index by " + prompt_index + ".");
    }
    markerParams.push_back(markerParam);

    // load the index from bgi
    const char *sql_count = "SELECT count(*) FROM Variant WHERE number_of_alleles=2";
    const char *sql = "SELECT chromosome,position,rsid,allele1,allele2,file_start_position,size_in_bytes FROM Variant WHERE number_of_alleles=2";

    // count the alleles
    rc = sqlite3_prepare_v2(db, sql_count, -1, &stmt, NULL);
    if(rc != SQLITE_OK){
        LOGGER.e(0, "bad index file: " + std::string(sqlite3_errmsg(db)) +
                "\nTry to regenerate the index by " + prompt_index + ".");
    }
    int n_variants = 0;
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
        n_variants = sqlite3_column_int(stmt, 0);
    }
    if(rc != SQLITE_DONE){
       LOGGER.e(0, "bad index file: " + std::string(sqlite3_errmsg(db)) +
                "\nTry to regenerate the index by " + prompt_index + ".");
    }
    rc = sqlite3_reset(stmt);
    if(n_variants == 0){
        LOGGER.w(0, "No biallelic variants.");
        return;
    }
    int cur_total_num_variants = chr.size() + n_variants;
    chr.reserve(cur_total_num_variants);
    name.reserve(cur_total_num_variants);
    gd.reserve(cur_total_num_variants);
    pd.reserve(cur_total_num_variants);
    a1.reserve(cur_total_num_variants);
    a2.reserve(cur_total_num_variants);
    byte_start.reserve(cur_total_num_variants);
    byte_size.reserve(cur_total_num_variants);
    LOGGER.i(0, to_string(n_variants) + " SNPs to be included from bgen index file.");

    // retrieve each variants
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if(rc != SQLITE_OK){
        LOGGER.e(0, "bad index file: " + std::string(sqlite3_errmsg(db)) +
                "\nTry to regenerate the index by " + prompt_index + ".");
    }

    int count_chr_error = 0;
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
        std::string snp_chr = RCSTR(sqlite3_column_text(stmt, 0));
        bool keep_snp = isAllowedChrLabel(snp_chr);
        if(!keep_snp) count_chr_error++;
        if(keep_snp){
            chr.push_back(snp_chr);
            name.push_back(RCSTR(sqlite3_column_text(stmt, 2)));
            gd.push_back(0);
            pd.push_back(sqlite3_column_int(stmt, 1));
            std::string snp_a1 = RCSTR(sqlite3_column_text(stmt, 3));
            std::string snp_a2 = RCSTR(sqlite3_column_text(stmt, 4));
            std::transform(snp_a1.begin(), snp_a1.end(), snp_a1.begin(), toupper);
            std::transform(snp_a2.begin(), snp_a2.end(), snp_a2.begin(), toupper);
            a1.push_back(snp_a1);
            a2.push_back(snp_a2);

            byte_start.push_back(sqlite3_column_int64(stmt, 5));
            uint64_t cur_size = sqlite3_column_int64(stmt, 6);
            byte_size.push_back(cur_size);
            if(maxGeno1ByteSize < cur_size){
                maxGeno1ByteSize = cur_size;
            }
        }
    }

    if (rc != SQLITE_DONE) {
        LOGGER.e(0, std::string(sqlite3_errmsg(db)) + "\nbad index file. You may regenerate it by " + prompt_index + ".");
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    num_marker = cur_total_num_variants - count_chr_error;
    int num_var_added = n_variants - count_chr_error;

    // check the first and last SNPs
    if(num_var_added > 0){
        int index1 = chr.size() - num_var_added;
        int index2 = index1 + num_var_added - 1;
        uint64_t pos1 = byte_start[index1];
        uint64_t pos2 = byte_start[index2];
        MarkerInfo m1 = extractBgenMarkerInfo(h_bgen, pos1);
        MarkerInfo m2 = extractBgenMarkerInfo(h_bgen, pos2);
        std::vector<string> allele1 = {a1[index1], a2[index1]};
        std::vector<string> allele2 = {a1[index2], a2[index2]};
        if(m1.chr != chr[index1] || m1.name != name[index1] || m1.pd != pd[index1] || m1.alleles != allele1){
            LOGGER.e(0, "the first variant in the bgen file is not consistent with that in the index file."
                "\nTry to regenerate the index by " + prompt_index + ".");
        }

        if(m2.chr != chr[index2] || m2.name != name[index2] || m2.pd != pd[index2] || m2.alleles != allele2){
            LOGGER.e(0, "the variants in bgen file aren't consistent with those in the index file."
                "\nTry to regenerate the index by " + prompt_index + ".");
        }
    }

    fclose(h_bgen);

    if(count_chr_error > 0){
        LOGGER << count_chr_error << " SNPs excluded due to filtering of chromosomes. " << std::endl;
    }

    LOGGER << "Total SNPs included: " << num_var_added  << "/" <<  num_marker << "." << std::endl;
 
}
#endif

uint64_t Marker::getMaxGenoMarkerUptrSize(){
    int suptr = sizeof(uintptr_t);
    return (maxGeno1ByteSize + suptr - 1) / suptr; 
}

MarkerInfo Marker::extractBgenMarkerInfo(FILE *h_bgen, uint64_t &pos){
    fseek(h_bgen, pos, SEEK_SET);
    auto Lid = read1Byte<uint16_t>(h_bgen);
    fseek(h_bgen, Lid, SEEK_CUR);

    auto len_rs = read1Byte<uint16_t>(h_bgen);
    unique_ptr<char[]> rsid(new char[len_rs + 1]());
    readBytes(h_bgen, len_rs, rsid.get());

    auto len_chr = read1Byte<uint16_t>(h_bgen);
    unique_ptr<char[]> snp_chr(new char[len_chr + 1]());
    
    readBytes(h_bgen, len_chr, snp_chr.get());

    std::string chr(snp_chr.get());

    auto snp_pos = read1Byte<uint32_t>(h_bgen);

    auto n_alleles = read1Byte<uint16_t>(h_bgen);
    std::vector<string> alleles;
    for(int i = 0; i < n_alleles; i++){
        auto len_a1 = read1Byte<uint32_t>(h_bgen);
        unique_ptr<char[]> snp_a1(new char[len_a1 + 1]());
        readBytes(h_bgen, len_a1, snp_a1.get());
        std::transform(snp_a1.get(), snp_a1.get() + len_a1, snp_a1.get(), toupper);
        alleles.push_back(snp_a1.get());
    }

    uint64_t snp_start = ftell(h_bgen);

    //auto len_comp = read1Byte<uint32_t>(h_bgen); 
    //fseek(h_bgen, len_comp, SEEK_CUR);

    MarkerInfo markerInfo;
    markerInfo.chr = chr;
    markerInfo.name = std::string(rsid.get());
    markerInfo.pd = snp_pos;
    markerInfo.alleles = alleles;

    pos = snp_start; // set the position easy for seek

    return markerInfo;
}
    

MarkerParam Marker::getMarkerParams(int part_num){
    if(part_num < markerParams.size()){
        return markerParams[part_num];
    }else{
        LOGGER.e(0, "the marker parameters are out of range.");
        return markerParams[0]; // dummy return
    }
}

#ifdef BGEN_SUPPORT
void Marker::read_bgen(string bgen_file){
    LOGGER << "Extracting biallelic SNPs from bgen [" << bgen_file << "]..." << std::endl;
    FILE* h_bgen = fopen(bgen_file.c_str(), "rb");
    if(!h_bgen){
        LOGGER.e(0, "can't open the bgen file to read");
    }
    uint32_t start_byte = read1Byte<uint32_t>(h_bgen);
    uint32_t start_data_block = start_byte + 4;

    uint32_t len_header = read1Byte<uint32_t>(h_bgen);

    uint32_t n_variants = read1Byte<uint32_t>(h_bgen);
    LOGGER << n_variants << " SNPs, ";

    auto n_sample = read1Byte<uint32_t>(h_bgen);
    LOGGER << n_sample << " samples in the bgen file." << std::endl;

    char magic[5];
    readBytes<char>(h_bgen, 4, magic);

    int32_t skip_byte = (int32_t)len_header - 20;
    if(skip_byte > 0){
        fseek(h_bgen, skip_byte, SEEK_CUR); 
    }else if(skip_byte < 0){
        LOGGER.e(0, "unusual header length - it might be an invalid bgen file.");
    }

    uint32_t flags = read1Byte<uint32_t>(h_bgen);
    int compress_block = flags & 3; // first 2 bit
    int layout = (flags >> 2) & 15; //2-5 bit
    int has_sample = (flags >> 31) & 1;

    LOGGER << "bgen version ";
    switch(layout){
        case 1:
            LOGGER << "1.1, ";
            break;
        case 2:
            LOGGER << "1.2, ";
            break;
        default:
            LOGGER << "unkown, ";
    }
    switch(compress_block){
        case 0:
            LOGGER << "no compress";
            break;
        case 1:
            LOGGER << "compressed by zlib";
            break;
        case 2:
            LOGGER << "compressed by zstd";
            break;
        default:
            LOGGER << "unknown format";
    }
    LOGGER << "." << std::endl;

    if(layout != 2){
        LOGGER.e(0, "GCTA only supports bgen version 1.2 and 1.3. Use QCTOOL to convert files with < v1.2 to a newer version.");
    }
               
    if(compress_block != 1){
        LOGGER.e(0, "non-zlid compressed files are not supported currently");
    }

    LOGGER << "Looking for biallelic alleles..." << std::endl;
    fseek(h_bgen, start_data_block, SEEK_SET);
    uint32_t count_chr_error = 0, count_multi_alleles = 0;

    for(int index = 0; index < n_variants; index++){
        auto Lid = read1Byte<uint16_t>(h_bgen);
        fseek(h_bgen, Lid, SEEK_CUR);

        auto len_rs = read1Byte<uint16_t>(h_bgen);
		unique_ptr<char[]> rsid(new char[len_rs + 1]());
        readBytes(h_bgen, len_rs, rsid.get());

        auto len_chr = read1Byte<uint16_t>(h_bgen);
		unique_ptr<char[]> snp_chr(new char[len_chr + 1]());
        readBytes(h_bgen, len_chr, snp_chr.get());
        std::string chr_str = std::string(snp_chr.get());
        bool keep_snp = isAllowedChrLabel(chr_str);
        if(!keep_snp) count_chr_error++;


        auto snp_pos = read1Byte<uint32_t>(h_bgen);

        auto n_alleles = read1Byte<uint16_t>(h_bgen);
        if(n_alleles != 2){
            count_multi_alleles++;
            keep_snp = false;
        }

        auto len_a1 = read1Byte<uint32_t>(h_bgen);
		unique_ptr<char[]> snp_a1(new char[len_a1 + 1]());
        readBytes(h_bgen, len_a1, snp_a1.get());

        auto len_a2 = read1Byte<uint32_t>(h_bgen);
		unique_ptr<char[]> snp_a2(new char[len_a2 + 1]());
        readBytes(h_bgen, len_a2, snp_a2.get());

        uint64_t snp_start = ftell(h_bgen);

        for(int index_allele = 2; index_allele < n_alleles; index_allele++){
            auto len_allele = read1Byte<uint32_t>(h_bgen);
            fseek(h_bgen, len_allele, SEEK_CUR);
        }

        auto len_comp = read1Byte<uint32_t>(h_bgen); 
        fseek(h_bgen, len_comp, SEEK_CUR);
        
        if(keep_snp){
            chr.push_back(chr_str);
            name.push_back(rsid.get());
            gd.push_back(0);
            pd.push_back(snp_pos);
            std::transform(snp_a1.get(), snp_a1.get() + len_a1, snp_a1.get(), toupper);
            std::transform(snp_a2.get(), snp_a2.get() + len_a2, snp_a2.get(), toupper);
            a1.push_back(snp_a1.get());
            a2.push_back(snp_a2.get());
            byte_start.push_back(snp_start);
        }
    }
    num_marker = name.size();
    index_extract.resize(name.size());
    std::iota(index_extract.begin(), index_extract.end(), 0);
    num_extract = index_extract.size();
    LOGGER.i(0, to_string(num_marker) + " SNPs to be included from bgen file.");
    LOGGER << count_chr_error << " SNPs excluded due to filtering of chromosomes." << std::endl;
    LOGGER << count_multi_alleles << " SNPs excluded due to multiple alleles." << std::endl; 
    fclose(h_bgen);
    if(num_extract == 0){
        LOGGER.e(0, "no SNP remains for further analysis.");
    }
}
#endif

uint32_t Marker::count_raw(int part) {
    if(part == -1){
        return num_marker;
    }else{
        return raw_limits[part];
    }
}

uint32_t Marker::count_extract(){
    return index_extract.size();
}

std::vector<uint32_t>& Marker::get_extract_index(){ // raw index
    return this->index_extract;
}

std::vector<uint32_t> Marker::get_extract_index_autosome(){ // returns the index in keep list not raw
    std::vector<uint32_t> auto_index;
    const size_t n = index_extract.size();
    if (homogametic_chrs.empty() && heterogametic_chrs.empty()) {
        auto_index.resize(n);
        std::iota(auto_index.begin(), auto_index.end(), static_cast<uint32_t>(0));
        return auto_index;
    }
    auto_index.reserve(n);
    std::string last_chr;
    bool has_last = false;
    bool last_is_auto = false;
    for (size_t i = 0; i < n; ++i) {
        const std::string& c = chr[index_extract[i]];
        bool is_auto;
        if (has_last && c == last_chr) {
            is_auto = last_is_auto;
        } else {
            last_chr = c;
            has_last = true;
            std::string norm;
            normalizeChrLabelInto(c, norm);
            last_is_auto = (homogametic_chrs.count(norm) == 0 && heterogametic_chrs.count(norm) == 0);
            is_auto = last_is_auto;
        }
        if (is_auto) auto_index.push_back(static_cast<uint32_t>(i));
    }
    return auto_index;
}

std::vector<uint32_t> Marker::get_extract_index_X(){
    std::vector<uint32_t> x_index;
    if (homogametic_chrs.empty()) {
        return x_index;
    }
    const size_t n = index_extract.size();
    x_index.reserve(n / 16 + 16);
    std::string last_chr;
    bool has_last = false;
    bool last_is_x = false;
    for (size_t i = 0; i < n; ++i) {
        const std::string& c = chr[index_extract[i]];
        bool is_x;
        if (has_last && c == last_chr) {
            is_x = last_is_x;
        } else {
            last_chr = c;
            has_last = true;
            last_is_x = isHomogameticChr(c);
            is_x = last_is_x;
        }
        if (is_x) x_index.push_back(static_cast<uint32_t>(i));
    }
    return x_index;
}


void Marker::extract_marker(std::vector<string> markers, bool isExtract) {
   std::vector<uint32_t> ori_index, marker_index;
   size_t numOriMarker = markers.size();
   removeDuplicateSort(markers);
   int numDup = numOriMarker - markers.size();
   if(numDup != 0) LOGGER.w(0, to_string(numDup) + " duplicated SNPs were ignored in the list." );

   vector_commonIndex_sorted1(name, markers, ori_index, marker_index);
   std::vector<uint32_t> remain_index;
   if(isExtract){
       std::set_intersection(index_extract.begin(), index_extract.end(),
                             ori_index.begin(), ori_index.end(),
                             std::back_inserter(remain_index));
   }else{
       std::set_difference(index_extract.begin(), index_extract.end(),
                           ori_index.begin(), ori_index.end(),
                           std::back_inserter(remain_index));
   }

   index_extract = remain_index;
   num_extract = index_extract.size();

   if(num_extract == 0){
       LOGGER.e(0, "no SNP remains.");
   }

   LOGGER.i(0, std::string("After ") + (isExtract? "extracting" : "excluding") +  " SNP, " +  to_string(num_extract) + " SNPs remain.");
}

void Marker::reset_exclude(){
    index_exclude.clear();
    if (index_extract.size() == num_marker) {
        num_exclude = 0;
        return;
    }
    index_exclude.reserve(num_marker - index_extract.size());
    uint32_t expected = 0;
    for (uint32_t idx : index_extract) {
        while (expected < idx) {
            index_exclude.push_back(expected++);
        }
        expected = idx + 1;
    }
    while (expected < num_marker) {
        index_exclude.push_back(expected++);
    }
    num_exclude = static_cast<uint32_t>(index_exclude.size());
}

// didn't check whether in extracted list or not;
void Marker::keep_raw_index(const std::vector<uint32_t>& keep_index) {
    index_extract.resize(keep_index.size());
    index_extract = keep_index;

    num_extract = index_extract.size();
    if(num_extract == 0){
        LOGGER.e(0, "no SNP remains.");
    }
    reset_exclude();
}

void Marker::keep_extracted_index(const std::vector<uint32_t>& keep_index) {
    std::vector<uint32_t> raw_index;
    raw_index.reserve(keep_index.size());
    for(auto index : keep_index){
        raw_index.push_back(index_extract[index]);
    }
    std::sort(raw_index.begin(), raw_index.end());
    keep_raw_index(raw_index);
}

string Marker::get_marker(int rawindex){
    std::string return_string = chr[rawindex] + "\t" + name[rawindex] + "\t" + 
        std::to_string(pd[rawindex]) + "\t";
    return return_string + a1[rawindex] + "\t" + a2[rawindex];
}

string Marker::getMarkerStrExtract(int extractindex){
    return get_marker(getRawIndex(extractindex));
}
 

bool Marker::isInExtract(uint32_t index) {
    if(index_extract.size() > index_exclude.size()){
        return !std::binary_search(index_exclude.begin(), index_exclude.end(), index);
    }else{
        return std::binary_search(index_extract.begin(), index_extract.end(), index);
    }
}

uint32_t Marker::getRawIndex(uint32_t extractedIndex){
    return index_extract[extractedIndex];
}

//TODO support multiple column SNP list, currently only take the first SNP list
// If multiple column provided, it will miss the correct SNP name.
std::vector<string> Marker::read_snplist(string snplist_file) {
    gcta_text_io::FastLineReader reader;
    if (!reader.open(snplist_file)) {
        LOGGER.e(0, "cannot open the file [" + snplist_file + "] to read");
    }
    std::vector<string> snplist;
    std::vector<std::string_view> toks;
    toks.reserve(4);

    int line_number = 0;
    size_t last_length = 0;
    bool reserved = false;
    std::string_view line;
    while (reader.next_line(line)) {
        if (!reserved) {
            reserved = true;
            if (reader.file_size() > 0) {
                const size_t avg_line = std::max<size_t>(line.size() + 1, 4);
                snplist.reserve(static_cast<size_t>(reader.file_size() / avg_line) + 64);
            }
        }
        line_number++;
        gcta_text_io::split_ws_exact(line, toks);
        if (toks.empty()) {
            LOGGER.e(0, "the SNP list file [" + snplist_file + "], line " + to_string(line_number) +
                        " has elements less than 1");
        }
        if (line_number > 1 && toks.size() != last_length) {
            LOGGER.w(0, "the SNP list file [" + snplist_file + "], line " + to_string(line_number) +
                        " has different elements");
        }
        snplist.emplace_back(toks[0]);
        last_length = toks.size();
    }
    return snplist;
}
