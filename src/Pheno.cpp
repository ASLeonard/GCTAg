/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   New implementation: holds phenotype information in plink format

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/

#include "Pheno.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include "constants.hpp"
#include "Logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <charconv>
#include <cstdio>
#include "utils.hpp"
#include <cstring>
#include <boost/algorithm/string.hpp>
#include <boost/crc.hpp>
#include <set>
#include "OptionIO.h"

using std::to_string;

int8_t Pheno::heterogametic_sex_code = 1;
int8_t Pheno::homogametic_sex_code = 2;

const uintptr_t k1LU = (uintptr_t)1;

int8_t Pheno::normalizeSexCode(int raw_code){
    if(raw_code == heterogametic_sex_code){
        return 1;
    }
    if(raw_code == homogametic_sex_code){
        return 2;
    }
    return 0;
}

void Pheno::loadSexCodeMappingFromFile(const string& file_path){
    std::ifstream fin(file_path.c_str());
    if(!fin.good()){
        LOGGER.e(0, "can't open sex chromosome file [" + file_path + "]");
    }

    bool found_homog = false;
    bool found_heterog = false;
    int homog_code = -1;
    int heterog_code = -1;

    string line;
    while(std::getline(fin, line)){
        if (!line.empty() && line.back() == '\r') line.pop_back();
        boost::trim(line);
        if(line.empty() || line[0] == '#'){
            continue;
        }

        vector<string> toks;
        boost::split(toks, line, boost::is_any_of("\t "), boost::token_compress_on);
        if(toks.size() < 3){
            LOGGER.e(0, "invalid --sex-chr-file format. Require at least 3 columns per non-comment line: <HOMOGAMETIC|HETEROGAMETIC> <PLINK sex code 1|2> <chromosome labels...>.");
        }

        string key = toks[0];
        boost::to_upper(key);

        int parsed_code = -1;
        try{
            parsed_code = std::stoi(toks[1]);
        }catch(std::invalid_argument&){
            LOGGER.e(0, "invalid PLINK sex code [" + toks[1] + "] in --sex-chr-file. Expected 1 or 2.");
        }catch(std::out_of_range&){
            LOGGER.e(0, "PLINK sex code [" + toks[1] + "] is out of range in --sex-chr-file.");
        }
        if(parsed_code != 1 && parsed_code != 2){
            LOGGER.e(0, "invalid PLINK sex code [" + toks[1] + "] in --sex-chr-file. Expected 1 or 2.");
        }

        if(key == "HOMOGAMETIC" || key == "HOMO" || key == "XX"){
            homog_code = parsed_code;
            found_homog = true;
        }else if(key == "HETEROGAMETIC" || key == "HETERO" || key == "XY"){
            heterog_code = parsed_code;
            found_heterog = true;
        }else{
            LOGGER.e(0, "invalid --sex-chr-file line prefix [" + toks[0] + "]. Use HOMOGAMETIC/HETEROGAMETIC.");
        }
    }

    if(!found_homog || !found_heterog){
        LOGGER.e(0, "invalid --sex-chr-file format. Both HOMOGAMETIC and HETEROGAMETIC lines are required.");
    }
    if(homog_code == heterog_code){
        LOGGER.e(0, "invalid --sex-chr-file format. HOMOGAMETIC and HETEROGAMETIC must use different PLINK sex codes.");
    }
    if(heterog_code != 1 || homog_code != 2){
        LOGGER.e(0, "invalid --sex-chr-file format. Fixed PLINK coding is required: HETEROGAMETIC=1 and HOMOGAMETIC=2.");
    }
}

Pheno::Pheno(const gcta_cli::GenoInputOptions& geno,
             const gcta_cli::SampleFilterOptions& sample_filter,
             const gcta_cli::PhenoCovarOptions* pheno_covar,
             bool filter_sex_flag) {
    heterogametic_sex_code = 1;
    homogametic_sex_code = 2;
    if (!sample_filter.sex_chr_file.empty()) {
        loadSexCodeMappingFromFile(sample_filter.sex_chr_file);
    }

    bool has_pheno = false;

    const string fam_file = !geno.fam.empty() ? geno.fam
        : (!geno.bfile.empty() ? geno.bfile + ".fam"
        : (!geno.bpfile.empty() ? geno.bpfile + ".fam" : ""));
    if (!fam_file.empty()) {
        requireFileReadable(geno.fam.empty() ? "--bfile/--bpfile" : "--fam", fam_file);
        this->read_fam(fam_file);
        has_pheno = true;
    }

    if (!geno.sample.empty()) {
        requireFileReadable("--sample", geno.sample);
        this->read_sample(geno.sample);
        has_pheno = true;
    }

    if (!geno.pfile.empty()) {
        const string psam_file = geno.pfile + ".psam";
        requireFileReadable("--pfile", psam_file);
        read_psam(psam_file);
        has_pheno = true;
    }

    if (!geno.mpfile.empty()) {
        read_checkMPSample(expandFileList(geno.mpfile, ".psam"));
        has_pheno = true;
    }

    if (!geno.mbfile.empty()) {
        read_checkMPSample(expandFileList(geno.mbfile, ".fam"));
        has_pheno = true;
    }

    if (!geno.mbpfile.empty()) {
        read_checkMPSample(expandFileList(geno.mbpfile, ".fam"));
        has_pheno = true;
    }

    if (!has_pheno) {
        LOGGER.e(0, "no phenotype file found.");
    }

    if (!sample_filter.keep.empty()) {
        requireFileReadable("--keep", sample_filter.keep);
        vector<string> keep_subjects = read_sublist(sample_filter.keep);
        LOGGER << "Get " << keep_subjects.size() << " samples from list [" << sample_filter.keep << "]." << std::endl;
        set_keep(keep_subjects, mark, index_keep, true);
    }

    if (!sample_filter.remove.empty()) {
        requireFileReadable("--remove", sample_filter.remove);
        vector<string> remove_subjects = read_sublist(sample_filter.remove);
        LOGGER << "Get " << remove_subjects.size() << " samples from list [" << sample_filter.remove << "]." << std::endl;
        set_keep(remove_subjects, mark, index_keep, false);
    }

    if (pheno_covar != nullptr && !pheno_covar->pheno.empty()) {
        requireFileReadable("--pheno", pheno_covar->pheno);
        vector<vector<double>> phenos;
        LOGGER.i(0, "Reading phenotype data from [" + pheno_covar->pheno + "]...");
        vector<string> pheno_subjects = read_sublist(pheno_covar->pheno, &phenos);
        if (hasVectorDuplicate(pheno_subjects)) {
            LOGGER.e(0, " duplicated IDs found in the phenotype data.");
        }

        int cur_pheno = pheno_covar->mpheno.value_or(1);
        if (cur_pheno <= 0 || cur_pheno > static_cast<int>(phenos.size())) {
            LOGGER.e(0, "the value specified for --mpheno can't be less than 0 or larger than the total number of columns in .pheno file.");
        }

        cur_pheno -= 1;
        update_pheno(pheno_subjects, phenos[cur_pheno]);

        // phenos is 0-indexed over data columns (col 0 = first data col after FID/IID).
        // Weight is expected in the column immediately after the selected pheno column.
        if (pheno_covar->weighted_phenotypes) {
            int weight_col = cur_pheno + 1;
            if (weight_col < static_cast<int>(phenos.size())) {
                update_weight(pheno_subjects, phenos[weight_col]);
                LOGGER.i(0, "Sample weights detected in phenotype file.");
            }
        }

        LOGGER.i(0, to_string(index_keep.size()) + " overlapping individuals with non-missing data to be included from the phenotype file.");
    }

    if (!geno.update_sex.empty()) {
        requireFileReadable("--update-sex", geno.update_sex);
        vector<vector<double>> phenos;
        LOGGER.i(0, "Reading gender information from [" + geno.update_sex + "]...");
        vector<string> subjects = read_sublist(geno.update_sex, &phenos);
        if (hasVectorDuplicate(subjects)) {
            LOGGER.e(0, "duplicated IDs in the gender information.");
        }
        vector<double> sex_info = phenos[0];
        update_sex(subjects, sex_info);
        LOGGER.i(0, to_string(index_keep.size()) + " individuals with valid sex information to be included from the phenotype file.");
    }

    if (filter_sex_flag) {
        filter_sex();
    }

    reinit();
        LOGGER << index_keep.size() << " individuals to be included. "
            << index_keep_heterogametic.size() << " sex-code-1, "
            << index_keep_sex.size() - index_keep_heterogametic.size() << " sex-code-2, "
            << index_keep.size() - index_keep_sex.size() << " unknown." << std::endl;

    /*
    vector<string> filter_marker;
    for(int i = 0; i < index_keep.size(); i++){
        filter_marker.push_back(mark[index_keep[i]]);
    }

    if(hasVectorDuplicate(filter_marker)){
        LOGGER.e(0, "duplicated sample IDs.");
    }
    */

}

void Pheno::filter_keep_index(vector<uint32_t>& k_index){
    if(k_index.size() == index_keep.size()){return;}
    vector<uint32_t> index_keep2(k_index.size(), 0);
    std::transform(k_index.begin(), k_index.end(), index_keep2.begin(), 
            [this](size_t pos){return this->index_keep[pos];});
    index_keep = index_keep2;

    reinit();
}

void Pheno::filter_sex(){
    vector<uint32_t> new_index;
    vector<uint32_t> heterogametic_index;
    new_index.reserve(index_keep.size());
    uint32_t valid_index = 0;
    for(int i = 0; i < index_keep.size(); i++){
        uint32_t index = index_keep[i];
        int8_t sex_item = sex[index];
        switch(sex_item){
            case 1:
                heterogametic_index.push_back(valid_index);
                new_index.push_back(index);
                valid_index++;
                break;
            case 2:
                new_index.push_back(index);
                valid_index++;
                break;
        }
    }

    new_index.shrink_to_fit();
    index_keep = new_index;
    index_keep_heterogametic = heterogametic_index;
    LOGGER.i(0, to_string(new_index.size()) + " individuals have gender information.");
}


void Pheno::reinit(){
    reinit_rm(index_keep, index_rm, num_ind);
    num_keep = index_keep.size();
    num_rm = index_rm.size();
    if(num_keep == 0){
        LOGGER.e(0, "0 individual remains for further analysis.");
    }
    //init_mask_block();
    index_keep_heterogametic.clear();
    index_keep_sex.clear();
    index_keep_heterogametic_extract.clear();

    uint32_t curSexIndex = 0;
    for(int i = 0; i < num_keep; i++){
        uint32_t curIndex = index_keep[i];
        int8_t curSex = sex[curIndex];
        if(curSex == 1){
            index_keep_heterogametic.push_back(curIndex);
            //index_keep_heterogametic_extract.push_back(curSexIndex);
            index_keep_heterogametic_extract.push_back(i);
            index_keep_sex.push_back(curIndex);
            curSexIndex++;
        }else if(curSex == 2){
            index_keep_sex.push_back(curIndex);
            curSexIndex++;
        }
    }
    
}


namespace gcta_pheno_detail {

bool read_file_bytes(const std::string& path, std::vector<char>& out) {
    out.clear();
    const int fd = ::open(path.c_str(), O_RDONLY
#ifdef O_CLOEXEC
                          | O_CLOEXEC
#endif
    );
    if (fd < 0) return false;
    struct stat st{};
    size_t hint = 0;
    if (::fstat(fd, &st) == 0 && st.st_size > 0) {
        hint = static_cast<size_t>(st.st_size);
        out.resize(hint);
    }
    constexpr size_t kChunk = 4ULL << 20;
    size_t total = 0;
    while (total < hint) {
        const size_t want = std::min(kChunk, hint - total);
        const ssize_t r = ::read(fd, out.data() + total, want);
        if (r > 0) {
            total += static_cast<size_t>(r);
        } else if (r == 0) {
            out.resize(total);
            ::close(fd);
            return true;
        } else if (errno == EINTR) {
            continue;
        } else {
            ::close(fd);
            out.clear();
            return false;
        }
    }
    // Probe EOF without extra heap allocation on regular files; grow cleanly if file grew or size was 0.
    char probe[4096];
    std::vector<char> chunk;
    while (true) {
        char* dst = chunk.empty() ? probe : chunk.data();
        const size_t cap = chunk.empty() ? sizeof(probe) : chunk.size();
        const ssize_t r = ::read(fd, dst, cap);
        if (r > 0) {
            out.insert(out.end(), dst, dst + r);
            if (chunk.empty()) chunk.resize(kChunk);
        } else if (r == 0) {
            break;
        } else if (errno == EINTR) {
            continue;
        } else {
            ::close(fd);
            out.clear();
            return false;
        }
    }
    ::close(fd);
    return true;
}

bool files_byte_equal(const std::vector<char>& ref_bytes, const std::string& other_path,
                      std::vector<char>& buf) {
    const int fd = ::open(other_path.c_str(), O_RDONLY
#ifdef O_CLOEXEC
                          | O_CLOEXEC
#endif
    );
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0 || static_cast<size_t>(st.st_size) != ref_bytes.size()) {
        ::close(fd);
        return false;
    }
    constexpr size_t kChunk = 4ULL << 20;
    const size_t target = std::min(kChunk, std::max<size_t>(ref_bytes.size(), 1));
    if (buf.size() < target) buf.resize(target);
    size_t offset = 0;
    while (offset < ref_bytes.size()) {
        const size_t want = std::min(target, ref_bytes.size() - offset);
        const ssize_t r = ::read(fd, buf.data(), want);
        if (r < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            return false;
        }
        if (r == 0) {
            ::close(fd);
            return false;
        }
        if (std::memcmp(buf.data(), ref_bytes.data() + offset, static_cast<size_t>(r)) != 0) {
            ::close(fd);
            return false;
        }
        offset += static_cast<size_t>(r);
    }
    // Confirm EOF
    char extra = 0;
    while (true) {
        const ssize_t r = ::read(fd, &extra, 1);
        if (r < 0 && errno == EINTR) continue;
        ::close(fd);
        return r == 0;
    }
}

struct MemLineReader {
    const char* data = nullptr;
    size_t size = 0;
    size_t pos = 0;

    uint64_t file_size() const noexcept { return static_cast<uint64_t>(size); }

    bool next_line(std::string_view& out) noexcept {
        if (pos >= size) return false;
        const char* start = data + pos;
        const size_t avail = size - pos;
        const void* nl = std::memchr(start, '\n', avail);
        size_t line_len = nl ? static_cast<size_t>(static_cast<const char*>(nl) - start) : avail;
        pos += nl ? (line_len + 1) : line_len;
        if (line_len > 0 && start[line_len - 1] == '\r') --line_len;
        out = std::string_view(start, line_len);
        return true;
    }
};

template <typename LineReader>
void parse_psam_reader(LineReader& reader, const std::string& psam_file,
                       int8_t sex_code_1, int8_t sex_code_2,
                       vector<string>& fid, vector<string>& pid, vector<string>& mark,
                       vector<string>& fa_id, vector<string>& mo_id,
                       vector<int8_t>& sex, vector<double>& pheno,
                       vector<uint32_t>& index_keep, int& num_ind, int& num_keep) {
    std::string_view line;
    std::string preLine;
    int curReadLine = 0;
    bool found_data = false;
    while (reader.next_line(line)) {
        curReadLine++;
        const size_t fnw = line.find_first_not_of("\r\t ");
        if (fnw != std::string_view::npos && line[fnw] == '#') {
            preLine.assign(line);
            continue;
        }
        if (fnw == std::string_view::npos) {
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
    if (ncol < 2) {
        LOGGER.e(0, "the number of fields in [" + psam_file + "] is less than 2.");
    }

    int iFID = -1, iIID = -1, iPAT = -1, iMAT = -1, iSEX = -1;
    if (head_toks.empty()) {
        if (ncol == 6) {
            iFID = 0;
            iIID = 1;
            iPAT = 2;
            iMAT = 3;
            iSEX = 4;
        } else {
            LOGGER.e(0, "only " + to_string(ncol) + " columns for the FAM file which has at least 6 columns.");
        }
    } else {
        auto find_col = [&](std::string_view key) -> int {
            for (int c = 0; c < ncol; ++c) {
                if (head_toks[static_cast<size_t>(c)] == key) return c;
            }
            return -1;
        };
        iFID = find_col("#FID");
        if (iFID >= 0) {
            iIID = find_col("IID");
        } else {
            // if not find FID, then use the IID instead, different from plink
            iIID = find_col("#IID");
            iFID = iIID;
        }
        if (iIID < 0) LOGGER.e(0, "can't find IID or #IID in the headers, which is invalid for a PSAM file.");

        iSEX = find_col("SEX");
        if (iSEX < 0) LOGGER.e(0, "the column for gender information is required in the PSAM file.");
        iPAT = find_col("PAT");
        iMAT = find_col("MAT");
    }

    if (reader.file_size() > 0) {
        const size_t avg_line = std::max<size_t>(line.size() + 1, 8);
        const size_t est_rows = static_cast<size_t>(reader.file_size() / avg_line) + 64;
        fid.reserve(est_rows);
        pid.reserve(est_rows);
        mark.reserve(est_rows);
        fa_id.reserve(est_rows);
        mo_id.reserve(est_rows);
        sex.reserve(est_rows);
        pheno.reserve(est_rows);
    }

    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    int line_number = nHeader;
    auto process_row = [&]() {
        line_number++;
        if (static_cast<int>(toks.size()) < ncol) {
            LOGGER.e(0, "the file [" + psam_file + "] contains different number of elements in line " + to_string(line_number) + ".");
        } else if (static_cast<int>(toks.size()) > ncol) {
            LOGGER.w(0, "the file [" + psam_file + "] contains extra number of elements in line " + to_string(line_number) + ".");
        }
        const std::string_view sv_fid = toks[static_cast<size_t>(iFID)];
        const std::string_view sv_iid = toks[static_cast<size_t>(iIID)];
        const std::string_view sv_sex = toks[static_cast<size_t>(iSEX)];
        fid.emplace_back(sv_fid);
        pid.emplace_back(sv_iid);
        std::string m;
        m.reserve(sv_fid.size() + 1 + sv_iid.size());
        m.append(sv_fid);
        m.push_back('\t');
        m.append(sv_iid);
        mark.push_back(std::move(m));

        int8_t sex_code = 0;
        if (sv_sex.size() == 1) {
            if (sv_sex[0] == '1') sex_code = sex_code_1;
            else if (sv_sex[0] == '2') sex_code = sex_code_2;
        }
        sex.push_back(sex_code);
        if (iPAT != -1) {
            fa_id.emplace_back(toks[static_cast<size_t>(iPAT)]);
        } else {
            fa_id.emplace_back("0");
        }
        if (iMAT != -1) {
            mo_id.emplace_back(toks[static_cast<size_t>(iMAT)]);
        } else {
            mo_id.emplace_back("0");
        }
        pheno.push_back(kNaN);
    };

    process_row();
    while (reader.next_line(line)) {
        gcta_text_io::split_ws_exact(line, toks);
        process_row();
    }

    const size_t nrows = fid.size();
    index_keep.resize(nrows);
    std::iota(index_keep.begin(), index_keep.end(), static_cast<uint32_t>(0));

    num_ind = static_cast<int>(nrows);
    num_keep = static_cast<int>(nrows);
    LOGGER.i(0, to_string(num_ind) + " individuals to be included from the sample file.");
}

// Token-level fallback comparison when raw bytes differ (e.g., CRLF vs LF or space vs tab).
// Mirrors original readTxtList comparison semantics without throwing inside OpenMP.
bool compare_sample_files_tokens(const std::vector<char>& ref_bytes, const std::string& other_path) {
    gcta_text_io::FastLineReader other_reader;
    if (!other_reader.open(other_path)) return false;

    MemLineReader ref_reader{ref_bytes.data(), ref_bytes.size(), 0};

    auto skip_to_first_data = [](auto& rdr, std::string& header_line, std::string_view& first_data) -> bool {
        header_line.clear();
        std::string_view line;
        while (rdr.next_line(line)) {
            const size_t fnw = line.find_first_not_of("\r\t ");
            if (fnw != std::string_view::npos && line[fnw] == '#') {
                header_line.assign(line);
                continue;
            }
            if (fnw == std::string_view::npos) return false;
            first_data = line;
            return true;
        }
        return false;
    };

    std::string ref_head, oth_head;
    std::string_view ref_line, oth_line;
    if (!skip_to_first_data(ref_reader, ref_head, ref_line)) return false;
    if (!skip_to_first_data(other_reader, oth_head, oth_line)) return false;

    std::vector<std::string_view> ref_toks, oth_toks;
    gcta_text_io::split_ws_exact(ref_line, ref_toks);
    gcta_text_io::split_ws_exact(oth_line, oth_toks);
    const size_t ncol = ref_toks.size();
    if (ncol < 2 || oth_toks.size() != ncol) return false;

    // Compare headers (both empty or exact token equality matching ncol)
    if (ref_head.empty() != oth_head.empty()) return false;
    if (!ref_head.empty()) {
        std::vector<std::string_view> rh, oh;
        gcta_text_io::split_ws_exact(ref_head, rh);
        gcta_text_io::split_ws_exact(oth_head, oh);
        if (rh.size() != ncol || oh.size() != ncol || rh != oh) return false;
    }

    auto rows_equal = [&]() -> bool {
        if (ref_toks.size() < ncol || oth_toks.size() < ncol) return false;
        for (size_t c = 0; c < ncol; ++c) {
            if (ref_toks[c] != oth_toks[c]) return false;
        }
        return true;
    };

    if (!rows_equal()) return false;
    while (true) {
        const bool r_ok = ref_reader.next_line(ref_line);
        const bool o_ok = other_reader.next_line(oth_line);
        if (r_ok != o_ok) return false;
        if (!r_ok) break;
        gcta_text_io::split_ws_exact(ref_line, ref_toks);
        gcta_text_io::split_ws_exact(oth_line, oth_toks);
        if (!rows_equal()) return false;
    }
    return !other_reader.io_error();
}

} // namespace gcta_pheno_detail

// TODO filter the non-number strings other than nan
vector<string> Pheno::read_sublist(string sublist_file, vector<vector<double>> *phenos, vector<int> *keep_row_p) {
    vector<string> subject_list;
    gcta_text_io::FastLineReader reader;
    if (!reader.open(sublist_file)) {
        LOGGER.e(0, "can't read [" + sublist_file + "]");
    }
    vector<int> keep_row;
    const string err_file = "the file [" + sublist_file + "]";

    std::string_view line;
    if (!reader.next_line(line)) {
        LOGGER.e(0, err_file + " is empty.");
    }

    std::vector<std::string_view> toks;
    toks.reserve(16);
    gcta_text_io::split_ws_exact(line, toks);
    int last_length = static_cast<int>(toks.size());
    int large_elements = 0;
    if (phenos) {
        if (last_length < 3) {
            LOGGER.e(0, err_file + " has less than 3 columns, where the first 2 columns should be FID, IID");
        }
        if (keep_row_p) {
            keep_row = *keep_row_p;
            phenos->resize(keep_row.size());
        } else {
            phenos->resize(static_cast<size_t>(last_length - 2));
            keep_row.resize(phenos->size());
            std::iota(keep_row.begin(), keep_row.end(), 0);
        }
        large_elements = keep_row[keep_row.size() - 1] + 1 + 2;
        if (large_elements > last_length) {
            LOGGER.e(0, err_file + " does not have enough columns to read");
        }
    } else {
        if (last_length < 2) {
            LOGGER.e(0, err_file + " has less than 2 columns.");
        }
    }

    if (reader.file_size() > 0) {
        const size_t avg_line = std::max<size_t>(line.size() + 1, 6);
        const size_t est_rows = static_cast<size_t>(reader.file_size() / avg_line) + 64;
        subject_list.reserve(est_rows);
        if (phenos) {
            for (auto& col : *phenos) col.reserve(est_rows);
        }
    }

    int line_number = 0;
    auto process_row = [&]() {
        line_number++;
        const int num_elements = static_cast<int>(toks.size());
        if (num_elements != last_length) {
            string errmsg = err_file + ", line " + to_string(line_number) +
                " has different number of columns.";
            LOGGER.w(0, errmsg);
        }
        if (num_elements < 2) {
            LOGGER.e(0, err_file + ", line " + to_string(line_number) +
                     " has less than 2 columns.");
        }

        const std::string_view f = toks[0];
        const std::string_view p = toks[1];
        std::string id;
        id.reserve(f.size() + 1 + p.size());
        id.append(f);
        id.push_back('\t');
        id.append(p);
        subject_list.push_back(std::move(id));

        if (phenos) {
            if (large_elements > num_elements) {
                LOGGER.e(0, err_file + ", line " + to_string(line_number) +
                         " has not enough elements");
            }
            for (size_t index = 0; index != keep_row.size(); ++index) {
                (*phenos)[index].push_back(
                    gcta_text_io::parse_pheno_double_sv(toks[static_cast<size_t>(keep_row[index] + 2)]));
            }
        }
        last_length = num_elements;
    };

    process_row();
    while (reader.next_line(line)) {
        gcta_text_io::split_ws_exact(line, toks);
        process_row();
    }
    return subject_list;
}

void Pheno::read_checkMPSample(const vector<string>& mfiles){
    if (mfiles.empty()) {
        return;
    }
    if (mfiles.size() == 1) {
        read_psam(mfiles[0]);
        return;
    }

    LOGGER.i(0, "Reading PLINK sample file from [" + mfiles[0] + "]...");
    std::vector<char> ref_bytes;
    if (!gcta_pheno_detail::read_file_bytes(mfiles[0], ref_bytes)) {
        LOGGER.e(0, "can't read [" + mfiles[0] + "].");
    }
    gcta_pheno_detail::MemLineReader ref_reader{ref_bytes.data(), ref_bytes.size(), 0};
    gcta_pheno_detail::parse_psam_reader(
        ref_reader, mfiles[0], normalizeSexCode(1), normalizeSexCode(2),
        fid, pid, mark, fa_id, mo_id, sex, pheno, index_keep, num_ind, num_keep);

    LOGGER << "Checking other sample files..." << std::endl;
    // Strictly sequential single-stream verification to avoid parallel open/read
    // storms on HPC parallel filesystems (Lustre/GPFS/NFS).
    std::vector<char> cmp_buf;
    bool errr = false;
    for (size_t i = 1; i < mfiles.size(); ++i) {
        const bool ok =
            gcta_pheno_detail::files_byte_equal(ref_bytes, mfiles[i], cmp_buf) ||
            gcta_pheno_detail::compare_sample_files_tokens(ref_bytes, mfiles[i]);
        if (!ok) {
            errr = true;
            LOGGER.w(0, "Sample file [" + mfiles[i] + "] is different from the first file.");
        }
    }
    if (errr) {
        LOGGER.e(0, "GCTA requires all files with same sample information.");
    } else {
        LOGGER.i(1, "All the files checked are OK.");
    }
}

void Pheno::read_psam(string psam_file){
    LOGGER.i(0, "Reading PLINK sample file from [" + psam_file + "]...");
    gcta_text_io::FastLineReader reader;
    if (!reader.open(psam_file)) {
        LOGGER.e(0, "can't read [" + psam_file + "].");
    }
    gcta_pheno_detail::parse_psam_reader(
        reader, psam_file, normalizeSexCode(1), normalizeSexCode(2),
        fid, pid, mark, fa_id, mo_id, sex, pheno, index_keep, num_ind, num_keep);
}

void Pheno::read_sample(string sample_file){
    LOGGER.i(0, "Reading Oxford sample information file from [" + sample_file + "]...");
    gcta_text_io::FastLineReader reader;
    if (!reader.open(sample_file)) {
        LOGGER.e(0, "cannot open sample file to read.");
    }
    std::string_view line;
    if (!reader.next_line(line)) {
        LOGGER.e(0, "invalid sample file");
    }
    size_t header_bytes = line.size() + 1;
    std::vector<std::string_view> toks;
    toks.reserve(16);
    gcta_text_io::split_ws_exact(line, toks);
    const size_t col_num = toks.size();
    if (col_num >= 4 && toks[0] == "ID_1" && toks[1] == "ID_2"
            && toks[2] == "missing" && toks[3] == "sex") {
        if (!reader.next_line(line)) {
            LOGGER.e(0, "invalid sample file");
        }
        header_bytes += line.size() + 1;
        gcta_text_io::split_ws_exact(line, toks);
        if (toks.size() != col_num
                || toks[0] != "0"
                || toks[1] != "0"
                || toks[2] != "0"
                || toks[3] != "D") {
            LOGGER.e(0, "invalid sample file");
        }
    } else {
        LOGGER.e(0, "invalid sample file");
    }

    const int8_t sex1 = normalizeSexCode(1);
    const int8_t sex2 = normalizeSexCode(2);
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const uint64_t fsize = reader.file_size();
    bool reserved = false;
    size_t sample_bytes = 0;

    int line_number = 0;
    while (reader.next_line(line)) {
        gcta_text_io::split_ws_exact(line, toks);
        if (toks.size() != col_num) {
            LOGGER.e(0, "line " + to_string(line_number + 3) + " has different number of columns.");
        }
        sample_bytes += line.size() + 1;
        if (!reserved && line_number == 15 && fsize > header_bytes) {
            reserved = true;
            const size_t avg_line = std::max<size_t>(sample_bytes / 16, 8);
            const size_t est_rows = static_cast<size_t>((fsize - header_bytes) / avg_line) + 64;
            fid.reserve(est_rows);
            pid.reserve(est_rows);
            fa_id.reserve(est_rows);
            mo_id.reserve(est_rows);
            mark.reserve(est_rows);
            sex.reserve(est_rows);
            pheno.reserve(est_rows);
        }
        const std::string_view sv_fid = toks[0];
        const std::string_view sv_pid = toks[1];
        const std::string_view sv_sex = toks[3];
        fid.emplace_back(sv_fid);
        pid.emplace_back(sv_pid);
        fa_id.emplace_back("0");
        mo_id.emplace_back("0");
        std::string m;
        m.reserve(sv_fid.size() + 1 + sv_pid.size());
        m.append(sv_fid);
        m.push_back('\t');
        m.append(sv_pid);
        mark.push_back(std::move(m));

        int8_t sex_code = 0;
        if (sv_sex.size() == 1) {
            if (sv_sex[0] == '1') sex_code = sex1;
            else if (sv_sex[0] == '2') sex_code = sex2;
        }
        sex.push_back(sex_code);
        pheno.push_back(kNaN);
        line_number++;
    }
    if (reader.io_error()) {
        LOGGER.e(0, "error reading sample file [" + sample_file + "].");
    }

    index_keep.resize(fid.size());
    std::iota(index_keep.begin(), index_keep.end(), static_cast<uint32_t>(0));
    num_ind = static_cast<int>(index_keep.size());
    num_keep = static_cast<int>(index_keep.size());
    LOGGER.i(0, to_string(num_ind) + " individuals to be included from the sample file.");
}


void Pheno::read_fam(string fam_file) {
    LOGGER.i(0, "Reading PLINK FAM file from [" + fam_file + "]...");
    gcta_text_io::FastLineReader reader;
    if (!reader.open(fam_file)) {
        LOGGER.e(0, "cannot open the file [" + fam_file + "] to read");
    }

    int line_number = 0;
    size_t last_length = 0;
    bool reserved = false;
    std::vector<std::string_view> toks;
    toks.reserve(8);
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    std::string_view line;
    while (reader.next_line(line)) {
        if (!reserved) {
            reserved = true;
            if (reader.file_size() > 0) {
                const size_t avg_line = std::max<size_t>(line.size() + 1, 8);
                const size_t est_rows = static_cast<size_t>(reader.file_size() / avg_line) + 64;
                fid.reserve(est_rows);
                pid.reserve(est_rows);
                mark.reserve(est_rows);
                fa_id.reserve(est_rows);
                mo_id.reserve(est_rows);
                sex.reserve(est_rows);
                pheno.reserve(est_rows);
                index_keep.reserve(est_rows);
            }
        }
        line_number++;
        gcta_text_io::split_ws_exact(line, toks);
        if (toks.size() < static_cast<size_t>(Constants::NUM_FAM_COL)) {
            LOGGER.e(0, "the fam file [" + fam_file + "], line " + to_string(line_number)
                   + " has elements less than " + to_string(Constants::NUM_FAM_COL));
        }
        if (line_number > 1 && toks.size() != last_length) {
            LOGGER.w(0, "the fam file [" + fam_file + "], line " + to_string(line_number) + " have different elements");
        }
        const std::string_view sv_fid = toks[0];
        const std::string_view sv_pid = toks[1];
        fid.emplace_back(sv_fid);
        pid.emplace_back(sv_pid);
        std::string m;
        m.reserve(sv_fid.size() + 1 + sv_pid.size());
        m.append(sv_fid);
        m.push_back('\t');
        m.append(sv_pid);
        mark.push_back(std::move(m));
        fa_id.emplace_back(toks[2]);
        mo_id.emplace_back(toks[3]);
        int sex_raw = 0;
        if (!gcta_text_io::parse_int_sv(toks[4], sex_raw)) {
            sex_raw = 0;
        }
        sex.push_back(normalizeSexCode(sex_raw));
        pheno.push_back(kNaN);
        last_length = toks.size();
        index_keep.push_back(static_cast<uint32_t>(line_number - 1));
    }
    num_ind = static_cast<int>(fid.size());
    num_bytes = (num_ind + 3) / 4;
    num_keep = static_cast<int>(index_keep.size());
    LOGGER.i(0, to_string(num_ind) + " individuals to be included from FAM file.");
}

uint32_t Pheno::getSeed(){
    boost::crc_32_type result;
    for(int i = 0; i < pid.size(); i++){
        string tmp = pid[i];
        const char* buf = tmp.data();
        result.process_bytes(buf, tmp.size());
    }
    return result.checksum();
}

vector<string> Pheno::get_id(int from_index, int to_index, string delim ){
    vector<string> out_id;
    out_id.reserve(to_index - from_index + 1);
    for(int index = from_index; index <= to_index; index++){
        int raw_index = index_keep[index];
        out_id.push_back(fid[raw_index] + delim + pid[raw_index]);
        //out_id.push_back(mark[raw_index]);
    }
    return out_id;
}

// ori_index, original raw count in fam file
PhenoMask Pheno::get_indi_mask(uint32_t ori_index){
    uint32_t pos_byte = ori_index % 4;
    switch(pos_byte){
        case 0:
            return PhenoMask(3, 0);
        case 1:
            return PhenoMask(12, 2);
        case 2:
            return PhenoMask(48, 4);
        default:
            return PhenoMask(192, 6);
    }
}

vector<uint32_t>& Pheno::get_index_keep() {
    return this->index_keep;
}
vector<uint32_t>& Pheno::getSexValidRawIndex(){
    return this->index_keep_sex;
}
vector<uint32_t>& Pheno::getHeterogameticRawIndex(){
    return this->index_keep_heterogametic;
}
vector<uint32_t>& Pheno::getHeterogameticExtractIndex(){
    return this->index_keep_heterogametic_extract;
}



void Pheno::save_pheno(string filename){
    LOGGER.i(0, "Saving individual information to [" + filename + "]...");
    std::ofstream out(filename.c_str(), std::ios::binary);
    if(!out.good()){
        LOGGER.e(0, "can't write to [" + filename + "].");
    }
    constexpr size_t kFlushBytes = 4ULL << 20;
    std::string buf;
    buf.reserve(kFlushBytes + 4096);
    char numbuf[64];
    for(const uint32_t i : index_keep){
        buf.append(mark[i]);
        buf.push_back('\t');
        buf.append(fa_id[i]);
        buf.push_back('\t');
        buf.append(mo_id[i]);
        buf.push_back('\t');
        {
            auto [end, ec] = std::to_chars(numbuf, numbuf + sizeof(numbuf), static_cast<int>(sex[i]));
            if (ec == std::errc{}) buf.append(numbuf, static_cast<size_t>(end - numbuf));
        }
        buf.push_back('\t');
        const double pv = pheno[i];
        if (std::isnan(pv)) {
            buf.append("nan", 3);
        } else if (std::isinf(pv)) {
            if (pv < 0) buf.append("-inf", 4);
            else buf.append("inf", 3);
        } else {
            auto [end, ec] = std::to_chars(numbuf, numbuf + sizeof(numbuf), pv,
                                           std::chars_format::general, 6);
            if (ec == std::errc{}) {
                buf.append(numbuf, static_cast<size_t>(end - numbuf));
            } else {
                const int n = std::snprintf(numbuf, sizeof(numbuf), "%.6g", pv);
                if (n > 0) buf.append(numbuf, static_cast<size_t>(n));
            }
        }
        buf.push_back('\n');
        if (buf.size() >= kFlushBytes) {
            out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
            buf.clear();
        }
    }
    if (!buf.empty()) {
        out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
    out.flush();
    if (!out.good()) {
        LOGGER.e(0, "error writing to [" + filename + "].");
    }
    LOGGER.i(0, to_string(index_keep.size()) + " individuals have been saved.");
}

void Pheno::get_pheno(vector<string>& ids, vector<double>& pheno){
    ids.clear();
    ids.reserve(index_keep.size());
    pheno.clear();
    pheno.reserve(index_keep.size());
    for(auto& index : index_keep){
        if(std::isfinite(this->pheno[index])){
            ids.push_back(mark[index]);
            pheno.push_back(this->pheno[index]);
        }
    }
}


uint8_t Pheno::extract_genobit(uint8_t *const buf, int index_in_keep) {
    //return 3;
    uint32_t raw_index = index_keep[index_in_keep];
    //PhenoMask mask = get_indi_mask(raw_index);
    uint8_t mask, shift;
    switch(raw_index % 4){
        case 0:
            mask = 3; shift = 0; break;
        case 1:
            mask = 12; shift = 2; break;
        case 2:
            mask = 48; shift = 4; break;
        default:
            mask = 192; shift = 6;
    }
    uint8_t extract_byte = *(buf + raw_index / 4);
    return (extract_byte & mask) >> shift;
}

uint32_t Pheno::count_raw(){
    return num_ind;
}

uint32_t Pheno::count_keep(){
    return num_keep;
}

uint32_t Pheno::count_heterogametic(){
    return index_keep_heterogametic.size();
}

int8_t Pheno::get_sex(uint32_t index){
    return sex[index_keep[index]];
}

// remove have larger priority than keep, once the SNP has been removed, it
// will never be kept again
void Pheno::set_keep(vector<string>& indi_marks, vector<string>& marks, vector<uint32_t>& keeps, bool isKeep) {
    int osize = indi_marks.size();
    removeDuplicateSort(indi_marks);
    int nDup = osize - indi_marks.size();
    if(nDup != 0){
        LOGGER.w(0, to_string(nDup) + " duplicated samples were ignored in the list.");
    }

    vector<uint32_t> keep_index, indi_index;
    vector_commonIndex_sorted1(marks, indi_marks, keep_index, indi_index);

    vector<uint32_t> remain_index;
    if(isKeep){
        std::set_intersection(keeps.begin(), keeps.end(),
                               keep_index.begin(), keep_index.end(),
                               std::back_inserter(remain_index));
    }else{
        std::set_difference(keeps.begin(), keeps.end(),
                            keep_index.begin(), keep_index.end(),
                            std::back_inserter(remain_index));
    }
    keeps = remain_index;

    LOGGER.i(0, string("After ") + (isKeep?"keeping":"removing") +  " individuals, " + to_string(keeps.size()) + " subjects remain.");
/* 
    std::sort(indi_marks.begin(), indi_marks.end());
    vector<uint32_t> pIN;
    uint32_t counter = 0;
    for(auto& item_mark : marks){
        if(std::binary_search(indi_marks.begin(), indi_marks.end(), item_mark)){
            pIN.push_back(counter);
        }
        counter++;
    }

    if(isKeep){
        auto common = [&pIN](int key) ->bool{
            return std::find(pIN.begin(), pIN.end(), key) == pIN.end();
        };
        keeps.erase(std::remove_if(keeps.begin(), keeps.end(), common), keeps.end());
    }else{
        auto diff = [&pIN](int key) ->bool{
            return std::find(pIN.begin(), pIN.end(), key) != pIN.end();
        };
        keeps.erase(std::remove_if(keeps.begin(), keeps.end(), diff), keeps.end());
    }
    */

}

void Pheno::update_sex(vector<string>& indi_marks, vector<double>& phenos){
    vector<uint32_t> pheno_index, update_index;
    vector_commonIndex_sorted1(mark, indi_marks, pheno_index, update_index);
    
    vector<uint32_t> common_index, pheno_index2;
    vector_commonIndex(index_keep, pheno_index, common_index, pheno_index2); 

    vector<uint32_t> indicies;
    indicies.reserve(common_index.size());
    for(int i = 0; i < common_index.size(); i++){
        uint32_t raw_index = index_keep[common_index[i]];
        int temp_update_pheno = std::round(phenos[update_index[pheno_index2[i]]]);
        temp_update_pheno = normalizeSexCode(temp_update_pheno);
        indicies.push_back(raw_index);
        sex[raw_index] = temp_update_pheno;
    }
    index_keep = indicies;
}


void Pheno::update_pheno(vector<string>& indi_marks, vector<double>& phenos){
    vector<uint32_t> pheno_index, update_index;
    vector_commonIndex_sorted1(mark, indi_marks, pheno_index, update_index);
    
    vector<uint32_t> common_index, pheno_index2;
    vector_commonIndex(index_keep, pheno_index, common_index, pheno_index2); 

    vector<uint32_t> indicies;
    indicies.reserve(common_index.size());
    for(int i = 0; i < common_index.size(); i++){
        uint32_t raw_index = index_keep[common_index[i]];
        double temp_update_pheno = phenos[update_index[pheno_index2[i]]];
        if(!std::isnan(temp_update_pheno)){
            indicies.push_back(raw_index);
            pheno[raw_index] = temp_update_pheno;
        }
    }
    index_keep = indicies;

    /*
    for(int i = 0; i < pheno_index.size(); i++){
        double temp_update_pheno = phenos[update_index[i]];
        int raw_index = pheno_index[i];
        if(std::binary_search(index_keep.begin(), index_keep.end(), raw_index) &&
                (!std::isnan(temp_update_pheno))){
            pIN.push_back(raw_index);
            pheno[raw_index] = temp_update_pheno;
        }
    }

    index_keep = pIN;

    for(auto& index : index_keep){
        auto lower = std::lower_bound(indi_marks.begin(), indi_marks.end(), mark[index]);
        int lower_index = lower - indi_marks.begin();
        if(lower != indi_marks.end() && (*lower) == mark[index] && ! std::isnan(phenos[lower_index])){
            pIN.push_back(index);
            pheno[index] = phenos[lower_index];
        }
    }

    index_keep = pIN;
    */
}

void Pheno::update_weight(vector<string>& indi_marks, vector<double>& weights) {
    if (weight.size() != num_ind)
        weight.assign(num_ind, 1.0);   // default 1.0, not NaN
    vector<uint32_t> pheno_index, update_index;
    vector_commonIndex_sorted1(mark, indi_marks, pheno_index, update_index);
    vector<uint32_t> common_index, pheno_index2;
    vector_commonIndex(index_keep, pheno_index, common_index, pheno_index2);
    for (int i = 0; i < (int)common_index.size(); i++) {
        uint32_t raw_index = index_keep[common_index[i]];
        double w = weights[update_index[pheno_index2[i]]];
        if (std::isfinite(w) && w > 0.0)
            weight[raw_index] = w;
        // else leave as 1.0
    }
}

Eigen::VectorXf Pheno::get_sqrt_weight_keep() const {
    Eigen::VectorXf out = Eigen::VectorXf::Ones(index_keep.size());
    if (weight.empty()) return out;
    for (int i = 0; i < (int)index_keep.size(); i++)
        out[i] = static_cast<float>(std::sqrt(weight[index_keep[i]]));
    return out;
}


void Pheno::reinit_rm(vector<uint32_t> &keeps, vector<uint32_t> &rms, int total_sample_number) {
    rms.clear();
    if (total_sample_number <= 0 || static_cast<size_t>(total_sample_number) == keeps.size()) {
        return;
    }
    rms.reserve(static_cast<size_t>(total_sample_number) - keeps.size());
    uint32_t expected = 0;
    for (uint32_t idx : keeps) {
        while (expected < idx) {
            rms.push_back(expected++);
        }
        expected = idx + 1;
    }
    while (expected < static_cast<uint32_t>(total_sample_number)) {
        rms.push_back(expected++);
    }
}


void Pheno::init_bmask_block(){
    int max_block = (index_keep[index_keep.size()-1] + 31) / 32;
    block_num = max_block;
    int max_index = max_block * 32 - 1;

    int start_rm_index;
    if(index_rm.size() == 0){
        start_rm_index = index_keep[index_keep.size()-1] + 1;
    }else{
        start_rm_index = index_rm[index_rm.size() - 1] + 1;
    }

    for(int rm_index = start_rm_index; rm_index <= max_index; rm_index++){
        index_rm.push_back(rm_index);
    }

    int cur_block = 0;
    auto last_it_keep = index_keep.begin();
    auto last_it_rm = index_rm.begin();
    while(cur_block < max_block){
        uint32_t start_val = cur_block * 32;
        uint32_t end_val = start_val + 31;
        auto it_keep = std::upper_bound(last_it_keep, index_keep.end(), end_val); 
        bool keep_found = true;
        uint32_t cur_index = *(it_keep - 1);
        if(cur_index >= start_val && cur_index <= end_val){
            keep_block_index.push_back(cur_block);
            uint64_t mask_item = 0xFFFFFFFFFFFFFFFF;
            uint64_t mask_add_item = 0;
            auto begin_it_rm = std::upper_bound(last_it_rm, index_rm.end(), start_val);
            auto end_it_rm = std::upper_bound(begin_it_rm, index_rm.end(), end_val);
            if((*begin_it_rm) >= start_val && (*end_it_rm) <= end_val){
                for(auto it = begin_it_rm; it <= end_it_rm; it++){
                    uint32_t cur_offset = (*it) - start_val;
                    int cur_byte = cur_offset / 4;
                    uint8_t *mask_pos = (uint8_t *) ((&mask_item) + cur_byte);
                    uint8_t *mask_add_pos = (uint8_t *)((&mask_add_item) + cur_byte);
                    int cur_bitpos = cur_offset % 4;
                    uint8_t mask_piece;
                    uint8_t mask_add_piece;
                    switch(cur_bitpos){
                        case 0:
                            mask_piece = 0b11111100;
                            mask_add_piece = 0b00000001;
                            break;
                        case 1:
                            mask_piece = 0b11110011;
                            mask_add_piece = 0b00000100;
                            break;
                        case 2:
                            mask_piece = 0b11001111;
                            mask_add_piece = 0b00010000;
                            break;
                        case 3:
                            mask_piece = 0b00111111;
                            mask_add_piece = 0b01000000;
                    }
                    (*mask_pos) &= mask_piece;
                    (*mask_add_pos) += mask_add_piece;
                }
            }
            mask_items.push_back(mask_item);
            mask_add_items.push_back(mask_add_item);
            last_it_rm = end_it_rm;
        }
        last_it_keep = it_keep;
        cur_block++;
    }
}


void Pheno::init_mask_block(){
    mask_block.clear();
    mask_add_block.clear();
    for(const auto &index : index_rm){
        int byte_pos = index / 4;
        Mask_t::iterator cur_item = mask_block.lower_bound(byte_pos);
        Mask_t::iterator cur_add_item = mask_add_block.lower_bound(byte_pos);
        uint8_t mask_item = 0;
        uint8_t mask_add_item = 0;
        switch(index % 4){
            case 0:
                mask_item = 0b11111100;
                mask_add_item = 0b00000001;
                break;
            case 1:
                mask_item = 0b11110011;
                mask_add_item = 0b00000100;
                break;
            case 2:
                mask_item = 0b11001111;
                mask_add_item = 0b00010000;
                break;
            case 3:
                mask_item = 0b00111111;
                mask_add_item = 0b01000000;
        }

        if(cur_item != mask_block.end() && !(mask_block.key_comp()(byte_pos, cur_item->first))){
            cur_item->second = cur_item->second & mask_item;
            cur_add_item->second = cur_add_item->second + mask_add_item;
        }else{
            mask_block.insert(cur_item, Mask_t::value_type(byte_pos, mask_item));
            mask_add_block.insert(cur_add_item, Mask_t::value_type(byte_pos, mask_add_item));
        }

    }
}

void Pheno::mask_geno_keep(uint8_t *const geno_1block, int num_blocks) {
    if(mask_block.size() == 0){
        return;
    }
    #pragma omp parallel for schedule(dynamic) 
    for(int cur_block = 0; cur_block < num_blocks; cur_block++){
        uint8_t *cur_pos =  geno_1block + cur_block * num_bytes;
        for(const auto &item : mask_block){
            uint8_t *mask_pos = cur_pos + item.first;
            *mask_pos = ((*mask_pos) & item.second) + mask_add_block[item.first];
        }
    }
}
//maskp must be zeroed
void Pheno::getMaskBit(uint64_t *maskp){
    for(auto keep_item : index_keep){
        uint32_t cur_qword = keep_item / 64;
        uint32_t cur_offset = keep_item % 64;
        maskp[cur_qword] |= k1LU << cur_offset;
    }
}

//maskp must be zeroed
void Pheno::getMaskBitHeterogametic(uint64_t *maskp){
    for(auto keep_item : index_keep_heterogametic){
        uint32_t cur_qword = keep_item / 64;
        uint32_t cur_offset = keep_item % 64;
        maskp[cur_qword] |= k1LU << cur_offset;
    }
}
