/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   New implementation: generate, read and process GRM.

   Depends on the class of genotype

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/

#include "GRM.h"
#include "Logger.h"
#include "cpu.h"
#include <Eigen/Dense>
#include <format>
#include <limits>
#include <memory>
#include <iterator>
#include <algorithm>
#include <ranges>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <numeric>
#include <unordered_set>
#include "utils.hpp"
#include "grm_binary_io.hpp"
#include <omp.h>
#include "third_party/plink-ng/2.0/include/plink2_bits.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <sstream>
#include <csignal>

using std::to_string;

namespace {

// Finds the largest `re` in (rs, full_re] such that a GRM tile spanning rows
// [rs, re) fits within budget_elems *actually touched* grm+N elements.
// This may overcommit if the virtual square allocation becomes resident.
//
// Solved as the positive root of re^2 + re - (2*budget_elems + rs*(rs+1)) = 0
// in floating-point arithmetic, then floored and clamped to full_re.
int solve_tile_budget_end(int rs, int full_re, double budget_elems) {
    const double drs = rs;
    const double disc = 1.0 + 4.0 * (2.0 * budget_elems + drs * (drs + 1.0));
    int re = static_cast<int>(std::floor((-1.0 + std::sqrt(disc)) / 2.0));
    return std::min(re, full_re);
}

// True when a single tile spanning the whole GRM [full_rs, full_re) already
// fits within budget_elems -- i.e. tiling would produce exactly one tile, so
// the dense (non-tiled) path should be used instead of the tiling machinery.
// Shared by the constructor (decides whether to allocate the member grm/N
// buffers up front) and processMakeGRM (decides which path actually runs)
// so the two decisions can't drift apart: if the constructor thinks tiling
// is active and leaves grm/N null, but processMakeGRM then takes the dense
// path expecting them allocated, every dense-path write goes through a null
// pointer.
bool grm_tile_budget_covers_whole(int full_rs, int full_re, double budget_elems) {
    const double drs = full_rs, dre = full_re;
    const double whole_elems = (dre * (dre + 1.0) - drs * (drs + 1.0)) / 2.0;
    return whole_elems <= budget_elems;
}

} // namespace

GRM::GRM(const std::string& grm_prefix, const std::string& keep, const std::string& remove,
         const std::string& out_prefix, bool detail)
    : o_name(out_prefix), keep_file(keep), remove_file(remove), cutoff_detail(detail) {
    if (!grm_prefix.empty()) {
        this->grm_file = grm_prefix;
        grm_ids = Pheno::read_sublist(grm_file + ".grm.id");
        num_subjects = grm_ids.size();

        uint64_t num_grm = num_subjects * (num_subjects + 1) / 2;
        UniqueFile file = open_file(grm_file + ".grm.bin", "rb");
        if (!file) {
            LOGGER.error(0, "can't open {}.grm.bin", grm_file);
        }
        const auto grm_file_size = getFileByteSize(file.get());
        if (!grm_file_size) {
            LOGGER.e(0, "Failed to read GRM binary size for [" + grm_file + "]: " + std::string(getUtilsErrorMessage(grm_file_size.error())));
        }
        if (num_grm * 4 != grm_file_size.value()) {
            LOGGER.e(0, "The IDs in GRM and the IDs in the GRM binary file do not match [" + grm_file + "]");
        }

        uint64_t num_grm_byte = num_grm * 4;
        uint64_t num_parts = (num_grm_byte + num_byte_GRM_read - 1) / num_byte_GRM_read;
        vector<uint32_t> parts = divide_parts(0, num_subjects - 1, num_parts);
        index_grm_pairs.reserve(parts.size());
        byte_part_grms.reserve(parts.size());

        index_grm_pairs.push_back(std::make_pair(0, parts[0]));
        byte_part_grms.push_back((0 + parts[0] + 2) * (parts[0] + 1) / 2);
        for(int index = 1; index != parts.size(); index++){
            index_grm_pairs.push_back(std::make_pair(parts[index - 1] + 1, parts[index]));
            byte_part_grms.push_back(((uint64_t)parts[index - 1] + parts[index] + 3) * (parts[index] - parts[index - 1]) / 2);
        }

        num_byte_buffer = *std::max_element(byte_part_grms.begin(), byte_part_grms.end());

        index_keep.resize(num_subjects);
        std::iota(index_keep.begin(), index_keep.end(), 0);

        if (!keep_file.empty()) {
            vector<string> keep_lists = Pheno::read_sublist(keep_file);
            LOGGER << "Get " << keep_lists.size() << " samples from list [" << keep_file << "]." << std::endl;
            Pheno::set_keep(keep_lists, grm_ids, index_keep, true);
        }

        if (!remove_file.empty()) {
            vector<string> rm_lists = Pheno::read_sublist(remove_file);
            LOGGER << "Get " << rm_lists.size() << " samples from list [" << remove_file << "]." << std::endl;
            Pheno::set_keep(rm_lists, grm_ids, index_keep, false);
        }
    }
}

void GRM::subtract_grm(string mgrm_file, string out_file){
   std::ifstream mgrm(mgrm_file.c_str());
    if(!mgrm){
        LOGGER.e(0, "can't open " + mgrm_file + " to read.");
    }
    string line;
    vector<string> files;
    vector<string> err_files;
    while(getline(mgrm, line)){
        boost::trim(line);
        if(!line.empty()){
            if(checkFileReadable(line+".grm.id") && checkFileReadable(line+".grm.bin") && checkFileReadable(line + ".grm.N.bin")){
                files.push_back(line);
            }else{
                err_files.push_back(line);
            }
        }
    }

    if(err_files.size() != 0){
        string out_err = "can't read GRM (*.grm.id, *.grm.bin) in ";
        out_err += boost::algorithm::join(err_files, ", ");
        out_err += ".";
        LOGGER.e(0, out_err);
    }
    if(files.size() != 2){
        LOGGER.e(0, "only 2 GRMs are supported currently.");
    }

    LOGGER.i(0, "Reading [" + files[0] + ".grm.id]...");
    vector<string> common_id = Pheno::read_sublist(files[0] + ".grm.id");
    LOGGER << common_id.size() << " samples have been read." << std::endl;
    LOGGER.i(0, "Reading [" + files[1] + ".grm.id]...");
    vector<string> common_id2 = Pheno::read_sublist(files[1] + ".grm.id");
    if(common_id != common_id2){
        LOGGER.e(0, "The sample IDs in the two GRMs are not same. You may try --unify-grm first.");
    }
    std::ofstream o_id(out_file + ".grm.id");
    if(!o_id) LOGGER.e(0, "can't write to [" + out_file + ".grm.id]");
    LOGGER.i(2, "Saving " + to_string(common_id.size()) + " individual IDs");
    std::ranges::copy(common_id, std::ostream_iterator<string>(o_id, "\n"));
    o_id.close();

    uint64_t grm_size = (1 + common_id.size()) * common_id.size() / 2;
    uint64_t grm_file_size = grm_size * 4; 

    UniqueFile h_grm1  = open_file(files[0] + ".grm.bin", "rb");
    UniqueFile h_grmN1 = open_file(files[0] + ".grm.N.bin", "rb");
    UniqueFile h_grm2  = open_file(files[1] + ".grm.bin", "rb");
    UniqueFile h_grmN2 = open_file(files[1] + ".grm.N.bin", "rb");
    if (!h_grm1 || grm_file_size != getFileSize(h_grm1.get())) {
        LOGGER.e(0, "The size of [" + files[0] + ".grm.bin] is not correct.");
    }
    if (!h_grmN1 || grm_file_size != getFileSize(h_grmN1.get())) {
        LOGGER.e(0, "The size of [" + files[0] + ".grm.N.bin] is not correct.");
    }
    if (!h_grm2 || grm_file_size != getFileSize(h_grm2.get())) {
        LOGGER.e(0, "The size of [" + files[1] + ".grm.bin] is not correct.");
    }
    if (!h_grmN2 || grm_file_size != getFileSize(h_grmN2.get())) {
        LOGGER.e(0, "The size of [" + files[1] + ".grm.N.bin] is not correct.");
    }

    uint64_t itemRead = 26214400;
    std::vector<float> buf1(itemRead);
    std::vector<float> buf2(itemRead);
    std::vector<float> buf(itemRead);
    std::vector<float> bufN1(itemRead);
    std::vector<float> bufN2(itemRead);
    std::vector<float> bufN(itemRead);
    
    int loop_num = (grm_size + itemRead - 1) / itemRead;
    uint64_t remain_item = grm_size % itemRead;
    LOGGER.i(0, "Subtracting GRMs...");
    UniqueFile ho_grm  = open_file(out_file + ".grm.bin", "wb");
    UniqueFile ho_grmN = open_file(out_file + ".grm.N.bin", "wb");
    if (!ho_grm) {
        LOGGER.error(0, "can't write to [{}.grm.bin].", out_file);
    }
    if (!ho_grmN) {
        LOGGER.error(0, "can't write to [{}.grm.N.bin].", out_file);
    }

    for(int i = 0; i < loop_num; i++){
        if(i == loop_num - 1 && remain_item != 0) itemRead = remain_item;
        readBytes(h_grm1.get(), itemRead, buf1.data());
        readBytes(h_grm2.get(), itemRead, buf2.data());
        readBytes(h_grmN1.get(), itemRead, bufN1.data());
        readBytes(h_grmN2.get(), itemRead, bufN2.data());
        // On the first chunk, check that file[0] has at least as many markers as file[1].
        // A negative N_diff means the files are in the wrong order (chr GRM listed first).
        if(i == 0 && bufN1[0] < bufN2[0]){
            LOGGER.e(0, "subtract_grm: the first GRM [" + files[0] + "] has fewer markers than "
                        "the second [" + files[1] + "] in the first pair. "
                        "The larger (all-chromosome) GRM must be listed first.");
        }
        for(int j = 0; j < itemRead; j++){
            bufN[j] = bufN1[j] - bufN2[j];
            // Guard against zero denominator: a pair where all markers in the subtracted
            // GRM are also missing in the full GRM (N_all == N_chr for that pair).
            buf[j] = (bufN[j] > 0.0f)
                ? (float)(((double)buf1[j] * bufN1[j] - (double)buf2[j] * bufN2[j]) / bufN[j])
                : 0.0f;
        }
        if(fwrite(buf.data(), sizeof(float), itemRead, ho_grm.get()) != itemRead){
            LOGGER.e(0, "can't write to [" + out_file + ".grm.bin].");
        }
        if(fwrite(bufN.data(), sizeof(float), itemRead, ho_grmN.get()) != itemRead){
            LOGGER.e(0, "can't write to [" + out_file + ".grm.N.bin].");
        }
    }
    ho_grm.reset();
    ho_grmN.reset();
    LOGGER.i(0, "The subtracted GRM has been written to [" + out_file + ".grm.bin, .grm.N.bin].");
}

void GRM::merge_grm(string mgrm_file, string out_file, int memory_budget_gb){
    std::ifstream mgrm(mgrm_file.c_str());
    if(!mgrm){
        LOGGER.e(0, "can't open " + mgrm_file + " to read.");
    }

    string line;
    vector<string> files;
    vector<string> err_files;
    while(getline(mgrm, line)){
        boost::trim(line);
        if(!line.empty()){
            if(checkFileReadable(line + ".grm.id") &&
               checkFileReadable(line + ".grm.bin") &&
               checkFileReadable(line + ".grm.N.bin")){
                files.push_back(line);
            }else{
                err_files.push_back(line);
            }
        }
    }

    if(err_files.size() != 0){
        string out_err = "can't read GRM (*.grm.id, *.grm.bin, *.grm.N.bin) in ";
        out_err += boost::algorithm::join(err_files, ", ");
        out_err += ".";
        LOGGER.e(0, out_err);
    }
    if(files.size() < 2){
        LOGGER.e(0, "streaming merge needs at least two GRMs in --mgrm.");
    }

    gcta_grm_io::merge_grms(files, out_file, memory_budget_gb);
}


void GRM::unify_grm(string mgrm_file, string out_file){
    std::ifstream mgrm(mgrm_file.c_str());
    if(!mgrm){
        LOGGER.e(0, "can't open " + mgrm_file + " to read.");
    }
    string line;
    vector<string> files;
    vector<string> err_files;
    while(getline(mgrm, line)){
        boost::trim(line);
        if(!line.empty()){
            if(checkFileReadable(line+".grm.id") && checkFileReadable(line+".grm.bin")){
                files.push_back(line);
            }else{
                err_files.push_back(line);
            }
        }
    }

    if(err_files.size() != 0){
        string out_err = "can't read GRM (*.grm.id, *.grm.bin) in ";
        out_err += boost::algorithm::join(err_files, ", ");
        out_err += ".";
        LOGGER.e(0, out_err);
    }
    if(files.size() < 2){
        LOGGER.e(0, "not enough valid GRMs to be unified.");
    }

    LOGGER.i(0, "Reading [" + files[0] + ".grm.id]...");
    vector<string> common_id = Pheno::read_sublist(files[0] + ".grm.id");
    LOGGER << common_id.size() << " samples have been read." << std::endl;
    vector<vector<string>> ids;
    ids.resize(files.size());
    
    ids[0] = common_id;
    for(int i = 1; i < files.size(); i++){
        string cur_file = files[i] + ".grm.id";
        LOGGER.i(0, "Reading [" + cur_file + "]...");
        ids[i] = Pheno::read_sublist(cur_file);
        LOGGER << ids[i].size() << " samples have been read." << std::endl;
        vector<uint32_t> index1, index2;
        vector_commonIndex_sorted1(common_id, ids[i], index1, index2); 
        vector<string> temp_common_id(index1.size());
        std::ranges::transform(index1, temp_common_id.begin(), [&common_id](uint32_t v){
                return common_id[v];});
        common_id = temp_common_id;
        LOGGER << common_id.size() << " common samples in GRMs" << std::endl;
    }
    if(!keep_file.empty()){
        LOGGER.i(0, "Keeping individuals listed in [" + keep_file + "]...");
        vector<string> keep_id = Pheno::read_sublist(keep_file);
        LOGGER << keep_id.size() << " samples have been read." << std::endl;
        vector<uint32_t> index1, index2;
        vector_commonIndex_sorted1(common_id, keep_id, index1, index2);
        vector<string> temp_common_id(index1.size());
        std::ranges::transform(index1, temp_common_id.begin(), [&common_id](uint32_t v){
                return common_id[v];});
        common_id = temp_common_id;
        LOGGER << common_id.size() << " common samples after merging." << std::endl;
    }
    if(!remove_file.empty()){
        LOGGER.i(0, "Excluding individuals listed in [" + remove_file + "]...");
        vector<string> remove_id = Pheno::read_sublist(remove_file);
        LOGGER << remove_id.size() << " samples have been read." << std::endl;
        vector<uint32_t> index1, index2;
        vector_commonIndex_sorted1(common_id, remove_id, index1, index2);
        vector<uint32_t> keeps(common_id.size());
        std::iota(keeps.begin(), keeps.end(), 0);
        vector<uint32_t> remain_index;
        std::set_difference(keeps.begin(), keeps.end(), index1.begin(), index1.end(), std::back_inserter(remain_index));
        vector<string> remain_ids(remain_index.size());
        std::ranges::transform(remain_index, remain_ids.begin(), [&common_id](uint32_t v){
                return common_id[v];});
        common_id = remain_ids;
        LOGGER << common_id.size() << " common samples after merging." << std::endl;
    }
    vector<vector<uint32_t>> grm_indices(files.size());
    //produce output file names
    vector<string> output_fileNames;
    const string& out_string = out_file;
    const auto basename_out_result = getFileName(out_string);
    if(!basename_out_result){
        LOGGER.e(0, "Invalid output path [" + out_string + "]: " + std::string(getUtilsErrorMessage(basename_out_result.error())));
    }
    const auto path_out_result = getPathName(out_string);
    if(!path_out_result){
        LOGGER.e(0, "Invalid output path [" + out_string + "]: " + std::string(getUtilsErrorMessage(path_out_result.error())));
    }
    const string basename_out = basename_out_result.value();
    const string path_out = path_out_result.value();
    for(auto & filename : files){
        const auto basename_result = getFileName(filename);
        if(!basename_result){
            LOGGER.e(0, "Invalid GRM file path [" + filename + "]: " + std::string(getUtilsErrorMessage(basename_result.error())));
        }
        const auto output_path_result = joinPath(path_out, basename_result.value() + "_" + basename_out);
        if(!output_path_result){
            LOGGER.e(0, "Failed to compose output path for [" + filename + "]: " + std::string(getUtilsErrorMessage(output_path_result.error())));
        }
        output_fileNames.push_back(output_path_result.value());
    }

    #pragma omp parallel for
    for(int i = 0; i < files.size(); i++){
        string id_file_name = output_fileNames[i] + ".grm.id";
        vector<uint32_t> index1;
        vector_commonIndex_sorted1(common_id, ids[i], index1, grm_indices[i]);
        std::ofstream out_id(id_file_name.c_str());
        for(auto & index: grm_indices[i]){
            out_id << ids[i][index] << std::endl;
        }
        out_id.close();
        LOGGER.i(0, "Unified individual IDs have been saved to [" + id_file_name + "].");
    }

    LOGGER.i(0, "Writing unified GRM in binary format...");
    #pragma omp parallel for
    for(int i = 0; i < grm_indices.size(); i++){
        vector<uint32_t> &p_index = grm_indices[i];
        uint32_t size_grm = p_index.size();
        uint32_t largest_grm_size = ids[i].size();
        string file_name = files[i] + ".grm.bin";
        string wfile_name = output_fileNames[i] + ".grm.bin";
        UniqueFile h_grm = open_file(file_name, "rb");
        if (!h_grm) {
            LOGGER.e(0, "can't read " + file_name + ".");
        }

        UniqueFile h_wgrm = open_file(wfile_name, "wb");
        if (!h_wgrm) {
            LOGGER.e(0, "can't write to " + wfile_name + ".");
        }

        std::vector<float> buf(largest_grm_size);
        std::vector<float> wbuf(size_grm);

        vector<uint64_t> start_pos(largest_grm_size);
        for(int j = 0; j < largest_grm_size; j++){
            uint64_t byte_start_grm = (1 + j) * j / 2 * sizeof(float);
            start_pos[j] = byte_start_grm;
        }

        for(int j = 0; j < size_grm; j++){
            uint64_t grm_index = p_index[j];
            uint64_t num_grm = grm_index + 1;
            uint64_t byte_start_grm = start_pos[grm_index];
            fseek(h_grm.get(), byte_start_grm, SEEK_SET);
            if(fread(buf.data(), sizeof(float), num_grm, h_grm.get()) != num_grm){
                LOGGER.e(0, "error in reading [" + file_name + "], in position " + to_string(ftell(h_grm.get())), ".");
            }
            uint64_t size_write = j + 1;
            for(int k = 0; k < size_write; k++){
                uint32_t cur_index = p_index[k];
                float grm_value;
                if(cur_index < num_grm){
                    grm_value = buf[cur_index];
                }else{
                    //fill the other parts
                    auto bytes_offset = start_pos[cur_index] + grm_index * sizeof(float);
                    fseek(h_grm.get(), bytes_offset, SEEK_SET); 
                    if(fread(&grm_value, sizeof(float), 1, h_grm.get()) != 1){
                        LOGGER.e(0, "error in reading [" + file_name + "], in position " + to_string(bytes_offset), ".");
                    }
                }
                wbuf[k] = grm_value;
            }
            //last element
            //*(wbuf + j) = *(buf + grm_index);

            if(size_write != fwrite(wbuf.data(), sizeof(float), size_write, h_wgrm.get())){
                LOGGER.e(0, "error in writing to [" + wfile_name + "], pos: " + std::to_string(ftell(h_wgrm.get())) + "."); 
            }
        }
        h_grm.reset();
        h_wgrm.reset();
        LOGGER.i(0, "GRM has been written to [" + wfile_name + "].");
    }

}

void GRM::prune_fam(float thresh, bool isSparse, float *value){
    LOGGER.i(0, "Pruning the GRM to a sparse matrix with a cutoff of " + to_string(thresh) + "...");
    LOGGER.i(0, "Total number of parts to be processed: " + to_string(index_grm_pairs.size()));
    UniqueFile grmFile = open_file(grm_file + ".grm.bin", "rb");
    if (!grmFile) LOGGER.error(0, "can't read [{}.grm.bin]", grm_file);

    std::ofstream o_id((o_name + ".grm.id").c_str());
    if(!o_id) LOGGER.e(0, "can't write to [" + o_name + ".grm.id]");
    std::ofstream o_fam;
    UniqueFile o_bk;
    if(isSparse){
        o_fam.open((o_name + ".grm.sp").c_str());
        if(!o_fam) LOGGER.e(0, "can't write to [" + o_name + ".grm.sp]");
    }else{
        o_bk = open_file(o_name + ".grm.bin", "wb");
        if (!o_bk) LOGGER.error(0, "can't write to [{}.grm.bin]", o_name);
    }

    //Save the kept IDs, which may change by --keep and --remove
    vector<string> keep_ID;
    keep_ID.reserve(index_keep.size());
    for(auto & index : index_keep){
        keep_ID.emplace_back(grm_ids[index]);
    }
    LOGGER.i(2, "Saving " + to_string(keep_ID.size()) + " individual IDs");
    std::ranges::copy(keep_ID, std::ostream_iterator<string>(o_id, "\n"));
    o_id.close();
    keep_ID.clear();
    keep_ID.shrink_to_fit();


    // Save pair1 par2 GRM
    std::unordered_set<int> keeps_ori(index_keep.begin(), index_keep.end());
    std::vector<float> grm_buf(num_byte_buffer);
    float cur_grm, *cur_grm_pos0;
    vector<float> rm_grm;
    vector<int> rm_grm_ID1, rm_grm_ID2;
    std::vector<float> out_grm_buf(num_byte_buffer);
    float *cur_grm_buf;
    int new_id1 = 0;
    for(int part_index = 0; part_index != index_grm_pairs.size(); part_index++){
        if(fread(grm_buf.data(), sizeof(float), byte_part_grms[part_index], grmFile.get()) != byte_part_grms[part_index]){
            LOGGER.e(0, "Failed to read GRM between line " + to_string(index_grm_pairs[part_index].first + 1) + " and "
                        + to_string(index_grm_pairs[part_index].second + 1));
        };
        if(part_index % 50 == 0){
            LOGGER.i(2, "Processing part " + to_string(part_index + 1));
        }
        cur_grm_pos0 = grm_buf.data();
        cur_grm_buf = out_grm_buf.data();

        for(int id1 = index_grm_pairs[part_index].first; id1 != index_grm_pairs[part_index].second + 1; id1++){
            if (keeps_ori.find(id1) == keeps_ori.end()) {
                cur_grm_pos0 += id1 + 1;
                continue;
            }
            int new_id2 = 0;
            if(value){
                for(int id2 : index_keep){
                    if(id2 > id1) break;
                    cur_grm = *(cur_grm_pos0 + id2);
                    if(cur_grm > thresh){
                        rm_grm_ID1.push_back(new_id1);
                        rm_grm_ID2.push_back(new_id2);
                        rm_grm.push_back(*value);
                        *(cur_grm_buf++) = (*value);
                    }else{
                        *(cur_grm_buf++) = 0.0;
                    }
                    new_id2++;
                }
 
            }else{
                for(int id2 : index_keep){
                    if(id2 > id1) break;
                    cur_grm = *(cur_grm_pos0 + id2);
                    if(cur_grm > thresh){
                        rm_grm_ID1.push_back(new_id1);
                        rm_grm_ID2.push_back(new_id2);
                        rm_grm.push_back(cur_grm);
                        *(cur_grm_buf++) = cur_grm;
                    }else{
                        *(cur_grm_buf++) = 0.0;
                    }
                    new_id2++;
                }
            }
            new_id1++;
            cur_grm_pos0 += id1 + 1;
        }
        if(!isSparse){
            int write_items = cur_grm_buf - out_grm_buf.data();
            if(fwrite(out_grm_buf.data(), sizeof(float), write_items, o_bk.get()) != write_items){
                LOGGER.e(0, "Failed to write the output"); 
            }
        }

    }
    grmFile.reset();
    o_bk.reset();

    if(isSparse){
        LOGGER.i(0, "Saving the sparse GRM (" + to_string(rm_grm.size()) + " pairs) to [" + o_name + ".grm.sp]");
        //    auto sorted_index = sort_indexes(rm_grm_ID2, rm_grm_ID1);
        //    for(auto index : sorted_index){
        o_fam << std::setprecision( std::numeric_limits<float>::digits10+2);
        for(int index = 0; index != rm_grm.size(); index++){
            o_fam << rm_grm_ID1[index] << "\t" << rm_grm_ID2[index] << "\t" << rm_grm[index] << std::endl;
        }
        o_fam.close();
        LOGGER.i(0, "Success:", "finished generating a sparse GRM");
        return;
    }else{
        LOGGER.i(0, "GRM has been saved to [" + o_name + ".grm.bin]");
    }

    UniqueFile NFile  = open_file(grm_file + ".grm.N.bin", "rb");
    UniqueFile ONFile = open_file(o_name + ".grm.N.bin", "wb");
    if (!NFile || !ONFile) {
        LOGGER.error(0, "can't open [{}.grm.N.bin] or [{}.grm.N.bin]", grm_file, o_name);
    }
    std::vector<float> N_buf(num_byte_buffer);
    float *cur_N_pos0;
    std::vector<float> out_N_buf(num_byte_buffer);
    float *cur_N_buf;
    for(int part_index = 0; part_index != index_grm_pairs.size(); part_index++){
        if(fread(N_buf.data(), sizeof(float), byte_part_grms[part_index], NFile.get()) != byte_part_grms[part_index]){
            LOGGER.w(0, "Reading GRM N failed between line " + to_string(index_grm_pairs[part_index].first + 1) + " and "
                        + to_string(index_grm_pairs[part_index].second + 1));
            LOGGER.i(0, "Stop pruning the GRM N");
            break;
        };
        if(part_index % 50 == 0){
            LOGGER.i(2, "Processing part " + to_string(part_index + 1));
        }
        cur_N_pos0 = N_buf.data();
        cur_N_buf = out_N_buf.data();

        for(int id1 = index_grm_pairs[part_index].first; id1 != index_grm_pairs[part_index].second + 1; id1++){
            if (keeps_ori.find(id1) == keeps_ori.end()) {
                cur_N_pos0 += id1 + 1;
                continue;
            }
            for(int id2 : index_keep){
                if(id2 > id1) break;
                *(cur_N_buf++) = *(cur_N_pos0 + id2);
            }
            cur_N_pos0 += id1 + 1;
        }

        int write_items = cur_N_buf - out_N_buf.data();
        if(fwrite(out_N_buf.data(), sizeof(float), write_items, ONFile.get()) != write_items){
            LOGGER.e(0, "Failed to write the GRM N"); 
        }

    }
    ONFile.reset();
    NFile.reset();
    LOGGER.i(0, "GRM N has been saved to [" + o_name + ".grm.N.bin]");
}



void GRM::cut_rel(float thresh, bool no_grm){
    LOGGER.i(0, "Pruning the GRM with a cutoff of " + to_string(thresh) + "...");
    LOGGER.i(0, "Total number of parts to be processed: " + to_string(index_grm_pairs.size()));
    UniqueFile grmFile = open_file(grm_file + ".grm.bin", "rb");
    if (!grmFile) LOGGER.error(0, "can't read [{}.grm.bin]", grm_file);
    // put this first to avoid unwritable disk
    std::ofstream o_keep;
    if(!no_grm){
        o_keep.open((o_name + ".grm.id").c_str());
        if((!o_keep)){
            LOGGER.e(0, "can't write to [" + o_name + ".grm.id]");
        }
    }

    const bool detail_flag = cutoff_detail;
    std::ofstream out_fam, o_single;
    if(detail_flag){
        out_fam.open((o_name + ".family.txt").c_str());
        o_single.open((o_name + ".singleton.txt").c_str());
        if((!out_fam) || (!o_single)){
            LOGGER.e(0, "can't write to [" + o_name + ".family.txt, .singleton.txt]");
        }
    }
 

    std::unordered_set<int> keeps_ori(index_keep.begin(), index_keep.end());

    std::vector<float> grm_buf(num_byte_buffer);
    float cur_grm, *cur_grm_pos0;
    vector<float> rm_grm;
    vector<int> rm_grm_ID1, rm_grm_ID2;
    for(int part_index = 0; part_index != index_grm_pairs.size(); part_index++){
        if(fread(grm_buf.data(), sizeof(float), byte_part_grms[part_index], grmFile.get()) != byte_part_grms[part_index]){
            LOGGER.e(0, "Failed to read GRM between line " + to_string(index_grm_pairs[part_index].first + 1) + " and "
                        + to_string(index_grm_pairs[part_index].second + 1));
        };
        if(part_index % 50 == 0){
            LOGGER.i(2, "Processing part " + to_string(part_index + 1));
        }
        cur_grm_pos0 = grm_buf.data();
        for(int id1 = index_grm_pairs[part_index].first; id1 != index_grm_pairs[part_index].second + 1; id1++){
            if (keeps_ori.find(id1) == keeps_ori.end()) {
                cur_grm_pos0 += id1 + 1;
                continue;
            }
            for(int id2 : index_keep){
                if(id2 >= id1) break;
                cur_grm = *(cur_grm_pos0 + id2);
                if(cur_grm > thresh){
                    rm_grm_ID1.push_back(id1);
                    rm_grm_ID2.push_back(id2);
                    rm_grm.push_back(cur_grm);
                }
            }
            cur_grm_pos0 += id1 + 1;
        }
    }

    if(detail_flag){
        for(int index = 0; index != rm_grm.size(); index++){
            out_fam << grm_ids[rm_grm_ID1[index]] << "\t" <<  grm_ids[rm_grm_ID2[index]] << "\t" << rm_grm[index] << std::endl;
        }
        out_fam.close();
        LOGGER.i(0, "Related family pairs have been saved to " + o_name + ".family.txt");
    }

    int i_buf;
    // copy from GCTA 1.26
    vector<int> rm_uni_ID(rm_grm_ID1);
    rm_uni_ID.insert(rm_uni_ID.end(), rm_grm_ID2.begin(), rm_grm_ID2.end());
    stable_sort(rm_uni_ID.begin(), rm_uni_ID.end());
    rm_uni_ID.erase(unique(rm_uni_ID.begin(), rm_uni_ID.end()), rm_uni_ID.end());
    std::map<int, int> rm_uni_ID_count;
    for (int i = 0; i < rm_uni_ID.size(); i++) {
        i_buf = count(rm_grm_ID1.begin(), rm_grm_ID1.end(), rm_uni_ID[i]) + count(rm_grm_ID2.begin(), rm_grm_ID2.end(), rm_uni_ID[i]);
        rm_uni_ID_count.insert(std::pair<int, int>(rm_uni_ID[i], i_buf));
    }

    // swapping
    std::map<int, int>::iterator iter1, iter2;
    for (int i = 0; i < rm_grm_ID1.size(); i++) {
        iter1 = rm_uni_ID_count.find(rm_grm_ID1[i]);
        iter2 = rm_uni_ID_count.find(rm_grm_ID2[i]);
        if (iter1->second < iter2->second) {
            i_buf = rm_grm_ID1[i];
            rm_grm_ID1[i] = rm_grm_ID2[i];
            rm_grm_ID2[i] = i_buf;
        }
    }

    stable_sort(rm_grm_ID1.begin(), rm_grm_ID1.end());
    rm_grm_ID1.erase(unique(rm_grm_ID1.begin(), rm_grm_ID1.end()), rm_grm_ID1.end());
    vector<string> removed_ID;
    removed_ID.reserve(rm_grm_ID1.size());
    for (auto &index : rm_grm_ID1) removed_ID.push_back(grm_ids[index]);

    auto diff = [&rm_grm_ID1](int value) ->bool{
        return std::binary_search(rm_grm_ID1.begin(), rm_grm_ID1.end(), value);
    };
    std::erase_if(index_keep, diff);

    vector<string> keep_ID;
    keep_ID.reserve(index_keep.size());
    for(auto & index : index_keep){
        keep_ID.emplace_back(grm_ids[index]);
    }

    LOGGER.i(0, "After pruning the GRM, there are " + to_string(keep_ID.size()) + " individuals (" + to_string(removed_ID.size()) + " individuals removed).");
    if(!no_grm){
        std::ranges::copy(keep_ID, std::ostream_iterator<string>(o_keep, "\n"));
        o_keep.close();
        LOGGER.i(2, "Pruned unrelated IDs have been saved to " + o_name + ".grm.id");
    }

    if(detail_flag){
        std::ranges::copy(keep_ID, std::ostream_iterator<string>(o_single, "\n"));
        o_single.close();
        LOGGER.i(2, "Pruned singleton IDs have been saved to " + o_name + ".singleton.txt");
    }

    if(no_grm) {
        return;
    }

    LOGGER.i(0, "Pruning GRM values, total parts " + std::to_string(index_grm_pairs.size()));
    UniqueFile grm_out_file = open_file(o_name + ".grm.bin", "wb");
    if(!grm_out_file){
        LOGGER.e(0, "can't open [" + o_name + ".grm.bin] to write");
    }
    fseek(grmFile.get(), 0, SEEK_SET);
    outBinFile(grmFile.get(), grm_out_file.get());
    grmFile.reset();
    grm_out_file.reset();
    LOGGER.i(2, "GRM values have been saved to [" + o_name + ".grm.bin]");

    LOGGER.i(0, "Pruning number of SNPs to calculate GRM, total parts " + std::to_string(index_grm_pairs.size()));
    UniqueFile NFile = open_file(grm_file + ".grm.N.bin", "rb");
    if(!NFile){
        LOGGER.w(2, "There is no [" + grm_file + ".grm.N.bin]");
        return;
    }
    UniqueFile N_out_file = open_file(o_name + ".grm.N.bin", "wb");
    if(!N_out_file){
        LOGGER.w(2, "can't open [" + o_name + ".grm.N.bin] to write. Ignore this step");
        return;
    }
    outBinFile(NFile.get(), N_out_file.get());
    NFile.reset();
    N_out_file.reset();
    LOGGER.i(2, "Number of SNPs has been saved to [" + o_name + ".grm.N.bin]");
}

void GRM::outBinFile(FILE *sFile, FILE *dFile) {
    std::unordered_set<int> keeps(index_keep.begin(), index_keep.end());
    std::vector<float> grm_buf(num_byte_buffer);
    std::vector<float> grm_out_buffer(num_byte_buffer);
    float *cur_grm_pos, *cur_out_pos;
    for(int part_index = 0; part_index != index_grm_pairs.size(); part_index++){
        if(fread(grm_buf.data(), sizeof(float), byte_part_grms[part_index], sFile) != byte_part_grms[part_index]){
            LOGGER.e(0, "Failed to read GRM between line " + to_string(index_grm_pairs[part_index].first + 1) + " and "
                        + to_string(index_grm_pairs[part_index].second + 1));
        };
        if(part_index % 50 == 0){
            LOGGER.i(2, "Processing part " + to_string(part_index + 1));
        }
        cur_grm_pos = grm_buf.data();
        cur_out_pos = grm_out_buffer.data();
        for(int id1 = index_grm_pairs[part_index].first; id1 != index_grm_pairs[part_index].second + 1; id1++) {
            if (keeps.find(id1) == keeps.end()) {
                cur_grm_pos += id1 + 1;
                continue;
            }
            for (int id2 : index_keep) {
                if(id2 > id1) break;
                *(cur_out_pos++) = *(cur_grm_pos + id2);
            }
            cur_grm_pos += id1 + 1;
        }
        uint64_t size_write = cur_out_pos - grm_out_buffer.data();
        if(fwrite(grm_out_buffer.data(), sizeof(float), size_write, dFile) != size_write){
            LOGGER.e(0, "Failed to write the output file, please check the disk condition or permission");
        };
    }
}


GRM::GRM(Pheno* pheno, Marker* marker, const gcta_cli::GrmMakeOptions& opts, const std::string& out_prefix) {
    GRM::nMarkerBlock = opts.n_markers.value_or(1024);
    this->pheno = pheno;
    this->marker = marker;
    this->geno = std::make_unique<Geno>(pheno, marker, opts.geno, opts.marker_filter, out_prefix);
    // Pay attention to not reflect the newest changes of keep;
    this->index_keep = pheno->get_index_keep();
    this->num_parts = opts.part ? static_cast<uint32_t>(opts.part->first) : 1u;
    this->part = opts.part ? static_cast<uint32_t>(opts.part->second) : 1u;

    bBLAS = true;
    vector<uint32_t> parts = divide_parts_mem(pheno->count_keep(), num_parts);

    if (parts.size() != num_parts) {
        LOGGER.w(0, "cannot divide into " + to_string(num_parts) + ". Use " + to_string(parts.size()) + " instead.");
        this->num_parts = parts.size();
    }

    if (part > this->num_parts) {
        LOGGER.e(0, "cannot calculate when" + to_string(part) + " is larger than " + to_string(this->num_parts));
    } else {
        if (part == 1) {
            part_keep_indices = std::make_pair(0, parts[0]);
        } else {
            part_keep_indices = std::make_pair(parts[part - 2] + 1, parts[part - 1]);
        }
    }

    this->num_count_handle = 2;

    // init the grm and N buffer
    num_individual = part_keep_indices.second - part_keep_indices.first + 1;
    num_grm = ((uint64_t) part_keep_indices.first + part_keep_indices.second + 2) * num_individual / 2;

    uint64_t fill_grm = (num_grm + num_count_handle - 1) / num_count_handle * num_count_handle;
    uint64_t fill_N = fill_grm;
    if(bBLAS){
        fill_grm = (uint64_t)num_individual * (part_keep_indices.second + 1);
    }

    // In bBLAS mode grm and N are managed per-tile in processMakeGRM; allocate
    // them here only when not using tiling (no --GRM-tile-budget active, or
    // the budget is large enough that processMakeGRM will fall back to the
    // dense path itself -- see grm_tile_budget_covers_whole, used both here
    // and there so this can't disagree with that).
    grm_tile_budget_bytes = opts.tile_budget_gb
        ? (*opts.tile_budget_gb * 1024.0 * 1024.0 * 1024.0)
        : 0.0;
    if (opts.sparse_cutoff.has_value()) {
        sparse_cutoff = static_cast<float>(*opts.sparse_cutoff);
    }
    bool ctor_tiling_enabled = grm_tile_budget_bytes > 0.0;
    if(ctor_tiling_enabled && grm_tile_budget_covers_whole(
            static_cast<int>(part_keep_indices.first),
            static_cast<int>(part_keep_indices.second) + 1,
            grm_tile_budget_bytes / 12.0)){
        ctor_tiling_enabled = false;
    }
    if(!bBLAS || !ctor_tiling_enabled){
        grm = make_aligned_uninitialized<double, 64>(fill_grm);
        if(!grm){
            LOGGER.error(0, "can't allocate enough memory to store the (parted) GRM: {:.2f}GB required.",
                         fill_grm * sizeof(double) / (1024.0 * 1024.0 * 1024.0));
        }
        if(bBLAS){
            // grm here is a dense (grm_m x grm_n) buffer for cblas_dsyrk/dgemm, not the
            // packed triangle fill_grm was sized for below (see the bBLAS override
            // above) -- most of it is upper-triangle padding that's never read back.
            // Skip zeroing it: calculate_GRM_blas's first call for this pass uses
            // beta=0, which zero-initializes exactly the elements it writes (lower
            // triangle for dsyrk, full rectangle for dgemm) without faulting in the
            // untouched upper triangle. See grm_first_accum.
            grm_first_accum = true;
        } else {
            std::memset(grm.get(), 0, fill_grm * sizeof(double));
        }

        // N stays packed-triangular size even in bBLAS mode (fill_N was captured
        // before the bBLAS override above), so this memset has no waste to fix.
        N = make_aligned_uninitialized<uint32_t, 64>(fill_N);
        if(!N){
            LOGGER.error(0, "can't allocate enough memory to store (parted) N: {:.2f}GB required.",
                         fill_N * sizeof(uint32_t) / (1024.0 * 1024.0 * 1024.0));
        }
        std::memset(N.get(), 0, fill_N * sizeof(uint32_t));
    }

    // +64 pads sub_miss past the last valid index by one full cache line (64 × sizeof(uint32_t) = 256 bytes).
    // This prevents OMP threads from writing to adjacent cache lines when two threads operate on
    // neighbouring sample indices near the end of the array (false-sharing guard).
    sub_miss.assign(index_keep.size() + 64, 0);

    //calculate each index in pair thread;
    int num_thread = omp_get_max_threads();
    index_grm_pairs.reserve(num_thread);
    vector<uint32_t> thread_parts = divide_parts(part_keep_indices.first, part_keep_indices.second, num_thread);
    if(num_thread != thread_parts.size()){
        LOGGER.w(0, "cannot run in " + to_string(num_thread) + " threads. Use " + to_string(thread_parts.size()) + " instead");
    }

    index_grm_pairs.push_back(std::make_pair(part_keep_indices.first, thread_parts[0]));
    for(int index = 1; index != thread_parts.size(); index++){
        index_grm_pairs.push_back(std::make_pair(thread_parts[index - 1] + 1, thread_parts[index]));
    }

    isDominance = opts.dominance;
    isMtd = (opts.alg == gcta_cli::GrmAlg::VanRaden);

    // Cache constants derived from part_keep_indices
    grm_n = static_cast<int>(part_keep_indices.second) + 1;
    grm_m = static_cast<int>(part_keep_indices.second) - static_cast<int>(part_keep_indices.first) + 1;
    grm_s_n = grm_n - grm_m;
    grm_bytes_std_geno = static_cast<int>(sizeof(float)) * grm_n;
    // Round up to a multiple of 64 elements so every column of stdGeno / stdGeno_i8
    // is 64-byte aligned and has capacity >= index_keep.size() (for --make-grm-part).
    const int ld_base = std::max(grm_n, static_cast<int>(index_keep.size()));
    stdGenoLD = (ld_base + 63) / 64 * 64;
    // Pre-allocate scratch buffers for calculate_GRM_blas
    validIndexBuf.reserve(nMarkerBlock);
    {
        const int markerPerN = static_cast<int>(sizeof(uintptr_t) * CHAR_BIT);
        const int numNSampleBlock = (grm_n + markerPerN - 1) / markerPerN;
        // Size covers all marker blocks at once so N_thread can be called in a
        // single parallel region across all blocks rather than 16 separate ones.
        const int numNblockMax = (nMarkerBlock + markerPerN - 1) / markerPerN;
        sampleMissBuf.resize(static_cast<size_t>(numNSampleBlock) * markerPerN * numNblockMax);
    }

    //t_print(begin, "  INIT finished");

    string fstring = bBLAS ? " v2 " : " ";
    string com_string = string("Computing the ") + (isDominance ? "dominance " : "") + "genetic relationship matrix (GRM)" + fstring + " using " + std::to_string(nMarkerBlock) + " markers per block ...";
    LOGGER.i(0, com_string);
    LOGGER.i(0, "Subset " + to_string(part) + "/" + to_string(num_parts) + ", no. subject " + to_string(part_keep_indices.first + 1) + "-" + to_string(part_keep_indices.second + 1));
    LOGGER.i(1, to_string(num_individual) + " samples, " + to_string(marker->count_extract()) + " markers, " + to_string(num_grm) + " GRM elements");

    o_name = out_prefix;

    if(isDominance){
        o_name += ".d";
    }

    output_id();
}


void GRM::output_id() {
    vector<string> out_id = pheno->get_id(part_keep_indices.first, part_keep_indices.second);

    string o_grm_id = o_name + ".grm.id";
    std::ofstream grm_id(o_grm_id.c_str());

    if (!grm_id) { LOGGER.e(0, "cannot open the file [" + o_grm_id + "] to write"); }
    std::copy(out_id.begin(), out_id.end(), std::ostream_iterator<string>(grm_id, "\n"));
    grm_id.close();
    LOGGER.i(0, "IDs for the GRM file have been saved in the file [" + o_grm_id + "]");

}

void GRM::calculate_GRM_blas(uintptr_t *buf, std::span<const uint32_t> markerIndex){
    int num_marker = markerIndex.size();

    if (use_int8_grm) {
        // BED int8 path: counts-only pass (VPOPCNTDQ on Zen 4) to obtain
        // af / mean / sd / validity over all keepSampleCT samples. The
        // genotypes themselves are decoded later straight from the raw .bed
        // records into the K-major GEMM operand (decodeBedI8KMajor), only for
        // valid markers -- no marker-major stdGeno_i8 staging buffer.
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < num_marker; i++){
            GenoBufItem &item = gbufitems[i];
            item.extractedMarkerIndex = markerIndex[i];
            item.geno_ptr = nullptr;
            item.geno_f32_ptr = nullptr;
            item.geno_i8_ptr = nullptr;
            item.stats_only = true;
            item.lazy_missing = true;
            geno->getGenoDouble(buf, i, &item);
        }
    } else {
        float* const std_geno_base = std::assume_aligned<64>(stdGeno.get());
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < num_marker; i++){
            GenoBufItem &item = gbufitems[i];
            item.extractedMarkerIndex = markerIndex[i];
            item.geno_ptr = nullptr;
            item.geno_i8_ptr = nullptr;
            item.geno_f32_ptr = std_geno_base + static_cast<size_t>(i) * stdGenoLD;
            item.stats_only = false;
            item.lazy_missing = true;   // `missing` only valid when nValidN < keep count
            geno->getGenoDouble(buf, i, &item);
        }
    }

    validIndexBuf.clear();
    for(int i = 0; i < num_marker; i++){
        if(gbufitems[i].valid){
            validIndexBuf.push_back(i);
        }
    }

    int curNumValidMarkers = static_cast<int>(validIndexBuf.size());

    // When all markers in the block pass filters (curNumValidMarkers == num_marker),
    // every column is already decoded in-place at stdGeno / stdGeno_i8 + i * stdGenoLD.
    // Otherwise, compact valid columns forward in-place (validIndexBuf[k] >= k).
    if (curNumValidMarkers < num_marker) {
        if (use_int8_grm) {
            // Nothing to compact: decodeBedI8KMajor reads validIndexBuf directly.
        } else {
            float* const std_geno_base = std::assume_aligned<64>(stdGeno.get());
            for (int k = 0; k < curNumValidMarkers; ++k) {
                const int src_idx = validIndexBuf[k];
                if (src_idx != k) {
                    std::memcpy(std_geno_base + static_cast<size_t>(k) * stdGenoLD,
                                std_geno_base + static_cast<size_t>(src_idx) * stdGenoLD,
                                static_cast<size_t>(grm_bytes_std_geno));
                }
            }
        }
    }
    if(!grm_skip_global_state){
        for(int i = 0; i < curNumValidMarkers; i++){
            sd.push_back(gbufitems[validIndexBuf[i]].sd);
        }
    }

    constexpr int kTile = 8192;
    const bool first_accum = grm_first_accum;
    double* const grm_base = std::assume_aligned<64>(grm.get());

    if (use_int8_grm) {
        int8_t* const kmajor_base = std::assume_aligned<64>(stdGeno_i8_kmajor_.get());
        const int ldK = std::max(64, (curNumValidMarkers + 63) / 64 * 64);

        double S_mu2 = 0.0;
        for (int m = 0; m < curNumValidMarkers; ++m) {
            const double mu_m = gbufitems[validIndexBuf[m]].mean;
            mu_vec_[m] = mu_m;
            S_mu2 += mu_m * mu_m;
        }

        // Fused cache-blocked (64x64) decode of the raw .bed records of the
        // valid markers straight into K-contiguous row-major stdGeno_i8_kmajor_
        // (grm_n x ldK) with per-sample row-sum accumulation
        // r_raw_[i] = sum_m g_{im} * mu_m (same accumulation order as before).
        geno->decodeBedI8KMajor(buf, validIndexBuf.data(), curNumValidMarkers, grm_n,
                                kmajor_base, ldK, mu_vec_.data(), r_raw_.data());

        bool block_has_missing = false;
        const uint32_t full_keep_ct = static_cast<uint32_t>(index_keep.size());
        for (int m = 0; m < curNumValidMarkers; ++m) {
            if (gbufitems[validIndexBuf[m]].nValidN < full_keep_ct) {
                block_has_missing = true;
                break;
            }
        }

        marker_miss_idx_.clear();
        if (block_has_missing) {
            // Rare path (not taken for imputed data): decode the full int8
            // column + missing bitmask of each marker with missing calls into a
            // compact marker-major buffer used by the missing corrections.
            miss_col_of_.assign(static_cast<size_t>(curNumValidMarkers), -1);
            std::vector<int> miss_markers;
            for (int m = 0; m < curNumValidMarkers; ++m) {
                if (gbufitems[validIndexBuf[m]].nValidN < full_keep_ct) {
                    miss_col_of_[m] = static_cast<int>(miss_markers.size());
                    miss_markers.push_back(m);
                }
            }
            const int n_miss_cols = static_cast<int>(miss_markers.size());
            if (n_miss_cols > stdGeno_i8_cols_) {
                stdGeno_i8 = make_aligned_uninitialized<int8_t, 64>(
                    static_cast<size_t>(n_miss_cols) * static_cast<size_t>(stdGenoLD));
                if (!stdGeno_i8) {
                    LOGGER.e(0, "can't allocate enough memory for the int8 missing-genotype columns.");
                }
                stdGeno_i8_cols_ = n_miss_cols;
            }
            int8_t* const miss_cols = stdGeno_i8.get();
            #pragma omp parallel for schedule(dynamic, 4)
            for (int j = 0; j < n_miss_cols; ++j) {
                const int bi = validIndexBuf[miss_markers[j]];
                GenoBufItem &item = gbufitems[bi];
                item.stats_only = false;
                item.lazy_missing = false;
                item.geno_i8_ptr = miss_cols + static_cast<size_t>(j) * stdGenoLD;
                geno->getGenoDouble(buf, bi, &item);
                item.geno_i8_ptr = nullptr;
            }
            std::memcpy(r_tilde_.data(), r_raw_.data(), static_cast<size_t>(grm_n) * sizeof(double));
            const int num_words = (grm_n + 63) / 64;
            const uintptr_t last_word_mask = (grm_n % 64 == 0)
                ? ~static_cast<uintptr_t>(0)
                : ((static_cast<uintptr_t>(1) << (grm_n % 64)) - 1);
            marker_miss_offset_[0] = 0;
            for (int m = 0; m < curNumValidMarkers; ++m) {
                const GenoBufItem& item = gbufitems[validIndexBuf[m]];
                if (item.nValidN < full_keep_ct && !item.missing.empty()) {
                    const double mu2 = mu_vec_[m] * mu_vec_[m];
                    const uintptr_t* miss_words = item.missing.data();
                    for (int w = 0; w < num_words; ++w) {
                        uintptr_t word = miss_words[w];
                        if (w == num_words - 1) word &= last_word_mask;
                        while (word != 0) {
                            const int bit = std::countr_zero(word);
                            const int s = w * 64 + bit;
                            r_tilde_[s] += mu2;
                            marker_miss_idx_.push_back(s);
                            word &= (word - 1);
                        }
                    }
                }
                marker_miss_offset_[m + 1] = static_cast<int>(marker_miss_idx_.size());
            }
            if (marker_miss_idx_.empty()) {
                block_has_missing = false;
            } else {
                // Build global per-sample missing-marker CSR once per marker block
                // instead of binary-searching marker_miss_idx_ on every column tile.
                std::fill(sample_miss_offset_.begin(), sample_miss_offset_.begin() + grm_n + 1, 0);
                for (int s : marker_miss_idx_) {
                    sample_miss_offset_[s + 1]++;
                }
                for (int s = 0; s < grm_n; ++s) {
                    sample_miss_offset_[s + 1] += sample_miss_offset_[s];
                }
                sample_miss_m_.resize(sample_miss_offset_[grm_n]);
                std::vector<int> write_pos(sample_miss_offset_.begin(),
                                           sample_miss_offset_.begin() + grm_n);
                for (int m = 0; m < curNumValidMarkers; ++m) {
                    const int mb = marker_miss_offset_[m];
                    const int me = marker_miss_offset_[m + 1];
                    for (int idx = mb; idx < me; ++idx) {
                        const int s = marker_miss_idx_[idx];
                        sample_miss_m_[write_pos[s]++] = m;
                    }
                }
            }
        }
        const double* r_row_vec = block_has_missing ? r_tilde_.data() : r_raw_.data();

        std::vector<int> row_miss_b;
        std::vector<int> row_miss_e;
        std::vector<int> active_row_miss_markers;
        if (block_has_missing) {
            row_miss_b.resize(curNumValidMarkers);
            row_miss_e.resize(curNumValidMarkers);
            active_row_miss_markers.reserve(curNumValidMarkers);
        }

        auto prepare_row_missing = [&](int global_r0, int rows) {
            if (!block_has_missing) return;
            #pragma omp parallel for schedule(static) if (curNumValidMarkers >= 256)
            for (int m = 0; m < curNumValidMarkers; ++m) {
                const int mb = marker_miss_offset_[m];
                const int me = marker_miss_offset_[m + 1];
                if (mb == me) {
                    row_miss_b[m] = row_miss_e[m] = mb;
                    continue;
                }
                const int b = static_cast<int>(std::lower_bound(
                    marker_miss_idx_.begin() + mb,
                    marker_miss_idx_.begin() + me,
                    global_r0) - marker_miss_idx_.begin());
                const int e = static_cast<int>(std::lower_bound(
                    marker_miss_idx_.begin() + b,
                    marker_miss_idx_.begin() + me,
                    global_r0 + rows) - marker_miss_idx_.begin());
                row_miss_b[m] = b;
                row_miss_e[m] = e;
            }
            active_row_miss_markers.clear();
            for (int m = 0; m < curNumValidMarkers; ++m) {
                if (row_miss_b[m] < row_miss_e[m]) {
                    active_row_miss_markers.push_back(m);
                }
            }
        };

        auto accum_col_with_centering = [&](int j, int row_start, int len,
                                            const int32_t* src_col, double* dst_col) {
            const double r_j_plus_S = S_mu2 - r_raw_[j];
            const double* r_row_slice = r_row_vec + row_start;
            if (first_accum) {
                for (int r = 0; r < len; ++r) {
                    dst_col[r] = static_cast<double>(src_col[r]) - r_row_slice[r] + r_j_plus_S;
                }
            } else {
                for (int r = 0; r < len; ++r) {
                    dst_col[r] += static_cast<double>(src_col[r]) - r_row_slice[r] + r_j_plus_S;
                }
            }
            if (!block_has_missing) return;

            // Step 2: markers missing in column j (indexed directly from block-level CSR)
            const int cm_b = sample_miss_offset_[j];
            const int cm_e = sample_miss_offset_[j + 1];
            for (int idx = cm_b; idx < cm_e; ++idx) {
                const int m = sample_miss_m_[idx];
                const double mu_m = mu_vec_[m];
                const double neg_mu2 = -mu_m * mu_m;
                const int8_t* col_m = stdGeno_i8.get()
                                      + static_cast<size_t>(miss_col_of_[m]) * stdGenoLD + row_start;
                for (int r = 0; r < len; ++r) {
                    dst_col[r] += static_cast<double>(col_m[r]) * mu_m + neg_mu2;
                }
                for (int r_idx = row_miss_b[m]; r_idx < row_miss_e[m]; ++r_idx) {
                    const int s = marker_miss_idx_[r_idx];
                    if (s >= row_start) {
                        dst_col[s - row_start] -= neg_mu2;
                    }
                }
            }

            // Step 3: markers missing in rows [row_start, row_start + len)
            // Read sample j's markers from L1-contiguous kmajor row.
            const int8_t* row_j_kmajor = kmajor_base + static_cast<size_t>(j) * ldK;
            for (int m : active_row_miss_markers) {
                const int8_t g_jm = row_j_kmajor[m];
                if (g_jm != 0) {
                    const double corr = mu_vec_[m] * static_cast<double>(g_jm);
                    for (int r_idx = row_miss_b[m]; r_idx < row_miss_e[m]; ++r_idx) {
                        const int s = marker_miss_idx_[r_idx];
                        if (s >= row_start) {
                            dst_col[s - row_start] += corr;
                        }
                    }
                }
            }
        };

        auto accum_i8_gemm_to_f64 = [&](int row_base, int M, int col_base, int N_cols, int K,
                                        double* C, int ldc) {
            for (int c0 = 0; c0 < N_cols; c0 += kTile) {
                const int cols = std::min(kTile, N_cols - c0);
                const int global_c0 = col_base + c0;
                for (int r0 = 0; r0 < M; r0 += kTile) {
                    const int rows = std::min(kTile, M - r0);
                    const int global_r0 = row_base + r0;
                    if (K > 0) {
                        prepare_row_missing(global_r0, rows);
                        gcta_gemm_s8s8s32_kmajor_nt(
                            rows, cols, K,
                            kmajor_base + static_cast<size_t>(global_r0) * ldK, ldK,
                            kmajor_base + static_cast<size_t>(global_c0) * ldK,
                            grm_i32_tile_.data(), rows,
                            i8_scratch_A_f32_, i8_scratch_B_f32_, grm_f32_tile_);
                    }
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        double* dst_col = C + r0 + static_cast<size_t>(c0 + c) * ldc;
                        if (K == 0) {
                            if (first_accum)
                                std::memset(dst_col, 0, static_cast<size_t>(rows) * sizeof(double));
                            continue;
                        }
                        const int32_t* src_col = grm_i32_tile_.data() + static_cast<size_t>(c) * rows;
                        accum_col_with_centering(global_c0 + c, global_r0, rows, src_col, dst_col);
                    }
                }
            }
        };

        auto accum_i8_syrk_lower_to_f64 = [&](int base, int N_dim, int K,
                                              double* C, int ldc) {
            for (int c0 = 0; c0 < N_dim; c0 += kTile) {
                const int cols = std::min(kTile, N_dim - c0);
                const int global_c0 = base + c0;
                if (K > 0) {
                    prepare_row_missing(global_c0, cols);
                    gcta_syrk_s8s8s32_lower_kmajor_nt(
                        cols, K,
                        kmajor_base + static_cast<size_t>(global_c0) * ldK, ldK,
                        grm_i32_tile_.data(), cols,
                        i8_scratch_A_f32_, grm_f32_tile_);
                }
                #pragma omp parallel for schedule(guided)
                for (int c = 0; c < cols; ++c) {
                    double* dst_col = C + (c0 + c) + static_cast<size_t>(c0 + c) * ldc;
                    const int len = cols - c;
                    if (K == 0) {
                        if (first_accum)
                            std::memset(dst_col, 0, static_cast<size_t>(len) * sizeof(double));
                        continue;
                    }
                    const int32_t* src_col = grm_i32_tile_.data() + c + static_cast<size_t>(c) * cols;
                    const int j = global_c0 + c;
                    accum_col_with_centering(j, j, len, src_col, dst_col);
                }
                const int rem_rows = N_dim - (c0 + cols);
                if (rem_rows > 0) {
                    accum_i8_gemm_to_f64(base + c0 + cols, rem_rows,
                                         global_c0, cols, K,
                                         C + (c0 + cols) + static_cast<size_t>(c0) * ldc, ldc);
                }
            }
        };

        if (grm_tiling_enabled) {
            if (grm_tile_rs > 0) {
                accum_i8_gemm_to_f64(grm_tile_rs, grm_tile_rows, 0, grm_tile_rs, curNumValidMarkers,
                                     grm_base, grm_tile_rows);
            }
            double *grm_diag = grm_base + static_cast<size_t>(grm_tile_rs) * grm_tile_rows;
            accum_i8_syrk_lower_to_f64(grm_tile_rs, grm_tile_rows, curNumValidMarkers,
                                       grm_diag, grm_tile_rows);
        } else if (part_keep_indices.first == 0) {
            accum_i8_syrk_lower_to_f64(0, grm_n, curNumValidMarkers,
                                       grm_base, grm_n);
        } else {
            const int row_base = static_cast<int>(part_keep_indices.first);
            accum_i8_gemm_to_f64(row_base, grm_m, 0, grm_s_n, curNumValidMarkers,
                                 grm_base, grm_m);
            double *grm_start = grm_base + static_cast<uint64_t>(grm_s_n) * grm_m;
            accum_i8_syrk_lower_to_f64(row_base, grm_m, curNumValidMarkers,
                                       grm_start, grm_m);
        }
    } else {
        const float* const std_geno_base = std::assume_aligned<64>(stdGeno.get());
        // Mixed-precision BLAS3: run cblas_ssyrk / cblas_sgemm (beta=0.0f) on L3-sized
        // 4096x4096 float32 tiles (grm_f32_tile_), then widen and accumulate into
        // double* grm. On the first accumulation for this grm buffer (grm_first_accum),
        // write directly to unzeroed memory instead of accumulating so untouched upper-
        // triangle pages of grm stay unfaulted in physical RAM.
        auto accum_sgemm_to_f64 = [&](int M, int N_cols, int K,
                                      const float* A, int lda,
                                      const float* B, int ldb,
                                      double* C, int ldc) {
            for (int c0 = 0; c0 < N_cols; c0 += kTile) {
                const int cols = std::min(kTile, N_cols - c0);
                for (int r0 = 0; r0 < M; r0 += kTile) {
                    const int rows = std::min(kTile, M - r0);
                    if (K > 0) {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    rows, cols, K,
                                    1.0f,
                                    A + r0, lda,
                                    B + c0, ldb,
                                    0.0f,
                                    grm_f32_tile_.data(), rows);
                    }
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < cols; ++c) {
                        double* dst_col = C + r0 + static_cast<size_t>(c0 + c) * ldc;
                        if (K == 0) {
                            if (first_accum)
                                std::memset(dst_col, 0, static_cast<size_t>(rows) * sizeof(double));
                            continue;
                        }
                        const float* src_col = grm_f32_tile_.data() + static_cast<size_t>(c) * rows;
                        if (first_accum) {
                            for (int r = 0; r < rows; ++r)
                                dst_col[r] = static_cast<double>(src_col[r]);
                        } else {
                            for (int r = 0; r < rows; ++r)
                                dst_col[r] += static_cast<double>(src_col[r]);
                        }
                    }
                }
            }
        };

        auto accum_ssyrk_lower_to_f64 = [&](int N_dim, int K,
                                            const float* A, int lda,
                                            double* C, int ldc) {
            for (int c0 = 0; c0 < N_dim; c0 += kTile) {
                const int cols = std::min(kTile, N_dim - c0);
                if (K > 0) {
                    cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                cols, K,
                                1.0f,
                                A + c0, lda,
                                0.0f,
                                grm_f32_tile_.data(), cols);
                }
                #pragma omp parallel for schedule(guided)
                for (int c = 0; c < cols; ++c) {
                    double* dst_col = C + (c0 + c) + static_cast<size_t>(c0 + c) * ldc;
                    const int len = cols - c;
                    if (K == 0) {
                        if (first_accum)
                            std::memset(dst_col, 0, static_cast<size_t>(len) * sizeof(double));
                        continue;
                    }
                    const float* src_col = grm_f32_tile_.data() + c + static_cast<size_t>(c) * cols;
                    if (first_accum) {
                        for (int r = 0; r < len; ++r)
                            dst_col[r] = static_cast<double>(src_col[r]);
                    } else {
                        for (int r = 0; r < len; ++r)
                            dst_col[r] += static_cast<double>(src_col[r]);
                    }
                }
                const int rem_rows = N_dim - (c0 + cols);
                if (rem_rows > 0) {
                    accum_sgemm_to_f64(rem_rows, cols, K,
                                       A + c0 + cols, lda,
                                       A + c0, lda,
                                       C + (c0 + cols) + static_cast<size_t>(c0) * ldc, ldc);
                }
            }
        };

        if(grm_tiling_enabled){
            if(grm_tile_rs > 0){
                accum_sgemm_to_f64(grm_tile_rows, grm_tile_rs, curNumValidMarkers,
                                   std_geno_base + grm_tile_rs, stdGenoLD,
                                   std_geno_base, stdGenoLD,
                                   grm_base, grm_tile_rows);
            }
            double *grm_diag = grm_base + static_cast<size_t>(grm_tile_rs) * grm_tile_rows;
            accum_ssyrk_lower_to_f64(grm_tile_rows, curNumValidMarkers,
                                     std_geno_base + grm_tile_rs, stdGenoLD,
                                     grm_diag, grm_tile_rows);
        } else if(part_keep_indices.first == 0){
            accum_ssyrk_lower_to_f64(grm_n, curNumValidMarkers,
                                     std_geno_base, stdGenoLD,
                                     grm_base, grm_n);
        }else{
            accum_sgemm_to_f64(grm_m, grm_s_n, curNumValidMarkers,
                               std_geno_base + part_keep_indices.first, stdGenoLD,
                               std_geno_base, stdGenoLD,
                               grm_base, grm_m);
            double *grm_start = grm_base + (uint64_t)grm_s_n * grm_m;
            accum_ssyrk_lower_to_f64(grm_m, curNumValidMarkers,
                                     std_geno_base + part_keep_indices.first, stdGenoLD,
                                     grm_start, grm_m);
        }
    }
    grm_first_accum = false;  // every later marker block for this buffer accumulates

    // Zero-missing fast path (post-imputation data): every contribution to
    // sub_miss and N below is popcount(0) == 0, so skip the bit transposes,
    // popcounts and N_thread passes entirely -- bit-identical result.
    const uint32_t keep_ct_all = static_cast<uint32_t>(index_keep.size());
    bool any_missing_in_block = false;
    for (int k = 0; k < curNumValidMarkers; ++k) {
        if (gbufitems[validIndexBuf[k]].nValidN < keep_ct_all) {
            any_missing_in_block = true;
            break;
        }
    }
    if (!any_missing_in_block) {
        if(!grm_skip_global_state){
            finished_marker += num_marker;
            numValidMarkers += curNumValidMarkers;
        }
        return;
    }

    const int markerPerN = sizeof(uintptr_t) * CHAR_BIT;
    const int numNblock = (curNumValidMarkers + markerPerN - 1) / markerPerN;
    const int numNSampleBlock = (grm_n + markerPerN - 1) / markerPerN;
    const int blockStride = numNSampleBlock * markerPerN;

    // Each sample block j owns disjoint 64-element slices in both sampleMissBuf
    // (512-byte aligned per block) and sub_miss (256-byte aligned), so transposing
    // and accumulating sub_miss across j is completely lock-free and race-free.
    #pragma omp parallel
    {
        std::array<uintptr_t, 64> transposeInput{};
        alignas(32) plink2::VecW transposeScratch[plink2::kPglBitTransposeBufvecs];

        #pragma omp for schedule(static)
        for(int j = 0; j < numNSampleBlock; j++){
            const int baseMissIndex = j * markerPerN;
            for(int i = 0; i < numNblock; i++){
                const int baseMarkerIndex = markerPerN * i;
                const int lastValidIndex  = std::min(markerPerN * (i + 1), curNumValidMarkers);
                uintptr_t *block_buf = sampleMissBuf.data() + i * blockStride;
                if (markerPerN == 64) {
                    for(int k = baseMarkerIndex; k < lastValidIndex; k++){
                        const GenoBufItem &it = gbufitems[validIndexBuf[k]];
                        transposeInput[static_cast<std::size_t>(k - baseMarkerIndex)] =
                            (it.nValidN < keep_ct_all) ? it.missing[j] : 0UL;
                    }
                    for(int k = lastValidIndex; k < markerPerN * (i + 1); k++){
                        transposeInput[static_cast<std::size_t>(k - baseMarkerIndex)] = 0UL;
                    }
                    // Transpose [marker x sample] missing bits into [sample x marker].
                    // Bit order inside each word is irrelevant for downstream AND+popcount.
                    plink2::TransposeBitblock(
                        transposeInput.data(), 1, 1,
                        static_cast<uint32_t>(markerPerN), static_cast<uint32_t>(markerPerN),
                        &block_buf[baseMissIndex], transposeScratch);
                } else {
                    for(int k = baseMarkerIndex; k < lastValidIndex; k++){
                        const GenoBufItem &it = gbufitems[validIndexBuf[k]];
                        block_buf[baseMissIndex + k - baseMarkerIndex] =
                            (it.nValidN < keep_ct_all) ? it.missing[j] : 0UL;
                    }
                    for(int k = lastValidIndex; k < markerPerN * (i + 1); k++){
                        block_buf[baseMissIndex + k - baseMarkerIndex] = 0UL;
                    }
                }
                if(!grm_skip_global_state){
                    for(int k = baseMissIndex; k < baseMissIndex + markerPerN; k++){
                        sub_miss[k] += std::popcount(block_buf[k]);
                    }
                }
            }
        }

        // Reuse the same parallel region: each thread processes its pair range across all blocks.
        #pragma omp for schedule(static)
        for(int index = 0; index < (int)index_grm_pairs.size(); index++){
            const auto index_pair = index_grm_pairs[index];
            for(int i = 0; i < numNblock; i++){
                N_thread(index_pair.first, index_pair.second, sampleMissBuf.data() + i * blockStride);
            }
        }
    }

    if(!grm_skip_global_state){
        finished_marker += num_marker;
        numValidMarkers += curNumValidMarkers;
    }
}

void GRM::deduce_GRM(){
    LOGGER.i(0, "The GRM computation is completed.");
    float thresh = -99.0f;
    bool isSparse = false;
    if(sparse_cutoff.has_value()){
        thresh = *sparse_cutoff;
        isSparse = true;
        LOGGER.i(0, "Saving sparse GRM with a cutoff " + to_string(thresh) + "...");
    }else{
        LOGGER.i(0, "Saving GRM...");
    }

    UniqueFile grm_out;
    UniqueFile N_out;
    if(isSparse){
        grm_out = open_file(o_name + ".grm.sp", "wb");
        if(!grm_out){
            LOGGER.error(0, "can't open {}.grm.sp to write", o_name);
        }
    }else{
        grm_out = open_file(o_name + ".grm.bin", "wb");
        N_out   = open_file(o_name + ".grm.N.bin", "wb");
        if((!grm_out) || (!N_out)){
            LOGGER.error(0, "can't open {}.grm.bin or .grm.N.bin to write", o_name);
        }
        // 4 MiB matches a typical Lustre stripe size and avoids sub-stripe metadata ops.
        setvbuf(grm_out.get(), nullptr, _IOFBF, 4 << 20);
        setvbuf(N_out.get(),   nullptr, _IOFBF, 4 << 20);
    }

    float mtd_weight = 1.0f;
    if(isMtd){
        float weight = 0.0f;
        if(!isDominance){
            for(uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i];
        }else{
            for(uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i] * sd[i];
        }
        if(weight > 0.0f) mtd_weight = static_cast<float>(numValidMarkers) / weight;
    }

    // Set tile bounds to cover the entire [part_keep_indices.first, part_keep_indices.second + 1)
    // pass and delegate to flush_grm_tile (grm_tiling_enabled == false selects packed N[] indexing).
    grm_tile_rs   = static_cast<int>(part_keep_indices.first);
    grm_tile_re   = static_cast<int>(part_keep_indices.second) + 1;
    grm_tile_rows = grm_m;
    grm_tile_cols = grm_n;

    std::vector<float> w_N;
    std::vector<float> grm_block;
    std::string sparse_buf;
    flush_grm_tile(grm_out.get(), N_out.get(), thresh, isSparse, mtd_weight,
                   w_N, grm_block, sparse_buf);

    grm_out.reset();
    N_out.reset();
    if(!isSparse){
        LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.bin]");
        LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + o_name + ".grm.N.bin]");
    }else{
        LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.sp]");
    }
}


// Write one tile's (or full non-tiled pass's) normalised rows to already-open output files.
// Uses grm (tile_rows × tile_cols column-major doubles) and N from object state:
//   - grm_tiling_enabled == true:  N is rectangular (tile_rows × tile_cols)
//   - grm_tiling_enabled == false: N is packed lower-triangular starting at tile_rs
//
// w_N/grm_block/sparse_buf are caller-owned scratch, sized once by
// processMakeGRM to the worst-case (final, widest) tile and reused across calls.
void GRM::flush_grm_tile(FILE *grm_out, FILE *N_out,
                         float thresh, bool isSparse, float mtd_weight,
                         std::vector<float>& w_N,
                         std::vector<float>& grm_block, std::string& sparse_buf){
    const int tile_rs   = grm_tile_rs;
    const int tile_re   = grm_tile_re;
    const int tile_rows = grm_tile_rows;
    const int tile_cols = grm_tile_cols;   // == tile_re

    if (!isSparse && static_cast<int>(w_N.size()) < tile_re) w_N.resize(tile_re);

    constexpr int BLAS_OUT_BLOCK = 1024;

    // Worst-case block for THIS tile: BLAS_OUT_BLOCK rows × tile_cols cols.
    const size_t needed = static_cast<size_t>(BLAS_OUT_BLOCK) * tile_cols;
    if (grm_block.size() < needed) grm_block.resize(needed);
    if(isSparse) sparse_buf.reserve(std::max(sparse_buf.capacity(),
        static_cast<size_t>(BLAS_OUT_BLOCK) * tile_cols / 20 * 25));

    const double* const grm_base = std::assume_aligned<64>(grm.get());
    const uint32_t* const N_base = std::assume_aligned<64>(N.get());

    const auto get_N_row = [&](int pair1) -> const uint32_t* {
        if (grm_tiling_enabled) {
            return N_base + static_cast<size_t>(pair1 - tile_rs) * tile_cols;
        }
        const size_t start_pos = static_cast<size_t>(pair1 + 1 + tile_rs) *
                                 static_cast<size_t>(pair1 - tile_rs) / 2;
        return N_base + start_pos;
    };

    for(int block_start = tile_rs; block_start < tile_re; block_start += BLAS_OUT_BLOCK){
        const int block_end  = std::min(block_start + BLAS_OUT_BLOCK, tile_re);
        const int block_rows = block_end - block_start;
        const int block_cols = block_end;           // lower-triangle max column
        const int local_block_start = block_start - tile_rs;

        // 1. Parallel column-major double -> row-major float transpose of the
        //    needed lower-triangle entries (pair2 <= pair1, i.e. c <= block_start + r).
        #pragma omp parallel for schedule(static)
        for(int c = 0; c < block_cols; ++c){
            const double *col_src = grm_base + static_cast<size_t>(c) * tile_rows + local_block_start;
            const int r_start = std::max(0, c - block_start);
            for(int r = r_start; r < block_rows; ++r){
                grm_block[static_cast<size_t>(r) * block_cols + c] = static_cast<float>(col_src[r]);
            }
        }

        // 2. Parallel in-place normalisation by pairwise non-missing count sub_N.
        #pragma omp parallel for schedule(guided)
        for(int pair1 = block_start; pair1 < block_end; ++pair1){
            const int local_r = pair1 - block_start;
            const uint32_t sub_miss1 = numValidMarkers - sub_miss[pair1];
            float *row = grm_block.data() + static_cast<size_t>(local_r) * block_cols;
            const uint32_t *N_row = get_N_row(pair1);
            for(int pair2 = 0; pair2 <= pair1; ++pair2){
                const uint32_t sub_N = N_row[pair2] + sub_miss1 - sub_miss[pair2];
                row[pair2] = sub_N ? (row[pair2] / static_cast<float>(sub_N)) * mtd_weight : 0.0f;
            }
        }

        if(isSparse) sparse_buf.clear();

        // 3. Sequential output emission.
        for(int pair1 = block_start; pair1 < block_end; pair1++){
            const int local_r = pair1 - block_start;
            const float *row  = grm_block.data() + static_cast<size_t>(local_r) * block_cols;

            if(!isSparse){
                const uint32_t sub_miss1 = numValidMarkers - sub_miss[pair1];
                const uint32_t *N_row    = get_N_row(pair1);
                for(int pair2 = 0; pair2 <= pair1; ++pair2){
                    w_N[pair2] = static_cast<float>(N_row[pair2] + sub_miss1 - sub_miss[pair2]);
                }
                fwrite(row,        sizeof(float), pair1 + 1, grm_out);
                fwrite(w_N.data(), sizeof(float), pair1 + 1, N_out);
            } else {
                for(int i = 0; i <= pair1; i++){
                    if(row[i] >= thresh){
                        std::format_to(std::back_inserter(sparse_buf),
                            "{}\t{}\t{:.{}g}\n",
                            pair1, i, row[i],
                            std::numeric_limits<float>::digits10 + 2);
                    }
                }
            }
        }
        if(isSparse && !sparse_buf.empty())
            fputs(sparse_buf.c_str(), grm_out);
    }
}


// N_thread: two distinct N[] layouts, selected by grm_tiling_enabled
// (true when an auto-sized --GRM-tile-budget is active).
//
//   grm_tiling_enabled  (tiled path):
//     N is a RECTANGULAR buffer of shape (grm_tile_rows × grm_tile_cols), allocated in
//     processMakeGRM with make_aligned_uninitialized and zeroed per tile.
//     Index: N[(pair1 - grm_tile_rs) * grm_tile_cols + pair2]
//     pair2 ranges over [0, pair1] (lower triangle), but the stride is grm_tile_cols
//     (full width), not the triangular row length.  flush_grm_tile reads it the same way.
//
//   !grm_tiling_enabled  (non-tiled path):
//     N is a PACKED TRIANGULAR buffer: row pair1 occupies positions
//     [tri_offset(pair1), tri_offset(pair1) + pair1].
//     startPos = (pair1+1+first) * (pair1-first) / 2  is the triangular row offset.
//     deduce_GRM advances po_N by (pair1+1) after each row, consuming the packed layout.
//
// INVARIANT: these two paths must never be mixed.  The tiled path is always active
// when grm_tiling_enabled; the non-tiled path is the fallback otherwise.
// If the two code paths are ever merged, this layout difference must be resolved first.
void GRM::N_thread(int grm_index_from, int grm_index_to, const uintptr_t* cur_cmask){
    uint32_t* const N_base = std::assume_aligned<64>(N.get());
    if(grm_tiling_enabled){
        // Rectangular N layout: N[(pair1 - grm_tile_rs) * grm_tile_cols + pair2]
        uint32_t *po_N_row = N_base + static_cast<size_t>(grm_index_from - grm_tile_rs) * grm_tile_cols;
        const uintptr_t *p_cmask1 = cur_cmask + grm_index_from;
        for(int index_pair1 = grm_index_from; index_pair1 != grm_index_to + 1;
                index_pair1++, po_N_row += grm_tile_cols){
            uintptr_t cmask1 = *p_cmask1++;
            if(cmask1){
                int count = index_pair1 + 1;
                const uintptr_t *p_cmask2 = cur_cmask;
                uint32_t *po_N = po_N_row;
                int index_pair2 = 0;
                for(; index_pair2 + 3 < count; index_pair2 += 4){
                    uintptr_t c0 = p_cmask2[0];
                    uintptr_t c1 = p_cmask2[1];
                    uintptr_t c2 = p_cmask2[2];
                    uintptr_t c3 = p_cmask2[3];
                    po_N[0] += std::popcount(cmask1 & c0);
                    po_N[1] += std::popcount(cmask1 & c1);
                    po_N[2] += std::popcount(cmask1 & c2);
                    po_N[3] += std::popcount(cmask1 & c3);
                    p_cmask2 += 4;
                    po_N += 4;
                }
                for(; index_pair2 < count; index_pair2++){
                    uintptr_t cmask = cmask1 & (*p_cmask2++);
                    if(cmask) *po_N += std::popcount(cmask);
                    po_N++;
                }
            }
        }
        return;
    }

    uint64_t startPos = ((uint64_t)grm_index_from + 1 + part_keep_indices.first) * (grm_index_from - part_keep_indices.first) / 2;

    uint32_t *po_N_start = N_base + startPos;
    //for(int cur_block = 0; cur_block != Constants::NUM_MARKER_READ / num_cmask_block; cur_block++){
    //uint64_t *cur_cmask = cmask_buf + cur_block * index_keep.size();
    uint32_t *po_N = po_N_start;
    const uintptr_t *p_cmask1 = cur_cmask + grm_index_from;
    for(int index_pair1 = grm_index_from; index_pair1 != grm_index_to + 1; index_pair1++){
        const uintptr_t *p_cmask2 = cur_cmask;
        uintptr_t cmask1 = *p_cmask1++;
        if(cmask1){
            int count = index_pair1 + 1;
            int index_pair2 = 0;
            // Unrolled: process 4 pairs per iteration
            for(; index_pair2 + 3 < count; index_pair2 += 4){
                uintptr_t c0 = p_cmask2[0];
                uintptr_t c1 = p_cmask2[1];
                uintptr_t c2 = p_cmask2[2];
                uintptr_t c3 = p_cmask2[3];
                po_N[0] += std::popcount(cmask1 & c0);
                po_N[1] += std::popcount(cmask1 & c1);
                po_N[2] += std::popcount(cmask1 & c2);
                po_N[3] += std::popcount(cmask1 & c3);
                p_cmask2 += 4;
                po_N += 4;
            }
            // Remainder
            for(; index_pair2 < count; index_pair2++){
                uintptr_t cmask = cmask1 & (*p_cmask2++);
                if(cmask) *po_N += std::popcount(cmask);
                po_N++;
            }
        }else{
            po_N += index_pair1 + 1;
        }
    }
    //}
}

vector<uint32_t> GRM::divide_parts_mem(uint32_t n_sample, uint32_t num_parts){
    int parts = num_parts;
    vector<double> n_part(parts);
    vector<double> n_each_part(parts);
    n_part[0] = 1;
    n_each_part[0] = 1;
    for(int i = 1; i < parts; i++){
        double previous_n = n_part[i - 1];
        //double cur_n = (sqrt(previous_n * previous_n + 4) - previous_n) / 2.0;
        //double cur_n = (sqrt(16.0 * previous_n * previous_n + 36.0) - 4.0 * previous_n) / 6.0;
        double cur_n = (sqrt(36.0 * previous_n * previous_n + 100.0) - 6.0 * previous_n) / 10.0;
        n_each_part[i] = cur_n;
        n_part[i] = previous_n + cur_n;
    }

    double x1 = n_sample / n_part[parts - 1];
    vector<uint32_t> div_parts(parts);
    uint32_t total_num = 0;
    for(int i = 0 ; i < parts - 1; i++){
        uint32_t cur_n = round(n_each_part[i] * x1);
        total_num = cur_n + total_num;
        div_parts[i] = total_num - 1;
    }
    div_parts[parts - 1] = n_sample - 1;
    return div_parts;
}

vector<uint32_t> GRM::divide_parts(uint32_t from, uint32_t to, uint32_t num_parts){
    vector<uint64_t> num_indi_grms;
    num_indi_grms.reserve(to - from + 1);
    for(uint64_t index = from; index != to + 1; index++){
        num_indi_grms.push_back((index + 2 + from) * (index + 1 -  from) / 2);
    }

    uint64_t indi_parts = num_indi_grms[to - from] / num_parts;
    vector<uint32_t> parts;
    parts.reserve(num_parts);
    vector<uint64_t>::iterator upper;
    for(uint32_t index = 1; index <= num_parts; index++){
        upper = std::upper_bound(num_indi_grms.begin(), num_indi_grms.end(), indi_parts * index);
        if(upper != num_indi_grms.begin()){
            upper--;
            if(*upper < indi_parts * index){
                upper++;
            }
        }
        parts.push_back(upper - num_indi_grms.begin() + from );
    }
    parts.erase(unique(parts.begin(), parts.end()), parts.end());
    return parts;
}

void GRM::processMakeGRM(bool is_homogametic){
    gbufitems.assign(nMarkerBlock, GenoBufItem{});
    {
        const int markerPerN = static_cast<int>(sizeof(uintptr_t) * CHAR_BIT);
        const int numNSampleBlock = (grm_n + markerPerN - 1) / markerPerN;
        const int numNblockMax = (nMarkerBlock + markerPerN - 1) / markerPerN;
        sampleMissBuf.resize(static_cast<size_t>(numNSampleBlock) * markerPerN * numNblockMax);
    }
    use_int8_grm = bBLAS && isMtd && !isDominance && !is_homogametic
                   && (geno->getGenoFormat() == "BED")
                   && (std::getenv("GCTA_DISABLE_INT8_GRM") == nullptr);
    const int max_tile_dim = std::min(8192, std::max(1, grm_n));
    const size_t geno_block_elems = static_cast<size_t>(nMarkerBlock) * stdGenoLD;
    if (use_int8_grm) {
        grm_bytes_std_geno = static_cast<int>(sizeof(int8_t)) * grm_n;
        this->num_byte_geno = sizeof(int8_t) * static_cast<uint64_t>(geno_block_elems);
        // Marker-major int8 columns are only materialised (lazily, compactly)
        // for markers with missing calls; the GEMM operand is decoded directly.
        stdGeno_i8.reset();
        stdGeno_i8_cols_ = 0;
        const int max_ldK = std::max(64, (nMarkerBlock + 63) / 64 * 64);
        stdGeno_i8_kmajor_ = make_aligned_uninitialized<int8_t, 64>(
            static_cast<size_t>(max_ldK) * static_cast<size_t>(stdGenoLD));
        if (!stdGeno_i8_kmajor_) {
            LOGGER.e(0, "can't allocate enough memory for the transposed int8 genotype buffer.");
        }
        grm_i32_tile_.resize(static_cast<size_t>(max_tile_dim) * max_tile_dim);
        mu_vec_.resize(nMarkerBlock);
        r_raw_.resize(grm_n);
        r_tilde_.resize(grm_n);
        marker_miss_offset_.resize(nMarkerBlock + 1);
        sample_miss_offset_.resize(static_cast<size_t>(grm_n) + 1);
    } else {
        grm_bytes_std_geno = static_cast<int>(sizeof(float)) * grm_n;
        this->num_byte_geno = sizeof(float) * static_cast<uint64_t>(geno_block_elems);
        stdGeno = make_aligned_uninitialized<float, 64>(geno_block_elems);
        if (!stdGeno) {
            LOGGER.e(0, "can't allocate enough memory for the genotype buffer.");
        }
        grm_f32_tile_.resize(static_cast<size_t>(max_tile_dim) * max_tile_dim);
    }
    
    vector<function<void (uintptr_t *, std::span<const uint32_t>)>> callBacks;
    callBacks.push_back([this](uintptr_t *buf, std::span<const uint32_t> idx){
        calculate_GRM_blas(buf, idx);
    });
    geno->setGRMMode(true, isDominance);
    const bool isSTD = !isMtd;
    const vector<uint32_t> processIndex = is_homogametic
        ? marker->get_extract_index_X()
        : marker->get_extract_index_autosome();
    sd.reserve(processIndex.size());

    grm_tiling_enabled = grm_tile_budget_bytes > 0.0;

    if(grm_tiling_enabled){
        // If the budget covers the *whole* GRM's triangular
        // footprint in a single tile, fall back to dense.
        const int full_rs = static_cast<int>(part_keep_indices.first);
        const int full_re = static_cast<int>(part_keep_indices.second) + 1;
        const double budget_elems = grm_tile_budget_bytes / 12.0;
        if(grm_tile_budget_covers_whole(full_rs, full_re, budget_elems)){
            LOGGER.info(0, "--GRM-tile-budget ({:.3f} GB) covers the whole GRM in one tile; using the dense path instead.",
                        grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0));
            grm_tiling_enabled = false;
        }
    }

    if(bBLAS && grm_tiling_enabled){
        // ── Block-tiled GRM: cap the grm+N tile buffer to --GRM-tile-budget
        // bytes, auto-sizing rows per tile since the lower-triangle column
        // width grows as tile_rs increases — this keeps memory usage roughly
        // uniform across tiles instead of shrinking-then-huge. ──

        float thresh = -99.0f;
        bool isSparse = false;
        if(sparse_cutoff.has_value()){
            thresh   = *sparse_cutoff;
            isSparse = true;
            LOGGER.i(0, "Computing sparse GRM with a cutoff " + to_string(thresh) + "...");
        } else {
            LOGGER.i(0, "Computing tiled GRM...");
        }

        UniqueFile grm_out;
        UniqueFile N_out;
        if(isSparse){
            grm_out = open_file(o_name + ".grm.sp", "wb");
            if(!grm_out) LOGGER.error(0, "can't open {}.grm.sp to write", o_name);
        } else {
            grm_out = open_file(o_name + ".grm.bin", "wb");
            N_out   = open_file(o_name + ".grm.N.bin", "wb");
            if(!grm_out || !N_out)
                LOGGER.error(0, "can't open {}.grm.bin or .grm.N.bin to write", o_name);
            // 4 MiB matches a typical Lustre stripe size and avoids sub-stripe metadata ops.
            setvbuf(grm_out.get(), nullptr, _IOFBF, 4 << 20);
            setvbuf(N_out.get(),   nullptr, _IOFBF, 4 << 20);
        }

        const int num_thread = omp_get_max_threads();
        const int full_rs = static_cast<int>(part_keep_indices.first);
        const int full_re = static_cast<int>(part_keep_indices.second) + 1;

        // Precompute every tile's [rs, re) boundary up front. This validates the
        // whole run (including the budget-too-small error case below) before any
        // genotype scanning starts.
        // 12 bytes/element = 8 (double grm) + 4 (uint32 N).
        const double budget_elems = grm_tile_budget_bytes / 12.0;
        vector<std::pair<int,int>> tile_ranges;
        int rs = full_rs;
        while(rs < full_re){
            const int re = solve_tile_budget_end(rs, full_re, budget_elems);
            if(re <= rs){
                const uint64_t need_bytes = static_cast<uint64_t>(rs + 1) * 12ULL;
                LOGGER.error(0, "--GRM-tile-budget is too small: the next row (sample offset {}) alone needs {:.3f} GB for its GRM+N row, but the budget is only {:.3f} GB. Increase --GRM-tile-budget.",
                             rs,
                             need_bytes / (1024.0 * 1024.0 * 1024.0),
                             grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0));
            }
            tile_ranges.push_back({rs, re});
            rs = re;
        }
        LOGGER.info(0, "Block-tiled GRM (memory budget {:.3f} GB): {} tile(s), rows/tile ranging {}-{}.",
                    grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0),
                    tile_ranges.size(),
                    tile_ranges.front().second - tile_ranges.front().first,
                    tile_ranges.back().second - tile_ranges.back().first);

        // flush_grm_tile's output-formatting scratch, sized once to the final
        // (widest) tile -- tile_ranges is monotonically widening in `re`, so
        // tile_ranges.back() is always the widest-column case.
        constexpr int BLAS_OUT_BLOCK = 1024;
        const int widest_tile_cols = tile_ranges.back().second;
        std::vector<float> flush_w_N(widest_tile_cols);
        std::vector<float> flush_grm_block(static_cast<size_t>(BLAS_OUT_BLOCK) * widest_tile_cols);
        std::string flush_sparse_buf;

        sub_miss.assign(index_keep.size() + 64, 0);
        numValidMarkers = 0;
        finished_marker = 0;
        sd.clear();
        grm_skip_global_state = false;  // first tile must accumulate global state
        float mtd_weight = 1.0f;        // computed once, after the first tile's decode (see below)

        for(const auto &range : tile_ranges){
            const int tile_rs = range.first;
            const int tile_re = range.second;
            grm_tile_rs   = tile_rs;
            grm_tile_re   = tile_re;
            grm_tile_rows = tile_re - tile_rs;
            grm_tile_cols = tile_re;   // lower-triangle: widest row needs cols [0, tile_re-1]

            // Fresh, exactly-sized allocation for this tile only
            const size_t tile_rect_elems = static_cast<size_t>(grm_tile_rows) * grm_tile_cols;
            grm = make_aligned_uninitialized<double, 64>(tile_rect_elems);
            if(!grm)
                LOGGER.e(0, "Can't allocate GRM tile buffer.");
            N = make_aligned_uninitialized<uint32_t, 64>(tile_rect_elems);
            if(!N)
                LOGGER.e(0, "Can't allocate N tile buffer.");

            const double drs = tile_rs, dre = tile_re;
            const size_t tile_elems = static_cast<size_t>((dre * (dre + 1.0) - drs * (drs + 1.0)) / 2.0);
            const double tile_gb    = tile_elems * 12.0 / (1024.0 * 1024.0 * 1024.0);
            LOGGER.info(0, "  Tile rows {}-{} ({:.2f} GB grm+N)", tile_rs, tile_re - 1, tile_gb);

            grm_first_accum = true;

            for(int r = 0; r < grm_tile_rows; r++){
                const int pair1 = tile_rs + r;               // global row index
                const int valid_cols = pair1 + 1;             // N_thread writes cols [0, pair1]
                std::memset(N.get() + static_cast<size_t>(r) * grm_tile_cols, 0,
                            static_cast<size_t>(valid_cols) * sizeof(uint32_t));
            }

            index_grm_pairs.clear();
            vector<uint32_t> tile_parts = divide_parts(tile_rs, tile_re - 1, num_thread);
            index_grm_pairs.push_back(std::make_pair(tile_rs, tile_parts[0]));
            for(int i = 1; i < (int)tile_parts.size(); i++)
                index_grm_pairs.push_back(std::make_pair(tile_parts[i-1] + 1, tile_parts[i]));

            const bool is_first_tile = !grm_skip_global_state;
            geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);

            grm_skip_global_state = true;

            if(is_first_tile && isMtd){
                float weight = 0.0f;
                if(!isDominance)
                    for(uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i];
                else
                    for(uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i] * sd[i];
                if(weight > 0.0f) mtd_weight = static_cast<float>(numValidMarkers) / weight;
            }

            flush_grm_tile(grm_out.get(), N_out.get(), thresh, isSparse, mtd_weight,
                           flush_w_N, flush_grm_block, flush_sparse_buf);

            grm.reset();
            N.reset();
        }

        grm_out.reset();
        N_out.reset();
        LOGGER << "  Used " << numValidMarkers << " valid SNPs." << std::endl;
        if(!isSparse){
            LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.bin]");
            LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + o_name + ".grm.N.bin]");
        } else {
            LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.sp]");
        }
    } else {
        // Non-tiled path (!grm_tiling_enabled): single pass, original behaviour.
        LOGGER << "Computing dense GRM..." << std::endl;
        geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);
        LOGGER << "  Used " << numValidMarkers << " valid SNPs." << std::endl;
        deduce_GRM();
    }

    gbufitems.clear();
    stdGeno.reset();
    stdGeno_i8.reset();
    stdGeno_i8_cols_ = 0;
    stdGeno_i8_kmajor_.reset();
    geno->setGRMMode(false, false);
}

void GRM::processMakeGRMX(){
    processMakeGRM(true);
}

void GRM::runMake(const gcta_cli::GrmMakeOptions& opts, const std::string& out_prefix) {
    if (opts.homogametic) {
        LOGGER.i(0, "Note: this function treats homogametic chromosomes as non-PAR regions.");
    } else {
        LOGGER.i(0, "Note: GRM is computed using the SNPs on the autosomes.");
    }
    const bool has_explicit_chr = opts.marker_filter.autosome || opts.marker_filter.autosome_sex ||
                                  opts.marker_filter.chr_homogametic || !opts.marker_filter.chr.empty();
    Pheno pheno(opts.geno, opts.sample_filter);
    Marker marker(opts.geno, opts.sample_filter, opts.marker_filter,
                  /*default_autosome=*/!opts.homogametic && !has_explicit_chr,
                  /*force_chr_homogametic=*/opts.homogametic && !has_explicit_chr);
    GRM grm(&pheno, &marker, opts, out_prefix);
    grm.processMakeGRM(opts.homogametic);
}

void GRM::runMerge(const gcta_cli::GrmMergeOptions& opts, const std::string& out_prefix) {
    GRM grm("", "", "", out_prefix);
    grm.merge_grm(opts.mgrm, out_prefix, opts.budget_gb);
}

void GRM::runPrune(const gcta_cli::GrmPruneOptions& opts, const std::string& out_prefix) {
    const bool no_grm = opts.singleton.has_value();
    const float cutoff = static_cast<float>(no_grm ? *opts.singleton : *opts.cutoff);
    const bool detail = no_grm || opts.cutoff_detail;
    GRM grm(opts.grm, opts.keep, opts.remove, out_prefix, detail);
    grm.cut_rel(cutoff, no_grm);
}

void GRM::runSparse(const gcta_cli::GrmSparseOptions& opts, const std::string& out_prefix) {
    GRM grm(opts.grm, opts.keep, opts.remove, out_prefix);
    float set_val = 0.0f;
    float* p = nullptr;
    if (opts.set_value.has_value()) {
        set_val = static_cast<float>(*opts.set_value);
        p = &set_val;
    }
    grm.prune_fam(static_cast<float>(opts.cutoff), !opts.dense, p);
}

void GRM::runSubtract(const gcta_cli::GrmCombineOptions& opts, const std::string& out_prefix) {
    GRM grm("", opts.keep, opts.remove, out_prefix);
    grm.subtract_grm(opts.mgrm, out_prefix);
}

void GRM::runUnify(const gcta_cli::GrmCombineOptions& opts, const std::string& out_prefix) {
    GRM grm("", opts.keep, opts.remove, out_prefix);
    grm.unify_grm(opts.mgrm, out_prefix);
}