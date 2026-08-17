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
#include <iterator>
#include <algorithm>
#include <ranges>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <numeric>
#include <unordered_set>
#include "utils.hpp"
#include "utils.hpp"
#include "grm_binary_io.hpp"
#include <omp.h>
#include "OptionIO.h"
#include "third_party/plink-ng/2.0/include/plink2_bits.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <sstream>
#include <csignal>

using std::to_string;

namespace {

// Finds the largest `re` in (rs, full_re] such that a GRM tile spanning rows
// [rs, re) with column width `re` (lower-triangle layout: row i needs columns
// [0, i]) fits within budget_elems combined grm+N elements, i.e.
//   (re - rs) * re <= budget_elems
// Solved as the positive root of re^2 - rs*re - budget_elems = 0 in 
// floating-point arithmetic, then floored and clamped to an integer full_re.
int solve_tile_budget_end(int rs, int full_re, double budget_elems) {
    const double drs = rs;
    const double disc = drs * drs + 4.0 * budget_elems;
    int re = static_cast<int>(std::floor((drs + std::sqrt(disc)) / 2.0));
    return std::min(re, full_re);
}

} // namespace

map<string, string> GRM::options;
map<string, double> GRM::options_d;
map<string, bool> GRM::options_b;
vector<string> GRM::processFunctions;


GRM::GRM(){
    bool has_single_grm = false;
    if(options.find("grm_file") != options.end()){
        has_single_grm = true;
    }
    if(has_single_grm){
        this->grm_file = options["grm_file"];
        grm_ids = Pheno::read_sublist(grm_file + ".grm.id");
        num_subjects = grm_ids.size();

        uint64_t num_grm = num_subjects * (num_subjects + 1) / 2;
        FILE *file = fopen((grm_file + ".grm.bin").c_str(), "rb");
        if(!file){
            LOGGER.e(0, "can't open " + grm_file + ".grm.bin");
        }
        const auto grm_file_size = getFileByteSize(file);
        if(!grm_file_size){
            LOGGER.e(0, "Failed to read GRM binary size for [" + grm_file + "]: " + std::string(getUtilsErrorMessage(grm_file_size.error())));
        }
        if(num_grm * 4 != grm_file_size.value()){
            LOGGER.e(0, "The IDs in GRM and the IDs in the GRM binary file do not match [" + grm_file + "]");
        }
        fclose(file);

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

        if(options.find("keep_file") != options.end()){
            vector<string> keep_lists = Pheno::read_sublist(options["keep_file"]);
            LOGGER << "Get " << keep_lists.size() << " samples from list [" << options["keep_file"] << "]." << std::endl;
            Pheno::set_keep(keep_lists, grm_ids, index_keep, true);
        }

        if(options.find("remove_file") != options.end()){
            vector<string> rm_lists = Pheno::read_sublist(options["remove_file"]);
            LOGGER << "Get " << rm_lists.size() << " samples from list [" << options["remove_file"] << "]." << std::endl;
            Pheno::set_keep(rm_lists, grm_ids, index_keep, false);
        }
    }
    if(options.find("mgrm") != options.end()){
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
    if(!o_id) LOGGER.e(0, "can't write to [" + options["out"] + ".grm.id]");
    LOGGER.i(2, "Saving " + to_string(common_id.size()) + " individual IDs");
    std::ranges::copy(common_id, std::ostream_iterator<string>(o_id, "\n"));
    o_id.close();

    uint64_t grm_size = (1 + common_id.size()) * common_id.size() / 2;
    uint64_t grm_file_size = grm_size * 4; 

    FILE * h_grm1 = fopen((files[0] + ".grm.bin").c_str(), "rb");
    FILE * h_grmN1 = fopen((files[0] + ".grm.N.bin").c_str(), "rb");
    FILE * h_grm2 = fopen((files[1] + ".grm.bin").c_str(), "rb");
    FILE * h_grmN2 = fopen((files[1] + ".grm.N.bin").c_str(), "rb");
    if(grm_file_size != getFileSize(h_grm1)){
        LOGGER.e(0, "The size of [" + files[0] + ".grm.bin] is not correct.");
    }
    if(grm_file_size != getFileSize(h_grmN1)){
        LOGGER.e(0, "The size of [" + files[0] + ".grm.N.bin] is not correct.");
    }
    if(grm_file_size != getFileSize(h_grm2)){
        LOGGER.e(0, "The size of [" + files[1] + ".grm.bin] is not correct.");
    }
    if(grm_file_size != getFileSize(h_grmN2)){
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
    FILE *ho_grm = fopen((out_file + ".grm.bin").c_str(), "wb");
    FILE *ho_grmN = fopen((out_file + ".grm.N.bin").c_str(), "wb");

    for(int i = 0; i < loop_num; i++){
        if(i == loop_num - 1 && remain_item != 0) itemRead = remain_item;
        readBytes(h_grm1, itemRead, buf1.data());
        readBytes(h_grm2, itemRead, buf2.data());
        readBytes(h_grmN1, itemRead, bufN1.data());
        readBytes(h_grmN2, itemRead, bufN2.data());
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
        if(fwrite(buf.data(), sizeof(float), itemRead, ho_grm) != itemRead){
            LOGGER.e(0, "can't write to [" + out_file + ".grm.bin].");
        }
        if(fwrite(bufN.data(), sizeof(float), itemRead, ho_grmN) != itemRead){
            LOGGER.e(0, "can't write to [" + out_file + ".grm.N.bin].");
        }
    }
    LOGGER.i(0, "The subtracted GRM has been written to [" + out_file + ".grm.bin, .grm.N.bin].");
}

void GRM::merge_grm_streaming(string mgrm_file, string out_file, int row_block_rows){
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

    gcta_grm_io::merge_grms_streaming(files, out_file, row_block_rows);
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
    if(options.find("keep_file") != options.end()){
        LOGGER.i(0, "Keeping individuals listed in [" + options["keep_file"] + "]...");
        vector<string> keep_id = Pheno::read_sublist(options["keep_file"]);
        LOGGER << keep_id.size() << " samples have been read." << std::endl;
        vector<uint32_t> index1, index2;
        vector_commonIndex_sorted1(common_id, keep_id, index1, index2);
        vector<string> temp_common_id(index1.size());
        std::ranges::transform(index1, temp_common_id.begin(), [&common_id](uint32_t v){
                return common_id[v];});
        common_id = temp_common_id;
        LOGGER << common_id.size() << " common samples after merging." << std::endl;
    }
    if(options.find("remove_file") != options.end()){
        LOGGER.i(0, "Excluding individuals listed in [" + options["remove_file"] + "]...");
        vector<string> remove_id = Pheno::read_sublist(options["remove_file"]);
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
    string out_string = options["out"];
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
        FILE *h_grm = fopen(file_name.c_str(), "rb");
        if(!h_grm){
            LOGGER.e(0, "can't read " + file_name + ".");
        }

        FILE *h_wgrm = fopen(wfile_name.c_str(), "wb");
        if(!h_wgrm){
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
            fseek(h_grm, byte_start_grm, SEEK_SET);
            if(fread(buf.data(), sizeof(float), num_grm, h_grm) != num_grm){
                LOGGER.e(0, "error in reading [" + file_name + "], in position " + to_string(ftell(h_grm)), ".");
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
                    fseek(h_grm, bytes_offset, SEEK_SET); 
                    if(fread(&grm_value, sizeof(float), 1, h_grm) != 1){
                        LOGGER.e(0, "error in reading [" + file_name + "], in position " + to_string(bytes_offset), ".");
                    }
                }
                wbuf[k] = grm_value;
            }
            //last element
            //*(wbuf + j) = *(buf + grm_index);

            if(size_write != fwrite(wbuf.data(), sizeof(float), size_write, h_wgrm)){
                LOGGER.e(0, "error in writing to [" + wfile_name + "], pos: " + std::to_string(ftell(h_wgrm)) + "."); 
            }
        }
        fclose(h_grm);
        fclose(h_wgrm);
        LOGGER.i(0, "GRM has been written to [" + wfile_name + "].");
    }

}

void GRM::prune_fam(float thresh, bool isSparse, float *value){
    LOGGER.i(0, "Pruning the GRM to a sparse matrix with a cutoff of " + to_string(thresh) + "...");
    LOGGER.i(0, "Total number of parts to be processed: " + to_string(index_grm_pairs.size()));
    FILE *grmFile = fopen((grm_file + ".grm.bin").c_str(), "rb");

    std::ofstream o_id((options["out"] + ".grm.id").c_str());
    if(!o_id) LOGGER.e(0, "can't write to [" + options["out"] + ".grm.id]");
    std::ofstream o_fam;
    FILE* o_bk;
    if(isSparse){
        o_fam.open((options["out"] + ".grm.sp").c_str());
        if(!o_fam) LOGGER.e(0, "can't write to [" + options["out"] + ".grm.sp]");
    }else{
        o_bk = fopen((options["out"] + ".grm.bin").c_str(), "wb");
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
        if(fread(grm_buf.data(), sizeof(float), byte_part_grms[part_index], grmFile) != byte_part_grms[part_index]){
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
            if(fwrite(out_grm_buf.data(), sizeof(float), write_items, o_bk) != write_items){
                LOGGER.e(0, "Failed to write the output"); 
            }
        }

    }
    fclose(grmFile);
    if(!isSparse){
        fclose(o_bk);
    }

    if(isSparse){
        LOGGER.i(0, "Saving the sparse GRM (" + to_string(rm_grm.size()) + " pairs) to [" + options["out"] + ".grm.sp]");
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
        LOGGER.i(0, "GRM has been saved to [" + options["out"] + ".grm.bin]");
    }

    FILE *NFile = fopen((grm_file + ".grm.N.bin").c_str(), "rb");
    FILE *ONFile = fopen((options["out"] + ".grm.N.bin").c_str(), "wb");
    std::vector<float> N_buf(num_byte_buffer);
    float *cur_N_pos0;
    std::vector<float> out_N_buf(num_byte_buffer);
    float *cur_N_buf;
    for(int part_index = 0; part_index != index_grm_pairs.size(); part_index++){
        if(fread(N_buf.data(), sizeof(float), byte_part_grms[part_index], NFile) != byte_part_grms[part_index]){
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
        if(fwrite(out_N_buf.data(), sizeof(float), write_items, ONFile) != write_items){
            LOGGER.e(0, "Failed to write the GRM N"); 
        }

    }
    fclose(ONFile);
    fclose(NFile);
    LOGGER.i(0, "GRM N has been saved to [" + options["out"] + ".grm.N.bin]");
}



void GRM::cut_rel(float thresh, bool no_grm){
    LOGGER.i(0, "Pruning the GRM with a cutoff of " + to_string(thresh) + "...");
    LOGGER.i(0, "Total number of parts to be processed: " + to_string(index_grm_pairs.size()));
    FILE *grmFile = fopen((grm_file + ".grm.bin").c_str(), "rb");
    // put this first to avoid unwritable disk
    std::ofstream o_keep;
    if(!no_grm){
        o_keep.open((options["out"] + ".grm.id").c_str());
        if((!o_keep)){
            LOGGER.e(0, "can't write to [" + options["out"] + ".grm.id]");
        }
    }

    bool detail_flag = false;
    std::ofstream out_fam, o_single;
    if(options.find("cutoff_detail") != options.end()){
        detail_flag = true;
        out_fam.open((options["out"] + ".family.txt").c_str());
        o_single.open((options["out"] + ".singleton.txt").c_str());
        if((!out_fam) || (!o_single)){
            LOGGER.e(0, "can't write to [" + options["out"] + ".family.txt, .singleton.txt]");
        }
    }
 

    std::unordered_set<int> keeps_ori(index_keep.begin(), index_keep.end());

    std::vector<float> grm_buf(num_byte_buffer);
    float cur_grm, *cur_grm_pos0;
    vector<float> rm_grm;
    vector<int> rm_grm_ID1, rm_grm_ID2;
    for(int part_index = 0; part_index != index_grm_pairs.size(); part_index++){
        if(fread(grm_buf.data(), sizeof(float), byte_part_grms[part_index], grmFile) != byte_part_grms[part_index]){
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
        LOGGER.i(0, "Related family pairs have been saved to " + options["out"] + ".family.txt");
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
        LOGGER.i(2, "Pruned unrelated IDs have been saved to " + options["out"] + ".grm.id");
    }

    if(detail_flag){
        std::ranges::copy(keep_ID, std::ostream_iterator<string>(o_single, "\n"));
        o_single.close();
        LOGGER.i(2, "Pruned singleton IDs have been saved to " + options["out"] + ".singleton.txt");
    }

    if(no_grm) {
        fclose(grmFile);
        return;
    }

    LOGGER.i(0, "Pruning GRM values, total parts " + std::to_string(index_grm_pairs.size()));
    FILE *grm_out_file = fopen((options["out"] + ".grm.bin").c_str(), "wb");
    if(!grm_out_file){
        LOGGER.e(0, "can't open [" + options["out"] + ".grm.bin] to write");
    }
    fseek(grmFile, 0, SEEK_SET);
    outBinFile(grmFile, grm_out_file);
    fclose(grmFile);
    fclose(grm_out_file);
    LOGGER.i(2, "GRM values have been saved to [" + options["out"] + ".grm.bin]");

    LOGGER.i(0, "Pruning number of SNPs to calculate GRM, total parts " + std::to_string(index_grm_pairs.size()));
    FILE *NFile = fopen((grm_file + ".grm.N.bin").c_str(), "rb");
    if(!NFile){
        LOGGER.w(2, "There is no [" + grm_file + ".grm.N.bin]");
        return;
    }
    FILE *N_out_file = fopen((options["out"] + ".grm.N.bin").c_str(), "wb");
    if(!N_out_file){
        LOGGER.w(2, "can't open [" + options["out"] + ".grm.N.bin] to write. Ignore this step");
        fclose(NFile);
        return;
    }
    outBinFile(NFile, N_out_file);
    fclose(NFile);
    fclose(N_out_file);
    LOGGER.i(2, "Number of SNPs has been saved to [" + options["out"] + ".grm.N.bin]");
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


GRM::GRM(Pheno* pheno, Marker* marker) {
    //clock_t begin = t_begin();
    this->pheno = pheno;
    this->marker = marker;
    this->geno = new Geno(pheno, marker);
    // Pay attention to not reflect the newest changes of keep;
    this->index_keep = pheno->get_index_keep();
    this->part = std::stoi(options["cur_part"]);
    this->num_parts = std::stoi(options["num_parts"]);

    // divide the genotype into equal length
    vector<uint32_t> parts;
    if(options.find("use_blas") != options.end()){
        bBLAS = true;
        parts = divide_parts_mem(pheno->count_keep(), num_parts);
    }else{
        bBLAS = false;
        parts = divide_parts(0, pheno->count_keep() - 1, num_parts);
    }

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

    // init the geno buffers
    /*
    if(!bBLAS){
        // use byte style coding
        this->num_byte_geno = sizeof(uint16_t) * Constants::NUM_MARKER_READ / num_marker_block * index_keep.size();
        lookup_GRM_table = new double[Constants::NUM_MARKER_READ / num_marker_block][num_lookup_table];

        int ret1 = posix_memalign((void **) &geno_buf, 32, num_byte_geno);
        int ret2 = posix_memalign((void **) &mask_buf, 32, num_byte_geno);
        if(ret1 != 0 | ret2 != 0){
            LOGGER.e(0, "can't allocate enough memory for genotype buffer.");
        }

   }else{
   }
   */

    reg_bit_width = 64;
    this->num_grm_handle = reg_bit_width / 64;
    this->num_count_handle = reg_bit_width / 32;
    this->num_block_handle = reg_bit_width / 16;
    this->num_marker_process_block = this->num_marker_block * this->num_block_handle;

    // init the count mask buffer;
    /*
    this->num_byte_cmask = sizeof(uint64_t) * Constants::NUM_MARKER_READ * index_keep.size() / num_cmask_block;
    int ret3 = posix_memalign((void **) &cmask_buf, 32, num_byte_cmask); 
    if(ret3 != 0){
        LOGGER.e(0, "can't allocate enough memory to store the mask.");
    }
    */

    // init the grm and N buffer
    num_individual = part_keep_indices.second - part_keep_indices.first + 1;
    num_grm = ((uint64_t) part_keep_indices.first + part_keep_indices.second + 2) * num_individual / 2;

    uint64_t fill_grm = (num_grm + num_count_handle - 1) / num_count_handle * num_count_handle;
    uint64_t fill_N = fill_grm;
    if(bBLAS){
        fill_grm = (uint64_t)num_individual * (part_keep_indices.second + 1);
    }

    // In bBLAS mode grm and N are managed per-tile in processMakeGRM; allocate
    // them here only when not using tiling (no --GRM-tile-budget active).
    const bool ctor_tiling_enabled = options_d.count("grm_tile_budget_bytes") > 0
                                    && options_d.at("grm_tile_budget_bytes") > 0;
    if(!bBLAS || !ctor_tiling_enabled){
        int ret_grm = posix_memalign((void **)&grm, 32, fill_grm * sizeof(double));
        if(ret_grm){
            LOGGER.e(0, "can't allocate enough memory to store the (parted) GRM: " + to_string(fill_grm*sizeof(double) / 1024.0/1024/1024) + "GB required.");
        }
        memset(grm, 0, fill_grm * sizeof(double));

        int ret_N = posix_memalign((void **)&N, 64, fill_N * sizeof(uint32_t));
        if(ret_N){
            LOGGER.e(0, "can't allocate enough memory to store (parted) N: " + to_string(fill_grm*sizeof(uint32_t) / 1024.0/1024/1024) + "GB required.");
        }
        memset(N, 0, fill_N * sizeof(uint32_t));
    }
    // grm/N == nullptr for bBLAS; posix_mem_free(nullptr) is a safe no-op.

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

    if(options_b.find("isDominance") != options_b.end()){
        isDominance = options_b["isDominance"];
    }

    if(options_b.find("isMtd") != options_b.end()){
        isMtd = options_b["isMtd"];
    }

    // Cache constants derived from part_keep_indices
    grm_n = static_cast<int>(part_keep_indices.second) + 1;
    grm_m = static_cast<int>(part_keep_indices.second) - static_cast<int>(part_keep_indices.first) + 1;
    grm_s_n = grm_n - grm_m;
    grm_bytes_std_geno = static_cast<int>(sizeof(double)) * grm_n;
    // Round grm_n up to next multiple of 8 (64 bytes) so every column of stdGeno
    // is 64-byte aligned. Required by single-threaded AVX/AVX-512 BLAS kernels.
    stdGenoLD = (grm_n + 7) / 8 * 8;
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

    o_name = options["out"];

    if(isDominance){
        o_name += ".d";
    }

    output_id();

#ifndef NDEBUG
    o_geno0 = fopen("./test.bin", "wb");
    o_mask0 = fopen("./test_mask.bin", "wb");
#endif
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

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_marker; i++){
        GenoBufItem &item = gbufitems[i];
        item.extractedMarkerIndex = markerIndex[i];
        geno->getGenoDouble(buf, i, &item);
    }

    validIndexBuf.clear();
    for(int i = 0; i < num_marker; i++){
        if(gbufitems[i].valid){
            validIndexBuf.push_back(i);
        }
    }

    int curNumValidMarkers = static_cast<int>(validIndexBuf.size());

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < curNumValidMarkers; i++){
        const double *src = gbufitems[validIndexBuf[i]].geno.data();
        double *dst = stdGeno + i * stdGenoLD;
        #pragma omp simd
        for (int j = 0; j < grm_n; ++j) {
            dst[j] = src[j];
        }
    }
    if(!grm_skip_global_state){
        for(int i = 0; i < curNumValidMarkers; i++){
            sd.push_back(gbufitems[validIndexBuf[i]].sd);
        }
    }

    static constexpr double alpha = 1.0;
    if(grm_tiling_enabled){
        // Block-tiled path: split into off-diagonal dgemm + diagonal dsyrk.
        //
        // grm buffer is column-major (grm_tile_rows × grm_tile_cols):
        //   cols [0, tile_rs)    — off-diagonal, filled by dgemm
        //   cols [tile_rs, tile_re) — diagonal block (lower triangle), filled by dsyrk
        //
        // A_tile = stdGeno[tile_rs : tile_re, :] — (tile_rows × k)
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> A_tile(
            stdGeno + grm_tile_rs, grm_tile_rows, curNumValidMarkers,
            Eigen::OuterStride<>(stdGenoLD));

        // Off-diagonal block: only present when tile starts after column 0.
        if(grm_tile_rs > 0){
            Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> A_top(
                stdGeno, grm_tile_rs, curNumValidMarkers,
                Eigen::OuterStride<>(stdGenoLD));
            Eigen::Map<Eigen::MatrixXd>(grm, grm_tile_rows, grm_tile_rs).noalias() +=
                alpha * (A_tile * A_top.transpose());
        }

        // Diagonal block (dsyrk — ~2× fewer flops than equivalent dgemm).
        // Only the lower triangle is written; flush_grm_tile only reads pair2 <= pair1.
        double *grm_diag = grm + static_cast<size_t>(grm_tile_rs) * grm_tile_rows;
        Eigen::Map<Eigen::MatrixXd>(grm_diag, grm_tile_rows, grm_tile_rows)
            .selfadjointView<Eigen::Lower>().rankUpdate(A_tile, alpha);
    } else if(part_keep_indices.first == 0){
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> A(stdGeno, grm_n, curNumValidMarkers, Eigen::OuterStride<>(stdGenoLD));
        Eigen::Map<Eigen::MatrixXd>(grm, grm_n, grm_n).selfadjointView<Eigen::Lower>().rankUpdate(A, alpha);
    }else{
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> A_top(stdGeno, grm_s_n, curNumValidMarkers, Eigen::OuterStride<>(stdGenoLD));
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> A_bot(stdGeno + part_keep_indices.first, grm_m, curNumValidMarkers, Eigen::OuterStride<>(stdGenoLD));
        Eigen::Map<Eigen::MatrixXd>(grm, grm_m, grm_s_n).noalias() += alpha * (A_bot * A_top.transpose());
        double *grm_start = grm + (uint64_t)grm_s_n * grm_m;
        Eigen::Map<Eigen::MatrixXd>(grm_start, grm_m, grm_m).selfadjointView<Eigen::Lower>().rankUpdate(A_bot, alpha);
    }

    const int markerPerN = sizeof(uintptr_t) * CHAR_BIT;
    const int numNblock = (curNumValidMarkers + markerPerN - 1) / markerPerN;
    const int numNSampleBlock = (grm_n + markerPerN - 1) / markerPerN;
    const int blockStride = numNSampleBlock * markerPerN;
    std::array<uintptr_t, 64> transposeInput{};
    std::vector<plink2::VecW> transposeScratch(plink2::kPglBitTransposeBufvecs);
    for(int i = 0; i < numNblock; i++){
        const int baseMarkerIndex = markerPerN * i;
        const int lastValidIndex  = std::min(markerPerN * (i + 1), curNumValidMarkers);
        uintptr_t *block_buf = sampleMissBuf.data() + i * blockStride;
        for(int j = 0; j < numNSampleBlock; j++){
            const int baseMissIndex = j * markerPerN;
            if (markerPerN == 64) {
                for(int k = baseMarkerIndex; k < lastValidIndex; k++){
                    transposeInput[static_cast<std::size_t>(k - baseMarkerIndex)] =
                        gbufitems[validIndexBuf[k]].missing[j];
                }
                for(int k = lastValidIndex; k < markerPerN * (i + 1); k++){
                    transposeInput[static_cast<std::size_t>(k - baseMarkerIndex)] = 0UL;
                }
                // Transpose [marker x sample] missing bits into [sample x marker].
                // Bit order inside each word is irrelevant for downstream AND+popcount.
                plink2::TransposeBitblock(
                    transposeInput.data(), 1, 1,
                    static_cast<uint32_t>(markerPerN), static_cast<uint32_t>(markerPerN),
                    &block_buf[baseMissIndex], transposeScratch.data());
            } else {
                for(int k = baseMarkerIndex; k < lastValidIndex; k++){
                    block_buf[baseMissIndex + k - baseMarkerIndex] = gbufitems[validIndexBuf[k]].missing[j];
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


    // Single parallel region: each thread processes its pair range across all blocks.
    // This reduces fork/join barriers from numNblock to 1, and keeps each thread's
    // N[] output range hot in cache across the inner block loop.
    #pragma omp parallel for schedule(static)
    for(int index = 0; index < (int)index_grm_pairs.size(); index++){
        const auto index_pair = index_grm_pairs[index];
        for(int i = 0; i < numNblock; i++){
            N_thread(index_pair.first, index_pair.second, sampleMissBuf.data() + i * blockStride);
        }
    }

    if(!grm_skip_global_state){
        finished_marker += num_marker;
        numValidMarkers += curNumValidMarkers;
    }
}

    /*
    int num_process_block = (num_marker + num_marker_block - 1) / num_marker_block;
    this->cur_num_block = (num_marker + num_marker_process_block - 1) / num_marker_process_block;
    num_process_block = cur_num_block;
    static int keep_sample_size = index_keep.size();
    static int geno_num64 = geno->num_item_1geno;
    static int geno_num8 = geno_num64 * 8;

    #pragma omp parallel for schedule(dynamic)
    for(int index_process_block = 0; index_process_block < num_process_block; index_process_block++){
        int base_marker_index = index_process_block * num_marker_process_block;
        uint8_t *cur_buf = (uint8_t*) (buf + base_marker_index * geno_num64);
        int num_marker_in_process_block = index_process_block == num_process_block - 1 ?
                                      (num_marker - base_marker_index): num_marker_process_block;
        // recode genotype
        for(int index_indi = 0; index_indi < part_keep_indices.second + 1; index_indi += 1) {
            uint32_t raw_index = index_indi;

            for (int marker_index = 0; marker_index != num_marker_in_process_block; marker_index++) {
                uint16_t temp_geno = (*(cur_buf  + marker_index * geno_num8 + raw_index / 4)
                        >> ((raw_index % 4) * 2)) & 3LU;
                if (temp_geno == 1) {
                    #pragma omp atomic
                    sub_miss[index_indi] += 1;
                    //cmask
                    #pragma omp atomic
                    cmask_buf[index_indi + ((base_marker_index + marker_index) / num_cmask_block)*keep_sample_size] |=
                            (1LLU << ((base_marker_index + marker_index) % num_cmask_block));
                }
            }
       }
    }
    */


/*
void GRM::calculate_GRM(uint64_t *buf, int num_marker) {
    //prepare the 7 freq arrays in current range;
    double * devs = new double[num_marker];
    if(!isDominance){
        #pragma omp parallel for schedule(dynamic)
        for (int index_marker = 0; index_marker < num_marker; index_marker++) {
            double af = geno->AFA1[finished_marker + index_marker];
            double mu = af * 2.0;
            double dev = mu * (1.0 - af);
            double rdev = (dev < 1.0e-50) ? 0 : (1.0 / dev);
            rdev = isMtd? 1.0 : rdev;
            GRM_table[index_marker][0] = (2.0 - mu) * (2.0 - mu) * rdev;  // 00 00
            GRM_table[index_marker][2] = (2.0 - mu) * (1.0 - mu) * rdev;  // 00 10
            GRM_table[index_marker][3] = (2.0 - mu) * (0.0 - mu) * rdev;  // 00 11
            GRM_table[index_marker][4] = (1.0 - mu) * (1.0 - mu) * rdev;  // 10 10
            GRM_table[index_marker][5] = (0.0 - mu) * (1.0 - mu) * rdev;  // 11 10
            GRM_table[index_marker][6] = (0.0 - mu) * (0.0 - mu) * rdev;  // 11 11
            GRM_table[index_marker][7] = 0.0;                             // 01 01
        }
    }else{
        #pragma omp parallel for schedule(dynamic)
        for (int index_marker = 0; index_marker < num_marker; index_marker++) {
            double af = geno->AFA1[finished_marker + index_marker];
            double mu = af * 2.0;
            double dev = mu * (1.0 - af);
            double rdev = (dev < 1.0e-50) ? 0 : (1.0 / dev);
            double rdev2 = rdev * rdev; 
            rdev2 = isMtd ? 1.0 : rdev2;
            double psq = 0.5 * mu * mu;
            double A00 = 2.0 * mu - 2 - psq;
            double A10 = mu - psq;
            double A11 = 0.0 - psq;

            GRM_table[index_marker][0] = A00 * A00 * rdev2;  // 00 00
            GRM_table[index_marker][2] = A00 * A10 * rdev2;  // 00 10
            GRM_table[index_marker][3] = A00 * A11 * rdev2;  // 00 11
            GRM_table[index_marker][4] = A10 * A10 * rdev2;  // 10 10
            GRM_table[index_marker][5] = A11 * A10 * rdev2;  // 11 10
            GRM_table[index_marker][6] = A11 * A11 * rdev2;  // 11 11
            GRM_table[index_marker][7] = 0.0;                // 01 01
        }
    }
 

    //init the remained arrays to 0;
    #pragma omp parallel for schedule(dynamic)
    for(int index_marker = num_marker; index_marker < Constants::NUM_MARKER_READ; index_marker++){
        for(int i = 0; i != 8; i++){
            GRM_table[index_marker][i] = 0.0;
        }
    }

    int num_block = (num_marker + num_marker_block - 1) / num_marker_block;

    int length = elements.size();

    //init GRM lookup table
    #pragma omp parallel for schedule(dynamic)
    for(int cur_block = 0; cur_block < num_block; cur_block++) {
        int cur_block_base = cur_block * num_marker_block;
        for (int e1 = 0; e1 != length; e1++) {
            for (int e2 = 0; e2 != length; e2++) {
                for (int e3 = 0; e3 != length; e3++) {
                    for (int e4 = 0; e4 != length; e4++) {
                        for (int e5 = 0; e5 != length; e5++) {
                            lookup_GRM_table[cur_block][elements[e5]
                                             | (elements[e4] << 3)
                                             | (elements[e3] << 6)
                                             | (elements[e2] << 9)
                                             | (elements[e1] << 12)] = GRM_table[cur_block_base][elements[e5]] +
                                                                       GRM_table[cur_block_base + 1][elements[e4]] +
                                                                       GRM_table[cur_block_base + 2][elements[e3]] +
                                                                       GRM_table[cur_block_base + 3][elements[e2]] +
                                                                       GRM_table[cur_block_base + 4][elements[e1]];
                        }
                    }
                }
            }
        }

    }


    memset(this->geno_buf, 0, num_byte_geno);
    memset(this->mask_buf, 0, num_byte_geno);
    memset(this->cmask_buf, 0, num_byte_cmask);

    int num_process_block = (num_marker + num_marker_block - 1) / num_marker_block;
    this->cur_num_block = (num_marker + num_marker_process_block - 1) / num_marker_process_block;
    num_process_block = cur_num_block;
    static int keep_sample_size = index_keep.size();

    static int geno_num64 = geno->num_item_1geno;
    static int needs_block_geno64 = (part_keep_indices.second + 31) / 32;
    static int super_block_size = num_block_handle * keep_sample_size;
    static int super_N_size = num_cmask_block / num_marker_block;
*/
/* Don't adapt this code, worse speed
    //#pragma omp parallel for schedule(dynamic) 
    for(int index_process_block = 0; index_process_block < num_process_block; index_process_block++){
        int super_block_num = index_process_block / num_block_handle * super_block_size;
        int super_block_item = index_process_block % num_block_handle;
        int super_block_start = super_block_num + super_block_item;
        int super_N_start = index_process_block / super_N_size * keep_sample_size;
        int super_N_move = index_process_block * num_marker_block % num_cmask_block;

        int base_marker_index = index_process_block * num_marker_block;
        int num_marker_in_process_block = index_process_block == num_process_block - 1 ?
                                      (num_marker - base_marker_index): num_marker_block;
        uint64_t * gbuf = buf + base_marker_index * geno_num64; 
        uint32_t base_sample = 0;
        for(int geno64_block = 0; geno64_block < needs_block_geno64; geno64_block++){
            int num_sample_geno = geno64_block == geno_num64 - 1 ?
                                  (keep_sample_size - 32 * geno64_block) : 32;

            uint64_t genot[num_marker_in_process_block];
            for(int index = 0; index < num_marker_in_process_block; index++){
                genot[index] = gbuf[geno64_block + index * geno_num64];
            }

            for(int index_sample = 0; index_sample < num_sample_geno; index_sample++){
                uint32_t real_index_sample = base_sample + index_sample;
                //if(real_index_sample > part_keep_indices.second)break;
               // uint32_t base_geno_buf = real_index_sample * num_process_block + index_process_block;
                uint32_t base_geno_buf = super_block_start + real_index_sample * num_block_handle;
                for(int index = 0; index < num_marker_in_process_block; index++){
                    uint16_t temp_geno = (uint16_t) (genot[index] >> (index_sample * 2)) & 3LU;
                    int move_bit = index * 3;
                    uint16_t t2_geno = temp_geno << move_bit;
                    //#pragma omp atomic
                    geno_buf[base_geno_buf] |= t2_geno;
                    if(temp_geno == 1){
                     //   #pragma omp critical
                        {
                            mask_buf[base_geno_buf] |= (7LU << move_bit);
                            sub_miss[real_index_sample]++;
                            cmask_buf[super_N_start + real_index_sample] |= (1LLU << (super_N_move + index));
                        }
                    }
                }
                for(int index = num_marker_in_process_block; index < num_marker_block; index++){
                    int move_bit = index * 3;
                    uint16_t m2_mask = (7LU << move_bit);
                    //#pragma omp atomic
                    mask_buf[base_geno_buf] |= m2_mask;
                }
            }

            base_sample += 32;

        }

    }
    */

/*
#ifndef NDEBUG
    if(finished_marker == 0 ){
        FILE *test = fopen("test_b0.bin", "wb");
        fwrite(geno_buf, 1, num_byte_geno, test);
        FILE *test_cmask = fopen("cmask_b0.bin", "wb");
        fwrite(cmask_buf, 1, num_byte_cmask, test_cmask);
        fclose(test);
        fclose(test_cmask);
    }
#endif
        
    static int geno_num8 = geno_num64 * 8;


    #pragma omp parallel for schedule(dynamic)
    for(int index_process_block = 0; index_process_block < num_process_block; index_process_block++){
        int base_marker_index = index_process_block * num_marker_process_block;
        uint8_t *cur_buf = (uint8_t*) (buf + base_marker_index * geno_num64);
        int base_buf_pos = num_block_handle * index_process_block * keep_sample_size;
        uint16_t *cur_geno = this->geno_buf + base_buf_pos;
        uint16_t *cur_mask = this->mask_buf + base_buf_pos;
        int num_marker_in_process_block = index_process_block == num_process_block - 1 ?
                                      (num_marker - base_marker_index): num_marker_process_block;
        // recode genotype
        for(int index_indi = 0; index_indi < part_keep_indices.second + 1; index_indi += 1) {
            uint32_t raw_index = index_indi;
            int indi_block = index_indi * num_block_handle;

            uint16_t *p_geno = cur_geno + indi_block;
            uint16_t *p_mask = cur_mask + indi_block;

            for (int marker_index = 0; marker_index != num_marker_in_process_block; marker_index++) {
                int cur_block = marker_index / num_marker_block;
                int cur_block_pos = (marker_index % num_marker_block) * 3;
                uint16_t temp_geno = (*(cur_buf  + marker_index * geno_num8 + raw_index / 4)
                        >> ((raw_index % 4) * 2)) & 3LU;
                uint16_t t2_geno = (temp_geno << cur_block_pos);
                #pragma omp atomic
                *(p_geno + cur_block) |= t2_geno;
                if (temp_geno == 1) {
                    #pragma omp atomic
                    *(p_mask + cur_block) |= (7LU << cur_block_pos);
                    #pragma omp atomic
                    sub_miss[index_indi] += 1;
                    //cmask
                    #pragma omp atomic
                    cmask_buf[index_indi + ((base_marker_index + marker_index) / num_cmask_block)*keep_sample_size] |=
                            (1LLU << ((base_marker_index + marker_index) % num_cmask_block));
                }
            }

            for (int marker_index = num_marker_in_process_block;
                 marker_index != num_marker_process_block; marker_index++) {
                int cur_block = marker_index / num_marker_block;
                int cur_block_pos = (marker_index % num_marker_block) * 3;
                #pragma omp atomic
                *(p_mask + cur_block) |= (7LU << cur_block_pos);

            }
        }
    }

    #pragma omp parallel for
    for(int index = 0; index < index_grm_pairs.size(); index++){
        auto index_pair = index_grm_pairs[index];
        grm_thread(index_pair.first, index_pair.second);
        //N_thread(index_pair.first, index_pair.second);
    }

    #pragma omp parallel for
    for(int index = 0; index < index_grm_pairs.size(); index++){
        auto index_pair = index_grm_pairs[index];
        //grm_thread(index_pair.first, index_pair.second);
        N_thread(index_pair.first, index_pair.second);
    }

    //finished_marker += num_marker;
//}
*/
/* don't use 
    //submit thread
    int pair_size = index_grm_pairs.size();
    for(int index = 0; index != pair_size - 1; index++){
        THREADS.AddJob(std::bind(&GRM::grm_thread, this, index_grm_pairs[index].first, index_grm_pairs[index].second));
    }

    for(int index = 0; index != pair_size - 1; index++){
        THREADS.AddJob(std::bind(&GRM::N_thread, this, index_grm_pairs[index].first, index_grm_pairs[index].second));
    }

    auto last_pair = index_grm_pairs[pair_size - 1];
    grm_thread(last_pair.first, last_pair.second);
    N_thread(last_pair.first, last_pair.second);

    THREADS.WaitAll();
    */

    //std::stringstream out_message;
    //out_message << std::fixed << std::setprecision(2) << finished_marker * 100.0 / geno->marker->count_extract();
    //LOGGER.i(0, out_message.str() + "% has been finished");



void GRM::deduce_GRM(){
    LOGGER.i(0, "The GRM computation is completed.");
    float thresh = -99;
    bool isSparse = false;
    if(options_d.find("sparse_cutoff") != options_d.end()){
        thresh = options_d["sparse_cutoff"];
        isSparse = true;
        LOGGER.i(0, "Saving sparse GRM with a cutoff " + to_string(thresh) + "...");
    }else{
        LOGGER.i(0, "Saving GRM...");
    }
    //Just for test
#ifndef NDEBUG
    fclose(o_geno0);
    fclose(o_mask0);
#endif

    //clock_t begin = t_begin();
    FILE *grm_out, *N_out;
    if(isSparse){
        string grm_name = o_name + ".grm.sp";
        grm_out = fopen(grm_name.c_str(), "wb");
        N_out = NULL;
        if(!grm_out){
            LOGGER.e(0, "can't open " + o_name + ".grm.sp to write");
        }
    }else{
        string grm_name = o_name + ".grm.bin";
        string N_name = o_name + ".grm.N.bin";
        grm_out = fopen(grm_name.c_str(), "wb");
        N_out = fopen(N_name.c_str(), "wb");
        if((!grm_out) || (!N_out)){
            LOGGER.e(0, "can't open " + o_name + ".grm.bin or .grm.N.bin to write");
        }
        // 4 MiB matches a typical Lustre stripe size and avoids sub-stripe metadata ops.
        setvbuf(grm_out, nullptr, _IOFBF, 4 << 20);
        setvbuf(N_out,   nullptr, _IOFBF, 4 << 20);
    }

    float mtd_weight = 1.0;
    if(options_b["isMtd"]){
        float weight = 0;
        if(!isDominance){
            for(int i = 0; i < numValidMarkers; i++){
                //float af = geno->AFA1[i];
                //float sd = 2.0 * af * (1.0 - af); 
                weight += sd[i];
            }
        }else{
            for(int i = 0; i < numValidMarkers; i++){
                //float af = geno->AFA1[i];
                //float sd = 2.0 * af * (1.0 - af); 
                weight += sd[i] * sd[i];
            }
        }
        mtd_weight = 1.0 / (weight / numValidMarkers);
    }

 
    /* X chr adjustment
    float weights[5];
    weights[2] = 0.5 * mtd_weight; // heterogametic-heterogametic
    weights[3] = sqrt(0.5) * mtd_weight; // heterogametic-homogametic;
    weights[4] = 1.0 * mtd_weight; // homogametic-homogametic;
    // don't need now.
    */

    uint32_t num_sample = index_keep.size();
    std::vector<float> w_grm(num_sample);
    std::vector<float> w_N(num_sample);

    uint32_t *po_N = N;
    //LOGGER << "mtd weight: " << mtd_weight << std::endl;
   
    /*
    std::ofstream osub("test1.txt");
    for(int i = 0; i < part_keep_indices.second + 1; i++){
        osub << sub_miss[i] << std::endl;
    }
    osub.close();
    */
    
    if(bBLAS){
        // The BLAS GRM buffer is column-major [grm_m × grm_n_write] (doubles).
        // Writing row by row requires stride-m reads (one cache miss per column step).
        // To keep memory low we process the output in blocks of BLAS_OUT_BLOCK rows:
        // each block allocates a row-major float slab of at most
        // BLAS_OUT_BLOCK × grm_n_write floats (~300 MB for n=74193, BLOCK=1024),
        // rather than a single grm_m × grm_n_write slab (~22 GB for the full matrix).
        const int grm_n_write = static_cast<int>(part_keep_indices.second + 1);
        constexpr int BLAS_OUT_BLOCK = 1024;

        // Worst-case block: BLAS_OUT_BLOCK rows × grm_n_write cols (last block).
        // Allocate once and reuse to avoid per-iteration malloc/free overhead.
        std::vector<float> grm_block(static_cast<size_t>(BLAS_OUT_BLOCK) * grm_n_write);
        // Sparse path: accumulate formatted text for an entire BLAS_OUT_BLOCK into a
        // pre-reserved string, then fputs once per block.  Avoids O(n²) small heap
        // allocations from constructing a new stringstream per row.
        // Reserve heuristic: BLAS_OUT_BLOCK rows × grm_n_write cols × ~25 chars/entry × 5% density.
        std::string sparse_buf;
        if(isSparse) sparse_buf.reserve(
            static_cast<size_t>(BLAS_OUT_BLOCK) * grm_n_write / 20 * 25);

        for(int block_start = static_cast<int>(part_keep_indices.first);
                block_start < grm_n_write;
                block_start += BLAS_OUT_BLOCK){
            const int block_end  = std::min(block_start + BLAS_OUT_BLOCK, grm_n_write);
            const int block_rows = block_end - block_start;
            // Lower-triangle output for row pair1 only needs columns 0..pair1,
            // so the widest row in this block needs block_end columns.
            const int block_cols = block_end;
            const int local_block_start = block_start - static_cast<int>(part_keep_indices.first);
            {
                using RowMajF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                Eigen::Map<const Eigen::MatrixXd> G_cm(grm, grm_m, grm_n_write);
                Eigen::Map<RowMajF>(grm_block.data(), block_rows, block_cols) =
                    G_cm.block(local_block_start, 0, block_rows, block_cols).cast<float>();
            }

            if(isSparse) sparse_buf.clear();

            for(int pair1 = block_start; pair1 < block_end; pair1++){
                const int local_block_r = pair1 - block_start;
                const uint32_t sub_miss1 = numValidMarkers - sub_miss[pair1];
                const float *row = grm_block.data() + static_cast<size_t>(local_block_r) * block_cols;

                for(int pair2 = 0; pair2 != pair1 + 1; pair2++){
                    const uint32_t sub_N = *(po_N + pair2) + sub_miss1 - sub_miss[pair2];
                    w_N[pair2]   = static_cast<float>(sub_N);
                    w_grm[pair2] = sub_N ? row[pair2] / sub_N * mtd_weight : 0.0f;
                }
                if(!isSparse){
                    fwrite(w_grm.data(), sizeof(float), pair1 + 1, grm_out);
                    fwrite(w_N.data(),  sizeof(float), pair1 + 1, N_out);
                }else{
                    for(int i = 0; i <= pair1; i++){
                        if(w_grm[i] >= thresh){
                            std::format_to(std::back_inserter(sparse_buf),
                                "{}\t{}\t{:.{}g}\n",
                                pair1, i, w_grm[i],
                                std::numeric_limits<float>::digits10 + 2);
                        }
                    }
                }
                po_N += pair1 + 1;
            }
            if(isSparse && !sparse_buf.empty())
                fputs(sparse_buf.c_str(), grm_out);
        } // end block loop
    }
    /* // don't need special case
    else{
        if(!options_b["homogametic_chr"]){
            for(int pair1 = part_keep_indices.first; pair1 != part_keep_indices.second + 1; pair1++){
                uint32_t sub_miss1 = finished_marker - sub_miss[pair1];
                for(int pair2 = 0; pair2 != pair1 + 1; pair2++){
                    uint32_t sub_N = *po_N + sub_miss1 - sub_miss[pair2];
                    w_N[pair2] = (float)sub_N;

                    if(sub_N){
                        w_grm[pair2] = (float)((*po_grm)/sub_N) * mtd_weight;
                    }else{
                        w_grm[pair2] = 0.0;
                    }
                    po_N++;
                    po_grm++;
                }
                //fwrite(w_grm, sizeof(float), pair1 + 1, grm_out);
                //fwrite(w_N, sizeof(float), pair1 + 1, N_out);
                write_GRM(w_grm.data(), w_N.data(), grm_out, N_out, pair1, thresh);
            }
        }else{
            for(uint32_t pair1 = part_keep_indices.first; pair1 != part_keep_indices.second + 1; pair1++){
                uint32_t sub_miss1 = finished_marker - sub_miss[pair1];
                int8_t weight1 = pheno->get_sex(pair1);
                for(uint32_t pair2 = 0; pair2 != pair1 + 1; pair2++){
                    uint32_t sub_N = *po_N + sub_miss1 - sub_miss[pair2];
                    w_N[pair2] = (float)sub_N;

                    if(sub_N){
                        int8_t weight2 = weight1 + pheno->get_sex(pair2);
                        w_grm[pair2] = weights[weight2] * ((*po_grm) /sub_N);

                    }else{
                        w_grm[pair2] = 0.0;
                    }
                    po_N++;
                    po_grm++;
                }
                //fwrite(w_grm, sizeof(float), pair1 + 1, grm_out);
                //fwrite(w_N, sizeof(float), pair1 + 1, N_out);
                write_GRM(w_grm.data(), w_N.data(), grm_out, N_out, pair1, thresh);
            }
        }
    }
    */


    if(grm_out)fclose(grm_out);
    if(N_out)fclose(N_out);
    //t_print(begin, "  GRM deduce finished");
    if(!isSparse){
        LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.bin]");
        LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + o_name + ".grm.N.bin]");
    }else{
        LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.sp]");
    }

}


// Write one tile's normalised rows to already-open output files.
// Uses grm (tile_rows × tile_cols column-major doubles) and
// N   (tile_rows × tile_cols uint32s, rectangular layout) from object state.
//
// w_grm/w_N/grm_block/sparse_buf are caller-owned scratch, sized once by
// processMakeGRM to the worst-case (final, widest) tile and reused across
// every call -- previously these were function-local vectors, reallocated
// fresh (and growing) on every tile call in the tiled loop, the same
// "allocate worst-case once, reuse" treatment the grm/N buffers just above
// this function in processMakeGRM already get, with a comment explaining
// exactly why (avoid the malloc/page-fault churn across many tiles).
void GRM::flush_grm_tile(FILE *grm_out, FILE *N_out,
                         float thresh, bool isSparse, float mtd_weight,
                         std::vector<float>& w_grm, std::vector<float>& w_N,
                         std::vector<float>& grm_block, std::string& sparse_buf){
    const int tile_rs   = grm_tile_rs;
    const int tile_re   = grm_tile_re;
    const int tile_rows = grm_tile_rows;
    const int tile_cols = grm_tile_cols;   // == tile_re

    if (static_cast<int>(w_grm.size()) < tile_re) w_grm.resize(tile_re);
    if (static_cast<int>(w_N.size())   < tile_re) w_N.resize(tile_re);

    constexpr int BLAS_OUT_BLOCK = 1024;

    // Worst-case block for THIS tile: BLAS_OUT_BLOCK rows × tile_cols cols.
    // grm_block is sized by the caller to the final (widest) tile, so this
    // is a no-op resize on every call after the first.
    const size_t needed = static_cast<size_t>(BLAS_OUT_BLOCK) * tile_cols;
    if (grm_block.size() < needed) grm_block.resize(needed);
    // Sparse path: accumulate formatted text for an entire BLAS_OUT_BLOCK into a
    // pre-reserved string, then fputs once per block.  Avoids O(n²) small heap
    // allocations from constructing a new stringstream per row.
    // Reserve heuristic: BLAS_OUT_BLOCK rows × tile_cols cols × ~25 chars/entry × 5% density.
    if(isSparse) sparse_buf.reserve(std::max(sparse_buf.capacity(),
        static_cast<size_t>(BLAS_OUT_BLOCK) * tile_cols / 20 * 25));

    for(int block_start = tile_rs; block_start < tile_re; block_start += BLAS_OUT_BLOCK){
        const int block_end  = std::min(block_start + BLAS_OUT_BLOCK, tile_re);
        const int block_rows = block_end - block_start;
        const int block_cols = block_end;           // lower-triangle max column
        const int local_block_start = block_start - tile_rs;
        {
            using RowMajF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<const Eigen::MatrixXd> G_cm(grm, tile_rows, tile_cols);
            Eigen::Map<RowMajF>(grm_block.data(), block_rows, block_cols) =
                G_cm.block(local_block_start, 0, block_rows, block_cols).cast<float>();
        }

        if(isSparse) sparse_buf.clear();

        for(int pair1 = block_start; pair1 < block_end; pair1++){
            const int local_r = pair1 - block_start;
            const int tile_r  = pair1 - tile_rs;
            const uint32_t sub_miss1 = numValidMarkers - sub_miss[pair1];
            const float    *row   = grm_block.data() + static_cast<size_t>(local_r) * block_cols;
            const uint32_t *N_row = N + static_cast<size_t>(tile_r) * tile_cols;

            for(int pair2 = 0; pair2 <= pair1; pair2++){
                const uint32_t sub_N = N_row[pair2] + sub_miss1 - sub_miss[pair2];
                w_N[pair2]   = static_cast<float>(sub_N);
                w_grm[pair2] = sub_N ? row[pair2] / sub_N * mtd_weight : 0.0f;
            }
            if(!isSparse){
                fwrite(w_grm.data(), sizeof(float), pair1 + 1, grm_out);
                fwrite(w_N.data(),   sizeof(float), pair1 + 1, N_out);
            } else {
                for(int i = 0; i <= pair1; i++){
                    if(w_grm[i] >= thresh){
                        std::format_to(std::back_inserter(sparse_buf),
                            "{}\t{}\t{:.{}g}\n",
                            pair1, i, w_grm[i],
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
//     processMakeGRM with posix_memalign and zeroed per tile.
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
    if(grm_tiling_enabled){
        // Rectangular N layout: N[(pair1 - grm_tile_rs) * grm_tile_cols + pair2]
        uint32_t *po_N_row = N + static_cast<size_t>(grm_index_from - grm_tile_rs) * grm_tile_cols;
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

    uint32_t *po_N_start = N + startPos;
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

void GRM::grm_thread(int grm_index_from, int grm_index_to) {

    double *po_grm;
    uint64_t geno, mask;
    uint64_t *p_geno1, *p_geno2, geno1;
    uint64_t *p_mask1, *p_mask2, mask1;
    uint64_t *cur_geno, *cur_mask;
    uint16_t geno_p1, geno_p2, geno_p3, geno_p4;
    double *table0, *table1, *table2, *table3;
    int cur_geno_block, start_geno_block;
    uint64_t grm_pos_offset = ((uint64_t)grm_index_from + 1 + part_keep_indices.first) * (grm_index_from - part_keep_indices.first) / 2;
    double *po_grm_start = grm + grm_pos_offset;
    for (int cur_block = 0; cur_block != cur_num_block; cur_block++) {
        cur_geno_block = cur_block * num_block_handle;
        start_geno_block = cur_geno_block * index_keep.size();

        table0 = lookup_GRM_table[cur_geno_block];
        table1 = lookup_GRM_table[cur_geno_block + 1];
        table2 = lookup_GRM_table[cur_geno_block + 2];
        table3 = lookup_GRM_table[cur_geno_block + 3];

        cur_geno = (uint64_t *) (geno_buf + start_geno_block);
        cur_mask = (uint64_t *) (mask_buf + start_geno_block);

        po_grm = po_grm_start;

        p_geno1 = cur_geno + grm_index_from;
        p_mask1 = cur_mask + grm_index_from;

        for (int index_pair1 = grm_index_from; index_pair1 != grm_index_to + 1; index_pair1++) {
            geno1 = *(p_geno1);
            mask1 = *(p_mask1);
            //Just for test
            #ifndef NDEBUG
            if(index_pair1 == 0){
                fwrite(p_geno1, sizeof(uint64_t), 1, o_geno0);
                fwrite(p_mask1, sizeof(uint64_t), 1, o_mask0);
            }
            #endif
            p_geno2 = cur_geno;
            p_mask2 = cur_mask;
            if(mask1) {
                for (int index_pair2 = 0; index_pair2 != index_pair1 + 1; index_pair2++) {
                    geno = geno1 + *(p_geno2);
                    mask = mask1 | *(p_mask2);

                    geno |= mask;

                    geno_p1 = (uint16_t) geno;
                    geno_p2 = (uint16_t) (geno >> 16);
                    geno_p3 = (uint16_t) (geno >> 32);
                    geno_p4 = (uint16_t) (geno >> 48);

                    *po_grm += table0[geno_p1] +
                               table1[geno_p2] +
                               table2[geno_p3] +
                               table3[geno_p4];


                    p_geno2++;
                    p_mask2++;
                    po_grm++;
                }
            }else{
                for (int index_pair2 = 0; index_pair2 != index_pair1 + 1; index_pair2++) {
                    geno = geno1 + *(p_geno2);

                    geno |= *(p_mask2);

                    geno_p1 = (uint16_t) geno;
                    geno_p2 = (uint16_t) (geno >> 16);
                    geno_p3 = (uint16_t) (geno >> 32);
                    geno_p4 = (uint16_t) (geno >> 48);

                    *po_grm += table0[geno_p1] +
                               table1[geno_p2] +
                               table2[geno_p3] +
                               table3[geno_p4];

                    p_geno2++;
                    p_mask2++;
                    po_grm++;
                }
            }

            p_geno1++;
            p_mask1++;
        }

    }
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

int GRM::registerOption(map<string, vector<string>>& options_in) {
    int return_value = 0;
    options["out"] = options_in["out"][0];

    if(options_in.find("--grm") != options_in.end()){
        if(options_in["--grm"].size() == 1){
            options["grm_file"] = options_in["--grm"][0];
        }else{
            LOGGER.e(0, "can't handle multiple GRM files");
        }
        options_in.erase("--grm");
        if(options_in.find("--grm-cutoff") != options_in.end() || 
                options_in.find("--grm-singleton") != options_in.end() ||
                options_in.find("--make-bK") != options_in.end()){
            if(options["grm_file"] == options["out"]){
                LOGGER.e(0, "it is not allowed to have the same file name for the input and the output files.");
            }
        }
    }

    addOneFileOption("keep_file", "", "--keep", options_in, options);
    addOneFileOption("remove_file", "", "--remove", options_in, options);
    addOneFileOption("mgrm", "", "--mgrm", options_in, options);

    if(options_in.find("--nMarkers") != options_in.end()){
        vector<string> markers = options_in["--nMarkers"];
        GRM::nMarkerBlock = std::stoi(markers[0]);
    }

    int num_parts = 1;
    int cur_part = 1;

    string part_grm = "--make-grm-part";
    string part_grm_d = "--make-grm-d-part";
    string part_grm_homogametic = "--make-grm-homogametic-part";
    bool bool_part_grm = options_in.find(part_grm) != options_in.end();
    bool bool_part_grm_d = options_in.find(part_grm_d) != options_in.end();
    bool bool_part_grm_homogametic = options_in.find(part_grm_homogametic) != options_in.end();
    string part_grm_symbol = "";
    bool isDominance = false;
    if(bool_part_grm){
        part_grm_symbol = part_grm;
        isDominance = false;
    }
    if(bool_part_grm_d){
        part_grm_symbol = part_grm_d;
        isDominance = true;
    }
    if(bool_part_grm_homogametic){
        part_grm_symbol = part_grm_homogametic;
        isDominance = false;
    }
    if(bool_part_grm && bool_part_grm_d && bool_part_grm_homogametic){
        LOGGER.e(0, "can't specify --make-grm-part, --make-grm-d-part or --make-grm-homogametic-part together.");
    }

    if(bool_part_grm || bool_part_grm_d || bool_part_grm_homogametic){
        if(options_in[part_grm_symbol].size() == 2){
            try{
                num_parts = std::stoi(options_in[part_grm_symbol][0]);
                cur_part = std::stoi(options_in[part_grm_symbol][1]);
            }catch(std::invalid_argument&){
                LOGGER.e(0, part_grm_symbol + " can only deal with integer value.");
            }
            if(num_parts <= 0 || cur_part <=0){
                LOGGER.e(0, part_grm_symbol + " arguments should be >= 1");
            }
            if(num_parts < cur_part){
                LOGGER.e(0, part_grm_symbol + "the 1st parameter (number of parts) can't be smaller than the 2nd parameter");
            }

            std::string s_parts = std::to_string(num_parts);
            std::string c_parts = std::to_string(cur_part);
            options["out"] = options["out"] + ".part_" + s_parts + "_" + std::string(s_parts.length() - c_parts.length(), '0') + c_parts;
            options_in["out"][0] = options["out"];
            options_in.erase(part_grm_symbol);
            if(!bool_part_grm_homogametic){
                std::map<string, vector<string>> t_option;
                t_option["--autosome"] = {};
                Marker::registerOption(t_option);
                processFunctions.push_back("make_grm");
                return_value++;
            }else{
                options_in["--make-grm-homogametic"] = {};
            }
 
        }else{
            LOGGER.e(0, part_grm_symbol + " takes two arguments: the number of total parts and the part number to be calculated currently");
        }
    }

    options["num_parts"] = std::to_string(num_parts);
    options["cur_part"] = std::to_string(cur_part);

    if(options_in.find("--grm-singleton") != options_in.end()){
        options_in["--make-grm"] = {};
    }

    string op_grm_homogametic = "--make-grm-homogametic";
    options_b["homogametic_chr"] = false;
    if(options_in.find(op_grm_homogametic) != options_in.end()){
        /*
        std::map<string, vector<string>> t_option;
        t_option["--chr-homogametic"] = {};
        t_option["--filter-sex"] = {};
        Pheno::registerOption(t_option);
        Marker::registerOption(t_option);
        Geno::registerOption(t_option);
        */
        processFunctions.push_back("make_grm_homogametic");
        options_b["homogametic_chr"] = true;
        options_in.erase(op_grm_homogametic);
        return_value++;
    }

   
    string op_grm_d = "--make-grm-d";
    if(options_in.find(op_grm_d) != options_in.end()){
        isDominance = true;
        processFunctions.push_back("make_grm");
        options_in.erase(op_grm_d);

        std::map<string, vector<string>> t_option;
        t_option["--autosome"] = {};
        Marker::registerOption(t_option);

        return_value++;
    }

    options_b["isMtd"] = false;
    string op_grm_mtd = "--make-grm-alg";
    if(options_in.find(op_grm_mtd) != options_in.end()){
        if(std::stoi(options_in[op_grm_mtd][0]) > 0)
        options_b["isMtd"] = true;
    }

        /*
    string op_grm_sparse = "--make-grm-sparse";
    if(options_in.find(op_grm_sparse) != options_in.end()){
        processFunctions.push_back("make_grm");
        return_value++;
        if(options_in[op_grm_sparse].size() == 0){
            options_d["grm_cutoff"] = 0.05;
        }else if(options_in[op_grm_sparse].size() >= 1){
            options_d["grm_cutoff"] = std::stod(options_in[op_grm_sparse][0]);
        }
        options_in.erase(op_grm_sparse);
    }
        
        */
    string op_grm_sparse = "--sparse-cutoff";
    if(options_in.find(op_grm_sparse) != options_in.end()){
        if(options_in[op_grm_sparse].size() == 0){
            options_d["sparse_cutoff"] = 0.05;
        }else if(options_in[op_grm_sparse].size() >= 1){
            options_d["sparse_cutoff"] = std::stod(options_in[op_grm_sparse][0]);
        }
    }
 

    string op_grm_unify = "--unify-grm";
    if(options_in.find(op_grm_unify) != options_in.end()){
        processFunctions.push_back("unify_grm");
        options_in.erase(op_grm_unify);
        return_value++;
    }

    string op_grm = "--subtract-grm";
    if(options_in.find(op_grm) != options_in.end()){
        processFunctions.push_back("subtract_grm");
        options_in.erase(op_grm);
        return_value++;
    }

    string op_merge_grm_streaming = "--merge-grm-streaming";
    if(options_in.find(op_merge_grm_streaming) != options_in.end()){
        if(options.find("mgrm") == options.end() || options["mgrm"].empty())
            LOGGER.e(0, "--merge-grm-streaming requires --mgrm <file>.");
        processFunctions.push_back("merge_grm_streaming");
        options_in.erase(op_merge_grm_streaming);
        return_value++;
    }

    const string op_merge_block_rows = "--merge-grm-row-block";
    options_d["merge_grm_row_block"] = 4000.0;
    if(options_in.find(op_merge_block_rows) != options_in.end()){
        if(options_in[op_merge_block_rows].size() != 1)
            LOGGER.e(0, op_merge_block_rows + " requires exactly one integer value.");
        int block_rows = 0;
        try{
            block_rows = std::stoi(options_in[op_merge_block_rows][0]);
        }catch(std::invalid_argument&){
            LOGGER.e(0, op_merge_block_rows + " must be an integer.");
        }
        if(block_rows <= 0)
            LOGGER.e(0, op_merge_block_rows + " must be > 0.");
        options_d["merge_grm_row_block"] = static_cast<double>(block_rows);
        options_in.erase(op_merge_block_rows);
    }

    if(options_in.find("--make-grm") != options_in.end()){
        isDominance = false;
        if(options.find("grm_file") == options.end()){
            processFunctions.push_back("make_grm");
            options_in.erase("--make-grm");
            std::map<string, vector<string>> t_option;
            if(options_in.find("--chr") != options_in.end()){
                t_option["--chr"] = options_in["--chr"];
                options_in.erase("--chr");
            } else {
                t_option["--autosome"] = {};
            }
            Marker::registerOption(t_option);
            return_value++;
        }else{
            if(options_in.find("--grm-cutoff") != options_in.end()){
                if(options_in["--grm-cutoff"].size() == 1){
                    options_d["grm_cutoff"] = std::stod(options_in["--grm-cutoff"][0]);
                }else{
                    LOGGER.e(0, "--grm-cutoff can't deal with more than one value currently");
                }
                if(options.find("grm_file") == options.end()){
                    LOGGER.e(0, "can't find the --grm flag that is essential to --grm-cutoff");
                }

                processFunctions.push_back("grm_cutoff");
                options_in.erase("--grm-cutoff");

                return_value++;
            }

            string opitem = "--grm-singleton";

            if(options_in.find(opitem) != options_in.end()){
                if(options_in[opitem].size() == 1){
                    options_d["grm_cutoff"] = std::stod(options_in[opitem][0]);
                }else{
                    LOGGER.e(0, opitem + " can't deal with more than one value currently.");
                }
                if(options.find("grm_file") == options.end()){
                    LOGGER.e(0, "can't find the --grm flag that is essential to " + opitem);
                }

                processFunctions.push_back("grm_cutoff");
                options_in.erase(opitem);
                options["no_grm"] = "true";
                options["cutoff_detail"] = "true";

                return_value++;
            }


            if(options_in.find("--cutoff-detail") != options_in.end()){
                options["cutoff_detail"] = "true";
            }
        }

    }


    vector<string> flags = {"--make-bK-sparse", "--make-bK"};
    vector<bool> isSparse = {true, false};
    for(int index = 0; index != flags.size(); index++){
        string curFlag = flags[index];
        if(options_in.find(curFlag) != options_in.end()){
            if(options.find("grm_file") == options.end()){
                LOGGER.e(0, "can't find the --grm flag that is essential to " + curFlag);
            }
            if(options_in[curFlag].size() == 1){
                options_d["grm_cutoff"] = std::stod(options_in[curFlag][0]);
                options_b["sparse"] = isSparse[index]; 
                processFunctions.push_back("make_fam");
            }else if(options_in[curFlag].size() == 2){
                options_d["grm_cutoff"] = std::stod(options_in[curFlag][0]);
                options_b["sparse"] = isSparse[index]; 
                options_d["grm_set_value"] = std::stod(options_in[curFlag][1]);
                processFunctions.push_back("make_fam");
            }else{
                LOGGER.e(0, curFlag + " can't deal with more than one value currently.");
            }
            return_value++;
        }
    }

    options_b["isDominance"] = isDominance;

    string op_tile_budget = "--GRM-tile-budget";
    if(options_in.find(op_tile_budget) != options_in.end()){
        if(options_in[op_tile_budget].size() != 1)
            LOGGER.e(0, op_tile_budget + " requires exactly one value, in GB, e.g. 10.");
        // Budget covers only the GRM tile buffer itself (grm + N, 8+4 bytes/elem),
        // not genotype buffers or other working memory.
        double budget_gb;
        try{
            budget_gb = std::stod(options_in[op_tile_budget][0]);
        } catch(...) {
            LOGGER.e(0, op_tile_budget + ": can't parse GB value '" + options_in[op_tile_budget][0] + "'.");
            budget_gb = 0; // unreachable, silences -Wmaybe-uninitialized
        }
        if(budget_gb <= 0)
            LOGGER.e(0, op_tile_budget + " must be a positive number of GB, e.g. 10.");
        options_d["grm_tile_budget_bytes"] = budget_gb * 1024.0 * 1024.0 * 1024.0;
        options_in.erase(op_tile_budget);
    }

    options["use_blas"] = "yes";
    /*
    auto it = std::ranges::find(processFunctions, "make_grm");
    if(it != processFunctions.end()){
        if((!options_b["isDominance"]) && (!options_b["isMtd"])){
        }
    }

    string op_blas = "--noblas";
    if(options_in.find(op_blas) != options_in.end()){
        if(options.find("use_blas")!=options.end()){
            options.erase("use_blas");
        }
        options_in.erase(op_blas);
    }
    */

    return return_value;

}

void GRM::processMakeGRM(){
    gbufitems = new GenoBufItem[nMarkerBlock];
    {
        const int markerPerN = static_cast<int>(sizeof(uintptr_t) * CHAR_BIT);
        const int numNSampleBlock = (grm_n + markerPerN - 1) / markerPerN;
        const int numNblockMax = (nMarkerBlock + markerPerN - 1) / markerPerN;
        sampleMissBuf.resize(static_cast<size_t>(numNSampleBlock) * markerPerN * numNblockMax);
    }
    /*
    uint32_t sampleCT, missPtrSize; 
    geno->setGenoItemSize(sampleCT, missPtrSize);
    for(int i = 0; i < nMarkerBlock; i++){
        gbufitems[i].geno.resize(sampleCT);
        gbufitems[i].missing.resize(missPtrSize);
    }
    */
    this->num_byte_geno = sizeof(double) * nMarkerBlock * stdGenoLD;
    int ret = posix_memalign((void **)&stdGeno, 64, num_byte_geno);
    if(ret != 0){
        LOGGER.e(0, "can't allocate enough memory for the genotype buffer.");
    }
    
    vector<function<void (uintptr_t *, std::span<const uint32_t>)>> callBacks;
    if(options.find("use_blas") != options.end()){
        callBacks.push_back([this](uintptr_t *buf, std::span<const uint32_t> idx){
            calculate_GRM_blas(buf, idx);
        });
    }else{
        //callBacks.push_back(bind(&GRM::calculate_GRM, &grm, _1, _2));
        LOGGER.e(0, "the original version has been deleted. Please use GCTA >= 1.92.4");
    }
    geno->setGRMMode(true, isDominance);
    const bool isSTD = !isMtd;
    const vector<uint32_t> processIndex = marker->get_extract_index_autosome();
    sd.reserve(processIndex.size());

    grm_tile_budget_bytes = (options_d.count("grm_tile_budget_bytes") > 0)
                    ? static_cast<double>(options_d["grm_tile_budget_bytes"]) : 0;

    grm_tiling_enabled = grm_tile_budget_bytes > 0;

    if(bBLAS && grm_tiling_enabled){
        // ── Block-tiled GRM: cap the grm+N tile buffer to --GRM-tile-budget
        // bytes, auto-sizing rows per tile since the lower-triangle column
        // width grows as tile_rs increases — this keeps memory usage roughly
        // uniform across tiles instead of shrinking-then-huge. ──

        float thresh = -99.0f;
        bool isSparse = false;
        if(options_d.find("sparse_cutoff") != options_d.end()){
            thresh   = static_cast<float>(options_d["sparse_cutoff"]);
            isSparse = true;
            LOGGER.i(0, "Saving sparse GRM with a cutoff " + to_string(thresh) + "...");
        } else {
            LOGGER.i(0, "Saving GRM...");
        }

        FILE *grm_out = nullptr, *N_out = nullptr;
        if(isSparse){
            grm_out = fopen((o_name + ".grm.sp").c_str(), "wb");
            if(!grm_out) LOGGER.e(0, "can't open " + o_name + ".grm.sp to write");
        } else {
            grm_out = fopen((o_name + ".grm.bin").c_str(), "wb");
            N_out   = fopen((o_name + ".grm.N.bin").c_str(), "wb");
            if(!grm_out || !N_out)
                LOGGER.e(0, "can't open " + o_name + ".grm.bin or .grm.N.bin to write");
            // 4 MiB matches a typical Lustre stripe size and avoids sub-stripe metadata ops.
            setvbuf(grm_out, nullptr, _IOFBF, 4 << 20);
            setvbuf(N_out,   nullptr, _IOFBF, 4 << 20);
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
                LOGGER.e(0, "--GRM-tile-budget is too small: the next row (sample offset "
                            + to_string(rs) + ") alone needs "
                            + to_string(need_bytes / (1024.0*1024.0*1024.0)).substr(0, 6)
                            + " GB for its GRM+N row, but the budget is only "
                            + to_string(grm_tile_budget_bytes / (1024.0*1024.0*1024.0)).substr(0, 6)
                            + " GB. Increase --GRM-tile-budget.");
            }
            tile_ranges.push_back({rs, re});
            rs = re;
        }
        LOGGER.i(0, "Block-tiled GRM (memory budget "
                    + to_string(grm_tile_budget_bytes / (1024.0*1024.0*1024.0)).substr(0, 6)
                    + " GB): " + to_string(tile_ranges.size()) + " tile(s), rows/tile ranging "
                    + to_string(tile_ranges.front().second - tile_ranges.front().first) + "-"
                    + to_string(tile_ranges.back().second - tile_ranges.back().first) + ".");

        // Allocate for the worst-case tile once and reuse — avoids repeated mmap/munmap
        // and the associated page-fault storms that dominate sys time. Every tile is
        // constructed to fit within budget_elems, so this scan is just a safety net
        // against any boundary-solver rounding.
        size_t max_tile_elems = 0;
        for(const auto &range : tile_ranges){
            max_tile_elems = std::max(max_tile_elems,
                                      static_cast<size_t>(range.second - range.first) * range.second);
        }
        if(posix_memalign((void**)&grm, 64, max_tile_elems * sizeof(double)) != 0)
            LOGGER.e(0, "Can't allocate GRM tile buffer.");
        if(posix_memalign((void**)&N, 64, max_tile_elems * sizeof(uint32_t)) != 0)
            LOGGER.e(0, "Can't allocate N tile buffer.");

        // flush_grm_tile's output-formatting scratch, sized once to the final
        // (widest) tile -- tile_ranges is monotonically widening, so
        // tile_ranges.back() is always the worst case. Reused across every
        // flush_grm_tile call below via resize-if-needed, same rationale as
        // the grm/N allocation just above (avoid malloc/page-fault churn
        // repeated once per tile).
        constexpr int BLAS_OUT_BLOCK = 1024;
        const int widest_tile_cols = tile_ranges.back().second;
        std::vector<float> flush_w_grm(widest_tile_cols);
        std::vector<float> flush_w_N(widest_tile_cols);
        std::vector<float> flush_grm_block(static_cast<size_t>(BLAS_OUT_BLOCK) * widest_tile_cols);
        std::string flush_sparse_buf;

        // sub_miss / sd / numValidMarkers / finished_marker depend only on global
        // genotype state (all markers × all samples), not on which tile row-range is
        // being processed.  Initialise them once here and let calculate_GRM_blas
        // populate them on the first tile; subsequent tiles reuse the values via
        // grm_skip_global_state so the expensive flip64+popcount work is done once.
        // +64: false-sharing guard — see matching comment in the constructor (non-tiled path).
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

            const size_t tile_elems = static_cast<size_t>(grm_tile_rows) * grm_tile_cols;
            const double tile_gb    = tile_elems * 12.0 / (1024.0 * 1024.0 * 1024.0);
            LOGGER.i(0, "  Tile rows " + to_string(tile_rs) + "-" + to_string(tile_re - 1)
                        + " (" + to_string(tile_gb).substr(0, 4) + " GB grm+N)");

            memset(grm, 0, tile_elems * sizeof(double));
            memset(N,   0, tile_elems * sizeof(uint32_t));

            // Rebuild thread pair ranges for this tile's row range [tile_rs, tile_re).
            // N_thread and the grm accumulation buffer are both sized for this tile, so the
            // pairs must index into the tile's local coordinate space (global indices
            // tile_rs .. tile_re-1), not the full part_keep_indices range.
            index_grm_pairs.clear();
            vector<uint32_t> tile_parts = divide_parts(tile_rs, tile_re - 1, num_thread);
            index_grm_pairs.push_back(std::make_pair(tile_rs, tile_parts[0]));
            for(int i = 1; i < (int)tile_parts.size(); i++)
                index_grm_pairs.push_back(std::make_pair(tile_parts[i-1] + 1, tile_parts[i]));

            const bool is_first_tile = !grm_skip_global_state;
            geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);

            // After the first tile sub_miss/sd/numValidMarkers are complete.
            // All subsequent tiles skip that work.
            grm_skip_global_state = true;

            // mtd_weight depends only on sd[]/numValidMarkers, which are frozen after
            // the first tile (see above) -- compute once here rather than every tile.
            if(is_first_tile && options_b["isMtd"]){
                float weight = 0.0f;
                if(!isDominance)
                    for(uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i];
                else
                    for(uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i] * sd[i];
                if(weight > 0.0f) mtd_weight = static_cast<float>(numValidMarkers) / weight;
            }

            flush_grm_tile(grm_out, N_out, thresh, isSparse, mtd_weight,
                          flush_w_grm, flush_w_N, flush_grm_block, flush_sparse_buf);
        }

        posix_mem_free(grm); grm = nullptr;
        posix_mem_free(N);   N   = nullptr;

        if(grm_out) fclose(grm_out);
        if(N_out)   fclose(N_out);
        LOGGER << "  Used " << numValidMarkers << " valid SNPs." << std::endl;
        if(!isSparse){
            LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.bin]");
            LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + o_name + ".grm.N.bin]");
        } else {
            LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.sp]");
        }
    } else {
        // Non-tiled path (!grm_tiling_enabled): single pass, original behaviour.
        LOGGER << "Computing GRM..." << std::endl;
        geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);
        LOGGER << "  Used " << numValidMarkers << " valid SNPs." << std::endl;
        deduce_GRM();
    }

    delete[] gbufitems;
    posix_mem_free(stdGeno);
    geno->setGRMMode(false, false);
}

void GRM::processMakeGRMX(){
    gbufitems = new GenoBufItem[nMarkerBlock];
    {
        const int markerPerN = static_cast<int>(sizeof(uintptr_t) * CHAR_BIT);
        const int numNSampleBlock = (grm_n + markerPerN - 1) / markerPerN;
        const int numNblockMax = (nMarkerBlock + markerPerN - 1) / markerPerN;
        sampleMissBuf.resize(static_cast<size_t>(numNSampleBlock) * markerPerN * numNblockMax);
    }
    /*
    uint32_t sampleCT, missPtrSize; 
    geno->setGenoItemSize(sampleCT, missPtrSize);
    for(int i = 0; i < nMarkerBlock; i++){
        gbufitems[i].geno.resize(sampleCT);
        gbufitems[i].missing.resize(missPtrSize);
    }
    */
    this->num_byte_geno = sizeof(double) * nMarkerBlock * stdGenoLD;
    int ret = posix_memalign((void **)&stdGeno, 64, num_byte_geno);
    if(ret != 0){
        LOGGER.e(0, "can't allocate enough memory for the genotype buffer.");
    }
    
    vector<function<void (uintptr_t *, std::span<const uint32_t>)>> callBacks;
    if(options.find("use_blas") != options.end()){
        callBacks.push_back([this](uintptr_t *buf, std::span<const uint32_t> idx){
            calculate_GRM_blas(buf, idx);
        });
    }else{
        //callBacks.push_back(bind(&GRM::calculate_GRM, &grm, _1, _2));
        LOGGER.e(0, "the original version has been deleted. Please use GCTA >= 1.92.4");
    }
    geno->setGRMMode(true, isDominance);
    bool isSTD = true;
    if(isMtd) isSTD = false;
    vector<uint32_t> processIndex = marker->get_extract_index_X();
    sd.reserve(processIndex.size());
    LOGGER << "Computing GRM..." << std::endl;
    geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);
    LOGGER << numValidMarkers << " valid SNPs are included."<< std::endl;
    deduce_GRM();
    delete[] gbufitems;
    posix_mem_free(stdGeno);
    geno->setGRMMode(false, false);

}

void GRM::processMain() {
    vector<function<void (uint64_t *, int)>> callBacks;
    for(auto &process_function : processFunctions){
        if(process_function == "make_grm"){
            LOGGER.i(0, "Note: GRM is computed using the SNPs on the autosomes.");
            Pheno pheno;
            Marker marker;
            GRM grm(&pheno, &marker);
            grm.processMakeGRM();
            return;
        }

        if(process_function == "make_grm_homogametic"){
            LOGGER.i(0, "Note: this function treats homogametic chromosomes as non-PAR regions.");

            Pheno pheno;
            Marker marker;
            GRM grm(&pheno, &marker);
            grm.processMakeGRMX();
            return;
        }


        if(process_function == "grm_cutoff"){
            GRM grm;
            bool no_grm = false;
            if(options.find("no_grm") != options.end()) {
                no_grm = true;
            }
            grm.cut_rel(options_d["grm_cutoff"], no_grm);
           // grm.loop_block(callBacks);

        }

        if(process_function == "make_fam"){
            GRM grm;
            float *grm_value = NULL;
            float value;
            if(options_d.find("grm_set_value") != options_d.end()){
                value = options_d["grm_set_value"];
                grm_value = &value;
            }
            grm.prune_fam(options_d["grm_cutoff"], options_b["sparse"], grm_value);
        }

        if(process_function == "unify_grm"){
            GRM grm;
            grm.unify_grm(options["mgrm"], options["out"]);
        }
        if(process_function == "subtract_grm"){
            GRM grm;
            grm.subtract_grm(options["mgrm"], options["out"]);
        }

        if(process_function == "merge_grm_streaming"){
            GRM grm;
            const int row_block_rows = static_cast<int>(options_d["merge_grm_row_block"]);
            grm.merge_grm_streaming(options["mgrm"], options["out"], row_block_rows);
        }
    }

}