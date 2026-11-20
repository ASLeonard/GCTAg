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
#include "OptionIO.h"
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
#include <array>
#include <bit>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include "utils.hpp"
#include "grm_binary_io.hpp"
#include <omp.h>
#include "third_party/plink-ng/2.0/include/plink2_bits.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <csignal>

using std::to_string;

namespace gcta_grm_detail {

constexpr size_t kMaxPosixIoChunk = 1ULL << 30; // 1 GiB (safe on macOS & Linux)

class GrmUniqueFd {
public:
    GrmUniqueFd() = default;
    explicit GrmUniqueFd(int fd) noexcept : fd_(fd) {}
    ~GrmUniqueFd() { reset(); }
    GrmUniqueFd(const GrmUniqueFd&) = delete;
    GrmUniqueFd& operator=(const GrmUniqueFd&) = delete;
    GrmUniqueFd(GrmUniqueFd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    GrmUniqueFd& operator=(GrmUniqueFd&& o) noexcept {
        if (this != &o) {
            reset(o.fd_);
            o.fd_ = -1;
        }
        return *this;
    }
    int get() const noexcept { return fd_; }
    explicit operator bool() const noexcept { return fd_ >= 0; }
    void reset(int new_fd = -1) noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = new_fd;
    }
private:
    int fd_ = -1;
};

inline GrmUniqueFd open_posix_read(const std::string& path) noexcept {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    return GrmUniqueFd(::open(path.c_str(), flags));
}

inline GrmUniqueFd open_posix_write_trunc(const std::string& path) noexcept {
    int flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    return GrmUniqueFd(::open(path.c_str(), flags, 0644));
}

inline bool write_exact_posix(int fd, const void* buf, size_t count, int& err_out) noexcept {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxPosixIoChunk, count - done);
        const ssize_t w = ::write(fd, p + done, step);
        if (w < 0) {
            if (errno == EINTR) continue;
            err_out = errno;
            return false;
        }
        if (w == 0) {
            err_out = EIO;
            return false;
        }
        done += static_cast<size_t>(w);
    }
    return true;
}

inline bool read_exact_posix(int fd, void* buf, size_t count, int& err_out, bool& eof_out) noexcept {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    eof_out = false;
    while (done < count) {
        const size_t step = std::min(kMaxPosixIoChunk, count - done);
        const ssize_t r = ::read(fd, p + done, step);
        if (r < 0) {
            if (errno == EINTR) continue;
            err_out = errno;
            return false;
        }
        if (r == 0) {
            eof_out = true;
            return false;
        }
        done += static_cast<size_t>(r);
    }
    return true;
}

inline std::pair<int, int> lower_tri_col_range(int cols, int tid, int nthreads) noexcept {
    if (cols <= 0 || nthreads <= 0) return {0, 0};
    if (nthreads == 1) return {0, cols};
    const double W = 0.5 * static_cast<double>(cols) * (static_cast<double>(cols) + 1.0);
    auto col_at = [&](int k) -> int {
        if (k <= 0) return 0;
        if (k >= nthreads) return cols;
        const double rem = W * (static_cast<double>(nthreads - k) / static_cast<double>(nthreads));
        const double tail = std::floor((-1.0 + std::sqrt(1.0 + 8.0 * rem)) * 0.5);
        return std::clamp(cols - static_cast<int>(tail), 0, cols);
    };
    return {col_at(tid), col_at(tid + 1)};
}

// Finds the largest `re` in (rs, full_re] such that a GRM tile spanning rows
// [rs, re) fits within budget_elems *actually touched* elements.
//
// Solved as the positive root of re^2 + re - (2*budget_elems + rs*(rs+1)) = 0
// in floating-point arithmetic, then floored and clamped to full_re.
inline int solve_tile_budget_end(int rs, int full_re, double budget_elems) {
    const double drs = rs;
    const double disc = 1.0 + 4.0 * (2.0 * budget_elems + drs * (drs + 1.0));
    int re = static_cast<int>(std::floor((-1.0 + std::sqrt(disc)) / 2.0));
    return std::min(re, full_re);
}

// True when a single tile spanning the whole GRM [full_rs, full_re) already
// fits within budget_elems -- i.e. tiling would produce exactly one tile.
inline bool grm_tile_budget_covers_whole(int full_rs, int full_re, double budget_elems) {
    const double drs = full_rs, dre = full_re;
    const double whole_elems = (dre * (dre + 1.0) - drs * (drs + 1.0)) / 2.0;
    return whole_elems <= budget_elems;
}

} // namespace gcta_grm_detail

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

        index_grm_pairs.push_back(std::make_pair(0, static_cast<int>(parts[0])));
        byte_part_grms.push_back((static_cast<uint64_t>(parts[0]) + 2ULL) * (static_cast<uint64_t>(parts[0]) + 1ULL) / 2ULL);
        for(size_t index = 1; index != parts.size(); index++){
            index_grm_pairs.push_back(std::make_pair(static_cast<int>(parts[index - 1] + 1), static_cast<int>(parts[index])));
            byte_part_grms.push_back((static_cast<uint64_t>(parts[index - 1]) + static_cast<uint64_t>(parts[index]) + 3ULL)
                                     * (static_cast<uint64_t>(parts[index]) - static_cast<uint64_t>(parts[index - 1])) / 2ULL);
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

    const uint64_t n_sub = static_cast<uint64_t>(common_id.size());
    const uint64_t grm_size = (1ULL + n_sub) * n_sub / 2ULL;
    const uint64_t grm_file_size = grm_size * sizeof(float);

    const std::string p_grm1  = files[0] + ".grm.bin";
    const std::string p_grmN1 = files[0] + ".grm.N.bin";
    const std::string p_grm2  = files[1] + ".grm.bin";
    const std::string p_grmN2 = files[1] + ".grm.N.bin";

    auto fd_grm1  = gcta_grm_detail::open_posix_read(p_grm1);
    auto fd_grmN1 = gcta_grm_detail::open_posix_read(p_grmN1);
    auto fd_grm2  = gcta_grm_detail::open_posix_read(p_grm2);
    auto fd_grmN2 = gcta_grm_detail::open_posix_read(p_grmN2);

    auto check_fd_size = [&](const gcta_grm_detail::GrmUniqueFd& ufd, const std::string& path) {
        struct stat st{};
        if (!ufd || ::fstat(ufd.get(), &st) != 0 || st.st_size < 0 ||
            static_cast<uint64_t>(st.st_size) != grm_file_size) {
            LOGGER.e(0, "The size of [" + path + "] is not correct.");
        }
    };
    check_fd_size(fd_grm1,  p_grm1);
    check_fd_size(fd_grmN1, p_grmN1);
    check_fd_size(fd_grm2,  p_grm2);
    check_fd_size(fd_grmN2, p_grmN2);

    constexpr uint64_t kChunkElems = 26214400ULL; // 100 MiB per buffer
    const uint64_t alloc_elems = std::min(grm_size, kChunkElems);
    AlignedRawBuffer<float, 64> buf1(alloc_elems);
    AlignedRawBuffer<float, 64> buf2(alloc_elems);
    AlignedRawBuffer<float, 64> bufN1(alloc_elems);
    AlignedRawBuffer<float, 64> bufN2(alloc_elems);

    const uint64_t loop_num = (grm_size + kChunkElems - 1ULL) / kChunkElems;
    const uint64_t remain_item = grm_size % kChunkElems;
    LOGGER.i(0, "Subtracting GRMs...");
    const std::string po_grm  = out_file + ".grm.bin";
    const std::string po_grmN = out_file + ".grm.N.bin";
    auto fdo_grm  = gcta_grm_detail::open_posix_write_trunc(po_grm);
    auto fdo_grmN = gcta_grm_detail::open_posix_write_trunc(po_grmN);
    if (!fdo_grm) {
        LOGGER.error(0, "can't write to [{}].", po_grm);
    }
    if (!fdo_grmN) {
        LOGGER.error(0, "can't write to [{}].", po_grmN);
    }

    for (uint64_t i = 0; i < loop_num; i++) {
        const uint64_t cur_items = (i == loop_num - 1 && remain_item != 0) ? remain_item : kChunkElems;
        const size_t cur_bytes = static_cast<size_t>(cur_items) * sizeof(float);

        bool ok_r1 = true, ok_r2 = true, ok_rn1 = true, ok_rn2 = true;
        int err_r1 = 0, err_r2 = 0, err_rn1 = 0, err_rn2 = 0;
        bool eof_r1 = false, eof_r2 = false, eof_rn1 = false, eof_rn2 = false;
        #pragma omp parallel sections num_threads(4)
        {
            #pragma omp section
            { ok_r1  = gcta_grm_detail::read_exact_posix(fd_grm1.get(),  buf1.data(),  cur_bytes, err_r1,  eof_r1); }
            #pragma omp section
            { ok_r2  = gcta_grm_detail::read_exact_posix(fd_grm2.get(),  buf2.data(),  cur_bytes, err_r2,  eof_r2); }
            #pragma omp section
            { ok_rn1 = gcta_grm_detail::read_exact_posix(fd_grmN1.get(), bufN1.data(), cur_bytes, err_rn1, eof_rn1); }
            #pragma omp section
            { ok_rn2 = gcta_grm_detail::read_exact_posix(fd_grmN2.get(), bufN2.data(), cur_bytes, err_rn2, eof_rn2); }
        }
        if (!ok_r1 || !ok_r2 || !ok_rn1 || !ok_rn2) {
            LOGGER.e(0, "subtract_grm: failed reading input GRM files.");
        }

        // On the first chunk, check that file[0] has at least as many markers as file[1].
        // A negative N_diff means the files are in the wrong order (chr GRM listed first).
        if (i == 0 && cur_items > 0 && bufN1[0] < bufN2[0]) {
            LOGGER.e(0, "subtract_grm: the first GRM [" + files[0] + "] has fewer markers than "
                        "the second [" + files[1] + "] in the first pair. "
                        "The larger (all-chromosome) GRM must be listed first.");
        }

        float* __restrict__ p1  = buf1.data();
        const float* __restrict__ p2  = buf2.data();
        float* __restrict__ pn1 = bufN1.data();
        const float* __restrict__ pn2 = bufN2.data();
        #pragma omp parallel for schedule(static)
        for (uint64_t j = 0; j < cur_items; j++) {
            const float n1 = pn1[j];
            const float n2 = pn2[j];
            const float nd = n1 - n2;
            pn1[j] = nd;
            // Guard against zero denominator: a pair where all markers in the subtracted
            // GRM are also missing in the full GRM (N_all == N_chr for that pair).
            p1[j] = (nd > 0.0f)
                ? static_cast<float>((static_cast<double>(p1[j]) * static_cast<double>(n1)
                                    - static_cast<double>(p2[j]) * static_cast<double>(n2))
                                   / static_cast<double>(nd))
                : 0.0f;
        }

        bool ok_wg = true, ok_wn = true;
        int err_wg = 0, err_wn = 0;
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            { ok_wg = gcta_grm_detail::write_exact_posix(fdo_grm.get(),  buf1.data(),  cur_bytes, err_wg); }
            #pragma omp section
            { ok_wn = gcta_grm_detail::write_exact_posix(fdo_grmN.get(), bufN1.data(), cur_bytes, err_wn); }
        }
        if (!ok_wg) {
            LOGGER.e(0, "can't write to [" + po_grm + "]: " + std::string(std::strerror(err_wg)));
        }
        if (!ok_wn) {
            LOGGER.e(0, "can't write to [" + po_grmN + "]: " + std::string(std::strerror(err_wn)));
        }
    }
    fdo_grm.reset();
    fdo_grmN.reset();
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
        for(uint64_t j = 0; j < largest_grm_size; j++){
            uint64_t byte_start_grm = (1ULL + j) * j / 2ULL * sizeof(float);
            start_pos[j] = byte_start_grm;
        }

        for(uint32_t j = 0; j < size_grm; j++){
            uint64_t grm_index = p_index[j];
            uint64_t num_grm = grm_index + 1;
            uint64_t byte_start_grm = start_pos[grm_index];
            fseeko(h_grm.get(), static_cast<off_t>(byte_start_grm), SEEK_SET);
            if(fread(buf.data(), sizeof(float), num_grm, h_grm.get()) != num_grm){
                LOGGER.e(0, "error in reading [" + file_name + "], in position " + to_string(ftello(h_grm.get())), ".");
            }
            uint64_t size_write = static_cast<uint64_t>(j) + 1ULL;
            for(uint64_t k = 0; k < size_write; k++){
                uint32_t cur_index = p_index[k];
                float grm_value;
                if(cur_index < num_grm){
                    grm_value = buf[cur_index];
                }else{
                    //fill the other parts
                    uint64_t bytes_offset = start_pos[cur_index] + grm_index * sizeof(float);
                    fseeko(h_grm.get(), static_cast<off_t>(bytes_offset), SEEK_SET);
                    if(fread(&grm_value, sizeof(float), 1, h_grm.get()) != 1){
                        LOGGER.e(0, "error in reading [" + file_name + "], in position " + to_string(bytes_offset), ".");
                    }
                }
                wbuf[k] = grm_value;
            }
            //last element
            //*(wbuf + j) = *(buf + grm_index);

            if(size_write != fwrite(wbuf.data(), sizeof(float), size_write, h_wgrm.get())){
                LOGGER.e(0, "error in writing to [" + wfile_name + "], pos: " + std::to_string(ftello(h_wgrm.get())) + ".");
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
            const size_t write_items = static_cast<size_t>(cur_grm_buf - out_grm_buf.data());
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

        const size_t write_items = static_cast<size_t>(cur_N_buf - out_N_buf.data());
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
    n_markers_explicit_ = opts.n_markers.has_value();
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

    // GRM dimensions for this part
    num_individual = part_keep_indices.second - part_keep_indices.first + 1;
    num_grm = ((uint64_t) part_keep_indices.first + part_keep_indices.second + 2) * num_individual / 2;

    grm_tile_budget_bytes = opts.tile_budget_gb
        ? (*opts.tile_budget_gb * 1024.0 * 1024.0 * 1024.0)
        : 0.0;
    if (opts.sparse_cutoff.has_value()) {
        sparse_cutoff = static_cast<float>(*opts.sparse_cutoff);
    }

    // +64 pads sub_miss past the last valid index by one full cache line (64 × sizeof(uint32_t) = 256 bytes).
    // This prevents OMP threads from writing to adjacent cache lines when two threads operate on
    // neighbouring sample indices near the end of the array (false-sharing guard).
    sub_miss.assign(index_keep.size() + 64, 0);

    //calculate each index in pair thread;
    int num_thread = omp_get_max_threads();
    index_grm_pairs.reserve(num_thread);
    vector<uint32_t> thread_parts = divide_parts(part_keep_indices.first, part_keep_indices.second, num_thread);
    if(num_thread != static_cast<int>(thread_parts.size())){
        LOGGER.w(0, "cannot run in " + to_string(num_thread) + " threads. Use " + to_string(thread_parts.size()) + " instead");
    }

    index_grm_pairs.push_back(std::make_pair(static_cast<int>(part_keep_indices.first), static_cast<int>(thread_parts[0])));
    for(size_t index = 1; index != thread_parts.size(); index++){
        index_grm_pairs.push_back(std::make_pair(static_cast<int>(thread_parts[index - 1] + 1), static_cast<int>(thread_parts[index])));
    }

    isDominance = opts.dominance;
    isMtd = (opts.alg == gcta_cli::GrmAlg::VanRaden);

    // Cache constants derived from part_keep_indices
    grm_n = static_cast<int>(part_keep_indices.second) + 1;
    grm_m = static_cast<int>(part_keep_indices.second) - static_cast<int>(part_keep_indices.first) + 1;
    grm_s_n = grm_n - grm_m;
    grm_bytes_std_geno = sizeof(float) * static_cast<size_t>(grm_n);
    // Round up to a multiple of 64 elements so every column of stdGeno / stdGeno_i8
    // is 64-byte aligned and has capacity >= index_keep.size() (for --make-grm-part).
    const int ld_base = std::max(grm_n, static_cast<int>(index_keep.size()));
    stdGenoLD = (ld_base + 63) / 64 * 64;

    // Auto-size default nMarkerBlock for VanRaden INT8 BED GRM (when --n-markers is not
    // explicitly passed): target <= 2 GiB for stdGeno_i8_kmajor_, clamped to [1024, 4096].
    // Keeps peak RSS modest while lifting VNNI GEMM arithmetic intensity well past the DRAM wall.
    const bool auto_use_int8 = bBLAS && isMtd && !isDominance && !opts.homogametic
                               && (geno->getGenoFormat() == "BED")
                               && (std::getenv("GCTA_DISABLE_INT8_GRM") == nullptr);
    if (n_markers_explicit_) {
        GRM::nMarkerBlock = *opts.n_markers;
    } else if (auto_use_int8) {
        constexpr uint64_t kTargetKMajorBytes = 2ULL << 30;
        const uint64_t raw_k = kTargetKMajorBytes / std::max<uint64_t>(1ULL, static_cast<uint64_t>(grm_n));
        GRM::nMarkerBlock = static_cast<int>(std::clamp<uint64_t>((raw_k / 64ULL) * 64ULL, 1024ULL, 4096ULL));
    } else {
        GRM::nMarkerBlock = 1024;
    }

    // Pre-allocate small scratch buffer; grm/grm_i32/N/sampleMissBuf are allocated in
    // processMakeGRM (and N/sampleMissBuf lazily only when missing genotypes occur).
    validIndexBuf.reserve(static_cast<size_t>(nMarkerBlock));

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

void GRM::ensure_N_allocated() {
    if (N) return;
    if (grm_tiling_enabled) {
        const size_t tile_rect_elems = static_cast<size_t>(grm_tile_rows) * static_cast<size_t>(grm_tile_cols);
        N = make_aligned_uninitialized<uint32_t, 64>(tile_rect_elems);
        if (!N) {
            LOGGER.error(0, "Can't allocate N tile buffer: {:.2f}GB required.",
                         tile_rect_elems * sizeof(uint32_t) / (1024.0 * 1024.0 * 1024.0));
        }
        // Parallel zero first-touch matching index_grm_pairs; upper-triangle pages remain unfaulted.
        uint32_t* const N_ptr = std::assume_aligned<64>(N.get());
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < static_cast<int>(index_grm_pairs.size()); ++idx) {
            const auto [p_from, p_to] = index_grm_pairs[idx];
            for (int pair1 = p_from; pair1 <= p_to; ++pair1) {
                const int r = pair1 - grm_tile_rs;
                std::memset(N_ptr + static_cast<size_t>(r) * grm_tile_cols, 0,
                            static_cast<size_t>(pair1 + 1) * sizeof(uint32_t));
            }
        }
    } else {
        const uint64_t fill_N = (num_grm + num_count_handle - 1) / num_count_handle * num_count_handle;
        N = make_aligned_uninitialized<uint32_t, 64>(fill_N);
        if (!N) {
            LOGGER.error(0, "can't allocate enough memory to store (parted) N: {:.2f}GB required.",
                         fill_N * sizeof(uint32_t) / (1024.0 * 1024.0 * 1024.0));
        }
        uint32_t* const N_ptr = std::assume_aligned<64>(N.get());
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < static_cast<int>(index_grm_pairs.size()); ++idx) {
            const auto [p_from, p_to] = index_grm_pairs[idx];
            const uint64_t off_start = (static_cast<uint64_t>(p_from) + 1ULL + part_keep_indices.first)
                                     * static_cast<uint64_t>(p_from - static_cast<int>(part_keep_indices.first)) / 2ULL;
            const uint64_t off_end   = (static_cast<uint64_t>(p_to) + 2ULL + part_keep_indices.first)
                                     * static_cast<uint64_t>(p_to + 1 - static_cast<int>(part_keep_indices.first)) / 2ULL;
            std::memset(N_ptr + off_start, 0, static_cast<size_t>(off_end - off_start) * sizeof(uint32_t));
        }
        if (fill_N > num_grm) {
            std::memset(N_ptr + num_grm, 0, static_cast<size_t>(fill_N - num_grm) * sizeof(uint32_t));
        }
    }
}

void GRM::promote_i32_to_f64() {
    if (!grm_is_i32_) return;
    const size_t tile_rect_elems = static_cast<size_t>(grm_tile_rows) * static_cast<size_t>(grm_tile_cols);
    if (grm_first_accum) {
        // Nothing accumulated into grm_i32 yet: free it first so peak RSS stays bounded.
        grm_i32.reset();
        grm = make_aligned_uninitialized<double, 64>(tile_rect_elems);
        if (!grm) {
            LOGGER.error(0, "can't allocate enough memory to promote GRM buffer to double: {:.2f}GB required.",
                         tile_rect_elems * sizeof(double) / (1024.0 * 1024.0 * 1024.0));
        }
    } else {
        grm = make_aligned_uninitialized<double, 64>(tile_rect_elems);
        if (!grm) {
            LOGGER.error(0, "can't allocate enough memory to promote GRM buffer to double: {:.2f}GB required.",
                         tile_rect_elems * sizeof(double) / (1024.0 * 1024.0 * 1024.0));
        }
        const int32_t* const src_base = std::assume_aligned<64>(grm_i32.get());
        double* const dst_base        = std::assume_aligned<64>(grm.get());
        const double* const R_ptr     = R_total_.data();
        const double* const R_row     = R_ptr + grm_tile_rs;
        const double S_tot            = S_total_;
        const int tile_rs             = grm_tile_rs;
        const int tile_rows           = grm_tile_rows;

        // Left off-diagonal rectangle: cols [0, tile_rs), all rows [0, tile_rows)
        if (tile_rs > 0) {
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < tile_rs; ++c) {
                const int32_t* __restrict__ src_col = src_base + static_cast<size_t>(c) * tile_rows;
                double* __restrict__ dst_col        = dst_base + static_cast<size_t>(c) * tile_rows;
                const double r_j_plus_S = S_tot - R_ptr[c];
                #pragma omp simd
                for (int r = 0; r < tile_rows; ++r) {
                    dst_col[r] = static_cast<double>(src_col[r]) - R_row[r] + r_j_plus_S;
                }
            }
        }
        // Right diagonal lower triangle: cols [tile_rs, tile_rs + tile_rows), rows [c_local, tile_rows)
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();
            const auto [c_beg, c_end] = gcta_grm_detail::lower_tri_col_range(tile_rows, tid, nth);
            for (int c_local = c_beg; c_local < c_end; ++c_local) {
                const int c = tile_rs + c_local;
                const int32_t* __restrict__ src_col = src_base + static_cast<size_t>(c) * tile_rows;
                double* __restrict__ dst_col        = dst_base + static_cast<size_t>(c) * tile_rows;
                const double r_j_plus_S = S_tot - R_ptr[c];
                #pragma omp simd
                for (int r = c_local; r < tile_rows; ++r) {
                    dst_col[r] = static_cast<double>(src_col[r]) - R_row[r] + r_j_plus_S;
                }
            }
        }
        grm_i32.reset();
    }
    grm_is_i32_ = false;
    const int max_tile_dim = std::min(4096, std::max(1, grm_n));
    const size_t tile_elems = static_cast<size_t>(max_tile_dim) * static_cast<size_t>(max_tile_dim);
    if (grm_i32_tile_.size() < tile_elems) {
        grm_i32_tile_.resize(tile_elems);
    }
}

void GRM::calculate_GRM_blas(uintptr_t *buf, std::span<const uint32_t> markerIndex){
    const int num_marker = static_cast<int>(markerIndex.size());

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

    const int curNumValidMarkers = static_cast<int>(validIndexBuf.size());

    if (curNumValidMarkers == 0) {
        if (!grm_skip_global_state) {
            finished_marker += num_marker;
        }
        return;
    }

    // Compact valid columns forward in-place for the FP32 path when any marker failed filters.
    if (!use_int8_grm && curNumValidMarkers < num_marker) {
        float* const std_geno_base = std::assume_aligned<64>(stdGeno.get());
        for (int k = 0; k < curNumValidMarkers; ++k) {
            const int src_idx = validIndexBuf[k];
            if (src_idx != k) {
                std::memcpy(std_geno_base + static_cast<size_t>(k) * stdGenoLD,
                            std_geno_base + static_cast<size_t>(src_idx) * stdGenoLD,
                            grm_bytes_std_geno);
            }
        }
    }
    if(!grm_skip_global_state){
        for(int i = 0; i < curNumValidMarkers; i++){
            sd.push_back(gbufitems[validIndexBuf[i]].sd);
        }
    }

    constexpr int kTile = 4096;
    const uint32_t full_keep_ct = static_cast<uint32_t>(index_keep.size());
    const int decode_rows = grm_tile_cols;  // == grm_tile_re (tiled) or grm_n (dense)
    bool any_missing_full = false;
    bool tile_has_missing = false;

    for (int m = 0; m < curNumValidMarkers; ++m) {
        if (gbufitems[validIndexBuf[m]].nValidN < full_keep_ct) {
            any_missing_full = true;
            break;
        }
    }
    if (any_missing_full) {
        grm_any_missing_seen_ = true;
    }

    if (use_int8_grm) {
        int8_t* const kmajor_base = std::assume_aligned<64>(stdGeno_i8_kmajor_.get());
        const int ldK = std::max(64, (curNumValidMarkers + 63) / 64 * 64);

        double S_mu2 = 0.0;
        for (int m = 0; m < curNumValidMarkers; ++m) {
            const double mu_m = gbufitems[validIndexBuf[m]].mean;
            mu_vec_[m] = mu_m;
            S_mu2 += mu_m * mu_m;
        }

        // Fused cache-blocked (64x64) decode of raw .bed records of valid markers
        // straight into K-contiguous row-major stdGeno_i8_kmajor_ (decode_rows x ldK)
        // with per-sample row-sum accumulation r_raw_[i] = sum_m g_{im} * mu_m.
        geno->decodeBedI8KMajor(buf, validIndexBuf.data(), curNumValidMarkers, decode_rows,
                                kmajor_base, ldK, mu_vec_.data(), r_raw_.data());

        marker_miss_idx_.clear();
        if (any_missing_full) {
            // Decode full int8 column + missing bitmask of each marker with missing calls.
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
            std::memcpy(r_tilde_.data(), r_raw_.data(), static_cast<size_t>(decode_rows) * sizeof(double));
            const int num_words = (decode_rows + 63) / 64;
            const uintptr_t last_word_mask = (decode_rows % 64 == 0)
                ? ~static_cast<uintptr_t>(0)
                : ((static_cast<uintptr_t>(1) << (decode_rows % 64)) - 1);
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
            tile_has_missing = !marker_miss_idx_.empty();
            if (tile_has_missing) {
                // Build global per-sample missing-marker CSR once per marker block.
                std::fill(sample_miss_offset_.begin(), sample_miss_offset_.begin() + decode_rows + 1, 0);
                for (int s : marker_miss_idx_) {
                    sample_miss_offset_[s + 1]++;
                }
                for (int s = 0; s < decode_rows; ++s) {
                    sample_miss_offset_[s + 1] += sample_miss_offset_[s];
                }
                sample_miss_m_.resize(sample_miss_offset_[decode_rows]);
                std::vector<int> write_pos(sample_miss_offset_.begin(),
                                           sample_miss_offset_.begin() + decode_rows);
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

        // Exactness fallback: promote int32_t -> double if missingness appears in this
        // tile's samples or if total accumulated markers could approach INT32_MAX / 4.
        if (grm_is_i32_ &&
            (tile_has_missing ||
             static_cast<uint64_t>(grm_tile_i32_markers_) + static_cast<uint64_t>(curNumValidMarkers) > kMaxInt32Markers)) {
            promote_i32_to_f64();
        }

        const bool first_accum = grm_first_accum;

        if (grm_is_i32_) {
            // Zero-missing exact int32_t in-place VNNI GRM accumulation (4 B/elem, beta=0/1).
            // r0-outer loop pre-packs A_strip once per strip across all c0 column tiles;
            // diagonal blocks are sub-tiled with a 1024x1024 (4 MiB) L3-resident scratch so
            // upper-triangle pages of grm_i32 remain 100% unfaulted.
            int32_t* const grm_i32_base = std::assume_aligned<64>(grm_i32.get());
            const int32_t beta = first_accum ? 0 : 1;
            constexpr int kDiagSub = 1024;
            if (grm_i32_diag_scratch_.size() < static_cast<size_t>(kDiagSub) * kDiagSub) {
                grm_i32_diag_scratch_.resize(static_cast<size_t>(kDiagSub) * kDiagSub);
            }

            for (int r0 = 0; r0 < grm_tile_rows; r0 += kTile) {
                const int rows = std::min(kTile, grm_tile_rows - r0);
                const int global_r0 = grm_tile_rs + r0;
                const int8_t* const A_strip = kmajor_base + static_cast<size_t>(global_r0) * ldK;

                if (global_r0 > 0) {
                    const int8_t* A_reord_ptr = nullptr;
                    if (global_r0 > kTile) {
                        const size_t rbytes = gcta_u8s8s32_reorder_bt_size(rows, curNumValidMarkers);
                        if (rbytes > 0) {
                            if (i8_reordered_B_.size() < rbytes) i8_reordered_B_.resize(rbytes);
                            gcta_u8s8s32_reorder_bt(rows, curNumValidMarkers, A_strip, ldK, i8_reordered_B_.data());
                            A_reord_ptr = i8_reordered_B_.data();
                        }
                    }
                    for (int c0 = 0; c0 < global_r0; c0 += kTile) {
                        const int cols = std::min(kTile, global_r0 - c0);
                        const int8_t* const B_strip = kmajor_base + static_cast<size_t>(c0) * ldK;
                        gcta_gemm_u8s8s32_kmajor_nt_beta_reordered(
                            rows, cols, curNumValidMarkers,
                            A_reord_ptr, A_strip, ldK,
                            B_strip,
                            grm_i32_base + r0 + static_cast<size_t>(c0) * grm_tile_rows,
                            grm_tile_rows, beta,
                            i8_scratch_A_f32_, i8_scratch_B_f32_, i8_scratch_C_f32_);
                    }
                }

                // Diagonal rows x rows lower triangle: sub-tile by kDiagSub = 1024
                for (int d0 = 0; d0 < rows; d0 += kDiagSub) {
                    const int db = std::min(kDiagSub, rows - d0);
                    const int8_t* const A_d = kmajor_base + static_cast<size_t>(global_r0 + d0) * ldK;
                    gcta_syrk_u8s8s32_lower_kmajor_nt_beta(
                        db, curNumValidMarkers, A_d, ldK,
                        grm_i32_diag_scratch_.data(), db, 0,
                        i8_scratch_A_f32_, i8_scratch_C_f32_);

                    const int32_t* const diag_src = grm_i32_diag_scratch_.data();
                    int32_t* const diag_dst = grm_i32_base + (r0 + d0)
                                              + static_cast<size_t>(global_r0 + d0) * grm_tile_rows;
                    #pragma omp parallel if (db >= 256)
                    {
                        const int tid = omp_get_thread_num();
                        const int nth = omp_get_num_threads();
                        const auto [c_beg, c_end] = gcta_grm_detail::lower_tri_col_range(db, tid, nth);
                        for (int c = c_beg; c < c_end; ++c) {
                            const int32_t* __restrict__ src_col = diag_src + c + static_cast<size_t>(c) * db;
                            int32_t* __restrict__ dst_col       = diag_dst + c + static_cast<size_t>(c) * grm_tile_rows;
                            const int len = db - c;
                            if (first_accum) {
                                std::memcpy(dst_col, src_col, static_cast<size_t>(len) * sizeof(int32_t));
                            } else {
                                #pragma omp simd
                                for (int r = 0; r < len; ++r) {
                                    dst_col[r] += src_col[r];
                                }
                            }
                        }
                    }

                    const int rem = rows - (d0 + db);
                    if (rem > 0) {
                        gcta_gemm_u8s8s32_kmajor_nt_beta(
                            rem, db, curNumValidMarkers,
                            kmajor_base + static_cast<size_t>(global_r0 + d0 + db) * ldK, ldK,
                            A_d,
                            grm_i32_base + (r0 + d0 + db) + static_cast<size_t>(global_r0 + d0) * grm_tile_rows,
                            grm_tile_rows, beta,
                            i8_scratch_A_f32_, i8_scratch_B_f32_, i8_scratch_C_f32_);
                    }
                }
            }

            S_total_ += S_mu2;
            double* __restrict__ R_tot = R_total_.data();
            const double* __restrict__ r_cur = r_raw_.data();
            #pragma omp parallel for schedule(static) if (decode_rows >= 4096)
            for (int i = 0; i < decode_rows; ++i) {
                R_tot[i] += r_cur[i];
            }
            grm_tile_i32_markers_ += static_cast<uint32_t>(curNumValidMarkers);
        } else {
            // Double-precision accumulation path (missing genotypes present or > 500M markers).
            double* const grm_base = std::assume_aligned<64>(grm.get());
            const int max_tile_dim = std::min(kTile, std::max(1, grm_n));
            const size_t need_tile = static_cast<size_t>(max_tile_dim) * static_cast<size_t>(max_tile_dim);
            if (grm_i32_tile_.size() < need_tile) {
                grm_i32_tile_.resize(need_tile);
            }

            const bool block_has_missing = tile_has_missing;
            const double* const r_row_vec = block_has_missing ? r_tilde_.data() : r_raw_.data();

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
                                                const int32_t* __restrict__ src_col,
                                                double* __restrict__ dst_col) {
                const double r_j_plus_S = S_mu2 - r_raw_[j];
                const double* __restrict__ r_row_slice = r_row_vec + row_start;
                if (first_accum) {
                    #pragma omp simd
                    for (int r = 0; r < len; ++r) {
                        dst_col[r] = static_cast<double>(src_col[r]) - r_row_slice[r] + r_j_plus_S;
                    }
                } else {
                    #pragma omp simd
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
                    const int8_t* __restrict__ col_m = stdGeno_i8.get()
                                          + static_cast<size_t>(miss_col_of_[m]) * stdGenoLD + row_start;
                    #pragma omp simd
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
                const int8_t* __restrict__ row_j_kmajor = kmajor_base + static_cast<size_t>(j) * ldK;
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

            // r0-outer loop: prepare_row_missing & A_strip reorder once per r0 strip.
            for (int r0 = 0; r0 < grm_tile_rows; r0 += kTile) {
                const int rows = std::min(kTile, grm_tile_rows - r0);
                const int global_r0 = grm_tile_rs + r0;
                const int8_t* const A_strip = kmajor_base + static_cast<size_t>(global_r0) * ldK;

                prepare_row_missing(global_r0, rows);

                if (global_r0 > 0) {
                    const int8_t* A_reord_ptr = nullptr;
                    if (global_r0 > kTile) {
                        const size_t rbytes = gcta_u8s8s32_reorder_bt_size(rows, curNumValidMarkers);
                        if (rbytes > 0) {
                            if (i8_reordered_B_.size() < rbytes) i8_reordered_B_.resize(rbytes);
                            gcta_u8s8s32_reorder_bt(rows, curNumValidMarkers, A_strip, ldK, i8_reordered_B_.data());
                            A_reord_ptr = i8_reordered_B_.data();
                        }
                    }
                    for (int c0 = 0; c0 < global_r0; c0 += kTile) {
                        const int cols = std::min(kTile, global_r0 - c0);
                        const int8_t* const B_strip = kmajor_base + static_cast<size_t>(c0) * ldK;
                        gcta_gemm_u8s8s32_kmajor_nt_beta_reordered(
                            rows, cols, curNumValidMarkers,
                            A_reord_ptr, A_strip, ldK,
                            B_strip,
                            grm_i32_tile_.data(), rows, 0,
                            i8_scratch_A_f32_, i8_scratch_B_f32_, i8_scratch_C_f32_);

                        #pragma omp parallel for schedule(static)
                        for (int c = 0; c < cols; ++c) {
                            double* dst_col = grm_base + r0 + static_cast<size_t>(c0 + c) * grm_tile_rows;
                            const int32_t* src_col = grm_i32_tile_.data() + static_cast<size_t>(c) * rows;
                            accum_col_with_centering(c0 + c, global_r0, rows, src_col, dst_col);
                        }
                    }
                }

                gcta_syrk_u8s8s32_lower_kmajor_nt_beta(
                    rows, curNumValidMarkers,
                    A_strip, ldK,
                    grm_i32_tile_.data(), rows, 0,
                    i8_scratch_A_f32_, i8_scratch_C_f32_);

                // Deterministic static equal-area lower-triangle accumulation.
                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    const int nth = omp_get_num_threads();
                    const auto [c_beg, c_end] = gcta_grm_detail::lower_tri_col_range(rows, tid, nth);
                    for (int c = c_beg; c < c_end; ++c) {
                        double* dst_col = grm_base + (r0 + c)
                                        + static_cast<size_t>(global_r0 + c) * grm_tile_rows;
                        const int32_t* src_col = grm_i32_tile_.data() + c + static_cast<size_t>(c) * rows;
                        const int j = global_r0 + c;
                        accum_col_with_centering(j, j, rows - c, src_col, dst_col);
                    }
                }
            }
        }
    } else {
        const float* const std_geno_base = std::assume_aligned<64>(stdGeno.get());
        double* const grm_base = std::assume_aligned<64>(grm.get());
        const bool first_accum = grm_first_accum;

        // Mixed-precision BLAS3: run cblas_ssyrk / cblas_sgemm (beta=0.0f) on L3-sized
        // 4096x4096 float32 tiles (grm_f32_tile_buf_), then widen and accumulate into
        // double* grm. Unified r0-outer loop with deterministic static equal-area diagonal.
        for (int r0 = 0; r0 < grm_tile_rows; r0 += kTile) {
            const int rows = std::min(kTile, grm_tile_rows - r0);
            const int global_r0 = grm_tile_rs + r0;
            const float* const A_strip = std_geno_base + global_r0;

            for (int c0 = 0; c0 < global_r0; c0 += kTile) {
                const int cols = std::min(kTile, global_r0 - c0);
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            rows, cols, curNumValidMarkers,
                            1.0f,
                            A_strip, stdGenoLD,
                            std_geno_base + c0, stdGenoLD,
                            0.0f,
                            grm_f32_tile_buf_.data(), rows);

                #pragma omp parallel for schedule(static)
                for (int c = 0; c < cols; ++c) {
                    double* __restrict__ dst_col = grm_base + r0 + static_cast<size_t>(c0 + c) * grm_tile_rows;
                    const float* __restrict__ src_col = grm_f32_tile_buf_.data() + static_cast<size_t>(c) * rows;
                    if (first_accum) {
                        #pragma omp simd
                        for (int r = 0; r < rows; ++r) dst_col[r] = static_cast<double>(src_col[r]);
                    } else {
                        #pragma omp simd
                        for (int r = 0; r < rows; ++r) dst_col[r] += static_cast<double>(src_col[r]);
                    }
                }
            }

            cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans,
                        rows, curNumValidMarkers,
                        1.0f,
                        A_strip, stdGenoLD,
                        0.0f,
                        grm_f32_tile_buf_.data(), rows);

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                const int nth = omp_get_num_threads();
                const auto [c_beg, c_end] = gcta_grm_detail::lower_tri_col_range(rows, tid, nth);
                for (int c = c_beg; c < c_end; ++c) {
                    double* __restrict__ dst_col = grm_base + (r0 + c)
                                                 + static_cast<size_t>(global_r0 + c) * grm_tile_rows;
                    const float* __restrict__ src_col = grm_f32_tile_buf_.data() + c + static_cast<size_t>(c) * rows;
                    const int len = rows - c;
                    if (first_accum) {
                        #pragma omp simd
                        for (int r = 0; r < len; ++r) dst_col[r] = static_cast<double>(src_col[r]);
                    } else {
                        #pragma omp simd
                        for (int r = 0; r < len; ++r) dst_col[r] += static_cast<double>(src_col[r]);
                    }
                }
            }
        }

        // Check whether any marker has missing bits within [0, decode_rows).
        if (any_missing_full) {
            const int num_words = (decode_rows + 63) / 64;
            const uintptr_t last_word_mask = (decode_rows % 64 == 0)
                ? ~static_cast<uintptr_t>(0)
                : ((static_cast<uintptr_t>(1) << (decode_rows % 64)) - 1);
            for (int m = 0; m < curNumValidMarkers && !tile_has_missing; ++m) {
                const GenoBufItem& it = gbufitems[validIndexBuf[m]];
                if (it.nValidN < full_keep_ct && !it.missing.empty()) {
                    for (int w = 0; w < num_words; ++w) {
                        uintptr_t word = it.missing[w];
                        if (w == num_words - 1) word &= last_word_mask;
                        if (word != 0) {
                            tile_has_missing = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    grm_first_accum = false;  // every later marker block for this buffer accumulates

    // Zero-missing fast path: skip bit transposes, popcounts, lazy N, and N_thread completely.
    if (!any_missing_full) {
        if (!grm_skip_global_state) {
            finished_marker += num_marker;
            numValidMarkers += static_cast<uint32_t>(curNumValidMarkers);
        }
        return;
    }

    // Missingness exists only outside [0, decode_rows) (e.g. earlier tile or --part):
    // leave N unallocated for this tile, but still populate global sub_miss on pass 0.
    if (!tile_has_missing) {
        if (!grm_skip_global_state) {
            const int num_words_all = (grm_n + 63) / 64;
            const uintptr_t last_mask_all = (grm_n % 64 == 0)
                ? ~static_cast<uintptr_t>(0)
                : ((static_cast<uintptr_t>(1) << (grm_n % 64)) - 1);
            #pragma omp parallel for schedule(static) if (num_words_all >= 64)
            for (int w = 0; w < num_words_all; ++w) {
                const int base_s = w * 64;
                for (int k = 0; k < curNumValidMarkers; ++k) {
                    const GenoBufItem& it = gbufitems[validIndexBuf[k]];
                    if (it.nValidN < full_keep_ct && !it.missing.empty()) {
                        uintptr_t word = it.missing[w];
                        if (w == num_words_all - 1) word &= last_mask_all;
                        while (word != 0) {
                            const int bit = std::countr_zero(word);
                            sub_miss[base_s + bit]++;
                            word &= (word - 1);
                        }
                    }
                }
            }
            finished_marker += num_marker;
            numValidMarkers += static_cast<uint32_t>(curNumValidMarkers);
        }
        return;
    }

    // Missing genotypes exist in [0, decode_rows): lazily allocate N and sampleMissBuf.
    ensure_N_allocated();

    const int markerPerN = static_cast<int>(sizeof(uintptr_t) * CHAR_BIT);
    const int numNblock = (curNumValidMarkers + markerPerN - 1) / markerPerN;
    const int active_samples = (!grm_skip_global_state) ? grm_n : decode_rows;
    const int numNSampleBlock = (active_samples + markerPerN - 1) / markerPerN;
    const size_t blockStride = static_cast<size_t>(numNSampleBlock) * static_cast<size_t>(markerPerN);
    const size_t need_miss_words = blockStride * static_cast<size_t>(numNblock);
    if (sampleMissBuf.size() < need_miss_words) {
        sampleMissBuf.resize(need_miss_words);
    }

    // Each sample block j owns disjoint 64-element slices in both sampleMissBuf
    // and sub_miss, so transposing and accumulating sub_miss across j is lock-free.
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
                uintptr_t *block_buf = sampleMissBuf.data() + static_cast<size_t>(i) * blockStride;
                if (markerPerN == 64) {
                    for(int k = baseMarkerIndex; k < lastValidIndex; k++){
                        const GenoBufItem &it = gbufitems[validIndexBuf[k]];
                        transposeInput[static_cast<std::size_t>(k - baseMarkerIndex)] =
                            (it.nValidN < full_keep_ct && !it.missing.empty()) ? it.missing[j] : 0UL;
                    }
                    for(int k = lastValidIndex; k < markerPerN * (i + 1); k++){
                        transposeInput[static_cast<std::size_t>(k - baseMarkerIndex)] = 0UL;
                    }
                    plink2::TransposeBitblock(
                        transposeInput.data(), 1, 1,
                        static_cast<uint32_t>(markerPerN), static_cast<uint32_t>(markerPerN),
                        &block_buf[baseMissIndex], transposeScratch);
                } else {
                    for(int k = baseMarkerIndex; k < lastValidIndex; k++){
                        const GenoBufItem &it = gbufitems[validIndexBuf[k]];
                        block_buf[baseMissIndex + k - baseMarkerIndex] =
                            (it.nValidN < full_keep_ct && !it.missing.empty()) ? it.missing[j] : 0UL;
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
        for(int index = 0; index < static_cast<int>(index_grm_pairs.size()); index++){
            const auto index_pair = index_grm_pairs[index];
            for(int i = 0; i < numNblock; i++){
                N_thread(index_pair.first, index_pair.second,
                         sampleMissBuf.data() + static_cast<size_t>(i) * blockStride);
            }
        }
    }

    if(!grm_skip_global_state){
        finished_marker += num_marker;
        numValidMarkers += static_cast<uint32_t>(curNumValidMarkers);
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

    const std::string grm_path = isSparse ? (o_name + ".grm.sp") : (o_name + ".grm.bin");
    const std::string N_path   = o_name + ".grm.N.bin";
    UniqueFile sparse_out;
    gcta_grm_detail::GrmUniqueFd grm_fd;
    gcta_grm_detail::GrmUniqueFd N_fd;
    if(isSparse){
        sparse_out = open_file(grm_path, "wb");
        if(!sparse_out){
            LOGGER.error(0, "can't open {} to write", grm_path);
        }
        setvbuf(sparse_out.get(), nullptr, _IOFBF, 4 << 20);
    }else{
        grm_fd = gcta_grm_detail::open_posix_write_trunc(grm_path);
        N_fd   = gcta_grm_detail::open_posix_write_trunc(N_path);
        if(!grm_fd || !N_fd){
            LOGGER.error(0, "can't open {}.grm.bin or .grm.N.bin to write", o_name);
        }
    }

    double mtd_weight = 1.0;
    if(isMtd){
        double weight = 0.0;
        if(!isDominance){
            for(uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i];
        }else{
            for(uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i] * sd[i];
        }
        if(weight > 0.0) mtd_weight = static_cast<double>(numValidMarkers) / weight;
    }

    // Set tile bounds to cover the entire [part_keep_indices.first, part_keep_indices.second + 1)
    // pass and delegate to flush_grm_tile (grm_tiling_enabled == false selects packed N[] indexing).
    grm_tile_rs   = static_cast<int>(part_keep_indices.first);
    grm_tile_re   = static_cast<int>(part_keep_indices.second) + 1;
    grm_tile_rows = grm_m;
    grm_tile_cols = grm_n;

    AlignedRawBuffer<float, 64> packed_grm;
    AlignedRawBuffer<float, 64> packed_N;
    std::vector<float> const_N_buf;
    std::string sparse_buf;
    flush_grm_tile(grm_fd.get(), N_fd.get(), sparse_out.get(), grm_path, N_path,
                   thresh, isSparse, mtd_weight,
                   packed_grm, packed_N, const_N_buf, sparse_buf);

    grm_fd.reset();
    N_fd.reset();
    sparse_out.reset();
    if(!isSparse){
        LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.bin]");
        LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + o_name + ".grm.N.bin]");
    }else{
        LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.sp]");
    }
}


// Write one tile's (or full non-tiled pass's) normalised lower-triangular rows to already-open
// output files using a cache-blocked (64x64 / 32x32) packed-row transpose and parallel POSIX I/O.
//   - Zero-missing fast path (N == nullptr): fuses scaling (and -R[i]-R[j]+S in int32_t mode)
//     into the 64x64 transpose and writes .grm.N.bin from a pre-filled 16 MiB constant buffer.
//   - Missing-data path (N != nullptr): reads N exactly once in parallel inside the 32x32 block
//     transpose to emit both packed_grm and packed_N, then writes both files in parallel.
void GRM::flush_grm_tile(int grm_fd, int N_fd, FILE *sparse_out,
                         const std::string& grm_path, const std::string& N_path,
                         float thresh, bool isSparse, double mtd_weight,
                         AlignedRawBuffer<float, 64>& packed_grm,
                         AlignedRawBuffer<float, 64>& packed_N,
                         std::vector<float>& const_N_buf,
                         std::string& sparse_buf){
    const int tile_rs   = grm_tile_rs;
    const int tile_re   = grm_tile_re;
    const int tile_rows = grm_tile_rows;
    const int tile_cols = grm_tile_cols;   // == tile_re

    constexpr size_t kConstNFloats = (16ULL << 20) / sizeof(float); // 16 MiB constant buffer

    // Edge case: 0 valid markers accumulated (all-missing / all-filtered run).
    if (grm_first_accum || numValidMarkers == 0) {
        if (!isSparse) {
            const uint64_t dre = static_cast<uint64_t>(tile_re);
            const uint64_t drs = static_cast<uint64_t>(tile_rs);
            const size_t total_elems = static_cast<size_t>((dre * (dre + 1ULL) - drs * (drs + 1ULL)) / 2ULL);
            const size_t total_bytes = total_elems * sizeof(float);
            const size_t buf_floats = std::min(total_elems, kConstNFloats);
            if (const_N_buf.size() < buf_floats) const_N_buf.resize(buf_floats);
            std::fill(const_N_buf.begin(), const_N_buf.begin() + static_cast<std::ptrdiff_t>(buf_floats), 0.0f);
            const size_t chunk_cap = buf_floats * sizeof(float);
            bool ok_grm = true, ok_N = true;
            int err_grm = 0, err_N = 0;
            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    size_t rem = total_bytes;
                    while (rem > 0) {
                        const size_t c = std::min(rem, chunk_cap);
                        if (!gcta_grm_detail::write_exact_posix(grm_fd, const_N_buf.data(), c, err_grm)) {
                            ok_grm = false;
                            break;
                        }
                        rem -= c;
                    }
                }
                #pragma omp section
                {
                    size_t rem = total_bytes;
                    while (rem > 0) {
                        const size_t c = std::min(rem, chunk_cap);
                        if (!gcta_grm_detail::write_exact_posix(N_fd, const_N_buf.data(), c, err_N)) {
                            ok_N = false;
                            break;
                        }
                        rem -= c;
                    }
                }
            }
            if (!ok_grm) LOGGER.e(0, "write() failed on [" + grm_path + "]: " + std::string(std::strerror(err_grm)));
            if (!ok_N)   LOGGER.e(0, "write() failed on [" + N_path   + "]: " + std::string(std::strerror(err_N)));
        } else if (0.0f >= thresh) {
            for (int pair1 = tile_rs; pair1 < tile_re; ++pair1) {
                sparse_buf.clear();
                for (int i = 0; i <= pair1; ++i) {
                    std::format_to(std::back_inserter(sparse_buf), "{}\t{}\t0\n", pair1, i);
                }
                if (!sparse_buf.empty() && fwrite(sparse_buf.data(), 1, sparse_buf.size(), sparse_out) != sparse_buf.size()) {
                    LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                }
            }
        }
        return;
    }

    const bool zero_missing = (N == nullptr);
    const double inv_scale = mtd_weight / static_cast<double>(numValidMarkers);

    // Pre-fill constant 16 MiB float(numValidMarkers) buffer for zero-missing .grm.N.bin output.
    if (!isSparse && zero_missing) {
        const float n_val = static_cast<float>(numValidMarkers);
        if (const_N_buf.size() != kConstNFloats || const_N_buf[0] != n_val) {
            const_N_buf.assign(kConstNFloats, n_val);
        }
    }

    // Bound packed_grm strip buffer to <= ~256 MiB (64M floats) even at N=1,000,000.
    constexpr size_t kTargetStripFloats = 64ULL << 20;
    const size_t raw_step = kTargetStripFloats / std::max<size_t>(1, static_cast<size_t>(tile_cols));
    const int strip_rows = static_cast<int>(std::clamp<size_t>((raw_step / 64ULL) * 64ULL, 64ULL, 1024ULL));

    const double* const grm_base       = grm ? std::assume_aligned<64>(grm.get()) : nullptr;
    const int32_t* const grm_i32_base  = grm_i32 ? std::assume_aligned<64>(grm_i32.get()) : nullptr;
    const uint32_t* const N_base       = N ? std::assume_aligned<64>(N.get()) : nullptr;
    const uint32_t* const sub_miss_ptr = sub_miss.data();
    const double* const R_ptr          = grm_is_i32_ ? R_total_.data() : nullptr;
    const double S_tot                 = S_total_;

    const auto get_N_row = [&](int pair1) -> const uint32_t* {
        if (grm_tiling_enabled) {
            return N_base + static_cast<size_t>(pair1 - tile_rs) * tile_cols;
        }
        const size_t start_pos = static_cast<size_t>(pair1 + 1 + tile_rs) *
                                 static_cast<size_t>(pair1 - tile_rs) / 2;
        return N_base + start_pos;
    };

    std::vector<size_t> row_off(static_cast<size_t>(strip_rows) + 1);

    for (int block_start = tile_rs; block_start < tile_re; block_start += strip_rows) {
        const int block_end         = std::min(block_start + strip_rows, tile_re);
        const int block_rows        = block_end - block_start;
        const int local_block_start = block_start - tile_rs;

        size_t total_packed = 0;
        for (int r = 0; r < block_rows; ++r) {
            row_off[static_cast<size_t>(r)] = total_packed;
            total_packed += static_cast<size_t>(block_start + r + 1);
        }
        row_off[static_cast<size_t>(block_rows)] = total_packed;

        if (packed_grm.size() < total_packed) {
            packed_grm.resize(total_packed);
        }
        if (!zero_missing && !isSparse && packed_N.size() < total_packed) {
            packed_N.resize(total_packed);
        }

        float* const out_grm = packed_grm.data();
        float* const out_N   = (!zero_missing && !isSparse) ? packed_N.data() : nullptr;

        if (zero_missing && grm_is_i32_) {
            // 64x64 L1-resident cache-blocked transpose: fuses centering (-R[i]-R[j]+S) and * inv_scale.
            constexpr int kBlk = 64;
            const int n_cblk = (block_end + kBlk - 1) / kBlk;
            #pragma omp parallel for schedule(static)
            for (int cb = 0; cb < n_cblk; ++cb) {
                const int c0 = cb * kBlk;
                const int c_len = std::min(kBlk, block_end - c0);
                alignas(64) float tmp[kBlk][kBlk];
                for (int r0 = 0; r0 < block_rows; r0 += kBlk) {
                    const int r_len = std::min(kBlk, block_rows - r0);
                    const int global_r0 = block_start + r0;
                    if (c0 >= global_r0 + r_len) continue;
                    const int local_r0 = local_block_start + r0;
                    const double* __restrict__ R_row = R_ptr + global_r0;
                    for (int dc = 0; dc < c_len; ++dc) {
                        const int c = c0 + dc;
                        const int dr_start = std::max(0, c - global_r0);
                        if (dr_start >= r_len) break;
                        const int32_t* __restrict__ col_src =
                            grm_i32_base + static_cast<size_t>(c) * tile_rows + local_r0;
                        const double r_j_plus_S = S_tot - R_ptr[c];
                        for (int dr = dr_start; dr < r_len; ++dr) {
                            tmp[dr][dc] = static_cast<float>(
                                (static_cast<double>(col_src[dr]) - R_row[dr] + r_j_plus_S) * inv_scale);
                        }
                    }
                    for (int dr = 0; dr < r_len; ++dr) {
                        const int pair1 = global_r0 + dr;
                        const int valid_c = std::min(c_len, pair1 + 1 - c0);
                        if (valid_c <= 0) continue;
                        std::memcpy(out_grm + row_off[static_cast<size_t>(r0 + dr)] + c0,
                                    tmp[dr], static_cast<size_t>(valid_c) * sizeof(float));
                    }
                }
            }
        } else if (zero_missing) {
            // 64x64 L1-resident cache-blocked transpose for zero-missing double* grm: fuses * inv_scale.
            constexpr int kBlk = 64;
            const int n_cblk = (block_end + kBlk - 1) / kBlk;
            #pragma omp parallel for schedule(static)
            for (int cb = 0; cb < n_cblk; ++cb) {
                const int c0 = cb * kBlk;
                const int c_len = std::min(kBlk, block_end - c0);
                alignas(64) float tmp[kBlk][kBlk];
                for (int r0 = 0; r0 < block_rows; r0 += kBlk) {
                    const int r_len = std::min(kBlk, block_rows - r0);
                    const int global_r0 = block_start + r0;
                    if (c0 >= global_r0 + r_len) continue;
                    const int local_r0 = local_block_start + r0;
                    for (int dc = 0; dc < c_len; ++dc) {
                        const int c = c0 + dc;
                        const int dr_start = std::max(0, c - global_r0);
                        if (dr_start >= r_len) break;
                        const double* __restrict__ col_src =
                            grm_base + static_cast<size_t>(c) * tile_rows + local_r0;
                        for (int dr = dr_start; dr < r_len; ++dr) {
                            tmp[dr][dc] = static_cast<float>(col_src[dr] * inv_scale);
                        }
                    }
                    for (int dr = 0; dr < r_len; ++dr) {
                        const int pair1 = global_r0 + dr;
                        const int valid_c = std::min(c_len, pair1 + 1 - c0);
                        if (valid_c <= 0) continue;
                        std::memcpy(out_grm + row_off[static_cast<size_t>(r0 + dr)] + c0,
                                    tmp[dr], static_cast<size_t>(valid_c) * sizeof(float));
                    }
                }
            }
        } else {
            // 32x32 L1-resident cache-blocked transpose for missing-data path:
            // reads N once in parallel to produce both packed_grm and packed_N.
            constexpr int kBlk = 32;
            const int n_cblk = (block_end + kBlk - 1) / kBlk;
            const uint32_t valid_m = numValidMarkers;
            #pragma omp parallel for schedule(static)
            for (int cb = 0; cb < n_cblk; ++cb) {
                const int c0 = cb * kBlk;
                const int c_len = std::min(kBlk, block_end - c0);
                alignas(64) double tmp_d[kBlk][kBlk];
                for (int r0 = 0; r0 < block_rows; r0 += kBlk) {
                    const int r_len = std::min(kBlk, block_rows - r0);
                    const int global_r0 = block_start + r0;
                    if (c0 >= global_r0 + r_len) continue;
                    const int local_r0 = local_block_start + r0;
                    for (int dc = 0; dc < c_len; ++dc) {
                        const int c = c0 + dc;
                        const int dr_start = std::max(0, c - global_r0);
                        if (dr_start >= r_len) break;
                        const double* __restrict__ col_src =
                            grm_base + static_cast<size_t>(c) * tile_rows + local_r0;
                        for (int dr = dr_start; dr < r_len; ++dr) {
                            tmp_d[dr][dc] = col_src[dr];
                        }
                    }
                    for (int dr = 0; dr < r_len; ++dr) {
                        const int pair1 = global_r0 + dr;
                        const int valid_c = std::min(c_len, pair1 + 1 - c0);
                        if (valid_c <= 0) continue;
                        const uint32_t sub_miss1 = valid_m - sub_miss_ptr[pair1];
                        const uint32_t* __restrict__ N_row = get_N_row(pair1) + c0;
                        const uint32_t* __restrict__ sm_c  = sub_miss_ptr + c0;
                        const size_t dst_off = row_off[static_cast<size_t>(r0 + dr)] + static_cast<size_t>(c0);
                        float* __restrict__ g_dst = out_grm + dst_off;
                        const double* __restrict__ td_row = tmp_d[dr];
                        if (out_N) {
                            float* __restrict__ n_dst = out_N + dst_off;
                            for (int dc = 0; dc < valid_c; ++dc) {
                                const uint32_t sub_N = N_row[dc] + sub_miss1 - sm_c[dc];
                                n_dst[dc] = static_cast<float>(sub_N);
                                g_dst[dc] = sub_N
                                    ? static_cast<float>((td_row[dc] / static_cast<double>(sub_N)) * mtd_weight)
                                    : 0.0f;
                            }
                        } else {
                            for (int dc = 0; dc < valid_c; ++dc) {
                                const uint32_t sub_N = N_row[dc] + sub_miss1 - sm_c[dc];
                                g_dst[dc] = sub_N
                                    ? static_cast<float>((td_row[dc] / static_cast<double>(sub_N)) * mtd_weight)
                                    : 0.0f;
                            }
                        }
                    }
                }
            }
        }

        if (!isSparse) {
            const size_t strip_bytes = total_packed * sizeof(float);
            const size_t const_N_bytes = const_N_buf.size() * sizeof(float);
            bool ok_grm = true, ok_N = true;
            int err_grm = 0, err_N = 0;
            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                {
                    ok_grm = gcta_grm_detail::write_exact_posix(grm_fd, out_grm, strip_bytes, err_grm);
                }
                #pragma omp section
                {
                    if (zero_missing) {
                        size_t rem = strip_bytes;
                        while (rem > 0) {
                            const size_t chunk = std::min(rem, const_N_bytes);
                            if (!gcta_grm_detail::write_exact_posix(N_fd, const_N_buf.data(), chunk, err_N)) {
                                ok_N = false;
                                break;
                            }
                            rem -= chunk;
                        }
                    } else {
                        ok_N = gcta_grm_detail::write_exact_posix(N_fd, out_N, strip_bytes, err_N);
                    }
                }
            }
            if (!ok_grm) {
                LOGGER.e(0, "write() failed on [" + grm_path + "]: " + std::string(std::strerror(err_grm)));
            }
            if (!ok_N) {
                LOGGER.e(0, "write() failed on [" + N_path + "]: " + std::string(std::strerror(err_N)));
            }
        } else {
            const int n_thr = (block_rows >= 64) ? omp_get_max_threads() : 1;
            if (n_thr > 1) {
                std::vector<std::string> thr_bufs(static_cast<size_t>(n_thr));
                #pragma omp parallel num_threads(n_thr)
                {
                    const int tid = omp_get_thread_num();
                    const int r_beg = (block_rows * tid) / n_thr;
                    const int r_end = (block_rows * (tid + 1)) / n_thr;
                    std::string& local_buf = thr_bufs[static_cast<size_t>(tid)];
                    for (int local_r = r_beg; local_r < r_end; ++local_r) {
                        const int pair1 = block_start + local_r;
                        const float* row = out_grm + row_off[static_cast<size_t>(local_r)];
                        for (int i = 0; i <= pair1; ++i) {
                            if (row[i] >= thresh) {
                                std::format_to(std::back_inserter(local_buf),
                                    "{}\t{}\t{:.{}g}\n",
                                    pair1, i, row[i],
                                    std::numeric_limits<float>::digits10 + 2);
                            }
                        }
                    }
                }
                for (const auto& tb : thr_bufs) {
                    if (!tb.empty() && fwrite(tb.data(), 1, tb.size(), sparse_out) != tb.size()) {
                        LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                    }
                }
            } else {
                sparse_buf.clear();
                for (int local_r = 0; local_r < block_rows; ++local_r) {
                    const int pair1 = block_start + local_r;
                    const float* row = out_grm + row_off[static_cast<size_t>(local_r)];
                    for (int i = 0; i <= pair1; ++i) {
                        if (row[i] >= thresh) {
                            std::format_to(std::back_inserter(sparse_buf),
                                "{}\t{}\t{:.{}g}\n",
                                pair1, i, row[i],
                                std::numeric_limits<float>::digits10 + 2);
                        }
                    }
                }
                if (!sparse_buf.empty() && fwrite(sparse_buf.data(), 1, sparse_buf.size(), sparse_out) != sparse_buf.size()) {
                    LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                }
            }
        }
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
    use_int8_grm = bBLAS && isMtd && !isDominance && !is_homogametic
                   && (geno->getGenoFormat() == "BED")
                   && (std::getenv("GCTA_DISABLE_INT8_GRM") == nullptr);
    if (!n_markers_explicit_ && !use_int8_grm && nMarkerBlock > 1024) {
        nMarkerBlock = 1024;
    }

    const vector<uint32_t> processIndex = is_homogametic
        ? marker->get_extract_index_X()
        : marker->get_extract_index_autosome();

    gbufitems.assign(static_cast<size_t>(nMarkerBlock), GenoBufItem{});
    validIndexBuf.reserve(static_cast<size_t>(nMarkerBlock));
    sampleMissBuf.reset(); // lazily allocated only when a block has missing calls

    constexpr int kTile = 4096;
    const int max_tile_dim = std::min(kTile, std::max(1, grm_n));
    const size_t geno_block_elems = static_cast<size_t>(nMarkerBlock) * static_cast<size_t>(stdGenoLD);

    if (use_int8_grm) {
        grm_bytes_std_geno = sizeof(int8_t) * static_cast<size_t>(grm_n);
        // Marker-major int8 columns are only materialised lazily for markers with missing calls.
        stdGeno_i8.reset();
        stdGeno_i8_cols_ = 0;
        const int eff_block_cap = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(nMarkerBlock), std::max<size_t>(1, processIndex.size())));
        const int max_ldK = std::max(64, (eff_block_cap + 63) / 64 * 64);
        const size_t kmajor_elems = static_cast<size_t>(max_ldK) * static_cast<size_t>(grm_n);
        this->num_byte_geno = sizeof(int8_t) * static_cast<uint64_t>(kmajor_elems);
        stdGeno_i8_kmajor_ = make_aligned_uninitialized<int8_t, 64>(kmajor_elems);
        if (!stdGeno_i8_kmajor_) {
            LOGGER.e(0, "can't allocate enough memory for the transposed int8 genotype buffer.");
        }
        grm_i32_tile_.reset(); // lazily allocated only if promoted to double* grm
        mu_vec_.resize(static_cast<size_t>(nMarkerBlock));
        r_raw_.resize(static_cast<size_t>(grm_n));
        r_tilde_.resize(static_cast<size_t>(grm_n));
        marker_miss_offset_.resize(static_cast<size_t>(nMarkerBlock) + 1);
        sample_miss_offset_.resize(static_cast<size_t>(grm_n) + 1);
    } else {
        grm_bytes_std_geno = sizeof(float) * static_cast<size_t>(grm_n);
        this->num_byte_geno = sizeof(float) * static_cast<uint64_t>(geno_block_elems);
        stdGeno = make_aligned_uninitialized<float, 64>(geno_block_elems);
        if (!stdGeno) {
            LOGGER.e(0, "can't allocate enough memory for the genotype buffer.");
        }
        grm_f32_tile_buf_.resize(static_cast<size_t>(max_tile_dim) * static_cast<size_t>(max_tile_dim));
    }

    vector<function<void (uintptr_t *, std::span<const uint32_t>)>> callBacks;
    callBacks.push_back([this](uintptr_t *buf, std::span<const uint32_t> idx){
        calculate_GRM_blas(buf, idx);
    });
    geno->setGRMMode(true, isDominance);
    const bool isSTD = !isMtd;
    sd.clear();
    sd.reserve(processIndex.size());
    sub_miss.assign(index_keep.size() + 64, 0);
    numValidMarkers = 0;
    finished_marker = 0;
    grm_any_missing_seen_ = false;
    grm_skip_global_state = false;

    const bool allow_i32_mode = use_int8_grm && (processIndex.size() <= kMaxInt32Markers);
    // Zero-missing lazy-N accounting: 4 B/elem for VanRaden INT8 (int32_t GRM, no N),
    // 8 B/elem for FP32/double GRM (no N). If missingness appears mid-run, remaining
    // tiles dynamically adapt to 12 B/elem (double GRM + uint32_t N).
    const double initial_bpe = allow_i32_mode ? 4.0 : 8.0;
    const int full_rs = static_cast<int>(part_keep_indices.first);
    const int full_re = static_cast<int>(part_keep_indices.second) + 1;

    grm_tiling_enabled = grm_tile_budget_bytes > 0.0;
    if (grm_tiling_enabled) {
        const double budget_elems = grm_tile_budget_bytes / initial_bpe;
        if (gcta_grm_detail::grm_tile_budget_covers_whole(full_rs, full_re, budget_elems)) {
            LOGGER.info(0, "--GRM-tile-budget ({:.3f} GB) covers the whole GRM in one tile ({:.0f} B/elem); using the dense path instead.",
                        grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0), initial_bpe);
            grm_tiling_enabled = false;
        }
    }

    if (bBLAS && grm_tiling_enabled) {
        // ── Block-tiled GRM: cap the active tile buffer to --GRM-tile-budget bytes ──
        float thresh = -99.0f;
        bool isSparse = false;
        if (sparse_cutoff.has_value()) {
            thresh   = *sparse_cutoff;
            isSparse = true;
            LOGGER.i(0, "Computing sparse GRM with a cutoff " + to_string(thresh) + "...");
        } else {
            LOGGER.i(0, "Computing tiled GRM...");
        }

        const std::string grm_path = isSparse ? (o_name + ".grm.sp") : (o_name + ".grm.bin");
        const std::string N_path   = o_name + ".grm.N.bin";
        UniqueFile sparse_out;
        gcta_grm_detail::GrmUniqueFd grm_fd;
        gcta_grm_detail::GrmUniqueFd N_fd;
        if (isSparse) {
            sparse_out = open_file(grm_path, "wb");
            if (!sparse_out) LOGGER.error(0, "can't open {} to write", grm_path);
            setvbuf(sparse_out.get(), nullptr, _IOFBF, 4 << 20);
        } else {
            grm_fd = gcta_grm_detail::open_posix_write_trunc(grm_path);
            N_fd   = gcta_grm_detail::open_posix_write_trunc(N_path);
            if (!grm_fd || !N_fd)
                LOGGER.error(0, "can't open {}.grm.bin or .grm.N.bin to write", o_name);
        }

        const int num_thread = omp_get_max_threads();

        // Pre-validate upfront under initial_bpe so an impossibly tiny budget fails before I/O.
        {
            const double budget_elems = grm_tile_budget_bytes / initial_bpe;
            vector<std::pair<int, int>> est_ranges;
            int sim_rs = full_rs;
            while (sim_rs < full_re) {
                const int sim_re = gcta_grm_detail::solve_tile_budget_end(sim_rs, full_re, budget_elems);
                if (sim_re <= sim_rs) {
                    const uint64_t need_bytes = static_cast<uint64_t>(sim_rs + 1) * static_cast<uint64_t>(initial_bpe);
                    LOGGER.error(0, "--GRM-tile-budget is too small: the next row (sample offset {}) alone needs {:.3f} GB ({:.0f} B/elem), but the budget is only {:.3f} GB. Increase --GRM-tile-budget.",
                                 sim_rs,
                                 need_bytes / (1024.0 * 1024.0 * 1024.0),
                                 initial_bpe,
                                 grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0));
                }
                est_ranges.push_back({sim_rs, sim_re});
                sim_rs = sim_re;
            }
            LOGGER.info(0, "Block-tiled GRM (memory budget {:.3f} GB, initial {:.0f} B/elem): {} tile(s), rows/tile ranging {}-{}.",
                        grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0),
                        initial_bpe,
                        est_ranges.size(),
                        est_ranges.front().second - est_ranges.front().first,
                        est_ranges.back().second - est_ranges.back().first);
        }

        AlignedRawBuffer<float, 64> flush_packed_grm;
        AlignedRawBuffer<float, 64> flush_packed_N;
        std::vector<float> flush_const_N;
        std::string flush_sparse_buf;

        double mtd_weight = 1.0;        // computed once, after the first tile's decode
        bool logged_missing_rebudget = false;
        int rs = full_rs;

        while (rs < full_re) {
            const double cur_bpe = grm_any_missing_seen_ ? 12.0 : initial_bpe;
            if (grm_any_missing_seen_ && !logged_missing_rebudget && rs > full_rs) {
                logged_missing_rebudget = true;
                LOGGER.info(0, "  Missing genotypes detected; adapting remaining tile budget to 12 B/elem (double GRM + N).");
            }
            const int re = gcta_grm_detail::solve_tile_budget_end(rs, full_re, grm_tile_budget_bytes / cur_bpe);
            if (re <= rs) {
                const uint64_t need_bytes = static_cast<uint64_t>(rs + 1) * static_cast<uint64_t>(cur_bpe);
                LOGGER.error(0, "--GRM-tile-budget is too small: the next row (sample offset {}) alone needs {:.3f} GB ({:.0f} B/elem), but the budget is only {:.3f} GB. Increase --GRM-tile-budget.",
                             rs,
                             need_bytes / (1024.0 * 1024.0 * 1024.0),
                             cur_bpe,
                             grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0));
            }

            grm_tile_rs   = rs;
            grm_tile_re   = re;
            grm_tile_rows = re - rs;
            grm_tile_cols = re;   // lower-triangle: widest row needs cols [0, re-1]

            bool known_tile_missing = false;
            if (grm_skip_global_state && grm_any_missing_seen_) {
                for (int s = 0; s < re; ++s) {
                    if (sub_miss[static_cast<size_t>(s)] > 0) {
                        known_tile_missing = true;
                        break;
                    }
                }
            }
            const bool start_i32 = allow_i32_mode && !known_tile_missing;

            const size_t tile_rect_elems = static_cast<size_t>(grm_tile_rows) * static_cast<size_t>(grm_tile_cols);
            grm.reset();
            grm_i32.reset();
            N.reset(); // always lazy!

            if (start_i32) {
                grm_i32 = make_aligned_uninitialized<int32_t, 64>(tile_rect_elems);
                if (!grm_i32) LOGGER.e(0, "Can't allocate int32 GRM tile buffer.");
                grm_is_i32_ = true;
                grm_tile_i32_markers_ = 0;
                R_total_.assign(static_cast<size_t>(grm_tile_cols), 0.0);
                S_total_ = 0.0;
            } else {
                grm = make_aligned_uninitialized<double, 64>(tile_rect_elems);
                if (!grm) LOGGER.e(0, "Can't allocate GRM tile buffer.");
                grm_is_i32_ = false;
            }

            const double drs = rs, dre = re;
            const double tile_elems = (dre * (dre + 1.0) - drs * (drs + 1.0)) / 2.0;
            const double est_bpe = start_i32 ? 4.0 : (known_tile_missing ? 12.0 : 8.0);
            const double tile_gb = tile_elems * est_bpe / (1024.0 * 1024.0 * 1024.0);
            LOGGER.info(0, "  Tile rows {}-{} ({:.2f} GB {})", rs, re - 1, tile_gb,
                        start_i32 ? "int32 GRM" : (known_tile_missing ? "double GRM+N" : "double GRM"));

            grm_first_accum = true;

            index_grm_pairs.clear();
            vector<uint32_t> tile_parts = divide_parts(static_cast<uint32_t>(rs), static_cast<uint32_t>(re - 1), static_cast<uint32_t>(num_thread));
            index_grm_pairs.push_back(std::make_pair(rs, static_cast<int>(tile_parts[0])));
            for (size_t i = 1; i < tile_parts.size(); ++i)
                index_grm_pairs.push_back(std::make_pair(static_cast<int>(tile_parts[i-1] + 1), static_cast<int>(tile_parts[i])));

            const bool is_first_tile = !grm_skip_global_state;
            geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);

            grm_skip_global_state = true;

            if (is_first_tile && isMtd) {
                double weight = 0.0;
                if (!isDominance)
                    for (uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i];
                else
                    for (uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i] * sd[i];
                if (weight > 0.0) mtd_weight = static_cast<double>(numValidMarkers) / weight;
            }

            flush_grm_tile(grm_fd.get(), N_fd.get(), sparse_out.get(), grm_path, N_path,
                           thresh, isSparse, mtd_weight,
                           flush_packed_grm, flush_packed_N, flush_const_N, flush_sparse_buf);

            grm.reset();
            grm_i32.reset();
            N.reset();
            rs = re;
        }

        grm_fd.reset();
        N_fd.reset();
        sparse_out.reset();
        LOGGER << "  Used " << numValidMarkers << " valid SNPs." << std::endl;
        if (!isSparse) {
            LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.bin]");
            LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + o_name + ".grm.N.bin]");
        } else {
            LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.sp]");
        }
    } else {
        // Non-tiled dense path (!grm_tiling_enabled): single pass.
        grm_tile_rs   = full_rs;
        grm_tile_re   = full_re;
        grm_tile_rows = grm_m;
        grm_tile_cols = grm_n;

        const size_t dense_rect_elems = static_cast<size_t>(grm_m) * static_cast<size_t>(grm_n);
        grm.reset();
        grm_i32.reset();
        N.reset(); // always lazy!

        if (allow_i32_mode) {
            grm_i32 = make_aligned_uninitialized<int32_t, 64>(dense_rect_elems);
            if (!grm_i32) {
                LOGGER.error(0, "can't allocate enough memory to store the (parted) int32 GRM: {:.2f}GB required.",
                             dense_rect_elems * sizeof(int32_t) / (1024.0 * 1024.0 * 1024.0));
            }
            grm_is_i32_ = true;
            grm_tile_i32_markers_ = 0;
            R_total_.assign(static_cast<size_t>(grm_n), 0.0);
            S_total_ = 0.0;
        } else {
            grm = make_aligned_uninitialized<double, 64>(dense_rect_elems);
            if (!grm) {
                LOGGER.error(0, "can't allocate enough memory to store the (parted) GRM: {:.2f}GB required.",
                             dense_rect_elems * sizeof(double) / (1024.0 * 1024.0 * 1024.0));
            }
            grm_is_i32_ = false;
        }
        grm_first_accum = true;

        LOGGER << "Computing dense GRM..." << std::endl;
        geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);
        LOGGER << "  Used " << numValidMarkers << " valid SNPs." << std::endl;
        deduce_GRM();
        grm.reset();
        grm_i32.reset();
        N.reset();
    }

    gbufitems.clear();
    stdGeno.reset();
    stdGeno_i8.reset();
    stdGeno_i8_cols_ = 0;
    stdGeno_i8_kmajor_.reset();
    sampleMissBuf.reset();
    grm_f32_tile_buf_.reset();
    grm_i32_tile_.reset();
    grm_i32_diag_scratch_.reset();
    i8_reordered_B_.reset();
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