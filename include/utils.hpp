/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   Some utils in header files

   Singleton pattern to use in the whole program.

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include <bit>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <tuple>
#include <ranges>
#include <sstream>
#include <cstdint>
#include <limits>

struct FileCloser {
    void operator()(FILE* f) const noexcept {
        if (f) std::fclose(f);
    }
};
using UniqueFile = std::unique_ptr<FILE, FileCloser>;

[[nodiscard]] inline UniqueFile open_file(const std::string& path, const char* mode) noexcept {
    return UniqueFile{std::fopen(path.c_str(), mode)};
}

enum class UtilsError {
    EmptyInput,
    NullFileHandle,
    FileSeekFailed,
    FileTellFailed
};

std::string_view getUtilsErrorMessage(UtilsError error);
std::string getHostName();
std::string getLocalTime();
[[nodiscard]] std::expected<std::string, UtilsError> getFileName(std::string_view path);
[[nodiscard]] std::expected<std::string, UtilsError> getPathName(std::string_view path);
[[nodiscard]] std::expected<std::string, UtilsError> joinPath(std::string_view dir, std::string_view path);
std::string getSSEvar();
std::string getOSName();
[[nodiscard]] std::expected<uint64_t, UtilsError> getFileByteSize(FILE * file);

template <typename T>
bool hasVectorDuplicate(const std::vector<T> &v){
    if(v.size() < 2){
        return false;
    }
    std::vector<T> t = v;
    std::ranges::sort(t);
    return std::adjacent_find(t.begin(), t.end()) != t.end();
}

template <typename T>
void removeDuplicateSort(std::vector<T> &t){
    std::ranges::sort(t);
    t.erase(std::unique(t.begin(), t.end()), t.end());
}


template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    std::vector<size_t> index(v.size());
    std::iota(index.begin(), index.end(), 0);

    std::ranges::sort(index, [&v](size_t item1, size_t item2) {
        return v[item1] < v[item2];
    });

    return index;
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v1, const std::vector<T> &v2) {
    if(v1.size() != v2.size()){
        return std::vector<size_t>();
    }

    std::vector<size_t> index(v1.size());
    std::iota(index.begin(), index.end(), 0);

    std::ranges::sort(index, [&v1, &v2](size_t item1, size_t item2) {
        return std::tie(v1[item1], v2[item1]) < std::tie(v1[item2], v2[item2]);
    });

    return index;
}

template <typename T, typename P>
void vector_commonIndex(const std::vector<T>& v1, const std::vector<T>& v2, std::vector<P>& k1, std::vector<P>& k2){
    if(v1.empty() || v2.empty()){
        k1.resize(0);
        k2.resize(0);
        return;
    }

    if(v1 == v2){
        k1.resize(v1.size());
        std::iota(k1.begin(), k1.end(), static_cast<P>(0));

        k2.resize(v2.size());
        std::iota(k2.begin(), k2.end(), static_cast<P>(0));
        return;
    }

    const std::vector<size_t> v1_index = sort_indexes(v1);
    const std::vector<size_t> v2_index = sort_indexes(v2);

    const size_t max_matches = std::min(v1.size(), v2.size());
    k1.resize(max_matches);
    k2.resize(max_matches);

    size_t i = 0;
    size_t j = 0;
    size_t out = 0;
    while(i < v1_index.size() && j < v2_index.size()){
        const auto &lhs = v1[v1_index[i]];
        const auto &rhs = v2[v2_index[j]];
        if(lhs < rhs){
            ++i;
        }else if(rhs < lhs){
            ++j;
        }else{
            k1[out] = static_cast<P>(v1_index[i]);
            k2[out] = static_cast<P>(v2_index[j]);
            ++out;
            ++i;
            ++j;
        }
    }

    k1.resize(out);
    k2.resize(out);
}

template <typename T, typename P>
void vector_commonIndex_sorted1(const std::vector<T>& v1, const std::vector<T>& v2, std::vector<P>& k1, std::vector<P>& k2){
    vector_commonIndex(v1, v2, k1, k2);
    if(k1.empty()){
        return;
    }
    if(std::ranges::is_sorted(k1)){
        return;
    }

    const std::vector<size_t> sorted_old_positions = sort_indexes(k1);
    std::vector<size_t> target_position(k1.size());
    for(size_t new_pos = 0; new_pos < sorted_old_positions.size(); ++new_pos){
        const size_t old_pos = sorted_old_positions[new_pos];
        target_position[old_pos] = new_pos;
    }

    // Apply the permutation in place to both output arrays.
    for(size_t i = 0; i < target_position.size(); ++i){
        while(target_position[i] != i){
            const size_t j = target_position[i];
            std::swap(k1[i], k1[j]);
            std::swap(k2[i], k2[j]);
            std::swap(target_position[i], target_position[j]);
        }
    }
}

/* Permute vector elements to all the combinations
 * @param elements: vector of each possible values
 * @param combines: out: combinations of these vectors;
 * @param temp and col:  only for internal use, don't change these values;
 * @example
 *   #include <vector>
     vector<vector<int>> elements = {
           {0, 1, 2},
           {10, 11, 12},
           {20, 21, 22}
     };
     vector<vector<int>> combines;
     permute_vector(elements, combines);
 * @depends vector c++11
 * @Bug report: zhili<zhilizheng@outlook.com>
 */
template <typename T>
void permute_vector(const std::vector<std::vector<T>> &elements, std::vector<std::vector<T>> &combines, std::vector<T> temp = {}, size_t col = 0){
    if(elements.empty()){
        combines.emplace_back();
        return;
    }

    const size_t e_size = elements.size();
    if(col > e_size){
        return;
    }
    if(col == e_size){
        combines.emplace_back(temp);
        return;
    }
    for(size_t i = col; i < e_size; ++i){
        if(elements[i].empty()){
            return;
        }
    }

    size_t additional_count = 1;
    bool overflow = false;
    for(size_t i = col; i < e_size; ++i){
        if(additional_count > std::numeric_limits<size_t>::max() / elements[i].size()){
            overflow = true;
            break;
        }
        additional_count *= elements[i].size();
    }
    if(!overflow){
        combines.reserve(combines.size() + additional_count);
    }

    if(temp.size() != e_size){
        temp.resize(e_size);
    }

    std::vector<size_t> indices(e_size, 0);
    for(size_t i = 0; i < col; ++i){
        if(elements[i].empty()){
            return;
        }
        auto found = std::ranges::find(elements[i], temp[i]);
        if(found == elements[i].end()){
            return;
        }
        indices[i] = static_cast<size_t>(found - elements[i].begin());
    }

    bool done = false;
    while(!done){
        for(size_t i = col; i < e_size; ++i){
            temp[i] = elements[i][indices[i]];
        }
        combines.emplace_back(temp);

        size_t pos = e_size;
        while(pos > col){
            --pos;
            ++indices[pos];
            if(indices[pos] < elements[pos].size()){
                break;
            }
            indices[pos] = 0;
            if(pos == col){
                done = true;
            }
        }
    }
}

template <typename T>
std::string to_string_precision(const T value, const int n = 0){
    std::ostringstream out;
    if(n > 0){
        out.precision(n);
    }
    out << std::fixed << value;
    return out.str();
} 

template <typename T>
int findElementVector(const std::vector<T> &vec, const T &element, bool &found){
    auto it = std::find(vec.begin(), vec.end(), element);
    if(it != vec.end()){
        found = true;
        return(it - vec.begin());
    }else{
        found = false;
        return -1;
    }
}
