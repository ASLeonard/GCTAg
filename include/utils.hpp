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
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <tuple>
#include <ranges>
#include <sstream>
#include <cstdint>
#include <limits>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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
    if constexpr (std::is_same_v<T, std::string>) {
        std::unordered_set<std::string_view> seen;
        seen.reserve(v.size());
        for (const auto& s : v) {
            if (!seen.emplace(s).second) {
                return true;
            }
        }
        return false;
    } else {
        std::vector<T> t = v;
        std::ranges::sort(t);
        return std::adjacent_find(t.begin(), t.end()) != t.end();
    }
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

    if (std::ranges::is_sorted(v1) && std::ranges::is_sorted(v2)) {
        const size_t max_matches = std::min(v1.size(), v2.size());
        k1.clear();
        k2.clear();
        k1.reserve(max_matches);
        k2.reserve(max_matches);
        size_t i = 0;
        size_t j = 0;
        while (i < v1.size() && j < v2.size()) {
            if (v1[i] < v2[j]) {
                ++i;
            } else if (v2[j] < v1[i]) {
                ++j;
            } else {
                k1.push_back(static_cast<P>(i));
                k2.push_back(static_cast<P>(j));
                ++i;
                ++j;
            }
        }
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
    if (v1.empty() || v2.empty()) {
        k1.clear();
        k2.clear();
        return;
    }
    if (v1 == v2) {
        k1.resize(v1.size());
        std::iota(k1.begin(), k1.end(), static_cast<P>(0));
        k2.resize(v2.size());
        std::iota(k2.begin(), k2.end(), static_cast<P>(0));
        return;
    }

    if constexpr (std::is_same_v<T, std::string>) {
        constexpr P kConsumed = std::numeric_limits<P>::max();
        if (v2.size() <= v1.size() * 2 && v2.size() < static_cast<size_t>(kConsumed)) {
            std::unordered_map<std::string_view, P> lookup;
            lookup.reserve(v2.size());
            bool has_dup = false;
            for (size_t j = 0; j < v2.size(); ++j) {
                if (!lookup.emplace(std::string_view(v2[j]), static_cast<P>(j)).second) {
                    has_dup = true;
                    break;
                }
            }
            if (!has_dup) {
                k1.clear();
                k2.clear();
                const size_t cap = std::min(v1.size(), v2.size());
                k1.reserve(cap);
                k2.reserve(cap);
                for (size_t i = 0; i < v1.size(); ++i) {
                    auto it = lookup.find(std::string_view(v1[i]));
                    if (it != lookup.end() && it->second != kConsumed) {
                        k1.push_back(static_cast<P>(i));
                        k2.push_back(it->second);
                        it->second = kConsumed;
                    }
                }
                return;
            }
        } else if (v1.size() < static_cast<size_t>(kConsumed)) {
            std::unordered_map<std::string_view, P> lookup;
            lookup.reserve(v1.size());
            bool has_dup = false;
            for (size_t i = 0; i < v1.size(); ++i) {
                if (!lookup.emplace(std::string_view(v1[i]), static_cast<P>(i)).second) {
                    has_dup = true;
                    break;
                }
            }
            if (!has_dup) {
                std::vector<std::pair<P, P>> pairs;
                pairs.reserve(std::min(v1.size(), v2.size()));
                for (size_t j = 0; j < v2.size(); ++j) {
                    auto it = lookup.find(std::string_view(v2[j]));
                    if (it != lookup.end() && it->second != kConsumed) {
                        pairs.emplace_back(it->second, static_cast<P>(j));
                        it->second = kConsumed;
                    }
                }
                std::ranges::sort(pairs, [](const auto& a, const auto& b) { return a.first < b.first; });
                k1.resize(pairs.size());
                k2.resize(pairs.size());
                for (size_t idx = 0; idx < pairs.size(); ++idx) {
                    k1[idx] = pairs[idx].first;
                    k2[idx] = pairs[idx].second;
                }
                return;
            }
        }
    }

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

namespace gcta_text_io {

inline char ascii_toupper(char c) noexcept {
    return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

inline void uppercase_inplace(std::string& s) noexcept {
    for (char& c : s) {
        c = ascii_toupper(c);
    }
}

inline void split_ws_exact(std::string_view line, std::vector<std::string_view>& out) {
    out.clear();
    const char* p = line.data();
    const char* const end = p + line.size();
    const char* tok_start = p;
    for (; p < end; ++p) {
        if (*p == '\t' || *p == ' ') {
            out.emplace_back(tok_start, static_cast<size_t>(p - tok_start));
            tok_start = p + 1;
        }
    }
    out.emplace_back(tok_start, static_cast<size_t>(end - tok_start));
}

inline bool parse_int_sv(std::string_view sv, int& out) noexcept {
    if (sv.empty()) return false;
    const char* p = sv.data();
    const char* const end = p + sv.size();
    if (*p == '+') {
        ++p;
        if (p == end) return false;
    }
    int val = 0;
    const auto res = std::from_chars(p, end, val);
    if (res.ec != std::errc{} || res.ptr != end) return false;
    out = val;
    return true;
}

inline bool parse_int64_sv(std::string_view sv, int64_t& out) noexcept {
    if (sv.empty()) return false;
    const char* p = sv.data();
    const char* const end = p + sv.size();
    if (*p == '+') {
        ++p;
        if (p == end) return false;
    }
    int64_t val = 0;
    const auto res = std::from_chars(p, end, val);
    if (res.ec != std::errc{} || res.ptr != end) return false;
    out = val;
    return true;
}

inline bool parse_float_sv(std::string_view sv, float& out) noexcept {
    if (sv.empty()) return false;
    if (sv.size() == 1 && sv[0] == '0') {
        out = 0.0f;
        return true;
    }
    const char* p = sv.data();
    const char* const end = p + sv.size();
    if (*p == '+') {
        ++p;
        if (p == end) return false;
    }
    float val = 0.0f;
    const auto res = std::from_chars(p, end, val);
    if (res.ec == std::errc{} && res.ptr == end) {
        out = val;
        return true;
    }
    if (res.ec == std::errc::result_out_of_range && res.ptr == end) {
        // Match std::stof/strtof behavior for subnormal underflow to 0.0f
        std::string tmp(sv);
        char* endp = nullptr;
        errno = 0;
        const float f = std::strtof(tmp.c_str(), &endp);
        if (endp == tmp.c_str() + tmp.size() && errno != ERANGE) {
            out = f;
            return true;
        }
        if (endp == tmp.c_str() + tmp.size() && f == 0.0f) {
            out = 0.0f;
            return true;
        }
    }
    return false;
}

inline double parse_pheno_double_sv(std::string_view sv) noexcept {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    if (sv.empty()) return kNaN;
    if (sv.size() == 2 && sv[0] == '-' && sv[1] == '9') return kNaN;
    const char* p = sv.data();
    const char* const end = p + sv.size();
    if (*p == '+') {
        ++p;
        if (p == end) return kNaN;
    }
    double val = 0.0;
    const auto res = std::from_chars(p, end, val);
    if (res.ec == std::errc{} && res.ptr == end) {
        return val;
    }
    if (res.ec == std::errc::result_out_of_range && res.ptr == end) {
        std::string tmp(sv);
        char* endp = nullptr;
        const double d = std::strtod(tmp.c_str(), &endp);
        if (endp == tmp.c_str() + tmp.size()) {
            return d;
        }
    }
    return kNaN;
}

class FastLineReader {
public:
    FastLineReader() = default;
    explicit FastLineReader(const std::string& path) { open(path); }
    ~FastLineReader() { close(); }

    FastLineReader(const FastLineReader&) = delete;
    FastLineReader& operator=(const FastLineReader&) = delete;

    bool open(const std::string& path) {
        close();
        fd_ = ::open(path.c_str(), O_RDONLY
#ifdef O_CLOEXEC
                     | O_CLOEXEC
#endif
        );
        if (fd_ < 0) return false;
        struct stat st{};
        if (::fstat(fd_, &st) == 0 && st.st_size > 0) {
            file_size_ = static_cast<uint64_t>(st.st_size);
        }
        const size_t init_cap = (file_size_ > 0)
            ? std::min<size_t>(kChunkSize, static_cast<size_t>(file_size_) + 4096)
            : kChunkSize;
        buf_.resize(std::max<size_t>(init_cap, 65536));
        pos_ = 0;
        len_ = 0;
        eof_ = false;
        io_error_ = false;
        return true;
    }

    void close() noexcept {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        pos_ = 0;
        len_ = 0;
        eof_ = false;
        file_size_ = 0;
    }

    bool is_open() const noexcept { return fd_ >= 0; }
    bool io_error() const noexcept { return io_error_; }
    uint64_t file_size() const noexcept { return file_size_; }

    bool next_line(std::string_view& out) {
        while (true) {
            if (pos_ < len_) {
                const char* start = buf_.data() + pos_;
                const size_t avail = len_ - pos_;
                const void* nl = std::memchr(start, '\n', avail);
                if (nl != nullptr) {
                    const size_t line_len = static_cast<const char*>(nl) - start;
                    pos_ += line_len + 1;
                    size_t trimmed = line_len;
                    if (trimmed > 0 && start[trimmed - 1] == '\r') --trimmed;
                    out = std::string_view(start, trimmed);
                    return true;
                }
            }
            if (eof_) {
                if (pos_ < len_) {
                    const char* start = buf_.data() + pos_;
                    size_t trimmed = len_ - pos_;
                    pos_ = len_;
                    if (trimmed > 0 && start[trimmed - 1] == '\r') --trimmed;
                    out = std::string_view(start, trimmed);
                    return true;
                }
                return false;
            }
            const size_t rem = len_ - pos_;
            if (rem > 0 && pos_ > 0) {
                std::memmove(buf_.data(), buf_.data() + pos_, rem);
            }
            pos_ = 0;
            len_ = rem;
            if (len_ == buf_.size()) {
                buf_.resize(buf_.size() * 2);
            }
            while (true) {
                const ssize_t r = ::read(fd_, buf_.data() + len_, buf_.size() - len_);
                if (r > 0) {
                    len_ += static_cast<size_t>(r);
                    break;
                }
                if (r == 0) {
                    eof_ = true;
                    break;
                }
                if (errno == EINTR) continue;
                io_error_ = true;
                eof_ = true;
                break;
            }
        }
    }

private:
    static constexpr size_t kChunkSize = 4ULL << 20; // 4 MiB
    int fd_ = -1;
    uint64_t file_size_ = 0;
    std::vector<char> buf_;
    size_t pos_ = 0;
    size_t len_ = 0;
    bool eof_ = false;
    bool io_error_ = false;
};

} // namespace gcta_text_io

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
