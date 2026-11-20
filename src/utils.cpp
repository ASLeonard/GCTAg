/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   Utils.

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/


#include <utils.hpp>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

#include <unistd.h>

std::string_view getUtilsErrorMessage(UtilsError error){
    switch(error){
    case UtilsError::EmptyInput:
        return "input path is empty";
    case UtilsError::NullFileHandle:
        return "file handle is null";
    case UtilsError::FileSeekFailed:
        return "failed to seek in file";
    case UtilsError::FileTellFailed:
        return "failed to read file position";
    }
    return "unknown utils error";
}

std::string getHostName(){
    std::string computerName;

    if (const char *temp = std::getenv("HOSTNAME"); temp != nullptr) {
        computerName = temp;
    } else {
        std::array<char, 512> buffer{};
        if (gethostname(buffer.data(), buffer.size()) == 0) {
            computerName = buffer.data();
        }
    }
    return computerName;
}

std::string getOSName(){
    #ifdef __linux__
    return "Linux";
    #elif __APPLE__ || __MACH__
    return "Mac";
    #else
    return "Other";
    #endif
}

std::string getSSEvar(){
    #ifdef __AVX512F__
    return "avx512";
    #elif __AVX2__
    return "avx2";
    #elif __AVX__
    return "avx";
    #elif __SSE4_1__
    return "sse4";
    #elif __SSE2__
    return "sse2";
    #else
    return "unknown";
    #endif
}

std::string getLocalTime(){
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
    localtime_r(&now_c, &local_tm);
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%T %Z on %a %b %d %Y");
    return oss.str();
}

std::expected<std::string, UtilsError> getFileName(std::string_view path){
    if(path.empty()){
        return std::unexpected(UtilsError::EmptyInput);
    }
    return std::filesystem::path(path).filename().string();
}

// get the path without the last slash
std::expected<std::string, UtilsError> getPathName(std::string_view path){
    if(path.empty()){
        return std::unexpected(UtilsError::EmptyInput);
    }
    return std::filesystem::path(path).parent_path().string();
}

std::expected<std::string, UtilsError> joinPath(std::string_view dir, std::string_view path){
    if(dir.empty() && path.empty()){
        return std::unexpected(UtilsError::EmptyInput);
    }
    if(dir.empty()){
        return std::string(path);
    }
    if(path.empty()){
        return std::string(dir);
    }
    return (std::filesystem::path(dir) / std::filesystem::path(path)).string();
}

std::expected<uint64_t, UtilsError> getFileByteSize(FILE * file) {
    if(file == nullptr){
        return std::unexpected(UtilsError::NullFileHandle);
    }

    if(fseeko(file, 0, SEEK_END) != 0){
        return std::unexpected(UtilsError::FileSeekFailed);
    }
    const auto end_pos = ftello(file);
    std::rewind(file);
    if(end_pos < 0){
        return std::unexpected(UtilsError::FileTellFailed);
    }
    return static_cast<uint64_t>(end_pos);
}


