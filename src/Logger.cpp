/*
   Global Singleton logger system with robust functions and thread safe.

      * open a logger file first, otherwise no output into log
      * e(int level, string message):  prompt error, and exit the program, level is the number of indent spaces
      * i:  prompt information
      * w:  prompt warning message
      * d:  debug message, only seen in the debug mode
      * m:  message that only show on the terminal that not log into logger file
      * l:  log into logger file only;
      * p:  progress, that always show in one line in the terminal but no output into logger file.
      * << :  use like std::cout

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/

#include "Logger.h"
#include <cstdlib>
#include <iostream>
#include <unistd.h>

using std::cout;

Logger::Logger() {
    if (isatty(STDOUT_FILENO)) {
        style_map_ = {"\033[0m", "\033[0;32m", "\r", "\033[0;33m", "\033[0;31m", "\033[0;34m"};
    } else {
        style_map_ = {"", "", "\n", "", "", ""};
    }
}

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

bool Logger::check() const {
    return m_logFile_.is_open() && !m_FileName_.empty();
}

void Logger::open(std::string_view ofile) {
    if (check()) {
        cout << "Logger has been set, not support to set another time" << endl;
    } else {
        m_FileName_ = std::string(ofile);
        m_logFile_.open(m_FileName_, std::ios::out);
        if (!check()) {
            error(0, "can't write to log file [{}].\nPlease check file/folder permission or disk quota.", ofile);
        }
    }
}

void Logger::close() {
    m_logFile_.close();
}

void Logger::flush() {
    m_logFile_.flush();
}

void Logger::Log(int level, Type type, std::string_view prompt, std::string_view message) {
    const std::string spaces(static_cast<std::size_t>(level) * 2, ' ');
    std::lock_guard<std::mutex> lock(log_mutex_);
    (*this) << spaces << type << prompt << INFO << message << endl;
}

[[noreturn]] void Logger::e(int level, std::string_view message, std::string_view title) {
    const std::string head = title.empty() ? std::string("Error: ") : (std::string(title) + " ");
    Log(level, ERROR, head, message);
    Log(level, INFO, "", "An error occurs, please check the options or data");
    std::exit(EXIT_FAILURE);
}

int Logger::precision(int p) {
    cout.precision(p);
    m_logFile_.precision(p);
    return static_cast<int>(cout.precision());
}

string Logger::setprecision(int p) {
    cout.precision(p);
    m_logFile_.precision(p);
    return {};
}

int Logger::precision() {
    return static_cast<int>(cout.precision());
}

void Logger::i(int level, std::string_view message, std::string_view title) {
    const std::string head = title.empty() ? std::string() : (std::string(title) + " ");
    Log(level, PROMPT, head, message);
}

void Logger::w(int level, std::string_view message, std::string_view title) {
    const std::string head = title.empty() ? std::string("Warning: ") : (std::string(title) + " ");
    Log(level, WARN, head, message);
}

void Logger::d(int level, std::string_view message, std::string_view title) {
#ifndef NDEBUG
    const std::string head = title.empty() ? std::string("Debug: ") : (std::string(title) + " ");
    Log(level, DEBUG, head, message);
#else
    (void)level;
    (void)message;
    (void)title;
#endif
}

void Logger::p(int level, std::string_view message, std::string_view title) {
    const std::string head = title.empty() ? std::string() : (std::string(title) + " ");
    const std::string spaces(static_cast<std::size_t>(level) * 2, ' ');
    std::lock_guard<std::mutex> lock(log_mutex_);
    m_stat_ = PROGRESS;
    (*this) << spaces << head << message << PROGRESS << std::flush;
    m_stat_ = INFO;
}

void Logger::m(int level, std::string_view message, std::string_view title) {
    const std::string head = title.empty() ? std::string() : (std::string(title) + " ");
    const std::string spaces(static_cast<std::size_t>(level) * 2, ' ');
    std::lock_guard<std::mutex> lock(log_mutex_);
    m_stat_ = PROGRESS;
    (*this) << spaces << PROMPT << head << INFO << message << endl;
    m_stat_ = INFO;
}

void Logger::l(int level, std::string_view message, std::string_view title) {
    const std::string head = title.empty() ? std::string() : (std::string(title) + " ");
    const std::string spaces(static_cast<std::size_t>(level) * 2, ' ');
    std::lock_guard<std::mutex> lock(log_mutex_);
    m_logFile_ << spaces << head << message << endl;
}

void Logger::ts(std::string_view key) {
    const auto now = std::chrono::steady_clock::now();
    if (auto it = time_map_.find(key); it != time_map_.end()) {
        it->second = now;
    } else {
        time_map_.emplace(std::string(key), now);
    }
}

float Logger::tp(std::string_view key) {
    const auto end = std::chrono::steady_clock::now();
    if (auto it = time_map_.find(key); it != time_map_.end()) {
        return std::chrono::duration_cast<std::chrono::duration<float>>(end - it->second).count();
    }
    w(2, "can't find key of time start");
    return 0.0f;
}

Logger& Logger::operator<<(Type type) {
    m_stat_ = type;
    cout << style_map_[static_cast<std::size_t>(type)];
    return *this;
}

Logger& Logger::operator<<(std::ostream& (*op)(std::ostream&)) {
    (*op)(cout);
    if (m_stat_ != PROGRESS) {
        (*op)(m_logFile_);
    }
    return *this;
}

Logger& Logger::operator<<(std::ios& (*pf)(std::ios&)) {
    cout << pf;
    m_logFile_ << pf;
    return *this;
}

Logger& Logger::operator<<(std::ios_base& (*pf)(std::ios_base&)) {
    cout << pf;
    m_logFile_ << pf;
    return *this;
}


