/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   Global Singleton logger system with robust functions and thread safe.

      * open a logger file first, otherwise no output into log
      * e(int level, string message):  prompt error, and exit the program, level is the number of indent spaces
      * i:  prompt information
      * w:  prompt warning message
      * d:  debug message, only seen in the debug mode
      * m:  message that only show on the terminal that not log into logger file
      * l:  log into logger file only
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

#ifndef GCTA2_LOGGER_H
#define GCTA2_LOGGER_H

#include <array>
#include <chrono>
#include <format>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#define LOGGER (Logger::instance())
#define LOGGER_P (&Logger::instance())

using std::string;
using std::endl;

class Logger {
public:
    enum Type { INFO = 0, PROMPT = 1, PROGRESS = 2, WARN = 3, ERROR = 4, DEBUG = 5 };

    void Log(int level, Type type, std::string_view prompt, std::string_view message);
    [[noreturn]] void e(int level, std::string_view message, std::string_view title = {});
    void i(int level, std::string_view message, std::string_view title = {});
    void d(int level, std::string_view message, std::string_view title = {});
    void w(int level, std::string_view message, std::string_view title = {});
    void p(int level, std::string_view message, std::string_view title = {});
    void m(int level, std::string_view message, std::string_view title = {});
    void l(int level, std::string_view message, std::string_view title = {});

    template <typename... Args>
    void info(int level, std::format_string<Args...> fmt, Args&&... args) {
        i(level, std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void warn(int level, std::format_string<Args...> fmt, Args&&... args) {
        w(level, std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    [[noreturn]] void error(int level, std::format_string<Args...> fmt, Args&&... args) {
        e(level, std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void debug(int level, std::format_string<Args...> fmt, Args&&... args) {
#ifndef NDEBUG
        d(level, std::format(fmt, std::forward<Args>(args)...));
#else
        (void)level;
        (void)fmt;
        ((void)args, ...);
#endif
    }

    void ts(std::string_view marker);
    float tp(std::string_view marker);

    Logger& operator<<(Type type);
    Logger& operator<<(std::ostream& (*manip)(std::ostream&));
    Logger& operator<<(std::ios& (*pf)(std::ios&));
    Logger& operator<<(std::ios_base& (*pf)(std::ios_base&));

    template <typename T>
    Logger& operator<<(const T& t) {
        std::cout << t;
        if (m_stat_ != PROGRESS) {
            m_logFile_ << t;
        }
        return *this;
    }

    int precision(int p);
    int precision();
    string setprecision(int p);

    void open(std::string_view ofile);
    void close();
    void flush();

    static Logger& instance();
    static Logger* GetLogger() { return &instance(); }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

private:
    Logger();
    bool check() const;

    std::array<std::string_view, 6> style_map_{};
    std::map<std::string, std::chrono::steady_clock::time_point, std::less<>> time_map_;
    std::mutex log_mutex_;
    Type m_stat_ = INFO;
    std::string m_FileName_;
    std::ofstream m_logFile_;
};

#endif //GCTA2_LOGGER_H
