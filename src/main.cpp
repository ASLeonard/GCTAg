/*
 *
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * 2010-present Jian Yang <jian.yang.qt@gmail.com> and others
 *
 * Mock layer of version 2 by Zhili Zheng <zhilizheng@outlook.com>
 *
 * Options parser, and bridge unsupported flags to version 1.

 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * A copy of the GNU General Public License is attached along with this program.
 * If not, see <http://www.gnu.org/licenses/>.
*/

#include "Cli.hpp"
#include "GRM.h"
#include "Logger.h"
#include "MLMA.h"
#include "PCA_stream.h"
#include "config.h"
#include "main/option.h"
#include "mem.hpp"
#include "utils.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

using std::string;
using std::to_string;
using std::vector;

void out_ver(bool flag_outFile) {
    auto log = [flag_outFile](int level, std::string_view msg, std::string_view title = {}) {
        if (flag_outFile) {
            LOGGER.l(level, msg, title);
        } else {
            LOGGER.m(level, msg, title);
        }
    };

    log(0, "*******************************************************************");
    log(0, "* Genome-wide Complex Trait Analysis -- Agriculture (GCTAg)");
    log(0, std::format("* {} ({})", GCTA_VERSION, getOSName()));
#if defined(__clang__)
    log(0, std::format("* Built at {} {}, by Clang {}.{}.{}",
                       __DATE__, __TIME__, __clang_major__, __clang_minor__, __clang_patchlevel__));
#elif defined(__GNUC__)
    log(0, std::format("* Built at {} {}, by GCC {}.{}",
                       __DATE__, __TIME__, __GNUC__, __GNUC_MINOR__));
#else
    log(0, std::format("* Built at {} {}", __DATE__, __TIME__));
#endif
#ifdef GCTA_TARGET_ARCH_STR
    log(0, std::string("* Target arch: ") + GCTA_TARGET_ARCH_STR, "");
#endif
#ifdef GCTA_BLAS_BACKEND_STR
    {
        std::string blas_line = std::string("* BLAS/LAPACK: ") + GCTA_BLAS_BACKEND_STR;
#ifdef GCTA_BLAS_VERSION_STR
        blas_line += std::string(" (") + GCTA_BLAS_VERSION_STR + ")";
#endif
        log(0, blas_line, "");
    }
#endif
    log(0, "* GCTA: (C) 2010-present, Yang Lab, Westlake University", "");
    log(0, "* GCTAg: (C) 2026-present, Alexander S. Leonard, ETH Zurich", "");
    log(0, "* Please report bugs at https://github.com/ASLeonard/GCTAg/issues", "");
    log(0, "*******************************************************************", "");
    log(0, string("at ") + getLocalTime() + ".", "Analysis started");
    log(0, "Hostname: " + getHostName(), "");
    log(0, "", "");
}

#ifdef GCTA_USE_JEMALLOC
extern "C" const char* malloc_conf = "thp:never";
#endif

namespace {

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

string compute_log_prefix(const gcta_cli::ParsedCli& parsed, bool& out_was_explicit) {
    string base = parsed.global.out_prefix();
    out_was_explicit = parsed.global.out.has_value();

    if (const auto* legacy = std::get_if<gcta_cli::LegacyOptions>(&parsed.command);
        legacy != nullptr && !out_was_explicit) {
        for (size_t i = 0; i + 1 < legacy->raw_args.size(); ++i) {
            if (legacy->raw_args[i] == "--out") {
                base = legacy->raw_args[i + 1];
                out_was_explicit = true;
                break;
            }
        }
    }

    if (const auto* gm = std::get_if<gcta_cli::GrmMakeOptions>(&parsed.command);
        gm != nullptr && gm->part.has_value()) {
        const string s_parts = to_string(gm->part->first);
        const string c_parts = to_string(gm->part->second);
        base += ".part_" + s_parts + "_" + string(s_parts.length() - c_parts.length(), '0') + c_parts;
    }

    return base;
}

int resolve_threads(const gcta_cli::ParsedCli& parsed, bool& is_threaded, bool& explicit_threads) {
    int thread_num = 1;
    is_threaded = false;
    explicit_threads = false;

    if (const char* env_threads = std::getenv("OMP_NUM_THREADS")) {
        try {
            const int v = std::stoi(env_threads);
            if (v > 0) {
                thread_num = v;
                is_threaded = true;
            }
        } catch (...) {
        }
    }

    if (parsed.global.threads.has_value()) {
        thread_num = *parsed.global.threads;
        is_threaded = true;
        explicit_threads = true;
        return thread_num;
    }

    if (const auto* legacy = std::get_if<gcta_cli::LegacyOptions>(&parsed.command)) {
        for (size_t i = 0; i + 1 < legacy->raw_args.size(); ++i) {
            if (legacy->raw_args[i] == "--threads" || legacy->raw_args[i] == "--thread-num") {
                try {
                    const int v = std::stoi(legacy->raw_args[i + 1]);
                    if (v > 0) {
                        thread_num = v;
                        is_threaded = true;
                        explicit_threads = true;
                    }
                } catch (...) {
                }
            }
        }
    }

    return thread_num;
}

void configure_threading(int thread_num, bool explicit_threads) {
    // Snapshot initial CPU affinity BEFORE any OpenMP runtime initialization
    // so background I/O threads can restore it instead of inheriting Core-0 binding.
    gcta_snapshot_initial_affinity();

#if defined(GCTA_USE_AOCL) && !defined(GCTA_LAPACKE_IS_MKL)
    // Short spin prevents microsecond futex storms between back-to-back
    // BLIS/AOCL-DLP/OMP regions while still sleeping during disk I/O.
    // overwrite=0 respects explicit user overrides (e.g. KMP_BLOCKTIME=0).
    setenv("KMP_BLOCKTIME", "5", 0);
    setenv("GOMP_SPINCOUNT", "50000", 0);
#else
    // Make OMP worker threads sleep immediately after a parallel region ends,
    // rather than spin-waiting. Spinning threads consume memory bandwidth and
    // starve BLAS (MKL/OpenBLAS/Accelerate) calls that follow parallel regions.
    setenv("KMP_BLOCKTIME", "0", 0);   // 0 = don't override user setting
    setenv("GOMP_SPINCOUNT", "0", 0);
#endif

#if defined(__linux__)
    const char* disable_numa_env = std::getenv("GCTA_DISABLE_NUMA");
    const bool disable_numa = (disable_numa_env != nullptr && std::strcmp(disable_numa_env, "0") != 0);
    if (thread_num > 1 && !disable_numa) {
        // Spread threads evenly across NUMA domains / CCDs to unlock full DDR5
        // channel bandwidth & L3 capacity; overwrite=0 respects user/Slurm env.
        setenv("OMP_PROC_BIND", "spread", 0);
        setenv("OMP_PLACES", "cores", 0);
    }
#endif

    const string thread_str = to_string(thread_num);
    setenv("OMP_NUM_THREADS", thread_str.c_str(), 1);
#if defined(GCTA_USE_AOCL)
    const int aocl_overwrite = explicit_threads ? 1 : 0;
    setenv("BLIS_NUM_THREADS", thread_str.c_str(), aocl_overwrite);
    setenv("DLP_NUM_THREADS", thread_str.c_str(), aocl_overwrite);
#else
    (void)explicit_threads;
#endif
#if defined(GCTA_USE_MKL) || defined(GCTA_LAPACKE_IS_MKL)
    // libmkl_rt.so reads MKL_THREADING_LAYER lazily on first call; match the
    // OpenMP runtime linked by this compiler while respecting user overrides.
#if defined(__clang__)
    setenv("MKL_THREADING_LAYER", "INTEL", 0);
#elif defined(__GNUC__)
    setenv("MKL_THREADING_LAYER", "GNU", 0);
#endif
#endif
    omp_set_num_threads(thread_num);
}

void dispatch_legacy(const char* prog_name,
                     const gcta_cli::LegacyOptions& legacy,
                     const string& log_prefix,
                     int thread_num,
                     bool is_threaded) {
    vector<string> args = legacy.raw_args;

    auto has_flag = [&args](std::string_view f) {
        return std::find(args.begin(), args.end(), f) != args.end();
    };
    if (!has_flag("--out")) {
        args.emplace_back("--out");
        args.emplace_back(log_prefix);
    }
    if (is_threaded && !has_flag("--threads") && !has_flag("--thread-num")) {
        args.emplace_back("--threads");
        args.emplace_back(to_string(thread_num));
    }

    vector<char*> legacy_argv;
    legacy_argv.reserve(args.size() + 1);
    legacy_argv.push_back(const_cast<char*>(prog_name));
    for (string& s : args) {
        legacy_argv.push_back(s.data());
    }

    try {
        option(static_cast<int>(legacy_argv.size()), legacy_argv.data());
    } catch (const string& err_msg) {
        LOGGER.e(0, err_msg);
    } catch (const char* err_msg) {
        LOGGER.e(0, string(err_msg));
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    gcta_cli::ParsedCli parsed;
    const int cli_rc = gcta_cli::parse_cli(argc, argv, parsed);
    if (cli_rc != 0) {
        return cli_rc;
    }
    if (std::holds_alternative<gcta_cli::HelpOrVersion>(parsed.command)) {
        return 0;
    }
    if (std::holds_alternative<std::monostate>(parsed.command)) {
        return EXIT_FAILURE;
    }

    out_ver(false);
    LOGGER.ts("main");

    bool out_explicit = false;
    const string log_prefix = compute_log_prefix(parsed, out_explicit);
    if (!out_explicit) {
        LOGGER.w(0, "Warning: No output file specified. Using default output file 'output'.");
    }

    bool is_threaded = false;
    bool explicit_threads = false;
    const int thread_num = resolve_threads(parsed, is_threaded, explicit_threads);
    configure_threading(thread_num, explicit_threads);

    LOGGER.open(log_prefix + ".log");
    out_ver(true);

    LOGGER.i(0, "Options: ");
    LOGGER.i(0, " ");
    for (size_t i = 1; i < parsed.raw_argv.size(); ++i) {
        LOGGER << parsed.raw_argv[i] << " ";
    }
    LOGGER << std::endl;
    LOGGER.i(0, "");

    if (is_threaded) {
        LOGGER.i(0, "The program will be running with up to " + to_string(thread_num) + " threads.");
    }

    std::visit(overloaded{
        [](std::monostate) {},
        [](const gcta_cli::HelpOrVersion&) {},
        [&log_prefix](const gcta_cli::GrmMakeOptions& opts) { GRM::runMake(opts, log_prefix); },
        [&log_prefix](const gcta_cli::GrmMergeOptions& opts) { GRM::runMerge(opts, log_prefix); },
        [&log_prefix](const gcta_cli::GrmPruneOptions& opts) { GRM::runPrune(opts, log_prefix); },
        [&log_prefix](const gcta_cli::GrmSparseOptions& opts) { GRM::runSparse(opts, log_prefix); },
        [&log_prefix](const gcta_cli::GrmSubtractOptions& opts) { GRM::runSubtract(opts, log_prefix); },
        [&log_prefix](const gcta_cli::GrmUnifyOptions& opts) { GRM::runUnify(opts, log_prefix); },
        [&log_prefix](const gcta_cli::PcaOptions& opts) { PCAStream::run(opts, log_prefix); },
        [&log_prefix](const gcta_cli::MlmaOptions& opts) { MLMA::run(opts, log_prefix); },
        [&](const gcta_cli::LegacyOptions& opts) {
            dispatch_legacy(argv[0], opts, log_prefix, thread_num, is_threaded);
        }
    }, parsed.command);

    LOGGER.i(0, "");
    LOGGER.i(0, "at " + getLocalTime(), "Analysis finished");
    float duration = LOGGER.tp("main");
    int hours = static_cast<int>(duration) / 3600;
    string time_str = (hours == 0) ? "" : (to_string(hours) + " hour" + ((hours == 1) ? " " : "s "));
    int mins = static_cast<int>(duration - 3600 * hours) / 60;
    time_str += (mins == 0) ? "" : (to_string(mins) + " minute" + ((mins == 1) ? " " : "s "));
    float seconds = duration - 3600 * hours - 60 * mins;
    time_str = time_str + ((time_str == "") ? to_string_precision(seconds, 2) : to_string(static_cast<int>(seconds))) + " sec";

#ifdef __linux__
    if (parsed.global.verbose) {
        float vmem = roundf(1000.0f * getVMPeakKB() / 1024.0f / 1024.0f) / 1000.0f;
        float mem = roundf(1000.0f * getMemPeakKB() / 1024.0f / 1024.0f) / 1000.0f;
        LOGGER.setprecision(3);
        LOGGER << "Peak memory: " << mem << " GB; Virtual memory: " << vmem << " GB." << std::endl;
    }
#endif

    LOGGER << "Overall computational time: " << time_str << "." << std::endl;
    return EXIT_SUCCESS;
}
