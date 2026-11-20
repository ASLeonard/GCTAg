/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   Mocks the same memory allocation api cross platform

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/


#include "mem.hpp"
#include "Logger.h"

#include <atomic>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <fcntl.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#if __has_include(<linux/mempolicy.h>)
#include <linux/mempolicy.h>
#endif
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
#ifndef MPOL_F_MEMS_ALLOWED
#define MPOL_F_MEMS_ALLOWED (1 << 2)
#endif
#elif defined(__APPLE__)
#include <unistd.h>
#endif

namespace gcta_mem_detail {

constexpr std::size_t kNumaInterleaveThreshold = 64ULL << 20;  // 64 MiB

std::size_t page_size() noexcept {
    static const std::size_t cached = []() noexcept -> std::size_t {
        long ps = ::sysconf(_SC_PAGESIZE);
        if (ps >= 4096) {
            const auto ups = static_cast<std::size_t>(ps);
            if (std::has_single_bit(ups)) return ups;
        }
        return 4096;
    }();
    return cached;
}

#if defined(__linux__)

struct AffinityState {
    std::atomic<bool> valid{false};
    std::size_t bytes{0};
    alignas(cpu_set_t) unsigned long mask[128]{};  // up to 8192 CPUs on 64-bit
};

AffinityState& affinity_state() noexcept {
    static AffinityState st;
    return st;
}

struct NumaPolicyState {
    static constexpr std::size_t kMaskWords = 16;  // 1024 NUMA node bits on 64-bit
    unsigned long nodemask[kMaskWords]{};
    unsigned long maxnode{0};
    int node_count{1};
    bool query_ok{false};
};

const NumaPolicyState& numa_policy_state() noexcept {
    static const NumaPolicyState st = []() noexcept {
        NumaPolicyState s;
#if defined(SYS_get_mempolicy) && defined(SYS_mbind)
        constexpr unsigned long kMaxNode =
            static_cast<unsigned long>(NumaPolicyState::kMaskWords * sizeof(unsigned long) * 8);
        if (::syscall(SYS_get_mempolicy, nullptr, s.nodemask, kMaxNode, nullptr,
                      static_cast<unsigned long>(MPOL_F_MEMS_ALLOWED)) == 0) {
            int count = 0;
            for (unsigned long w : s.nodemask) {
                count += std::popcount(w);
            }
            if (count >= 1) {
                s.node_count = count;
                s.maxnode = kMaxNode;
                s.query_ok = true;
            }
        }
#endif
        return s;
    }();
    return st;
}

bool numa_disabled() noexcept {
    const char* env = std::getenv("GCTA_DISABLE_NUMA");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

#endif  // __linux__

}  // namespace gcta_mem_detail

void gcta_snapshot_initial_affinity() noexcept {
#if defined(__linux__)
    static std::once_flag once;
    std::call_once(once, []() noexcept {
        auto& st = gcta_mem_detail::affinity_state();
        std::size_t bytes = sizeof(cpu_set_t);
        while (bytes <= sizeof(st.mask)) {
            std::memset(st.mask, 0, bytes);
            if (::sched_getaffinity(0, bytes, reinterpret_cast<cpu_set_t*>(st.mask)) == 0) {
                st.bytes = bytes;
                st.valid.store(true, std::memory_order_release);
                return;
            }
            if (errno != EINVAL) {
                return;
            }
            bytes *= 2;
        }
    });
#endif
}

void gcta_restore_initial_affinity() noexcept {
#if defined(__linux__)
    const auto& st = gcta_mem_detail::affinity_state();
    if (!st.valid.load(std::memory_order_acquire)) {
        return;
    }
    (void)::sched_setaffinity(0, st.bytes, reinterpret_cast<const cpu_set_t*>(st.mask));
#endif
}

int gcta_numa_node_count() noexcept {
#if defined(__linux__)
    return gcta_mem_detail::numa_policy_state().node_count;
#else
    return 1;
#endif
}

void gcta_numa_interleave(void* ptr, std::size_t bytes) noexcept {
#if defined(__linux__) && defined(SYS_get_mempolicy) && defined(SYS_mbind)
    if (ptr == nullptr || bytes < gcta_mem_detail::kNumaInterleaveThreshold) {
        return;
    }
    if (gcta_mem_detail::numa_disabled()) {
        return;
    }
    const auto& st = gcta_mem_detail::numa_policy_state();
    if (!st.query_ok || st.node_count <= 1) {
        return;
    }

    const std::uintptr_t page_sz = static_cast<std::uintptr_t>(gcta_mem_detail::page_size());
    const std::uintptr_t raw_start = reinterpret_cast<std::uintptr_t>(ptr);
    const std::uintptr_t raw_end = raw_start + bytes;
    if (raw_end < raw_start) {
        return;
    }

    const std::uintptr_t aligned_start = (raw_start + page_sz - 1) & ~(page_sz - 1);
    const std::uintptr_t aligned_end   = raw_end & ~(page_sz - 1);
    if (aligned_end <= aligned_start) {
        return;
    }

    const unsigned long aligned_len = static_cast<unsigned long>(aligned_end - aligned_start);
    if (::syscall(SYS_mbind,
                  reinterpret_cast<void*>(aligned_start),
                  aligned_len,
                  static_cast<int>(MPOL_INTERLEAVE),
                  st.nodemask,
                  st.maxnode,
                  0U) == 0) {
        static std::once_flag log_once;
        std::call_once(log_once, [&st]() {
            try {
                LOGGER.i(0, "NUMA: interleaving large shared buffers across " +
                            std::to_string(st.node_count) + " allowed nodes.");
            } catch (...) {
            }
        });
    }
#else
    (void)ptr;
    (void)bytes;
#endif
}

void* gcta_alloc_numa_interleaved(std::size_t bytes, std::size_t align) noexcept {
    if (bytes == 0) return nullptr;
    std::size_t eff_align = align < sizeof(void*) ? sizeof(void*) : align;
    std::size_t eff_bytes = bytes;
    if (bytes >= gcta_mem_detail::kNumaInterleaveThreshold) {
        const std::size_t page_sz = gcta_mem_detail::page_size();
        if (eff_align < page_sz) eff_align = page_sz;
        if (bytes > std::numeric_limits<std::size_t>::max() - (page_sz - 1)) return nullptr;
        eff_bytes = (bytes + page_sz - 1) & ~(page_sz - 1);
    }
    void* p = nullptr;
    if (posix_memalign(&p, eff_align, eff_bytes) != 0) return nullptr;
    if (bytes >= gcta_mem_detail::kNumaInterleaveThreshold) {
        gcta_numa_interleave(p, eff_bytes);
    }
    return p;
}

void gcta_fadvise_dontneed(int fd, off_t off, off_t len) noexcept {
#if defined(__linux__) && defined(POSIX_FADV_DONTNEED)
    if (fd >= 0) {
        (void)::posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED);
    }
#else
    (void)fd;
    (void)off;
    (void)len;
#endif
}

void gcta_fadvise_sequential(int fd) noexcept {
#if defined(__linux__) && defined(POSIX_FADV_SEQUENTIAL)
    if (fd >= 0) {
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
#else
    (void)fd;
#endif
}

void gcta_fadvise_random(int fd) noexcept {
#if defined(__linux__) && defined(POSIX_FADV_RANDOM)
    if (fd >= 0) {
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    }
#else
    (void)fd;
#endif
}

#if defined(__APPLE__)
#include <mach/mach.h>

// Returns current RSS in KB using Mach task API
static long macosRSSKB() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<long>(info.phys_footprint / 1024);
    }
    return -1;
}

int getVMemKB()    { return static_cast<int>(macosRSSKB()); }
int getMemKB()     { return static_cast<int>(macosRSSKB()); }
int getMemPeakKB() { return static_cast<int>(macosRSSKB()); }
int getVMPeakKB()  { return static_cast<int>(macosRSSKB()); }

#else // Linux

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

static int readProcStatus(const char* key, int keylen) {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return -1;
    int result = -1;
    char line[128];
    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, key, keylen) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

// limit to 2TB
int getVMemKB()    { return readProcStatus("VmSize:", 7); }
int getMemKB()     { return readProcStatus("VmRSS:",  6); }
int getMemPeakKB() { return readProcStatus("VmHWM:",  6); }
int getVMPeakKB()  { return readProcStatus("VmPeak:", 7); }

#endif // __APPLE__






