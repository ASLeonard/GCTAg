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
#include <cstdlib>
#include <cstdio>
#include <cstring>

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
