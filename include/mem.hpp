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

#ifndef GCTA2_MEM_HPP
#define GCTA2_MEM_HPP

#include <bit>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <stdlib.h>
#include <sys/types.h>
#include <type_traits>
#include <utility>

// ---------------------------------------------------------------------------
// AlignedAllocator<T, Align>
//
// Standard C++17/20 over-aligned allocator backed by ::operator new/delete
// with std::align_val_t. Guarantees Align-byte alignment for std::vector
// buffers passed to plink2/SIMD/BLAS code and hits jemalloc/amdalloc's
// sized+aligned fast path on deallocation.
// ---------------------------------------------------------------------------
template <typename T, std::size_t Align = 64>
struct AlignedAllocator {
    static_assert(Align >= alignof(T) && std::has_single_bit(Align),
                  "Alignment must be a power of two >= alignof(T)");

    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Align}));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        if (!p) return;
#if defined(__cpp_sized_deallocation)
        ::operator delete(p, n * sizeof(T), std::align_val_t{Align});
#else
        (void)n;
        ::operator delete(p, std::align_val_t{Align});
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };
};

template <typename T, typename U, std::size_t A>
constexpr bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return true;
}

template <typename T, typename U, std::size_t A>
constexpr bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return false;
}

// ---------------------------------------------------------------------------
// In-program Linux NUMA, thread affinity, and POSIX fadvise helpers
// (Zero external libnuma dependency; safe no-ops on macOS / single-node.)
// ---------------------------------------------------------------------------

// Snapshot the process's initial CPU affinity mask before OpenMP initializes
// (and potentially pins thread 0 to a single core under OMP_PROC_BIND).
void gcta_snapshot_initial_affinity() noexcept;

// Restore the snapshotted initial CPU affinity mask on the calling thread
// (e.g., background I/O threads so they never compete with OMP thread 0).
void gcta_restore_initial_affinity() noexcept;

// Number of cpuset-allowed NUMA memory nodes (>= 1 on Linux; 1 on macOS).
int gcta_numa_node_count() noexcept;

// Apply MPOL_INTERLEAVE across all cpuset-allowed NUMA nodes for interior
// full pages of [ptr, ptr + bytes) when bytes >= 64 MiB, allowed nodes > 1,
// and GCTA_DISABLE_NUMA is not set. Silent no-op otherwise.
void gcta_numa_interleave(void* ptr, std::size_t bytes) noexcept;

// Internal raw allocator used by make_numa_interleaved_uninitialized:
// allocates uninitialized aligned storage via posix_memalign (rounded up to
// page alignment and page-multiple length when bytes >= 64 MiB so mbind never
// spans a partial page owned by another heap allocation) and applies
// gcta_numa_interleave before any page is faulted. Free with std::free.
void* gcta_alloc_numa_interleaved(std::size_t bytes, std::size_t align) noexcept;

// POSIX file page-cache hints (no-op on macOS or invalid fd).
void gcta_fadvise_dontneed(int fd, off_t off = 0, off_t len = 0) noexcept;
void gcta_fadvise_sequential(int fd) noexcept;
void gcta_fadvise_random(int fd) noexcept;

// ---------------------------------------------------------------------------
// AlignedUniquePtr<T, Align> / make_aligned_uninitialized<T, Align> /
// make_numa_interleaved_uninitialized<T, Align>
//
// RAII owner for large uninitialized SIMD/cache-line-aligned buffers.
// Backed by posix_memalign / std::free so both standard and page-aligned
// NUMA-interleaved allocations share one deleter without allocator mismatch.
// Does NOT zero-fill or touch allocated pages, preserving OS lazy demand
// paging (critical for triangular/tiled GRM buffers where the upper triangle
// must stay un-faulted in physical RAM).
// ---------------------------------------------------------------------------
template <typename T, std::size_t Align = 64>
struct AlignedDeleter {
    static_assert(Align >= alignof(T) && std::has_single_bit(Align),
                  "Alignment must be a power of two >= alignof(T)");
    static_assert(std::is_trivially_destructible_v<T>,
                  "AlignedDeleter requires trivially destructible T");

    void operator()(T* p) const noexcept {
        if (p) {
            std::free(p);
        }
    }
};

template <typename T, std::size_t Align = 64>
using AlignedUniquePtr = std::unique_ptr<T[], AlignedDeleter<T, Align>>;

template <typename T, std::size_t Align = 64>
[[nodiscard]] AlignedUniquePtr<T, Align> make_aligned_uninitialized(std::size_t count) noexcept {
    static_assert(std::is_trivially_default_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>,
                  "make_aligned_uninitialized requires trivial T");
    if (count == 0) return nullptr;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) return nullptr;
    const std::size_t bytes = count * sizeof(T);
    const std::size_t eff_align = Align < sizeof(void*) ? sizeof(void*) : Align;
    void* p = nullptr;
    if (posix_memalign(&p, eff_align, bytes) != 0) return nullptr;
    return AlignedUniquePtr<T, Align>{static_cast<T*>(p)};
}

template <typename T, std::size_t Align = 64>
[[nodiscard]] AlignedUniquePtr<T, Align> make_numa_interleaved_uninitialized(std::size_t count) noexcept {
    static_assert(std::is_trivially_default_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>,
                  "make_numa_interleaved_uninitialized requires trivial T");
    if (count == 0) return nullptr;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) return nullptr;
    void* p = gcta_alloc_numa_interleaved(count * sizeof(T), Align);
    return AlignedUniquePtr<T, Align>{static_cast<T*>(p)};
}

// ---------------------------------------------------------------------------
// AlignedRawBuffer<T, Align>
//
// Lightweight RAII uninitialized aligned buffer with std::vector-like
// .data()/.size()/.resize()/.clear() plus .reset() and optional NUMA
// interleaving. Never zero-fills on resize (eliminating serial Core-0 page
// faults on multi-GB I/O or scratch buffers). Note: growing beyond capacity()
// reallocates uninitialized storage and does NOT preserve previous contents.
// ---------------------------------------------------------------------------
template <typename T, std::size_t Align = 64>
class AlignedRawBuffer {
    static_assert(Align >= alignof(T) && std::has_single_bit(Align),
                  "Alignment must be a power of two >= alignof(T)");
    static_assert(std::is_trivially_default_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>,
                  "AlignedRawBuffer requires trivial T");

public:
    using value_type = T;

    AlignedRawBuffer() noexcept = default;
    explicit AlignedRawBuffer(bool interleave) noexcept : interleave_(interleave) {}
    explicit AlignedRawBuffer(std::size_t count, bool interleave = false)
        : interleave_(interleave) {
        resize(count);
    }

    AlignedRawBuffer(const AlignedRawBuffer&) = delete;
    AlignedRawBuffer& operator=(const AlignedRawBuffer&) = delete;

    AlignedRawBuffer(AlignedRawBuffer&& other) noexcept
        : ptr_(std::move(other.ptr_)),
          size_(std::exchange(other.size_, 0)),
          capacity_(std::exchange(other.capacity_, 0)),
          interleave_(other.interleave_) {}

    AlignedRawBuffer& operator=(AlignedRawBuffer&& other) noexcept {
        if (this != &other) {
            ptr_ = std::move(other.ptr_);
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, 0);
            interleave_ = other.interleave_;
        }
        return *this;
    }

    ~AlignedRawBuffer() = default;

    [[nodiscard]] T* data() noexcept { return ptr_.get(); }
    [[nodiscard]] const T* data() const noexcept { return ptr_.get(); }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    T& operator[](std::size_t i) noexcept { return ptr_[i]; }
    const T& operator[](std::size_t i) const noexcept { return ptr_[i]; }

    T* begin() noexcept { return ptr_.get(); }
    const T* begin() const noexcept { return ptr_.get(); }
    T* end() noexcept { return ptr_.get() + size_; }
    const T* end() const noexcept { return ptr_.get() + size_; }

    void set_interleave(bool interleave) noexcept { interleave_ = interleave; }
    [[nodiscard]] bool interleave() const noexcept { return interleave_; }

    void resize(std::size_t n) {
        if (n <= capacity_) {
            size_ = n;
            return;
        }
        // Release old allocation first to avoid transient 2x peak RSS.
        ptr_.reset();
        size_ = 0;
        capacity_ = 0;

        AlignedUniquePtr<T, Align> fresh = interleave_
            ? make_numa_interleaved_uninitialized<T, Align>(n)
            : make_aligned_uninitialized<T, Align>(n);
        if (!fresh) {
            throw std::bad_alloc();
        }
        ptr_ = std::move(fresh);
        size_ = n;
        capacity_ = n;
    }

    void clear() noexcept { size_ = 0; }

    void reset() noexcept {
        ptr_.reset();
        size_ = 0;
        capacity_ = 0;
    }

private:
    AlignedUniquePtr<T, Align> ptr_;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
    bool interleave_ = false;
};

// These functions are only for test purpose, don't forget to remove calls
int getVMemKB();
int getMemKB();

int getVMPeakKB();
int getMemPeakKB();

#endif //GCTA2_MEM_HPP


