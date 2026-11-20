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
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

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
// AlignedUniquePtr<T, Align> / make_aligned_uninitialized<T, Align>
//
// RAII owner for large uninitialized SIMD/cache-line-aligned buffers.
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
            ::operator delete(p, std::align_val_t{Align});
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
    void* p = ::operator new(count * sizeof(T), std::align_val_t{Align}, std::nothrow);
    return AlignedUniquePtr<T, Align>{static_cast<T*>(p)};
}

// These functions are only for test purpose, don't forget to remove calls
int getVMemKB();
int getMemKB();

int getVMPeakKB();
int getMemPeakKB();

#endif //GCTA2_MEM_HPP


