cmake_minimum_required(VERSION 4.5)

set(GCTA_GCC_TOOLCHAIN "" CACHE PATH
    "GCC installation prefix for Clang/AOCC on Linux (auto-detected from g++ on PATH if unset)")

# Clang/AOCC on Linux delegates C++ headers, libstdc++, and libgcc_s to a host
# GCC installation, and only searches /usr unless --gcc-toolchain=<prefix> is
# passed. Inject it before project() so CMake's initial compiler link test finds
# -lstdc++ on HPC systems where GCC lives in modules or scratch directories.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    if(DEFINED CMAKE_CXX_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL "")
        set(_pre_cxx_candidate "${CMAKE_CXX_COMPILER}")
    elseif(NOT "$ENV{CXX}" STREQUAL "")
        set(_pre_cxx_candidate "$ENV{CXX}")
    else()
        set(_pre_cxx_candidate "c++")
    endif()
    find_program(_pre_cxx_prog NAMES "${_pre_cxx_candidate}")
    if(_pre_cxx_prog)
        execute_process(
            COMMAND "${_pre_cxx_prog}" --version
            OUTPUT_VARIABLE _pre_cxx_ver
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_pre_cxx_ver MATCHES "(AOCC|[Cc]lang)")
            if(NOT GCTA_GCC_TOOLCHAIN)
                find_program(_path_gxx NAMES g++ g++-16 g++-15 g++-14)
                if(_path_gxx)
                    file(REAL_PATH "${_path_gxx}" _path_gxx_real)
                    cmake_path(GET _path_gxx_real PARENT_PATH _path_gxx_bin)
                    cmake_path(GET _path_gxx_bin  PARENT_PATH _path_gxx_prefix)
                    if(_path_gxx_prefix AND NOT _path_gxx_prefix STREQUAL "/usr"
                                        AND NOT _path_gxx_prefix STREQUAL "/")
                        set(GCTA_GCC_TOOLCHAIN "${_path_gxx_prefix}" CACHE PATH
                            "GCC installation prefix for Clang/AOCC on Linux (auto-detected from g++ on PATH if unset)" FORCE)
                        message(STATUS "Pre-project: auto-detected GCTA_GCC_TOOLCHAIN=${GCTA_GCC_TOOLCHAIN} (from ${_path_gxx})")
                    endif()
                endif()
            endif()

            set(_tc_arg "")
            if(GCTA_GCC_TOOLCHAIN)
                if(NOT IS_DIRECTORY "${GCTA_GCC_TOOLCHAIN}")
                    message(FATAL_ERROR "GCTA_GCC_TOOLCHAIN does not exist: ${GCTA_GCC_TOOLCHAIN}")
                endif()
                set(_tc_arg "--gcc-toolchain=${GCTA_GCC_TOOLCHAIN}")
                foreach(_v IN ITEMS CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
                    if(NOT "${${_v}}" MATCHES "(^| )--gcc-toolchain=")
                        string(STRIP "${${_v}} ${_tc_arg}" _updated_flags)
                        set(${_v} "${_updated_flags}" CACHE STRING "" FORCE)
                    endif()
                endforeach()
                message(STATUS "Pre-project: using ${_tc_arg} for Clang/AOCC")
            endif()

            if(NOT CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
                set(_stdcxx_probe_cmd "${_pre_cxx_prog}")
                if(_tc_arg)
                    list(APPEND _stdcxx_probe_cmd "${_tc_arg}")
                endif()
                list(APPEND _stdcxx_probe_cmd "-print-file-name=libstdc++.so")
                execute_process(
                    COMMAND ${_stdcxx_probe_cmd}
                    OUTPUT_VARIABLE _stdcxx_probe
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                if(NOT _stdcxx_probe OR _stdcxx_probe STREQUAL "libstdc++.so")
                    message(FATAL_ERROR
                        "${_pre_cxx_prog} could not locate libstdc++.so (got '${_stdcxx_probe}').\n"
                        "On Linux, Clang/AOCC requires a GCC installation for libstdc++ and C++ headers, "
                        "and does not search PATH automatically.\n"
                        "Load a GCC module (e.g. 'module load gcc/14' or newer) so g++ is on PATH, "
                        "or pass -DGCTA_GCC_TOOLCHAIN=/path/to/gcc-prefix.")
                endif()
            endif()
        endif()
    endif()
    unset(_pre_cxx_prog CACHE)
    unset(_path_gxx CACHE)
endif()

project(GCTAg LANGUAGES C CXX)

# Prevent CMake from finding stale configs in build-tree or user package registries.
# This is especially important on HPC clusters where previous partial builds can
# leave broken zstd-config.cmake or other config files in the build directory.
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY        TRUE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY TRUE)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

macro(gcta_add_rpath)
    foreach(_gcta_rpath_dir IN ITEMS ${ARGN})
        list(APPEND CMAKE_BUILD_RPATH   "${_gcta_rpath_dir}")
        list(APPEND CMAKE_INSTALL_RPATH "${_gcta_rpath_dir}")
    endforeach()
endmacro()

function(gcta_pin_compiler_rtlibs _label)
    set(_tc_arg "")
    if(GCTA_GCC_TOOLCHAIN)
        set(_tc_arg "--gcc-toolchain=${GCTA_GCC_TOOLCHAIN}")
    endif()
    foreach(_rtlib IN LISTS ARGN)
        execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" ${_tc_arg} -print-file-name=${_rtlib}
            OUTPUT_VARIABLE _rtlib_path
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_rtlib_path AND NOT _rtlib_path STREQUAL "${_rtlib}")
            file(REAL_PATH "${_rtlib_path}" _rtlib_real)
            cmake_path(GET _rtlib_real PARENT_PATH _rtlib_dir)
            gcta_add_rpath("${_rtlib_dir}")
            message(STATUS "${_label}: ${_rtlib} -> ${_rtlib_real}")
        elseif(NOT _rtlib STREQUAL "libgomp.so.1")
            message(WARNING "${_label}: -print-file-name=${_rtlib} did not resolve")
        endif()
    endforeach()
    set(CMAKE_BUILD_RPATH   "${CMAKE_BUILD_RPATH}"   PARENT_SCOPE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" PARENT_SCOPE)
endfunction()

function(gcta_detect_blas_version _backend _blas_lib _blas_include_dir _out_var)
    # Relies on CMP0159=NEW so file(STRINGS ... REGEX) populates CMAKE_MATCH_<n>.
    set(_ver "unknown")

    if(_backend STREQUAL "OpenBLAS" AND EXISTS "${_blas_include_dir}/openblas_config.h")
        file(STRINGS "${_blas_include_dir}/openblas_config.h" _line LIMIT_COUNT 1
            REGEX "^#define[ \t]+OPENBLAS_VERSION[ \t]+\"([^\"]+)\"")
        if(_line)
            set(_ver "${CMAKE_MATCH_1}")
        endif()
    elseif(_backend STREQUAL "MKL" AND EXISTS "${_blas_include_dir}/mkl_version.h")
        set(_mkl_ver_h "${_blas_include_dir}/mkl_version.h")
        file(STRINGS "${_mkl_ver_h}" _l1 LIMIT_COUNT 1 REGEX "^#define[ \t]+__INTEL_MKL__[ \t]+([0-9]+)")
        if(_l1)
            set(_mkl_maj "${CMAKE_MATCH_1}")
        endif()
        file(STRINGS "${_mkl_ver_h}" _l2 LIMIT_COUNT 1 REGEX "^#define[ \t]+__INTEL_MKL_MINOR__[ \t]+([0-9]+)")
        if(_l2)
            set(_mkl_min "${CMAKE_MATCH_1}")
        endif()
        file(STRINGS "${_mkl_ver_h}" _l3 LIMIT_COUNT 1 REGEX "^#define[ \t]+__INTEL_MKL_UPDATE__[ \t]+([0-9]+)")
        if(_l3)
            set(_mkl_upd "${CMAKE_MATCH_1}")
        endif()
        if(DEFINED _mkl_maj AND DEFINED _mkl_min AND DEFINED _mkl_upd)
            set(_ver "${_mkl_maj}.${_mkl_min}.${_mkl_upd}")
        endif()
    elseif(_backend STREQUAL "AOCL")
        if(EXISTS "${_blas_include_dir}/blis.h")
            file(STRINGS "${_blas_include_dir}/blis.h" _line LIMIT_COUNT 1
                REGEX "^#define[ \t]+BLIS_VERSION_STRING[ \t]+\"([^\"]+)\"")
            if(_line)
                set(_ver "${CMAKE_MATCH_1}")
            endif()
        endif()
        if(_ver STREQUAL "unknown" AND _blas_lib)
            cmake_path(GET _blas_lib PARENT_PATH _blas_lib_dir)
            foreach(_manifest IN ITEMS "${_blas_lib_dir}/aocl_manifest.txt" "${_blas_lib_dir}/../aocl_manifest.txt")
                if(EXISTS "${_manifest}")
                    file(STRINGS "${_manifest}" _line LIMIT_COUNT 1
                        REGEX "(AOCL|Version|version)[^0-9]*([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
                    if(_line)
                        set(_ver "AOCL ${CMAKE_MATCH_2}")
                        break()
                    endif()
                endif()
            endforeach()
        endif()
    elseif(_backend STREQUAL "Accelerate")
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-version
            OUTPUT_VARIABLE _macos_sdk_ver
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_macos_sdk_ver)
            set(_ver "SDK ${_macos_sdk_ver}")
        endif()
    endif()

    if(_ver STREQUAL "unknown" AND _blas_lib)
        cmake_path(GET _blas_lib FILENAME _blas_name)
        if(_blas_name MATCHES "\\.so\\.([0-9][0-9.]*)$")
            set(_ver "soname ${CMAKE_MATCH_1}")
        endif()
    endif()

    set(${_out_var} "${_ver}" PARENT_SCOPE)
endfunction()

set(GCTA_COMPILER_IS_AOCC FALSE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(FATAL_ERROR
        "AppleClang is not supported. On macOS, use Homebrew LLVM (Clang >= 23) or GCC >= 16.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Compiling with GCC ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16.0)
        message(FATAL_ERROR "GCTAg requires GCC >= 16 (found ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
    if(LINUX)
        # Pin libgomp, libstdc++, and libgcc_s to this GCC's copies so HPC
        # LD_LIBRARY_PATH cannot load incompatible system copies at runtime.
        gcta_pin_compiler_rtlibs("GCC" libgomp.so.1 libstdc++.so.6 libgcc_s.so.1)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Detect AOCC (AMD's Clang-based compiler) from --version since CMake reports ID "Clang".
    if(LINUX)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --version
            OUTPUT_VARIABLE _cxx_ver_str
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_cxx_ver_str MATCHES "AOCC")
            set(GCTA_COMPILER_IS_AOCC TRUE)
            if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 21.0)
                message(FATAL_ERROR
                    "GCTAg requires AOCC >= 6.0 (Clang >= 21) (found Clang ${CMAKE_CXX_COMPILER_VERSION} in ${_cxx_ver_str})")
            endif()
            cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _aocc_bin)
            cmake_path(GET _aocc_bin          PARENT_PATH _aocc_root)
            set(GCTA_AOCC_ROOT "${_aocc_root}" CACHE PATH "AOCC installation root" FORCE)
            message(STATUS "Detected AOCC compiler (Clang ${CMAKE_CXX_COMPILER_VERSION}, root: ${GCTA_AOCC_ROOT})")
            gcta_pin_compiler_rtlibs("AOCC" libomp.so libstdc++.so.6 libgcc_s.so.1)
        endif()
    endif()

    if(NOT GCTA_COMPILER_IS_AOCC)
        message(STATUS "Compiling with Clang ${CMAKE_CXX_COMPILER_VERSION}")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23.0)
            message(FATAL_ERROR "GCTAg requires Clang >= 23 (found ${CMAKE_CXX_COMPILER_VERSION})")
        endif()
        if(APPLE)
            cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _llvm_bin)
            cmake_path(GET _llvm_bin          PARENT_PATH _llvm_prefix)
            if(EXISTS "${_llvm_prefix}/lib/c++")
                link_directories("${_llvm_prefix}/lib/c++")
                gcta_add_rpath("${_llvm_prefix}/lib/c++" "${_llvm_prefix}/lib")
                message(STATUS "Using LLVM libc++ / libomp from ${_llvm_prefix}")
            endif()
        endif()
    endif()
else()
    message(FATAL_ERROR "GCTAg requires GCC >= 16, Clang >= 23, or AOCC >= 6.0 (found ${CMAKE_CXX_COMPILER_ID})")
endif()

# Use the compiler's matched archiver/ranlib for static targets (Pgenlib, zstd, sqlite3)
# so LTO archives stay consistent with the compiler toolchain.
if(CMAKE_CXX_COMPILER_AR)
    set(CMAKE_AR "${CMAKE_CXX_COMPILER_AR}")
endif()
if(CMAKE_CXX_COMPILER_RANLIB)
    set(CMAKE_RANLIB "${CMAKE_CXX_COMPILER_RANLIB}")
endif()

# On macOS, Homebrew Clang/GCC may default to CommandLineTools' SDK while
# xcrun (_CMAKE_OSX_SYSROOT_PATH) resolves to Xcode.app's SDK; register both
# as implicit include dirs so FindZLIB does not inject -isystem <sdk>/usr/include
# ahead of the compiler's C++ standard library headers.
if(APPLE AND _CMAKE_OSX_SYSROOT_PATH)
    list(APPEND CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES   "${_CMAKE_OSX_SYSROOT_PATH}/usr/include")
    list(APPEND CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "${_CMAKE_OSX_SYSROOT_PATH}/usr/include")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Defaulting to Release build")
endif()

# Release-build arch flags are applied per-target on GCTAg below,
# after add_executable, so they don't bleed into FetchContent dependencies.

set(GCTA_BLAS_BACKEND "" CACHE STRING "BLAS/LAPACK backend (required): OpenBLAS, MKL, Accelerate, or AOCL")
set_property(CACHE GCTA_BLAS_BACKEND PROPERTY STRINGS OpenBLAS MKL Accelerate AOCL)
set(GCTA_BLAS_LIBRARY "" CACHE FILEPATH "Primary BLAS .so/.a for the selected backend (OpenBLAS/MKL/AOCL).")
set(GCTA_LAPACKE_LIBRARY "" CACHE FILEPATH "Separate LAPACKE .so/.a, if not bundled in GCTA_BLAS_LIBRARY (e.g. apt OpenBLAS). Leave empty for MKL/conda.")
set(GCTA_BLAS_INCLUDE_DIR "" CACHE PATH "Directory containing cblas.h / mkl.h / blis.h for the selected backend.")
set(Boost_DIR "" CACHE PATH "Boost CMake config dir (BoostConfig.cmake). If set, uses system Boost instead of fetching.")
option(GCTA_UNITY_BUILD "Enable unity builds for faster compilation" ON)
option(GCTA_ENABLE_IPO "Enable IPO/LTO in Release builds" ON)
option(BGEN_SUPPORT "Enable BGEN file format support (requires SQLite)" OFF)
set(USE_JEMALLOC "" CACHE STRING
    "Path to libjemalloc.so / libamdalloc.so (or ON/amdalloc with AOCC to auto-select bundled libamdalloc.so). Links with thp:never to avoid Linux THP over-allocation on large triangular allocations.")


# Initialize plink-ng submodule if needed
if(EXISTS "${CMAKE_SOURCE_DIR}/.git" AND NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/plink-ng/2.0/include/pgenlib_ffi_support.h")
    find_package(Git QUIET)
    if(GIT_FOUND)
        message(STATUS "Initializing plink-ng submodule...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMAND_ERROR_IS_FATAL ANY)
    else()
        message(FATAL_ERROR "plink-ng submodule not initialized. Please run: git submodule update --init --recursive")
    endif()
endif()

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/plink-ng/2.0/include/pgenlib_ffi_support.h")
    message(FATAL_ERROR "plink-ng submodule is missing. Please run: git submodule update --init --recursive")
endif()

# Fetch 3rd parties
include(FetchContent)
# Once all tarballs are cached locally, skip re-checking download URLs on
# every re-configure.  This is faster and works on HPC compute nodes that
# have no outbound internet.  Override with -DFETCHCONTENT_UPDATES_DISCONNECTED=OFF.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Skip re-checking FetchContent URLs on re-configure (set OFF to force update check)")
if(CMAKE_VERBOSE_MAKEFILE)
    set(FETCHCONTENT_QUIET OFF)
else()
    set(FETCHCONTENT_QUIET ON)
endif()

FetchContent_Declare(
    eigen
    URL https://github.com/eigen-mirror/eigen/archive/refs/tags/5.0.1.tar.gz
    SYSTEM)
FetchContent_Declare(
    spectra
    URL https://github.com/yixuan/spectra/archive/refs/tags/v1.2.0.tar.gz
    SYSTEM)
FetchContent_Declare(
    zstd
    URL https://github.com/facebook/zstd/archive/refs/tags/v1.5.7.tar.gz
    SOURCE_SUBDIR "build/cmake"
    SYSTEM
    EXCLUDE_FROM_ALL)
FetchContent_Declare(
    Boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-cmake.tar.xz
    SYSTEM
    EXCLUDE_FROM_ALL)
if(BGEN_SUPPORT)
    FetchContent_Declare(
        sqlite
        URL https://www.sqlite.org/2026/sqlite-amalgamation-3510200.zip
        EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
    stdexec
    URL https://github.com/NVIDIA/stdexec/archive/refs/tags/nvhpc-26.05.tar.gz
    SYSTEM
    EXCLUDE_FROM_ALL)

# Always use the system shared zlib — it is a base system library with a stable
# ABI (libz.so.1 since 1997) and is always present on HPC clusters. This avoids
# the CMAKE_SKIP_INSTALL_RULES / alias dance required by a fetched static zlib.
find_package(ZLIB REQUIRED)

block()
    set(ZSTD_BUILD_PROGRAMS OFF)
    set(ZSTD_BUILD_TESTS    OFF)
    set(ZSTD_BUILD_SHARED   OFF)
    set(ZSTD_BUILD_STATIC   ON)
    set(SPECTRA_BUILD_TESTS OFF)
    set(BUILD_TESTING       OFF)
    set(BOOST_IOSTREAMS_ENABLE_ZLIB  ON  CACHE BOOL "" FORCE)
    set(BOOST_IOSTREAMS_ENABLE_BZIP2 OFF CACHE BOOL "" FORCE)
    set(BOOST_IOSTREAMS_ENABLE_LZMA  OFF CACHE BOOL "" FORCE)
    set(BOOST_IOSTREAMS_ENABLE_ZSTD  OFF CACHE BOOL "" FORCE)
    if(Boost_DIR)
        FetchContent_MakeAvailable(zstd)
    else()
        # These vars are only meaningful to Boost's own CMake build system
        set(BOOST_ENABLE_CMAKE ON)
        set(BOOST_ENABLE_COMPATIBILITY_TARGETS ON)
        set(BOOST_INCLUDE_LIBRARIES "algorithm;math;crc;iostreams")
        FetchContent_MakeAvailable(zstd Boost)
    endif()
endblock()

# Eigen and Spectra have entangled install rules (Spectra exports reference
# Eigen targets) that break with EXCLUDE_FROM_ALL. Suppress via
# CMAKE_SKIP_INSTALL_RULES instead, which must be set in the parent scope.
set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_MakeAvailable(eigen spectra)
set(CMAKE_SKIP_INSTALL_RULES OFF)

# sqlite has no options to isolate, and sqlite_SOURCE_DIR must be visible
# in the parent scope for the add_library(sqlite3) call below.
if(BGEN_SUPPORT)
    FetchContent_MakeAvailable(sqlite)
endif()
FetchContent_MakeAvailable(stdexec)

# If the user supplied a Boost_DIR, prefer that over fetching Boost.
if(Boost_DIR)
    message(STATUS "Using external Boost from ${Boost_DIR}")
    find_package(Boost REQUIRED COMPONENTS iostreams
        CONFIG
        PATHS "${Boost_DIR}"
        NO_DEFAULT_PATH)
endif()

if(BGEN_SUPPORT)
    add_library(sqlite3 STATIC ${sqlite_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3 PUBLIC ${sqlite_SOURCE_DIR})
endif()

find_package(Threads REQUIRED)

if(APPLE AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(FATAL_ERROR "GCTAg on macOS requires Apple Silicon (arm64); found ${CMAKE_SYSTEM_PROCESSOR}.")
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

set(GCTA_BLAS_LIBS "")
set(GCTA_BLAS_INCLUDE_DIRS "")
set(GCTA_BLAS_DEFINES "")
set(GCTA_LAPACKE_IS_MKL FALSE)

if(APPLE AND NOT GCTA_BLAS_BACKEND)
    set(GCTA_BLAS_BACKEND "Accelerate")
endif()

set(_gcta_valid_blas_backends OpenBLAS MKL Accelerate AOCL)
if(NOT GCTA_BLAS_BACKEND IN_LIST _gcta_valid_blas_backends)
    message(FATAL_ERROR
        "GCTA_BLAS_BACKEND='${GCTA_BLAS_BACKEND}' is invalid or unset. "
        "Choose one of: OpenBLAS, MKL, Accelerate, AOCL.\n"
        "Linux:  -DGCTA_BLAS_BACKEND=AOCL -DGCTA_BLAS_LIBRARY=<...> -DGCTA_BLAS_INCLUDE_DIR=<...>\n"
        "macOS:  -DGCTA_BLAS_BACKEND=Accelerate")
endif()

# Validate platform constraints
if(APPLE AND NOT GCTA_BLAS_BACKEND STREQUAL "Accelerate")
    message(FATAL_ERROR "On macOS, only GCTA_BLAS_BACKEND=Accelerate is supported (found '${GCTA_BLAS_BACKEND}').")
elseif(NOT APPLE AND GCTA_BLAS_BACKEND STREQUAL "Accelerate")
    message(FATAL_ERROR "Accelerate is only available on macOS. Use AOCL, MKL, or OpenBLAS.")
endif()

if(GCTA_BLAS_BACKEND STREQUAL "Accelerate")
    list(APPEND GCTA_BLAS_LIBS    "-framework Accelerate")
    list(APPEND GCTA_BLAS_DEFINES GCTA_USE_ACCELERATE ACCELERATE_NEW_LAPACK EIGEN_USE_BLAS)
    set(GCTA_BLAS_PROVIDER "Accelerate" CACHE INTERNAL "")

else()
    # OpenBLAS, MKL, and AOCL all require the user to supply explicit paths.
    # This avoids the fragile FindBLAS / BLA_VENDOR linker-probe machinery.
    # For MKL, pass libmkl_rt.so (the single-library interface).
    #   With GCC:  set MKL_THREADING_LAYER=GNU at runtime to use libgomp.
    #   With AOCC: set MKL_THREADING_LAYER=IOMP5 at runtime to use libomp.
    # For AOCL, use the gcc/ library variant with GCC and the aocc/ variant with AOCC.
    if(NOT GCTA_BLAS_LIBRARY)
        message(FATAL_ERROR
            "GCTA_BLAS_LIBRARY must be set for the ${GCTA_BLAS_BACKEND} backend "
            "(path to the primary .so/.a, e.g. libopenblas.so, libmkl_rt.so, libblis-mt.so).")
    endif()
    if(NOT GCTA_BLAS_INCLUDE_DIR)
        message(FATAL_ERROR
            "GCTA_BLAS_INCLUDE_DIR must be set for the ${GCTA_BLAS_BACKEND} backend "
            "(directory containing cblas.h / mkl.h / blis.h).")
    endif()
    if(NOT EXISTS "${GCTA_BLAS_LIBRARY}")
        message(FATAL_ERROR "GCTA_BLAS_LIBRARY does not exist: ${GCTA_BLAS_LIBRARY}")
    endif()
    if(NOT IS_DIRECTORY "${GCTA_BLAS_INCLUDE_DIR}")
        message(FATAL_ERROR "GCTA_BLAS_INCLUDE_DIR does not exist: ${GCTA_BLAS_INCLUDE_DIR}")
    endif()

    message(STATUS "${GCTA_BLAS_BACKEND} library:     ${GCTA_BLAS_LIBRARY}")
    if(GCTA_LAPACKE_LIBRARY)
        message(STATUS "${GCTA_BLAS_BACKEND} LAPACKE library: ${GCTA_LAPACKE_LIBRARY}")
    endif()
    message(STATUS "${GCTA_BLAS_BACKEND} include dir: ${GCTA_BLAS_INCLUDE_DIR}")

    if(LINUX)
        cmake_path(GET GCTA_BLAS_LIBRARY PARENT_PATH _blas_lib_dir)
        gcta_add_rpath("${_blas_lib_dir}")
        add_link_options("LINKER:-rpath-link,${_blas_lib_dir}")
        if(GCTA_LAPACKE_LIBRARY)
            cmake_path(GET GCTA_LAPACKE_LIBRARY PARENT_PATH _lapack_lib_dir)
            gcta_add_rpath("${_lapack_lib_dir}")
            add_link_options("LINKER:-rpath-link,${_lapack_lib_dir}")
        endif()
    endif()

    list(APPEND GCTA_BLAS_INCLUDE_DIRS "${GCTA_BLAS_INCLUDE_DIR}")
    list(APPEND GCTA_BLAS_LIBS         "${GCTA_BLAS_LIBRARY}")
    if(GCTA_LAPACKE_LIBRARY)
        if(NOT EXISTS "${GCTA_LAPACKE_LIBRARY}")
            message(FATAL_ERROR "GCTA_LAPACKE_LIBRARY does not exist: ${GCTA_LAPACKE_LIBRARY}")
        endif()
        list(APPEND GCTA_BLAS_LIBS "${GCTA_LAPACKE_LIBRARY}")
        cmake_path(GET GCTA_LAPACKE_LIBRARY FILENAME _lapacke_lib_name)
        string(TOLOWER "${_lapacke_lib_name}" _lapacke_lib_lower)
        if(_lapacke_lib_lower MATCHES "mkl")
            set(GCTA_LAPACKE_IS_MKL TRUE)
            list(APPEND GCTA_BLAS_DEFINES GCTA_LAPACKE_IS_MKL)
        endif()
    endif()

    if(GCTA_BLAS_BACKEND STREQUAL "OpenBLAS")
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_OPENBLAS EIGEN_USE_LAPACKE EIGEN_USE_BLAS)
        set(GCTA_BLAS_PROVIDER "OpenBLAS" CACHE INTERNAL "")
    elseif(GCTA_BLAS_BACKEND STREQUAL "AOCL")
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_AOCL EIGEN_USE_LAPACKE EIGEN_USE_BLAS)
        set(GCTA_BLAS_PROVIDER "AOCL" CACHE INTERNAL "")

        # AOCL ships separate aocc/ (libomp) and gcc/ (libgomp) builds; catch mismatches early.
        cmake_path(GET GCTA_BLAS_LIBRARY PARENT_PATH _aocl_lib_dir)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND EXISTS "${_aocl_lib_dir}/libomp.so")
            message(FATAL_ERROR
                "AOCL variant mismatch: GCC compiler but AOCC-built AOCL (libomp.so present) at ${_aocl_lib_dir}. "
                "Use the gcc/ AOCL variant.")
        endif()
        if(GCTA_COMPILER_IS_AOCC AND NOT EXISTS "${_aocl_lib_dir}/libomp.so"
                                 AND EXISTS "${_aocl_lib_dir}/libgomp.so")
            message(WARNING
                "AOCL variant mismatch: AOCC compiler but GCC-built AOCL (no libomp.so) at ${_aocl_lib_dir}. "
                "Consider using the aocc/ variant for best performance.")
        endif()
    else()
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_MKL EIGEN_USE_MKL_ALL)
        set(GCTA_BLAS_PROVIDER "MKL" CACHE INTERNAL "")
    endif()
endif()

message(STATUS "BLAS/LAPACK provider: ${GCTA_BLAS_PROVIDER}")
gcta_detect_blas_version("${GCTA_BLAS_PROVIDER}" "${GCTA_BLAS_LIBRARY}" "${GCTA_BLAS_INCLUDE_DIR}" GCTA_BLAS_VERSION)
message(STATUS "BLAS/LAPACK version: ${GCTA_BLAS_VERSION}")

# jemalloc / amdalloc (optional): interposes malloc/free/new/delete via linking alone —
# no header/API usage, so only the .so path is needed. thp:never is applied
# via a malloc_conf symbol defined in src/main.cpp, gated on GCTA_USE_JEMALLOC.
set(GCTA_USE_JEMALLOC FALSE)
if(USE_JEMALLOC)
    string(TOLOWER "${USE_JEMALLOC}" _jemalloc_lower)
    if(_jemalloc_lower MATCHES "^(on|true|1|yes|amdalloc)$")
        if(NOT GCTA_COMPILER_IS_AOCC)
            message(FATAL_ERROR
                "USE_JEMALLOC=${USE_JEMALLOC} auto-selection is only supported with AOCC (bundled libamdalloc.so). "
                "Pass the explicit path to libjemalloc.so instead.")
        endif()
        unset(_aocc_amdalloc CACHE)
        find_library(_aocc_amdalloc NAMES amdalloc
            HINTS "${GCTA_AOCC_ROOT}/lib" "${GCTA_AOCC_ROOT}/lib64" NO_DEFAULT_PATH)
        if(NOT _aocc_amdalloc)
            message(FATAL_ERROR
                "USE_JEMALLOC=${USE_JEMALLOC} with AOCC could not find libamdalloc.so under "
                "${GCTA_AOCC_ROOT}/lib or ${GCTA_AOCC_ROOT}/lib64. Pass the explicit .so path instead.")
        endif()
        set(_jemalloc_target "${_aocc_amdalloc}")
    else()
        set(_jemalloc_target "${USE_JEMALLOC}")
    endif()

    if(NOT EXISTS "${_jemalloc_target}")
        message(FATAL_ERROR "USE_JEMALLOC=${USE_JEMALLOC} (${_jemalloc_target}) does not exist")
    endif()
    if(IS_DIRECTORY "${_jemalloc_target}")
        message(FATAL_ERROR
            "USE_JEMALLOC=${USE_JEMALLOC} is a directory — pass the libjemalloc.so/libamdalloc.so file itself, e.g. -DUSE_JEMALLOC=/opt/jemalloc/lib/libjemalloc.so")
    endif()
    file(REAL_PATH "${_jemalloc_target}" _jemalloc_realpath)
    if(NOT EXISTS "${_jemalloc_realpath}")
        message(FATAL_ERROR "USE_JEMALLOC=${USE_JEMALLOC} resolves to ${_jemalloc_realpath}, which does not exist (broken symlink?)")
    endif()
    set(GCTA_JEMALLOC_LIB "${_jemalloc_realpath}")
    set(GCTA_USE_JEMALLOC TRUE)
    if(LINUX)
        cmake_path(GET GCTA_JEMALLOC_LIB PARENT_PATH _jemalloc_dir)
        gcta_add_rpath("${_jemalloc_dir}")
    endif()
    message(STATUS "jemalloc: ${GCTA_JEMALLOC_LIB} (thp:never)")
endif()

add_executable(GCTAg)

target_sources(GCTAg PRIVATE
            # entry point
        src/main.cpp
        # src/ module
        src/Covar.cpp
        src/FastFAM.cpp
        src/MLMA_stream.cpp
        #src/MLMA_loco.cpp # Disabled as out of sync with MLMA_stream
        src/pca_stream.cpp
        src/RemlEngine.cpp
        src/GRM.cpp
        src/Geno.cpp
        src/BedBackend.cpp
        src/BgenBackend.cpp
        src/PgenBackend.cpp
        src/LD.cpp
        src/Logger.cpp
        src/Marker.cpp
        src/OptionIO.cpp
        src/Pheno.cpp
        src/StatLib.cpp
        src/ThreadPool.cpp
        src/mem.cpp
        src/tables.cpp
        src/utils.cpp
        # main/ legacy module
        main/CommFunc.cpp
        main/StatFunc.cpp
        main/StrFunc.cpp
        main/bivar_reml.cpp
        main/data.cpp
        main/edata.cpp
        main/ejma.cpp
        main/est_hsq.cpp
        main/gbat.cpp
        main/grm.cpp
        main/gsmr.cpp
        main/gwas_simu.cpp
        main/joint_meta.cpp
        main/ld.cpp
        main/LinAlg.cpp
        main/mbat.cpp
        main/mlm_assoc.cpp
        main/mtcojo.cpp
        main/option.cpp
        main/pc_adjust.cpp
        main/popu_genet.cpp
        main/raw_geno.cpp
        main/reml_within_family.cpp
        main/sbat.cpp)

target_include_directories(GCTAg PRIVATE
    ${CMAKE_SOURCE_DIR}
    include
    ${GCTA_BLAS_INCLUDE_DIRS}
    third_party/plink-ng/2.0/include
    third_party/plink-ng/2.0/simde)

target_precompile_headers(GCTAg PRIVATE
    <Eigen/Dense>
    <boost/algorithm/string.hpp>
    <boost/math/distributions.hpp>)

# ── GCTAg target properties ───────────────────────────────────────────────
# IPO check must run before set_target_properties.
set(GCTA_IPO_ENABLED OFF)
if(GCTA_ENABLE_IPO AND CMAKE_BUILD_TYPE MATCHES "^[Rr]elease$")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT GCTA_IPO_SUPPORTED OUTPUT GCTA_IPO_OUTPUT LANGUAGES CXX)
    if(GCTA_IPO_SUPPORTED)
        set(GCTA_IPO_ENABLED ON)
    else()
        message(STATUS "IPO/LTO not supported: ${GCTA_IPO_OUTPUT}")
    endif()
endif()

set_target_properties(GCTAg PROPERTIES
    UNITY_BUILD                   ${GCTA_UNITY_BUILD}
    INTERPROCEDURAL_OPTIMIZATION  ${GCTA_IPO_ENABLED})

target_compile_definitions(GCTAg PRIVATE
    ${GCTA_BLAS_DEFINES}
    GCTA_BLAS_BACKEND_STR="${GCTA_BLAS_PROVIDER}"
    GCTA_BLAS_VERSION_STR="${GCTA_BLAS_VERSION}"
    $<$<BOOL:${GCTA_USE_JEMALLOC}>:GCTA_USE_JEMALLOC>
    $<$<BOOL:${BGEN_SUPPORT}>:BGEN_SUPPORT>)

target_compile_options(GCTAg PRIVATE
    -pipe
    $<$<CONFIG:Release>:$<IF:$<BOOL:${APPLE}>,-mcpu=native,-march=native>>)

# Both GCC (GNU ld) and Clang (lld or GNU ld) default to DT_RUNPATH (--enable-new-dtags),
# which searches LD_LIBRARY_PATH before RPATH and breaks HPC libstdc++/BLAS pinning.
target_link_options(GCTAg PRIVATE
    $<$<BOOL:${LINUX}>:LINKER:--disable-new-dtags>)


add_library(Pgenlib STATIC)

target_sources(Pgenlib PRIVATE
            third_party/Pgenlib/PgenReader.cpp
        third_party/plink-ng/2.0/include/pgenlib_ffi_support.cc
        third_party/plink-ng/2.0/include/pgenlib_misc.cc
        third_party/plink-ng/2.0/include/pgenlib_read.cc
        third_party/plink-ng/2.0/include/pgenlib_write.cc
        third_party/plink-ng/2.0/include/plink2_base.cc
        third_party/plink-ng/2.0/include/plink2_simd.cc
        third_party/plink-ng/2.0/include/plink2_bgzf.cc
        third_party/plink-ng/2.0/include/plink2_bitmap.cc
        third_party/plink-ng/2.0/include/plink2_bits.cc
        third_party/plink-ng/2.0/include/plink2_fmath.cc
        third_party/plink-ng/2.0/include/plink2_htable.cc
        third_party/plink-ng/2.0/include/plink2_memory.cc
        third_party/plink-ng/2.0/include/plink2_stats.cc
        third_party/plink-ng/2.0/include/plink2_string.cc
        third_party/plink-ng/2.0/include/plink2_text.cc
        third_party/plink-ng/2.0/include/plink2_thread.cc
        third_party/plink-ng/2.0/include/plink2_zstfile.cc
        third_party/plink-ng/2.0/include/pvar_ffi_support.cc)
# Unity builds must be disabled for Pgenlib: it is a separate static library
# with its own include paths and must never be merged into a GCTAg unity TU.
# Without this, CMake can concatenate PgenReader.cpp and PgenBackend.cpp into
# the same translation unit, causing PgenReader.cpp to fail to find GCTA
# headers such as GenoBackendFactory.h.
set_target_properties(Pgenlib PROPERTIES UNITY_BUILD OFF)
target_include_directories(Pgenlib PUBLIC "${CMAKE_SOURCE_DIR}/third_party/plink-ng/2.0/include")
target_compile_options(Pgenlib PRIVATE
    $<$<CONFIG:Release>:$<IF:$<BOOL:${APPLE}>,-mcpu=native,-march=native>>)


target_link_libraries(GCTAg PRIVATE
    Boost::iostreams
    Boost::headers
    ZLIB::ZLIB
    Eigen3::Eigen
    Spectra
    STDEXEC::stdexec
    libzstd_static
    $<$<BOOL:${GCTA_USE_JEMALLOC}>:${GCTA_JEMALLOC_LIB}>
    OpenMP::OpenMP_CXX
    $<$<BOOL:${BGEN_SUPPORT}>:sqlite3>
    Pgenlib
    Threads::Threads
    ${GCTA_BLAS_LIBS})

# Remove any duplicate entries added by compiler probes, BLAS detection, and
# toolchain logic (e.g. BLAS lib dir may match OpenMP dir on some installs).
list(REMOVE_DUPLICATES CMAKE_BUILD_RPATH)
list(REMOVE_DUPLICATES CMAKE_INSTALL_RPATH)

# Installation
if(LINUX)
    # PREPEND so $ORIGIN/../lib is first but BLAS-specific paths appended
    # earlier (e.g. the conda OpenBLAS or MKL lib dir) are preserved.
    # Using set() here would overwrite those previously-appended entries.
    list(PREPEND CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
elseif(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
endif()

if(LINUX OR APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
    install(TARGETS GCTAg
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
endif()

# Build configuration summary
if(GCTA_COMPILER_IS_AOCC)
    set(GCTA_COMPILER_IS_AOCC_STR " (AOCC)")
else()
    set(GCTA_COMPILER_IS_AOCC_STR "")
endif()
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "GCTAg Build Configuration Summary")
message(STATUS "==========================================")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  BLAS/LAPACK:       ${GCTA_BLAS_PROVIDER} (${GCTA_BLAS_VERSION})")
if(GCTA_LAPACKE_LIBRARY)
    message(STATUS "  LAPACKE library:   ${GCTA_LAPACKE_LIBRARY}")
endif()
message(STATUS "  Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}${GCTA_COMPILER_IS_AOCC_STR} (${CMAKE_CXX_COMPILER})")
message(STATUS "  Archiver (AR):     ${CMAKE_AR}")
message(STATUS "  Ranlib:            ${CMAKE_RANLIB}")
message(STATUS "  IPO/LTO:           ${GCTA_IPO_ENABLED}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  OpenMP:            ${OpenMP_CXX_VERSION}")
message(STATUS "  BGEN support:      ${BGEN_SUPPORT}")
if(GCTA_USE_JEMALLOC)
    message(STATUS "  jemalloc:          ${GCTA_JEMALLOC_LIB} (thp:never)")
else()
    message(STATUS "  jemalloc:          OFF (default allocator)")
endif()
if(LINUX)
    message(STATUS "  RPATH type:        DT_RPATH (--disable-new-dtags, all Linux)")
    message(STATUS "  Build RPATH:       ${CMAKE_BUILD_RPATH}")
endif()
if(GCTA_BLAS_PROVIDER STREQUAL "MKL" OR GCTA_LAPACKE_IS_MKL)
    message(STATUS "")
    message(STATUS "  NOTE (MKL): set MKL_THREADING_LAYER at runtime:")
    message(STATUS "    GCC:  export MKL_THREADING_LAYER=GNU")
    message(STATUS "    AOCC: export MKL_THREADING_LAYER=IOMP5")
endif()

message(STATUS "==========================================")
message(STATUS "")

set_property(DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/main"
    "${CMAKE_SOURCE_DIR}/third_party/Pgenlib")
