cmake_minimum_required(VERSION 3.28...4.3)

if(GCTA_GCC_TOOLCHAIN AND DEFINED CMAKE_CXX_COMPILER)
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" --version
        OUTPUT_VARIABLE _pre_cxx_ver ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_pre_cxx_ver MATCHES "AOCC")
        foreach(_v IN ITEMS CMAKE_CXX_FLAGS CMAKE_C_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
            if(NOT "${${_v}}" MATCHES "--gcc-toolchain")
                string(APPEND ${_v} " --gcc-toolchain=${GCTA_GCC_TOOLCHAIN}")
                set(${_v} "${${_v}}" CACHE STRING "" FORCE)
            endif()
        endforeach()
        message(STATUS "Pre-project: injected --gcc-toolchain=${GCTA_GCC_TOOLCHAIN} for AOCC compiler test")
    endif()
endif()

project(GCTAg LANGUAGES C CXX)

# Prevent CMake from finding stale configs in build-tree or user package registries.
# This is especially important on HPC clusters where previous partial builds can
# leave broken zstd-config.cmake or other config files in the build directory.
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY        TRUE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY TRUE)

# Define platform booleans early — used throughout including in compiler detection.
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(gcta_configure_binutils_from_dir _bin_dir _tool_kind _label _warn_if_missing)
    if(_tool_kind STREQUAL "GNU")
        set(_ar_name gcc-ar)
        set(_ranlib_name gcc-ranlib)
        set(_nm_name gcc-nm)
    elseif(_tool_kind STREQUAL "LLVM")
        set(_ar_name llvm-ar)
        set(_ranlib_name llvm-ranlib)
        set(_nm_name llvm-nm)
    else()
        message(FATAL_ERROR "Unknown binutils tool kind: ${_tool_kind}")
    endif()

    find_program(_tool_ar     NAMES ${_ar_name}     HINTS "${_bin_dir}" NO_DEFAULT_PATH)
    find_program(_tool_ranlib NAMES ${_ranlib_name} HINTS "${_bin_dir}" NO_DEFAULT_PATH)
    find_program(_tool_nm     NAMES ${_nm_name}     HINTS "${_bin_dir}" NO_DEFAULT_PATH)

    if(_tool_ar)
        set(CMAKE_AR "${_tool_ar}" PARENT_SCOPE)
        set(CMAKE_AR "${_tool_ar}" CACHE FILEPATH "Archiver" FORCE)
        message(STATUS "${_label}: using archiver ${_tool_ar}")
    elseif(_warn_if_missing)
        message(WARNING "${_label}: ${_ar_name} not found next to ${CMAKE_CXX_COMPILER} — LTO may be suboptimal for static libs")
    endif()

    if(_tool_ranlib)
        set(CMAKE_RANLIB "${_tool_ranlib}" PARENT_SCOPE)
        set(CMAKE_RANLIB "${_tool_ranlib}" CACHE FILEPATH "Ranlib" FORCE)
    endif()

    if(_tool_nm)
        set(CMAKE_NM "${_tool_nm}" PARENT_SCOPE)
        set(CMAKE_NM "${_tool_nm}" CACHE FILEPATH "NM" FORCE)
    endif()
endfunction()

function(gcta_detect_blas_version _backend _blas_lib _blas_include_dir _out_var)
    # Best-effort backend-specific version probe for configure-time diagnostics.
    set(_ver "unknown")

    if(_backend STREQUAL "OpenBLAS")
        set(_openblas_cfg "${_blas_include_dir}/openblas_config.h")
        if(EXISTS "${_openblas_cfg}")
            file(STRINGS "${_openblas_cfg}" _openblas_ver_line REGEX "^#define[ \t]+OPENBLAS_VERSION[ \t]+\".*\"")
            if(_openblas_ver_line MATCHES "\"([^\"]+)\"")
                set(_ver "${CMAKE_MATCH_1}")
            endif()
        endif()
        if(_ver STREQUAL "unknown")
            get_filename_component(_blas_name "${_blas_lib}" NAME)
            if(_blas_name MATCHES "\\.so\\.([0-9][0-9.]*)$")
                set(_ver "soname ${CMAKE_MATCH_1}")
            elseif(_blas_name MATCHES "\\.dylib\\.([0-9][0-9.]*)$")
                set(_ver "dylib ${CMAKE_MATCH_1}")
            endif()
        endif()

    elseif(_backend STREQUAL "MKL")
        set(_mkl_ver_h "${_blas_include_dir}/mkl_version.h")
        if(EXISTS "${_mkl_ver_h}")
            file(STRINGS "${_mkl_ver_h}" _mkl_major_line REGEX "^#define[ \t]+__INTEL_MKL__[ \t]+[0-9]+")
            file(STRINGS "${_mkl_ver_h}" _mkl_minor_line REGEX "^#define[ \t]+__INTEL_MKL_MINOR__[ \t]+[0-9]+")
            file(STRINGS "${_mkl_ver_h}" _mkl_update_line REGEX "^#define[ \t]+__INTEL_MKL_UPDATE__[ \t]+[0-9]+")
            if(_mkl_major_line MATCHES "([0-9]+)$")
                set(_mkl_major "${CMAKE_MATCH_1}")
            endif()
            if(_mkl_minor_line MATCHES "([0-9]+)$")
                set(_mkl_minor "${CMAKE_MATCH_1}")
            endif()
            if(_mkl_update_line MATCHES "([0-9]+)$")
                set(_mkl_update "${CMAKE_MATCH_1}")
            endif()
            if(DEFINED _mkl_major AND DEFINED _mkl_minor AND DEFINED _mkl_update)
                set(_ver "${_mkl_major}.${_mkl_minor}.${_mkl_update}")
            endif()
        endif()

    elseif(_backend STREQUAL "AOCL")
        # AOCL typically exposes a BLIS version macro via blis.h.
        set(_blis_h "${_blas_include_dir}/blis.h")
        if(EXISTS "${_blis_h}")
            file(STRINGS "${_blis_h}" _blis_ver_line REGEX "^#define[ \t]+BLIS_VERSION_STRING[ \t]+\".*\"")
            if(_blis_ver_line MATCHES "\"([^\"]+)\"")
                set(_ver "${CMAKE_MATCH_1}")
            endif()
        endif()
        if(_ver STREQUAL "unknown")
            get_filename_component(_blas_name "${_blas_lib}" NAME)
            if(_blas_name MATCHES "\\.so\\.([0-9][0-9.]*)$")
                set(_ver "soname ${CMAKE_MATCH_1}")
            endif()
        endif()

    elseif(_backend STREQUAL "Accelerate")
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-version
            OUTPUT_VARIABLE _macos_sdk_ver
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_macos_sdk_ver)
            set(_ver "SDK ${_macos_sdk_ver}")
        endif()
    endif()

    set(${_out_var} "${_ver}" PARENT_SCOPE)
endfunction()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Compiling with GCC ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "GCTA requires GCC >= 13 for reliable C++23 support (found ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
    # Use gcc-ar/gcc-ranlib so static libs (Pgenlib, sqlite3) also get LTO-aware
    # archiving. CMake only sets CMAKE_CXX_COMPILER_AR for executables when IPO is ON.
    get_filename_component(_gcc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    gcta_configure_binutils_from_dir("${_gcc_bin}" "GNU" "GCC" TRUE)
    if(LINUX)
        # Pin libgomp to this GCC's copy so HPC LD_LIBRARY_PATH doesn't shadow it.
        # Uses the same -print-file-name probe as the AOCC libomp / toolchain logic.
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgomp.so.1
            OUTPUT_VARIABLE _gcc_gomp_path
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_gcc_gomp_path AND NOT _gcc_gomp_path STREQUAL "libgomp.so.1")
            file(REAL_PATH "${_gcc_gomp_path}" _gcc_gomp_real)
            get_filename_component(_gcc_gomp_dir "${_gcc_gomp_real}" DIRECTORY)
            list(APPEND CMAKE_BUILD_RPATH   "${_gcc_gomp_dir}")
            list(APPEND CMAKE_INSTALL_RPATH "${_gcc_gomp_dir}")
            message(STATUS "GCC: libgomp.so.1 -> ${_gcc_gomp_real}")
        endif()
        foreach(_gcc_rtlib IN ITEMS libstdc++.so.6 libgcc_s.so.1)
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${_gcc_rtlib}
                OUTPUT_VARIABLE _gcc_rtlib_path
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_gcc_rtlib_path AND NOT _gcc_rtlib_path STREQUAL "${_gcc_rtlib}")
                file(REAL_PATH "${_gcc_rtlib_path}" _gcc_rtlib_real)
                get_filename_component(_gcc_rtlib_dir "${_gcc_rtlib_real}" DIRECTORY)
                list(APPEND CMAKE_BUILD_RPATH   "${_gcc_rtlib_dir}")
                list(APPEND CMAKE_INSTALL_RPATH "${_gcc_rtlib_dir}")
                message(STATUS "GCC: ${_gcc_rtlib} -> ${_gcc_rtlib_real}")
            else()
                message(WARNING "GCC: -print-file-name=${_gcc_rtlib} did not resolve")
            endif()
        endforeach()
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Compiling with Clang ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
        message(FATAL_ERROR "GCTA requires Clang >= 17 for reliable C++23 support (found ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
    if(APPLE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        # Derive LLVM prefix from the selected compiler path (.../bin/clang++ -> ...).
        get_filename_component(_llvm_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(_llvm_prefix "${_llvm_bin_dir}" DIRECTORY)
        if(EXISTS "${_llvm_prefix}/lib/c++")
            if(NOT CMAKE_CXX_FLAGS MATCHES "(^| )-stdlib=libc\\+\\+($| )")
                string(APPEND CMAKE_CXX_FLAGS " -stdlib=libc++")
            endif()
            if(NOT CMAKE_EXE_LINKER_FLAGS MATCHES "(^| )-L${_llvm_prefix}/lib/c\\+\\+($| )")
                string(APPEND CMAKE_EXE_LINKER_FLAGS " -L${_llvm_prefix}/lib/c++")
            endif()
            link_directories("${_llvm_prefix}/lib/c++")
            list(APPEND CMAKE_BUILD_RPATH "${_llvm_prefix}/lib/c++")
            message(STATUS "Using LLVM libc++ from ${_llvm_prefix}/lib/c++")
        endif()
    endif()
else()
    message(FATAL_ERROR "GCTA can only compile with GCC or Clang")
endif()

# Detect AOCC (AMD's Clang-based compiler). It reports CMAKE_CXX_COMPILER_ID
# as "Clang" but has its own libomp and a clang-specific AOCL library variant.
# We probe the --version string since there is no dedicated CMake variable.
set(GCTA_COMPILER_IS_AOCC FALSE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND LINUX)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE _cxx_ver_str
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_cxx_ver_str MATCHES "AOCC")
        set(GCTA_COMPILER_IS_AOCC TRUE)
        # Derive the AOCC install root from the compiler path (bin/../)
        get_filename_component(_aocc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(GCTA_AOCC_ROOT "${_aocc_bin}" DIRECTORY CACHE)
        message(STATUS "Detected AOCC compiler (root: ${GCTA_AOCC_ROOT})")

        # AOCC requires its own llvm-ar / llvm-ranlib for static library creation.
        # The system 'ar' cannot handle LLVM bitcode objects produced by AOCC,
        # causing "Error running link command: no such file or directory" at archive time.
        find_program(_aocc_ar NAMES llvm-ar HINTS "${GCTA_AOCC_ROOT}/bin" NO_DEFAULT_PATH)
        if(NOT _aocc_ar)
            message(WARNING "AOCC: llvm-ar not found under ${GCTA_AOCC_ROOT}/bin — static library builds may fail")
        endif()
        gcta_configure_binutils_from_dir("${GCTA_AOCC_ROOT}/bin" "LLVM" "AOCC" FALSE)

        # AOCC (Clang 17) does not define __cpp_concepts when using libstdc++ headers,
        # which gates <expected> and other C++23 headers. Explicitly define both macros
        # as belt-and-suspenders; suppress the resulting -Wbuiltin-macro-redefined.
        add_compile_definitions(__cpp_concepts=202002L __cpp_lib_expected=202211L)
        add_compile_options(-Wno-builtin-macro-redefined)
        message(STATUS "AOCC: applied libstdc++ C++23 compatibility shims (__cpp_concepts, __cpp_lib_expected)")
    endif()
endif()

# For upstream Clang (not AOCC), detect LLVM tooling next to the compiler so
# ar/nm/ranlib come from the same LLVM install rather than PATH.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT GCTA_COMPILER_IS_AOCC)
    get_filename_component(_clang_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    gcta_configure_binutils_from_dir("${_clang_bin}" "LLVM" "Clang" FALSE)
endif()

# AOCC is LLVM 17 and cannot parse headers from GCC 15+. If your default GCC is
# too new, point AOCC at an older GCC toolchain:
#   -DGCTA_GCC_TOOLCHAIN=/path/to/gcc-13-or-14-root
# The path should be the GCC prefix — the directory containing bin/gcc.
set(GCTA_GCC_TOOLCHAIN "" CACHE PATH
    "GCC installation root for AOCC (required when system GCC >= 15). Must contain bin/gcc.")
if(GCTA_COMPILER_IS_AOCC AND GCTA_GCC_TOOLCHAIN)
    if(NOT IS_DIRECTORY "${GCTA_GCC_TOOLCHAIN}")
        message(FATAL_ERROR "GCTA_GCC_TOOLCHAIN does not exist: ${GCTA_GCC_TOOLCHAIN}")
    endif()
    message(STATUS "AOCC: using GCC toolchain at ${GCTA_GCC_TOOLCHAIN}")

    # Validate the toolchain GCC version: AOCC (Clang 17) requires GCC 12–14.
    # GCC < 12 lacks C++23 <expected>; GCC >= 15 has headers Clang 17 cannot parse.
    find_program(_aocc_tc_gcc NAMES gcc HINTS "${GCTA_GCC_TOOLCHAIN}/bin" NO_DEFAULT_PATH)
    if(_aocc_tc_gcc)
        execute_process(
            COMMAND ${_aocc_tc_gcc} --version
            OUTPUT_VARIABLE _aocc_tc_gcc_ver
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_aocc_tc_gcc_ver MATCHES "([0-9]+)\\.[0-9]+\\.[0-9]+")
            set(_aocc_tc_gcc_major "${CMAKE_MATCH_1}")
            if(_aocc_tc_gcc_major LESS 12)
                message(FATAL_ERROR
                    "GCTA_GCC_TOOLCHAIN GCC ${_aocc_tc_gcc_major} is too old — AOCC requires GCC 12–14.")
            elseif(_aocc_tc_gcc_major GREATER_EQUAL 15)
                message(FATAL_ERROR
                    "GCTA_GCC_TOOLCHAIN GCC ${_aocc_tc_gcc_major} is too new — AOCC (Clang 17) cannot parse GCC 15+ headers.")
            else()
                message(STATUS "AOCC: GCC toolchain version ${_aocc_tc_gcc_major} (OK, range 12–14)")
            endif()
        else()
            message(WARNING "AOCC: could not parse GCC version from ${_aocc_tc_gcc} — proceeding without version check")
        endif()
    else()
        message(WARNING "AOCC: gcc not found under ${GCTA_GCC_TOOLCHAIN}/bin — cannot validate toolchain version")
    endif()

    add_compile_options(--gcc-toolchain=${GCTA_GCC_TOOLCHAIN})
    add_link_options(--gcc-toolchain=${GCTA_GCC_TOOLCHAIN})

    # Use the compiler's own -print-file-name to locate libstdc++.so.6 and
    # libgcc_s.so.1 within the toolchain.  This is more reliable than directory
    # globs: the compiler applies the same search logic (multilib, sysroot,
    # --gcc-toolchain) it uses at link time, so we get the exact file it will
    # link against.  Spack and other toolchain managers emit paths with ../../../../
    # traversals (e.g. lib/gcc/<triple>/<ver>/../../../../lib64/); file(REAL_PATH)
    # normalises these to canonical absolute paths before adding to RPATH.
    foreach(_tc_lib IN ITEMS libstdc++.so.6 libgcc_s.so.1)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER}
                    --gcc-toolchain=${GCTA_GCC_TOOLCHAIN}
                    -print-file-name=${_tc_lib}
            OUTPUT_VARIABLE _tc_lib_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        # The compiler echoes back the bare filename when it cannot find the file.
        if(_tc_lib_path AND NOT _tc_lib_path STREQUAL "${_tc_lib}")
            file(REAL_PATH "${_tc_lib_path}" _tc_lib_real)
            get_filename_component(_tc_lib_dir "${_tc_lib_real}" DIRECTORY)
            list(APPEND CMAKE_BUILD_RPATH   "${_tc_lib_dir}")
            list(APPEND CMAKE_INSTALL_RPATH "${_tc_lib_dir}")
            message(STATUS "AOCC: ${_tc_lib} -> ${_tc_lib_real}")
        else()
            message(WARNING
                "AOCC: -print-file-name=${_tc_lib} did not resolve in ${GCTA_GCC_TOOLCHAIN} — "
                "runtime may load the system library.")
        endif()
    endforeach()
    message(STATUS "AOCC: added GCC toolchain lib dirs to rpath")
elseif(GCTA_COMPILER_IS_AOCC AND NOT GCTA_GCC_TOOLCHAIN)
    # Warn if the GCC associated with this toolchain is too new for AOCC.
    set(_gcc_probe_cmd "gcc")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-prog-name=gcc
        OUTPUT_VARIABLE _gcc_probe_path
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_gcc_probe_path AND NOT _gcc_probe_path STREQUAL "gcc" AND EXISTS "${_gcc_probe_path}")
        set(_gcc_probe_cmd "${_gcc_probe_path}")
    endif()
    execute_process(
        COMMAND ${_gcc_probe_cmd} --version
        OUTPUT_VARIABLE _gcc_ver_str
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_gcc_ver_str MATCHES "([0-9]+)\\.[0-9]+\\.[0-9]+")
        if(CMAKE_MATCH_1 GREATER_EQUAL 15)
            message(WARNING
                "AOCC detected but ambient GCC ${CMAKE_MATCH_1} is too new — "
                "set -DGCTA_GCC_TOOLCHAIN=/path/to/gcc-13-or-14.")
        endif()
    endif()
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
option(GCTA_LIST_TARGETS "List imported and local targets during configure" OFF)
option(GCTA_UNITY_BUILD "Enable unity builds for faster compilation" ON)
option(GCTA_ENABLE_IPO "Enable IPO/LTO in Release builds" ON)
option(GCTA_VERBOSE_BLAS "Print verbose BLAS detection information during configure" OFF)
option(BGEN_SUPPORT "Enable BGEN file format support (requires SQLite)" OFF)


# Initialize plink-ng submodule if needed
if(EXISTS "${CMAKE_SOURCE_DIR}/.git" AND NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/plink-ng/2.0/include/pgenlib_ffi_support.h")
    find_package(Git QUIET)
    if(GIT_FOUND)
        message(STATUS "Initializing plink-ng submodule...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}")
        endif()
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
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(
    spectra
    URL https://github.com/yixuan/spectra/archive/refs/tags/v1.2.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(
    zstd
    URL https://github.com/facebook/zstd/archive/refs/tags/v1.5.7.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR "build/cmake"
    EXCLUDE_FROM_ALL)
FetchContent_Declare(
    Boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.91.0/boost-1.91.0-cmake.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
if(BGEN_SUPPORT)
    FetchContent_Declare(
        sqlite
        URL https://www.sqlite.org/2026/sqlite-amalgamation-3510200.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
    stdexec
    URL https://github.com/NVIDIA/stdexec/archive/refs/tags/nvhpc-26.05.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
    set(BOOST_IOSTREAMS_ENABLE_ZLIB ON)
    if(Boost_DIR)
        FetchContent_MakeAvailable(zstd)
    else()
        # These two vars are only meaningful to Boost's own CMake build system
        set(BOOST_ENABLE_CMAKE ON)
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

# Resolve zstd link target.
# Check for the actual CMake target rather than relying on zstd_POPULATED,
# which CMake 4.x no longer sets reliably after a block() scope.
# FetchContent's zstd build always creates 'libzstd'; system installs use
# the zstd:: namespace targets.
if(TARGET libzstd)
    # Built from source by FetchContent — target carries its own include dirs
    set(GCTA_ZSTD_INCLUDE_DIRS "")
    set(GCTA_ZSTD_LIBS libzstd)
elseif(TARGET zstd::libzstd_shared)
    set(GCTA_ZSTD_INCLUDE_DIRS "")
    set(GCTA_ZSTD_LIBS zstd::libzstd_shared)
elseif(TARGET zstd::libzstd_static)
    set(GCTA_ZSTD_INCLUDE_DIRS "")
    set(GCTA_ZSTD_LIBS zstd::libzstd_static)
else()
    message(FATAL_ERROR
        "zstd target not found — FetchContent may have failed. "
        "Try: module unload zstd && rm -rf build && cmake -B build ...")
endif()

if(BGEN_SUPPORT)
    add_library(sqlite3 STATIC ${sqlite_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3 PUBLIC ${sqlite_SOURCE_DIR})
endif()

find_package(Threads REQUIRED)

# find_package(OpenMP) cannot locate libomp on macOS or steer away from libgomp
# on AOCC, so we pre-populate its cache variables before the call.
if(APPLE)
    # AppleClang needs -Xpreprocessor; all other Clang variants use plain -fopenmp.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(OpenMP_CXX_FLAGS    "-Xpreprocessor -fopenmp" CACHE STRING "" FORCE)
    else()
        set(OpenMP_CXX_FLAGS    "-fopenmp"                CACHE STRING "" FORCE)
    endif()
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)

    # Probe compiler prefix first (Homebrew Clang ships its own libomp),
    # then fall back to a standalone Homebrew libomp install.
    cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _omp_bin_dir)
    cmake_path(GET _omp_bin_dir       PARENT_PATH _omp_prefix)
    find_path(OpenMP_CXX_INCLUDE_DIR NAMES omp.h
        HINTS "/opt/homebrew/opt/libomp/include"
              "/usr/local/opt/libomp/include"
              "${_omp_prefix}/include")
    find_library(OpenMP_omp_LIBRARY NAMES omp libomp
        HINTS "/opt/homebrew/opt/libomp/lib"
              "/usr/local/opt/libomp/lib"
              "${_omp_prefix}/lib")

    if(NOT OpenMP_omp_LIBRARY OR NOT OpenMP_CXX_INCLUDE_DIR)
        message(FATAL_ERROR "OpenMP not found on macOS. Install with: brew install libomp")
    endif()
    cmake_path(GET OpenMP_omp_LIBRARY PARENT_PATH _omp_lib_dir)
    list(APPEND CMAKE_BUILD_RPATH   "${_omp_lib_dir}")
    list(APPEND CMAKE_INSTALL_RPATH "${_omp_lib_dir}")
endif()

if(GCTA_COMPILER_IS_AOCC)
    # Pre-steer CMake to AOCC's libomp so it doesn't fall back to system libgomp.
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
    find_library(OpenMP_omp_LIBRARY NAMES omp
        HINTS "${GCTA_AOCC_ROOT}/lib" NO_DEFAULT_PATH)
    if(OpenMP_omp_LIBRARY)
        message(STATUS "AOCC OpenMP library: ${OpenMP_omp_LIBRARY}")
        cmake_path(GET OpenMP_omp_LIBRARY PARENT_PATH _aocc_omp_dir)
        list(APPEND CMAKE_BUILD_RPATH   "${_aocc_omp_dir}")
        list(APPEND CMAKE_INSTALL_RPATH "${_aocc_omp_dir}")
    else()
        message(WARNING
            "AOCC: libomp.so not found under ${GCTA_AOCC_ROOT}/lib — "
            "may fall back to system libgomp, causing runtime crashes.")
    endif()
endif()

find_package(OpenMP REQUIRED)

set(GCTA_BLAS_LIBS "")
set(GCTA_BLAS_INCLUDE_DIRS "")
set(GCTA_BLAS_DEFINES "")

if(NOT GCTA_BLAS_BACKEND)
    message(FATAL_ERROR
        "GCTA_BLAS_BACKEND is required. Choose: OpenBLAS, MKL, Accelerate, or AOCL.\n"
        "Linux:  -DGCTA_BLAS_BACKEND=OpenBLAS -DGCTA_BLAS_LIBRARY=<...> -DGCTA_BLAS_INCLUDE_DIR=<...>\n"
        "macOS:  -DGCTA_BLAS_BACKEND=Accelerate")
endif()
if(GCTA_BLAS_BACKEND STREQUAL "Auto")
    message(FATAL_ERROR "GCTA_BLAS_BACKEND=Auto is no longer supported; choose OpenBLAS, MKL, Accelerate, or AOCL explicitly.")
endif()
set(_GCTA_BLAS "${GCTA_BLAS_BACKEND}")

# Validate platform constraints
if(APPLE AND _GCTA_BLAS STREQUAL "MKL")
    message(FATAL_ERROR "MKL is not supported on macOS. Use Accelerate or OpenBLAS.")
endif()
if(APPLE AND _GCTA_BLAS STREQUAL "AOCL")
    message(FATAL_ERROR "AOCL is not supported on macOS. Use Accelerate or OpenBLAS.")
endif()
if(NOT APPLE AND _GCTA_BLAS STREQUAL "Accelerate")
    message(FATAL_ERROR "Accelerate is only available on macOS. Use MKL, OpenBLAS, or AOCL.")
endif()

if(_GCTA_BLAS STREQUAL "Accelerate")
    # Accelerate is a macOS system framework — no user-supplied paths needed.
    set(BLA_VENDOR Apple)
    find_package(BLAS   REQUIRED)
    find_package(LAPACK REQUIRED)

    # Default to 13.3 (minimum for ACCELERATE_NEW_LAPACK) if not already set.
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3" CACHE STRING "Minimum macOS deployment target")
    endif()

    if(NOT CMAKE_OSX_SYSROOT)
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    message(STATUS "Accelerate SDK: ${CMAKE_OSX_SYSROOT}")

    # Framework headers (e.g. <vecLib/cblas_new.h>) are resolved via -F.
    list(APPEND GCTA_BLAS_INCLUDE_DIRS "-F${CMAKE_OSX_SYSROOT}/System/Library/Frameworks")
    list(APPEND GCTA_BLAS_LIBS         ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    list(APPEND GCTA_BLAS_DEFINES      GCTA_USE_ACCELERATE ACCELERATE_NEW_LAPACK EIGEN_USE_BLAS)
    set(GCTA_BLAS_PROVIDER "Accelerate" CACHE INTERNAL "")

elseif(_GCTA_BLAS STREQUAL "OpenBLAS" OR _GCTA_BLAS STREQUAL "MKL" OR _GCTA_BLAS STREQUAL "AOCL")
    # OpenBLAS, MKL, and AOCL all require the user to supply explicit paths.
    # This avoids the fragile FindBLAS / BLA_VENDOR linker-probe machinery.
    # For MKL, pass libmkl_rt.so (the single-library interface).
    #   With GCC:  set MKL_THREADING_LAYER=GNU at runtime to use libgomp.
    #   With AOCC: set MKL_THREADING_LAYER=IOMP5 at runtime to use libomp.
    # For AOCL, use the gcc/ library variant with GCC and the clang/ variant with AOCC.
    if(NOT GCTA_BLAS_LIBRARY)
        message(FATAL_ERROR
            "GCTA_BLAS_LIBRARY must be set for the ${_GCTA_BLAS} backend "
            "(path to the primary .so/.a, e.g. libopenblas.so, libmkl_rt.so, libblis-mt.so).")
    endif()
    if(NOT GCTA_BLAS_INCLUDE_DIR)
        message(FATAL_ERROR
            "GCTA_BLAS_INCLUDE_DIR must be set for the ${_GCTA_BLAS} backend "
            "(directory containing cblas.h / mkl.h / blis.h).")
    endif()
    if(NOT EXISTS "${GCTA_BLAS_LIBRARY}")
        message(FATAL_ERROR "GCTA_BLAS_LIBRARY does not exist: ${GCTA_BLAS_LIBRARY}")
    endif()
    if(NOT IS_DIRECTORY "${GCTA_BLAS_INCLUDE_DIR}")
        message(FATAL_ERROR "GCTA_BLAS_INCLUDE_DIR does not exist: ${GCTA_BLAS_INCLUDE_DIR}")
    endif()

    message(STATUS "${_GCTA_BLAS} library:     ${GCTA_BLAS_LIBRARY}")
    if(GCTA_LAPACKE_LIBRARY)
        message(STATUS "${_GCTA_BLAS} LAPACKE library: ${GCTA_LAPACKE_LIBRARY}")
    endif()
    message(STATUS "${_GCTA_BLAS} include dir: ${GCTA_BLAS_INCLUDE_DIR}")

    if(LINUX)
        get_filename_component(_blas_lib_dir "${GCTA_BLAS_LIBRARY}" DIRECTORY)
        list(APPEND CMAKE_BUILD_RPATH   "${_blas_lib_dir}")
        list(APPEND CMAKE_INSTALL_RPATH "${_blas_lib_dir}")
        if(NOT GCTA_COMPILER_IS_AOCC)
            add_link_options("LINKER:-rpath-link,${_blas_lib_dir}")
        endif()
        if(GCTA_LAPACKE_LIBRARY)
            get_filename_component(_lapack_lib_dir "${GCTA_LAPACKE_LIBRARY}" DIRECTORY)
            list(APPEND CMAKE_BUILD_RPATH   "${_lapack_lib_dir}")
            list(APPEND CMAKE_INSTALL_RPATH "${_lapack_lib_dir}")
            if(NOT GCTA_COMPILER_IS_AOCC)
                add_link_options("LINKER:-rpath-link,${_lapack_lib_dir}")
            endif()
        endif()
    endif()

    list(APPEND GCTA_BLAS_INCLUDE_DIRS "${GCTA_BLAS_INCLUDE_DIR}")
    list(APPEND GCTA_BLAS_LIBS         "${GCTA_BLAS_LIBRARY}")
    if(GCTA_LAPACKE_LIBRARY)
        if(NOT EXISTS "${GCTA_LAPACKE_LIBRARY}")
            message(FATAL_ERROR "GCTA_LAPACKE_LIBRARY does not exist: ${GCTA_LAPACKE_LIBRARY}")
        endif()
        list(APPEND GCTA_BLAS_LIBS "${GCTA_LAPACKE_LIBRARY}")
    endif()

    if(_GCTA_BLAS STREQUAL "OpenBLAS")
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_OPENBLAS EIGEN_USE_LAPACKE EIGEN_USE_BLAS)
        set(GCTA_BLAS_PROVIDER "OpenBLAS" CACHE INTERNAL "")
    elseif(_GCTA_BLAS STREQUAL "AOCL")
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_AOCL EIGEN_USE_LAPACKE EIGEN_USE_BLAS)
        set(GCTA_BLAS_PROVIDER "AOCL" CACHE INTERNAL "")

        # AOCL ships two separate library builds:
        #   aocc/  — linked against libomp  (LLVM OpenMP, AOCC's runtime)
        #   gcc/   — linked against libgomp (GCC's OpenMP runtime)
        # Using the AOCC variant with GCC (or vice-versa) produces link errors:
        #   "libomp.so not found" / "undefined reference to __kmpc_*"
        # Detect the mismatch early via the presence of libomp.so in the lib dir.
        get_filename_component(_aocl_lib_dir "${GCTA_BLAS_LIBRARY}" DIRECTORY)
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

        # AOCL transitive dependencies: libflame needs libaoclutils and libblis-mt.
        # Skip any dep whose path is already in GCTA_BLAS_LIBS (e.g. libblis-mt
        # is the primary BLAS library and must not be linked a second time).
        get_filename_component(_aocl_lib_dir "${GCTA_BLAS_LIBRARY}" DIRECTORY)
        foreach(_aocl_dep IN ITEMS libaoclutils.so libblis-mt.so)
            set(_aocl_dep_path "${_aocl_lib_dir}/${_aocl_dep}")
            set(_aocl_dep_path64 "${_aocl_lib_dir}/../lib64/${_aocl_dep}")
            if("${_aocl_dep_path}" IN_LIST GCTA_BLAS_LIBS)
                message(STATUS "AOCL: skipping ${_aocl_dep} (already linked as primary library)")
            elseif(EXISTS "${_aocl_dep_path}")
                list(APPEND GCTA_BLAS_LIBS "${_aocl_dep_path}")
                message(STATUS "AOCL: found transitive dep ${_aocl_dep}")
            elseif(EXISTS "${_aocl_dep_path64}")
                list(APPEND GCTA_BLAS_LIBS "${_aocl_dep_path64}")
                message(STATUS "AOCL: found transitive dep ${_aocl_dep} (lib64)")
            else()
                message(WARNING "AOCL: transitive dep not found: ${_aocl_dep_path}")
            endif()
        endforeach()
    else()
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_MKL EIGEN_USE_MKL_ALL)
        set(GCTA_BLAS_PROVIDER "MKL" CACHE INTERNAL "")
    endif()

else()
    message(FATAL_ERROR "Unknown GCTA_BLAS_BACKEND: '${GCTA_BLAS_BACKEND}'. Use OpenBLAS, MKL, Accelerate, or AOCL.")
endif()

message(STATUS "BLAS/LAPACK provider: ${GCTA_BLAS_PROVIDER}")
gcta_detect_blas_version("${GCTA_BLAS_PROVIDER}" "${GCTA_BLAS_LIBRARY}" "${GCTA_BLAS_INCLUDE_DIR}" GCTA_BLAS_VERSION)
message(STATUS "BLAS/LAPACK version: ${GCTA_BLAS_VERSION}")

# Verbose BLAS logging: print detection details when requested
if(CMAKE_VERBOSE_MAKEFILE OR GCTA_VERBOSE_BLAS)
    message(STATUS "---- BLAS detection details ----")
    message(STATUS "  GCTA_BLAS_BACKEND:    ${GCTA_BLAS_BACKEND}")
    message(STATUS "  Resolved backend:     ${_GCTA_BLAS}")
    message(STATUS "  GCTA_BLAS_LIBRARY:    ${GCTA_BLAS_LIBRARY}")
    message(STATUS "  GCTA_LAPACKE_LIBRARY: ${GCTA_LAPACKE_LIBRARY}")
    message(STATUS "  GCTA_BLAS_INCLUDE_DIR:${GCTA_BLAS_INCLUDE_DIR}")
    message(STATUS "  Defines:              ${GCTA_BLAS_DEFINES}")
    message(STATUS "  Link list:")
    foreach(_lib IN LISTS GCTA_BLAS_LIBS)
        message(STATUS "    ${_lib}")
    endforeach()
    message(STATUS "---- end BLAS detection details ----")
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
    third_party/plink-ng/2.0/simde
    # module-mode FindBoost.cmake fallback (empty when using config-mode Boost targets)
    ${Boost_INCLUDE_DIRS})

# Boost: prefer modern config-mode targets; fall back to module-mode Boost_LIBRARIES.
# $<TARGET_EXISTS:...> (3.27) avoids the imperative if(TARGET)/list(APPEND) pattern.
target_link_libraries(GCTAg PRIVATE
    $<$<TARGET_EXISTS:Boost::iostreams>:Boost::iostreams>
    $<$<TARGET_EXISTS:Boost::headers>:Boost::headers>
    $<$<BOOL:${Boost_LIBRARIES}>:${Boost_LIBRARIES}>
    $<$<BOOL:${Boost_DIR}>:ZLIB::ZLIB>   # system Boost::iostreams needs system zlib
    Eigen3::Eigen
    Spectra
    STDEXEC::stdexec
    ${GCTA_ZSTD_LIBS})

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

target_include_directories(GCTAg PRIVATE
    ${GCTA_ZSTD_INCLUDE_DIRS}
    ${GCTA_BLAS_INCLUDE_DIRS})

target_compile_definitions(GCTAg PRIVATE
    ${GCTA_BLAS_DEFINES}
    GCTA_BLAS_BACKEND_STR="${GCTA_BLAS_PROVIDER}"
    GCTA_BLAS_VERSION_STR="${GCTA_BLAS_VERSION}")

# LTO flavour: AOCC requires full LTO; upstream Clang prefers ThinLTO; GCC uses auto.
# Folded into the same target_compile/link_options calls so all flags are co-located.
set(_lto_compile_flag "")
set(_lto_link_flag    "")
if(GCTA_IPO_ENABLED)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(GCTA_COMPILER_IS_AOCC)
            set(_lto_compile_flag -flto=full)
            set(_lto_link_flag    -flto=full)
        else()
            set(_lto_compile_flag -flto=thin)
            set(_lto_link_flag    -flto=thin)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(_lto_compile_flag -flto=auto)
        set(_lto_link_flag    -flto=auto)
    endif()
endif()

target_compile_options(GCTAg PRIVATE
    $<$<BOOL:${APPLE}>:-Wno-nullability-completeness>
    -pipe
    $<$<CONFIG:Release>:$<IF:$<BOOL:${APPLE}>,-mcpu=native,-march=native>>
    $<$<CONFIG:Release>:-O3>
    $<$<AND:$<CONFIG:Release>,$<BOOL:${_lto_compile_flag}>>:${_lto_compile_flag}>)

target_link_options(GCTAg PRIVATE
    $<$<AND:$<CONFIG:Release>,$<BOOL:${_lto_link_flag}>>:${_lto_link_flag}>)

target_link_libraries(GCTAg PRIVATE OpenMP::OpenMP_CXX)

# Both GCC (GNU ld) and Clang (lld or GNU ld) default to DT_RUNPATH (--enable-new-dtags).
# DT_RUNPATH is searched *after* LD_LIBRARY_PATH, so HPC module paths prepended
# to LD_LIBRARY_PATH win over the baked-in RPATH, breaking libstdc++/BLAS pinning.
# --disable-new-dtags writes DT_RPATH instead, searched *before* LD_LIBRARY_PATH.
# Applied to all Linux builds regardless of Clang variant (AOCC or upstream).
if(LINUX)
    target_link_options(GCTAg PRIVATE "LINKER:--disable-new-dtags")
    message(STATUS "GCTAg: using DT_RPATH (--disable-new-dtags) for HPC library pinning")
endif()


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
if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # plink2_base.h requires FMA3 whenever 32-byte AVX2 float vectors are enabled.
    target_compile_options(Pgenlib PRIVATE -mavx2 -mfma -mbmi -mbmi2 -mlzcnt)
endif()


if(BGEN_SUPPORT)
    target_link_libraries(GCTAg PRIVATE sqlite3)
    target_compile_definitions(GCTAg PRIVATE BGEN_SUPPORT)
endif()
target_link_libraries(GCTAg PRIVATE Pgenlib Threads::Threads)

# Link BLAS early so OpenBLAS/Accelerate/MKL cblas symbols take precedence
# over other cblas implementations in dyld (macOS) and ld.so (Linux) symbol resolution order
target_link_libraries(GCTAg PRIVATE ${GCTA_BLAS_LIBS})

# On Linux, re-link BLAS to ensure it appears last in NEEDED and
# its cblas symbols take precedence via ld.so's first-match ordering.
if(LINUX)
    target_link_libraries(GCTAg PRIVATE ${GCTA_BLAS_LIBS})
endif()

# List all targets (verbose, only if requested)
if(CMAKE_VERBOSE_MAKEFILE OR GCTA_LIST_TARGETS)
    foreach(prop IN ITEMS IMPORTED_TARGETS BUILDSYSTEM_TARGETS)
        get_property(_targets DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY ${prop})
        foreach(t IN LISTS _targets)
            message(STATUS "${prop}: ${t}")
        endforeach()
    endforeach()
endif()

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

if(LINUX)
    install(FILES ${CMAKE_SOURCE_DIR}/resources/GCTAg.desktop
        DESTINATION share/applications)
    install(FILES
        ${CMAKE_SOURCE_DIR}/resources/GCTAg.png
        DESTINATION share/icons/hicolor/256x256/apps)
    install(FILES
        ${CMAKE_SOURCE_DIR}/resources/GCTAg.svg
        DESTINATION share/icons/hicolor/scalable/apps)
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
message(STATUS "  Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}${GCTA_COMPILER_IS_AOCC_STR} (${CMAKE_CXX_COMPILER})")
message(STATUS "  Archiver (AR):     ${CMAKE_AR}")
message(STATUS "  Ranlib:            ${CMAKE_RANLIB}")
message(STATUS "  NM:                ${CMAKE_NM}")
if(GCTA_COMPILER_IS_AOCC)
    message(STATUS "  GCC toolchain:     ${GCTA_GCC_TOOLCHAIN} (--gcc-toolchain)")
    message(STATUS "  libstdc++ shims:   __cpp_concepts=202002L, __cpp_lib_expected=202211L")
endif()
message(STATUS "  IPO/LTO:           ${GCTA_IPO_ENABLED}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  OpenMP:            ${OpenMP_CXX_VERSION}")
message(STATUS "  BGEN support:      ${BGEN_SUPPORT}")
if(LINUX)
    message(STATUS "  RPATH type:        DT_RPATH (--disable-new-dtags, all Linux)")
    message(STATUS "  Build RPATH:       ${CMAKE_BUILD_RPATH}")
endif()
if(GCTA_BLAS_PROVIDER STREQUAL "MKL")
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
