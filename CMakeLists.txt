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

project(GCTA LANGUAGES C CXX)

# Prevent CMake from finding stale configs in build-tree or user package registries.
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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Compiling with GCC ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "GCTA requires GCC >= 13 for reliable C++23 support (found ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
    get_filename_component(_gcc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    gcta_configure_binutils_from_dir("${_gcc_bin}" "GNU" "GCC" TRUE)
    if(LINUX)
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
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Compiling with Clang ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
        message(FATAL_ERROR "GCTA requires Clang >= 17 for reliable C++23 support (found ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
    if(APPLE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
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

set(GCTA_COMPILER_IS_AOCC FALSE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND LINUX)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE _cxx_ver_str
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_cxx_ver_str MATCHES "AOCC")
        set(GCTA_COMPILER_IS_AOCC TRUE)
        get_filename_component(_aocc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(GCTA_AOCC_ROOT "${_aocc_bin}" DIRECTORY CACHE)
        message(STATUS "Detected AOCC compiler (root: ${GCTA_AOCC_ROOT})")

        find_program(_aocc_ar NAMES llvm-ar HINTS "${GCTA_AOCC_ROOT}/bin" NO_DEFAULT_PATH)
        if(NOT _aocc_ar)
            message(WARNING "AOCC: llvm-ar not found under ${GCTA_AOCC_ROOT}/bin — static library builds may fail")
        endif()
        gcta_configure_binutils_from_dir("${GCTA_AOCC_ROOT}/bin" "LLVM" "AOCC" FALSE)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT GCTA_COMPILER_IS_AOCC)
    get_filename_component(_clang_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    gcta_configure_binutils_from_dir("${_clang_bin}" "LLVM" "Clang" FALSE)
endif()

set(GCTA_GCC_TOOLCHAIN "" CACHE PATH
    "GCC installation root for AOCC to use as its C++ header/sysroot provider. \
Required when the system GCC is >= 15 (AOCC/Clang 17 cannot parse GCC 15+ libstdc++ headers). \
Example: /opt/gcc-13  (must contain bin/gcc)")
if(GCTA_COMPILER_IS_AOCC AND GCTA_GCC_TOOLCHAIN)
    if(NOT IS_DIRECTORY "${GCTA_GCC_TOOLCHAIN}")
        message(FATAL_ERROR "GCTA_GCC_TOOLCHAIN does not exist: ${GCTA_GCC_TOOLCHAIN}")
    endif()
    message(STATUS "AOCC: using GCC toolchain at ${GCTA_GCC_TOOLCHAIN}")
    add_compile_options(--gcc-toolchain=${GCTA_GCC_TOOLCHAIN})
    add_link_options(--gcc-toolchain=${GCTA_GCC_TOOLCHAIN})

    foreach(_tc_lib IN ITEMS libstdc++.so.6 libgcc_s.so.1)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER}
                    --gcc-toolchain=${GCTA_GCC_TOOLCHAIN}
                    -print-file-name=${_tc_lib}
            OUTPUT_VARIABLE _tc_lib_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_tc_lib_path AND NOT _tc_lib_path STREQUAL "${_tc_lib}")
            file(REAL_PATH "${_tc_lib_path}" _tc_lib_real)
            get_filename_component(_tc_lib_dir "${_tc_lib_real}" DIRECTORY)
            list(APPEND CMAKE_BUILD_RPATH   "${_tc_lib_dir}")
            list(APPEND CMAKE_INSTALL_RPATH "${_tc_lib_dir}")
            message(STATUS "AOCC: ${_tc_lib} -> ${_tc_lib_real}")
        else()
            message(WARNING
                "AOCC: -print-file-name=${_tc_lib} did not resolve to a path "
                "in ${GCTA_GCC_TOOLCHAIN} — runtime may load the system library. "
                "Use -DGCTA_STATIC_STDCXX=ON to avoid the runtime dependency.")
        endif()
    endforeach()
    message(STATUS "AOCC: added GCC toolchain lib dirs to rpath")
elseif(GCTA_COMPILER_IS_AOCC AND NOT GCTA_GCC_TOOLCHAIN)
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
                "AOCC detected but GCC ${CMAKE_MATCH_1} (${_gcc_probe_cmd}) is too new — "
                "AOCC (Clang 17) cannot parse GCC 15+ libstdc++ headers.\n"
                "Set -DGCTA_GCC_TOOLCHAIN=/path/to/gcc-13-or-14 to fix compile errors.")
        endif()
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Defaulting to Release build")
endif()

set(GCTA_BLAS_BACKEND "" CACHE STRING "BLAS/LAPACK backend (required): OpenBLAS, MKL, Accelerate, or AOCL")
set_property(CACHE GCTA_BLAS_BACKEND PROPERTY STRINGS OpenBLAS MKL Accelerate AOCL)
set(GCTA_BLAS_LIBRARY "" CACHE FILEPATH "Primary BLAS library (OpenBLAS/MKL only).")
set(GCTA_LAPACKE_LIBRARY "" CACHE FILEPATH "Separate LAPACKE library, if needed.")
set(GCTA_BLAS_INCLUDE_DIR "" CACHE PATH "Directory containing BLAS/LAPACK headers.")
set(Boost_DIR "" CACHE PATH "Exact Boost CMake config directory.")
option(GCTA_LIST_TARGETS "List imported and local targets during configure" OFF)
option(GCTA_UNITY_BUILD "Enable unity builds for faster compilation" ON)
option(GCTA_ENABLE_IPO "Enable IPO/LTO in Release builds" ON)
option(GCTA_VERBOSE_BLAS "Print verbose BLAS detection information during configure" OFF)

# Fetch 3rd parties
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Skip re-checking FetchContent URLs on re-configure (set OFF to force update check)")
if(CMAKE_VERBOSE_MAKEFILE)
    set(FETCHCONTENT_QUIET OFF)
else()
    set(FETCHCONTENT_QUIET ON)
endif()

FetchContent_Declare(
    zlib
    URL https://github.com/madler/zlib/archive/refs/tags/v1.3.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
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
FetchContent_Declare(
    sqlite
    URL https://www.sqlite.org/2026/sqlite-amalgamation-3510200.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
set(ZLIB_BUILD_EXAMPLES OFF)
FetchContent_MakeAvailable(zlib)

block()
    set(ZSTD_BUILD_PROGRAMS OFF)
    set(ZSTD_BUILD_TESTS    OFF)
    set(ZSTD_BUILD_SHARED   ON)
    set(SPECTRA_BUILD_TESTS OFF)
    set(BUILD_TESTING       OFF)
    set(BOOST_IOSTREAMS_ENABLE_ZLIB ON)
    if(zlib_POPULATED AND TARGET ZLIB::ZLIB)
        set(ZLIB_ROOT        "${zlib_BINARY_DIR}"                    CACHE PATH   "" FORCE)
        set(ZLIB_LIBRARY     ZLIB::ZLIB                              CACHE STRING "" FORCE)
        set(ZLIB_INCLUDE_DIRS "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}" CACHE PATH   "" FORCE)
    endif()

    if(Boost_DIR)
        FetchContent_MakeAvailable(zstd)
    else()
        set(BOOST_ENABLE_CMAKE ON)
        set(BOOST_INCLUDE_LIBRARIES "algorithm;math;crc")
        FetchContent_MakeAvailable(zstd Boost)
    endif()
endblock()

set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_MakeAvailable(eigen spectra)
set(CMAKE_SKIP_INSTALL_RULES OFF)
FetchContent_MakeAvailable(sqlite)

if(Boost_DIR)
    message(STATUS "Using external Boost from ${Boost_DIR}")
    find_package(Boost REQUIRED COMPONENTS iostreams
        CONFIG
        PATHS "${Boost_DIR}"
        NO_DEFAULT_PATH)
    find_package(ZLIB REQUIRED)
endif()

if(zlib_POPULATED)
    set(GCTA_ZLIB_INCLUDE_DIRS "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")
    if(NOT TARGET ZLIB::ZLIB)
        if(TARGET zlibstatic)
            add_library(ZLIB::ZLIB ALIAS zlibstatic)
        elseif(TARGET zlib)
            add_library(ZLIB::ZLIB ALIAS zlib)
        endif()
    endif()
else()
    set(GCTA_ZLIB_INCLUDE_DIRS "${ZLIB_INCLUDE_DIRS}")
endif()

if(zlib_POPULATED AND TARGET ZLIB::ZLIB)
    set(ZLIB_ROOT         "${zlib_BINARY_DIR}"                     CACHE PATH   "" FORCE)
    set(ZLIB_LIBRARY      ZLIB::ZLIB                               CACHE STRING "" FORCE)
    set(ZLIB_INCLUDE_DIRS "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}"  CACHE PATH   "" FORCE)
    set(ZLIB_USE_STATIC_LIBS ON                                    CACHE BOOL   "" FORCE)
endif()

if(TARGET libzstd)
    set(GCTA_ZSTD_INCLUDE_DIRS "")
    set(GCTA_ZSTD_LIBS libzstd)
elseif(TARGET zstd::libzstd_shared)
    set(GCTA_ZSTD_INCLUDE_DIRS "")
    set(GCTA_ZSTD_LIBS zstd::libzstd_shared)
elseif(TARGET zstd::libzstd_static)
    set(GCTA_ZSTD_INCLUDE_DIRS "")
    set(GCTA_ZSTD_LIBS zstd::libzstd_static)
else()
    message(FATAL_ERROR "zstd is not available.")
endif()

find_package(Threads REQUIRED)

if(NOT WIN32)
    find_package(GSL REQUIRED)
endif()

add_library(sqlite3 STATIC ${sqlite_SOURCE_DIR}/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${sqlite_SOURCE_DIR})

# Find OpenMP (Require CXX component)
if(APPLE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        get_filename_component(_omp_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(_omp_prefix "${_omp_bin_dir}" DIRECTORY)
        find_path(OpenMP_CXX_INCLUDE_DIR
            NAMES omp.h
            HINTS "${_omp_prefix}/include"
            NO_DEFAULT_PATH)
        find_library(OpenMP_omp_LIBRARY
            NAMES omp libomp
            HINTS "${_omp_prefix}/lib"
            NO_DEFAULT_PATH)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(NOT OpenMP_CXX_LIB_NAMES)
            set(OpenMP_CXX_LIB_NAMES "gomp")
        endif()
        if(NOT OpenMP_CXX_FLAGS)
            set(OpenMP_CXX_FLAGS "-fopenmp")
        endif()
    else()
        if(NOT OpenMP_CXX_LIB_NAMES)
            set(OpenMP_CXX_LIB_NAMES "omp")
        endif()
        if(NOT OpenMP_CXX_FLAGS)
            if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            else()
                set(OpenMP_CXX_FLAGS "-fopenmp")
            endif()
        endif()
    endif()

    if(NOT OpenMP_CXX_INCLUDE_DIR)
        find_path(OpenMP_CXX_INCLUDE_DIR NAMES omp.h)
    endif()
    if(NOT OpenMP_omp_LIBRARY)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            find_library(OpenMP_omp_LIBRARY NAMES gomp)
        else()
            find_library(OpenMP_omp_LIBRARY NAMES omp libomp)
        endif()
    endif()
endif()

if(WIN32)
    if(NOT OpenMP_CXX_FLAGS)
        set(OpenMP_CXX_FLAGS "-fopenmp")
    endif()
    if(NOT OpenMP_CXX_LIB_NAMES)
        set(OpenMP_CXX_LIB_NAMES "omp")
    endif()
    if(DEFINED LLVM_ROOT)
        if(NOT OpenMP_omp_LIBRARY)
            set(OpenMP_omp_LIBRARY "${LLVM_ROOT}/lib/libomp.lib")
        endif()
        if(NOT OpenMP_CXX_INCLUDE_DIR)
            set(OpenMP_CXX_INCLUDE_DIR "${LLVM_ROOT}/include")
        endif()
    endif()
endif()

if(GCTA_COMPILER_IS_AOCC)
    set(OpenMP_CXX_LIB_NAMES "omp")
    find_library(_aocc_omp_lib
        NAMES omp
        HINTS "${GCTA_AOCC_ROOT}/lib"
        NO_DEFAULT_PATH)
    if(_aocc_omp_lib)
        set(OpenMP_omp_LIBRARY "${_aocc_omp_lib}")
        message(STATUS "AOCC OpenMP library: ${_aocc_omp_lib}")
        get_filename_component(_aocc_omp_dir "${_aocc_omp_lib}" DIRECTORY)
        list(APPEND CMAKE_BUILD_RPATH   "${_aocc_omp_dir}")
        list(APPEND CMAKE_INSTALL_RPATH "${_aocc_omp_dir}")
    endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

set(GCTA_BLAS_LIBS "")
set(GCTA_BLAS_INCLUDE_DIRS "")
set(GCTA_BLAS_DEFINES "")

if(NOT GCTA_BLAS_BACKEND)
    message(FATAL_ERROR "GCTA_BLAS_BACKEND is required: OpenBLAS, MKL, Accelerate, or AOCL.")
endif()
set(_GCTA_BLAS "${GCTA_BLAS_BACKEND}")

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
    set(BLA_VENDOR Apple)
    find_package(BLAS   REQUIRED)
    find_package(LAPACK REQUIRED)

    if(NOT CMAKE_OSX_SYSROOT)
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    find_path(GCTA_ACCELERATE_INCLUDE_DIR
        NAMES Accelerate.h cblas.h
        HINTS "${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
              "${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/vecLib.framework/Headers"
        NO_DEFAULT_PATH
        DOC "vecLib / Accelerate headers directory")
    if(NOT GCTA_ACCELERATE_INCLUDE_DIR)
        message(FATAL_ERROR "Accelerate/vecLib headers not found inside SDK: ${CMAKE_OSX_SYSROOT}")
    endif()
    message(STATUS "Accelerate include dir: ${GCTA_ACCELERATE_INCLUDE_DIR}")

    list(APPEND GCTA_BLAS_INCLUDE_DIRS "${GCTA_ACCELERATE_INCLUDE_DIR}")
    list(APPEND GCTA_BLAS_LIBS         ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    list(APPEND GCTA_BLAS_DEFINES      GCTA_USE_ACCELERATE ACCELERATE_NEW_LAPACK EIGEN_USE_BLAS)
    set(GCTA_BLAS_PROVIDER "Accelerate" CACHE INTERNAL "")

elseif(_GCTA_BLAS STREQUAL "OpenBLAS" OR _GCTA_BLAS STREQUAL "MKL" OR _GCTA_BLAS STREQUAL "AOCL")
    if(NOT GCTA_BLAS_LIBRARY)
        message(FATAL_ERROR "GCTA_BLAS_LIBRARY must be set for the ${_GCTA_BLAS} backend.")
    endif()
    if(NOT GCTA_BLAS_INCLUDE_DIR)
        message(FATAL_ERROR "GCTA_BLAS_INCLUDE_DIR must be set for the ${_GCTA_BLAS} backend.")
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
        add_link_options("LINKER:-rpath-link,${_blas_lib_dir}")
        if(GCTA_LAPACKE_LIBRARY)
            get_filename_component(_lapack_lib_dir "${GCTA_LAPACKE_LIBRARY}" DIRECTORY)
            list(APPEND CMAKE_BUILD_RPATH   "${_lapack_lib_dir}")
            list(APPEND CMAKE_INSTALL_RPATH "${_lapack_lib_dir}")
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
    endif()

    if(_GCTA_BLAS STREQUAL "OpenBLAS")
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_OPENBLAS EIGEN_USE_LAPACKE EIGEN_USE_BLAS)
        set(GCTA_BLAS_PROVIDER "OpenBLAS" CACHE INTERNAL "")
    elseif(_GCTA_BLAS STREQUAL "AOCL")
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_AOCL EIGEN_USE_LAPACKE EIGEN_USE_BLAS)
        set(GCTA_BLAS_PROVIDER "AOCL" CACHE INTERNAL "")

        get_filename_component(_aocl_lib_dir "${GCTA_BLAS_LIBRARY}" DIRECTORY)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND EXISTS "${_aocl_lib_dir}/libomp.so")
            message(FATAL_ERROR "AOCL compiler/variant mismatch: you are using GCC but the AOCL library is for AOCC.")
        endif()

        foreach(_aocl_dep IN ITEMS libaoclutils.so libblis-mt.so)
            set(_aocl_dep_path "${_aocl_lib_dir}/${_aocl_dep}")
            if(EXISTS "${_aocl_dep_path}" AND NOT "${_aocl_dep_path}" IN_LIST GCTA_BLAS_LIBS)
                list(APPEND GCTA_BLAS_LIBS "${_aocl_dep_path}")
            endif()
        endforeach()
    elseif(_GCTA_BLAS STREQUAL "MKL")
        list(APPEND GCTA_BLAS_DEFINES GCTA_USE_MKL EIGEN_USE_MKL_ALL)
        set(GCTA_BLAS_PROVIDER "MKL" CACHE INTERNAL "")
    endif()
endif()

message(STATUS "BLAS/LAPACK provider: ${GCTA_BLAS_PROVIDER}")

# ----------------- Target Definitions -------------------
set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/include)
set(MAIN_SOURCE "${CMAKE_SOURCE_DIR}/src/main.cpp")

# Glob all .cpp files in src/ except main.cpp
file(GLOB SRCS "${CMAKE_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM SRCS "${MAIN_SOURCE}")
list(APPEND SRCS "${CMAKE_SOURCE_DIR}/src/acat.c")

set(GCTA_LIBS "")
foreach(lib_src ${SRCS})
    get_filename_component(file_name "${lib_src}" NAME_WE)
    string(TOLOWER ${file_name} lib_name)
    add_library(${lib_name} STATIC "${lib_src}")

    target_include_directories(${lib_name} PUBLIC
        ${COMMON_INCLUDES}
        "${CMAKE_SOURCE_DIR}/third_party/plink-ng/2.0/include"
        "${CMAKE_SOURCE_DIR}/third_party/plink-ng/2.0/simde")

    list(APPEND GCTA_LIBS ${lib_name})
endforeach(lib_src)

add_executable(gcta64 ${MAIN_SOURCE})
target_compile_definitions(gcta64 PRIVATE EIGEN_MAX_ALIGN_BYTES=16)
add_subdirectory(main)
target_include_directories(mainV1 PUBLIC ${COMMON_INCLUDES})

set(GCTA_TARGETS gcta64 mainV1 ${GCTA_LIBS})

# ---------------- Compiler settings, LTO/IPO -------------------
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

foreach(target IN LISTS GCTA_TARGETS)
    set_target_properties(${target} PROPERTIES
        UNITY_BUILD ${GCTA_UNITY_BUILD}
        INTERPROCEDURAL_OPTIMIZATION ${GCTA_IPO_ENABLED})
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${COMMON_INCLUDES}
        ${GCTA_ZSTD_INCLUDE_DIRS}
        ${GCTA_ZLIB_INCLUDE_DIRS}
        ${GCTA_BLAS_INCLUDE_DIRS}
        $<$<NOT:$<BOOL:${WIN32}>>:${GSL_INCLUDE_DIRS}>)
    target_compile_definitions(${target} PRIVATE
        ${GCTA_BLAS_DEFINES}
        GCTA_BLAS_BACKEND_STR="${GCTA_BLAS_PROVIDER}")
    target_compile_options(${target} PRIVATE
        $<$<BOOL:${APPLE}>:-Wno-nullability-completeness>
        -pipe
        $<$<CONFIG:Release>:$<IF:$<BOOL:${APPLE}>,-mcpu=native,-march=native>>
        $<$<CONFIG:Release>:-O3>)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX
        $<$<NOT:$<BOOL:${WIN32}>>:GSL::gsl>)
endforeach()

if(GCTA_IPO_ENABLED AND CMAKE_BUILD_TYPE MATCHES "^[Rr]elease$")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(GCTA_COMPILER_IS_AOCC)
            foreach(target IN LISTS GCTA_TARGETS)
                target_compile_options(${target} PRIVATE -flto=full)
                target_link_options(${target}    PRIVATE -flto=full)
            endforeach()
        else()
            foreach(target IN LISTS GCTA_TARGETS)
                target_compile_options(${target} PRIVATE -flto=thin)
                target_link_options(${target}    PRIVATE -flto=thin)
            endforeach()
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        foreach(target IN LISTS GCTA_TARGETS)
            target_compile_options(${target} PRIVATE -flto=auto)
            target_link_options(${target} PRIVATE -flto=auto)
        endforeach()
    endif()
endif()

# Resolve Boost libraries to link
set(_GCTA_BOOST_LINKS "")
if(TARGET Boost::headers)
    list(APPEND _GCTA_BOOST_LINKS Boost::headers)
endif()
if(DEFINED Boost_LIBRARIES AND NOT "${Boost_LIBRARIES}" STREQUAL "")
    list(APPEND _GCTA_BOOST_LINKS ${Boost_LIBRARIES})
endif()

if(DEFINED Boost_INCLUDE_DIRS)
    foreach(target IN LISTS GCTA_TARGETS)
        target_include_directories(${target} PRIVATE ${Boost_INCLUDE_DIRS})
    endforeach()
endif()

foreach(target IN LISTS GCTA_TARGETS)
    target_link_libraries(${target} PRIVATE
        ${_GCTA_BOOST_LINKS}
        $<$<BOOL:${Boost_DIR}>:ZLIB::ZLIB>
        Eigen3::Eigen
        Spectra
        sqlite3
        ${GCTA_ZSTD_LIBS}
        ZLIB::ZLIB)
endforeach()

# Pgenlib definition
add_library(Pgenlib
    third_party/Pgenlib/PgenReader.cpp
    third_party/plink-ng/2.0/include/pgenlib_ffi_support.cc
    third_party/plink-ng/2.0/include/pgenlib_misc.cc
    third_party/plink-ng/2.0/include/pgenlib_read.cc
    third_party/plink-ng/2.0/include/pgenlib_write.cc
    third_party/plink-ng/2.0/include/plink2_base.cc
    third_party/plink-ng/2.0/include/plink2_bgzf.cc
    third_party/plink-ng/2.0/include/plink2_bitmap.cc
    third_party/plink-ng/2.0/include/plink2_bits.cc
    third_party/plink-ng/2.0/include/plink2_fmath.cc
    third_party/plink-ng/2.0/include/plink2_htable.cc
    third_party/plink-ng/2.0/include/plink2_memory.cc
    third_party/plink-ng/2.0/include/plink2_simd.cc
    third_party/plink-ng/2.0/include/plink2_stats.cc
    third_party/plink-ng/2.0/include/plink2_string.cc
    third_party/plink-ng/2.0/include/plink2_text.cc
    third_party/plink-ng/2.0/include/plink2_thread.cc
    third_party/plink-ng/2.0/include/plink2_zstfile.cc
    third_party/plink-ng/2.0/include/pvar_ffi_support.cc)

set_target_properties(Pgenlib PROPERTIES UNITY_BUILD OFF)
target_include_directories(Pgenlib PUBLIC "${CMAKE_SOURCE_DIR}/third_party/plink-ng/2.0/include")
if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(Pgenlib PRIVATE -mfma -mbmi -mbmi2 -mlzcnt)
endif()
target_include_directories(Pgenlib PRIVATE ${GCTA_ZLIB_INCLUDE_DIRS})


target_link_libraries(gcta64 PRIVATE
    mainV1
    ${GCTA_LIBS}
    Pgenlib
    Threads::Threads
    ${GCTA_BLAS_LIBS}
    ${GCTA_ZSTD_LIBS})

if(NOT WIN32)
    target_link_libraries(gcta64 PRIVATE GSL::gsl)
endif()

target_link_libraries(gcta64 PRIVATE sqlite3)

if(LINUX)
    list(PREPEND CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
elseif(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
endif()

if(LINUX OR APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
    install(TARGETS gcta64 mainV1 ${GCTA_LIBS}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
endif()
