# GCTAg

`GCTAg` is a fork of [GCTA](https://yanglab.westlake.edu.cn/software/gcta/) that scales to much larger cohorts and exploits the typically dense genetic relationship matrices (GRMs) of agricultural datasets (hence GCT**Ag**). It adds new algorithms (e.g. Hutch++ trace estimation, Woodbury low-rank approximations) and substantially optimised linear algebra to the existing GCTA GRM, REML, MLMA and PCA workflows.

- A summary of the new flags: [docs/changes/GCTAg.md](docs/changes/GCTAg.md)
- Notes on the linear algebra optimisations (incomplete): [docs/development/linalg_optimizations.md](docs/development/linalg_optimizations.md)
- A manuscript describing this work is in preparation.

## Contents

- [Building GCTAg](#building-gctag)
- [Documentation](#documentation)
- [Citation](#citation)
- [About GCTA](#about-gcta)
- [Credits](#credits)
- [License](#license)
- [Questions and help requests](#questions-and-help-requests)

## Building GCTAg

x86\_64 and ARM (including Apple Silicon) CPUs are supported on Linux and macOS. Windows is not currently supported.

### Requirements

**Provided by your system**

| Requirement | Notes |
|---|---|
| C++23 compiler | GCC >= 13 or Clang >= 17 (AppleClang must report version >= 17; Homebrew LLVM also works). AMD's AOCC is supported, see [AOCC](#aocc-amd-compiler). |
| CMake >= 3.28 |  |
| Git | Needed to fetch the `plink-ng` submodule. |
| OpenMP | libgomp with GCC; libomp with Clang (on macOS: `brew install libomp`). |
| zlib | The system shared zlib is always used; it is not downloaded. |
| BLAS/LAPACK backend | Chosen explicitly, see [BLAS backends](#blas-backends). It is not auto-detected. |
| Linux kernel >= 2.6.28 | Linux only. |

**Downloaded automatically by CMake** (FetchContent; internet access is required on the first configure)

| Dependency | Version |
|---|---|
| [Eigen](https://gitlab.com/libeigen/eigen) | 5.0.1 |
| [Spectra](https://spectralib.org/) | 1.2.0 |
| [zstd](https://github.com/facebook/zstd) | 1.5.7 (built as a static library) |
| [Boost](https://www.boost.org/) | 1.92.0 (`algorithm`, `math`, `crc`, `iostreams`) |
| [stdexec](https://github.com/NVIDIA/stdexec) | `nvhpc-26.05` |
| [SQLite](https://www.sqlite.org/) | 3.51.2 amalgamation (only with `-DBGEN_SUPPORT=ON`) |

To use an existing Boost (>= 1.90, with the `iostreams` component) instead of downloading one, pass `-DBoost_DIR=<directory containing BoostConfig.cmake>`.

### Get the source

Clone with submodules (GitHub source archives do **not** include them):

```sh
git clone --recurse-submodules https://github.com/ASLeonard/GCTAg.git
cd GCTAg
```

If you already cloned without `--recurse-submodules`:

```sh
git submodule update --init --recursive
```

### BLAS backends

The backend is required and is set with `GCTA_BLAS_BACKEND`. CMake stops with an error if it is missing.

| `GCTA_BLAS_BACKEND` | Platform | Additional variables |
|---|---|---|
| `AOCL` (recommended on AMD EPYC / Zen4) | Linux | `GCTA_BLAS_LIBRARY` (e.g. `libblis-mt.so`), `GCTA_BLAS_INCLUDE_DIR` |
| `MKL` (2017 or above) | Linux | `GCTA_BLAS_LIBRARY` (`libmkl_rt.so`), `GCTA_BLAS_INCLUDE_DIR` |
| `OpenBLAS` | Linux | `GCTA_BLAS_LIBRARY`, `GCTA_BLAS_INCLUDE_DIR`, and `GCTA_LAPACKE_LIBRARY` if LAPACKE is not bundled in the OpenBLAS library (e.g. distribution packages) |
| `Accelerate` | macOS | none (uses the system framework; requires macOS 13.3 or later) |

The library's directory is added to the binary's RPATH, so the binary resolves the BLAS you built against at run time rather than a system default.

### Configure

**Linux, AOCL** (for example on the ETH Euler cluster):

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
      -DGCTA_BLAS_BACKEND=AOCL \
      -DGCTA_BLAS_LIBRARY=/path/to/aocl/lib/libblis-mt.so \
      -DGCTA_BLAS_INCLUDE_DIR=/path/to/aocl/include \
      -B build/Release -S .
```

**Linux, MKL:**

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
      -DGCTA_BLAS_BACKEND=MKL \
      -DGCTA_BLAS_LIBRARY=/path/to/mkl/lib/libmkl_rt.so \
      -DGCTA_BLAS_INCLUDE_DIR=/path/to/mkl/include \
      -B build/Release -S .
```

At run time, tell `libmkl_rt` which OpenMP layer to use: `export MKL_THREADING_LAYER=GNU` for GCC builds, or `IOMP5` for AOCC builds.

**Linux, OpenBLAS:**

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
      -DGCTA_BLAS_BACKEND=OpenBLAS \
      -DGCTA_BLAS_LIBRARY=/path/to/libopenblas.so \
      -DGCTA_LAPACKE_LIBRARY=/path/to/liblapacke.so \
      -DGCTA_BLAS_INCLUDE_DIR=/path/to/openblas/include \
      -B build/Release -S .
```

**macOS, Accelerate:**

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
      -DGCTA_BLAS_BACKEND=Accelerate \
      -B build/Release -S .
```

### Compile

```sh
cmake --build build/Release
```

The executable is written to `build/Release/GCTAg`.

Release builds use `-O3` with `-march=native` (`-mcpu=native` on macOS), so **the binary is only guaranteed to run on the CPU generation it was built on**. On a cluster, build on the same node type you will run on.

### Install (optional)

```sh
cmake --install build/Release --prefix /path/to/prefix
```

This installs `bin/GCTAg`. On Linux it also installs the desktop entry and icons.

### Optional build settings

| CMake option | Default | Description |
|---|---|---|
| `BGEN_SUPPORT` | `OFF` | Enable BGEN input (SQLite is downloaded automatically). |
| `USE_JEMALLOC` | empty | Path to the `libjemalloc.so` **file**. Links jemalloc with `thp:never` to avoid Linux transparent-huge-page over-allocation on very large triangular allocations such as the GRM. |
| `GCTA_ENABLE_IPO` | `ON` | Link-time optimisation in Release builds. Set `OFF` to reduce link time and memory use. |
| `GCTA_UNITY_BUILD` | `ON` | Unity builds for faster compilation. |
| `Boost_DIR` | empty | Use an existing Boost instead of downloading one. |
| `GCTA_GCC_TOOLCHAIN` | empty | GCC installation root for AOCC, see below. |
| `GCTA_VERBOSE_BLAS` | `OFF` | Print BLAS detection details during configure. |
| `GCTA_LIST_TARGETS` | `OFF` | List imported and local CMake targets during configure. |

### AOCC (AMD compiler)

AOCC is detected automatically from the compiler's `--version` output:

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/path/to/aocc/bin/clang \
      -DCMAKE_CXX_COMPILER=/path/to/aocc/bin/clang++ \
      -DGCTA_GCC_TOOLCHAIN=/path/to/gcc-12-to-14-prefix \
      -DGCTA_BLAS_BACKEND=AOCL \
      -DGCTA_BLAS_LIBRARY=/path/to/aocl/lib/libblis-mt.so \
      -DGCTA_BLAS_INCLUDE_DIR=/path/to/aocl/include \
      -B build/Release -S .
```

- AOCC is Clang 17 based and cannot parse the headers of GCC 15 or newer. If your default GCC is that new, set `GCTA_GCC_TOOLCHAIN` to the prefix (the directory containing `bin/gcc`) of a GCC 12-14 installation. CMake rejects anything outside that range.
- Use the `aocc/` variant of AOCL with AOCC and the `gcc/` variant with GCC. A mismatch is a configure error with GCC and a warning with AOCC.

### Building on HPC clusters

- Dependencies are downloaded once and cached in the build directory. Later re-configures do not need internet access (`FETCHCONTENT_UPDATES_DISCONNECTED=ON` is the default). For fully offline builds, configure once on a node with internet access, or point CMake at local copies with `-DFETCHCONTENT_SOURCE_DIR_EIGEN=...` (likewise `SPECTRA`, `ZSTD`, `BOOST`, `STDEXEC`, `SQLITE`).
- On Linux the BLAS, `libgomp` and `libstdc++` directories are written into the binary as `DT_RPATH`, which takes precedence over `LD_LIBRARY_PATH`. Modules loaded at run time therefore cannot shadow the libraries the binary was built against.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `GCTA_BLAS_BACKEND is required` | Add `-DGCTA_BLAS_BACKEND=<AOCL\|MKL\|OpenBLAS\|Accelerate>`. |
| `GCTA_BLAS_LIBRARY must be set ...` | Pass the full path to the library file (not a directory) and the include directory. |
| `zstd target not found` | A system zstd module can conflict with the downloaded one. Run `module unload zstd`, delete the build directory, and reconfigure. |
| `plink-ng submodule is missing` | Run `git submodule update --init --recursive`. |
| `OpenMP not found on macOS` | `brew install libomp`. |
| AOCC fails to parse standard library headers | Set `GCTA_GCC_TOOLCHAIN` to a GCC 12-14 prefix. |

## Documentation

- [Summary of new flags](docs/changes/GCTAg.md)
- [Linear algebra optimisations](docs/development/linalg_optimizations.md)
- Documentation for the unchanged upstream GCTA functionality: <https://yanglab.westlake.edu.cn/software/gcta/>

## Citation

A manuscript describing GCTAg is in preparation. Until it is published, please cite the repository together with the upstream GCTA paper:

> Yang J, Lee SH, Goddard ME, Visscher PM. GCTA: a tool for genome-wide complex trait analysis. *Am J Hum Genet* 88, 76-82 (2011).

Publications for the individual GCTA modules (GREML, COJO, mtCOJO, MLMA-LOCO, fastBAT, fastGWA, ...) are listed at <https://yanglab.westlake.edu.cn/software/gcta/index.html#Overview>.

## About GCTA

GCTA (Genome-wide Complex Trait Analysis) is a software package that was initially developed to estimate the proportion of phenotypic variance explained by all genome-wide SNPs for a complex trait, and has since been extended to many other analyses of data from genome-wide association studies (GWASs). See the [GCTA website](https://yanglab.westlake.edu.cn/software/gcta/) for more information.

## Credits

Jian Yang developed the original version (before v1.90) of GCTA (with support from Peter Visscher, Mike Goddard and Hong Lee) and currently maintains the upstream software.

Zhili Zheng programmed the fastGWA, fastGWA-GLMM and fastGWA-BB modules, rewrote the I/O and GRM modules, improved the GREML and bivariate GREML modules, extended the PCA module, and improved the SBLUP module.

Zhihong Zhu programmed the mtCOJO and GSMR modules and improved the COJO module.

Longda Jiang and Hailing Fang developed the ACAT-V module.

Jian Zeng rewrote the GCTA-HEreg module.

Andrew Bakshi contributed to the GCTA-fastBAT module.

Angli Xue improved the GSMR module.

Robert Maier improved the GCTA-SBLUP module.

Wujuan Zhong and Judong Shen programmed the fastGWA-GE module.

Alex Leonard programmed the `GCTAg` fork, improving performance in existing `GCTA` bottlenecks and adding new features (Hutch++, Woodbury, etc.) to scale to larger cohorts.

Contributions to the development of the methods implemented in GCTA (e.g. GREML methods, COJO, mtCOJO, MLMA-LOCO, fastBAT, fastGWA and fastGWA-GLMM) can be found in the corresponding publications (<https://yanglab.westlake.edu.cn/software/gcta/index.html#Overview>).

## License

GPLv3. Some parts of the code are released under LGPL, as detailed in the individual files.

## Questions and help requests

For bug reports, questions and feature requests about `GCTAg`, please open an [issue](https://github.com/ASLeonard/GCTAg/issues).

For questions about the original GCTA software, contact Jian Yang at <jian.yang@westlake.edu.cn>.