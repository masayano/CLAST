# CLAST

CUDA-implemented large-scale alignment search tool.

## Build Status

This repository provides a modern Linux build path based on `CMake`.

What is verified in this branch:

- containerized CUDA development environment with `nvcc`
- CMake-based Linux build entry
- successful compile with CUDA 12.4 and Boost 1.74
- tiny FASTA smoke inputs for runtime verification

What is still required for runtime execution:

- an NVIDIA driver visible to the runtime environment
- GPU access for Docker or a native Linux/WSL CUDA setup

## Repository Layout

Current source layout in this branch:

- `src/cli/`: CLI entrypoint and command-line parsing
- `src/host/`: host-side orchestration and FASTA loading
- `src/host/seq/`: `CHostSeqList` (base), `query` / `target` (pairs with `src/device/seq/`)
- `src/device/`: CUDA device-side data structures and kernels support
- `src/device/hit/`: hit list and Thrust host seed helpers
- `src/device/seq/`: `CDeviceSeqList` (base), `query` / `target` specializations
- `src/kernel/`: CUDA kernel translation units
- `src/util/`: shared utilities (e.g. reverse complement, FASTA string and file size helpers)
- `src/test_support/`: test-only support code
- `tests/unit/device/hit/`: unit tests for `src/device/hit/*`; compiled into `clast_unit_tests` (CUDA required)
- `tests/unit/util/`: utility unit tests; `resultSortingTest` and `utilAddSequenceTest` compiled into `clast_unit_tests`, the rest into `clast_host_unit_tests`
- `tests/unit/host/`: host-side unit tests; `CHostSchedularTest` and `CHostResultHolderTest` compiled into `clast_unit_tests`, `CFASTALoaderTest` into `clast_host_unit_tests`
- `tests/unit/cli/`: CLI unit tests; compiled into `clast_host_unit_tests` (no GPU required)
- `tests/smoke/`: tiny FASTA files and runtime smoke scripts for `ctest`
- `tools/preprocess_db/`: database preprocessing helper source
- `tools/divide_query/`: query-splitting helper source

## Modern Linux Build

### Option 1: Dev Container

The repository includes `.devcontainer/` for a CUDA-enabled development
container based on `nvidia/cuda:12.4.1-devel-ubuntu22.04`.

### Option 2: Docker Scripts

Build the development image (required once, from the repository root):

```bash
docker build -t clast-cuda-dev .devcontainer
```

Compile CLAST (Ninja, artifacts under `build/` on the host; default `BUILD_TESTING` is
`Off` in this script so `cmake` does not need to fetch GoogleTest; set
`CLAST_CMAKE_BUILD_TESTING=On` if you also want the unit test binary in one step):

```bash
./scripts/docker-build.sh
```

Run the current minimal test suite (uses `build/`; first run needs network to
fetch GoogleTest for configuration):

```bash
./scripts/docker-test.sh
```

Run the tiny smoke test (requires a GPU; uses `CLAST_DOCKER_GPU_FLAG` if you
need to change `docker run` GPU options):

```bash
./scripts/docker-smoke.sh
```

On **Windows (PowerShell)**, the `.sh` files are not run by bash by default, so
use the wrappers: `.\scripts\docker-build.ps1`, `.\scripts\docker-test.ps1`, or
`.\scripts\docker-smoke.ps1`. The wrapper **prefers Git for Windows’ `bash`**
over WSL (Docker works with either; WSL can report `Wsl/Service/E_UNEXPECTED`
while Git bash is fine). To try WSL first instead, set
`$env:CLAST_POWERSHELL_BASH_ORDER = "wsl-first"`, or run
`"C:\Program Files\Git\bin\bash.exe" ./scripts/docker-build.sh` from the repo root.

If the runtime environment has no GPU driver, the smoke script will fail with a
CUDA driver/runtime mismatch. Compilation should still succeed.
The `ctest` suite keeps that smoke path as a conditional test: it passes the
build artifact check and skips runtime smoke when no NVIDIA adapter is visible.

### Option 3: Native Linux

Requirements:

1. CUDA Toolkit with `nvcc`
2. CMake 3.22+
3. Boost
4. Ninja or Make

Build:

```bash
cmake -S . -B build -G Ninja -DCLAST_CUDA_ARCHITECTURES=75
cmake --build build -j 4
```

Override `CLAST_CUDA_ARCHITECTURES` as needed for your GPU generation.

## Usage

Check `PARAMETER_GUIDE` and run `clast` with at least:

- `-t` target FASTA
- `-q` query FASTA
- `-o` output path

## CAUTION

   Result may contain odd result due to GPU memory error.

## FORM OF RESULT FILE

   Result file is sepalated by tab.

0: queryLabel  
1: query side start index  
2: query side hit length  
3: query strand ("+" or "-")  
4: targetLabel  
5: target side start index  
6: target side hit length  
7: identity ("match num" / "query side hit length" * 100 %)  
8: score  
9: E-value

## FOR LARGE REFERENCE SEQUENCES

If your database contains large (refer "-tRAM" and "-tVRAM" option) sequences,  
you may need the preprocessing helper under
`tools/preprocess_db/preprocessDB.cpp` before executing CLAST.
You can learn how to use that helper by building it and running it without any
option.


## LICENSE

GNU GPL

## VERSION

0.1.0 Feb.5,  2014:  
    first version.

0.1.1 Jan.19, 2015:  
    (1) more dense alignment is now available.  
    (2) florting point value is now available for tRAM, qRAM, tVRAM, and qVRAM in command line parameter

0.1.2 Apr.26, 2015:  
    ~~Add "alignmentHits.cuh.7.0" for CUDA 7.0.~~  
    ~~Now CLAST can be built on CUDA 7.0 if it will be renamed "alignmentHits.cuh".~~  
    ~~But it does not work.~~  
    ~~Please use CUDA 5.5 or CUDA 4.x, and Fermi or Kepler architecture GPU.~~

0.1.3 Nov.19, 2015:  
    Fixed a bug.  
    Remove "alignmentHits.cuh.7.0".  
    Remove "doc/READ_ME*".  
    Remove "Makefile.＊".  
    Edit "README.md".

0.1.4 Nov.20, 2015:  
    Refactored "krnlAlignment.cu" and "common.hpp".

0.1.5 Nov.24, 2015:  
    Refactored "CFASTALoader.cpp".

0.2.0 May.6, 2026:  
    Modernized the codebase. Ported to CUDA 12 with a CMake-based build system,
    reorganized sources into a structured layout, and substantially expanded the
    test suite with unit tests and smoke tests.

0.2.1 May.6, 2026:  
    Minor README fixes.

0.2.2 May.6, 2026:  
    Eliminated dead code.

0.2.3 Jun.11, 2026:  
    Fixed two out-of-bounds device-memory reads (cudaErrorIllegalAddress) that
    could crash on multi-contig reference targets: a strided overread in
    "createHashIndex" and a permutation_iterator lookup at index -1 in
    "createRawSeedList".

0.2.4 Jun.11, 2026:  
    Fixed PARAMETER_GUIDE: corrected the "-gap" default value (8, not 16),
    and documented that "-tVRAM"/"-qVRAM" only take effect up to the chunk
    size set by "-tRAM"/"-qRAM".
