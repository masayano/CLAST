# CLAST

CUDA-implemented large-scale alignment search tool.

## Minimal Port Status

This branch adds a modern Linux build path without removing the historical
`Makefile`.

What is verified in this branch:

- containerized CUDA development environment with `nvcc`
- CMake-based Linux build entry
- successful compile with CUDA 12.4 and Boost 1.74
- tiny FASTA smoke inputs for runtime verification

What is still required for runtime execution:

- an NVIDIA driver visible to the runtime environment
- GPU access for Docker or a native Linux/WSL CUDA setup

## Modern Linux Build

### Option 1: Dev Container

The repository includes `.devcontainer/` for a CUDA-enabled development
container based on `nvidia/cuda:12.4.1-devel-ubuntu22.04`.

### Option 2: Docker Scripts

Build the development image:

```bash
docker build -t clast-cuda-dev .devcontainer
```

Compile CLAST:

```bash
./scripts/docker-build.sh
```

Run the current minimal test suite:

```bash
./scripts/docker-test.sh
```

Run the tiny smoke test:

```bash
./scripts/docker-smoke.sh
```

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

## Legacy Build

The original NVIDIA-sample-style `Makefile` is kept for reference. It still
assumes older CUDA layouts and architecture flags, so the CMake path above is
the recommended starting point for modern Linux work.

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
you need to preprocess your database by "preprocessDB" before execute CLAST.  
You can learn how to use it by exeute it without any option.


## LICENSE

GNU GPL

## VERSION

0.1.0 Feb.5,  2014:  
    first version.

0.1.1 Jan.19, 2015:  
    (1) more dense alignment is now available.  
    (2) florting point value is now available for tRAM, qRAM, tVRAM, and qVRAM in command line parameter

0.1.2 Apr.26, 2015:  
    ~~Add "CDeviceHitList_alignmentHits.cuh.7.0" for CUDA 7.0.~~  
    ~~Now CLAST can be built on CUDA 7.0 if it will be renamed "CDeviceHitList_alignmentHits.cuh".~~  
    ~~But it does not work.~~  
    ~~Please use CUDA 5.5 or CUDA 4.x, and Fermi or Kepler architecture GPU.~~

0.1.3 Nov.19, 2015:  
    Fixed a bug.  
    Remove "CDeviceHitList_alignmentHits.cuh.7.0".  
    Remove "doc/READ_ME*".  
    Remove "Makefile.＊".  
    Edit "README.md".

0.1.4 Nov.20, 2015:  
    Refactored "krnlAlignment.cu" and "common.hpp".

0.1.5 Nov.24, 2015:  
    Refactored "CFASTALoader.cpp". 
