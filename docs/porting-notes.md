# CLAST Porting Notes

## Scope

This branch targets a minimal port of CLAST to a modern Linux CUDA toolchain.
The goal is to recover buildability first, then prove a small end-to-end run.

Out of scope for this phase:

- algorithm changes
- score model changes
- Windows-native support
- scientific validation of alignment quality

## Current Baseline

- The original build is driven by `Makefile` and `findcudalib.mk`.
- The original `Makefile` hard-codes legacy GPU architectures from `sm_20` to
  `sm_50`.
- Runtime and CLI behavior are defined by `main.cu`, `CHostSetting.cu`, and
  `PARAMETER_GUIDE`.
- The codebase relies on legacy Thrust functor base classes such as
  `thrust::binary_function` and `thrust::unary_function`.

## Modernization Strategy

1. Add a reproducible CUDA development environment with `nvcc`.
2. Add a modern CMake build entry without deleting the legacy `Makefile`.
3. Fix compile breaks caused by modern CUDA/Thrust APIs.
4. Verify the port with a tiny FASTA smoke test.

## Known Compatibility Hotspots

### Build System

- `Makefile` assumes `/usr/local/cuda` and old `-gencode` values.
- `findcudalib.mk` is tuned for older Unix-style CUDA installs.

### CUDA / Thrust

- `utilResultSorting.cuh`
- `CStridedRange.cuh`
- `CDeviceHitList_sortSeeds.cuh`

These files inherit from Thrust function object base classes removed from newer
toolchains.

### Runtime

- `main.cu` uses a busy-wait sleep loop for optional GPU cooldown.
- CLAST still assumes a GPU-oriented execution flow even for smoke tests.

## Verification Targets

For this branch, the minimum success bar is:

1. `nvcc --version` works in the documented development environment.
2. `cmake -S . -B build` succeeds.
3. `cmake --build build -j` succeeds.
4. `clast` can process a tiny FASTA pair without crashing.

## Verification Results In This Environment

- `docker run --rm clast-cuda-dev nvcc --version` succeeded.
- `cmake` configure and full build succeeded in the Docker image.
- On this Windows host, Docker bind mounts work best when the build directory is
  kept inside the container, such as `/tmp/clast-build`.
- A smoke run reached CLAST startup and parsed the tiny FASTA files correctly,
  but runtime execution failed with `cudaErrorInsufficientDriver` because this
  environment does not expose an NVIDIA adapter to Docker.

This means the branch is verified for compile-time modernization here, while
full runtime verification still requires a GPU-enabled Linux or WSL setup.
