#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CLAST_DOCKER_IMAGE:-clast-cuda-dev}"
JOBS="${CLAST_BUILD_JOBS:-4}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm \
  -e CLAST_BUILD_JOBS="${JOBS}" \
  -v "${ROOT_DIR}:/workspaces/CLAST" \
  -w /workspaces/CLAST \
  "${IMAGE}" \
  bash -lc 'cmake -S . -B /tmp/clast-build -G Ninja -DCLAST_CUDA_ARCHITECTURES=75 && \
    cmake --build /tmp/clast-build -j "${CLAST_BUILD_JOBS}" && \
    ctest --test-dir /tmp/clast-build --output-on-failure'
