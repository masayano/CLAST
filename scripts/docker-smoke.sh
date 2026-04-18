#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CLAST_DOCKER_IMAGE:-clast-cuda-dev}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${CLAST_SMOKE_OUTPUT:-/tmp/clast-smoke.tsv}"
GPU_FLAG="${CLAST_DOCKER_GPU_FLAG:---gpus all}"

docker run --rm ${GPU_FLAG} \
  -v "${ROOT_DIR}:/workspaces/CLAST" \
  -w /workspaces/CLAST \
  "${IMAGE}" \
  bash -lc "cmake -S . -B /tmp/clast-build -G Ninja >/tmp/configure.log && \
    cmake --build /tmp/clast-build -j 4 >/tmp/build.log && \
    /tmp/clast-build/clast -t tests/smoke/target.fa -q tests/smoke/query.fa -o \"${OUTPUT_PATH}\""
