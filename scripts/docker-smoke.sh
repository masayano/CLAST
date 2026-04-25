#!/usr/bin/env bash
set -euo pipefail

IMAGE="${CLAST_DOCKER_IMAGE:-clast-cuda-dev}"
JOBS="${CLAST_BUILD_JOBS:-4}"
BUILD_DIR="${CLAST_CMAKE_BUILD_DIR:-build}"
_s=${BASH_SOURCE[0]}
case $_s in
  */*) _s=${_s%/*} ;;
  *) _s=. ;;
esac
_s=$(cd -- "$_s" && pwd)
# shellcheck source=docker-common.sh
source "$_s/docker-common.sh"
unset _s
OUTPUT_PATH="${CLAST_SMOKE_OUTPUT:-build/clast-smoke.tsv}"
GPU_FLAG="${CLAST_DOCKER_GPU_FLAG:---gpus all}"

docker run --rm -i "${DOCKER_TT[@]}" ${GPU_FLAG} \
  -v "${DOCKER_BIND_SRC}:/workspaces/CLAST" \
  "${IMAGE}" \
  bash -c "set -euo pipefail
printf '%s\n' \"[clast] configure + build (no tests) in ${BUILD_DIR}...\"
cmake -S . -B \"${BUILD_DIR}\" -G Ninja -DCLAST_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=Off &&
cmake --build \"${BUILD_DIR}\" -j ${JOBS} &&
printf '%s\n' \"[clast] smoke: writing ${OUTPUT_PATH}...\"
\"${BUILD_DIR}/clast\" -t tests/smoke/target.fa -q tests/smoke/query.fa -o \"${OUTPUT_PATH}\"
printf '%s\n' \"[clast] ok: /workspaces/CLAST/${OUTPUT_PATH}\""
