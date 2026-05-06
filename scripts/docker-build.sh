#!/usr/bin/env bash
# Build clast only; use docker-test.sh for ctest. Default BUILD_TESTING=Off (no GTest fetch).
set -euo pipefail

IMAGE="${CLAST_DOCKER_IMAGE:-clast-cuda-dev}"
JOBS="${CLAST_BUILD_JOBS:-4}"
CLAST_CMAKE_BUILD_TESTING="${CLAST_CMAKE_BUILD_TESTING:-Off}"
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

docker run --rm -i "${DOCKER_TT[@]}" \
  -v "${DOCKER_BIND_SRC}:/workspaces/CLAST" \
  "${IMAGE}" \
  bash -c "set -euo pipefail
printf '%s\n' \"[clast] cmake configure -> ${BUILD_DIR} (BUILD_TESTING=${CLAST_CMAKE_BUILD_TESTING})...\"
cmake -S . -B \"${BUILD_DIR}\" -G Ninja -DCLAST_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=${CLAST_CMAKE_BUILD_TESTING} &&
printf '%s\n' \"[clast] cmake --build -j${JOBS}...\" &&
cmake --build \"${BUILD_DIR}\" -j ${JOBS} &&
printf '%s\n' \"[clast] ok: /workspaces/CLAST/${BUILD_DIR}/clast\""
