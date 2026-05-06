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

docker run --rm -i "${DOCKER_TT[@]}" \
  -v "${DOCKER_BIND_SRC}:/workspaces/CLAST" \
  "${IMAGE}" \
  bash -c "set -euo pipefail
printf '%s\n' \"[clast] cmake (tests on) + build + ctest in ${BUILD_DIR}...\"
cmake -S . -B \"${BUILD_DIR}\" -G Ninja -DCLAST_CUDA_ARCHITECTURES=75 &&
cmake --build \"${BUILD_DIR}\" -j ${JOBS} &&
ctest --test-dir \"${BUILD_DIR}\" --output-on-failure"
