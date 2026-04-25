# shellcheck shell=bash
# Sourced from docker-*.sh only. BASH_SOURCE[0] = this file; BASH_SOURCE[1] = caller.
if [[ ${#BASH_SOURCE[@]} -lt 2 ]]; then
  echo "error: source docker-common.sh from docker-*.sh" >&2
  exit 1
fi
if [[ -z "${IMAGE:-}" ]]; then
  echo "error: IMAGE must be set before sourcing docker-common.sh" >&2
  exit 1
fi
_entry=${BASH_SOURCE[1]}
_s=${_entry}
case $_s in
  */*) _s=${_s%/*} ;;
  *) _s=. ;;
esac
ROOT_DIR="$(cd "$_s/.." && pwd)"
export MSYS2_ARG_CONV_EXCL="${MSYS2_ARG_CONV_EXCL:+$MSYS2_ARG_CONV_EXCL:}docker"
if command -v cygpath >/dev/null 2>&1; then
  DOCKER_BIND_SRC="$(cygpath -w "$ROOT_DIR")"
else
  DOCKER_BIND_SRC="$ROOT_DIR"
fi
DOCKER_TT=()
if [ -t 1 ]; then
  DOCKER_TT=(-t)
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is not in PATH" >&2
  exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "error: Docker image \"${IMAGE}\" not found. Build: docker build -t ${IMAGE} .devcontainer" >&2
  exit 1
fi
unset _entry _s
