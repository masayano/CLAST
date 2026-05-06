#!/usr/bin/env bash
set -euo pipefail

binary_path="${1:?missing binary path}"
query_path="${2:?missing query path}"
target_path="${3:?missing target path}"

test -x "${binary_path}"
test -s "${query_path}"
test -s "${target_path}"
