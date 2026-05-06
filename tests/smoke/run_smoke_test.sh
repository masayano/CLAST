#!/usr/bin/env bash
set -euo pipefail

binary_path="${1:?missing binary path}"
target_path="${2:?missing target path}"
query_path="${3:?missing query path}"
output_path="${4:?missing output path}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "Skipping CLAST smoke test because no NVIDIA adapter is visible."
  exit 125
fi

rm -f "${output_path}"
"${binary_path}" -t "${target_path}" -q "${query_path}" -o "${output_path}"
test -s "${output_path}"
