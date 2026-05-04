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

smoke_dir="$(dirname "${BASH_SOURCE[0]}")"
output_stem="${output_path%.tsv}"

run_and_check() {
  local mode="$1"        # "global" or "local"
  local local_flag="$2"  # 0 or 1
  local out="${output_stem}.${mode}.tsv"
  local expected="${smoke_dir}/clast-smoke.${mode}.expected.tsv"

  echo "--- smoke test: ${mode} alignment ---"
  rm -f "${out}"
  "${binary_path}" -t "${target_path}" -q "${query_path}" -o "${out}" -local "${local_flag}"
  test -s "${out}"

  python3 - "${out}" "${expected}" <<'PYEOF'
import sys, re

def normalize_field(s):
    m = re.fullmatch(r'(\d+)\(([0-9.]+)%\)', s)
    if m and '.' in m.group(2):
        return f"{m.group(1)}({float(m.group(2)):.2g}%)"
    if re.fullmatch(r'[+-]?[0-9]*\.?[0-9]+[eE][+-]?[0-9]+', s):
        return f"{float(s):.2g}"
    return s

def normalize_line(line):
    return "\t".join(normalize_field(f) for f in line.rstrip("\n").split("\t"))

got_path, exp_path = sys.argv[1], sys.argv[2]
got = sorted(normalize_line(l) for l in open(got_path) if l.strip())
exp = sorted(normalize_line(l) for l in open(exp_path) if l.strip())

if got != exp:
    print("SMOKE TEST FAILED: output does not match expected")
    print(f"\n--- expected ({exp_path}) ---")
    for l in exp: print(l)
    print(f"\n--- got ({got_path}) ---")
    for l in got: print(l)
    sys.exit(1)

print(f"OK (floats compared at 2 sig figs)")
PYEOF
}

run_and_check global 0
run_and_check local  1

echo "All smoke tests passed."
