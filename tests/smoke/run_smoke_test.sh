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

expected_path="$(dirname "${BASH_SOURCE[0]}")/clast-smoke.expected.tsv"
python3 - "${output_path}" "${expected_path}" <<'PYEOF'
import sys, re, math

def normalize_field(s):
    # identity field: "99(99%)" or "100(99.0099%)" — normalize only decimal percentages
    m = re.fullmatch(r'(\d+)\(([0-9.]+)%\)', s)
    if m and '.' in m.group(2):
        return f"{m.group(1)}({float(m.group(2)):.2g}%)"
    # scientific-notation e-values
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

print("Smoke test output matches expected (floats compared at 2 sig figs).")
PYEOF
