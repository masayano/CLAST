#!/usr/bin/env python3
"""Generate richer smoke-test inputs for CLAST.

Fetches the first 1000 bp of four phylogenetically distant RefSeq
sequences from NCBI E-utilities, writes them as targets, and produces
12 mutation-derived queries (3 per source: substitution, deletion,
insertion).

Re-running this script with the same SEED and the same RefSeq accessions
produces bit-identical output.
"""

from __future__ import annotations

import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SEED = 20260425
TARGET_LENGTH = 1000
QUERY_LENGTH = 100

SOURCES = [
    ("Ecoli_K12",        "NC_000913.3"),
    ("Scerevisiae_chrI", "NC_001133.9"),
    ("Mjannaschii",      "NC_000909.1"),
    ("Lambda",           "NC_001416.1"),
]

EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={accession}&rettype=fasta&retmode=text"
    "&seq_start=1&seq_stop={stop}"
)
NCBI_REQUEST_INTERVAL_SEC = 1.0
BASES = "ACGT"

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = REPO_ROOT / "tests" / "smoke" / "target.fa"
QUERY_PATH = REPO_ROOT / "tests" / "smoke" / "query.fa"


def parse_fasta_single(text: str) -> str:
    seq_chars: list[str] = []
    saw_header = False
    for line in text.splitlines():
        if line.startswith(">"):
            if saw_header:
                break
            saw_header = True
            continue
        if line.startswith(";"):
            continue
        seq_chars.append(line.strip())
    if not saw_header:
        raise RuntimeError("FASTA response missing header line")
    return "".join(seq_chars).upper()


def fetch_first_n_bp(accession: str, n: int) -> str:
    url = EFETCH_URL.format(accession=accession, stop=n)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"NCBI efetch returned HTTP {response.status} for {accession}"
                )
            text = response.read().decode("ascii", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"NCBI efetch failed for {accession}: {exc}") from exc

    seq = parse_fasta_single(text)
    if len(seq) != n:
        raise RuntimeError(
            f"Expected {n} bp from {accession}, got {len(seq)}"
        )
    return seq


def mutate_substitute(rng: random.Random, seq: str) -> str:
    pos = rng.randrange(len(seq))
    original = seq[pos]
    candidates = [b for b in BASES if b != original]
    new_base = rng.choice(candidates)
    return seq[:pos] + new_base + seq[pos + 1 :]


def mutate_delete(rng: random.Random, seq: str) -> str:
    pos = rng.randrange(len(seq))
    return seq[:pos] + seq[pos + 1 :]


def mutate_insert(rng: random.Random, seq: str) -> str:
    pos = rng.randrange(len(seq) + 1)
    new_base = rng.choice(BASES)
    return seq[:pos] + new_base + seq[pos:]


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")


def main() -> int:
    target_records: list[tuple[str, str]] = []
    for i, (name, accession) in enumerate(SOURCES):
        if i > 0:
            time.sleep(NCBI_REQUEST_INTERVAL_SEC)
        print(f"[fetch] {name} ({accession})", file=sys.stderr)
        seq = fetch_first_n_bp(accession, TARGET_LENGTH)
        target_records.append((name, seq))

    write_fasta(TARGET_PATH, target_records)
    print(f"[write] {TARGET_PATH} ({len(target_records)} records)", file=sys.stderr)

    rng = random.Random(SEED)
    query_records: list[tuple[str, str]] = []
    for name, target_seq in target_records:
        start = rng.randrange(TARGET_LENGTH - QUERY_LENGTH + 1)
        segment = target_seq[start : start + QUERY_LENGTH]
        query_records.append((f"{name}_sub", mutate_substitute(rng, segment)))
        query_records.append((f"{name}_del", mutate_delete(rng, segment)))
        query_records.append((f"{name}_ins", mutate_insert(rng, segment)))

    write_fasta(QUERY_PATH, query_records)
    print(f"[write] {QUERY_PATH} ({len(query_records)} records)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
