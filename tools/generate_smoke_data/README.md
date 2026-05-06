# Smoke-test Data Generator

This is a one-shot tool that produces the FASTA inputs used by the
CLAST smoke test in `tests/smoke/`. Routine CLAST development does not
run this script — the generated files are committed to the repository.

## What it produces

- `tests/smoke/target.fa` — four reference sequences (1000 bp each),
  fetched from NCBI:
  - `Ecoli_K12` — Escherichia coli K-12 MG1655, accession `NC_000913.3`
  - `Scerevisiae_chrI` — Saccharomyces cerevisiae chromosome I, accession `NC_001133.9`
  - `Mjannaschii` — Methanocaldococcus jannaschii DSM 2661, accession `NC_000909.1`
  - `Lambda` — Bacteriophage λ, accession `NC_001416.1`
- `tests/smoke/query.fa` — twelve queries, three per source:
  - `<source>_sub` — 100 bp with one base substituted
  - `<source>_del` —  99 bp with one base deleted
  - `<source>_ins` — 101 bp with one base inserted

The query base segment (100 bp) is extracted from a random position in
the source target. All randomness is seeded with `SEED = 20260425`.

## How to run

From the repository root:

```bash
python3 tools/generate_smoke_data/generate.py
```

Requires Python 3 (standard library only) and outbound HTTPS access to
`eutils.ncbi.nlm.nih.gov`. The script overwrites `tests/smoke/target.fa`
and `tests/smoke/query.fa` unconditionally; review the diff with
`git diff` before committing.

## When to re-run

Only when the smoke data needs to change intentionally:

- adding or removing a source sequence
- changing query construction rules
- a pinned RefSeq accession is retired by NCBI (rare for these entries)

## Reproducibility

The script's output is bit-identical across runs as long as:

- the `SEED` constant in `generate.py` is unchanged
- the pinned RefSeq accessions are still served by NCBI with the same
  bytes (RefSeq accessions are versioned, e.g. `NC_000913.3` rather
  than `NC_000913`, so this is normally guaranteed)

If NCBI returns a different sequence for any pinned accession, the
script will abort with a length-mismatch error rather than silently
producing different output.
