#!/usr/bin/env python3
"""
Convert LaTeX formulas from Parquet to equations.jsonl format.

Each output line looks like:
    {
      "id": 0,
      "source_file": "train-00000-of-00006.parquet",
      "kind": "inline",
      "env": null,
      "equation": "<LaTeX string>"
    }

Usage examples:

    # Single parquet file, LaTeX column named 'latex_formula'
    python parquet_to_equations_jsonl.py \
        --in train-00000-of-00006.parquet \
        --out equations_from_hf.jsonl \
        --column latex_formula

    # All train shards, one JSONL (this is what you want)
    python parquet_to_equations_jsonl.py \
        --in 'train-*.parquet' \
        --out equations_from_hf.jsonl \
        --column latex_formula

    # Directory with multiple parquet files
    python parquet_to_equations_jsonl.py \
        --in data/latex_formulas/train \
        --out equations_from_hf.jsonl \
        --column latex_formula

Optional:
    --max-rows 100000    # for quick testing
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List
import glob

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="in_spec",
        required=True,
        help=(
            "Input .parquet spec: a single file, a directory, or a glob "
            "pattern (e.g., 'train-*.parquet')."
        ),
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output JSONL file (equations.jsonl style).",
    )
    p.add_argument(
        "--column",
        dest="col_name",
        required=True,
        help="Name of the column in the parquet that contains the LaTeX formula.",
    )
    p.add_argument(
        "--max-rows",
        dest="max_rows",
        type=int,
        default=None,
        help="Optional maximum total rows to convert (for testing).",
    )
    return p.parse_args()


def iter_parquet_files(spec: str) -> Iterable[Path]:
    """
    Resolve the --in spec into a list of .parquet files.

    spec can be:
      - a single .parquet file path
      - a directory containing .parquet files
      - a glob pattern (e.g. 'train-*.parquet')
    """
    p = Path(spec)

    # Case 1: exact file
    if p.is_file() and p.suffix == ".parquet":
        yield p
        return

    # Case 2: directory
    if p.is_dir():
        for q in sorted(p.glob("*.parquet")):
            if q.is_file():
                yield q
        return

    # Case 3: treat as glob pattern
    matches = sorted(Path(m) for m in glob.glob(spec))
    if not matches:
        raise FileNotFoundError(
            f"No .parquet files found for spec {spec!r} "
            "(not a file, not a directory, and glob had no matches)."
        )
    for q in matches:
        if q.is_file() and q.suffix == ".parquet":
            yield q


def main():
    args = parse_args()
    in_spec = args.in_spec
    out_path = Path(args.out_path)
    col_name = args.col_name
    max_rows = args.max_rows

    total_written = 0
    next_id = 0

    parquet_files: List[Path] = list(iter_parquet_files(in_spec))
    if not parquet_files:
        raise RuntimeError(f"No .parquet files resolved from spec {in_spec!r}")

    print(f"[INFO] Found {len(parquet_files)} parquet file(s).")
    print(f"[INFO] Writing JSONL to {out_path}")
    print(f"[INFO] Using column '{col_name}' for equations.")

    with out_path.open("w", encoding="utf-8") as fout:
        for pf in parquet_files:
            if max_rows is not None and total_written >= max_rows:
                break

            print(f"[INFO] Reading {pf} ...")
            df = pd.read_parquet(pf)

            if col_name not in df.columns:
                raise KeyError(
                    f"Column '{col_name}' not found in {pf}. "
                    f"Available columns: {list(df.columns)}"
                )

            # Limit rows if close to max_rows
            if max_rows is not None:
                remaining = max_rows - total_written
                if remaining <= 0:
                    break
                df = df.head(remaining)

            for _, row in df.iterrows():
                eq = row[col_name]
                if not isinstance(eq, str):
                    # Skip non-string / missing formulas
                    continue

                rec = {
                    "id": next_id,
                    "source_file": pf.name,
                    "kind": "inline",
                    "env": None,
                    "equation": eq,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                next_id += 1
                total_written += 1

                if max_rows is not None and total_written >= max_rows:
                    break

    print(f"[INFO] Done. Wrote {total_written} equations to {out_path}")


if __name__ == "__main__":
    main()
