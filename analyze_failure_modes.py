#!/usr/bin/env python3
"""
Analyze failure modes for the real-equation pipeline:

    original equations  ->  MiTeX (Typst)  ->  LEL functions

We assume three JSONL files:

  1) --orig: equations_from_hf.jsonl (or equations.jsonl)
     Each line: {"id": ..., "equation": "<latex ...>", ...}

  2) --mitex: hf_mitex_equations.jsonl
     Each line: {"id": ..., "equation": "...", "typst_equation": "...", ...}

  3) --lel: hf_lel_functions.jsonl
     Each line: {"id": ..., "lel_equation": "...", ...}

We report:

  - Counts for each stage.
  - IDs that failed at the MiTeX stage.
  - IDs that failed at the LEL stage.
  - A small random sample of failures from each category, with the equations.

Usage example:

    python analyze_failure_modes.py \
        --orig equations_from_hf.jsonl \
        --mitex hf_mitex_equations.jsonl \
        --lel hf_lel_functions.jsonl \
        --sample 10
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True,
                   help="Original equations JSONL (e.g., equations_from_hf.jsonl).")
    p.add_argument("--mitex", required=True,
                   help="MiTeX-success JSONL (e.g., hf_mitex_equations.jsonl).")
    p.add_argument("--lel", required=True,
                   help="LEL-success JSONL (e.g., hf_lel_functions.jsonl).")
    p.add_argument("--sample", type=int, default=10,
                   help="How many examples to print per failure category.")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    args = parse_args()

    orig_path = Path(args.orig)
    mitex_path = Path(args.mitex)
    lel_path = Path(args.lel)

    print(f"[INFO] Loading original equations from {orig_path}")
    orig_recs = load_jsonl(orig_path)
    print(f"[INFO] Loading MiTeX equations from {mitex_path}")
    mitex_recs = load_jsonl(mitex_path)
    print(f"[INFO] Loading LEL functions from {lel_path}")
    lel_recs = load_jsonl(lel_path)

    # Index by id
    orig_by_id = {rec["id"]: rec for rec in orig_recs}
    mitex_by_id = {rec["id"]: rec for rec in mitex_recs}
    lel_by_id = {rec["id"]: rec for rec in lel_recs}

    orig_ids = set(orig_by_id.keys())
    mitex_ids = set(mitex_by_id.keys())
    lel_ids = set(lel_by_id.keys())

    n_orig = len(orig_ids)
    n_mitex = len(mitex_ids)
    n_lel = len(lel_ids)

    print()
    print("=== Coverage Summary ===")
    print(f"Original equations:       {n_orig}")
    print(f"After MiTeX (Typst):      {n_mitex} ({n_mitex / n_orig:.2%} of original)")
    print(f"After LEL parsing:        {n_lel} ({n_lel / n_orig:.2%} of original)")
    if n_mitex > 0:
        print(f"  ... LEL / MiTeX:        {n_lel} ({n_lel / n_mitex:.2%} of MiTeX-success)")

    # Failure sets
    mitex_fail_ids = sorted(orig_ids - mitex_ids)
    lel_fail_ids = sorted(mitex_ids - lel_ids)
    success_ids = sorted(lel_ids)

    print()
    print("=== Failure Categories ===")
    print(f"MiTeX failures (orig but not MiTeX):         {len(mitex_fail_ids)}")
    print(f"LEL parse failures (MiTeX but not LEL):     {len(lel_fail_ids)}")
    print(f"Pipeline successes (survive all stages):    {len(success_ids)}")

    # Helper to sample IDs
    def sample_ids(ids, k):
        if not ids:
            return []
        if len(ids) <= k:
            return list(ids)
        return random.sample(ids, k)

    # Sample MiTeX failures
    print()
    print("=== Sample: MiTeX failures (raw LaTeX equations) ===")
    for eq_id in sample_ids(mitex_fail_ids, args.sample):
        rec = orig_by_id.get(eq_id, {})
        eq = rec.get("equation", "<no equation field>")
        src = rec.get("source_file", "<unknown>")
        print(f"- id={eq_id} source={src}")
        print(f"  equation: {eq}")
        print()

    # Sample LEL failures
    print()
    print("=== Sample: LEL parse failures (MiTeX succeeded, LEL failed) ===")
    for eq_id in sample_ids(lel_fail_ids, args.sample):
        rec = mitex_by_id.get(eq_id, {})
        orig_eq = rec.get("equation", "<no original equation field>")
        typ = rec.get("typst_equation", "<no typst_equation field>")
        src = rec.get("source_file", "<unknown>")
        print(f"- id={eq_id} source={src}")
        print(f"  original: {orig_eq}")
        print(f"  typst:    {typ}")
        print()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
