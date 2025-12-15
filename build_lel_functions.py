#!/usr/bin/env python3
"""
Filter MiTeX-processed equations down to those that match the LEL function form.

Input JSONL (e.g. mitex_equations.jsonl):
    {"id": 4802, "typst_equation": "r(t) = FK(q(t))", ...}

Output JSONL (e.g. lel_functions.jsonl):
    {"id": 4802, "typst_equation": "...", "lel_equation": "r(t) = FK(q(t))", ...}

Usage:
    python build_lel_functions.py \
        --in mitex_equations.jsonl \
        --out lel_functions.jsonl \
        --max 5000
"""

import argparse
import json
from pathlib import Path

from lel_core import parse_function_def_from_string


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract LEL-style function definitions from MiTeX equations."
    )
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input JSONL with 'typst_equation' field (e.g. mitex_equations.jsonl).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output JSONL with added 'lel_equation' field (e.g. lel_functions.jsonl).",
    )
    p.add_argument(
        "--max",
        dest="max_items",
        type=int,
        default=None,
        help="Optional maximum number of input records to scan (for testing).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    max_items = args.max_items

    num_total = 0      # records with a typst_equation
    num_parsed = 0     # successfully parsed as LEL

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if max_items is not None and idx >= max_items:
                break
            if not line.strip():
                continue

            rec = json.loads(line)
            eq = rec.get("typst_equation")
            if not eq:
                continue

            num_total += 1

            # Try to parse as a LEL function definition
            try:
                _ = parse_function_def_from_string(eq)
            except SyntaxError:
                # Not in our LEL subset, skip
                continue

            num_parsed += 1
            rec["lel_equation"] = eq
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Total with typst_equation:", num_total)
    print("Successfully parsed as LEL:", num_parsed)
    print("Output written to:", out_path)


if __name__ == "__main__":
    main()
