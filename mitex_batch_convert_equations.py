#!/usr/bin/env python3
"""
Run LaTeX equations through MiTeX CLI and attach the resulting (cleaned) math string.

Only equations that successfully pass through MiTeX and yield a Typst math line
are written to the output JSONL.

Input JSONL (e.g. equations.jsonl):
    {"id": 0, "source_file": "...", "kind": "env", "env": "equation",
     "equation": "f(x) = x^2 + 1"}

Output JSONL (e.g. mitex_equations.jsonl):
    {"id": 0, ..., "equation": "f(x) = x^2 + 1",
     "typst_equation": "f(x) = x^2 + 1"}

Usage:
    python mitex_batch_convert_equations.py \
        --in equations.jsonl \
        --out mitex_equations.jsonl \
        --max 10
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input equations JSONL file (from extract_equations.py).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output JSONL file with added 'typst_equation' field.",
    )
    p.add_argument(
        "--max",
        dest="max_items",
        type=int,
        default=None,
        help="Optional max number of equations to process (for testing).",
    )
    return p.parse_args()


# ----------------- cleaning helpers ----------------- #

def strip_line_comments(eq: str) -> str:
    """Remove full-line and trailing % comments."""
    lines = []
    for line in eq.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        if "%" in line:
            line = line.split("%", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def remove_simple_wrappers(eq: str) -> str:
    """Remove some common wrappers that are irrelevant for math content."""
    eq = eq.replace(r"\begin{equation}", "")
    eq = eq.replace(r"\end{equation}", "")
    eq = eq.replace(r"\begin{equation*}", "")
    eq = eq.replace(r"\end{equation*}", "")
    eq = eq.replace(r"\begin{align}", "")
    eq = eq.replace(r"\end{align}", "")
    eq = eq.replace(r"\begin{align*}", "")
    eq = eq.replace(r"\end{align*}", "")

    eq = eq.replace(r"\nonumber", "")

    eq = re.sub(r"\\label\{[^}]*\}", "", eq)

    return eq


def simplify_wrapped_commands(eq: str) -> str:
    """
    Turn things like \mathcal{X}, \mathrm{foo}, etc. into just X, foo.
    """
    wrapper_cmds = [
        "mathcal",
        "mathrm",
        "mathbf",
        "mathbb",
        "operatorname",
        "text",
        "operatornamewithlimits",
    ]
    for cmd in wrapper_cmds:
        pattern = rf"\\{cmd}\{{([^}}]+)\}}"
        eq = re.sub(pattern, r"\1", eq)
    return eq


def remove_vector_macros(eq: str) -> str:
    """
    Remove vector macros like \v, \vg, etc.:
        \vg \phi -> \phi
        \v r -> r
    """
    eq = re.sub(r"\\vg\s*", "", eq)
    eq = re.sub(r"\\v\s*", "", eq)
    return eq


def simplify_dot_accents(eq: str) -> str:
    """
    Simplify \dot{x}, \ddot{x} to just x.
    """
    eq = re.sub(r"\\ddot\{([^}]+)\}", r"\1", eq)
    eq = re.sub(r"\\dot\{([^}]+)\}", r"\1", eq)
    return eq


def clean_equation(eq: str) -> str:
    """
    Combine all cleaning steps into a single pass.
    """
    eq = strip_line_comments(eq)
    eq = remove_simple_wrappers(eq)
    eq = simplify_wrapped_commands(eq)
    eq = remove_vector_macros(eq)
    eq = simplify_dot_accents(eq)

    eq = re.sub(r"[ \t]+", " ", eq)
    return eq.strip()


# ----------------- MiTeX interaction ----------------- #

def run_mitex_direct(raw_eq: str) -> Optional[str]:
    """
    Given a raw LaTeX equation string, clean it, write to a temp file,
    run `mitex compile eq.tex out.typ`, and extract a Typst math line.

    Returns None if anything fails.
    """
    cleaned = clean_equation(raw_eq)
    if not cleaned:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        eq_path = tmpdir_path / "eq.tex"
        out_path = tmpdir_path / "out.typ"

        eq_path.write_text(cleaned + "\n", encoding="utf-8")

        cmd = ["mitex", "compile", str(eq_path), str(out_path)]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        if proc.returncode != 0:
            # MiTeX couldn't handle this equation
            return None

        typst_text = out_path.read_text(encoding="utf-8", errors="ignore")
        return extract_and_clean_typst_math(typst_text)


def extract_and_clean_typst_math(text: str) -> Optional[str]:
    """
    Given a .typ file produced by MiTeX, return a cleaned math string.

    We:
      - Skip preamble lines starting with '#' or '//'
      - Take the first remaining non-empty line
      - Unescape things like '\(' -> '(', '\)' -> ')', '\^' -> '^'
    """
    lines = [l.rstrip() for l in text.splitlines()]
    candidate = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("//"):
            continue
        candidate = stripped
        break

    if candidate is None:
        return None

    replacements = {
        r"\(": "(",
        r"\)": ")",
        r"\^": "^",
    }
    for old, new in replacements.items():
        candidate = candidate.replace(old, new)

    return candidate


# ----------------- main ----------------- #

def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    num_ok = 0
    num_fail = 0
    num_total = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if args.max_items is not None and idx >= args.max_items:
                break

            if not line.strip():
                continue

            record = json.loads(line)
            num_total += 1

            eq = record.get("equation", "").strip()
            eq_id = record.get("id", idx)

            if not eq:
                num_fail += 1
                print(f"[SKIP] {eq_id} (empty equation)")
                continue

            typst_eq = run_mitex_direct(eq)

            if typst_eq is None:
                num_fail += 1
                print(f"[FAIL] {eq_id}")
                continue

            # Only keep successful ones
            record["typst_equation"] = typst_eq
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            num_ok += 1
            print(f"[OK] {eq_id}")

    print(f"[INFO] Processed {num_total} equations from {in_path}")
    print(f"[INFO]   Saved (OK): {num_ok}")
    print(f"[INFO]   Skipped/Failed: {num_fail}")
    print(f"[INFO] Output written to {out_path}")


if __name__ == "__main__":
    main()
