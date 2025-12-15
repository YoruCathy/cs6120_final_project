#!/usr/bin/env python3
"""
Extract LaTeX equations from a folder of .tex files (e.g., downloaded from arXiv)
and save them into a JSONL file.

Supported equation forms:
- \begin{equation} ... \end{equation}
- \begin{equation*} ... \end{equation*}
- \begin{align}, align*, gather, multline (and their * forms)
- \[ ... \]
- $$ ... $$

Output format (JSONL):
{"id": 0, "source_file": "2501.01234/main.tex", "kind": "env", "env": "equation", "equation": "f(x) = x^2 + 1"}
{"id": 1, "source_file": "2501.01234/main.tex", "kind": "display", "env": "\\[", "equation": "E = mc^2"}

Usage:
    python extract_equations.py \
        --root <path> \
        --out equations.jsonl
"""

import argparse
import json
import os
import re
from typing import Iterable, List, Dict, Any, Tuple


DISPLAY_ENVIRONMENTS = [
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract LaTeX equations from .tex files into a JSONL file."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing downloaded .tex files (will be scanned recursively).",
    )
    parser.add_argument(
        "--out",
        default="equations.jsonl",
        help="Output JSONL file (default: equations.jsonl).",
    )
    parser.add_argument(
        "--include-inline",
        action="store_true",
        help="Also extract inline math $...$ (experimental; can be noisy).",
    )
    return parser.parse_args()


def read_file_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_env_equations(text: str) -> List[Dict[str, Any]]:
    """
    Extract equations from LaTeX environments like:
        \begin{equation} ... \end{equation}
        \begin{align*} ... \end{align*}
    Returns a list of dicts: {"kind": "env", "env": env_name, "equation": content}
    """
    results = []

    # Pattern for \begin{env} ... \end{env}
    # We capture env name and content between, non-greedy.
    env_names_pattern = "|".join(re.escape(env) for env in DISPLAY_ENVIRONMENTS)
    env_pattern = re.compile(
        r"\\begin\{(" + env_names_pattern + r")\}(.*?)\\end\{\1\}",
        re.DOTALL | re.MULTILINE,
    )

    for m in env_pattern.finditer(text):
        env_name = m.group(1)
        body = m.group(2).strip()
        if body:
            results.append(
                {
                    "kind": "env",
                    "env": env_name,
                    "equation": body,
                }
            )

    return results


def extract_bracket_equations(text: str) -> List[Dict[str, Any]]:
    """
    Extract \[ ... \] display math.
    """
    results = []

    bracket_pattern = re.compile(r"\\\[(.*?)\\\]", re.DOTALL | re.MULTILINE)

    for m in bracket_pattern.finditer(text):
        body = m.group(1).strip()
        if body:
            results.append(
                {
                    "kind": "display",
                    "env": "\\[",
                    "equation": body,
                }
            )

    return results


def extract_dollar_equations(text: str) -> List[Dict[str, Any]]:
    """
    Extract $$ ... $$ display math.
    """
    results = []

    # Use a non-greedy match; $$...$$ pairs
    dollar_pattern = re.compile(r"\$\$(.*?)\$\$", re.DOTALL | re.MULTILINE)

    for m in dollar_pattern.finditer(text):
        body = m.group(1).strip()
        if body:
            results.append(
                {
                    "kind": "display",
                    "env": "$$",
                    "equation": body,
                }
            )

    return results


def extract_inline_equations(text: str) -> List[Dict[str, Any]]:
    """
    Extract inline $...$ equations.
    Tries to avoid $$...$$ by insisting on single-dollar context.

    This is heuristic and can be noisy, which is why it's optional.
    """
    results = []

    # Rough heuristic: a single $ ... $ where it's not $$.
    inline_pattern = re.compile(
        r"(?<!\$)\$(.+?)(?<!\\)\$(?!\$)",
        re.DOTALL,
    )

    for m in inline_pattern.finditer(text):
        body = m.group(1).strip()
        if body:
            results.append(
                {
                    "kind": "inline",
                    "env": "$",
                    "equation": body,
                }
            )

    return results


def iter_tex_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".tex"):
                yield os.path.join(dirpath, name)


def extract_from_file(path: str, include_inline: bool) -> List[Dict[str, Any]]:
    text = read_file_text(path)

    equations: List[Dict[str, Any]] = []
    equations.extend(extract_env_equations(text))
    equations.extend(extract_bracket_equations(text))
    equations.extend(extract_dollar_equations(text))

    if include_inline:
        equations.extend(extract_inline_equations(text))

    # Attach source_file info later in the main loop.
    return equations


def main():
    args = parse_args()

    root = os.path.abspath(args.root)
    out_path = os.path.abspath(args.out)
    include_inline = args.include_inline

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    eq_id = 0
    num_files = 0
    num_equations = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for tex_path in iter_tex_files(root):
            rel_path = os.path.relpath(tex_path, root)
            num_files += 1

            equations = extract_from_file(tex_path, include_inline=include_inline)
            if not equations:
                continue

            for eq in equations:
                record = {
                    "id": eq_id,
                    "source_file": rel_path,
                    "kind": eq["kind"],
                    "env": eq["env"],
                    "equation": eq["equation"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                eq_id += 1
                num_equations += 1

    print(f"[INFO] Scanned {num_files} .tex files under {root}")
    print(f"[INFO] Extracted {num_equations} equations")
    print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
