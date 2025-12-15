#!/usr/bin/env python3
"""
Evaluate LEL → LLVM pipeline on a set of real equations.

For each record in a JSONL file (default: lel_functions.jsonl) with field
    "lel_equation": "f(x) = x^2 + 1"

we:

  1. Parse into FunctionDef using lel_core.parse_function_def_from_string
  2. Build an LLVM module with build_module_for_functiondef
  3. Generate a C driver that:
       - declares double f(double, ...)
       - declares + defines stub functions for any callees (identity on first arg)
       - tests f on a few test inputs against Python reference (same stub semantics)
  4. Compile with clang and run it.
  5. Count pass / fail.

Calls to other functions are treated as black-box stubs with identity semantics:
    g(x, y, ...) ≈ x

Usage:

    python eval_lel_functions.py \
        --in lel_functions.jsonl \
        --max-functions 20
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from lel_core import (
    FunctionDef,
    Expr,
    Var,
    Const,
    UnaryOp,
    BinOp,
    Call,
    Compare,
    IfExpr,
    parse_function_def_from_string,
)
from lel_codegen_llvm import build_module_for_functiondef
import time
import statistics

# ---------------------------------------------------------------------------
# Python reference evaluator for Expr (with stub calls)
# ---------------------------------------------------------------------------

def eval_expr_py(
    expr: Expr,
    env: Dict[str, float],
    func: Optional[FunctionDef] = None,
    depth: int = 0,
    max_depth: int = 20,
) -> float:
    """
    Evaluate an Expr in Python for given variable environment.

    - env: mapping from variable name -> float
    - func: optional FunctionDef (unused for now)
    - Calls are treated as stubs: we evaluate their first argument and return it.
    """
    if depth > max_depth:
        raise RecursionError("Maximum eval depth exceeded (possible recursion).")

    if isinstance(expr, Const):
        return float(expr.value)

    if isinstance(expr, Var):
        if expr.name not in env:
            raise ValueError(f"Unknown variable {expr.name!r} in reference eval")
        return float(env[expr.name])

    if isinstance(expr, UnaryOp):
        val = eval_expr_py(expr.operand, env, func, depth + 1, max_depth)
        if expr.op == "-":
            return -val
        raise NotImplementedError(f"Unsupported unary op {expr.op!r}")

    if isinstance(expr, BinOp):
        left = eval_expr_py(expr.left, env, func, depth + 1, max_depth)
        right = eval_expr_py(expr.right, env, func, depth + 1, max_depth)

        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "^":
            return math.pow(left, right)

        raise NotImplementedError(f"Unsupported binary op {expr.op!r}")

    if isinstance(expr, Call):
        # Stub semantics: return the value of the first argument (if any),
        # or 0.0 if there are no arguments.
        if not expr.args:
            return 0.0
        return eval_expr_py(expr.args[0], env, func, depth + 1, max_depth)

    if isinstance(expr, Compare):
        left = eval_expr_py(expr.left, env, func, depth + 1, max_depth)
        right = eval_expr_py(expr.right, env, func, depth + 1, max_depth)
        if expr.op == ">":
            return 1.0 if left > right else 0.0
        elif expr.op == "<":
            return 1.0 if left < right else 0.0
        else:
            raise NotImplementedError(f"Unsupported compare op {expr.op!r}")

    if isinstance(expr, IfExpr):
        cond_val = eval_expr_py(expr.cond, env, func, depth + 1, max_depth)
        if cond_val != 0.0:
            return eval_expr_py(expr.then_branch, env, func, depth + 1, max_depth)
        else:
            return eval_expr_py(expr.else_branch, env, func, depth + 1, max_depth)


    raise NotImplementedError(f"Unknown Expr node type: {type(expr)}")


# ---------------------------------------------------------------------------
# Test case generation
# ---------------------------------------------------------------------------

def generate_test_cases(params: List[str]) -> List[Dict[str, float]]:
    """
    Generate a small set of numeric test cases for the given parameter list.
    """
    if len(params) == 0:
        return [{}]

    if len(params) == 1:
        p = params[0]
        values = [-3.0, -1.0, 0.1, 2.0, 4.0]
        return [{p: v} for v in values]

    if len(params) == 2:
        p, q = params
        combos = [
            (-1.0, 0.5),
            (0.1, 0.11),
            (2.4, -1.0),
            (2.2, 1.0),
        ]
        return [{p: a, q: b} for a, b in combos]

    cases = []
    env0 = {name: 0.0 for name in params}
    cases.append(env0)
    env1 = {name: float(i + 1) for i, name in enumerate(params)}
    cases.append(env1)
    return cases


# ---------------------------------------------------------------------------
# Call collection for stub generation
# ---------------------------------------------------------------------------
def has_self_call(expr: Expr, func_name: str) -> bool:
    """
    Return True if expr contains a Call to func_name (i.e., self recursion).
    """
    if isinstance(expr, Call):
        if expr.func_name == func_name:
            return True
        return any(has_self_call(a, func_name) for a in expr.args)

    if isinstance(expr, UnaryOp):
        return has_self_call(expr.operand, func_name)

    if isinstance(expr, BinOp):
        return has_self_call(expr.left, func_name) or has_self_call(expr.right, func_name)

    # Var / Const: no recursion
    return False

def collect_call_arities(expr: Expr, out: Dict[str, int]) -> None:
    """
    Walk an Expr tree and record, for each function name, the maximum
    number of arguments it is called with.
    """
    if isinstance(expr, Call):
        name = expr.func_name
        arity = len(expr.args)
        current = out.get(name, 0)
        if arity > current:
            out[name] = arity
        for a in expr.args:
            collect_call_arities(a, out)
    elif isinstance(expr, UnaryOp):
        collect_call_arities(expr.operand, out)
    elif isinstance(expr, BinOp):
        collect_call_arities(expr.left, out)
        collect_call_arities(expr.right, out)
    # Const / Var: nothing to do


# ---------------------------------------------------------------------------
# C driver generation (with stubs)
# ---------------------------------------------------------------------------

def make_c_driver_source(
    fd: FunctionDef,
    test_cases: List[Dict[str, float]],
    expected_values: List[float],
    call_arities: Dict[str, int],
    eps: float = 1e-6,
) -> str:
    """
    Generate C code for a driver that:
      - declares + defines stub functions for all callees (except fd.name)
      - declares the target function
      - tests it on given cases with expected values
    """
    assert len(test_cases) == len(expected_values)
    n_params = len(fd.params)

    # Prototype for main LEL function
    if n_params == 0:
        proto = f"double {fd.name}(void);"
    else:
        proto_args = ", ".join(["double"] * n_params)
        proto = f"double {fd.name}({proto_args});"

    # Stub functions for any callees other than the main function
    stub_defs: List[str] = []
    for name, arity in call_arities.items():
        if name == fd.name:
            continue
        # Build argument list: (double a0, double a1, ...)
        if arity == 0:
            params = "void"
            ret_expr = "0.0"
        else:
            param_list = [f"double a{i}" for i in range(arity)]
            params = ", ".join(param_list)
            # identity on first arg
            ret_expr = "a0"
        stub_code = f"double {name}({params}) {{ return {ret_expr}; }}"
        stub_defs.append(stub_code)

    stubs_joined = "\n".join(stub_defs)

    # Test cases
    tests_c = []
    for idx, (env, exp) in enumerate(zip(test_cases, expected_values)):
        if n_params == 0:
            call = f"{fd.name}()"
        else:
            arg_vals = [env[name] for name in fd.params]
            arg_str = ", ".join(f"{v:.17g}" for v in arg_vals)
            call = f"{fd.name}({arg_str})"

        test_block = f"""
        {{
            double got = {call};
            double expected = {exp:.17g};
            if (fabs(got - expected) > {eps:.1e}) {{
                printf("FAIL {fd.name} case {idx}: got %f expected %f\\n", got, expected);
                return 1;
            }}
        }}
        """
        tests_c.append(test_block)

    tests_joined = "\n".join(tests_c)

    src = f"""
    #include <math.h>
    #include <stdio.h>

    {proto}

    {stubs_joined}

    int main(void) {{
        {tests_joined}
        printf("PASS {fd.name}\\n");
        return 0;
    }}
    """
    return src


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    func_name: str
    equation: str
    status: str      # "ok", "parse_error", "compile_error", "runtime_error"
    detail: str = ""


def evaluate_function(
    equation: str,
    record_id: str,
    tmpdir: Path,
) -> EvalResult:
    eq_display = equation

    # 1) Parse as LEL FunctionDef
    try:
        src = equation.strip()
        if src.endswith(";") or src.endswith("."):
            src = src[:-1].strip()
        fd = parse_function_def_from_string(src)
    except Exception as e:
        return EvalResult(
            func_name=f"<unknown:{record_id}>",
            equation=eq_display,
            status="parse_error",
            detail=str(e),
        )
    # Skip self-recursive functions (we don't support executing recursion yet)
    if has_self_call(fd.body, fd.name):
        return EvalResult(
            func_name=fd.name,
            equation=eq_display,
            status="unsupported",
            detail="Self-recursive definitions are not evaluated (potential infinite recursion).",
        )

    # 2) Generate test cases
    test_cases = generate_test_cases(fd.params)

    # 3) Compute expected values in Python using stub semantics
    expected_values: List[float] = []
    try:
        for env in test_cases:
            val = eval_expr_py(fd.body, env, func=fd)
            if not (math.isfinite(val)):
                raise ValueError(f"Non-finite value {val}")
            expected_values.append(val)
    except Exception as e:
        return EvalResult(
            func_name=fd.name,
            equation=eq_display,
            status="runtime_error",
            detail=f"Reference eval failed: {e}",
        )

    # 4) Collect call arities for stub generation
    call_arities: Dict[str, int] = {}
    collect_call_arities(fd.body, call_arities)

    # 5) Build LLVM module and write .ll
    module = build_module_for_functiondef(fd, module_name=f"lel_{fd.name}")
    ll_path = tmpdir / f"{fd.name}_{record_id}.ll"
    ll_path.write_text(str(module), encoding="utf-8")

    # 6) Generate C driver with stubs
    driver_src = make_c_driver_source(fd, test_cases, expected_values, call_arities)
    driver_path = tmpdir / f"driver_{fd.name}_{record_id}.c"
    driver_path.write_text(driver_src, encoding="utf-8")

    # 7) Compile with clang
    exe_path = tmpdir / f"test_{fd.name}_{record_id}"
    compile_cmd = [
        "clang",
        "-O2",
        str(ll_path),
        str(driver_path),
        "-o",
        str(exe_path),
        "-lm",
    ]
    comp = subprocess.run(
        compile_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if comp.returncode != 0:
        return EvalResult(
            func_name=fd.name,
            equation=eq_display,
            status="compile_error",
            detail=comp.stderr.strip(),
        )

    # 8) Run the executable
    run = subprocess.run(
        [str(exe_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if run.returncode != 0:
        detail = (run.stdout + "\n" + run.stderr).strip()
        return EvalResult(
            func_name=fd.name,
            equation=eq_display,
            status="runtime_error",
            detail=detail,
        )

    return EvalResult(
        func_name=fd.name,
        equation=eq_display,
        status="ok",
        detail=run.stdout.strip(),
    )


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LEL→LLVM pipeline on functions from a JSONL file."
    )
    p.add_argument(
        "--in",
        dest="in_path",
        default="lel_functions.jsonl",
        help="Input JSONL with 'lel_equation' (default: lel_functions.jsonl).",
    )
    p.add_argument(
        "--max-functions",
        type=int,
        default=1000,
        help="Maximum number of functions to evaluate (default: 20).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_path)

    results: List[EvalResult] = []
    total_considered = 0
    timings: List[float] = []   # end-to-end times for functions we actually run

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        with in_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if args.max_functions is not None and total_considered >= args.max_functions:
                    break
                if not line.strip():
                    continue

                rec = json.loads(line)
                eq = rec.get("lel_equation") or rec.get("typst_equation")
                if not eq:
                    continue

                record_id = str(rec.get("id", total_considered))
                total_considered += 1

                # --- timing starts here ---
                start_t = time.perf_counter()
                res = evaluate_function(eq, record_id, tmpdir)
                end_t = time.perf_counter()
                elapsed = end_t - start_t
                # Only count functions where we actually ran the full pipeline
                # (parse + ref eval + LLVM + clang + run). That corresponds to
                # ones that made it past parse and ref eval, i.e. not "parse_error"
                # or "unsupported".
                if res.status in ("ok", "compile_error", "runtime_error"):
                    timings.append(elapsed)
                # --- timing ends here ---

                results.append(res)
                print(f"[{res.status.upper()}] {record_id}: {res.func_name}")

    # Summary
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print("\n=== Summary ===")
    print(f"Total evaluated (up to max): {len(results)}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")

    # Optional: print details for non-ok results
    for r in results:
        if r.status != "ok":
            print(f"\n--- {r.status.upper()} for {r.func_name} ---")
            print(f"Equation: {r.equation}")
            print(f"Detail: {r.detail}")

    # --- Performance summary ---
    print("\n=== Performance summary (end-to-end) ===")
    if timings:
        total_funcs = len(timings)
        total_time = sum(timings)
        mean_time = total_time / total_funcs
        median_time = statistics.median(timings)
        min_time = min(timings)
        max_time = max(timings)

        print(f"Functions timed: {total_funcs}")
        print(f"Total time:      {total_time:.3f} s")
        print(f"Avg per func:    {mean_time * 1000:.1f} ms")
        print(f"Median per func: {median_time * 1000:.1f} ms")
        print(f"Min / Max:       {min_time * 1000:.1f} ms / {max_time * 1000:.1f} ms")
    else:
        print("No functions reached end-to-end timing (no ok/compile_error/runtime_error cases).")



if __name__ == "__main__":
    main()
