#!/usr/bin/env python3
"""
LLVM code generator for LEL (LaTeX Equation Language).

Given a function definition string like:

    f(x) = x^2 + 1
    f(x) = (x > 0) ? (x^2) : 0

we:

  1. Parse it into a FunctionDef using lel_core
  2. Build an LLVM module with a function:

         double f(double x);

  3. Emit LLVM IR to a .ll file.
"""

from __future__ import annotations

import argparse
from typing import Dict

from llvmlite import ir

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

# ---------------------------------------------------------------------------
# Expression codegen
# ---------------------------------------------------------------------------

def codegen_expr(
    expr: Expr,
    builder: ir.IRBuilder,
    env: Dict[str, ir.Value],
    module: ir.Module,
    current_fn: ir.Function,
) -> ir.Value:
    """
    Generate LLVM IR for an Expr, returning an ir.Value (double).

    env: mapping from variable name -> ir.Value (function arguments)
    module: LLVM module (needed for pow / external calls)
    current_fn: the LLVM function we are inside (for recursion)
    """
    double = ir.DoubleType()

    if isinstance(expr, Const):
        return ir.Constant(double, expr.value)

    if isinstance(expr, Var):
        if expr.name not in env:
            # Free symbol: treat as constant 0.0
            return ir.Constant(double, 0.0)
        return env[expr.name]

    if isinstance(expr, UnaryOp):
        if expr.op == "-":
            val = codegen_expr(expr.operand, builder, env, module, current_fn)
            return builder.fsub(ir.Constant(double, 0.0), val, name="neg")
        else:
            raise NotImplementedError(f"Unsupported unary op {expr.op!r}")

    if isinstance(expr, BinOp):
        left = codegen_expr(expr.left, builder, env, module, current_fn)
        right = codegen_expr(expr.right, builder, env, module, current_fn)

        if expr.op == "+":
            return builder.fadd(left, right, name="addtmp")
        if expr.op == "-":
            return builder.fsub(left, right, name="subtmp")
        if expr.op == "*":
            return builder.fmul(left, right, name="multmp")
        if expr.op == "/":
            return builder.fdiv(left, right, name="divtmp")
        if expr.op == "^":
            # Use llvm.pow.f64 intrinsic: double pow(double, double)
            pow_fn = module.globals.get("llvm.pow.f64")
            if pow_fn is None:
                pow_ty = ir.FunctionType(double, [double, double])
                pow_fn = ir.Function(module, pow_ty, name="llvm.pow.f64")
            return builder.call(pow_fn, [left, right], name="powtmp")

        raise NotImplementedError(f"Unsupported binary op {expr.op!r}")

    if isinstance(expr, Call):
        # Evaluate arguments first
        arg_vals = [
            codegen_expr(arg, builder, env, module, current_fn)
            for arg in expr.args
        ]

        # Self-call (recursion)
        if expr.func_name == current_fn.name:
            return builder.call(current_fn, arg_vals, name="calltmp")

        # Call to some *other* function: treat as extern double foo(double,...)
        callee = module.globals.get(expr.func_name)
        if callee is None:
            fn_ty = ir.FunctionType(double, [double] * len(arg_vals))
            callee = ir.Function(module, fn_ty, name=expr.func_name)
        return builder.call(callee, arg_vals, name="calltmp")

    if isinstance(expr, Compare):
        left = codegen_expr(expr.left, builder, env, module, current_fn)
        right = codegen_expr(expr.right, builder, env, module, current_fn)
        if expr.op == ">":
            cmp_val = builder.fcmp_ordered(">", left, right, name="cmptmp")
        elif expr.op == "<":
            cmp_val = builder.fcmp_ordered("<", left, right, name="cmptmp")
        else:
            raise NotImplementedError(f"Unsupported compare op {expr.op!r}")
        # Convert i1 to double 0.0/1.0
        return builder.uitofp(cmp_val, double, name="cmp_as_double")

    if isinstance(expr, IfExpr):
        # Evaluate condition as double and compare to 0.0
        cond_val = codegen_expr(expr.cond, builder, env, module, current_fn)
        zero = ir.Constant(double, 0.0)
        cond_bool = builder.fcmp_ordered("!=", cond_val, zero, name="ifcond")

        fn = current_fn
        then_block = fn.append_basic_block(name="then")
        else_block = fn.append_basic_block(name="else")
        merge_block = fn.append_basic_block(name="ifcont")

        builder.cbranch(cond_bool, then_block, else_block)

        # THEN block
        builder.position_at_start(then_block)
        then_val = codegen_expr(expr.then_branch, builder, env, module, current_fn)
        builder.branch(merge_block)
        then_block_end = builder.block

        # ELSE block
        builder.position_at_start(else_block)
        else_val = codegen_expr(expr.else_branch, builder, env, module, current_fn)
        builder.branch(merge_block)
        else_block_end = builder.block

        # MERGE block
        builder.position_at_start(merge_block)
        phi = builder.phi(double, name="iftmp")
        phi.add_incoming(then_val, then_block_end)
        phi.add_incoming(else_val, else_block_end)
        return phi

    raise NotImplementedError(f"Unknown Expr node type: {type(expr)}")


# ---------------------------------------------------------------------------
# Function + module construction
# ---------------------------------------------------------------------------

def build_module_for_functiondef(fd: FunctionDef, module_name: str = "lel_module") -> ir.Module:
    """
    Given a FunctionDef, construct an LLVM module with a single function.

    Every parameter is a double, and the return type is double.
    """
    double = ir.DoubleType()
    module = ir.Module(name=module_name)

    # All parameters are double-typed
    param_types = [double for _ in fd.params]
    fn_ty = ir.FunctionType(double, param_types)
    fn = ir.Function(module, fn_ty, name=fd.name)

    # Name the arguments
    env: Dict[str, ir.Value] = {}
    for i, param_name in enumerate(fd.params):
        arg = fn.args[i]
        arg.name = param_name
        env[param_name] = arg

    # Entry block
    block = fn.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # Codegen body and return
    ret_val = codegen_expr(fd.body, builder, env, module, fn)
    builder.ret(ret_val)

    return module


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate LLVM IR (.ll) from a LEL function definition string."
    )
    p.add_argument(
        "equation",
        help='Function definition, e.g. "f(x) = x^2 + 1"',
    )
    p.add_argument(
        "--out",
        "-o",
        required=True,
        help="Output .ll file path.",
    )
    p.add_argument(
        "--module-name",
        default="lel_module",
        help="Optional LLVM module name.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    src = args.equation.strip()
    # Optionally strip trailing ; or .
    if src.endswith(";") or src.endswith("."):
        src = src[:-1].strip()

    # Parse into FunctionDef using lel_core
    fd = parse_function_def_from_string(src)

    # Build LLVM module
    module = build_module_for_functiondef(fd, module_name=args.module_name)

    # Write IR
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(module))
    print(f"[INFO] Wrote LLVM IR for {fd.name} to {out_path}")


if __name__ == "__main__":
    main()
