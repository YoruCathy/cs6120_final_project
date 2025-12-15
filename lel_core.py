#!/usr/bin/env python3
"""
Core AST + parser for LEL (LaTeX Equation Language) front-end.

We parse normalized math strings of the form:

    f(x) = x^2 + 1
    g(x, y) = (x + y) / 2
    h() = 42
    f(x) = (x > 0) ? (x^2) : 0

into an AST:

    FunctionDef(name='f', params=['x'], body=BinOp(...))

Grammar (informal):

    function_def  -> (IDENT '(' param_list? ')' | IDENT) '=' expr

    param_list    -> IDENT (',' IDENT)*

    expr          -> ternary

    ternary       -> comparison ('?' comparison ':' comparison)?

    comparison    -> add (('<' | '>') add)*

    add           -> mul (('+' | '-') mul)*
    mul           -> unary (('*' | '/') unary)*
    unary         -> '-' unary | power
    power         -> atom ('^' power)?          # right-associative

    atom          -> NUMBER
                   | IDENT
                   | IDENT '(' arg_list? ')'
                   | '(' expr ')'

    arg_list      -> expr (',' expr)*
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# AST definitions
# ---------------------------------------------------------------------------

class Expr:
    """Base class for expressions."""
    pass


@dataclass
class Var(Expr):
    name: str


@dataclass
class Const(Expr):
    value: float


@dataclass
class UnaryOp(Expr):
    op: str   # '-'
    operand: Expr


@dataclass
class BinOp(Expr):
    op: str   # '+', '-', '*', '/', '^'
    left: Expr
    right: Expr


@dataclass
class Call(Expr):
    func_name: str
    args: List[Expr]


@dataclass
class Compare(Expr):
    op: str   # '<' or '>'
    left: Expr
    right: Expr


@dataclass
class IfExpr(Expr):
    cond: Expr
    then_branch: Expr
    else_branch: Expr


@dataclass
class FunctionDef:
    name: str
    params: List[str]
    body: Expr


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind: str   # 'IDENT', 'NUMBER', 'SYMBOL', 'EOF'
    value: str

    def __repr__(self) -> str:
        return f"Token({self.kind!r}, {self.value!r})"


def tokenize_expr(src: str) -> List[Token]:
    """
    Turn a math string into a flat list of tokens.

    - Identifiers: [A-Za-z]+
    - Numbers: integer or float, e.g. 42, 3.14, .5
    - Symbols: single characters in ()=,+-*/^?:<>
    """
    tokens: List[Token] = []
    i = 0
    n = len(src)

    while i < n:
        c = src[i]

        if c.isspace():
            i += 1
            continue

        # Identifiers: letter followed by letters/digits/underscore
        if c.isalpha():
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            tokens.append(Token("IDENT", src[i:j]))
            i = j
            continue

        # Numbers: int or float
        if c.isdigit() or (c == '.' and i + 1 < n and src[i + 1].isdigit()):
            j = i + 1
            dot_seen = (c == '.')
            while j < n:
                if src[j].isdigit():
                    j += 1
                elif src[j] == '.' and not dot_seen:
                    dot_seen = True
                    j += 1
                else:
                    break
            tokens.append(Token("NUMBER", src[i:j]))
            i = j
            continue

        # Single-character symbols
        if c in "()=,+-*/^?:<>":
            tokens.append(Token("SYMBOL", c))
            i += 1
            continue

        # Anything else: not in LEL v1
        raise SyntaxError(f"Unexpected character in expression: {c!r}")

    tokens.append(Token("EOF", ""))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser for expressions
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, expected_kind: Optional[str] = None,
                expected_value: Optional[str] = None) -> Token:
        tok = self.current
        if expected_kind is not None and tok.kind != expected_kind:
            raise SyntaxError(
                f"Expected token kind {expected_kind}, got {tok.kind} ({tok.value!r})"
            )
        if expected_value is not None and tok.value != expected_value:
            raise SyntaxError(
                f"Expected token value {expected_value!r}, got {tok.value!r}"
            )
        self.pos += 1
        return tok

    def match(self, kind: str, value: Optional[str] = None) -> bool:
        tok = self.current
        if tok.kind != kind:
            return False
        if value is not None and tok.value != value:
            return False
        return True

    # expr        -> ternary
    def parse_expr(self) -> Expr:
        return self.parse_ternary()

    # ternary     -> comparison ('?' comparison ':' comparison)?
    def parse_ternary(self) -> Expr:
        node = self.parse_comparison()
        if self.match("SYMBOL", "?"):
            self.consume("SYMBOL", "?")
            then_e = self.parse_comparison()
            self.consume("SYMBOL", ":")
            else_e = self.parse_comparison()
            node = IfExpr(cond=node, then_branch=then_e, else_branch=else_e)
        return node

    # comparison  -> add (('<' | '>') add)*
    def parse_comparison(self) -> Expr:
        node = self.parse_add()
        while self.match("SYMBOL") and self.current.value in ("<", ">"):
            op = self.consume("SYMBOL").value
            right = self.parse_add()
            node = Compare(op=op, left=node, right=right)
        return node

    # add         -> mul (('+' | '-') mul)*
    def parse_add(self) -> Expr:
        node = self.parse_mul()
        while self.match("SYMBOL") and self.current.value in ("+", "-"):
            op = self.consume("SYMBOL").value
            right = self.parse_mul()
            node = BinOp(op=op, left=node, right=right)
        return node

    # mul         -> unary (('*' | '/') unary)*
    def parse_mul(self) -> Expr:
        node = self.parse_unary()
        while self.match("SYMBOL") and self.current.value in ("*", "/"):
            op = self.consume("SYMBOL").value
            right = self.parse_unary()
            node = BinOp(op=op, left=node, right=right)
        return node

    # unary       -> '-' unary | power
    def parse_unary(self) -> Expr:
        if self.match("SYMBOL", "-"):
            self.consume("SYMBOL", "-")
            operand = self.parse_unary()
            return UnaryOp(op="-", operand=operand)
        return self.parse_power()

    # power       -> atom ('^' power)?   # right-associative
    def parse_power(self) -> Expr:
        left = self.parse_atom()
        if self.match("SYMBOL", "^"):
            self.consume("SYMBOL", "^")
            right = self.parse_power()
            return BinOp(op="^", left=left, right=right)
        return left

    # atom        -> NUMBER | IDENT | IDENT '(' arg_list ')' | '(' expr ')'
    def parse_atom(self) -> Expr:
        tok = self.current

        if tok.kind == "NUMBER":
            self.consume("NUMBER")
            return Const(float(tok.value))

        if tok.kind == "IDENT":
            name = self.consume("IDENT").value
            # function call?
            if self.match("SYMBOL", "("):
                self.consume("SYMBOL", "(")
                args: List[Expr] = []
                if not self.match("SYMBOL", ")"):
                    args.append(self.parse_expr())
                    while self.match("SYMBOL", ","):
                        self.consume("SYMBOL", ",")
                        args.append(self.parse_expr())
                self.consume("SYMBOL", ")")
                return Call(func_name=name, args=args)
            else:
                return Var(name=name)

        if tok.kind == "SYMBOL" and tok.value == "(":
            self.consume("SYMBOL", "(")
            expr = self.parse_expr()
            self.consume("SYMBOL", ")")
            return expr

        raise SyntaxError(f"Unexpected token in atom: {tok.kind} {tok.value!r}")


# ---------------------------------------------------------------------------
# Top-level: parse a function definition
# ---------------------------------------------------------------------------

import re

FUNC_NAME_RE = r"[A-Za-z][A-Za-z0-9_]*"
PARAM_NAME_RE = r"[A-Za-z][A-Za-z0-9_]*"

def parse_function_def_from_string(src: str) -> FunctionDef:
    src = src.strip()
    if "=" not in src:
        raise SyntaxError("Function definition must contain '='.")
    lhs, rhs = src.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    # First, try function-header form: name(params)
    m = re.match(rf"^({FUNC_NAME_RE})\s*\(([^)]*)\)\s*$", lhs)
    if m:
        func_name = m.group(1)
        arg_str = m.group(2).strip()
        if arg_str == "":
            params: List[str] = []
        else:
            raw_params = [p.strip() for p in arg_str.split(",")]
            if any(not re.fullmatch(PARAM_NAME_RE, p) for p in raw_params):
                raise SyntaxError(f"Invalid parameter list: {arg_str!r}")
            params = raw_params
    else:
        # Fallback: treat plain NAME = expr as a 0-arg function
        if not re.fullmatch(FUNC_NAME_RE, lhs):
            raise SyntaxError(f"Invalid function header: {lhs!r}")
        func_name = lhs
        params = []

    tokens = tokenize_expr(rhs)
    parser = Parser(tokens)
    body = parser.parse_expr()
    if not parser.match("EOF"):
        raise SyntaxError(f"Unexpected extra tokens at end of expression: {parser.current}")
    return FunctionDef(name=func_name, params=params, body=body)


# ---------------------------------------------------------------------------
# Tiny manual test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "f(x) = x^2 + 1",
        "g(x, y) = (x + y) / 2",
        "h() = 42",
        "k(x) = -(x^3 - 1) / 2",
        "fact(n) = n * fact(n - 1)",
        "f(x) = (x > 0) ? (x^2) : 0",
        "g(x) = (x < 1) ? (x) : (2 * x)",
    ]
    for t in tests:
        print("====", t)
        try:
            fd = parse_function_def_from_string(t)
            print(fd)
        except SyntaxError as e:
            print("Syntax error:", e)
