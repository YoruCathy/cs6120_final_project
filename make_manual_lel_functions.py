#!/usr/bin/env python3
import json
from pathlib import Path

OUT = Path("manual_lel_functions.jsonl")

CASES = [
    # ----- Arithmetic & polynomials -----
    ("arith_1", "f1(x) = x + 1"),
    ("arith_2", "f2(x) = x^2 + 2*x + 1"),
    ("arith_3", "f3(x) = x^3 - 3*x + 2"),
    ("arith_4", "lin(x) = 3*x - 5"),
    ("arith_5", "avg2(x, y) = (x + y) / 2"),
    ("arith_6", "lin3(x, y, z) = x + 2*y - 3*z"),

    # ----- Powers & fractions -----
    ("power_1", "pow2(x) = x^2"),
    ("power_2", "pow_chain(x) = x^2^3"),  # parses as x^(2^3) with right-assoc ^
    ("frac_1", "ratio(x, y) = (x^2 - y^2) / (x + y)"),
    ("frac_2", "harmonic(x, y) = 2 / (1/x + 1/y)"),

    # ----- Zero-arg functions -----
    ("zero_1", "c = 42"),
    ("zero_2", "offset() = -5.5"),

    # ----- Conditionals (ternary) -----
    ("cond_1", "relu(x) = (x > 0) ? (x) : 0"),
    ("cond_2", "step_pos(x) = (x > 0) ? 1 : 0"),
    ("cond_3", "step_neg(x) = (x < 0) ? 1 : 0"),
    ("cond_4", "absval(x) = (x > 0) ? (x) : (0 - x)"),

    # ----- Calls (exercise Call + stub semantics) -----
    ("call_1", "use_relu(x) = relu(x)"),
    ("call_2", "shift_then_relu(x) = relu(x + 1)"),
    ("call_3", "use_avg(x, y) = avg2(x, y)"),
    ("call_4", "combine(x, y) = f1(x) + f2(y)"),

    # ----- Recursive defs (language supports, eval should mark unsupported) -----
    ("rec_1", "fact(n) = n * fact(n - 1)"),
    ("rec_2", "accum(t) = t + accum(t - 1)"),

    # ----- Mixed multi-var polynomials -----
    ("mix_1", "poly2(x, y) = x^2 + y^2 + 2*x*y"),
    ("mix_2", "triple(x, y, z) = x*y + y*z + z*x"),
]


def main():
    with OUT.open("w", encoding="utf-8") as f:
        for id_, eq in CASES:
            rec = {
                "id": id_,
                "lel_equation": eq,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
