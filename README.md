# LaTeX Equation Language (LEL): LaTeX → LLVM

This repo is my CS 6120 final project: a tiny compiler pipeline that:

1. **Normalizes** LaTeX equations using [MiTeX](https://github.com/mitex-rs/mitex),
2. **Parses** a small subset into a custom **LaTeX Equation Language (LEL)**,
3. **Compiles** LEL functions to **LLVM IR**, and
4. **Evaluates** correctness by linking with C test drivers and running them.

It supports a scalar subset of equations of the form:

```text
name(arg1, arg2, ...) = expr
name = expr
```

where `expr` includes arithmetic, powers, function calls, and simple ternary conditionals.

---

## Files in this repo

Core language & backend:

- `lel_core.py` – LEL tokenizer, parser, and AST.
- `lel_codegen_llvm.py` – LLVM IR codegen for a single LEL function.

Data + front-end:

- `parquet_to_equations_jsonl.py` – Convert HuggingFace `latex-formulas` Parquet shards to a JSONL of LaTeX equations.
- `download_arxiv_robotics_2025_tex.py` – Download LaTeX sources for 2025 `cs.RO` arXiv papers.
- `extract_equations.py` – Extract math environments from `.tex` into `equations.jsonl`.
- `mitex_batch_convert_equations.py` – Run MiTeX on each equation and attach a `typst_equation` field.
- `build_lel_functions.py` – Filter MiTeX-normalized equations down to the LEL subset (`*_lel_functions.jsonl`).

Evaluation & analysis:

- `make_manual_lel_functions.py` – Generate a tiny curated set of hand-written LEL functions.
- `eval_lel_functions.py` – End-to-end evaluation (parse → LLVM → clang → C driver → run tests).
- `analyze_failure_modes.py` – Sample and categorize why MiTeX-success equations fail the LEL parser.

---

## Setup

You’ll need:

- **Python** 3.10+
- **Rust** (for installing MiTeX)
- **LLVM / Clang** available as `clang` on your `PATH`
- Some Python packages: `pandas`, `pyarrow`

### 1. Install MiTeX CLI

```bash
brew install rust           # if you don’t already have Rust
cargo install --git https://github.com/mitex-rs/mitex mitex-cli

# make sure cargo binaries are on your PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### 2. Python dependencies

```bash
pip install pandas pyarrow
```

### 3. Optional: download preprocessed data

Download prebuilt JSONL files (if available) from this
[Google Drive link](https://drive.google.com/drive/folders/160Ebony_H_DMlIwIGVA-3mH3CokfMLMM?usp=drive_link)
into the repo root.

---

## 1. Tiny hand-written test set

This is the smallest way to see the pipeline work end-to-end.

Generate the manual LEL suite:

```bash
python make_manual_lel_functions.py
# writes manual_lel_functions.jsonl
```

Run the evaluator:

```bash
python eval_lel_functions.py --in manual_lel_functions.jsonl
```

You should see per-function status and a summary, e.g.:

```text
[OK] arith_1: f1
[OK] cond_1: relu
...
=== Summary ===
Total evaluated (up to max): 24
  ok: ...
  runtime_error: ...
  unsupported: ...
```

---

## 2. Using the HuggingFace `latex-formulas` dataset

Dataset page: <https://huggingface.co/datasets/OleehyO/latex-formulas>

### 2.1 Convert Parquet → JSONL (LaTeX only)

Download the cleaned shards locally (e.g., `train-00000-of-00006.parquet` … `train-00005-of-00006.parquet`), then:

```bash
python parquet_to_equations_jsonl.py   --in 'train-*.parquet'   --out equations_from_hf.jsonl   --column latex_formula
```

This produces `equations_from_hf.jsonl` with records like:

```json
{"id": 0, "equation": "\begin{align*} ... \end{align*}", ...}
```

### 2.2 Normalize with MiTeX

```bash
python mitex_batch_convert_equations.py   --in equations_from_hf.jsonl   --out hf_mitex_equations.jsonl
```

Now each record has a `typst_equation` field (MiTeX-normalized math).

### 2.3 Filter to LEL subset

```bash
python build_lel_functions.py   --in hf_mitex_equations.jsonl   --out hf_lel_functions.jsonl
```

Only equations that match the LEL grammar are kept.

### 2.4 End-to-end evaluation

```bash
python eval_lel_functions.py   --in hf_lel_functions.jsonl
```

This will:

- parse each `lel_equation`,
- generate LLVM IR,
- compile IR + a C driver with `clang`,
- run tests against the Python reference evaluator, and
- report `ok`, `runtime_error`, `compile_error`, or `unsupported`.

---

## 3. Using arXiv cs.RO (robotics) LaTeX sources

### 3.1 Download LaTeX sources

```bash
python download_arxiv_robotics_2025_tex.py   --out arxiv_tex_cs_RO_2025
```

### 3.2 Extract equations from `.tex`

```bash
python extract_equations.py   --root arxiv_tex_cs_RO_2025   --out equations.jsonl
```

### 3.3 Normalize with MiTeX

```bash
python mitex_batch_convert_equations.py   --in equations.jsonl   --out mitex_equations.jsonl
```

### 3.4 Filter to LEL subset

```bash
python build_lel_functions.py   --in mitex_equations.jsonl   --out lel_functions.jsonl
```

### 3.5 Evaluate LEL → LLVM on arXiv equations

```bash
python eval_lel_functions.py   --in lel_functions.jsonl
```

---

## 4. Failure-mode analysis

To understand why many real equations don’t make it into the LEL subset (or fail evaluation), you can sample and categorize failures:

```bash
python analyze_failure_modes.py
```

(See comments in that script for which input JSONL it expects and how it samples.)

Typical categories include:

- no function header (`name(args) = expr`),
- unsupported constructs (integrals, matrices, MiTeX macros),
- identifier / normalization issues (e.g., weird subscripts, dotted names).

---

## Notes / limitations

- LEL currently only supports scalar expressions (no vectors/matrices, integrals, sums, products, etc.).
- Function calls are treated as **stubs** in evaluation: they return their first argument.
- Self-recursive definitions are detected and marked as `unsupported` in `eval_lel_functions.py` to avoid non-terminating runs.
- The most difficult part is the **front end**: dealing with real-world LaTeX + MiTeX normalization quirks, not the LLVM backend itself.
