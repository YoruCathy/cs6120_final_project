# cs6120_final_project
## Setup
Install
```
brew install rust
```
```
cargo install --git https://github.com/mitex-rs/mitex mitex-cli
```
```
export PATH="$HOME/.cargo/bin:$PATH"
```
Download data from this [link](https://drive.google.com/drive/folders/160Ebony_H_DMlIwIGVA-3mH3CokfMLMM?usp=drive_link).
## Use the tiny hand-written test set
```
python ./make_manual_lel_functions.py 
```
```
python eval_lel_functions.py --in manual_lel_functions.jsonl
```

## Use latex-formulas dataset
https://huggingface.co/datasets/OleehyO/latex-formulas
```
python parquet_to_equations_jsonl.py \
  --in 'train-*.parquet' \
  --out equations_from_hf.jsonl \
  --column latex_formula
```
```
python mitex_batch_convert_equations.py \
  --in equations_from_hf.jsonl  \
  --out hf_mitex_equations.jsonl
```
```
python build_lel_functions.py \          
  --in hf_mitex_equations.jsonl \
  --out hf_lel_functions.jsonl  
```
```
python eval_lel_functions.py \          
  --in hf_lel_functions.jsonl    
```
## 