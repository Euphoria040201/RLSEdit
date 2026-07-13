# RLSEdit

Code for running the **RLSEdit** main editing algorithm.

## Requirements
**At least one A40 48G GPU.**

- torch==2.6.0
- einops==0.8.1
- higher==0.2.1
- hydra-core==1.3.2
- transformers==4.51.3
- datasets==2.21.0
- matplotlib==3.10.3
- spacy==3.4.1
- scipy==1.15.2
- scikit-learn==1.6.1
- nltk==3.9.1

Install dependencies with:

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

## How to run RLSEdit

The main entry point is [`experiments/evaluate.py`](experiments/evaluate.py), which loads hyperparameters from `hparams/RLSEdit/` and applies [`RLSEdit/RLSEdit_main.py`](RLSEdit/RLSEdit_main.py).

### Direct command

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u -m experiments.evaluate \
  --alg_name RLSEdit \
  --model_name /path/to/Meta-Llama-3-8B-Instruct \
  --hparams_fname Llama3-8B.json \
  --ds_name mcf \
  --dataset_size_limit 100 \
  --num_edits 100 \
  --downstream_eval_steps 200 \
  --reasoning_eval_steps 200 \
  --reasoning_eval_num_tests 100 \
  --save_every 0 \
  --checkpoint_subdir Llama3-8B_mcf_RLSEdit_ne100_ds100
```

Key arguments:

| Flag | Meaning |
|------|---------|
| `--alg_name RLSEdit` | Select the RLSEdit algorithm |
| `--model_name` | Hugging Face model id or local path |
| `--hparams_fname` | JSON file under `hparams/RLSEdit/` (e.g. `Llama3-8B.json`, `Qwen2.5-7B.json`) |
| `--ds_name` | Dataset: `mcf`, `cf`, `zsre`, or `mquake` |
| `--num_edits` | Edits applied per batch |
| `--dataset_size_limit` | Truncate dataset to first `n` records |

Optional environment overrides for RLSEdit (see `RLSEdit_main.py`):

```bash
export RLSEDIT_MOM2_UPDATE_WEIGHT=12000   # overrides mom2_update_weight
export RLSEDIT_LAMBDA_REG=0              # overrides lambda_reg
```

Available RLSEdit hyperparameter files:

```
hparams/RLSEdit/
  EleutherAI_gpt-j-6B.json
  gpt2-xl.json
  Llama2-7B.json
  Llama3.2-3B.json
  Llama3-8B.json
  Qwen2.5-7B.json
```

### Full pipeline (edit → eval checkpoints → plot/summarize)

Edit the config at the top of [`run_qwen2p5_pipeline.sh`](run_qwen2p5_pipeline.sh) (`MODEL_ID`, `HPARAMS_FNAME`, `ALG_NAME=RLSEdit`, etc.), then:

```bash
LOG_DIR="$PWD/logs"
install -d "$LOG_DIR"
STAMP="$(date +%m%d_%H%M%S)"
OUTLOG="logs/pipeline_${STAMP}.log"
PIDF="logs/pipeline.pid"
nohup ./run_qwen2p5_pipeline.sh > "$OUTLOG" 2>&1 & echo $! | tee "$PIDF"
```

This runs editing, checkpoint evaluation, forgetting plots, and summarization.

## Results

Results from each run are stored at `results/<method_name>/run_<run_id>`:

```bash
results/
|__ RLSEdit/
    |__ run_<run_id>/
        |__ params.json
        |__ case_0.json
        |__ case_1.json
        |__ ...
```

### Summarize results

```bash
python -m experiments.summarize --dir_name RLSEdit --runs 000
```

## Acknowledgment
Our code is based on [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit).
