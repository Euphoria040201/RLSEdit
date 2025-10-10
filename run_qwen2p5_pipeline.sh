#!/usr/bin/env bash
set -euo pipefail
NUM_EDITS=1
ROOT="/work/xinyu/Project"
LOG_DIR="$ROOT/logs"
dataset="mcf"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS_FNAME="Llama3-8B.json"
ALG_NAME="EvoEdit"
MODEL_NAME="Llama3-8B"
CKPT_SUBDIR="checkpoints_${MODEL_NAME}_evo_${dataset}_${ALG_NAME}_ne${NUM_EDITS}"
GPU_ID=1
RUN_DIR="results/${ALG_NAME}/run_000"

install -d "$LOG_DIR"
cd "$ROOT"

STAMP="$(date +%m%d_%H%M%S)"
LOG="${LOG_DIR}/${MODEL_NAME}_evo_ne${NUM_EDITS}_${ALG_NAME}_${dataset}_${STAMP}.log"
PIDF="${LOG}.pid"

echo "==[Stage 1 / evaluate]=================================================="
echo "Logs: $LOG"
echo "PID file: $PIDF"
echo "Started at: $(date)"

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python3 -u -m experiments.evaluate \
  --alg_name "$ALG_NAME" \
  --model_name "$MODEL_ID" \
  --hparams_fname "$HPARAMS_FNAME" \
  --ds_name "$dataset" \
  --dataset_size_limit 500 \
  --num_edits "$NUM_EDITS" \
  --downstream_eval_steps 0 \
  --save_every 2000 \
  --checkpoint_subdir "$CKPT_SUBDIR" \
  > "$LOG" 2>&1 & echo $! | tee "$PIDF"

PID1="$(cat "$PIDF")"
echo "evaluate PID: $PID1"
echo "Waiting for evaluate (PID=$PID1) to finish..."
wait "$PID1"
echo "evaluate finished at: $(date)"

echo "==[Stage 2 / eval_run_checkpoints]======================================="
if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: RUN_DIR does not exist: $RUN_DIR" >&2
  exit 1
fi

STAMP2="$(date +%m%d_%H%M)"
EVAL_LOG="${RUN_DIR}/eval_batch_${STAMP2}.log"
EVAL_PIDF="${RUN_DIR}/eval_batch.pid"

echo "Run dir: $RUN_DIR"
echo "Eval logs: $EVAL_LOG"
echo "Eval PID file: $EVAL_PIDF"
echo "Started at: $(date)"

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python3 -u -m experiments.eval_run_checkpoints \
  --run_dir "$RUN_DIR" \
  --ds_name "$dataset" \
  --dataset_size_limit 2000 \
  --generation_test_interval 5 \
  --trust_remote_code \
  --skip_existing \
  > "$EVAL_LOG" 2>&1 & echo $! | tee "$EVAL_PIDF"

PID2="$(cat "$EVAL_PIDF")"
echo "eval_run_checkpoints PID: $PID2"
echo "Waiting for eval_run_checkpoints (PID=$PID2) to finish..."
wait "$PID2"
echo "eval_run_checkpoints finished at: $(date)"

echo "==[Stage 3 / post-processing]==========================================="
echo "Plot forgetting curves..."
python3 -u experiments/plot_forgetting.py \
  --run_dir "$RUN_DIR" \
  --ds_name mcf

echo "Summarize run..."
python -m experiments.summarize --dir_name ROME --runs "$(basename "$RUN_DIR")"

echo "======================================================================="
echo "ALL DONE at: $(date)"
