#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG ======
NUM_EDITS=100
ROOT="/work/xinyu/Project"
LOG_DIR="$ROOT/logs"
dataset="mcf"
RLSEDIT_MOM2_UPDATE_WEIGHT=15000
# Optional: override lambda_reg (controls lam**2 term); leave empty to use hparams
RLSEDIT_LAMBDA_REG=""
# 用绝对路径（你之前的权重在 /work/xinyu/models/...）
MODEL_ID="/work/xinyu/models/Qwen2.5-7B-Instruct"
HPARAMS_FNAME="Qwen2.5-7B.json"
ALG_NAME="RLSEdit"
MODEL_NAME="Qwen2.5-7B"
DATASET_SIZE=15000
CKPT_SUBDIR="${MODEL_NAME}_${dataset}_${ALG_NAME}_ne${NUM_EDITS}_ds${DATASET_SIZE}_mu${RLSEDIT_MOM2_UPDATE_WEIGHT}"
# 原来是 {0,1,...} 会当普通字符串；改成 bash 数组
GPU_IDS=(6 7)
RUN_DIR="results/${ALG_NAME}/run_000"

# RLSEdit-specific: scale for mom2_update_weight (override via env)


# 如果你想让 Stage 2 只用一个 GPU 跑，指定这里：
GPU_FOR_STAGE2=0

# ====== PREP ======
install -d "$LOG_DIR"
cd "$ROOT"

if [[ "$ALG_NAME" == "RLSEdit" ]]; then
  export RLSEDIT_MOM2_UPDATE_WEIGHT
  echo "RLSEdit mom2_update_weight override: ${RLSEDIT_MOM2_UPDATE_WEIGHT}"
  if [[ -n "${RLSEDIT_LAMBDA_REG}" ]]; then
    export RLSEDIT_LAMBDA_REG
    echo "RLSEdit lambda_reg override: ${RLSEDIT_LAMBDA_REG}"
  fi
fi

echo "==[Stage 1 / evaluate - per GPU parallel]==============================="
echo "Logs dir: $LOG_DIR"
echo "Started at: $(date)"

PIDS=()   # 收集所有后台进程 PID

for GPU_ID in "${GPU_IDS[@]}"; do
  STAMP="$(date +%m%d_%H%M%S)"
  LOG="${LOG_DIR}/${CKPT_SUBDIR}_${STAMP}_GPU${GPU_ID}.log"
  PIDF="${LOG}.pid"

  echo "Launching evaluate on GPU ${GPU_ID}"
  echo "  Log: $LOG"
  echo "  PID file: $PIDF"

  # 每块卡一个进程并行跑
  CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python3 -u -m experiments.evaluate \
    --alg_name "$ALG_NAME" \
    --model_name "$MODEL_ID" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name "$dataset" \
    --dataset_size_limit "$DATASET_SIZE" \
    --num_edits "$NUM_EDITS" \
    --downstream_eval_steps 0 \
    --save_every 0 \
    --checkpoint_subdir "$CKPT_SUBDIR" \
    > "$LOG" 2>&1 & echo $! | tee "$PIDF"

  PIDS+=("$(cat "$PIDF")")
done

# 等全部 GPU 的 evaluate 结束
echo "Waiting for all evaluate jobs to finish..."
for pid in "${PIDS[@]}"; do
  echo "  waiting PID=$pid"
  if ! wait "$pid"; then
    echo "[FATAL] evaluate failed (PID=$pid). Check its log under $LOG_DIR" >&2
    exit 3
  fi
done
echo "All evaluate jobs finished at: $(date)"

echo "==[Stage 2 / eval_run_checkpoints]======================================="
if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: RUN_DIR does not exist: $RUN_DIR" >&2
  exit 4
fi

STAMP2="$(date +%m%d_%H%M)"
EVAL_LOG="${RUN_DIR}/eval_batch_${STAMP2}.log"
EVAL_PIDF="${RUN_DIR}/eval_batch.pid"

echo "Run dir: $RUN_DIR"
echo "Eval logs: $EVAL_LOG"
echo "Eval PID file: $EVAL_PIDF"
echo "Started at: $(date)"

CUDA_VISIBLE_DEVICES=${GPU_FOR_STAGE2} nohup python3 -u -m experiments.eval_run_checkpoints \
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
wait "$PID2" || { echo "[FATAL] eval_run_checkpoints 失败，检查日志：$EVAL_LOG" >&2; exit 5; }
echo "eval_run_checkpoints finished at: $(date)"

echo "==[Stage 3 / post-processing]==========================================="
echo "Plot forgetting curves..."
python3 -u experiments/plot_forgetting.py \
  --run_dir "$RUN_DIR" \
  --ds_name mcf

echo "Summarize run..."
python -m experiments.summarize --dir_name "$ALG_NAME" --runs "$(basename "$RUN_DIR")"

echo "======================================================================="
echo "ALL DONE at: $(date)"
