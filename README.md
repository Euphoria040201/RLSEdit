# AlphaEdit
- Code for [``EvoEdit``]

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

## Quick Start
source /work/xinyu/envs/ml261/bin/activate
# 提示符出现 (ml261) 即成功
conda deactivate
conda activate /work/xinyu/envs/ml261/
cd /work/xinyu/Project
LOG_DIR="$PWD/logs"
install -d "$LOG_DIR"
STAMP="$(date +%m%d_%H%M%S)"
OUTLOG="logs/pipeline_${STAMP}.log"
PIDF="logs/pipeline.pid"
nohup ./run_qwen2p5_pipeline.sh > "$OUTLOG" 2>&1 & echo $! | tee "$PIDF" 

This command runs a bash script, which include editing the model, evaluate the model, and plot/summarize the results. 

Results from each run are stored at `results/<method_name>/run_<run_id>` in a specific format:
```bash
results/
|__ AlphaEdit/
    |__ run_<run_id>/
        |__ params.json
        |__ case_0.json
        |__ case_1.json
        |__ ...
        |__ case_2000.json
```

#### 2. Summarize the results  
To summarize the results, you can use [`experiments/summarize.py`](experiments/summarize.py):

    python -m experiments.summarize --dir_name ROME --runs 002

## Acknowledgment
Our code is based on  [``AlphaEdit``](https://github.com/jianghoucheng/AlphaEdit).

nohup python compute.py --gpu 7 --mem-frac 0.85 --compute &


cd /work/xinyu/Project/tools
mkdir -p logs

HF_DATASETS_CACHE=/work/xinyu/hf_cache/datasets_gsm8k_clean \
CUDA_VISIBLE_DEVICES=5 nohup python eval_gsm8k_batch.py \
  --models-dir /work/xinyu/Project/results/AlphaEdit/run_074/REASONING_Llama3-8B_mcf_AlphaEdit_ne100_ds10000_mu15000 \
  --pattern "edits_*" \
  --pre-model meta-llama/Meta-Llama-3-8B-Instruct \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --device cuda \
  --split test \
  --shots 8 \
  --cot \
  --temperature 0.0 \
  --max-new-tokens 256 \
  --output-json logs/gsm8k_run074_AlphaEdit_all_edits_8shot_cot.json \
  --write-per-model \
  > logs/gsm8k_run074_AlphaEdi_all_edits_8shot_cot.out 2>&1 &
