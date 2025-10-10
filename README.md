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

    python -m experiments.summarize --dir_name EvoEdit --runs run_000

## Acknowledgment
Our code is based on  [``AlphaEdit``](https://github.com/jianghoucheng/AlphaEdit).

