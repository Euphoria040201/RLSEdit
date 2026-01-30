import os
import sys
import json
import traceback
from pathlib import Path

# Ensure project root (one level up) is on sys.path so `import glue_eval.*` finds the package
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from glue_eval.glue_eval import GLUEEval


def run_eval_on_model(
    model_dir: str,
    out_dir: str,
    gen_len: int = 3,
    few_shots: int = 0,
    number_of_tests=None,
    tasks=None,
    perplexity_flag: bool = False,
):
    model_dir = str(model_dir)
    print(f"Loading model from {model_dir}")
    # Try loading tokenizer robustly; retry with different flags if necessary
    tokenizer = None
    load_attempts = [dict(use_fast=False), dict(use_fast=True), dict()]
    for opts in load_attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, **opts)
            break
        except Exception:
            tokenizer = None

    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer for model at {model_dir}")

    # If tokenizer is not a proper tokenizer instance (e.g., boolean or missing methods), try alternatives
    if isinstance(tokenizer, bool) or not hasattr(tokenizer, 'encode') or not hasattr(tokenizer, 'decode'):
        # Attempt to load from tokenizer files directly
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        except Exception:
            pass

    if isinstance(tokenizer, bool) or not hasattr(tokenizer, 'encode') or not hasattr(tokenizer, 'decode'):
        # Final fallback: create minimal wrapper around tokenizer methods if available
        if hasattr(tokenizer, 'convert_tokens_to_ids') and hasattr(tokenizer, 'convert_ids_to_tokens'):
            class _TokenizerWrapper:
                def __init__(self, tok):
                    self.tok = tok
                def __call__(self, text, **kwargs):
                    ids = self.tok.convert_tokens_to_ids(self.tok.tokenize(text))
                    return {"input_ids": ids}
                def encode(self, *a, **k):
                    return self.tok.convert_tokens_to_ids(self.tok.tokenize(a[0]))
                def decode(self, ids, **k):
                    return "".join(self.tok.convert_ids_to_tokens(ids))
            tokenizer = _TokenizerWrapper(tokenizer)
        else:
            raise RuntimeError(f"Tokenizer for {model_dir} is invalid or unsupported: {tokenizer}")

    # try load in float16 if possible to reduce memory
    dtype = torch.float16
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    except Exception:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir)
        except Exception as e:
            print(f"Failed to load model {model_dir}: {e}")
            traceback.print_exc()
            return False

    # ensure context-length lookup works in existing evaluation code
    try:
        model.config._name_or_path = 'llama3'
    except Exception:
        pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    evaluator = GLUEEval(
        model,
        tokenizer,
        number_of_tests=number_of_tests,
        sst_number_of_few_shots=few_shots,
        mrpc_number_of_few_shots=few_shots,
        cola_number_of_few_shots=few_shots,
        rte_number_of_few_shots=few_shots,
        mmlu_number_of_few_shots=few_shots,
        sentiment_analysis_number_of_few_shots=few_shots,
        nli_number_of_few_shots=few_shots,
        dialogue_number_of_few_shots=few_shots,
    )

    task_flags = {
        "sst": False,
        "mmlu": False,
        "mrpc": False,
        "cola": False,
        "rte": False,
        "nli": False,
        "sentiment_analysis": False,
        "dialogue": False,
    }
    if tasks is None:
        tasks = ["sst", "mrpc", "cola", "rte", "mmlu", "sentiment_analysis", "nli"]
    for task in tasks:
        if task not in task_flags:
            raise ValueError(f"Unknown task: {task}")
        task_flags[task] = True

    print(f"Running GLUE evaluation for {model_dir} (gen_len={gen_len}, tasks={tasks})")
    glue_results = {'model_dir': model_dir}
    result_dict = evaluator.evaluate(
        glue_results,
        os.path.join(out_dir, f"{Path(model_dir).name}_glue_eval.json"),
        perplexity_flag=perplexity_flag,
        sst_flag=task_flags["sst"],
        mmlu_flag=task_flags["mmlu"],
        mrpc_flag=task_flags["mrpc"],
        cola_flag=task_flags["cola"],
        rte_flag=task_flags["rte"],
        nli_flag=task_flags["nli"],
        sentiment_analysis_flag=task_flags["sentiment_analysis"],
        dialogue_flag=task_flags["dialogue"],
        gen_len=gen_len,
    )

    os.makedirs(out_dir, exist_ok=True)
    base_name = Path(model_dir).name
    json_path = os.path.join(out_dir, f"{base_name}_glue_eval.json")

    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"Saved results to {json_path}")
    return True


def run_batch(
    root_models_dir: str,
    out_dir: str = None,
    gen_len: int = 3,
    few_shots: int = 0,
    number_of_tests=None,
    tasks=None,
    perplexity_flag: bool = False,
):
    root = Path(root_models_dir)
    if out_dir is None:
        out_dir = root / 'glue_eval_results'
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    entries = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(entries) == 0:
        print(f"No model directories found under {root_models_dir}")
        return

    for model_path in entries:
        try:
            run_eval_on_model(
                model_path,
                out_dir,
                gen_len=gen_len,
                few_shots=few_shots,
                number_of_tests=number_of_tests,
                tasks=tasks,
                perplexity_flag=perplexity_flag,
            )
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Batch run GLUE eval for models in a directory')
    parser.add_argument('models_root', help='Root directory containing model subdirectories')
    parser.add_argument('--out', help='Output directory to save results', default=None)
    parser.add_argument('--gen_len', type=int, default=3)
    parser.add_argument('--few_shots', type=int, default=0)
    parser.add_argument('--tests', type=int, default=None, help='Number of test examples to run (default: all)')
    parser.add_argument(
        '--tasks',
        default=None,
        help='Comma-separated tasks (sst,mrpc,cola,rte,mmlu,nli,sentiment_analysis,dialogue). Default: all except dialogue.',
    )
    parser.add_argument('--perplexity', action='store_true', help='Also compute perplexity on wikitext.')

    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(',')] if args.tasks else None
    run_batch(
        args.models_root,
        out_dir=args.out,
        gen_len=args.gen_len,
        few_shots=args.few_shots,
        number_of_tests=args.tests,
        tasks=tasks,
        perplexity_flag=args.perplexity,
    )
