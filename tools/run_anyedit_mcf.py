#!/usr/bin/env python
"""
Sequential AnyEdit (UNKE) on MultiCounterFact, single-example batches.

Example:
  CUDA_VISIBLE_DEVICES=0 python tools/run_anyedit_mcf.py \
    --data /work/xinyu/Project/data/multi_counterfact.json \
    --model /work/xinyu/models/Meta-Llama-3-8B-Instruct \
    --hparams /work/xinyu/AnyEdit/hparams/unke/Llama3-8B-Instruct.json \
    --alpaca /work/xinyu/AnyEdit/data/alpaca_data.json \
    --limit 5 \
    --out /work/xinyu/AnyEdit/output/anyedit_mcf_seq.json

Note: AnyEdit's UNKE implementation uses hardcoded CUDA in some calls;
set CUDA_VISIBLE_DEVICES to pick the GPU before running.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add AnyEdit repo to path
ANYEDIT_ROOT = Path("/work/xinyu/AnyEdit")
sys.path.append(str(ANYEDIT_ROOT))

from unke import unkeHyperParams, apply_unke_to_model  # noqa: E402
from dsets.unke import get_llama_without_answer, get_list_llama_without_answer  # noqa: E402


def adapt_multi_counterfact(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if limit:
        raw = raw[:limit]
    adapted = []
    for item in raw:
        req = item.get("requested_rewrite", {})
        subj = req.get("subject", "")
        prompt = req.get("prompt", "{}")
        question_plain = prompt.replace("{}", subj)
        para_candidates = item.get("paraphrase_prompts") or item.get("generation_prompts") or [question_plain]
        para_plain = para_candidates[0].replace("{}", subj)
        sub_qs_plain = item.get("attribute_prompts") or item.get("generation_prompts") or []
        sub_qs_plain = [q.replace("{}", subj) for q in sub_qs_plain]
        if not sub_qs_plain:
            sub_qs_plain = [question_plain]
        answer = req.get("target_new", {}).get("str", "")

        adapted.append(
            {
                "question": get_llama_without_answer(question_plain),
                "para_question": get_llama_without_answer(para_plain),
                "sub_question": get_list_llama_without_answer(sub_qs_plain, False),
                "answer": answer + "<|eot_id|>",
            }
        )
    return adapted


def load_ex_data(alpaca_path: Path) -> list[str]:
    with alpaca_path.open("r", encoding="utf-8") as f:
        ex_datas = json.load(f)
    return [get_llama_without_answer(i["instruction"] + i["input"]) + i["output"] for i in ex_datas]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AnyEdit UNKE sequentially on MultiCounterFact.")
    ap.add_argument("--data", required=True, type=Path, help="Path to multi_counterfact.json")
    ap.add_argument("--alpaca", required=True, type=Path, help="Path to alpaca_data.json")
    ap.add_argument("--model", required=True, type=str, help="HF model path/id for base model")
    ap.add_argument("--hparams", required=True, type=Path, help="Path to AnyEdit UNKE hparams JSON")
    ap.add_argument("--limit", type=int, help="Optional limit on number of edits")
    ap.add_argument("--out", required=True, type=Path, help="Where to save edited predictions JSON")
    args = ap.parse_args()

    # Load hparams
    hparams = unkeHyperParams.from_json(args.hparams)

    # Load data
    edit_data = adapt_multi_counterfact(args.data, args.limit)
    ex_datas = load_ex_data(args.alpaca)

    # Load model/tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)

    edited_data = []
    for idx, record in enumerate(edit_data):
        batch = [record]
        random_elements = random.sample(ex_datas, hparams.ex_data_num)
        apply_unke_to_model(model, tok, hparams, batch, random_elements)

        # Generate predictions
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        question = tokenizer([record["question"], record["para_question"]], return_tensors="pt", padding=True)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=question["input_ids"].to(model.device),
                attention_mask=question["attention_mask"].to(model.device),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=512,
            )
        gen_trim = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(question["input_ids"], generated_ids)]
        output = tokenizer.batch_decode(gen_trim, skip_special_tokens=True)
        record["original_prediction"] = output[0]
        record["para_prediction"] = output[1]

        edited_data.append(record)
        if idx < 5:
            print(f"[Info] idx={idx} question: {record['question']}")
            print(output[0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(edited_data, f, ensure_ascii=False, indent=2)
    print(f"Saved edited data to {args.out}")


if __name__ == "__main__":
    main()
