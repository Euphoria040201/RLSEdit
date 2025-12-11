#!/usr/bin/env python
import argparse
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ANSWER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass
class EvalResult:
    name: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def normalize_number(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = cleaned.lstrip("$")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.rstrip(".")
    try:
        decimal_value = Decimal(cleaned)
        as_float = format(decimal_value, "f")
        trimmed = as_float.rstrip("0").rstrip(".")
        return trimmed or "0"
    except InvalidOperation:
        return cleaned


def extract_final_number(text: str) -> Optional[str]:
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def load_gsm8k(split: str, data_path: Optional[Path]) -> Iterable[dict]:
    if data_path:
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    yield {"question": record["question"], "answer": record["answer"]}
                except json.JSONDecodeError:
                    # Fallback for tab-separated question<TAB>answer
                    if "\t" in line:
                        q, a = line.split("\t", 1)
                        yield {"question": q, "answer": a}
                    else:
                        raise
    else:
        dataset = load_dataset("gsm8k", "main", split=split)
        for record in dataset:
            yield {"question": record["question"], "answer": record["answer"]}


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _load_tokenizer(model_path: str, base_model: Optional[str]) -> AutoTokenizer:
    try:
        print(f"[INFO] Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        if base_model is None:
            print(f"[ERROR] Failed to load tokenizer from {model_path}: {exc}")
            raise
        print(f"[WARN] Falling back to tokenizer from base model {base_model}: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    model_path: str, base_model: Optional[str], device: str, torch_dtype: torch.dtype
) -> AutoModelForCausalLM:
    config = None
    if base_model:
        try:
            print(f"[INFO] Loading config from base model {base_model}")
            config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        except Exception as exc:
            print(f"[WARN] Could not load base config {base_model}: {exc}")

    try:
        print(f"[INFO] Loading model weights from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model from {model_path}: {exc}")
        raise

    model.to(device)
    model.eval()
    return model

def evaluate_model(
    model_path: str,
    dataset: Iterable[dict],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    limit: Optional[int],
    base_model: Optional[str],
) -> EvalResult:
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = _load_tokenizer(model_path, base_model)
    model = _load_model(model_path, base_model, device, torch_dtype)

    correct = 0
    total = 0
    for idx, sample in enumerate(dataset):
        if limit is not None and idx >= limit:
            break

        if idx % 50 == 0:
            print(f"[INFO] Example {idx} | correct so far: {correct}/{total}")

        prompt = build_prompt(sample["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        do_sample = temperature > 0
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        prediction_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        gold = extract_final_number(sample["answer"])
        pred = extract_final_number(prediction_text)
        correct += int(gold is not None and pred == gold)
        total += 1

    return EvalResult(name=model_path, correct=correct, total=total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two models on GSM8K and report exact-match accuracy."
    )
    parser.add_argument("--pre-model", required=True, help="Path or Hugging Face id for the pre-edit model.")
    parser.add_argument("--post-model", required=True, help="Path or Hugging Face id for the post-edit model.")
    parser.add_argument(
        "--base-model",
        help="Optional base model to provide config/tokenizer if the edited checkpoints miss config.json.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to use (default: test).")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional local gsm8k JSONL file with fields 'question' and 'answer'.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of examples to evaluate (useful for quick tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = list(load_gsm8k(args.split, args.data_path))
    print(f"[INFO] Loaded {len(dataset)} samples from {'local file' if args.data_path else 'huggingface'} split {args.split}")

    pre_result = evaluate_model(
        model_path=args.pre_model,
        dataset=dataset,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
        base_model=args.base_model,
    )

    post_result = evaluate_model(
        model_path=args.post_model,
        dataset=dataset,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
        base_model=args.base_model,
    )

    summary = {
        "pre_model": {
            "path": args.pre_model,
            "accuracy": pre_result.accuracy,
            "correct": pre_result.correct,
            "total": pre_result.total,
        },
        "post_model": {
            "path": args.post_model,
            "accuracy": post_result.accuracy,
            "correct": post_result.correct,
            "total": post_result.total,
        },
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
