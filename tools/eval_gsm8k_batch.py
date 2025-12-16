#!/usr/bin/env python
import argparse
import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional, List, Tuple, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ANSWER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"Final answer:\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE
)


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
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def parse_gsm8k_answer(answer_text: str) -> Tuple[str, Optional[str]]:
    if "####" in answer_text:
        rationale, final = answer_text.split("####", 1)
        final_num = extract_final_number(final)
        return rationale.strip(), final_num
    return answer_text.strip(), extract_final_number(answer_text)


def _load_dataset_no_verify(name: str, config: str, split: str):
    """
    关掉 split 校验，绕过 NonMatchingSplitsSizesError。
    兼容不同 datasets 版本。
    """
    try:
        return load_dataset(name, config, split=split, verification_mode="no_checks")
    except TypeError:
        return load_dataset(name, config, split=split, ignore_verifications=True)


def load_gsm8k(split: str, data_path: Optional[Path]) -> List[dict]:
    if data_path:
        out = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    out.append({"question": record["question"], "answer": record["answer"]})
                except json.JSONDecodeError:
                    if "\t" in line:
                        q, a = line.split("\t", 1)
                        out.append({"question": q, "answer": a})
                    else:
                        raise
        return out
    else:
        ds = _load_dataset_no_verify("gsm8k", "main", split=split)
        return [{"question": r["question"], "answer": r["answer"]} for r in ds]


def format_assistant_answer(rationale: str, final_num: Optional[str], use_cot: bool) -> str:
    final_line = f"Final answer: {final_num}" if final_num is not None else "Final answer: "
    if use_cot:
        rationale = rationale.strip()
        return f"{rationale}\n{final_line}" if rationale else final_line
    return final_line


def build_fewshot_messages_from_indices(k: int, seed: int, use_cot: bool) -> List[dict]:
    if k <= 0:
        return []
    train_ds = _load_dataset_no_verify("gsm8k", "main", split="train")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(train_ds)), k)

    msgs: List[dict] = []
    for i in idxs:
        q = train_ds[i]["question"]
        rat, final_num = parse_gsm8k_answer(train_ds[i]["answer"])
        a = format_assistant_answer(rat, final_num, use_cot=use_cot)
        msgs.append({"role": "user", "content": f"Question: {q}"})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def build_query_messages(question: str, use_cot: bool) -> List[dict]:
    if use_cot:
        user = (
            f"Question: {question}\n"
            "Please think step by step, and end your response with exactly one line:\n"
            "Final answer: <number>"
        )
    else:
        user = (
            f"Question: {question}\n"
            "End your response with exactly one line:\n"
            "Final answer: <number>"
        )
    return [{"role": "user", "content": user}]


def encode_chat_or_plain(tokenizer: AutoTokenizer, messages: List[dict], device: str):
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(prompt, return_tensors="pt").to(device)
    else:
        parts = []
        for m in messages:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        parts.append("ASSISTANT:")
        prompt = "\n".join(parts)
        return tokenizer(prompt, return_tensors="pt").to(device)


def _load_tokenizer(base_for_tokenizer: str) -> AutoTokenizer:
    print(f"[INFO] Loading tokenizer from {base_for_tokenizer}")
    tok = AutoTokenizer.from_pretrained(base_for_tokenizer, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_model(model_path: str, base_model: Optional[str], device: str, torch_dtype: torch.dtype) -> AutoModelForCausalLM:
    config = None
    if base_model:
        try:
            print(f"[INFO] Loading config from base model {base_model}")
            config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        except Exception as exc:
            print(f"[WARN] Could not load base config {base_model}: {exc}")

    print(f"[INFO] Loading model weights from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return model


def eval_loaded_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[dict],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    limit: Optional[int],
    fewshot_msgs: List[dict],
    cot: bool,
) -> EvalResult:
    correct = 0
    total = 0
    do_sample = temperature > 0

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that solves math word problems. "
            "Always finish with exactly one line: Final answer: <number>."
        ),
    }

    for idx, sample in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        if idx % 50 == 0:
            print(f"[INFO]   Example {idx} | correct so far: {correct}/{total}")

        messages = [system_msg] + fewshot_msgs + build_query_messages(sample["question"], use_cot=cot)
        inputs = encode_chat_or_plain(tokenizer, messages, device=device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen = outputs[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(gen, skip_special_tokens=True)

        gold = extract_final_number(sample["answer"])
        pred = extract_final_number(pred_text)
        correct += int(gold is not None and pred == gold)
        total += 1

    return EvalResult(name="model", correct=correct, total=total)


def list_model_dirs(models_dir: Path, pattern: str) -> List[Path]:
    cands = [p for p in models_dir.glob(pattern) if p.is_dir()]
    def key_fn(p: Path):
        # edits_010000 -> 10000
        m = re.search(r"(\d+)$", p.name)
        return int(m.group(1)) if m else 10**18
    cands.sort(key=key_fn)
    return cands


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch evaluate multiple edited checkpoints under a directory on GSM8K.")
    p.add_argument("--models-dir", type=Path, required=True, help="Directory containing subfolders like edits_002000, edits_010000, ...")
    p.add_argument("--pattern", default="edits_*", help="Glob pattern under --models-dir (default: edits_*)")

    p.add_argument("--pre-model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Pre-edit model path or HF id (for baseline eval). Default: Llama3-8B-Instruct")
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Base model for config if edited ckpts miss config.json. Default: Llama3-8B-Instruct")

    p.add_argument("--split", default="test")
    p.add_argument("--data-path", type=Path, help="Optional local gsm8k JSONL file with fields question/answer.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--limit", type=int, help="Optional cap on #examples (quick test).")

    p.add_argument("--shots", type=int, default=8)
    p.add_argument("--cot", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--output-json", type=Path, default=Path("gsm8k_batch_results.json"),
                   help="Where to write the merged JSON results (default: gsm8k_batch_results.json)")
    p.add_argument("--write-per-model", action="store_true",
                   help="Also write gsm8k_eval.json inside each edits_xxxxxx folder.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_gsm8k(args.split, args.data_path)
    src = "local file" if args.data_path else "huggingface"
    print(f"[INFO] Loaded {len(dataset)} samples from {src} split={args.split}")

    model_dirs = list_model_dirs(args.models_dir, args.pattern)
    if not model_dirs:
        raise SystemExit(f"[ERROR] No model dirs matched {args.models_dir}/{args.pattern}")

    print(f"[INFO] Found {len(model_dirs)} model dirs under {args.models_dir} matching {args.pattern}")
    for p in model_dirs[:10]:
        print(f"  - {p}")
    if len(model_dirs) > 10:
        print(f"  ... (+{len(model_dirs)-10} more)")

    # Tokenizer: use base model tokenizer for everything (faster & consistent)
    tokenizer = _load_tokenizer(args.base_model or args.pre_model)

    # Few-shot messages precomputed once (text only)
    fewshot_msgs = build_fewshot_messages_from_indices(args.shots, args.seed, use_cot=bool(args.cot))
    print(f"[INFO] Few-shot: shots={args.shots}, cot={bool(args.cot)}, seed={args.seed}")

    device = args.device
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    results: Dict[str, Any] = {
        "setup": {
            "models_dir": str(args.models_dir),
            "pattern": args.pattern,
            "split": args.split,
            "data_path": str(args.data_path) if args.data_path else None,
            "device": device,
            "shots": args.shots,
            "cot": bool(args.cot),
            "seed": args.seed,
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "limit": args.limit,
        },
        "baseline": None,
        "models": [],
    }

    # Baseline (pre-model)
    if args.pre_model:
        print(f"[INFO] Evaluating baseline pre-model: {args.pre_model}")
        pre_model = _load_model(args.pre_model, args.base_model, device, torch_dtype)
        pre_res = eval_loaded_model(
            model=pre_model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
            fewshot_msgs=fewshot_msgs,
            cot=bool(args.cot),
        )
        results["baseline"] = {
            "path": args.pre_model,
            "accuracy": pre_res.accuracy,
            "correct": pre_res.correct,
            "total": pre_res.total,
        }
        del pre_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Each edited ckpt
    for mi, mdir in enumerate(model_dirs):
        print(f"\n[INFO] ({mi+1}/{len(model_dirs)}) Evaluating: {mdir}")
        model = _load_model(str(mdir), args.base_model, device, torch_dtype)
        res = eval_loaded_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
            fewshot_msgs=fewshot_msgs,
            cot=bool(args.cot),
        )
        record = {
            "path": str(mdir),
            "name": mdir.name,
            "accuracy": res.accuracy,
            "correct": res.correct,
            "total": res.total,
        }
        results["models"].append(record)

        if args.write_per_model:
            outp = mdir / "gsm8k_eval.json"
            outp.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"[INFO] Wrote per-model result: {outp}")

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[INFO] Wrote merged results to: {args.output_json}")
    print(json.dumps({
        "baseline_acc": results["baseline"]["accuracy"] if results["baseline"] else None,
        "num_models": len(results["models"]),
        "best_acc": max([m["accuracy"] for m in results["models"]], default=None),
    }, indent=2))


if __name__ == "__main__":
    main()
