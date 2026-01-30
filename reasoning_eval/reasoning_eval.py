import json
import os
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset

ANSWER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"Final answer:\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


@dataclass
class EvalResult:
    name: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _normalize_number(raw: str) -> str:
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


def _extract_final_number(text: str) -> Optional[str]:
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return _normalize_number(m.group(1))
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def _extract_boxed(text: str) -> Optional[str]:
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _extract_math_answer(text: str) -> Optional[str]:
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    return _extract_final_number(text)


def _build_gsm8k_query_messages(question: str, use_cot: bool) -> List[dict]:
    if use_cot:
        user = (
            f"Question: {question}\n"
            "Please think step by step, and end your response with exactly one line:\n"
            "Final answer: <answer>"
        )
    else:
        user = (
            f"Question: {question}\n"
            "End your response with exactly one line:\n"
            "Final answer: <answer>"
        )
    return [{"role": "user", "content": user}]


def _build_math_query_messages(problem: str, use_cot: bool) -> List[dict]:
    if use_cot:
        user = (
            f"Problem: {problem}\n"
            "Please think step by step, and end your response with exactly one line:\n"
            "Final answer: <answer>"
        )
    else:
        user = (
            f"Problem: {problem}\n"
            "End your response with exactly one line:\n"
            "Final answer: <answer>"
        )
    return [{"role": "user", "content": user}]


def _encode_chat_or_plain(tokenizer, messages: List[dict], device: str):
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(prompt, return_tensors="pt").to(device)
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("ASSISTANT:")
    prompt = "\n".join(parts)
    return tokenizer(prompt, return_tensors="pt").to(device)


def _pick_indices(total: int, limit: Optional[int], seed: int) -> Sequence[int]:
    if limit is None or limit >= total:
        return range(total)
    rng = random.Random(seed)
    return rng.sample(range(total), limit)


def _parse_gsm8k_answer(answer_text: str) -> Tuple[str, Optional[str]]:
    if "####" in answer_text:
        rationale, final = answer_text.split("####", 1)
        final_num = _extract_final_number(final)
        return rationale.strip(), final_num
    return answer_text.strip(), _extract_final_number(answer_text)


def _format_assistant_answer(rationale: str, final_ans: Optional[str], use_cot: bool) -> str:
    final_line = f"Final answer: {final_ans}" if final_ans is not None else "Final answer: "
    if use_cot:
        rationale = rationale.strip()
        return f"{rationale}\n{final_line}" if rationale else final_line
    return final_line


def _build_gsm8k_fewshot_messages(k: int, seed: int, use_cot: bool) -> List[dict]:
    if k <= 0:
        return []
    train_ds = _load_dataset_no_verify("gsm8k", "main", split="train")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(train_ds)), min(k, len(train_ds)))

    msgs: List[dict] = []
    for i in idxs:
        q = train_ds[i]["question"]
        rat, final_num = _parse_gsm8k_answer(train_ds[i]["answer"])
        a = _format_assistant_answer(rat, final_num, use_cot=use_cot)
        msgs.append({"role": "user", "content": f"Question: {q}"})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def _build_math_fewshot_messages(
    k: int,
    seed: int,
    use_cot: bool,
    dataset_name: str,
) -> List[dict]:
    if k <= 0:
        return []
    try:
        train_ds = _load_dataset_no_verify(dataset_name, None, split="train")
    except Exception:
        train_ds = _load_dataset_no_verify(dataset_name, None, split="test")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(train_ds)), min(k, len(train_ds)))

    msgs: List[dict] = []
    for i in idxs:
        problem = train_ds[i]["problem"]
        solution = train_ds[i]["solution"]
        final_ans = _extract_math_answer(solution)
        a = _format_assistant_answer(solution, final_ans, use_cot=use_cot)
        msgs.append({"role": "user", "content": f"Problem: {problem}"})
        msgs.append({"role": "assistant", "content": a})
    return msgs


class ReasoningEval:
    def __init__(
        self,
        model,
        tokenizer,
        number_of_tests: Optional[int] = None,
        gsm8k_split: str = "test",
        math_split: str = "test",
        math_dataset: str = "HuggingFaceH4/MATH-500",
        use_cot: bool = True,
        seed: int = 0,
        gsm8k_shots: int = 8,
        math_shots: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.number_of_tests = number_of_tests
        self.gsm8k_split = gsm8k_split
        self.math_split = math_split
        self.math_dataset = math_dataset
        self.use_cot = use_cot
        self.seed = seed
        self.gsm8k_shots = gsm8k_shots
        self.math_shots = math_shots
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._gsm8k = None
        self._math = None
        default_gsm8k = os.path.join("data", "reasoning", "gsm8k_test.jsonl")
        default_math = os.path.join("data", "reasoning", "math_500_test.jsonl")
        self.gsm8k_data_path = os.environ.get("REASONING_GSM8K_PATH") or (
            default_gsm8k if os.path.exists(default_gsm8k) else None
        )
        self.math_data_path = os.environ.get("REASONING_MATH_PATH") or (
            default_math if os.path.exists(default_math) else None
        )

    def _load_gsm8k(self) -> List[dict]:
        if self._gsm8k is None:
            if self.gsm8k_data_path:
                data = _load_local_json_records(self.gsm8k_data_path)
                self._gsm8k = [{"question": r["question"], "answer": r["answer"]} for r in data]
            else:
                ds = _load_dataset_no_verify("gsm8k", "main", split=self.gsm8k_split)
                self._gsm8k = [{"question": r["question"], "answer": r["answer"]} for r in ds]
        return self._gsm8k

    def _load_math(self) -> List[dict]:
        if self._math is None:
            if self.math_data_path:
                data = _load_local_json_records(self.math_data_path)
                self._math = [{"problem": r["problem"], "solution": r["solution"]} for r in data]
            else:
                ds = _load_dataset_no_verify(self.math_dataset, None, split=self.math_split)
                self._math = [{"problem": r["problem"], "solution": r["solution"]} for r in ds]
        return self._math

    def _eval_dataset(
        self,
        dataset: List[dict],
        question_key: str,
        answer_key: str,
        answer_parser,
        name: str,
        build_query_messages,
        fewshot_messages: List[dict],
        system_message: Optional[dict],
    ) -> EvalResult:
        correct = 0
        total = 0
        do_sample = self.temperature > 0
        device = next(self.model.parameters()).device

        for idx in _pick_indices(len(dataset), self.number_of_tests, self.seed):
            sample = dataset[idx]
            messages = []
            if system_message:
                messages.append(system_message)
            messages.extend(fewshot_messages)
            messages.extend(build_query_messages(sample[question_key], use_cot=self.use_cot))
            inputs = _encode_chat_or_plain(self.tokenizer, messages, device=device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            gen = outputs[0][inputs["input_ids"].shape[1]:]
            pred_text = self.tokenizer.decode(gen, skip_special_tokens=True)

            gold = answer_parser(sample[answer_key])
            pred = answer_parser(pred_text)
            correct += int(gold is not None and pred == gold)
            total += 1

        return EvalResult(name=name, correct=correct, total=total)

    def evaluate(
        self,
        results: Dict[str, Any],
        record_path: str,
        gsm8k_flag: bool = True,
        math_flag: bool = True,
    ) -> Dict[str, Any]:
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant that solves math word problems. "
                "Always finish with exactly one line: Final answer: <answer>."
            ),
        }
        gsm8k_fewshot = _build_gsm8k_fewshot_messages(
            self.gsm8k_shots, self.seed, use_cot=self.use_cot
        )
        math_fewshot = _build_math_fewshot_messages(
            self.math_shots, self.seed, use_cot=self.use_cot, dataset_name=self.math_dataset
        )

        if gsm8k_flag:
            try:
                ds = self._load_gsm8k()
                res = self._eval_dataset(
                    dataset=ds,
                    question_key="question",
                    answer_key="answer",
                    answer_parser=_extract_final_number,
                    name="gsm8k",
                    build_query_messages=_build_gsm8k_query_messages,
                    fewshot_messages=gsm8k_fewshot,
                    system_message=system_message,
                )
                results["gsm8k"] = {
                    "accuracy": res.accuracy,
                    "correct": res.correct,
                    "total": res.total,
                    "shots": self.gsm8k_shots,
                    "limit": self.number_of_tests,
                }
            except Exception as exc:
                results["gsm8k_error"] = str(exc)

        if math_flag:
            try:
                ds = self._load_math()
                res = self._eval_dataset(
                    dataset=ds,
                    question_key="problem",
                    answer_key="solution",
                    answer_parser=_extract_math_answer,
                    name="math",
                    build_query_messages=_build_math_query_messages,
                    fewshot_messages=math_fewshot,
                    system_message=system_message,
                )
                results["math"] = {
                    "accuracy": res.accuracy,
                    "correct": res.correct,
                    "total": res.total,
                    "shots": self.math_shots,
                    "limit": self.number_of_tests,
                }
            except Exception as exc:
                results["math_error"] = str(exc)

        return results
def _load_dataset_no_verify(name: str, config: Optional[str], split: str):
    try:
        return load_dataset(name, config, split=split, verification_mode="no_checks")
    except TypeError:
        return load_dataset(name, config, split=split, ignore_verifications=True)


def _load_local_json_records(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()
        if not raw:
            return records
        if raw[0] == "[":
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            raise ValueError("JSON must be a list of objects.")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
