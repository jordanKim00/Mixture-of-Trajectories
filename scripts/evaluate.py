from __future__ import annotations

"""Unified evaluation on the RoE-matched suite (see scripts/prepare_eval_data.py).

Task protocols:
  math (gsm8k, svamp, addsub, singleeq, multiarith):
      greedy generation, last-number extraction (RoE convention).
  choice (arc_easy, arc_challenge, openbookqa, siqa, hellaswag):
      per-choice continuation log-likelihood; reports raw and length-normalized
      accuracy.
  code (humaneval, humanevalplus):
      completion-style greedy generation; samples are written for offline
      execution with evalplus (this script does not execute generated code).

Modes:
  --mode fused  trajectory wrapper (optionally with --adapter checkpoint)
  --mode base   raw frozen HF model, no wrapper (clean baseline)

Fused generation uses per-trajectory KV caching (B*N cache rows, prompt-frozen
context gate, running global mean for the aggregator query), so decode cost is
O(len) per trajectory like standard generation.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import TrajectoryEnsembleForCausalLM  # noqa: E402

MATH_TASKS = ("gsm8k", "svamp", "addsub", "singleeq", "multiarith")
CHOICE_TASKS = ("arc_easy", "arc_challenge", "openbookqa", "siqa", "hellaswag")
CODE_TASKS = ("humaneval", "humanevalplus")
ALL_TASKS = MATH_TASKS + CHOICE_TASKS + CODE_TASKS

CODE_STOP_MARKERS = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert ")


def extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    candidates = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not candidates:
        return None
    return candidates[-1].replace(",", "")


def numbers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) <= 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


def truncate_code(completion: str) -> str:
    cut = len(completion)
    for marker in CODE_STOP_MARKERS:
        idx = completion.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return completion[:cut]


def load_eval_jsonl(eval_dir: Path, task: str, limit: int) -> List[Dict[str, object]]:
    path = eval_dir / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_eval_data.py first")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


class FusedRunner:
    """Trajectory wrapper runner (fused logits)."""

    name = "fused"

    def __init__(self, args) -> None:
        self.model = TrajectoryEnsembleForCausalLM.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            num_trajectories=args.num_trajectories,
            disable_seed_noise=args.disable_seed_noise,
            local_files_only=args.local_files_only,
        )
        if args.adapter:
            self.model.load_adapter(args.adapter)
        self.model.eval()
        self.input_device = self.model.input_device

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids.to(self.input_device),
            attention_mask=attention_mask.to(self.input_device),
        )
        return output.logits

    @torch.no_grad()
    def greedy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        return self.model.greedy_generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )


class BaseRunner:
    """Raw frozen HF model runner (no wrapper)."""

    name = "base"

    def __init__(self, args) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        self.model.eval()
        self.input_device = self.model.model.embed_tokens.weight.device

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids.to(self.input_device),
            attention_mask=attention_mask.to(self.input_device),
        )
        return output.logits.float()

    @torch.no_grad()
    def greedy(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: Optional[int]) -> torch.Tensor:
        return self.model.generate(
            input_ids=input_ids.to(self.input_device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )


def math_prompt(tokenizer, question: str) -> str:
    message = (
        f"{question}\n"
        "Please reason step by step, and give the final numeric answer after '####'."
    )
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return message + "\nAnswer:"


def choice_texts(record: Dict[str, object]) -> tuple[str, List[str]]:
    question = str(record["question"])
    choices = [str(choice) for choice in record["choices"]]  # type: ignore[index]
    if record["task"] == "hellaswag":
        return question, [" " + choice for choice in choices]
    return f"Question: {question}\nAnswer:", [" " + choice for choice in choices]


@torch.no_grad()
def score_choices(runner, tokenizer, prompt: str, choices: List[str]) -> tuple[List[float], List[float]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    sums: List[float] = []
    means: List[float] = []
    for choice in choices:
        choice_ids = tokenizer(choice, add_special_tokens=False).input_ids
        if not choice_ids:
            sums.append(float("-inf"))
            means.append(float("-inf"))
            continue
        full = torch.tensor([prompt_ids + choice_ids], dtype=torch.long)
        attention = torch.ones_like(full)
        logits = runner.logits(full, attention)
        log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = full[:, 1:].to(log_probs.device)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        choice_lp = token_lp[len(prompt_ids) - 1 :]
        sums.append(float(choice_lp.sum()))
        means.append(float(choice_lp.mean()))
    return sums, means


def run_math_task(runner, tokenizer, task: str, rows, args, out_dir: Path) -> Dict[str, object]:
    correct = 0
    records = []
    eos = tokenizer.eos_token_id
    for idx, row in enumerate(rows):
        prompt = math_prompt(tokenizer, str(row["question"]))
        input_ids = torch.tensor([tokenizer(prompt, add_special_tokens=False).input_ids], dtype=torch.long)
        generated = runner.greedy(input_ids, args.max_new_tokens, eos)
        completion = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
        pred = extract_last_number(completion)
        ok = numbers_match(pred, str(row["answer"]))
        correct += int(ok)
        records.append(
            {"task": task, "idx": idx, "pred": pred, "gold": row["answer"], "correct": ok, "completion": completion}
        )
        if (idx + 1) % 10 == 0:
            print(f"[{task}] {idx + 1}/{len(rows)} acc={correct / (idx + 1):.3f}")
    _write_records(out_dir / f"{task}.jsonl", records)
    return {"task": task, "n": len(rows), "accuracy": correct / max(len(rows), 1)}


def run_choice_task(runner, tokenizer, task: str, rows, args, out_dir: Path) -> Dict[str, object]:
    correct_sum = 0
    correct_norm = 0
    records = []
    for idx, row in enumerate(rows):
        prompt, choices = choice_texts(row)
        sums, means = score_choices(runner, tokenizer, prompt, choices)
        pred_sum = int(max(range(len(sums)), key=sums.__getitem__))
        pred_norm = int(max(range(len(means)), key=means.__getitem__))
        gold = int(row["answer_idx"])  # type: ignore[arg-type]
        correct_sum += int(pred_sum == gold)
        correct_norm += int(pred_norm == gold)
        records.append(
            {"task": task, "idx": idx, "pred": pred_sum, "pred_norm": pred_norm, "gold": gold, "scores": sums}
        )
        if (idx + 1) % 50 == 0:
            print(f"[{task}] {idx + 1}/{len(rows)} acc_norm={correct_norm / (idx + 1):.3f}")
    _write_records(out_dir / f"{task}.jsonl", records)
    return {
        "task": task,
        "n": len(rows),
        "accuracy": correct_sum / max(len(rows), 1),
        "accuracy_norm": correct_norm / max(len(rows), 1),
    }


def run_code_task(runner, tokenizer, task: str, rows, args, out_dir: Path) -> Dict[str, object]:
    records = []
    eos = tokenizer.eos_token_id
    for idx, row in enumerate(rows):
        prompt = str(row["prompt"])
        input_ids = torch.tensor([tokenizer(prompt, add_special_tokens=False).input_ids], dtype=torch.long)
        generated = runner.greedy(input_ids, args.max_new_tokens, eos)
        completion = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
        records.append({"task_id": row["task_id"], "completion": truncate_code(completion)})
        if (idx + 1) % 10 == 0:
            print(f"[{task}] generated {idx + 1}/{len(rows)}")
    _write_records(out_dir / f"{task}_samples.jsonl", records)
    return {
        "task": task,
        "n": len(rows),
        "note": f"samples saved to {task}_samples.jsonl; score offline with evalplus",
    }


def _write_records(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate on the RoE-matched suite.")
    parser.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-moe-16b-chat")
    parser.add_argument("--mode", choices=["fused", "base"], default="fused")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--disable_seed_noise", action="store_true")
    parser.add_argument("--num_trajectories", type=int, choices=[3, 5], default=3)
    parser.add_argument("--tasks", default="all", help="Comma list or 'all'/'math'/'choice'/'code'.")
    parser.add_argument("--eval_dir", default=str(ROOT / "data" / "eval"))
    parser.add_argument("--out_dir", default=None, help="Default: results/<mode>[_adapter].")
    parser.add_argument("--limit", type=int, default=0, help="Per-task example cap (0 = all).")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.tasks == "all":
        tasks = list(ALL_TASKS)
    elif args.tasks == "math":
        tasks = list(MATH_TASKS)
    elif args.tasks == "choice":
        tasks = list(CHOICE_TASKS)
    elif args.tasks == "code":
        tasks = list(CODE_TASKS)
    else:
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        unknown = [task for task in tasks if task not in ALL_TASKS]
        if unknown:
            raise ValueError(f"unknown tasks: {unknown}")

    run_name = args.out_dir
    if run_name is None:
        suffix = "_adapter" if args.adapter else ""
        suffix += "_noseed" if args.disable_seed_noise else ""
        run_name = str(ROOT / "results" / f"{args.mode}{suffix}")
    out_dir = Path(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    runner = FusedRunner(args) if args.mode == "fused" else BaseRunner(args)

    eval_dir = Path(args.eval_dir)
    metrics: List[Dict[str, object]] = []
    started = time.time()
    for task in tasks:
        rows = load_eval_jsonl(eval_dir, task, args.limit)
        print(f"[eval] {task}: {len(rows)} examples ({args.mode})")
        if task in MATH_TASKS:
            metrics.append(run_math_task(runner, tokenizer, task, rows, args, out_dir))
        elif task in CHOICE_TASKS:
            metrics.append(run_choice_task(runner, tokenizer, task, rows, args, out_dir))
        else:
            metrics.append(run_code_task(runner, tokenizer, task, rows, args, out_dir))
        print(f"[eval] {task}: {metrics[-1]}")

    summary = {
        "mode": args.mode,
        "adapter": args.adapter,
        "disable_seed_noise": args.disable_seed_noise,
        "num_trajectories": args.num_trajectories,
        "limit": args.limit,
        "elapsed_sec": round(time.time() - started, 1),
        "metrics": metrics,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
