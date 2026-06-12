from __future__ import annotations

"""Download the RoE-matched evaluation suite into data/eval/*.jsonl.

Tasks follow the RoE paper setting (arXiv:2509.17238):
  math (generation):     gsm8k, svamp, addsub, singleeq, multiarith
  commonsense (choice):  arc_easy, arc_challenge, openbookqa, siqa, hellaswag
  code (generation):     humaneval, humanevalplus

Unified schemas (one JSON object per line):
  generation math: {"task", "question", "answer"}
  multiple choice: {"task", "question", "choices", "answer_idx"}
  code:            {"task", "task_id", "prompt", "entry_point", "test"}

SIQA and HellaSwag use validation splits because their test labels are hidden.
Each download failure is reported and skipped so one gated/renamed dataset does
not block the rest of the suite.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "eval"

LETTER_TO_IDX = {letter: idx for idx, letter in enumerate("ABCDEFGH")}


def _clean_number(text: str) -> Optional[str]:
    if text is None:
        return None
    match = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text))
    if not match:
        return str(text).strip() or None
    return match[-1].replace(",", "")


def _rows_gsm8k() -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("openai/gsm8k", "main", split="test"):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        yield {"task": "gsm8k", "question": row["question"], "answer": gold}


def _rows_svamp() -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("ChilleD/SVAMP", split="test"):
        question = f"{row['Body'].strip()} {row['Question'].strip()}"
        yield {"task": "svamp", "question": question, "answer": _clean_number(row["Answer"])}


def _rows_multiarith() -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("ChilleD/MultiArith", split="test"):
        yield {
            "task": "multiarith",
            "question": row["question"].strip(),
            "answer": _clean_number(row["final_ans"]),
        }


def _rows_lila(config: str, task: str) -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("allenai/lila", config, split="test"):
        yield {
            "task": task,
            "question": row["input"].strip(),
            "answer": _clean_number(row["output_answer"]),
        }


def _rows_arc(config: str, task: str) -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("allenai/ai2_arc", config, split="test"):
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            continue
        yield {
            "task": task,
            "question": row["question"],
            "choices": list(row["choices"]["text"]),
            "answer_idx": labels.index(row["answerKey"]),
        }


def _rows_openbookqa() -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("allenai/openbookqa", "main", split="test"):
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            continue
        yield {
            "task": "openbookqa",
            "question": row["question_stem"],
            "choices": list(row["choices"]["text"]),
            "answer_idx": labels.index(row["answerKey"]),
        }


def _rows_siqa() -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    # Parquet mirror; allenai/social_i_qa's loader script breaks on datasets>=3.
    for row in load_dataset("lighteval/siqa", split="validation"):
        yield {
            "task": "siqa",
            "question": f"{row['context']} {row['question']}",
            "choices": [row["answerA"], row["answerB"], row["answerC"]],
            "answer_idx": int(row["label"]) - 1,
        }


def _rows_hellaswag() -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True):
        yield {
            "task": "hellaswag",
            "question": row["ctx"],
            "choices": list(row["endings"]),
            "answer_idx": int(row["label"]),
        }


def _rows_humaneval() -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("openai/openai_humaneval", split="test"):
        yield {
            "task": "humaneval",
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
        }


def _rows_humanevalplus() -> Iterable[Dict[str, object]]:
    from datasets import load_dataset

    for row in load_dataset("evalplus/humanevalplus", split="test"):
        yield {
            "task": "humanevalplus",
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row.get("test", ""),
        }


TASK_LOADERS: Dict[str, Callable[[], Iterable[Dict[str, object]]]] = {
    "gsm8k": _rows_gsm8k,
    "svamp": _rows_svamp,
    "addsub": lambda: _rows_lila("addsub", "addsub"),
    "singleeq": lambda: _rows_lila("singleq", "singleeq"),
    "multiarith": _rows_multiarith,
    "arc_easy": lambda: _rows_arc("ARC-Easy", "arc_easy"),
    "arc_challenge": lambda: _rows_arc("ARC-Challenge", "arc_challenge"),
    "openbookqa": _rows_openbookqa,
    "siqa": _rows_siqa,
    "hellaswag": _rows_hellaswag,
    "humaneval": _rows_humaneval,
    "humanevalplus": _rows_humanevalplus,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare RoE-matched eval data.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task names or 'all'. Available: " + ",".join(TASK_LOADERS),
    )
    args = parser.parse_args()

    if args.tasks == "all":
        tasks: List[str] = list(TASK_LOADERS)
    else:
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        unknown = [task for task in tasks if task not in TASK_LOADERS]
        if unknown:
            raise ValueError(f"unknown tasks: {unknown}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    for task in tasks:
        out_path = out_dir / f"{task}.jsonl"
        try:
            count = 0
            with out_path.open("w", encoding="utf-8") as handle:
                for record in TASK_LOADERS[task]():
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            print(f"[eval] {task}: {count} examples -> {out_path}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(task)
            if out_path.exists():
                out_path.unlink()
            print(f"[eval] {task}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)

    if failures:
        print(f"[eval] failed tasks: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
