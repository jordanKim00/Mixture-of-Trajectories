from __future__ import annotations

"""Build the staged adapter-training mixtures under data/train/.

Stage 1 (continuation warm-up): plain text documents from broad corpora, so the
aggregator first adapts to the frozen hidden distribution.
Stage 3 (SFT specialization): prompt+completion pairs across hard multi-domain
instruction data, excluding everything in the RoE-matched eval suite.
Stage 2 (hard-prefix mining) needs the GPU model and lives in
scripts/mine_hard_prefixes.py; it consumes the pool built here.

Every emitted example passes the n-gram decontamination filter against
data/eval/*.jsonl (run scripts/prepare_eval_data.py first; the script warns if
the eval set is missing). Sources are streamed with per-source caps derived
from the bucket ratios, and any failing source is skipped with a warning so a
gated dataset does not block the rest.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from decontamination import build_eval_ngram_index, is_contaminated  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "train"
DEFAULT_EVAL = ROOT / "data" / "eval"

MAX_TEXT_CHARS = 6000


def _stream(dataset_id: str, config: Optional[str] = None, split: str = "train", **kwargs):
    from datasets import load_dataset

    return load_dataset(dataset_id, config, split=split, streaming=True, **kwargs)


def _truncate(text: str) -> str:
    return text[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# Stage 1: continuation corpora -> {"text": ...}
# ---------------------------------------------------------------------------

def _docs_fineweb_edu() -> Iterator[Dict[str, str]]:
    for row in _stream("HuggingFaceFW/fineweb-edu", "sample-10BT"):
        yield {"text": _truncate(row["text"])}


def _docs_slimpajama() -> Iterator[Dict[str, str]]:
    for row in _stream("DKYoon/SlimPajama-6B"):
        yield {"text": _truncate(row["text"])}


def _docs_openwebmath() -> Iterator[Dict[str, str]]:
    for row in _stream("open-web-math/open-web-math"):
        yield {"text": _truncate(row["text"])}


def _docs_stack_smol() -> Iterator[Dict[str, str]]:
    for row in _stream("bigcode/the-stack-smol", "data/python"):
        yield {"text": _truncate(row["content"])}


def _docs_wikipedia() -> Iterator[Dict[str, str]]:
    for row in _stream("wikimedia/wikipedia", "20231101.en"):
        yield {"text": _truncate(row["text"])}


STAGE1_SOURCES: List[tuple[str, float, Callable[[], Iterator[Dict[str, str]]]]] = [
    ("fineweb_edu", 0.40, _docs_fineweb_edu),
    ("slimpajama", 0.20, _docs_slimpajama),
    ("openwebmath", 0.15, _docs_openwebmath),
    ("stack_python", 0.15, _docs_stack_smol),
    ("wikipedia", 0.10, _docs_wikipedia),
]


# ---------------------------------------------------------------------------
# Stage 3: SFT pairs -> {"prompt": ..., "completion": ...} or {"messages": ...}
# ---------------------------------------------------------------------------

def _sft_tulu3() -> Iterator[Dict[str, object]]:
    for row in _stream("allenai/tulu-3-sft-mixture"):
        messages = row.get("messages")
        if isinstance(messages, list) and messages and messages[-1].get("role") == "assistant":
            yield {"messages": messages}


def _sft_numinamath() -> Iterator[Dict[str, object]]:
    for row in _stream("AI-MO/NuminaMath-CoT"):
        yield {"prompt": row["problem"], "completion": " " + row["solution"]}


def _sft_codefeedback() -> Iterator[Dict[str, object]]:
    for row in _stream("m-a-p/CodeFeedback-Filtered-Instruction"):
        yield {"prompt": row["query"], "completion": " " + row["answer"]}


def _sft_commonsenseqa() -> Iterator[Dict[str, object]]:
    for row in _stream("tau/commonsense_qa"):
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            continue
        answer = row["choices"]["text"][labels.index(row["answerKey"])]
        options = " / ".join(row["choices"]["text"])
        yield {
            "prompt": f"Question: {row['question']}\nOptions: {options}\nAnswer:",
            "completion": f" {answer}",
        }


def _sft_piqa() -> Iterator[Dict[str, object]]:
    for row in _stream("ybisk/piqa", trust_remote_code=True):
        answer = row["sol1"] if int(row["label"]) == 0 else row["sol2"]
        yield {"prompt": f"Goal: {row['goal']}\nAnswer:", "completion": f" {answer}"}


def _sft_winogrande() -> Iterator[Dict[str, object]]:
    for row in _stream("allenai/winogrande", "winogrande_xl", trust_remote_code=True):
        answer = row["option1"] if row["answer"] == "1" else row["option2"]
        yield {
            "prompt": f"Fill in the blank: {row['sentence']}\nAnswer:",
            "completion": f" {answer}",
        }


def _sft_sciq() -> Iterator[Dict[str, object]]:
    for row in _stream("allenai/sciq"):
        support = (row.get("support") or "").strip()
        prefix = f"{support}\n" if support else ""
        yield {
            "prompt": f"{prefix}Question: {row['question']}\nAnswer:",
            "completion": f" {row['correct_answer']}",
        }


def _sft_hotpotqa() -> Iterator[Dict[str, object]]:
    for row in _stream("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True):
        yield {"prompt": f"Question: {row['question']}\nAnswer:", "completion": f" {row['answer']}"}


STAGE3_SOURCES: List[tuple[str, float, Callable[[], Iterator[Dict[str, object]]]]] = [
    ("tulu3", 0.25, _sft_tulu3),
    ("numinamath", 0.20, _sft_numinamath),
    ("codefeedback", 0.20, _sft_codefeedback),
    ("commonsenseqa", 0.05, _sft_commonsenseqa),
    ("piqa", 0.04, _sft_piqa),
    ("winogrande", 0.03, _sft_winogrande),
    ("sciq", 0.10, _sft_sciq),
    ("hotpotqa", 0.13, _sft_hotpotqa),
]


def _example_text(example: Dict[str, object]) -> str:
    if "messages" in example:
        return " ".join(str(m.get("content", "")) for m in example["messages"])  # type: ignore[index]
    return f"{example.get('prompt', '')} {example.get('completion', '')} {example.get('text', '')}"


def _collect(
    sources: List[tuple[str, float, Callable[[], Iterator[Dict[str, object]]]]],
    total: int,
    ngram_index,
    seed: int,
) -> List[Dict[str, object]]:
    collected: List[Dict[str, object]] = []
    for name, ratio, loader in sources:
        target = max(1, int(round(total * ratio)))
        kept = 0
        dropped = 0
        try:
            for example in loader():
                text = _example_text(example)
                if len(text.strip()) < 32:
                    continue
                if is_contaminated(text, ngram_index):
                    dropped += 1
                    continue
                example["source"] = name
                collected.append(example)
                kept += 1
                if kept >= target:
                    break
            print(f"[train] {name}: kept {kept}/{target} (decontaminated {dropped})")
        except Exception as exc:  # noqa: BLE001 - skip broken/gated sources
            print(f"[train] {name}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
    random.Random(seed).shuffle(collected)
    return collected


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare staged adapter training data.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--eval_dir", default=str(DEFAULT_EVAL))
    parser.add_argument("--stage1_size", type=int, default=20000)
    parser.add_argument("--stage3_size", type=int, default=20000)
    parser.add_argument("--stages", default="1,3", help="Comma list out of {1,3}.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ngram_index = build_eval_ngram_index(args.eval_dir)
    if not ngram_index:
        print(
            "[train] WARNING: no eval n-gram index found; run scripts/prepare_eval_data.py "
            "first so decontamination can be applied.",
            file=sys.stderr,
        )
    else:
        print(f"[train] decontamination index: {len(ngram_index)} n-grams")

    stages = {stage.strip() for stage in args.stages.split(",")}
    if "1" in stages:
        rows = _collect(STAGE1_SOURCES, args.stage1_size, ngram_index, args.seed)
        count = _write_jsonl(out_dir / "stage1_continuation.jsonl", rows)
        print(f"[train] stage1_continuation.jsonl: {count} examples")
    if "3" in stages:
        rows = _collect(STAGE3_SOURCES, args.stage3_size, ngram_index, args.seed + 1)
        count = _write_jsonl(out_dir / "stage3_sft.jsonl", rows)
        print(f"[train] stage3_sft.jsonl: {count} examples")
    print(
        "[train] stage2 (hard-prefix mining) requires the GPU model: "
        "python scripts/mine_hard_prefixes.py --pool data/train/stage1_continuation.jsonl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
