from __future__ import annotations

"""N-gram decontamination against the evaluation suite.

The RoE-matched evaluation tasks (GSM8K, SVAMP, AddSub, SingleEq, MultiArith,
ARC-Easy/Challenge, OpenBookQA, Social-IQA, HellaSwag, HumanEval, HumanEval+)
must not leak into adapter training data. Exact-match filtering is not enough
for open instruction mixtures, so training examples are dropped when any
normalized word n-gram overlaps the eval corpus.

This is a word-shingle filter, not MinHash/embedding dedup; it catches verbatim
and near-verbatim leaks, which is the dominant contamination mode for these
benchmarks.
"""

import json
import re
from pathlib import Path
from typing import Iterable, Set

DEFAULT_NGRAM = 8

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def text_ngrams(text: str, n: int = DEFAULT_NGRAM) -> Set[tuple]:
    words = normalize_text(text)
    if len(words) < n:
        # Short eval questions are protected by their full word tuple.
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _eval_record_texts(record: dict) -> Iterable[str]:
    for key in ("question", "prompt", "context"):
        value = record.get(key)
        if isinstance(value, str) and value:
            yield value
    choices = record.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, str) and choice:
                yield choice
    answer = record.get("answer")
    if isinstance(answer, str) and len(normalize_text(answer)) >= 4:
        yield answer


def build_eval_ngram_index(eval_dir: str | Path, n: int = DEFAULT_NGRAM) -> Set[tuple]:
    """Collect n-gram shingles from every data/eval/*.jsonl record."""

    index: Set[tuple] = set()
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        return index
    for jsonl_file in sorted(eval_path.glob("*.jsonl")):
        with jsonl_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for text in _eval_record_texts(record):
                    index.update(text_ngrams(text, n=n))
    return index


def is_contaminated(text: str, index: Set[tuple], n: int = DEFAULT_NGRAM) -> bool:
    if not index or not text:
        return False
    words = normalize_text(text)
    if not words:
        return False
    if len(words) < n:
        return tuple(words) in index
    return any(tuple(words[i : i + n]) in index for i in range(len(words) - n + 1))
