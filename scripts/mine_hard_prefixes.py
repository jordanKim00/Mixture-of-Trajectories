from __future__ import annotations

"""Stage 2 hard-prefix mining (requires the DeepSeek-MoE GPU model).

Scores every pooled example with the trajectory wrapper and keeps the hardest
fraction. The score follows the method's selective-fusion philosophy: prefer
prefixes where the frozen base route struggles AND the seeded trajectories
actually disagree, because those are the only tokens where latent aggregation
can earn fused-CE gains.

Raw components per example:
    base_ce          (base route difficulty)
    gold_nll_std     (trajectory predictive disagreement)
    route_divergence (1 - first-layer exact top-k match)
    router_entropy   (route ambiguity at the seed layer)

Selection z-normalizes each component across the pool before weighting,
because raw base_ce is ~50x larger than the disagreement terms and would
otherwise dominate — which selects junk text (spam, listings) that is hard to
predict for the wrong reason. Junk is additionally filtered with
data_quality.looks_like_junk, and a base-CE percentile cap drops un-learnable
noise ("hard because garbage" rather than "hard but learnable").

`--rerank_from existing.jsonl` re-runs only the selection on stored mining
metadata (no GPU), for tuning weights/filters after an expensive scoring run.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data_quality import junk_signals  # noqa: E402


def load_pool(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty pool: {path}")
    return rows


def example_text(example: Dict[str, object]) -> str:
    if example.get("text"):
        return str(example["text"])
    if example.get("messages"):
        return "\n".join(str(m.get("content", "")) for m in example["messages"])  # type: ignore[index]
    return f"{example.get('prompt', '')}{example.get('completion', '')}"


def _percentile(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("inf")
    rank = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[rank]


def _zscores(values: List[float]) -> List[float]:
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
    std = var**0.5
    if std < 1e-12:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


COMPONENTS = ("base_ce", "gold_nll_std", "route_divergence", "router_entropy")


def select_hard(scored: List[Dict[str, object]], args) -> List[Dict[str, object]]:
    """Filter junk, cap un-learnable noise, z-normalize, weight, sort."""

    kept: List[Dict[str, object]] = []
    junk_dropped = 0
    for row in scored:
        signals = junk_signals(example_text(row))
        if signals:
            junk_dropped += 1
            continue
        kept.append(row)
    print(f"[mine] junk filter dropped {junk_dropped}/{len(scored)}")
    if not kept:
        return []

    ce_values = [row["mining"]["base_ce"] for row in kept]  # type: ignore[index]
    ce_cap = _percentile(ce_values, args.base_ce_percentile_cap)
    capped = [row for row in kept if row["mining"]["base_ce"] <= ce_cap]  # type: ignore[index]
    print(
        f"[mine] base_ce cap (p{args.base_ce_percentile_cap:.0f} = {ce_cap:.3f}) "
        f"dropped {len(kept) - len(capped)}/{len(kept)}"
    )
    if not capped:
        return []

    weights = {
        "base_ce": args.weight_base_ce,
        "gold_nll_std": args.weight_nll_std,
        "route_divergence": args.weight_route_div,
        "router_entropy": args.weight_entropy,
    }
    z_by_component = {
        name: _zscores([row["mining"][name] for row in capped])  # type: ignore[index]
        for name in COMPONENTS
    }
    for idx, row in enumerate(capped):
        mining = dict(row["mining"])  # type: ignore[arg-type]
        mining["score_raw"] = mining.get("score")
        mining["score"] = sum(weights[name] * z_by_component[name][idx] for name in COMPONENTS)
        row["mining"] = mining
    capped.sort(key=lambda row: row["mining"]["score"], reverse=True)  # type: ignore[index]
    return capped[: max(1, int(len(capped) * args.keep_fraction))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine hard prefixes for stage 2 training.")
    parser.add_argument("--pool", default=None, help="JSONL pool (stage1/stage3 output).")
    parser.add_argument(
        "--rerank_from",
        default=None,
        help="Previously mined JSONL with mining metadata; reselect without the GPU.",
    )
    parser.add_argument("--out", default=str(ROOT / "data" / "train" / "stage2_hard_prefix.jsonl"))
    parser.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-moe-16b-chat")
    parser.add_argument("--adapter", default=None, help="Optional trained adapter checkpoint.")
    parser.add_argument("--num_trajectories", type=int, choices=[3, 5], default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Score at most this many examples (0 = all).")
    parser.add_argument("--keep_fraction", type=float, default=0.25)
    parser.add_argument("--weight_base_ce", type=float, default=1.0)
    parser.add_argument("--weight_nll_std", type=float, default=2.0)
    parser.add_argument("--weight_route_div", type=float, default=1.0)
    parser.add_argument("--weight_entropy", type=float, default=0.25)
    parser.add_argument(
        "--base_ce_percentile_cap",
        type=float,
        default=95.0,
        help="Drop examples above this base-CE percentile (un-learnable noise guard).",
    )
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0.0 < args.keep_fraction <= 1.0:
        raise ValueError("--keep_fraction must be in (0, 1]")
    if (args.pool is None) == (args.rerank_from is None):
        raise ValueError("provide exactly one of --pool (GPU scoring) or --rerank_from (reselect)")

    if args.rerank_from:
        scored = load_pool(Path(args.rerank_from))
        missing = [row for row in scored if "mining" not in row]
        if missing:
            raise ValueError(f"{len(missing)} rows lack mining metadata; rerun GPU scoring")
        selected = select_hard(scored, args)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[mine] reranked: kept {len(selected)}/{len(scored)} -> {out_path}")
        return 0

    import torch
    from transformers import AutoTokenizer

    from src import TrajectoryEnsembleForCausalLM

    pool = load_pool(Path(args.pool))
    if args.limit > 0:
        pool = pool[: args.limit]
    before_junk = len(pool)
    pool = [example for example in pool if not junk_signals(example_text(example))]
    print(f"[mine] pre-scoring junk filter: {before_junk} -> {len(pool)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = TrajectoryEnsembleForCausalLM.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        num_trajectories=args.num_trajectories,
        local_files_only=args.local_files_only,
    )
    if args.adapter:
        model.load_adapter(args.adapter)
    model.eval()

    scored: List[Dict[str, object]] = []
    with torch.no_grad():
        for idx, example in enumerate(pool):
            text = example_text(example)[:8000]
            encoded = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt")
            input_ids = encoded.input_ids.to(model.input_device)
            if input_ids.shape[1] < 8:
                continue
            attention_mask = encoded.attention_mask.to(model.input_device)
            labels = input_ids.clone()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_route_stats=True,
                return_trajectory_prediction_stats=True,
            )
            pred = output.trajectory_prediction_stats or {}
            components = output.loss_components or {}
            base_ce = float(components.get("base_ce", 0.0))
            nll_std = float(pred.get("gold_nll_std_mean", 0.0))
            seed_layer = model.router_noise.target_layer_idx
            route = (output.route_stats or {}).get(seed_layer, {})
            exact = route.get("topk_exact_match_with_base") or [1.0]
            route_div = 1.0 - sum(exact) / len(exact)
            entropy = route.get("entropy_by_traj") or [0.0]
            base_entropy = float(entropy[0])
            score = (
                args.weight_base_ce * base_ce
                + args.weight_nll_std * nll_std
                + args.weight_route_div * route_div
                + args.weight_entropy * base_entropy
            )
            record = dict(example)
            record["mining"] = {
                "score": score,
                "base_ce": base_ce,
                "gold_nll_std": nll_std,
                "route_divergence": route_div,
                "router_entropy": base_entropy,
            }
            scored.append(record)
            if (idx + 1) % 50 == 0:
                print(f"[mine] scored {idx + 1}/{len(pool)}")

    selected = select_hard(scored, args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[mine] kept {len(selected)}/{len(scored)} hard prefixes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
