from __future__ import annotations

"""Greedy generation with a full per-token trajectory trace for viz/index.html.

The sequence is generated first, then a single instrumented forward pass over
the complete sequence captures, for every position:
  - first-layer-and-later MoE top-k expert ids and gate weights per trajectory
  - aggregator alpha (alt trajectories + null abstention)
  - per-trajectory next-token top-1 predictions and gold-free logprobs
  - fused and base next-token top-1
  - base-vs-alt final hidden cosine

Token semantics for the viewer: the record at position t describes the
computation that PREDICTS token t+1, so clicking a generated token shows the
routing of the forward position that emitted it.

Output: a single JSON file (default viz/traces/<name>.json) consumed by the
static page in viz/index.html.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import TrajectoryEnsembleForCausalLM  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with a per-token trajectory trace.")
    parser.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-moe-16b-chat")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--num_trajectories", type=int, choices=[3, 5], default=3)
    parser.add_argument("--prompt", default="If a train travels 60 miles in 2 hours, what is its speed?")
    parser.add_argument("--chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out", default=None, help="Default: viz/traces/trace.json")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, local_files_only=args.local_files_only
    )
    model = TrajectoryEnsembleForCausalLM.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        num_trajectories=args.num_trajectories,
        local_files_only=args.local_files_only,
    )
    adapter_config = None
    if args.adapter:
        adapter_config = model.load_adapter(args.adapter)
    model.eval()

    if args.chat_template:
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            prompt_text = args.prompt
    else:
        prompt_text = args.prompt

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    print(f"[trace] prompt tokens: {prompt_len}; generating {args.max_new_tokens} tokens...")
    generated = model.greedy_generate(
        input_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    sequence = generated[0]
    seq_len = sequence.shape[0]
    print(f"[trace] sequence length {seq_len}; running instrumented forward...")

    with torch.no_grad():
        output = model(
            input_ids=sequence.unsqueeze(0),
            attention_mask=torch.ones(1, seq_len, dtype=torch.long, device=sequence.device),
            return_base_logits=True,
            return_token_routing=True,
            return_pre_norm_hidden=True,
            return_aggregator_alpha=True,
        )

        pre_norm = output.pre_norm_by_traj[0]  # (N, S, D)
        num_traj = pre_norm.shape[0]
        traj_top1 = []
        for traj_idx in range(num_traj):
            hidden = model.deepseek_model.norm(pre_norm[traj_idx])
            logits = model.lm_head(hidden).float()
            log_probs = F.log_softmax(logits, dim=-1)
            top = log_probs.max(dim=-1)
            traj_top1.append(
                {
                    "ids": top.indices.cpu().tolist(),
                    "logprobs": [round(value, 4) for value in top.values.cpu().tolist()],
                }
            )

        fused_lp = F.log_softmax(output.logits[0].float(), dim=-1)
        fused_top = fused_lp.max(dim=-1)
        base_lp = F.log_softmax(output.base_logits[0].float(), dim=-1)
        base_top = base_lp.max(dim=-1)

        base_hidden = pre_norm[0].float()
        cosines = []
        for alt_idx in range(1, num_traj):
            cos = F.cosine_similarity(base_hidden, pre_norm[alt_idx].float(), dim=-1)
            cosines.append([round(value, 4) for value in cos.cpu().tolist()])

        alpha = output.aggregator_alpha[0].float().cpu()  # (P(+null), S)
        routing = {}
        for layer_idx, payload in sorted(output.token_routing.items()):
            routing[str(layer_idx)] = {
                "topk_idx": payload["topk_idx"][0].tolist(),  # (N, S, k)
                "topk_weight": [
                    [[round(weight, 3) for weight in token] for token in traj]
                    for traj in payload["topk_weight"][0].tolist()
                ],
            }

    tokens = [
        {"text": tokenizer.decode([token_id]), "id": int(token_id), "is_generated": idx >= prompt_len}
        for idx, token_id in enumerate(sequence.cpu().tolist())
    ]

    seed_scales = model.router_noise.noise_scale_value().detach().float().cpu().view(-1).tolist()
    trace = {
        "meta": {
            "model": args.model_name_or_path,
            "num_trajectories": num_traj,
            "n_experts": model.router_noise.n_experts,
            "top_k": model.top_k,
            "moe_layers": sorted(int(layer) for layer in output.token_routing),
            "seed_target_layer": model.router_noise.target_layer_idx,
            "seed_scales": [round(value, 4) for value in seed_scales],
            "include_null_candidate": model.aggregator.include_null_candidate,
            "prompt_len": prompt_len,
            "adapter": args.adapter,
            "adapter_config": adapter_config,
            "prompt": args.prompt,
        },
        "tokens": tokens,
        "alpha": [[round(value, 4) for value in row] for row in alpha.tolist()],
        "traj_top1": traj_top1,
        "fused_top1": {
            "ids": fused_top.indices.cpu().tolist(),
            "logprobs": [round(value, 4) for value in fused_top.values.cpu().tolist()],
        },
        "base_top1": {
            "ids": base_top.indices.cpu().tolist(),
            "logprobs": [round(value, 4) for value in base_top.values.cpu().tolist()],
        },
        "base_alt_cosine": cosines,
        "routing": routing,
        "token_strings": {
            str(token_id): tokenizer.decode([token_id])
            for entry in (traj_top1 + [
                {"ids": fused_top.indices.cpu().tolist()},
                {"ids": base_top.indices.cpu().tolist()},
            ])
            for token_id in entry["ids"]
        },
    }

    out_path = Path(args.out) if args.out else ROOT / "viz" / "traces" / "trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, ensure_ascii=False)
    completion = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
    print(f"[trace] completion: {completion!r}")
    print(f"[trace] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("[trace] view: python -m http.server -d viz 8000  ->  http://localhost:8000/?trace=traces/trace.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
