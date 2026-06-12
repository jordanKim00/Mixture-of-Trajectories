from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import TrajectoryEnsembleForCausalLM


def encode_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        return tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="No-cache greedy generation smoke.")
    parser.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-moe-16b-chat")
    parser.add_argument("--num_trajectories", type=int, choices=[3, 5], default=3)
    parser.add_argument("--agg_dim", type=int, default=256)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--noise_scale_max", type=float, default=1.0)
    parser.add_argument("--train_router_mode", choices=["hard", "soft_all", "st_topk"], default="st_topk")
    parser.add_argument("--soft_temperature", type=float, default=1.0)
    parser.add_argument("--residual_scale_max", type=float, default=0.25)
    parser.add_argument("--include_null_aggregation_candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aggregator_value_mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--disable_seed_noise", action="store_true")
    parser.add_argument("--context_seed_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context_scale_max_delta", type=float, default=0.5)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default="다음 수를 더하시오: 17 + 25 =")
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify_against_no_cache",
        action="store_true",
        help="Also run the uncached path and compare generated token ids and timing.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = TrajectoryEnsembleForCausalLM.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        num_trajectories=args.num_trajectories,
        agg_dim=args.agg_dim,
        noise_scale=args.noise_scale,
        noise_scale_max=args.noise_scale_max,
        train_router_mode=args.train_router_mode,
        soft_temperature=args.soft_temperature,
        residual_scale_max=args.residual_scale_max,
        include_null_aggregation_candidate=args.include_null_aggregation_candidate,
        aggregator_value_mode=args.aggregator_value_mode,
        disable_seed_noise=args.disable_seed_noise,
        context_seed_gate=args.context_seed_gate,
        context_scale_max_delta=args.context_scale_max_delta,
        local_files_only=args.local_files_only,
    )
    model.eval()

    input_ids = encode_prompt(tokenizer, args.prompt).to(model.input_device)
    attention_mask = torch.ones_like(input_ids, device=model.input_device)

    import time

    started = time.time()
    generated = model.greedy_generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=args.use_cache,
    )
    elapsed = time.time() - started
    print(f"[cache={args.use_cache}] {elapsed:.2f}s")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))

    if args.verify_against_no_cache and args.use_cache:
        started = time.time()
        reference = model.greedy_generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        elapsed_ref = time.time() - started
        match = generated.shape == reference.shape and bool((generated == reference).all().item())
        print(f"[cache=False] {elapsed_ref:.2f}s | tokens match cached run: {match}")
        if not match:
            print("cached :", tokenizer.decode(generated[0], skip_special_tokens=True))
            print("nocache:", tokenizer.decode(reference[0], skip_special_tokens=True))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
