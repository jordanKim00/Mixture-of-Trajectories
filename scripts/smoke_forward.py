from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

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
    parser = argparse.ArgumentParser(description="Forward smoke for trajectory aggregation.")
    parser.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-moe-16b-chat")
    parser.add_argument("--num_trajectories", type=int, choices=[3, 5], default=3)
    parser.add_argument("--agg_dim", type=int, default=256)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--noise_scale_max", type=float, default=1.0)
    parser.add_argument("--noise_init_std", type=float, default=0.02)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--train_router_mode", choices=["hard", "soft_all", "st_topk"], default="st_topk")
    parser.add_argument("--soft_temperature", type=float, default=1.0)
    parser.add_argument("--residual_scale_max", type=float, default=0.25)
    parser.add_argument("--include_null_aggregation_candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aggregator_value_mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--disable_seed_noise", action="store_true")
    parser.add_argument("--context_seed_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context_scale_max_delta", type=float, default=0.5)
    parser.add_argument("--kl_direction", choices=["base_to_fused", "symmetric"], default="base_to_fused")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default="다음 수를 더하시오: 17 + 25 =")
    parser.add_argument("--backward_check", action="store_true")
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
        noise_init_std=args.noise_init_std,
        noise_scale=args.noise_scale,
        noise_scale_max=args.noise_scale_max,
        top_k=args.top_k,
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
    labels = input_ids.clone()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        kl_beta=1.0,
        kl_direction=args.kl_direction,
        return_base_logits=True,
        return_route_stats=True,
        return_aggregator_stats=True,
        return_trajectory_stats=True,
        return_trajectory_prediction_stats=True,
    )

    max_identity_diff = (output.logits - output.base_logits).abs().max().item()
    noise_layers = [
        layer_idx
        for layer_idx, stats in output.route_stats.items()
        if stats["noise_applied"]
    ]
    expected_noise_layers = [] if args.disable_seed_noise else [model.router_noise.target_layer_idx]
    if noise_layers != expected_noise_layers:
        raise AssertionError(f"noise applied at {noise_layers}, expected {expected_noise_layers}")

    print("[forward] ok")
    print(f"  logits_shape={tuple(output.logits.shape)}")
    print(f"  base_identity_max_abs_diff={max_identity_diff:.6e}")
    print(f"  patched_moe_layers={model.patched_moe_layers}")
    print(f"  noise_layers={noise_layers}")
    print(f"  aggregator={json.dumps(output.aggregator_stats, ensure_ascii=False)}")
    print(f"  trajectory={json.dumps(output.trajectory_stats, ensure_ascii=False)}")
    print(f"  trajectory_prediction={json.dumps(output.trajectory_prediction_stats, ensure_ascii=False)}")
    print(f"  path={json.dumps(output.path_stats, ensure_ascii=False)[:1000]}")
    first_layer = model.router_noise.target_layer_idx
    print(
        "  first_layer_route="
        + json.dumps(output.route_stats[first_layer], ensure_ascii=False)[:1000]
    )

    if args.backward_check:
        model.train()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            kl_beta=1.0,
            kl_direction=args.kl_direction,
        )
        output.loss.backward()
        backbone_grads = [
            name
            for name, param in model.base_model.named_parameters()
            if param.grad is not None
        ]
        if backbone_grads:
            raise AssertionError(f"backbone gradients found: {backbone_grads[:5]}")
        trainable_grad_count = sum(
            1 for param in model.trainable_parameters() if param.grad is not None
        )
        if trainable_grad_count == 0:
            raise AssertionError("no adapter gradients found")
        print(f"[backward] ok trainable_grad_tensors={trainable_grad_count}")


if __name__ == "__main__":
    main()
