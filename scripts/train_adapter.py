from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import TrajectoryEnsembleForCausalLM


@dataclass
class TrainingExample:
    text: Optional[str] = None
    prompt: Optional[str] = None
    target: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None


TARGET_KEYS = ("completion", "response", "answer", "output", "target")


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _messages_from_obj(obj: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return None
    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = _as_text(message.get("role"))
        content = _as_text(message.get("content"))
        if role is None or content is None:
            return None
        normalized.append({"role": role, "content": content})
    return normalized or None


def _example_from_json(obj: Dict[str, Any]) -> Optional[TrainingExample]:
    messages = _messages_from_obj(obj)
    if messages is not None:
        return TrainingExample(messages=messages)

    target = next((_as_text(obj.get(key)) for key in TARGET_KEYS if obj.get(key) is not None), None)
    if target is not None:
        prompt = _as_text(obj.get("prompt"))
        if prompt is None and obj.get("question") is not None:
            prompt = _as_text(obj.get("question"))
        if prompt is None and obj.get("instruction") is not None:
            instruction = _as_text(obj.get("instruction")) or ""
            extra_input = _as_text(obj.get("input"))
            prompt = instruction if extra_input is None else f"{instruction}\n\n{extra_input}"
        if prompt is None:
            prompt = _as_text(obj.get("text"))
        if prompt is not None:
            return TrainingExample(prompt=prompt, target=target)

    text = _as_text(obj.get("text") or obj.get("prompt") or obj.get("question") or obj.get("instruction"))
    if text is not None:
        return TrainingExample(text=text)
    return None


def load_examples(path: Optional[str]) -> List[TrainingExample]:
    if path is None:
        return [
            TrainingExample(prompt="다음 수를 더하시오: 17 + 25 =", target=" 42"),
            TrainingExample(
                prompt="If a train travels 60 miles in 2 hours, what is its speed?",
                target=" 30 miles per hour.",
            ),
        ]
    examples: List[TrainingExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                example = _example_from_json(obj)
                if example is not None:
                    examples.append(example)
            else:
                examples.append(TrainingExample(text=line))
    if not examples:
        raise ValueError(f"no training examples found in {path}")
    return examples


def batches(items: List[TrainingExample], batch_size: int, seed: int) -> Iterable[List[TrainingExample]]:
    rng = random.Random(seed)
    order = list(range(len(items)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            yield [items[idx] for idx in order[start : start + batch_size]]


def _fallback_render_messages(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _append_eos(text: str, tokenizer, append_eos: bool) -> str:
    eos = getattr(tokenizer, "eos_token", None)
    if append_eos and eos and not text.endswith(eos):
        return text + eos
    return text


def _render_example(
    example: TrainingExample,
    tokenizer,
    append_eos_to_target: bool,
) -> Tuple[str, Optional[str], bool]:
    if example.messages is not None:
        messages = example.messages
        if messages and messages[-1]["role"].lower() == "assistant":
            prompt_messages = messages[:-1]
            rendered_with_chat_template = True
            try:
                prompt = tokenizer.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                rendered_with_chat_template = False
                prompt = _fallback_render_messages(prompt_messages)
                if prompt:
                    prompt = prompt + "\nassistant: "
            target = _append_eos(messages[-1]["content"], tokenizer, append_eos_to_target)
            return str(prompt), target, rendered_with_chat_template
        rendered_with_chat_template = True
        try:
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
        except Exception:
            rendered_with_chat_template = False
            text = _fallback_render_messages(messages)
        return str(text), None, rendered_with_chat_template

    if example.prompt is not None and example.target is not None:
        return example.prompt, _append_eos(example.target, tokenizer, append_eos_to_target), False
    if example.text is not None:
        return example.text, None, False
    raise ValueError("empty training example")


def _tokenize_example(
    example: TrainingExample,
    tokenizer,
    max_length: int,
    train_on_prompt: bool,
    append_eos_to_target: bool,
) -> Tuple[List[int], List[int]]:
    prompt_or_text, target, rendered_with_chat_template = _render_example(
        example,
        tokenizer,
        append_eos_to_target,
    )
    add_prompt_special_tokens = not rendered_with_chat_template
    if target is None:
        input_ids = tokenizer(
            prompt_or_text,
            add_special_tokens=add_prompt_special_tokens,
            truncation=False,
        ).input_ids
        input_ids = input_ids[:max_length]
        return input_ids, list(input_ids)

    prompt_ids = tokenizer(
        prompt_or_text,
        add_special_tokens=add_prompt_special_tokens,
        truncation=False,
    ).input_ids
    target_ids = tokenizer(target, add_special_tokens=False, truncation=False).input_ids
    input_ids = prompt_ids + target_ids
    input_ids = input_ids[:max_length]
    if train_on_prompt:
        return input_ids, list(input_ids)
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * len(prompt_ids) + target_ids
    labels = labels[:max_length]
    return input_ids, labels


def encode_batch(
    examples: List[TrainingExample],
    tokenizer,
    max_length: int,
    train_on_prompt: bool,
    append_eos_to_target: bool,
    min_target_tokens: int,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor | int]]:
    encoded: List[Tuple[List[int], List[int]]] = []
    skipped = 0
    for example in examples:
        input_ids, labels = _tokenize_example(
            example=example,
            tokenizer=tokenizer,
            max_length=max_length,
            train_on_prompt=train_on_prompt,
            append_eos_to_target=append_eos_to_target,
        )
        valid_next_token_labels = sum(1 for label in labels[1:] if label != -100)
        if not input_ids or valid_next_token_labels < min_target_tokens:
            skipped += 1
            continue
        encoded.append((input_ids, labels))
    if not encoded:
        return None

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer needs pad_token_id or eos_token_id")
    batch_max_len = max(len(input_ids) for input_ids, _ in encoded)
    input_rows: List[List[int]] = []
    attention_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    padding_side = getattr(tokenizer, "padding_side", "right")
    for input_ids, labels in encoded:
        pad_len = batch_max_len - len(input_ids)
        if padding_side == "left":
            input_rows.append([pad_token_id] * pad_len + input_ids)
            attention_rows.append([0] * pad_len + [1] * len(input_ids))
            label_rows.append([-100] * pad_len + labels)
        else:
            input_rows.append(input_ids + [pad_token_id] * pad_len)
            attention_rows.append([1] * len(input_ids) + [0] * pad_len)
            label_rows.append(labels + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
        "skipped": skipped,
        "encoded": len(encoded),
    }


def beta_at(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    ratio = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return start + ratio * (end - start)


def freeze_seed_noise(model: TrajectoryEnsembleForCausalLM) -> None:
    """Keep the seed perturbation active but untrained.

    This is the frozen-random-seed negative control: trajectories still diverge
    through the initialized seed bias, but only the aggregator learns. It
    separates "the perturbation must be learned" from "the perturbation must
    exist" (`--disable_seed_noise`). The context gate is frozen with the rest of
    the module so the realized seed scale stays fixed.
    """

    model.router_noise.requires_grad_(False)


def grad_summary(module: torch.nn.Module) -> Dict[str, object]:
    total_sq = 0.0
    tensors = 0
    tensors_with_grad = 0
    for parameter in module.parameters():
        tensors += 1
        if parameter.grad is None:
            continue
        tensors_with_grad += 1
        grad = parameter.grad.detach().float()
        total_sq += float(grad.pow(2).sum().cpu().item())
    return {
        "l2_norm": total_sq**0.5,
        "tensors": tensors,
        "tensors_with_grad": tensors_with_grad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first-router noise and final aggregator only.")
    parser.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-moe-16b-chat")
    parser.add_argument(
        "--train_file",
        default=None,
        help=(
            "Plain text or JSONL. JSONL supports text, prompt+completion, "
            "question+answer, instruction+output, and messages."
        ),
    )
    parser.add_argument("--out_dir", default="outputs/adapter")
    parser.add_argument("--num_trajectories", type=int, choices=[3, 5], default=3)
    parser.add_argument("--agg_dim", type=int, default=256)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--noise_scale_max", type=float, default=1.0)
    parser.add_argument("--noise_init_std", type=float, default=0.02)
    parser.add_argument("--seed_init_mode", choices=["gaussian", "orthogonal"], default="orthogonal")
    parser.add_argument(
        "--seed_inject_mode",
        choices=["first", "all"],
        default="first",
        help="'all' attaches an independent trainable seed to every MoE router layer.",
    )
    parser.add_argument("--train_router_mode", choices=["hard", "soft_all", "st_topk"], default="st_topk")
    parser.add_argument("--soft_temperature", type=float, default=1.0)
    parser.add_argument("--residual_scale_max", type=float, default=0.25)
    parser.add_argument("--include_null_aggregation_candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aggregator_value_mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--aggregator_relative_keys", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable_seed_noise", action="store_true")
    parser.add_argument("--freeze_seed_noise", action="store_true")
    parser.add_argument("--context_seed_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context_scale_max_delta", type=float, default=0.5)
    parser.add_argument("--kl_beta_start", type=float, default=1.0)
    parser.add_argument("--kl_beta_end", type=float, default=0.1)
    parser.add_argument("--kl_direction", choices=["base_to_fused", "symmetric"], default="base_to_fused")
    parser.add_argument("--kl_advantage_tau", type=float, default=0.0)
    parser.add_argument("--residual_l2_weight", type=float, default=1e-3)
    parser.add_argument("--trajectory_oracle_aux_weight", type=float, default=0.0)
    parser.add_argument("--trajectory_oracle_temperature", type=float, default=0.5)
    parser.add_argument("--trajectory_oracle_include_base", action="store_true")
    parser.add_argument("--aggregator_oracle_align_weight", type=float, default=0.0)
    parser.add_argument("--aggregator_oracle_align_temperature", type=float, default=0.5)
    parser.add_argument("--noise_diversity_weight", type=float, default=0.01)
    parser.add_argument("--noise_l2_weight", type=float, default=1e-4)
    parser.add_argument("--context_gate_l2_weight", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--train_on_prompt", action="store_true")
    parser.add_argument("--append_eos_to_target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_target_tokens", type=int, default=1)
    parser.add_argument("--log_trajectory_prediction_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_length < 2:
        raise ValueError("--max_length must be at least 2 for next-token training")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.log_every < 1:
        raise ValueError("--log_every must be positive")
    if args.min_target_tokens < 1:
        raise ValueError("--min_target_tokens must be positive")
    for name in (
        "residual_l2_weight",
        "kl_advantage_tau",
        "trajectory_oracle_aux_weight",
        "aggregator_oracle_align_weight",
        "noise_diversity_weight",
        "noise_l2_weight",
        "context_gate_l2_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.freeze_seed_noise and args.disable_seed_noise:
        raise ValueError("--freeze_seed_noise conflicts with --disable_seed_noise")
    if args.trajectory_oracle_temperature <= 0:
        raise ValueError("--trajectory_oracle_temperature must be positive")
    if args.aggregator_oracle_align_temperature <= 0:
        raise ValueError("--aggregator_oracle_align_temperature must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.train_file)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        tokenizer.padding_side = "right"

    model = TrajectoryEnsembleForCausalLM.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        num_trajectories=args.num_trajectories,
        agg_dim=args.agg_dim,
        noise_init_std=args.noise_init_std,
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
        seed_init_mode=args.seed_init_mode,
        seed_inject_mode=args.seed_inject_mode,
        aggregator_relative_keys=args.aggregator_relative_keys,
        local_files_only=args.local_files_only,
    )
    if args.freeze_seed_noise:
        freeze_seed_noise(model)
    model.train()
    trainable = [param for param in model.trainable_parameters() if param.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr)
    log_path = out_dir / "train_log.jsonl"

    stream = batches(examples, args.batch_size, args.seed)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        for step in range(args.steps):
            encoded = None
            attempts = 0
            max_attempts = max(8, len(examples) // max(args.batch_size, 1) + 2)
            while encoded is None:
                batch_examples = next(stream)
                encoded = encode_batch(
                    examples=batch_examples,
                    tokenizer=tokenizer,
                    max_length=args.max_length,
                    train_on_prompt=args.train_on_prompt,
                    append_eos_to_target=args.append_eos_to_target,
                    min_target_tokens=args.min_target_tokens,
                    device=model.input_device,
                )
                attempts += 1
                if encoded is None and attempts >= max_attempts:
                    raise ValueError(
                        "all sampled examples were skipped; lower --min_target_tokens "
                        "or increase --max_length"
                    )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = encoded["labels"]
            beta = beta_at(step, args.steps, args.kl_beta_start, args.kl_beta_end)
            log_this_step = step % args.log_every == 0

            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                kl_beta=beta,
                kl_direction=args.kl_direction,
                kl_advantage_tau=args.kl_advantage_tau,
                residual_l2_weight=args.residual_l2_weight,
                trajectory_oracle_aux_weight=args.trajectory_oracle_aux_weight,
                trajectory_oracle_temperature=args.trajectory_oracle_temperature,
                trajectory_oracle_include_base=args.trajectory_oracle_include_base,
                aggregator_oracle_align_weight=args.aggregator_oracle_align_weight,
                aggregator_oracle_align_temperature=args.aggregator_oracle_align_temperature,
                return_route_stats=log_this_step,
                return_aggregator_stats=log_this_step,
                return_trajectory_stats=log_this_step,
                return_trajectory_prediction_stats=(
                    log_this_step and args.log_trajectory_prediction_stats
                ),
            )
            reg_loss, reg_metrics = model.adapter_regularization(
                noise_diversity_weight=args.noise_diversity_weight,
                noise_l2_weight=args.noise_l2_weight,
                context_gate_l2_weight=args.context_gate_l2_weight,
            )
            total_loss = output.loss + reg_loss.to(output.loss.device)
            total_loss.backward()
            grad_record = None
            if log_this_step:
                grad_record = {
                    "router_noise": grad_summary(model.router_noise),
                    "aggregator": grad_summary(model.aggregator),
                }
            optimizer.step()

            if log_this_step:
                loss_record = dict(output.loss_components)
                loss_record.update(reg_metrics)
                loss_record["total_with_reg"] = float(total_loss.detach().float().cpu().item())
                record = {
                    "step": step,
                    "batch": {
                        "encoded": int(encoded["encoded"]),
                        "skipped": int(encoded["skipped"]),
                        "tokens": int(attention_mask.sum().detach().cpu().item()),
                        "target_tokens": int(labels[:, 1:].ne(-100).sum().detach().cpu().item()),
                    },
                    "loss": loss_record,
                    "aggregator": output.aggregator_stats,
                    "trajectory": output.trajectory_stats,
                    "trajectory_prediction": output.trajectory_prediction_stats,
                    "grad": grad_record,
                    "route_stats": output.route_stats,
                    "path_stats": output.path_stats,
                }
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_handle.flush()
                print(json.dumps(record["loss"], ensure_ascii=False))

    model.save_adapter(str(out_dir / "trajectory_adapter.pt"))
    adapter_config = model.adapter_config()
    adapter_config["freeze_seed_noise"] = bool(args.freeze_seed_noise)
    with open(out_dir / "adapter_config.json", "w", encoding="utf-8") as handle:
        json.dump(adapter_config, handle, ensure_ascii=False, indent=2)
    print(f"[saved] {out_dir / 'trajectory_adapter.pt'}")


if __name__ == "__main__":
    main()
