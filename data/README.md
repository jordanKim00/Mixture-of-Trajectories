# Data Layout

```
data/
  eval/    RoE-matched evaluation suite (scripts/prepare_eval_data.py)
  train/   staged adapter-training mixtures (scripts/prepare_train_data.py)
```

## Evaluation suite (RoE setting, arXiv:2509.17238)

| group | tasks | protocol |
|---|---|---|
| math | gsm8k, svamp, addsub, singleeq, multiarith | greedy generation, last-number extraction |
| commonsense | arc_easy, arc_challenge, openbookqa, siqa, hellaswag | per-choice log-likelihood |
| code | humaneval, humanevalplus | completion samples for offline evalplus |

SIQA and HellaSwag use validation splits (hidden test labels). AddSub and
SingleEq come from `allenai/lila` (`addsub`, `singleq` configs). These twelve
tasks are the contamination-protected set: nothing here may appear in any
training stage.

## Training mixtures

The adapter trains ~3M parameters, so the budget question is coverage and
difficulty shape, not token count. Defaults are 20k examples per stage; scale
with `--stage1_size/--stage3_size` only after the POC shows the evidence-chain
metrics moving.

Stage 1 — continuation warm-up (`stage1_continuation.jsonl`): plain text so the
aggregator and seeds adapt to the frozen hidden distribution under the KL rail.

| bucket | ratio | source |
|---|---|---|
| web/educational | 40% | HuggingFaceFW/fineweb-edu (sample-10BT) |
| broad general | 20% | DKYoon/SlimPajama-6B |
| math/science text | 15% | open-web-math/open-web-math |
| code | 15% | codeparrot/codeparrot-clean (the-stack is gated) |
| encyclopedic | 10% | wikimedia/wikipedia 20231101.en |

Stage 2 — hard-prefix mining (`stage2_hard_prefix.jsonl`): GPU step, run after
the eval suite and stage 1 exist:

```bash
python scripts/mine_hard_prefixes.py --pool data/train/stage1_continuation.jsonl
```

Scores each example with the wrapper on four components (`base_ce`,
`gold_nll_std`, `route_divergence`, `router_entropy`), then selects with three
guards learned from the first mining run, where raw base-CE selected spam:

- junk filter (`scripts/data_quality.py`): spam keywords, keyword stuffing
  (near-zero function-word ratio), repetitive or symbol-heavy text — "hard
  because garbage" is not "hard reasoning";
- base-CE percentile cap (default p95): drops un-learnable noise;
- per-component z-normalization before weighting (default `1.0*ce + 2.0*dis +
  1.0*div + 0.25*ent`), because raw base-CE is ~50x larger than the
  disagreement terms and otherwise dominates.

`--rerank_from mined.jsonl` reapplies selection on stored metadata without the
GPU, for tuning weights/filters after an expensive scoring run.

Stage 3 — SFT specialization (`stage3_sft.jsonl`): target-only labels
(prompt masked with -100) via the existing train_adapter.py loaders.

| bucket | ratio | source |
|---|---|---|
| general instruction | 25% | allenai/tulu-3-sft-mixture |
| math reasoning | 20% | AI-MO/NuminaMath-CoT |
| code instruction | 20% | m-a-p/CodeFeedback-Filtered-Instruction |
| commonsense (eval-disjoint) | 12% | tau/commonsense_qa, ChilleD/StrategyQA, allenai/winogrande |
| science | 10% | allenai/sciq |
| multi-hop | 13% | hotpotqa/hotpot_qa (distractor) |

PIQA was replaced by StrategyQA (its hub loader script is broken on
datasets>=3) and FinQA is omitted until a reliably hosted mirror is chosen;
both substitutions keep the bucket intent (implicit commonsense reasoning,
numeric reasoning lives in NuminaMath).

## Task-family holdout (strict)

The twelve eval task families are fully held out from ALL training stages:
no test sets, no validation sets, **no train splits, and no rephrased
derivatives** inside open mixtures. Enforcement is two-layer:

1. provenance filtering at load time (`EVAL_FAMILY_MARKERS` in
   `prepare_train_data.py`): rows whose `source`/`dataset` tags reference
   gsm8k/svamp/addsub/singleeq/multiarith/arc/openbookqa/siqa/hellaswag/
   humaneval (and mbpp, as a code-eval relative) are dropped — this catches
   e.g. the GSM8K-train-derived subset inside NuminaMath-CoT and math/QA
   subsets inside Tulu-3;
2. the 8-gram decontamination filter against `data/eval/` catches verbatim
   and near-verbatim leaks that carry no provenance tag.

This keeps the comparison with training-free RoE clean: the adapter never
sees any data from the evaluated task families, so gains cannot come from
task-distribution memorization.

## Decontamination

`scripts/prepare_train_data.py` builds a normalized word 8-gram index over
every record in `data/eval/` (questions, choices, prompts, long answers) and
drops any training example sharing a single shingle. Always regenerate training
data after changing the eval suite. This catches verbatim and near-verbatim
leaks; if a stage-3 source is later suspected of paraphrase-level leakage, add
MinHash/embedding dedup before scaling up.
