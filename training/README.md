# AegisDesk Training Guide

This directory contains the Round 2 training path for the expanded AegisDesk benchmark.

## Benchmark Story

- train on `9` canonical fixtures
- evaluate on `27` judged fixtures
- keep `18` held-out `generalization` fixtures out of the training path
- keep `3` `showcase` fixtures outside the main score-report story

The source of truth for fixture splits is:
- [training/support_rl_manifest.json](support_rl_manifest.json)

## Main Files

- `train_grpo_aegisdesk.py`: TRL/OpenEnv GRPO starter
- `self_improve.py`: baseline -> harvest -> DPO pairs -> train -> re-evaluate
- `trajectory_harvester.py`: collects trajectories from benchmark runs
- `dpo_pair_generator.py`: builds `(chosen, rejected)` preference pairs
- `adaptive_scheduler.py`: curriculum weighting helpers
- `AegisDesk_Training.ipynb`: notebook path for HF Jobs / Colab
- `HF_JOBS_RUNBOOK.md`: stage-by-stage cloud GPU plan with hardware, timeouts, and artifact gates
- `strongest_submission.py`: numbered 10-step execution path
- `check_training_readiness.py`: corpus, manifest, dependency, and endpoint readiness doctor

## Data Pipeline

Use:

```bash
python scripts/fetch_real_datasets.py
```

This fetch/build path now targets:
- Bitext
- ABCD
- Sierra tau-bench / tau2-bench few-shot data
- Schema-Guided Dialogue
- HelpSteer2
- optional DialogStudio / MultiWOZ samples

Derived outputs:
- `training/raw/*`
- `training/data/support_sft.jsonl`
- `training/data/support_pref.jsonl`
- `training/data/dataset_build_report.json`
- `training/support_rl_manifest.json`

Verified build output in this workspace:
- `support_sft.jsonl`: `15,124` rows
- `support_pref.jsonl`: `7,119` rows
- `dataset_build_report.json`: source-level counts and target checks

## Numbered Execution Path

To see the 10-step strongest-submission workflow:

```bash
python training/strongest_submission.py --list
```

To validate local readiness before GPU work:

```bash
python training/check_training_readiness.py \
  --env-url https://i4mgr00t-meta.hf.space \
  --output training/readiness_report.json
```

## Recommended Stack

1. Unsloth QLoRA SFT on `support_sft.jsonl`
2. Unsloth DPO or ORPO on `support_pref.jsonl`
3. TRL `GRPOTrainer` on the canonical 9-fixture pack

Concrete entrypoints:
- `train_unsloth_sft.py`
- `train_unsloth_dpo.py`
- `train_grpo_aegisdesk.py`

Default model target:
- `Qwen/Qwen3-8B`

Fallback:
- `Qwen/Qwen3-4B`

## Run GRPO

Champion path:

```bash
accelerate launch training/train_grpo_aegisdesk.py \
  --phase champion \
  --rl-manifest training/support_rl_manifest.json \
  --env-url https://i4mgr00t-meta.hf.space \
  --model Qwen/Qwen3-8B \
  --report-to trackio \
  --run-name aegisdesk-grpo-champion
```

Stabilization path:

```bash
accelerate launch training/train_grpo_aegisdesk.py \
  --phase stabilize \
  --rl-manifest training/support_rl_manifest.json \
  --env-url https://i4mgr00t-meta.hf.space \
  --model Qwen/Qwen3-8B \
  --report-to trackio \
  --run-name aegisdesk-grpo-stabilize
```

## Run SFT

```bash
python training/train_unsloth_sft.py \
  --dataset training/data/support_sft.jsonl \
  --output training/outputs/sft-qwen3-8b \
  --model Qwen/Qwen3-8B \
  --report-to trackio \
  --run-name aegisdesk-sft
```

## Run DPO

```bash
python training/train_unsloth_dpo.py \
  --dataset training/data/support_pref.jsonl \
  --output training/outputs/dpo-qwen3-8b \
  --model Qwen/Qwen3-8B \
  --report-to trackio \
  --run-name aegisdesk-dpo
```

## Run The Self-Improvement Loop

Dry-run:

```bash
python training/self_improve.py --rounds 1 --dry-run
```

Real loop:

```bash
python training/self_improve.py \
  --rounds 1 \
  --seeds 3 \
  --env-url https://i4mgr00t-meta.hf.space \
  --results-path training/benchmark_results.json
```

This now benchmarks the full `27` judged fixtures while keeping harvesting/training on the canonical `9`.

## Hardware Plan

Preferred:
- HF Jobs GPU
- L4 or A10G class hardware
- `Qwen/Qwen3-8B`
- SFT timeout target: `3h`
- DPO timeout target: `2h`
- GRPO timeout target: `4h`

Fallback:
- `t4-medium`
- `Qwen/Qwen3-4B`

Suggested run order:
1. `Qwen/Qwen3-8B` SFT on `L4` or `A10G`
2. `Qwen/Qwen3-8B` DPO on the same class of GPU
3. GRPO stabilize run on canonical `9`
4. GRPO champion run on canonical `9` plus private curriculum variants
5. Baseline and champion evaluation across `27` judged fixtures with `3` seeds

## Current Truth

The repo now contains the correct training structure and evaluation split, but it does **not** yet contain a real champion run. The remaining work is to produce and check in:
- `training/benchmark_results.json`
- reward curve PNG
- loss curve PNG
- per-track delta figure
