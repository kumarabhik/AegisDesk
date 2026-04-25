# AegisDesk HF Jobs Runbook

This runbook turns the training plan into a concrete sequence for Hugging Face Jobs.

If you want the numbered local execution map for these stages, use:

```bash
python training/strongest_submission.py --list
python training/check_training_readiness.py --env-url https://i4mgr00t-meta.hf.space
```

## Goal

Train a champion `Qwen/Qwen3-8B` adapter stack that:
- trains on the canonical `9` fixtures plus private curriculum variants where allowed
- never trains on the `18` held-out `generalization` fixtures
- improves on both canonical and held-out judged means across the full `27`-fixture benchmark

## Inputs

- SFT corpus: `training/data/support_sft.jsonl`
- Preference corpus: `training/data/support_pref.jsonl`
- RL split: `training/support_rl_manifest.json`
- Environment URL: `https://i4mgr00t-meta.hf.space`

Current verified corpus sizes:
- `support_sft.jsonl`: `15,124` rows
- `support_pref.jsonl`: `7,119` rows

## Stage Plan

### Stage 0: Data verification

Run before GPU work:

```bash
python scripts/fetch_real_datasets.py --verify-only
python scripts/fetch_real_datasets.py
python -m pytest -q
openenv validate
python oracle_demo.py --pack benchmark --seed 11
```

Expected outputs:
- `training/data/support_sft.jsonl`
- `training/data/support_pref.jsonl`
- `training/data/dataset_build_report.json`
- `training/support_rl_manifest.json`

### Stage 1: SFT smoke

Purpose:
- confirm tokenizer/template path
- confirm Hub push works
- catch format issues before longer runs

Recommended hardware:
- `t4-medium` or `L4`

Timeout:
- `90m`

Command:

```bash
python training/train_unsloth_sft.py \
  --dataset training/data/support_sft.jsonl \
  --output training/outputs/sft-smoke \
  --model Qwen/Qwen3-8B \
  --epochs 0.1 \
  --report-to trackio \
  --run-name aegisdesk-sft-smoke
```

Promotion rule:
- proceed only if loss decreases and checkpoint save succeeds

### Stage 2: SFT champion

Purpose:
- produce the main instruction-tuned adapter

Recommended hardware:
- `L4` or `A10G`

Timeout:
- `3h`

Command:

```bash
python training/train_unsloth_sft.py \
  --dataset training/data/support_sft.jsonl \
  --output training/outputs/sft-qwen3-8b \
  --model Qwen/Qwen3-8B \
  --epochs 1.5 \
  --report-to trackio \
  --run-name aegisdesk-sft-champion
```

Expected artifact:
- SFT adapter checkpoint pushed to the Hub or saved for the next stage

### Stage 3: DPO champion

Purpose:
- improve response quality and preference alignment before RL

Recommended hardware:
- `L4` or `A10G`

Timeout:
- `2h`

Command:

```bash
python training/train_unsloth_dpo.py \
  --dataset training/data/support_pref.jsonl \
  --output training/outputs/dpo-qwen3-8b \
  --model Qwen/Qwen3-8B \
  --epochs 1.0 \
  --report-to trackio \
  --run-name aegisdesk-dpo-champion
```

Promotion rule:
- proceed only if preference loss is stable and samples remain policy-compliant

### Stage 4: GRPO stabilize

Purpose:
- validate online RL plumbing on the canonical `9` only

Recommended hardware:
- `A10G`

Timeout:
- `2h`

Command:

```bash
accelerate launch training/train_grpo_aegisdesk.py \
  --phase stabilize \
  --rl-manifest training/support_rl_manifest.json \
  --env-url https://i4mgr00t-meta.hf.space \
  --model Qwen/Qwen3-8B \
  --report-to trackio \
  --run-name aegisdesk-grpo-stabilize
```

Success signal:
- reward improves without a spike in invalid or forbidden actions

### Stage 5: GRPO champion

Purpose:
- run the best RL pass on canonical `9` plus private curriculum variants

Recommended hardware:
- `A10G`

Timeout:
- `4h`

Command:

```bash
accelerate launch training/train_grpo_aegisdesk.py \
  --phase champion \
  --rl-manifest training/support_rl_manifest.json \
  --env-url https://i4mgr00t-meta.hf.space \
  --model Qwen/Qwen3-8B \
  --report-to trackio \
  --run-name aegisdesk-grpo-champion
```

Hard guardrail:
- the `18` held-out `generalization` fixtures must remain excluded

### Stage 6: Baseline and champion evaluation

Purpose:
- generate the submission evidence package

Command:

```bash
python training/self_improve.py \
  --rounds 1 \
  --seeds 3 \
  --env-url https://i4mgr00t-meta.hf.space \
  --results-path training/benchmark_results.json
python training/plot_benchmark_results.py \
  --results training/benchmark_results.json
```

Expected outputs:
- `training/benchmark_results.json`
- `training/per_track_delta.png`
- `training/canonical_vs_held_out_summary.md`

## Hardware Defaults

Primary champion path:
- model: `Qwen/Qwen3-8B`
- GPU class: `L4` or `A10G`

Fallback path:
- model: `Qwen/Qwen3-4B`
- GPU class: `t4-medium`

Use the fallback only if quota or time blocks the `8B` run.

## Artifact Checklist

Check in or upload links for:
- SFT adapter
- DPO adapter
- GRPO champion adapter
- `training/benchmark_results.json`
- reward curve PNG
- loss curve PNG
- `training/per_track_delta.png`
- `training/canonical_vs_held_out_summary.md`

## Promotion Rules

Do not call the model a winner unless all of these are true:
- canonical mean improves
- held-out mean improves
- security slice does not regress badly
- invalid action rate does not materially worsen
- forbidden action hit rate does not materially worsen

If only canonical improves:
- frame the result as promising but not yet generalized

If canonical and held-out both improve:
- frame the result as real transfer to unseen support variants
