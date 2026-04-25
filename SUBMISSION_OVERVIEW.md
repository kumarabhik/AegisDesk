# Submission Overview

`AegisDesk` is a deterministic OpenEnv benchmark for B2B SaaS support operations. It asks a concrete and useful question:

> Can an agent investigate the right evidence, follow policy, avoid unsafe shortcuts, communicate clearly, and finish a support case correctly when the workflow looks like an enterprise support console instead of a toy task?

## What Judges See

- `30` surfaced fixtures
- `27` judged fixtures
- `3` showcase fixtures
- deterministic rubric grading
- dense reward shaping
- a live Space
- an interactive console
- an oracle trajectory viewer

## Core Benchmark Story

The main story is not just “9 tasks.” It is now:

- train on `9` canonical enterprise fixtures
- test on `18` held-out judged variants
- show whether improvement transfers to unseen fixture variants

This makes AegisDesk stronger than a single-pack benchmark because it can now separate:
- memorizing canonical fixtures
- improving on structurally related but unseen variants

## Why The Environment Is Novel

The benchmark combines:
- structured operational actions
- record inspection and evidence gathering
- world-state and policy-window reasoning
- multi-agent customer follow-up behavior
- long-horizon phase ordering
- deterministic security and escalation rules

This is a more realistic RL training target than a static instruction benchmark and more auditable than a free-form LLM judge environment.

## What Is Already Strong

- deterministic grading only
- OpenEnv-compatible server
- held-out judged variants are now first-class via `fixture_id`
- `/tasks` and `/benchmark-card` now tell the truth about the catalog
- oracle reports cover every surfaced fixture
- local suite passes with `57` tests
- `openenv validate` passes
- real external-data corpora are now built in-repo:
  - `support_sft.jsonl` with `15,124` rows
  - `support_pref.jsonl` with `7,119` rows

## What Still Needs To Be Shown

The remaining top-finish gap is evidence:
- real SFT / preference / GRPO training run
- real trained-vs-baseline benchmark deltas
- real reward and loss plots
- checked-in `training/benchmark_results.json`

That is why the correct current claim is:

> AegisDesk is now a strong benchmark and a strong submission framework.

Not yet:

> AegisDesk already proves the best RL model in the field.
