# AegisDesk Round 2 Slide Deck

## Slide 1 — Title

**AegisDesk**

Deterministic OpenEnv benchmark for B2B SaaS support operations

- public env: `support_ops_env`
- live Space: `https://i4mgr00t-meta.hf.space`
- thesis: train on realistic enterprise workflows, not toy agent tasks

## Slide 2 — Problem

Most agent benchmarks either:
- score free-form text with fuzzy judges
- avoid real operational constraints
- or fail to separate memorization from transfer

AegisDesk asks:

> Can an agent investigate correctly, follow policy, and resolve enterprise support cases safely under deterministic grading?

## Slide 3 — Benchmark Shape

- `30` surfaced fixtures total
- `27` judged fixtures
- `3` showcase fixtures

Judged benchmark:
- `3` core canonical fixtures
- `6` Round 2 canonical fixtures
- `18` held-out judged variants

Story:
- train on `9` canonical fixtures
- test on `18` held-out variants

## Slide 4 — Why It Is Hard

Agents must:
- select the right ticket from distractors
- inspect the right records before acting
- obey escalation rules
- handle customer follow-up messages
- respect world state and policy windows
- complete long-horizon phases in order

## Slide 5 — Deterministic Reward

```text
reward =
  progress_delta
  + behavior_adjustment
  + phase_bonus
  + (qa_score × 0.1 × 0.15)
```

Key point:
- no LLM judge
- no fuzzy semantic evaluator
- rubric and reward stay auditable

## Slide 6 — Public Surfaces

- `/tasks`
- `/benchmark-card`
- `/console`
- `/trajectory-viewer`
- `/trajectory-report`

Current verified state:
- `57` passing tests
- `openenv validate` passes
- oracle coverage for every surfaced fixture

## Slide 7 — Training Path

Three-stage plan:

1. Unsloth QLoRA SFT on `support_sft.jsonl`
2. Unsloth DPO/ORPO on `support_pref.jsonl`
3. TRL GRPO on the canonical `9` training fixtures

Guardrail:
- held-out `generalization` fixtures are excluded from training

## Slide 8 — Real Data Sources

- Bitext
- ABCD
- Sierra tau-bench / tau2-bench
- Schema-Guided Dialogue
- HelpSteer2
- optional DialogStudio / MultiWOZ

Repo path:
- `python scripts/fetch_real_datasets.py`

Current built corpora:
- `support_sft.jsonl`: `15,124` rows
- `support_pref.jsonl`: `7,119` rows

## Slide 9 — What Is Proven Now

- strong deterministic environment
- stronger benchmark taxonomy
- held-out transfer story implemented
- truthful public catalog and benchmark card
- stronger training/data/reporting structure

## Slide 10 — Remaining Gap

Still needed before the final top-finish claim:
- real GPU-backed training run
- `training/benchmark_results.json`
- reward curve PNG
- loss curve PNG
- positive delta on both canonical and held-out judged fixtures

Current truthful claim:

> AegisDesk is benchmark-strong and submission-ready in structure.

Not yet:

> AegisDesk already proves the strongest RL model without further training evidence.
