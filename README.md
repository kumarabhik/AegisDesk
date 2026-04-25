---
title: AegisDesk
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
base_path: /home
fullWidth: true
header: mini
short_description: OpenEnv benchmark for SaaS support operations
suggested_hardware: cpu-basic
pinned: true
tags:
  - openenv
  - benchmark
  - agents
  - fastapi
  - rl
---

# AegisDesk

`AegisDesk` is the public benchmark name for `support_ops_env`, an OpenEnv environment for B2B SaaS support operations. Agents must choose the right ticket, inspect the right records, follow policy, avoid unsafe shortcuts, communicate clearly, and finish with a deterministic resolution.

The benchmark now exposes:
- `30` surfaced fixtures
- `27` judged fixtures
- `3` showcase fixtures

The main training/evaluation story is:
- train on `9` canonical enterprise fixtures
- test on `18` held-out judged variants
- keep `3` showcase fixtures outside the main score-report pack

The YAML fixture files under `server/task_data/` are episode specifications for deterministic evaluation. They are not the whole training corpus. The real training data now lives in the derived corpora under `training/data/`.

## Judge Quick Start

- Live Space: `https://i4mgr00t-meta.hf.space`
- Hugging Face Space repo: `https://huggingface.co/spaces/I4mGr00T/Meta`
- GitHub repo: `https://github.com/kumarabhik/AegisDesk`
- Landing page: `https://i4mgr00t-meta.hf.space/home`
- Interactive console: `https://i4mgr00t-meta.hf.space/console`
- Oracle trajectory viewer: `https://i4mgr00t-meta.hf.space/trajectory-viewer`
- Benchmark card: `https://i4mgr00t-meta.hf.space/benchmark-card`
- Design doc: [design_doc.md](design_doc.md)
- Round 1 design archive: [ROUND1_DESIGN_DOC.md](ROUND1_DESIGN_DOC.md)
- Results log: [RESULTS.md](RESULTS.md)
- Submission overview: [SUBMISSION_OVERVIEW.md](SUBMISSION_OVERVIEW.md)
- Slide deck: [ROUND2_SLIDE_DECK.md](ROUND2_SLIDE_DECK.md)
- Training guide: [training/README.md](training/README.md)
- HF Jobs runbook: [training/HF_JOBS_RUNBOOK.md](training/HF_JOBS_RUNBOOK.md)

## Current Readiness

Verified locally in this workspace:
- `python -m pytest -q` -> `57 passed`
- `openenv validate` -> `[OK] meta: Ready for multi-mode deployment`
- `/tasks` returns the truthful surfaced catalog with `fixture_id`, `task_id`, `track`, `judged`, and `oracle_available`
- `/trajectory-report` works for every surfaced fixture
- `oracle_demo.py` supports `core`, `v2`, `benchmark`, `generalization`, `showcase`, and `all`
- `python scripts/fetch_real_datasets.py` builds:
  - `training/data/support_sft.jsonl` with `15,124` rows
  - `training/data/support_pref.jsonl` with `7,119` rows
  - `training/support_rl_manifest.json`

Main remaining gap:
- the benchmark is strong, but **real checked-in training evidence is still pending**. The repo now has the data path, manifest, and reporting path, but not a finished GPU-backed champion run.

## Benchmark Packs

| Pack | Count | Role |
|---|---:|---|
| `core` | 3 | canonical baseline fixtures |
| `v2` | 6 | canonical Round 2 fixtures |
| `generalization` | 18 | held-out judged variants |
| `showcase` | 3 | demo-only showcase fixtures |
| `benchmark` | 27 | official judged benchmark = `core + v2 + generalization` |
| `all` | 30 | full surfaced catalog = `benchmark + showcase` |

## Surfaced Fixture Catalog

| Fixture ID | Track | Task Family | Judged |
|---|---|---|---:|
| `billing_seat_adjustment` | `core` | `billing_seat_adjustment` | yes |
| `login_incident_triage` | `core` | `login_incident_triage` | yes |
| `suspicious_admin_request` | `core` | `suspicious_admin_request` | yes |
| `customer_escalation_chain` | `v2` | `customer_escalation_chain` | yes |
| `multi_tier_billing_dispute` | `v2` | `multi_tier_billing_dispute` | yes |
| `data_breach_response_lifecycle` | `v2` | `data_breach_response_lifecycle` | yes |
| `contract_renewal_negotiation` | `v2` | `contract_renewal_negotiation` | yes |
| `service_reinstatement_review` | `v2` | `service_reinstatement_review` | yes |
| `api_partner_access_audit` | `v2` | `api_partner_access_audit` | yes |
| `billing_seat_adjustment_v1` | `generalization` | `billing_seat_adjustment` | yes |
| `billing_seat_adjustment_v2` | `generalization` | `billing_seat_adjustment` | yes |
| `login_incident_triage_v1` | `generalization` | `login_incident_triage` | yes |
| `login_incident_triage_v2` | `generalization` | `login_incident_triage` | yes |
| `suspicious_admin_request_v1` | `generalization` | `suspicious_admin_request` | yes |
| `suspicious_admin_request_v2` | `generalization` | `suspicious_admin_request` | yes |
| `customer_escalation_chain_v1` | `generalization` | `customer_escalation_chain` | yes |
| `customer_escalation_chain_v2` | `generalization` | `customer_escalation_chain` | yes |
| `multi_tier_billing_dispute_v1` | `generalization` | `multi_tier_billing_dispute` | yes |
| `multi_tier_billing_dispute_v2` | `generalization` | `multi_tier_billing_dispute` | yes |
| `data_breach_response_lifecycle_v1` | `generalization` | `data_breach_response_lifecycle` | yes |
| `data_breach_response_lifecycle_v2` | `generalization` | `data_breach_response_lifecycle` | yes |
| `contract_renewal_negotiation_v1` | `generalization` | `contract_renewal_negotiation` | yes |
| `contract_renewal_negotiation_v2` | `generalization` | `contract_renewal_negotiation` | yes |
| `service_reinstatement_review_v1` | `generalization` | `service_reinstatement_review` | yes |
| `service_reinstatement_review_v2` | `generalization` | `service_reinstatement_review` | yes |
| `api_partner_access_audit_v1` | `generalization` | `api_partner_access_audit` | yes |
| `api_partner_access_audit_v2` | `generalization` | `api_partner_access_audit` | yes |
| `admin_role_transfer_verification` | `showcase` | `admin_role_transfer_verification` | no |
| `api_rate_limit_escalation` | `showcase` | `api_rate_limit_escalation` | no |
| `tax_exemption_credit_review` | `showcase` | `tax_exemption_credit_review` | no |

## Why This Benchmark Is Interesting

AegisDesk is not a browser toy and not a pure text-only judge benchmark. It targets a real operational capability gap:
- selecting the correct case from distractors
- inspecting the right records before mutating anything
- following escalation rules instead of unsafe direct fulfillment
- handling policy windows and world state
- reacting to injected customer follow-ups
- completing long-horizon workflows in the right order

That maps naturally to the Round 2 themes:
- Multi-Agent Interactions
- Long-Horizon Planning and Instruction Following
- World Modeling
- Self-Improving Agent Systems

## Contract Highlights

### Identity model

- `task_id` = task family
- `fixture_id` = exact episode identity
- canonical fixtures use `fixture_id == task_id`
- variants use `fixture_id = <task_id>_v<n>`

### Public routes

- `/tasks` -> surfaced fixture catalog
- `/benchmark-card` -> machine-readable benchmark summary
- `/console` -> manual operator UI
- `/trajectory-viewer` -> oracle viewer
- `/trajectory-report?fixture_id=...` -> scored oracle trace

### Reward model

```text
reward =
  progress_delta
  + behavior_adjustment
  + phase_bonus
  + (qa_score × 0.1 × 0.15)
```

There are no LLM judges and no fuzzy grading.

## Training Story

The intended top-finish training path is:

1. Unsloth QLoRA SFT on `support_sft.jsonl`
2. Unsloth DPO or ORPO on `support_pref.jsonl`
3. TRL `GRPOTrainer` on the canonical `9` training fixtures
4. Evaluate on the full `27` judged fixtures

Repo assets:
- RL manifest: [training/support_rl_manifest.json](training/support_rl_manifest.json)
- GRPO trainer: [training/train_grpo_aegisdesk.py](training/train_grpo_aegisdesk.py)
- Self-improve loop: [training/self_improve.py](training/self_improve.py)
- Trajectory harvester: [training/trajectory_harvester.py](training/trajectory_harvester.py)
- Dataset fetch/build script: [scripts/fetch_real_datasets.py](scripts/fetch_real_datasets.py)
- Notebook: [training/AegisDesk_Training.ipynb](training/AegisDesk_Training.ipynb)

### Real dataset sources wired into the repo

- Bitext customer support dataset
- ABCD
- Sierra tau-bench / tau2-bench few-shot trajectories
- Schema-Guided Dialogue
- HelpSteer2
- optional DialogStudio / MultiWOZ samples

Current derived corpus sizes from `python scripts/fetch_real_datasets.py`:
- `support_sft.jsonl`: `15,124` rows
  - Bitext: `5,776`
  - ABCD: `5,000`
  - tau-bench / tau2-bench: `69`
  - SGD: `4,000`
  - AegisDesk oracle traces: included in the combined corpus build
- `support_pref.jsonl`: `7,119` rows
  - HelpSteer2: `7,118`
  - AegisDesk harvested DPO pairs: appended when available

### Training discipline

- the `18` judged `generalization` fixtures are excluded from SFT, preference tuning, and GRPO
- the `9` canonical fixtures are the main training pack
- private non-surfaced variants may be used as curriculum fixtures

## Local Usage

Install:

```bash
pip install -e .
```

Serve locally:

```bash
python -m server.app
```

Useful routes:
- `http://127.0.0.1:7860/home`
- `http://127.0.0.1:7860/console`
- `http://127.0.0.1:7860/trajectory-viewer`
- `http://127.0.0.1:7860/benchmark-card`

## Validation

```bash
python -m pytest -q
openenv validate
python verify_space.py --base-url http://127.0.0.1:7860
python submission_audit.py --space-url https://i4mgr00t-meta.hf.space
```

Oracle packs:

```bash
python oracle_demo.py --pack benchmark
python oracle_demo.py --pack generalization
python oracle_demo.py --pack showcase
python oracle_demo.py --pack all
```

## Evidence Path

The repo now contains the structure for a strong submission, but the following are still pending a real run:
- `training/benchmark_results.json`
- reward curve PNG
- loss curve PNG
- per-track delta figure
- champion-vs-baseline narrative in the slide deck

This is the main remaining blocker between “strong benchmark” and “top-finish-safe submission.”

## Repo Pointers

- Server: [server/app.py](server/app.py)
- Runtime: [server/environment.py](server/environment.py)
- Reward logic: [server/reward.py](server/reward.py)
- Fixture catalog: [server/fixtures.py](server/fixtures.py)
- Oracle tooling: [oracle_tools.py](oracle_tools.py)
- Source-of-truth doc: [design_doc.md](design_doc.md)
