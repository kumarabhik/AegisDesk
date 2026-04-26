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

# AegisDesk 🛡️

**A deterministic RL benchmark for B2B SaaS support operations.**

AegisDesk tests whether an agent can do what a real enterprise support operator does: pick the right ticket from a noisy inbox, inspect the right records before touching anything, follow policy (escalation rules, discount limits, security gates), coordinate across customers and internal teams, and close the case with a verifiable, auditable resolution — under a step budget.

No LLM judges. No fuzzy grading. Same fixture + same actions = same score, always.

## Quick Links

| | |
|---|---|
| Live Space | https://i4mgr00t-meta.hf.space |
| Interactive console | https://i4mgr00t-meta.hf.space/console |
| Oracle trajectory viewer | https://i4mgr00t-meta.hf.space/trajectory-viewer |
| Benchmark card | https://i4mgr00t-meta.hf.space/benchmark-card |
| GitHub | https://github.com/kumarabhik/AegisDesk |

## Benchmark At a Glance

| | |
|---|---|
| Surfaced fixtures | **30** |
| Judged fixtures | **27** |
| Training fixtures | **9** (canonical only) |
| Held-out generalization fixtures | **18** (never used in training) |
| Showcase fixtures | **3** (demo only) |
| Grading | deterministic rubric — no LLM judge |
| Baseline score | **0.27** (Qwen2.5-72B on 3 core tasks, zero-shot) |

## The Core Benchmark Story

The main claim is not just "9 tasks." It is:

> Train on **9 canonical** enterprise support fixtures, evaluate on **18 held-out judged variants**, and show whether improvement transfers to unseen fixture variants.

This separates **memorization of canonical fixtures** from **genuine generalization** to structurally similar but unseen episodes — a stronger RL evaluation story than a single-pack benchmark.

## Task Mix

| Task | Track | Theme | Max Steps |
|---|---|---|---:|
| `billing_seat_adjustment` | core | Baseline | 12 |
| `login_incident_triage` | core | World Modeling | 12 |
| `suspicious_admin_request` | core | Baseline | 12 |
| `customer_escalation_chain` | v2 | Multi-Agent | 15 |
| `multi_tier_billing_dispute` | v2 | Multi-Agent | 15 |
| `data_breach_response_lifecycle` | v2 | Long-Horizon | 30 |
| `contract_renewal_negotiation` | v2 | Long-Horizon | 25 |
| `service_reinstatement_review` | v2 | World Modeling | 12 |
| `api_partner_access_audit` | v2 | World Modeling | 15 |

Each task has 2 held-out judged generalization variants (`_v1`, `_v2`) that are excluded from training.

## Why This Is Hard

A random or naive agent scores near **0.0**. The zero-shot frontier model baseline is **0.27**. The difficulty comes from:

- **Distractor inbox** — multiple tickets are visible; picking the wrong one wastes steps
- **Mandatory evidence inspection** — rubric items require specific records to be read before mutating state
- **Policy windows** — discounts, escalation triggers, and security gates are fixture-dependent
- **Phase ordering** — long-horizon tasks (breach response, contract renewal) penalize out-of-order actions
- **Multi-agent follow-up** — `CustomerSimAgent` injects follow-up messages that must be handled
- **Forbidden action traps** — some actions are terminal and immediately lock the score

## What Makes It Novel

AegisDesk combines things most benchmarks treat separately:

- structured operational actions (not free-form text)
- mandatory record inspection before mutation
- world-state and policy-window reasoning
- multi-agent customer simulation
- long-horizon phase ordering with step budgets
- deterministic security and escalation rules
- dense per-step reward shaping

This is a more realistic RL training target than a static instruction benchmark and more auditable than a free-form LLM-judge environment.

## Reward Model

```
reward = progress_delta
       + behavior_adjustment
       + phase_bonus
       + (qa_score × 0.1 × 0.15)
```

- `progress_delta`: rubric progress change this step
- `behavior_adjustment`: sum of behavior penalties (invalid payload: −0.05, loop: −0.03, …)
- `phase_bonus`: +0.05 per newly completed investigation phase (in declared order only)
- `qa_score × 0.1 × 0.15`: QualityReviewAgent contribution

No LLM in the reward path. All rubric items are fixture-defined and deterministic.

## Benchmark Packs

| Pack | Count | Role |
|---|---:|---|
| `core` | 3 | canonical baseline fixtures |
| `v2` | 6 | canonical Round 2 fixtures |
| `generalization` | 18 | held-out judged variants (never trained on) |
| `showcase` | 3 | demo-only fixtures |
| `benchmark` | 27 | full judged set = `core + v2 + generalization` |
| `all` | 30 | full surfaced catalog |

## Full Fixture Catalog

| Fixture ID | Track | Judged |
|---|---|:---:|
| `billing_seat_adjustment` | core | ✓ |
| `login_incident_triage` | core | ✓ |
| `suspicious_admin_request` | core | ✓ |
| `customer_escalation_chain` | v2 | ✓ |
| `multi_tier_billing_dispute` | v2 | ✓ |
| `data_breach_response_lifecycle` | v2 | ✓ |
| `contract_renewal_negotiation` | v2 | ✓ |
| `service_reinstatement_review` | v2 | ✓ |
| `api_partner_access_audit` | v2 | ✓ |
| `billing_seat_adjustment_v1` | generalization | ✓ |
| `billing_seat_adjustment_v2` | generalization | ✓ |
| `login_incident_triage_v1` | generalization | ✓ |
| `login_incident_triage_v2` | generalization | ✓ |
| `suspicious_admin_request_v1` | generalization | ✓ |
| `suspicious_admin_request_v2` | generalization | ✓ |
| `customer_escalation_chain_v1` | generalization | ✓ |
| `customer_escalation_chain_v2` | generalization | ✓ |
| `multi_tier_billing_dispute_v1` | generalization | ✓ |
| `multi_tier_billing_dispute_v2` | generalization | ✓ |
| `data_breach_response_lifecycle_v1` | generalization | ✓ |
| `data_breach_response_lifecycle_v2` | generalization | ✓ |
| `contract_renewal_negotiation_v1` | generalization | ✓ |
| `contract_renewal_negotiation_v2` | generalization | ✓ |
| `service_reinstatement_review_v1` | generalization | ✓ |
| `service_reinstatement_review_v2` | generalization | ✓ |
| `api_partner_access_audit_v1` | generalization | ✓ |
| `api_partner_access_audit_v2` | generalization | ✓ |
| `admin_role_transfer_verification` | showcase | — |
| `api_rate_limit_escalation` | showcase | — |
| `tax_exemption_credit_review` | showcase | — |

## Training Pipeline

Training corpus (built from public datasets via `scripts/fetch_real_datasets.py`):

| Corpus | Rows | Sources |
|---|---:|---|
| `support_sft.jsonl` | 15,124 | Bitext, ABCD, tau-bench, SGD, oracle traces |
| `support_pref.jsonl` | 7,119 | HelpSteer2, harvested DPO pairs |

**Training order:**
1. SFT on `support_sft.jsonl` (Qwen2.5-7B-Instruct + LoRA 4-bit)
2. Preference tuning on `support_pref.jsonl`
3. GRPO on 9 canonical training fixtures (live environment reward signal)
4. Evaluate on all 27 judged fixtures

**Key discipline:** the 18 `generalization` fixtures are excluded from SFT, preference tuning, and GRPO. They are only used for evaluation.

**Training notebook:** [training/AegisDesk_Kaggle_GRPO.ipynb](training/AegisDesk_Kaggle_GRPO.ipynb)  
**Results:** [training/benchmark_results.json](training/benchmark_results.json)

## API

```bash
# Health
GET  /

# Start episode
POST /reset   {"task_id": "billing_seat_adjustment", "seed": 42}

# Step
POST /step    {"action_type": "open_ticket", "ticket_id": "TKT-001"}

# Internal state (debug)
POST /state

# Surfaced fixture catalog
GET  /tasks

# Machine-readable benchmark summary
GET  /benchmark-card

# Scored oracle trace for any surfaced fixture
GET  /trajectory-report?fixture_id=billing_seat_adjustment
```

## Local Setup

```bash
git clone https://github.com/kumarabhik/AegisDesk.git
cd AegisDesk
pip install -e .
python -m server.app
# → http://127.0.0.1:7860/home
```

## Validation

```bash
python -m pytest -q          # 57 passed
openenv validate             # [OK] meta: Ready for multi-mode deployment
python oracle_demo.py --pack benchmark --seed 11   # all 27 judged fixtures
python oracle_demo.py --pack all --seed 11         # all 30 surfaced fixtures
```

## Repo Layout

| Path | Purpose |
|---|---|
| `server/app.py` | FastAPI application |
| `server/environment.py` | Gym-style reset/step/state |
| `server/reward.py` | Dense reward shaping |
| `server/grader.py` | Deterministic rubric grader |
| `server/fixtures.py` | Fixture catalog registry |
| `server/task_data/*.yaml` | Episode specifications |
| `training/train_grpo_aegisdesk.py` | GRPO training script |
| `training/self_improve.py` | Self-improvement pipeline |
| `training/AegisDesk_Kaggle_GRPO.ipynb` | Kaggle training notebook |
| `oracle_tools.py` | Oracle trajectory tools |
| `oracle_demo.py` | Oracle runner CLI |
| `scripts/fetch_real_datasets.py` | Corpus builder |
| `design_doc.md` | Full design narrative |
