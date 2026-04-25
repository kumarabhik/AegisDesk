# AegisDesk Round 2 Design Doc

## Overview

`AegisDesk` is the public benchmark name for `support_ops_env`, an OpenEnv environment for B2B SaaS support operations. The benchmark tests whether an agent can inspect the right evidence, obey policy, avoid unsafe shortcuts, coordinate across multiple records and stakeholders, and finish a case with a deterministic, auditable resolution.

This document is the Round 2 source of truth for:
- what the benchmark is
- what is publicly surfaced
- how the 27 judged fixtures are organized
- how training data should be built
- how reward and grading work
- what is verified now
- what is still missing before a true top-finish submission

For the archived Round 1 design narrative, see [ROUND1_DESIGN_DOC.md](ROUND1_DESIGN_DOC.md).

The main shift in this pass is that AegisDesk no longer behaves like “9 canonical tasks plus some hidden variant files.” It now has a first-class fixture identity model and a clear public story:
- train on `9` canonical enterprise tasks
- evaluate on `18` held-out judged variants
- keep `3` showcase fixtures outside the main score-report pack

Important distinction:
- the YAML files in `server/task_data/` are benchmark episode specifications
- the real training data is built separately into `training/data/support_sft.jsonl` and `training/data/support_pref.jsonl`

## Submission Snapshot

### Project identity

- Public benchmark name: `AegisDesk`
- Internal environment id: `support_ops_env`
- Live Space: `https://i4mgr00t-meta.hf.space`
- GitHub repo: `https://github.com/kumarabhik/AegisDesk`
- Framework: OpenEnv
- Runtime: FastAPI + in-memory deterministic fixtures

### Current verified repo state

- `python -m pytest -q` -> `57 passed`
- `openenv validate` -> `[OK] meta: Ready for multi-mode deployment`
- `/tasks` exposes `30` surfaced fixtures
- `/benchmark-card` reports `27` judged fixtures and `30` surfaced fixtures
- `/trajectory-report` supports canonical task ids and exact `fixture_id`s
- `oracle_demo.py` supports `core`, `v2`, `benchmark`, `generalization`, `showcase`, and `all`
- `python scripts/fetch_real_datasets.py` completes and builds real corpora plus the RL manifest

### Current benchmark story

- `3` core canonical fixtures
- `6` Round 2 canonical fixtures
- `18` held-out judged generalization fixtures
- `3` showcase fixtures

### Main truth right now

AegisDesk is a strong environment and a much better benchmark than it was during the earlier 9-task-only pass. The main remaining risk is still not environment correctness. The main remaining risk is **proof of learning**:
- no checked-in real reward curve PNGs yet
- no checked-in real loss curve PNGs yet
- no checked-in trained-vs-baseline deltas yet
- no checked-in `training/benchmark_results.json` from a real GPU run yet

That means the project is now **benchmark-strong but evidence-incomplete**.

## Why This Is A Good Project Now

The project is already good for four reasons:

1. The environment is real enough to matter.
   It models enterprise support operations rather than a toy instruction-following game. The agent has to choose the correct ticket, inspect the right records, honor policy windows, coordinate via escalation, and finish safely.

2. The grading is deterministic.
   There are no LLM judges, no fuzzy semantic scoring, and no hidden language-model evaluator in the reward path. The same fixture plus the same action sequence yields the same score.

3. The task mix covers meaningful capability axes.
   The canonical fixtures span baseline billing, incident handling, security judgment, multi-agent billing coordination, long-horizon breach/renewal workflows, and world-model-aware decision making.

4. The environment is trainable.
   It already exposes dense rewards, an OpenEnv-compatible server, an oracle viewer, a baseline inference path, trajectory harvesting, DPO pair generation, GRPO training, and a notebook path.

The benchmark is not “good because the design doc is long.” It is good because the environment and evaluation contract are clear, auditable, and interesting.

## Why This Still Is Not Fully Top-Finish-Safe

The remaining top-finish blockers are now sharply defined:

1. Real before/after training evidence is still missing.
2. The reward/training story is stronger than before, but the strongest artifact path still depends on a real HF Jobs or comparable GPU run.
3. The docs can now describe a stronger benchmark truthfully, but they still cannot claim a winning model until one exists.

Plainly:

> AegisDesk is now ready to be judged as a serious benchmark.
> AegisDesk is not yet ready to claim that its RL model is “the best” without a real trained result.

## Judge Rubric Map

| Criterion | Weight | Current state | Why it is credible now | Main remaining gap |
|---|---:|---|---|---|
| Environment Innovation | 40% | Strong | 27 judged fixtures, real enterprise workflow, deterministic scoring, held-out generalization story | Need trained evidence to prove the benchmark teaches transferable behavior |
| Storytelling & Presentation | 30% | Stronger than before | landing page, console, benchmark card, oracle viewer, design doc, slide deck, README | Needs final artifact linking once training results exist |
| Showing Improvement in Rewards | 20% | Blocked on evidence | benchmark/report path now exists for 27 judged fixtures | Need a real baseline vs champion comparison |
| Reward & Training Pipeline | 10% | Implemented | dataset fetch path, corpora builders, manifest, self-improve loop, GRPO starter, notebook | Need a real completed run with saved outputs |

## Benchmark Taxonomy

### Public pack definitions

- `core`: 3 canonical baseline fixtures
- `v2`: 6 canonical Round 2 fixtures
- `generalization`: 18 held-out judged variants
- `showcase`: 3 legacy demo fixtures

### Judged benchmark definition

The official judged benchmark is:
- `benchmark = core + v2 + generalization`
- `judged_total = 27`

The full surfaced catalog is:
- `all = benchmark + showcase`
- `surfaced_total = 30`

### Training discipline

- canonical training fixtures: `9`
- held-out judged generalization fixtures: `18`
- non-judged private curriculum variants: `18`
- showcase fixtures: excluded from the main score-report story

## Surfaced Fixture Catalog

| Fixture ID | Task Family | Track | Judged | Difficulty | Role |
|---|---|---|---:|---|---|
| `billing_seat_adjustment` | `billing_seat_adjustment` | `core` | yes | `easy` | canonical baseline billing |
| `login_incident_triage` | `login_incident_triage` | `core` | yes | `medium` | canonical incident-aware restraint |
| `suspicious_admin_request` | `suspicious_admin_request` | `core` | yes | `hard` | canonical security judgment |
| `customer_escalation_chain` | `customer_escalation_chain` | `v2` | yes | `medium_hard` | canonical multi-agent billing |
| `multi_tier_billing_dispute` | `multi_tier_billing_dispute` | `v2` | yes | `medium` | canonical reconciliation |
| `data_breach_response_lifecycle` | `data_breach_response_lifecycle` | `v2` | yes | `hard` | canonical long-horizon breach response |
| `contract_renewal_negotiation` | `contract_renewal_negotiation` | `v2` | yes | `medium_hard` | canonical long-horizon blocker resolution |
| `service_reinstatement_review` | `service_reinstatement_review` | `v2` | yes | `easy_medium` | canonical world-model policy check |
| `api_partner_access_audit` | `api_partner_access_audit` | `v2` | yes | `medium` | canonical world-aware access audit |
| `billing_seat_adjustment_v1` | `billing_seat_adjustment` | `generalization` | yes | `easy` | held-out billing variant |
| `billing_seat_adjustment_v2` | `billing_seat_adjustment` | `generalization` | yes | `easy` | held-out billing variant |
| `login_incident_triage_v1` | `login_incident_triage` | `generalization` | yes | `medium` | held-out incident variant |
| `login_incident_triage_v2` | `login_incident_triage` | `generalization` | yes | `medium` | held-out incident variant |
| `suspicious_admin_request_v1` | `suspicious_admin_request` | `generalization` | yes | `hard` | held-out security variant |
| `suspicious_admin_request_v2` | `suspicious_admin_request` | `generalization` | yes | `hard` | held-out security variant |
| `customer_escalation_chain_v1` | `customer_escalation_chain` | `generalization` | yes | `medium_hard` | held-out multi-agent variant |
| `customer_escalation_chain_v2` | `customer_escalation_chain` | `generalization` | yes | `medium_hard` | held-out multi-agent variant |
| `multi_tier_billing_dispute_v1` | `multi_tier_billing_dispute` | `generalization` | yes | `medium` | held-out reconciliation variant |
| `multi_tier_billing_dispute_v2` | `multi_tier_billing_dispute` | `generalization` | yes | `medium` | held-out reconciliation variant |
| `data_breach_response_lifecycle_v1` | `data_breach_response_lifecycle` | `generalization` | yes | `hard` | held-out breach-response variant |
| `data_breach_response_lifecycle_v2` | `data_breach_response_lifecycle` | `generalization` | yes | `hard` | held-out breach-response variant |
| `contract_renewal_negotiation_v1` | `contract_renewal_negotiation` | `generalization` | yes | `medium_hard` | held-out renewal variant |
| `contract_renewal_negotiation_v2` | `contract_renewal_negotiation` | `generalization` | yes | `medium_hard` | held-out renewal variant |
| `service_reinstatement_review_v1` | `service_reinstatement_review` | `generalization` | yes | `easy_medium` | held-out reinstatement variant |
| `service_reinstatement_review_v2` | `service_reinstatement_review` | `generalization` | yes | `easy_medium` | held-out reinstatement variant |
| `api_partner_access_audit_v1` | `api_partner_access_audit` | `generalization` | yes | `medium` | held-out access-audit variant |
| `api_partner_access_audit_v2` | `api_partner_access_audit` | `generalization` | yes | `medium` | held-out access-audit variant |
| `admin_role_transfer_verification` | `admin_role_transfer_verification` | `showcase` | no | `hard` | showcase security demo |
| `api_rate_limit_escalation` | `api_rate_limit_escalation` | `showcase` | no | `medium` | showcase incident demo |
| `tax_exemption_credit_review` | `tax_exemption_credit_review` | `showcase` | no | `easy` | showcase billing demo |

## Private Curriculum Fixtures

These fixture ids exist locally but are intentionally not surfaced in `/tasks`:

- `billing_seat_adjustment_v3`
- `billing_seat_adjustment_v4`
- `customer_escalation_chain_v3`
- `customer_escalation_chain_v4`
- `api_partner_access_audit_v3`
- `api_partner_access_audit_v4`
- `contract_renewal_negotiation_v3`
- `contract_renewal_negotiation_v4`
- `data_breach_response_lifecycle_v3`
- `data_breach_response_lifecycle_v4`
- `login_incident_triage_v3`
- `login_incident_triage_v4`
- `multi_tier_billing_dispute_v3`
- `multi_tier_billing_dispute_v4`
- `service_reinstatement_review_v3`
- `service_reinstatement_review_v4`
- `suspicious_admin_request_v3`
- `suspicious_admin_request_v4`

These are allowed for curriculum experiments, but they must stay out of the official judged story unless the surfaced benchmark definition is intentionally revised.

## Fixture Identity Contract

### Old behavior

Previously the loader keyed fixtures only by `task_id`. That made variant files unsafe because any `_v1` or `_v2` file that reused the same `task_id` could collapse onto the canonical fixture identity.

### New behavior

Every fixture now has a first-class `fixture_id`.

Rules:
- canonical fixtures use `fixture_id == task_id`
- variants use `fixture_id == <task_id>_v<n>`
- `task_id` remains the task family
- `fixture_id` is the public identity for exact episodes

### Resolution behavior

- `POST /reset` prefers `fixture_id`
- if only `task_id` is given, the server resolves to the canonical fixture for that family
- `/trajectory-report` accepts `fixture_id` and falls back to canonical `task_id`
- `/tasks` returns both `fixture_id` and `task_id`

## Public Interface Contract

### `/tasks`

Returns the surfaced public catalog with:
- `fixture_id`
- `task_id`
- `track`
- `judged`
- `difficulty`
- `task_brief`
- `max_steps`
- `reply_template_id`
- `reply_checklist`
- `oracle_available`

### `/benchmark-card`

Returns:
- `core=3`
- `v2=6`
- `generalization=18`
- `showcase=3`
- `judged_total=27`
- `surfaced_total=30`

### `/reset`

Accepted payload:

```json
{
  "task_id": "billing_seat_adjustment",
  "fixture_id": "billing_seat_adjustment_v1",
  "seed": 7
}
```

Resolution rules:
- if `fixture_id` is present, it wins
- if only `task_id` is present, canonical fixture is used
- if both are omitted, environment falls back to default cycle behavior

### `/trajectory-report`

Accepted query style:

- `/trajectory-report?task_id=billing_seat_adjustment&seed=7`
- `/trajectory-report?fixture_id=billing_seat_adjustment_v1&seed=7`

### Oracle CLI

Supported packs:
- `core`
- `v2`
- `benchmark`
- `generalization`
- `showcase`
- `extended` as a compatibility alias for `showcase`
- `all`

## Oracle Coverage

The oracle layer no longer depends only on hardcoded canonical task ids.

The current behavior is:
- read the fixture
- resolve exact `fixture_id`
- parse the fixture’s `oracle_reference_path`
- construct typed `SupportAction` instances
- run the local or remote oracle trajectory

This matters because it makes the 18 held-out judged variants first-class oracle citizens. They are not merely visible in the catalog; they can actually be scored and inspected.

## Runtime Invariants

These invariants remain non-negotiable:

1. Deterministic grading only.
   No LLM judge is permitted in the reward or final score path.

2. No live data during environment execution.
   Task execution stays fixture-backed and in-memory.

3. No hidden truth leakage into observations.
   Public observations must not expose rubric internals, forbidden-action lists, or oracle steps.

4. Safe action enforcement remains local to fixtures.
   Unsafe behavior is modeled through per-fixture forbidden rules, not vague global heuristics.

## Reward Model

The reward formula remains:

```text
reward =
  progress_delta
  + behavior_adjustment
  + phase_bonus
  + (qa_score × 0.1 × 0.15)
```

Where:
- `progress_delta` is the rubric progress gained this step
- `behavior_adjustment` is the signed sum of penalties
- `phase_bonus` is `+0.05` per newly completed ordered phase
- `qa_score × 0.1 × 0.15` is the deterministic `QualityReviewAgent` contribution

### Current reward-side guarantees

- ordered phase completion is enforced
- phase bonuses only fire for newly completed ordered phases
- repeated irrelevant inspection is penalized
- repeated action signatures are penalized
- invalid payloads are penalized
- world-policy terminal violations still terminate deterministically
- guard fields like `requires_escalation_first` and `requires_checks_first` are honored

## Task Design Themes

### Core

- canonical billing correction
- canonical incident-aware restraint
- canonical security escalation

### Round 2 canonical

- multi-agent billing follow-up
- multi-party billing reconciliation
- long-horizon breach lifecycle
- long-horizon renewal blocker resolution
- world-aware reinstatement
- world-aware API access review

### Held-out generalization

These fixtures deliberately change ids, entities, numbers, and local details while preserving the structure of the task family. They are meant to test whether learning transfers beyond memorizing exact canonical fixture tokens.

### Showcase

These remain useful for:
- judge walkthroughs
- oracle demos
- qualitative story depth

But they are not part of the official 27-fixture benchmark delta story.

## Real Dataset Provenance

The Round 2 training path now targets these sources:

| Source | Status in repo | Purpose |
|---|---|---|
| Bitext customer support dataset | fetch script implemented | support phrasing, billing/account/access SFT seeds, simulator realism |
| ABCD | fetch script implemented | policy-constrained workflows, action taxonomy, simulator realism |
| Sierra tau-bench / tau2-bench few-shot trajectories | fetch script implemented | tool-use SFT seeds |
| Schema-Guided Dialogue | fetch script implemented | long-horizon stateful task-oriented dialogue seeds |
| HelpSteer2 preferences | fetch script implemented | preference tuning / response-quality supervision |
| DialogStudio / MultiWOZ sample | optional fetch path implemented | optional extra task-oriented SFT seed |

## Derived Training Corpora

## External Data Acquisition Checklist

When preparing the real training run, the expected raw inputs should land under `training/raw/` with source-specific subfolders or archives. The fetch/build path is designed around these concrete artifacts:

- Bitext export or parquet snapshot for customer-support utterances
- ABCD dialogue and ontology files for policy-constrained support flows
- Sierra tau-bench / tau2-bench prompt or trajectory examples for tool-use seeding
- Schema-Guided Dialogue archive for schema-following and state tracking
- HelpSteer2 rows for preference supervision
- optional DialogStudio or MultiWOZ sample for extra dialogue variety

The important invariant is not the exact download mechanism. The important invariant is that the derived corpora builder can trace every row in `support_sft.jsonl` and `support_pref.jsonl` back to one of these explicit sources or to AegisDesk oracle traces.

Verified local build in this workspace:
- `support_sft.jsonl`: `15,124` rows
- `support_pref.jsonl`: `7,119` rows
- `dataset_build_report.json`: written with source-level counts and corpus targets

### `training/data/support_sft.jsonl`

Built from:
- Bitext support examples
- ABCD sample dialogues
- Sierra tau few-shot conversations
- SGD dialogue rows
- AegisDesk oracle step demonstrations from canonical plus private curriculum fixtures
- optional DialogStudio/MultiWOZ examples if enabled

Current local row counts:
- Bitext: `5,776`
- ABCD: `5,000`
- tau-bench / tau2-bench: `69`
- SGD: `4,000`
- combined `support_sft.jsonl`: `15,124`

### `training/data/support_pref.jsonl`

Built from:
- HelpSteer2 preference pairs
- locally harvested AegisDesk DPO pairs when available

Current local row counts:
- HelpSteer2: `7,118`
- combined `support_pref.jsonl`: `7,119`

### `training/support_rl_manifest.json`

Defines:
- canonical training fixtures
- held-out judged fixtures
- showcase fixtures
- private curriculum fixtures
- allowed GRPO fixtures
- excluded-from-training held-out fixtures

## Training Stack

The intended top-finish stack is hybrid, not single-stage:

### Stage 1: Unsloth QLoRA SFT

- target default: `Qwen/Qwen3-8B`
- fallback: `Qwen/Qwen3-4B`
- input: `support_sft.jsonl`

### Stage 2: Unsloth DPO or ORPO

- input: `support_pref.jsonl`
- goal: shape better replies and preference-aligned action choices before RL

### Stage 3: TRL GRPO on OpenEnv

- environment: `support_ops_env`
- canonical training pack: `9` canonical fixtures
- optional private curriculum fixtures: allowed by manifest
- held-out `generalization` fixtures: excluded

## Training Discipline Rules

These are important enough to state explicitly:

1. The 18 `generalization` fixtures must never be used for SFT.
2. The 18 `generalization` fixtures must never be used for DPO/ORPO.
3. The 18 `generalization` fixtures must never be used for GRPO curriculum.
4. The 18 `generalization` fixtures are evaluation fixtures, not training fixtures.
5. Private curriculum variants may be used experimentally, but only if they remain outside the surfaced benchmark contract.

## Hardware Plan

### Preferred

- HF Jobs GPU
- `Qwen/Qwen3-8B`
- L4 or A10G class hardware

### Fallback

- HF Jobs `t4-medium`
- `Qwen/Qwen3-4B`

### Repo truth

The repo is ready for this path:
- training notebook exists
- `training/train_grpo_aegisdesk.py` now supports an RL manifest
- `training/self_improve.py` evaluates the 27 judged fixtures and can emit `training/benchmark_results.json`
- `scripts/fetch_real_datasets.py` now builds the SFT and preference corpora locally

What it does **not** yet have is a checked-in finished GPU run.

## Evidence Plan

The winning evidence package should include:

- baseline run over 27 judged fixtures
- champion run over 27 judged fixtures
- reward curve PNG
- loss curve PNG
- per-track delta figure
- compact canonical-vs-held-out comparison table
- `training/benchmark_results.json`

## Ablation Table Template

This table is intentionally present even though the real values are still pending:

| Configuration | Canonical mean | Held-out mean | Overall mean | Status |
|---|---:|---:|---:|---|
| Baseline | pending | pending | pending | not checked in yet |
| SFT only | pending | pending | pending | not checked in yet |
| SFT + preference tuning | pending | pending | pending | not checked in yet |
| SFT + preference tuning + GRPO | pending | pending | pending | not checked in yet |

The correct next step is to fill this table with real numbers, not invented ones.

## Current Verified Wins

What is true now:

- fixture identity redesign is implemented
- benchmark card and task catalog reflect the new 30/27 story
- held-out generalization fixtures are surfaced and judged
- oracle tooling covers all surfaced fixtures
- local tests pass with `57` passing tests
- OpenEnv validation still passes
- training scripts now separate canonical training fixtures from held-out judged fixtures
- dataset fetch/build infrastructure is stronger and more explicit than before
- real derived corpora are now generated in-repo with `15,124` SFT rows and `7,119` preference rows

## Current Missing Evidence

What is still not true now:

- there is no real champion benchmark file checked in yet
- there is no real trained-vs-baseline delta checked in yet
- there are no real reward/loss plots checked in yet
- there is no honest basis yet to claim “best RL model” in the submission narrative

## Submission Acceptance Targets

The benchmark should be considered top-finish-safe only when all of these are true:

1. Positive mean delta on the `9` canonical judged fixtures.
2. Positive mean delta on the `18` held-out judged fixtures.
3. No severe regression on security-sensitive tasks like:
   - `suspicious_admin_request`
   - `api_partner_access_audit`
4. `training/benchmark_results.json` is checked in with real values.
5. Reward/loss plots are checked in and linked from the README.
6. README, RESULTS, slide deck, and this document all report the same counts and same evidence paths.

## Strongest Submission Path

If the goal is not merely “strong” but “strongest plausible submission,” the next work should be done in this order:

1. Verify the benchmark contract with `pytest`, `openenv validate`, and oracle coverage on the judged pack.
2. Verify the external dataset sources are reachable.
3. Rebuild `support_sft.jsonl`, `support_pref.jsonl`, `dataset_build_report.json`, and `training/support_rl_manifest.json`.
4. Run the training readiness doctor to validate corpus counts, manifest invariants, and endpoint counts.
5. Run a real `Qwen/Qwen3-8B` SFT smoke job and confirm checkpoint persistence.
6. Run a real `Qwen/Qwen3-8B` SFT champion job and save the adapter.
7. Run a real `Qwen/Qwen3-8B` DPO champion job and save the adapter.
8. Run a GRPO stabilize pass on the canonical `9` fixtures only.
9. Run a GRPO champion pass on the canonical `9` plus private curriculum variants, while keeping all `18` held-out `generalization` fixtures excluded.
10. Evaluate baseline and champion across all `27` judged fixtures with `3` seeds, write `training/benchmark_results.json`, generate the plots, and sync the judge-facing docs to the real numbers.

Important status rule:
- before Step 10 is finished, the project should be described as “strong benchmark, evidence-incomplete”
- after Step 10 is finished with positive canonical and held-out deltas, the project can be described as “top-finish-ready”

Execution helpers:
- `python training/strongest_submission.py --list`
- `python training/check_training_readiness.py --env-url https://i4mgr00t-meta.hf.space`

## Final Push Checklist

### Already done

- fixture identity migration
- public pack redesign
- 30-surfaced / 27-judged catalog
- oracle coverage for held-out variants
- API contract updates
- self-improve benchmark output path
- RL manifest
- stronger dataset fetch/build script

### Still to do with real compute

- run real SFT
- run real preference tuning
- run real GRPO
- benchmark baseline vs champion
- save plots
- check in evidence artifacts

## Session Continuity Protocol

Every assistant continuing this project should:

1. Read `CLAUDE.md`.
2. Read this design doc.
3. Read `roadmap.md`.
4. Verify repo truth before making claims.
5. Keep the benchmark story truthful.

## Session Log

### Session 2026-04-23 — v1 environment complete, deployed to HF Space

Built the full v1 environment: deterministic core tasks, OpenEnv-compliant server, Docker path, baseline inference runner, local tests, and live HF Space deployment.

### Session 2026-04-23 — v2 scaffolded for Round 2 themes

Added the 6 v2 tasks plus multi-agent hooks, world-state support, long-horizon phase tracking, and the self-improvement training modules.

### Session 2026-04-25 — fixture generator, notebook, CLAUDE.md

Added the fixture generator, training notebook, and repo-level working rules. At that point the codebase had strong breadth, but the repo still had stale task taxonomy and incomplete oracle/demo coverage.

### Session 2026-04-25 — Round 2 hardening pass

Normalized the surfaced task taxonomy to `core=3`, `v2=6`, `extended=3`. Added v2 oracle coverage, repaired phase-bonus ordering, honored guarded forbidden-action semantics, made `/tasks` expose truthful `oracle_available`, updated `/benchmark-card` counts, refreshed judge-facing docs, added a slide deck, and brought the local suite to `53` passing tests with `openenv validate` still passing.

### Session 2026-04-25 — top-finish benchmark expansion

Implemented first-class `fixture_id` support across the models, fixture loader, environment, client, FastAPI routes, oracle tooling, and tests. Surfaced the benchmark as `30` fixtures with `27` judged fixtures and `3` showcase fixtures. Promoted 18 held-out variants into a public `generalization` pack, updated the oracle viewer and CLI, brought the suite to `57` passing tests, kept `openenv validate` green, added an RL manifest, upgraded the training scripts to respect the canonical-vs-held-out split, and rewrote the dataset fetch path around real external sources and derived corpora builders.

Remaining blocker after this session: real checked-in training evidence from an actual GPU run.

### Session 2026-04-25 — continuity sync after compaction

Verified the repo state after compaction instead of trusting memory, then aligned the remaining project-control docs with the implemented benchmark. Updated `CLAUDE.md` to reflect the `fixture_id` contract, the `30` surfaced fixtures / `27` judged fixtures taxonomy, and the new `generalization` plus `showcase` packs. Updated `roadmap.md` so Phase 20 no longer claims the fixture loader is blocked on `task_id`, Phase 21 now matches the actual benchmark/API/oracle state, and the Round 2 submission definition now tells the canonical-vs-held-out story truthfully. Remaining blocker is unchanged: real GPU-backed training evidence still needs to be generated and checked in.

### Session 2026-04-25 — benchmark credibility and real corpus build

Expanded the public benchmark to `30` surfaced fixtures with `27` judged fixtures and `18` held-out generalization variants. Verified `python -m pytest -q`, `openenv validate`, `python oracle_demo.py --pack benchmark --seed 11`, and `python oracle_demo.py --pack all --seed 11`. Fixed the dataset builder import path, then ran `python scripts/fetch_real_datasets.py` to completion, generating `training/data/support_sft.jsonl` with `15,124` rows, `training/data/support_pref.jsonl` with `7,119` rows, `training/data/dataset_build_report.json`, and an updated `training/support_rl_manifest.json`.
