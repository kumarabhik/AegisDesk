# AegisDesk — Agent Working Rules

This file is read automatically by Claude Code, Cursor, Codex, and any other AI coding
assistant that respects CLAUDE.md / agent instruction files. Every rule here exists
because a previous session learned it the hard way. Follow all of them exactly.

---

## 1. Session Continuity Protocol (CRITICAL)

**This is the most important rule in this file.**

Context windows get compacted. When that happens, the assistant loses the live conversation
history and must reconstruct intent from scratch. Hallucinations follow — wrong file paths,
invented function names, stale assumptions about what has already been built.

### Rule: every compaction summary must be persisted to `design_doc.md`

When a context window is compacted (or when a session ends and a new one begins on the
same task), the compaction summary **must be appended** to the
`## Session Log` section at the bottom of `design_doc.md` before continuing work.

Format:

```markdown
### Session <ISO-date> — <one-line description of what the session did>
<paste the compaction summary verbatim, or a concise version of it>
```

Why this matters:
- `design_doc.md` is the single source of truth for what has been built, why, and what is next.
- It is read at the start of every session by every assistant (Claude, Codex, Cursor).
- A stale `design_doc.md` causes hallucinations across ALL assistants, not just the one
  whose context was compacted.
- Git history tells you what changed; `design_doc.md` tells you WHY and WHAT IS STILL PENDING.

### Rule: verify before asserting

Before stating that a file, function, or class exists, **read it or grep for it**.
Memory and compaction summaries describe state at a point in time. Code changes.
"The memory says X exists" is not the same as "X exists now."

---

## 2. Project Identity

- Public name: **AegisDesk**
- Internal environment ID: `support_ops_env`
- Framework: **OpenEnv** (Meta/PyTorch) — Gym-style `reset()`, `step()`, `state()` API
- Deployment: FastAPI server, Docker, Hugging Face Spaces at `https://i4mgr00t-meta.hf.space`
- GitHub: `https://github.com/kumarabhik/AegisDesk`
- HF Space: `https://huggingface.co/spaces/I4mGr00T/Meta`

---

## 3. Architecture Rules

### Never break determinism
- All task scoring must be deterministic: same fixture + same action sequence = same score.
- No LLM judges. No embedding similarity. No fuzzy text scoring. Ever.
- `WorldStateEngine`, `CustomerSimAgent`, and `QualityReviewAgent` are all seeded and
  fixture-driven. Keep them that way.

### No live data during environment execution
- The environment is fully in-memory and fixture-backed.
- Do not add any external API calls, database reads, or network requests inside
  `server/environment.py`, `server/grader.py`, `server/reward.py`, or any fixture loader.

### Reward formula (v2) — do not change without updating design_doc.md
```
reward = progress_delta + behavior_adjustment + phase_bonus + (qa_score × 0.1 × 0.15)
```
- `progress_delta` = rubric progress change this step
- `behavior_adjustment` = sum of behavior penalties (invalid payload: -0.05, loop: -0.03, etc.)
- `phase_bonus` = +0.05 per newly completed investigation phase (in declared order only)
- `qa_score × 0.1 × 0.15` = QualityReviewAgent contribution (WEIGHT=0.15)

### Forbidden action model
- Forbidden actions are defined in each fixture YAML under `forbidden_actions`.
- A `terminal: true` forbidden action sets `done=True` immediately and locks the score.
- Do not add global forbidden action logic to the grader — keep it fixture-local.

### Observation vs. state boundary
- `SupportObservation` (returned by `reset()`/`step()`) must NEVER contain hidden rubric
  truth, forbidden action lists, or oracle paths.
- `SupportState` (returned by `state()`) may contain grader internals for debugging.

---

## 4. File Layout Rules

### Fixture files
- Live in `server/task_data/<fixture_id>.yaml`
- Canonical fixtures use `fixture_id == task_id`
- Generated variants use the naming convention `<task_id>_v<n>.yaml` and keep the
  parent `task_id`
- Canonical task families must be listed in `server/fixtures.py` under
  `CANONICAL_TASK_IDS` or `V2_TASK_IDS`
- Surfaced held-out variants must be listed under `GENERALIZATION_FIXTURE_IDS`
- Showcase-only fixtures must be listed under `SHOWCASE_TASK_IDS`
- Never register a fixture that has not been validated by `validate_fixture()` in
  `scripts/generate_fixtures.py`

### Training files
- `training/train_grpo_aegisdesk.py` — main GRPO training script
- `training/self_improve.py` — end-to-end self-improvement CLI
- `training/trajectory_harvester.py`, `dpo_pair_generator.py`, `adaptive_scheduler.py`
- `training/check_training_readiness.py` — validates corpora, manifest invariants, and training prerequisites
- `training/strongest_submission.py` — numbered 10-step strongest-submission workflow
- `training/data/` — seed data (tau-bench SFT, Bitext utterances, ABCD taxonomy)
- `training/AegisDesk_Training.ipynb` — Colab/HF Jobs training notebook

### Script files
- `scripts/generate_fixtures.py` — LLM-driven fixture variant generator
- `verify_space.py`, `submission_audit.py` — deployment verification helpers

---

## 5. Task and Reward Rules

### Current public benchmark roster

The surfaced catalog is now `30` fixtures total:

| Pack | Count | Notes |
|---|---:|---|
| `core` | 3 | canonical benchmark fixtures |
| `v2` | 6 | canonical Round 2 benchmark fixtures |
| `generalization` | 18 | held-out judged variants, never used for training |
| `showcase` | 3 | demo-only fixtures, not part of the judged benchmark |

Judged benchmark:
- `benchmark = core + v2 + generalization = 27` fixtures
- canonical training pack = `core + v2 = 9` fixtures
- surfaced full catalog = `benchmark + showcase = 30` fixtures

Canonical task families:
| Task ID | Track | Theme | Max Steps |
|---|---|---|---|
| `billing_seat_adjustment` | core | Baseline | 12 |
| `login_incident_triage` | core | World Modeling | 12 |
| `suspicious_admin_request` | core | Baseline | 12 |
| `customer_escalation_chain` | v2 | Multi-Agent | 15 |
| `multi_tier_billing_dispute` | v2 | Multi-Agent | 15 |
| `data_breach_response_lifecycle` | v2 | Long-Horizon | 30 |
| `contract_renewal_negotiation` | v2 | Long-Horizon | 25 |
| `service_reinstatement_review` | v2 | World Modeling | 12 |
| `api_partner_access_audit` | v2 | World Modeling | 15 |

Held-out judged variants:
- `billing_seat_adjustment_v1`
- `billing_seat_adjustment_v2`
- `login_incident_triage_v1`
- `login_incident_triage_v2`
- `suspicious_admin_request_v1`
- `suspicious_admin_request_v2`
- `customer_escalation_chain_v1`
- `customer_escalation_chain_v2`
- `multi_tier_billing_dispute_v1`
- `multi_tier_billing_dispute_v2`
- `data_breach_response_lifecycle_v1`
- `data_breach_response_lifecycle_v2`
- `contract_renewal_negotiation_v1`
- `contract_renewal_negotiation_v2`
- `service_reinstatement_review_v1`
- `service_reinstatement_review_v2`
- `api_partner_access_audit_v1`
- `api_partner_access_audit_v2`

### Claim discipline
- Do not describe the project as “strongest”, “best RL model”, or “winning model” unless a real GPU-backed run has produced:
  - `training/benchmark_results.json`
  - reward/loss plots
  - positive deltas on both canonical and held-out judged fixtures
- Before that evidence exists, the truthful description is: **strong benchmark, evidence-incomplete**.

### v1 baseline score
The pre-training baseline mean score over 3 core tasks is **0.27** (measured on live Space,
`Qwen/Qwen2.5-72B-Instruct`). Use this as the comparison baseline in all training reports.

---

## 6. Code Style Rules

- Python 3.11. Pydantic v2. FastAPI. No new mandatory dependencies without updating
  `requirements.txt` AND `server/requirements.txt`.
- Do not add comments that explain WHAT the code does — only add comments for non-obvious
  WHY (hidden constraint, workaround, subtle invariant).
- Do not write multi-paragraph docstrings. One short line maximum.
- Prefer editing existing files over creating new ones.
- Do not introduce abstractions beyond what the task requires.
- `extra="ignore"` on `ForbiddenActionSpec` and `TaskFixture` — do not revert to `"forbid"`.

---

## 7. Git and Deployment Rules

- Remote `origin` = HF Space (auto-deploys on push to `main`)
- Remote `github` = GitHub repo
- Never push to `origin` without verifying the Space build does not break
  `/reset`, `/step`, and `/state` endpoints.
- Never commit `.env` files, `HF_TOKEN`, or any secret.
- After any training run, commit reward curve PNGs and `training/benchmark_results.json`
  to `github` remote so evidence is preserved.

---

## 8. What NOT to do

- Do not mock `WorldStateEngine`, `CustomerSimAgent`, or `QualityReviewAgent` in
  integration tests — these are deterministic and can be exercised directly.
- Do not add a second scoring channel at episode end — the returned reward IS the
  dense per-step reward, not a separate terminal score signal.
- Do not change `task_id` values in generated fixture variants — variants share the
  same `task_id` as their parent, while the public episode identity is `fixture_id`.
- Do not rename `CANONICAL_TASK_IDS`, `V2_TASK_IDS`, `GENERALIZATION_FIXTURE_IDS`,
  or `SHOWCASE_TASK_IDS` in `fixtures.py` — these are referenced by the grader, the
  console, the oracle tooling, and the `/tasks` endpoint.
- Do not add a `--no-verify` flag to git commits to bypass hooks.

---

## 9. Submission Deadline

**3:00 PM April 26, 2026** — no commits after this time.
After the deadline, the repo state at `main` HEAD is what judges evaluate.
