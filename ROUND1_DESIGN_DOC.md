# AegisDesk Round 1 Design Document

This file preserves the Round 1 design narrative for `AegisDesk`.

It is kept as an archive alongside the current [Round 2 design doc](design_doc.md) so the project retains both:
- the original Round 1 benchmark framing
- the current Round 2 benchmark, training, and evidence plan

This Round 1 archive is based on the pre-v2 `design_doc.md` state from the project history on GitHub.

## Overview

`AegisDesk` is the public-facing name for `support_ops_env`, a standalone OpenEnv environment that simulates a real B2B SaaS support-operations workflow. An agent receives a small inbox of 2-3 support tickets, identifies the primary case to resolve, investigates internal records, applies safe operational actions, drafts a structured response, and finalizes a policy-compliant resolution.

The Round 1 environment was designed to evaluate agents on realistic support work:
- triaging competing tickets
- gathering the right evidence before acting
- following operational and security policies
- avoiding unsafe or destructive actions
- producing a deterministic, gradable resolution

The core idea in Round 1 was to fill the gap between toy workflows and overly open-ended browser tasks by providing realistic multi-step decisions with deterministic scoring and fast reproducibility.

## Round 1 Status Snapshot

At the time of the Round 1 design:
- local tests passed with `33` tests
- `openenv validate` passed
- local Docker build/run had been verified
- the exact official pre-validation script passed locally
- the public GitHub repo was live at `https://github.com/kumarabhik/AegisDesk`
- the Hugging Face Space was live at `https://i4mgr00t-meta.hf.space/`
- the live Space responded successfully to `/`, `/reset`, `/step`, and `/state`
- the app exposed `/tasks` and `/console` for task discovery and manual evaluation
- the hackathon submission path used `HF_TOKEN` with `https://router.huggingface.co/v1`
- reusable verification helpers existed at `verify_space.py` and `submission_audit.py`
- local pre-validation mirrors existed at `validate-submission.sh` and `validate-submission.ps1`
- `inference.py` emitted tagged `[START]`, `[STEP]`, and `[END]` stdout lines
- the latest captured live HF-router run reported rounded scores of `0.28`, `0.28`, and `0.25`, with a rounded mean of `0.27`

## Round 1 Goals

- Implement the full OpenEnv interface with typed Pydantic models and validator-friendly packaging.
- Simulate a real support workflow with clear state transitions and hidden internal truth.
- Ship exactly `3` canonical tasks with deterministic graders and meaningful difficulty progression.
- Provide dense reward shaping based on rubric progress, not binary end-only success.
- Support reproducible local runs, Docker execution, Hugging Face Spaces deployment, and a baseline inference script using the OpenAI client.
- Keep the hackathon submission path compliant with the guidance to use a Hugging Face token plus the Hugging Face router rather than requiring a paid OpenAI key.

## Round 1 Non-Goals

- No live integration with SaaS systems, ticketing providers, or external databases.
- No model-based or free-text semantic grading.
- No large-scale inbox management, async work queues, or human handoff systems in v1.
- No dependence on external network access during environment execution.

## Environment Concept

The Round 1 environment represented an internal support console for a fictional B2B SaaS company. Each episode included:
- a small inbox with mixed urgency tickets
- one primary ticket that should be resolved or safely escalated
- supporting records such as account snapshots, invoices, incident status, audit logs, approved contacts, and knowledge base articles

The benchmark stressed several real-world agent capabilities:
- selective attention under competing inbox items
- safe action sequencing before mutation
- grounded decisions using retrieved records
- escalation when direct action would be risky
- structured communication with deterministic content checks

## OpenEnv Interface

### Core models

Round 1 was built around three typed models:
- `SupportAction`
- `SupportObservation`
- `SupportState`

The action space supported:
- `open_ticket`
- `inspect_record`
- `search_kb`
- `set_priority`
- `set_status`
- `add_tag`
- `apply_credit`
- `escalate`
- `draft_reply`
- `finalize_resolution`

Expected action fields included:
- `action_type`
- `ticket_id`
- `record_id`
- `query`
- `priority`
- `status`
- `tag`
- `amount`
- `currency`
- `escalation_team`
- `template_id`
- `reply_checklist`
- `resolution_code`
- `freeform_note`

Round 1 validation rules:
- only fields relevant to the given `action_type` were permitted
- required fields had to be present for each action type
- numeric amounts had to be positive and bounded
- enumerated values had to be rejected if unknown
- invalid actions should not crash the environment and instead return an error plus reward penalty

### Observation design

The public observation returned by `reset()` and `step()` included:
- `task_brief`
- `inbox`
- `active_ticket_id`
- `focus_panel`
- `available_record_ids`
- `action_history`
- `step_count`
- `remaining_steps`
- `last_action_error`

Observation rules:
- only records explicitly opened by the agent could appear in detail
- hidden rubric truth and forbidden action lists could not appear
- distractor tickets remained visible in summary form unless opened

### Internal state design

The internal `state()` payload was allowed to expose grader details for debugging, including:
- episode identity
- selected ticket
- viewed records
- mutations and credits
- escalations
- reply draft state
- rubric progress
- rubric breakdown
- unsafe action log
- behavior penalties
- done/final score

## Episode Mechanics

Each `reset()`:
- loaded a single fixture-backed episode
- populated an inbox with `2-3` tickets
- reset all mutation and progress trackers
- set `step_count = 0`
- set `remaining_steps = 12`

Each `step(action)`:
- validated the action model
- applied the requested transition if legal
- computed rubric progress and behavior adjustments
- returned updated observation, reward, done, and info

An episode ended when:
- the agent called `finalize_resolution`
- the step limit was reached
- a fixture-marked catastrophic unsafe action was taken

## Round 1 Task Set

### Canonical judged tasks

Round 1 shipped exactly three canonical judged tasks:

1. `billing_seat_adjustment` (`easy`)
   - resolve an overbilling complaint by inspecting account and invoice data, issuing the exact credit, updating metadata, and finalizing safely

2. `login_incident_triage` (`medium`)
   - handle a VIP login-failure report during an active authentication incident without taking unsafe remediation shortcuts

3. `suspicious_admin_request` (`hard`)
   - detect a likely account-takeover scenario, inspect verification/security records, escalate to security, and refuse unsafe fulfillment

### Extended demo pack

The Round 1 repo also included an additive extended pack for demos and manual evaluation:
- `tax_exemption_credit_review`
- `api_rate_limit_escalation`
- `admin_role_transfer_verification`

These were useful for richer walkthroughs, but the judged Round 1 core remained the three canonical tasks above.

## Grader Design

Round 1 used a single rubric engine for:
- terminal grading
- incremental dense reward
- debug visibility in `state()`

Each fixture defined weighted rubric checks totaling `1.0`.

Typical rubric categories:
- correct ticket selection
- required evidence inspected
- metadata changes
- safe operational action
- forbidden action avoidance
- reply checklist completion
- correct finalization code

Deterministic grading rules:
- same fixture plus same action sequence always yielded the same score
- scores were clamped into `[0.0, 1.0]`
- no LLM judge, embedding similarity, or fuzzy text scoring

## Round 1 Reward Design

The Round 1 dense reward formula was:

```text
reward = (current_rubric_progress - previous_rubric_progress) + behavior_adjustments
```

Behavior penalties applied for:
- invalid action payloads
- repeated irrelevant inspection
- repeated no-op loops
- unnecessary destructive mutations
- task-defined unsafe actions

Illustrative penalty magnitudes in the Round 1 design:
- invalid payload: `-0.05`
- repeated irrelevant inspect: `-0.02`
- repeated loop/no-op: `-0.03`
- unsafe destructive action: `-0.10`

The main Round 1 design goal was that:
- progress should be visible before finalization
- partially informed work should score above zero
- harmful shortcuts should score worse than cautious escalation
- reward should remain reproducible and auditable

## Reply Grading

Round 1 deliberately kept the reply step structured so it could be scored deterministically.

Recommended reply contract:
- `template_id` selected a controlled response pattern
- `reply_checklist` listed structured intent tokens
- optional `freeform_note` was stored for realism but ignored by the grader

This kept the communication step realistic while avoiding free-text semantic judging.

## Baseline Inference Design

Round 1 required a root `inference.py` script that:
- used the OpenAI Python client
- supported the hackathon-preferred path:
  - `HF_TOKEN`
  - `API_BASE_URL=https://router.huggingface.co/v1`
  - `MODEL_NAME`
- optionally supported compatibility fallback with `OPENAI_API_KEY`
- ran all three canonical tasks in fixed order
- printed per-task scores and the overall mean

Round 1 reproducibility constraints:
- deterministic task order
- deterministic environment seeds
- default `temperature=0`
- no external retrieval
- machine-readable final scores

## Packaging and Deployment

The Round 1 deployment targets were:
- local Docker
- OpenEnv validation
- Hugging Face Spaces

Key packaging requirements:
- Python `3.11`
- lightweight dependencies
- service exposed on port `7860`
- root `Dockerfile`
- `openenv.yaml`

Expected Hugging Face Space settings:
- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=<HF-routable-model>`
- secret `HF_TOKEN=<hugging-face-token>`

## Testing Strategy

Round 1 tests were expected to verify:
- `reset()` initialized a clean episode with `2-3` tickets
- `state()` exposed internal mutations without leaking grader truth in observation
- each canonical task returned deterministic scores in `[0.0, 1.0]`
- repeated irrelevant inspection reduced reward
- forbidden actions reduced score and optionally terminated
- finalizing early without required evidence could not achieve full score
- same seed plus same action sequence yielded identical results
- Docker container booted and served the environment
- `inference.py` ran against the environment contract without shape errors

Oracle paths were also expected for each task:
- one near-perfect sequence per canonical task
- at least one negative path per task

## Round 1 Acceptance Criteria

The Round 1 environment was considered complete when:
- OpenEnv validation passed
- the `3` canonical tasks were implemented and graded deterministically
- the reward function gave non-trivial partial progress
- `inference.py` ran end to end with reproducible outputs
- Docker built locally and the service responded correctly
- the README and deployment artifacts were aligned with actual behavior
- the hackathon submission path was validated with `HF_TOKEN` plus the Hugging Face router

## Round 1 Historical Note

This archived file is intentionally preserved because the current [design_doc.md](design_doc.md) is the Round 2 source of truth. Keeping the Round 1 design separately makes the project history easier to understand for judges, collaborators, and future coding sessions.
