---
title: AegisDesk
sdk: docker
app_port: 7860
tags:
  - openenv
---

# AegisDesk

`AegisDesk` is the public-facing name for `support_ops_env`, a real-world OpenEnv benchmark for B2B SaaS support operations. Each episode presents a small inbox of 2-3 tickets. The agent must identify the primary case, inspect the right internal records, take safe operational actions, draft a structured customer reply, and finalize the case.

If you prefer a longer, blog-style explanation of what the project does, how the environment works, and how to run and verify it end to end, read `PROJECT_WALKTHROUGH.md`.
If you want a more professional, judge-facing explanation of the benchmark's purpose and what makes it distinctive, read `SUBMISSION_OVERVIEW.md`.

## Why this environment exists
- It models a genuine human workflow instead of a toy task.
- It evaluates prioritization, policy compliance, and safe escalation.
- It stays deterministic and cheap to run because all task data is fixture-driven.

## Action space
The environment uses one typed `SupportAction` model with these supported actions:
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

Important action fields:
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

## Observation space
Each observation includes:
- `task_brief`
- `inbox`
- `active_ticket_id`
- `focus_panel`
- `available_record_ids`
- `action_history`
- `step_count`
- `remaining_steps`
- `last_action_error`

## Tasks
The initial release ships 3 deterministic tasks:

1. `billing_seat_adjustment` (easy)
- Resolve an overbilling complaint by inspecting account and invoice data, issuing the exact credit, updating metadata, and finalizing safely.

2. `login_incident_triage` (medium)
- Handle a VIP login-failure report during an active authentication incident without taking unsafe remediation shortcuts.

3. `suspicious_admin_request` (hard)
- Detect a likely account-takeover scenario, inspect verification/security records, escalate to security, and refuse unsafe fulfillment.

## Reward design
Reward is dense and deterministic:

```text
reward = rubric_progress_delta + behavior_adjustments
```

Behavior penalties apply for:
- invalid action payloads
- repeated loops
- repeated inspection of irrelevant records
- forbidden unsafe actions

Final task scores are deterministic in `[0.0, 1.0]`.

## Project layout
Key files:
- `models.py`
- `client.py`
- `server/environment.py`
- `server/app.py`
- `server/grader.py`
- `server/reward.py`
- `server/task_data/*.yaml`
- `inference.py`
- `verify_space.py`
- `submission_audit.py`
- `run_local_stack.py`
- `env_doctor.py`
- `.env.example`
- `PROJECT_WALKTHROUGH.md`
- `SUBMISSION_OVERVIEW.md`
- `openenv.yaml`

## Setup
Install dependencies:

```bash
pip install -e .
```

If you want a simple starting point for environment variables, copy values from `.env.example` into your shell or local environment manager. The file does not contain secrets, only placeholders and the expected variable names.

If you want a non-secret check of whether your local inference configuration is ready, run:

```bash
python env_doctor.py
env-doctor
```

Run the server locally:

```bash
python -m server.app
```

Or with uvicorn:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

If you want a one-command local bootstrap that starts the server if needed, waits for health, verifies the API, and then shuts the local process down again, run:

```bash
python run_local_stack.py
run-local-stack
```

## Baseline inference
The root `inference.py` script uses the OpenAI client.
It now emits structured stdout lines tagged as `[START]`, `[STEP]`, and `[END]` for evaluator-friendly parsing.

Hackathon-preferred configuration:
- `HF_TOKEN`
- `MODEL_NAME`
- `API_BASE_URL` (optional, defaults to `https://router.huggingface.co/v1` when `HF_TOKEN` is used)
- `ENV_BASE_URL` (optional if connecting to a running server; otherwise it uses the local in-process environment)

Compatibility fallback configuration:
- `OPENAI_API_KEY`
- `MODEL_NAME`
- `API_BASE_URL`

For local runs with other compatible providers, `inference.py` also accepts these aliases:
- Groq: `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL`
- xAI/Grok-style aliases: `XAI_API_KEY`, `XAI_MODEL`, `XAI_BASE_URL`, `GROK_API_KEY`, `GROK_MODEL`, `GROK_BASE_URL`

This keeps the benchmark compliant with the required OpenAI client while aligning the official submission path with the Hugging Face router.

Hackathon-ready example:

```bash
HF_TOKEN=...
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=<hf-routable-model>
python inference.py
```

Run it with:

```bash
python inference.py
```

## Validation and testing
Recommended checks:

```bash
python -m pytest
openenv validate
python verify_space.py --base-url http://127.0.0.1:7860
verify-space --base-url http://127.0.0.1:7860
python run_local_stack.py
run-local-stack
python env_doctor.py
env-doctor
submission-audit --space-url https://i4mgr00t-meta.hf.space
docker build -t support-ops-env .
docker run -p 7860:7860 support-ops-env
```

Live Space verification:

```bash
python verify_space.py --base-url https://i4mgr00t-meta.hf.space
verify-space --base-url https://i4mgr00t-meta.hf.space
submission-audit --space-url https://i4mgr00t-meta.hf.space
```

If you want a local mirror of the hackathon pre-validation flow, use one of these:

```bash
./validate-submission.sh https://i4mgr00t-meta.hf.space
```

On Windows, prefer the PowerShell version:

```powershell
powershell -ExecutionPolicy Bypass -File .\validate-submission.ps1 -PingUrl https://i4mgr00t-meta.hf.space -RepoDir .
```

## Verification status
Verified in this workspace:
- `python -m pytest` passes with 31 tests
- `openenv validate` passes
- FastAPI smoke checks for `/` and `/reset` pass
- FastAPI health checks for `/` and `/health` pass
- `run-local-stack` provides a one-command local start plus verification path
- `env-doctor` provides a non-secret environment readiness check
- `uv.lock` has been generated
- `inference.py` resolves `HF_TOKEN` plus the Hugging Face router as the preferred submission path
- `inference.py` emits tagged `[START]`, `[STEP]`, and `[END]` stdout lines
- `inference.py` still supports standard OpenAI env vars and compatible Groq/xAI aliases as fallbacks
- `inference.py` has been run successfully against a Groq-compatible backend
- `inference.py` has been run successfully against the Hugging Face router using `HF_TOKEN`
- `docker build -t support-ops-env .` succeeds
- `docker run -p 7860:7860 support-ops-env` succeeds
- live container checks for `/`, `/reset`, `/step`, and `/state` succeed
- `submission-audit --space-url https://i4mgr00t-meta.hf.space` succeeds
- local mirror validator scripts exist at `validate-submission.sh` and `validate-submission.ps1`
- Hugging Face Space deployment is live and verified at `https://i4mgr00t-meta.hf.space/`
- live Space checks for `/`, `/reset`, `/step`, and `/state` succeed

## Baseline scores
Recorded baseline run in this workspace:
- Provider path: OpenAI client against `https://api.groq.com/openai/v1`
- Credential source: `GROQ_API_KEY`
- Model: `llama3-8b-8192`
- `billing_seat_adjustment`: `0.2750`
- `login_incident_triage`: `0.2750`
- `suspicious_admin_request`: `0.2500`
- Overall mean: `0.2667`

This recorded run used the local compatibility fallback rather than the hackathon-preferred HF router path.

Recorded hackathon-path baseline run in this workspace:
- Provider path: OpenAI client against `https://router.huggingface.co/v1`
- Credential source: `HF_TOKEN`
- Model: `Qwen/Qwen2.5-7B-Instruct-1M`
- Environment target: `https://i4mgr00t-meta.hf.space`
- `billing_seat_adjustment`: `0.2750`
- `login_incident_triage`: `0.2750`
- `suspicious_admin_request`: `0.2500`
- Overall mean: `0.2667`

Equivalent compatibility configuration for future runs:

```bash
OPENAI_API_KEY=...
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama3-8b-8192
python inference.py
```

Equivalent hackathon configuration:

```bash
HF_TOKEN=...
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-1M
ENV_BASE_URL=https://i4mgr00t-meta.hf.space
python inference.py
```

## Final submission checklist
- Space URL is live: `https://i4mgr00t-meta.hf.space/`
- `python verify_space.py --base-url https://i4mgr00t-meta.hf.space` succeeds
- `verify-space --base-url https://i4mgr00t-meta.hf.space` succeeds
- `submission-audit --space-url https://i4mgr00t-meta.hf.space` succeeds
- `python inference.py` succeeds with `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`, and `ENV_BASE_URL` set
- `python inference.py` emits `[START]`, `[STEP]`, and `[END]` log lines
- `python -m pytest` passes
- `openenv validate` passes
- the latest Hugging Face Space commit is the intended submission revision
