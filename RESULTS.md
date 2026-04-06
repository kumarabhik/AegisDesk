# AegisDesk Results

Generated on April 7, 2026.

Project links:
- GitHub: `https://github.com/kumarabhik/AegisDesk`
- Hugging Face Space: `https://huggingface.co/spaces/I4mGr00T/Meta`
- Live app: `https://i4mgr00t-meta.hf.space`

## Official Pre-Validation Result

Command:

```bash
bash official-validate-submission.sh https://i4mgr00t-meta.hf.space .
```

Result:

```text
========================================
  OpenEnv Submission Validator
========================================
[20:39:40] Repo:     /c/Users/kumar/Downloads/XOXO/meta
[20:39:40] Ping URL: https://i4mgr00t-meta.hf.space

[20:39:40] Step 1/3: Pinging HF Space (https://i4mgr00t-meta.hf.space/reset) ...
[20:39:42] PASSED -- HF Space is live and responds to /reset
[20:39:42] Step 2/3: Running docker build ...
[20:39:42]   Found Dockerfile in /c/Users/kumar/Downloads/XOXO/meta
[20:39:48] PASSED -- Docker build succeeded
[20:39:48] Step 3/3: Running openenv validate ...
[20:39:52] PASSED -- openenv validate passed
[20:39:52]   [OK] meta: Ready for multi-mode deployment

========================================
  All 3/3 checks passed!
  Your submission is ready to submit.
========================================
```

## Verification Summary

- `python -m pytest -q`:

```text
33 passed in 15.62s
```

- `python verify_space.py --base-url https://i4mgr00t-meta.hf.space`:

```json
{
  "base_url": "https://i4mgr00t-meta.hf.space",
  "env_name": "support_ops_env",
  "opened_ticket_id": "TICKET-1001",
  "root_status": "ok",
  "selected_ticket_id": "TICKET-1001",
  "step_done": false,
  "task_id": "billing_seat_adjustment"
}
```

- `python submission_audit.py --space-url https://i4mgr00t-meta.hf.space`:

```json
{
  "overall_ok": true,
  "openenv_validate": {
    "ok": true,
    "stdout": "[OK] meta: Ready for multi-mode deployment"
  },
  "pytest": {
    "ok": true,
    "summary": "33 passed"
  },
  "live_verify": {
    "ok": true
  },
  "space_url": "https://i4mgr00t-meta.hf.space"
}
```

## Live Inference Baseline

Command:

```bash
python inference.py
```

Model used:

```text
Qwen/Qwen2.5-7B-Instruct-1M
```

Rounded per-task scores from the latest live run:

| Task | Score |
| --- | ---: |
| `billing_seat_adjustment` | `0.28` |
| `login_incident_triage` | `0.28` |
| `suspicious_admin_request` | `0.25` |
| Mean | `0.27` |

Observed stdout:

```text
[START] task=billing_seat_adjustment env=support_ops_env model=Qwen/Qwen2.5-7B-Instruct-1M
[END] success=true steps=12 score=0.28 rewards=0.15,0.12,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03
[START] task=login_incident_triage env=support_ops_env model=Qwen/Qwen2.5-7B-Instruct-1M
[END] success=true steps=12 score=0.28 rewards=0.15,0.12,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03
[START] task=suspicious_admin_request env=support_ops_env model=Qwen/Qwen2.5-7B-Instruct-1M
[END] success=true steps=12 score=0.25 rewards=0.15,0.10,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03
```

## Latency Benchmark

Command:

```bash
python measure_latency.py --runs 5
```

Result:

```json
{
  "cold_start_mode": {
    "first_reset_seconds": 0.020340760005638003,
    "first_tasks_seconds": 0.016205099993385375,
    "startup_seconds": 0.05221500000334345,
    "time_to_first_reset_seconds": 0.07255576000898145
  },
  "comparison": {
    "first_reset_latency_ms_saved": 8.78,
    "first_reset_latency_pct_saved": 43.16,
    "first_tasks_latency_ms_saved": 4.8,
    "first_tasks_latency_pct_saved": 29.6,
    "startup_overhead_ms": -43.94,
    "startup_overhead_pct": -84.14,
    "time_to_first_reset_ms_saved": 52.72,
    "time_to_first_reset_pct_saved": 72.66
  },
  "prewarmed_mode": {
    "first_reset_seconds": 0.011561020003864542,
    "first_tasks_seconds": 0.011408619995927438,
    "startup_seconds": 0.008278979995520786,
    "time_to_first_reset_seconds": 0.01983999999938533
  },
  "runs": 5
}
```

Notes:
- Prewarming is enabled through the app lifespan startup path.
- The strongest improvement in this sample is combined `startup + first /reset`, which dropped by about `72.66%`.
- The benchmark is local and compares the current app in cold-start mode versus prewarmed mode.
