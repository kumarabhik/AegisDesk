# AegisDesk Results

Generated on April 10, 2026.

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

### Previous baseline (Qwen2.5-7B-Instruct-1M, minimal prompt)

The initial run used a 7B model with a minimal 5-line system prompt. The agent opened
the correct ticket and inspected one record, then looped on penalty steps because it
lacked the action schema and reply requirements needed to complete the workflow.

| Task | Score |
| --- | ---: |
| `billing_seat_adjustment` | `0.28` |
| `login_incident_triage` | `0.28` |
| `suspicious_admin_request` | `0.25` |
| Mean | `0.27` |

### Baseline improvements (April 10, 2026)

Three targeted changes were made to close the gap between achievable and observed scores:

1. **`reply_requirements` added to `SupportObservation`** — the `template_id` and
   `reply_checklist` values required by `draft_reply` are now part of the observation
   returned by every `reset()` and `step()` call. Previously these values were held only
   in the internal fixture and were unreachable by any agent, making ~0.10 rubric weight
   per task effectively unscoreable without cheating.

2. **System prompt rewritten** — the new prompt includes the complete action schema with
   JSON examples, the recommended six-step workflow, and explicit safety rules.
   A 7B model with the old prompt ran out of useful actions after step 2; the new prompt
   teaches the model to inspect all records, apply the correct operational action, use
   the reply requirements, and finalize.

3. **Default model upgraded to `Qwen/Qwen2.5-72B-Instruct`** — the 72B model follows
   structured instructions reliably and can extract numeric values (credit amounts,
   priority levels) from focus panel content, which the 7B model struggled with.

These changes should improve on the measured `0.27` mean baseline, but no fresh
post-change live run is captured in this file yet. Re-run `python inference.py` with
`HF_TOKEN` set to replace this section with measured updated scores.

Command:

```bash
python inference.py
```

Model used (updated default):

```text
Qwen/Qwen2.5-72B-Instruct
```

## Latency Benchmark

Command:

```bash
python measure_latency.py --runs 2
```

Result:

```json
{
  "cold_start_mode": {
    "cached_trajectory_report_seconds": 0.020746949998283526,
    "first_benchmark_card_seconds": 0.01580764999926032,
    "first_reset_seconds": 0.01940669999930833,
    "first_tasks_seconds": 0.30213720000028843,
    "first_trajectory_report_seconds": 0.034839199999623816,
    "startup_seconds": 27.290732949999438,
    "time_to_first_reset_seconds": 27.310139649998746
  },
  "comparison": {
    "benchmark_card_latency_ms_saved": -8.04,
    "benchmark_card_latency_pct_saved": -50.89,
    "cached_trajectory_report_ms_saved": -6.76,
    "cached_trajectory_report_pct_saved": -32.6,
    "first_reset_latency_ms_saved": 8.36,
    "first_reset_latency_pct_saved": 43.09,
    "first_tasks_latency_ms_saved": 274.33,
    "first_tasks_latency_pct_saved": 90.8,
    "first_trajectory_report_ms_saved": 17.63,
    "first_trajectory_report_pct_saved": 50.6,
    "startup_overhead_ms": -16036.56,
    "startup_overhead_pct": -58.76,
    "time_to_first_reset_ms_saved": 16044.92,
    "time_to_first_reset_pct_saved": 58.75
  },
  "notes": [
    "Positive *_ms_saved values mean prewarming reduced the measured latency.",
    "The benchmark-card and trajectory-report probes measure the new cached read-only endpoints.",
    "Positive startup_overhead_ms means startup took longer because work was shifted earlier.",
    "time_to_first_reset combines startup time and the first /reset call."
  ],
  "prewarmed_mode": {
    "cached_trajectory_report_seconds": 0.027510050000273623,
    "first_benchmark_card_seconds": 0.023851750000176253,
    "first_reset_seconds": 0.011045150000427384,
    "first_tasks_seconds": 0.027805399999124347,
    "first_trajectory_report_seconds": 0.01721125000040047,
    "startup_seconds": 11.254177299999355,
    "time_to_first_reset_seconds": 11.265222449999783
  },
  "runs": 2
}
```

Notes:
- Prewarming is enabled through the app lifespan startup path and now also warms deterministic read-only benchmark payloads.
- The biggest measured improvements in this sample were:
  - first `/tasks`: about `274.33 ms` faster
  - first `/trajectory-report`: about `17.63 ms` faster
  - combined `startup + first /reset`: about `16044.92 ms` faster, or `58.75%`
- Two tiny read-only probes (`/benchmark-card` and cached `/trajectory-report`) were slightly slower in this 2-run sample; those differences are small enough to treat as measurement noise unless confirmed with a larger run count.
- The benchmark is local and compares the current app in cold-start mode versus prewarmed mode.
