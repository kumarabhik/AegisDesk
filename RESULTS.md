# Results Log

## Verified Local State

### Test suite

Command:

```bash
python -m pytest -q
```

Observed result:

```text
57 passed in 10.41s
```

### OpenEnv validation

Command:

```bash
openenv validate
```

Observed result:

```text
[OK] meta: Ready for multi-mode deployment
```

## Benchmark Contract

Current surfaced catalog:
- `core = 3`
- `v2 = 6`
- `generalization = 18`
- `showcase = 3`
- `judged_total = 27`
- `surfaced_total = 30`

Current oracle/demo coverage:
- `/trajectory-report` works for all surfaced fixtures
- `oracle_demo.py` supports `core`, `v2`, `benchmark`, `generalization`, `showcase`, and `all`

## What Is Real Now

- fixture identity is now first-class via `fixture_id`
- `/reset` accepts canonical `task_id` and exact `fixture_id`
- `/tasks` returns `fixture_id`, `task_id`, `track`, `judged`, and `oracle_available`
- held-out judged variants are now part of the public benchmark story
- `python oracle_demo.py --pack benchmark --seed 11` succeeds on all `27` judged fixtures
- `python oracle_demo.py --pack all --seed 11` succeeds on all `30` surfaced fixtures
- `python scripts/fetch_real_datasets.py` now builds:
  - `training/data/support_sft.jsonl` with `15,124` rows
  - `training/data/support_pref.jsonl` with `7,119` rows
  - `training/data/dataset_build_report.json`
  - `training/support_rl_manifest.json`
- the self-improvement path now targets:
  - train on `9` canonical fixtures
  - evaluate on `27` judged fixtures

## What Is Still Missing

The following evidence is still pending a real GPU-backed run:
- `training/benchmark_results.json`
- reward curve PNG
- loss curve PNG
- per-track delta figure
- trained-vs-baseline deltas across the 27 judged fixtures

## Truthful Submission Status

AegisDesk is now much stronger as a benchmark than it was in the earlier 12-surfaced / 9-judged state.

It is accurate to say:
- the environment is credible
- the benchmark contract is stronger
- the held-out generalization story is implemented
- the local validation posture is solid

It is not yet accurate to say:
- the final RL model has already proven strong improvement
- the repo contains the final champion evidence package
