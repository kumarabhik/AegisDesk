"""Validate that AegisDesk is ready for the strongest-submission training path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.fixtures import benchmark_task_ids, generalization_fixture_ids, showcase_fixture_ids


DATA_DIR = ROOT / "training" / "data"
MANIFEST_PATH = ROOT / "training" / "support_rl_manifest.json"
DATASET_REPORT_PATH = DATA_DIR / "dataset_build_report.json"
SUPPORT_SFT_PATH = DATA_DIR / "support_sft.jsonl"
SUPPORT_PREF_PATH = DATA_DIR / "support_pref.jsonl"

SFT_TARGET_MIN = 15_000
SFT_TARGET_MAX = 20_000
PREF_TARGET_MIN = 5_000

EXPECTED_COUNTS = {
    "core": 3,
    "v2": 6,
    "generalization": 18,
    "showcase": 3,
    "judged_total": 27,
    "surfaced_total": 30,
}

OPTIONAL_LOCAL_MODULES = [
    "torch",
    "transformers",
    "accelerate",
    "unsloth",
    "trl",
    "datasets",
    "matplotlib",
]


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_paths() -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    summary: dict[str, Any] = {
        "files": {},
        "counts": {},
    }
    for path in [SUPPORT_SFT_PATH, SUPPORT_PREF_PATH, MANIFEST_PATH, DATASET_REPORT_PATH]:
        exists = path.exists()
        summary["files"][path.name] = exists
        if not exists:
            errors.append(f"Missing required file: {path}")

    if SUPPORT_SFT_PATH.exists():
        sft_rows = _count_jsonl(SUPPORT_SFT_PATH)
        summary["counts"]["support_sft_rows"] = sft_rows
        if sft_rows < SFT_TARGET_MIN:
            errors.append(f"support_sft.jsonl below target: {sft_rows} < {SFT_TARGET_MIN}")
        if sft_rows > SFT_TARGET_MAX:
            errors.append(f"support_sft.jsonl above target: {sft_rows} > {SFT_TARGET_MAX}")

    if SUPPORT_PREF_PATH.exists():
        pref_rows = _count_jsonl(SUPPORT_PREF_PATH)
        summary["counts"]["support_pref_rows"] = pref_rows
        if pref_rows < PREF_TARGET_MIN:
            errors.append(f"support_pref.jsonl below target: {pref_rows} < {PREF_TARGET_MIN}")

    return errors, summary


def _check_dataset_report() -> tuple[list[str], dict[str, Any]]:
    if not DATASET_REPORT_PATH.exists():
        return [], {}
    payload = _load_json(DATASET_REPORT_PATH)
    errors: list[str] = []
    files = payload.get("files", {})
    summary = {
        "targets": payload.get("targets", {}),
        "files": files,
    }
    if files.get("support_sft.jsonl") != _count_jsonl(SUPPORT_SFT_PATH):
        errors.append("dataset_build_report.json does not match support_sft.jsonl row count")
    if files.get("support_pref.jsonl") != _count_jsonl(SUPPORT_PREF_PATH):
        errors.append("dataset_build_report.json does not match support_pref.jsonl row count")
    return errors, summary


def _check_manifest() -> tuple[list[str], dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return [], {}
    payload = _load_json(MANIFEST_PATH)
    errors: list[str] = []

    canonical_train = payload.get("canonical_train_fixture_ids", [])
    held_out = payload.get("held_out_generalization_fixture_ids", [])
    showcase = payload.get("showcase_fixture_ids", [])
    judged = payload.get("judged_fixture_ids", [])
    allowed_grpo = payload.get("allowed_grpo_fixture_ids", [])
    excluded_from_training = payload.get("excluded_from_training_fixture_ids", [])
    security_slice = payload.get("security_slice_fixture_ids", [])
    core_fixture_ids = payload.get("core_fixture_ids", [])
    v2_fixture_ids = payload.get("v2_fixture_ids", [])

    if len(core_fixture_ids) != EXPECTED_COUNTS["core"]:
        errors.append(f"Manifest core count mismatch: {len(core_fixture_ids)} != 3")
    if len(v2_fixture_ids) != EXPECTED_COUNTS["v2"]:
        errors.append(f"Manifest v2 count mismatch: {len(v2_fixture_ids)} != 6")
    if len(canonical_train) != 9:
        errors.append(f"Manifest canonical training count mismatch: {len(canonical_train)} != 9")
    if len(held_out) != EXPECTED_COUNTS["generalization"]:
        errors.append(f"Manifest held-out count mismatch: {len(held_out)} != 18")
    if len(showcase) != EXPECTED_COUNTS["showcase"]:
        errors.append(f"Manifest showcase count mismatch: {len(showcase)} != 3")
    if len(judged) != EXPECTED_COUNTS["judged_total"]:
        errors.append(f"Manifest judged count mismatch: {len(judged)} != 27")
    if len(set(held_out) & set(allowed_grpo)) != 0:
        errors.append("Held-out generalization fixtures leaked into allowed_grpo_fixture_ids")
    if set(held_out) != set(excluded_from_training):
        errors.append("excluded_from_training_fixture_ids does not match held-out generalization set")
    if len(security_slice) != 6:
        errors.append(f"Manifest security slice count mismatch: {len(security_slice)} != 6")
    if set(held_out) != set(generalization_fixture_ids()):
        errors.append("Manifest held-out fixtures do not match server.fixtures generalization pack")
    if set(showcase) != set(showcase_fixture_ids()):
        errors.append("Manifest showcase fixtures do not match server.fixtures showcase pack")
    if set(judged) != set(benchmark_task_ids()):
        errors.append("Manifest judged fixtures do not match server.fixtures benchmark pack")

    summary = {
        "canonical_train_count": len(canonical_train),
        "held_out_count": len(held_out),
        "showcase_count": len(showcase),
        "judged_count": len(judged),
        "security_slice_count": len(security_slice),
    }
    return errors, summary


def _check_local_modules() -> dict[str, bool]:
    return {module_name: _module_available(module_name) for module_name in OPTIONAL_LOCAL_MODULES}


def _check_env_vars() -> dict[str, bool]:
    return {
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        "MODEL_NAME": bool(os.environ.get("MODEL_NAME")),
        "API_BASE_URL": bool(os.environ.get("API_BASE_URL")),
    }


def _check_env_endpoint(env_url: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    tasks_resp = requests.get(f"{env_url}/tasks", timeout=20)
    benchmark_resp = requests.get(f"{env_url}/benchmark-card", timeout=20)
    tasks_resp.raise_for_status()
    benchmark_resp.raise_for_status()

    tasks_payload = tasks_resp.json()
    benchmark_payload = benchmark_resp.json()
    task_counts = benchmark_payload.get("task_counts", {})
    tasks = tasks_payload.get("tasks", [])
    surfaced_count = len(tasks)
    judged_count = sum(1 for task in tasks if task.get("judged"))

    legacy_tasks_schema = bool(tasks) and "fixture_id" not in tasks[0] and "judged" not in tasks[0]
    legacy_benchmark_card = "extended" in task_counts and "total" in task_counts and "generalization" not in task_counts

    if legacy_tasks_schema:
        errors.append(
            "/tasks endpoint is serving a legacy schema without fixture_id/judged fields; "
            "deploy the current repo before Step 10 evaluation"
        )
    if legacy_benchmark_card:
        errors.append(
            "/benchmark-card endpoint is serving a legacy extended/total taxonomy; "
            "deploy the current repo before Step 10 evaluation"
        )
    if legacy_tasks_schema or legacy_benchmark_card:
        summary = {
            "env_url": env_url,
            "surfaced_count": surfaced_count,
            "judged_count": judged_count,
            "task_counts": task_counts,
            "legacy_tasks_schema": legacy_tasks_schema,
            "legacy_benchmark_card": legacy_benchmark_card,
        }
        return errors, summary

    if surfaced_count != EXPECTED_COUNTS["surfaced_total"]:
        errors.append(f"/tasks surfaced count mismatch: {surfaced_count} != 30")
    if judged_count != EXPECTED_COUNTS["judged_total"]:
        errors.append(f"/tasks judged count mismatch: {judged_count} != 27")
    for key, expected in EXPECTED_COUNTS.items():
        if task_counts.get(key) != expected:
            errors.append(f"/benchmark-card {key} mismatch: {task_counts.get(key)} != {expected}")

    summary = {
        "env_url": env_url,
        "surfaced_count": surfaced_count,
        "judged_count": judged_count,
        "task_counts": task_counts,
        "legacy_tasks_schema": legacy_tasks_schema,
        "legacy_benchmark_card": legacy_benchmark_card,
    }
    return errors, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", help="Optional environment base URL to validate against /tasks and /benchmark-card")
    parser.add_argument("--output", help="Optional JSON output path for the readiness report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    critical_errors: list[str] = []
    advisory_items: list[str] = []

    path_errors, path_summary = _check_paths()
    critical_errors.extend(path_errors)

    report_errors, report_summary = _check_dataset_report()
    critical_errors.extend(report_errors)

    manifest_errors, manifest_summary = _check_manifest()
    critical_errors.extend(manifest_errors)

    env_summary: dict[str, Any] = {}
    if args.env_url:
        endpoint_errors, env_summary = _check_env_endpoint(args.env_url.rstrip("/"))
        critical_errors.extend(endpoint_errors)

    modules = _check_local_modules()
    env_vars = _check_env_vars()
    for module_name, available in modules.items():
        if not available:
            advisory_items.append(f"Local module not installed: {module_name}")
    if not env_vars["HF_TOKEN"]:
        advisory_items.append("HF_TOKEN not set in local environment")
    if not env_vars["MODEL_NAME"]:
        advisory_items.append("MODEL_NAME not set in local environment")

    payload = {
        "status": "ready" if not critical_errors else "not_ready",
        "critical_errors": critical_errors,
        "advisories": advisory_items,
        "paths": path_summary,
        "dataset_report": report_summary,
        "manifest": manifest_summary,
        "local_modules": modules,
        "env_vars": env_vars,
        "env_endpoint": env_summary,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("AegisDesk training readiness")
    print(f"Status: {payload['status']}")
    if path_summary.get("counts"):
        print(f"SFT rows: {path_summary['counts'].get('support_sft_rows', 0)}")
        print(f"Preference rows: {path_summary['counts'].get('support_pref_rows', 0)}")
    print(f"Judged fixtures: {manifest_summary.get('judged_count', 0)}")
    print(f"Held-out fixtures: {manifest_summary.get('held_out_count', 0)}")
    if critical_errors:
        print("\nCritical issues:")
        for issue in critical_errors:
            print(f"- {issue}")
    if advisory_items:
        print("\nAdvisories:")
        for issue in advisory_items:
            print(f"- {issue}")

    return 0 if not critical_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
