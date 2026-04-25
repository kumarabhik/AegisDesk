"""List and optionally execute the 10-step strongest-submission path for AegisDesk."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Stage:
    number: int
    slug: str
    title: str
    kind: str
    description: str
    commands: list[list[str]]
    expected_artifacts: list[str]


def build_stages(env_url: str, model_name: str, results_path: str) -> list[Stage]:
    return [
        Stage(
            number=1,
            slug="benchmark-verify",
            title="Verify benchmark contract",
            kind="local",
            description="Confirm tests, OpenEnv validation, and oracle coverage on the judged benchmark.",
            commands=[
                [sys.executable, "-m", "pytest", "-q"],
                ["openenv", "validate"],
                [sys.executable, "oracle_demo.py", "--pack", "benchmark", "--seed", "11"],
            ],
            expected_artifacts=[],
        ),
        Stage(
            number=2,
            slug="dataset-source-verify",
            title="Verify external dataset reachability",
            kind="local",
            description="Check that the real external data sources are reachable before rebuilding corpora.",
            commands=[
                [sys.executable, "scripts/fetch_real_datasets.py", "--verify-only"],
            ],
            expected_artifacts=[],
        ),
        Stage(
            number=3,
            slug="corpus-build",
            title="Rebuild corpora and RL split",
            kind="local",
            description="Rebuild the SFT corpus, preference corpus, dataset build report, and RL manifest.",
            commands=[
                [sys.executable, "scripts/fetch_real_datasets.py"],
            ],
            expected_artifacts=[
                "training/data/support_sft.jsonl",
                "training/data/support_pref.jsonl",
                "training/data/dataset_build_report.json",
                "training/support_rl_manifest.json",
            ],
        ),
        Stage(
            number=4,
            slug="readiness-doctor",
            title="Run training readiness doctor",
            kind="local",
            description="Validate corpus counts, manifest invariants, optional endpoint counts, and local training prerequisites.",
            commands=[
                [
                    sys.executable,
                    "training/check_training_readiness.py",
                    "--env-url",
                    env_url,
                    "--output",
                    "training/readiness_report.json",
                ],
            ],
            expected_artifacts=["training/readiness_report.json"],
        ),
        Stage(
            number=5,
            slug="sft-smoke",
            title="Run SFT smoke",
            kind="gpu",
            description="Use a short SFT pass to validate tokenizer/template behavior and checkpoint persistence.",
            commands=[
                [
                    sys.executable,
                    "training/train_unsloth_sft.py",
                    "--dataset",
                    "training/data/support_sft.jsonl",
                    "--output",
                    "training/outputs/sft-smoke",
                    "--model",
                    model_name,
                    "--epochs",
                    "0.1",
                    "--report-to",
                    "trackio",
                    "--run-name",
                    "aegisdesk-sft-smoke",
                ],
            ],
            expected_artifacts=["training/outputs/sft-smoke"],
        ),
        Stage(
            number=6,
            slug="sft-champion",
            title="Run SFT champion",
            kind="gpu",
            description="Train the main instruction-tuned adapter on the full support SFT corpus.",
            commands=[
                [
                    sys.executable,
                    "training/train_unsloth_sft.py",
                    "--dataset",
                    "training/data/support_sft.jsonl",
                    "--output",
                    "training/outputs/sft-qwen3-8b",
                    "--model",
                    model_name,
                    "--epochs",
                    "1.5",
                    "--report-to",
                    "trackio",
                    "--run-name",
                    "aegisdesk-sft-champion",
                ],
            ],
            expected_artifacts=["training/outputs/sft-qwen3-8b"],
        ),
        Stage(
            number=7,
            slug="dpo-champion",
            title="Run DPO champion",
            kind="gpu",
            description="Train the preference-tuned adapter on the support preference corpus.",
            commands=[
                [
                    sys.executable,
                    "training/train_unsloth_dpo.py",
                    "--dataset",
                    "training/data/support_pref.jsonl",
                    "--output",
                    "training/outputs/dpo-qwen3-8b",
                    "--model",
                    model_name,
                    "--epochs",
                    "1.0",
                    "--report-to",
                    "trackio",
                    "--run-name",
                    "aegisdesk-dpo-champion",
                ],
            ],
            expected_artifacts=["training/outputs/dpo-qwen3-8b"],
        ),
        Stage(
            number=8,
            slug="grpo-stabilize",
            title="Run GRPO stabilize",
            kind="gpu",
            description="Validate online RL on the canonical nine-task training pack only.",
            commands=[
                [
                    "accelerate",
                    "launch",
                    "training/train_grpo_aegisdesk.py",
                    "--phase",
                    "stabilize",
                    "--rl-manifest",
                    "training/support_rl_manifest.json",
                    "--env-url",
                    env_url,
                    "--model",
                    model_name,
                    "--report-to",
                    "trackio",
                    "--run-name",
                    "aegisdesk-grpo-stabilize",
                ],
            ],
            expected_artifacts=[],
        ),
        Stage(
            number=9,
            slug="grpo-champion",
            title="Run GRPO champion",
            kind="gpu",
            description="Train on the canonical pack plus private curriculum variants while keeping the held-out pack excluded.",
            commands=[
                [
                    "accelerate",
                    "launch",
                    "training/train_grpo_aegisdesk.py",
                    "--phase",
                    "champion",
                    "--rl-manifest",
                    "training/support_rl_manifest.json",
                    "--env-url",
                    env_url,
                    "--model",
                    model_name,
                    "--report-to",
                    "trackio",
                    "--run-name",
                    "aegisdesk-grpo-champion",
                ],
            ],
            expected_artifacts=[],
        ),
        Stage(
            number=10,
            slug="benchmark-and-evidence",
            title="Run evaluation and generate evidence",
            kind="local",
            description="Evaluate the baseline/champion path across all judged fixtures and generate the evidence artifacts.",
            commands=[
                [
                    sys.executable,
                    "training/self_improve.py",
                    "--rounds",
                    "1",
                    "--seeds",
                    "3",
                    "--env-url",
                    env_url,
                    "--results-path",
                    results_path,
                ],
                [
                    sys.executable,
                    "training/plot_benchmark_results.py",
                    "--results",
                    results_path,
                ],
            ],
            expected_artifacts=[
                results_path,
                "training/per_track_delta.png",
                "training/canonical_vs_held_out_summary.md",
            ],
        ),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", default="https://i4mgr00t-meta.hf.space")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--results-path", default="training/benchmark_results.json")
    parser.add_argument("--step", type=int, help="Run or print a single step by number")
    parser.add_argument("--list", action="store_true", help="List all numbered steps")
    parser.add_argument("--status", action="store_true", help="Print artifact presence for each step")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    return parser.parse_args(argv)


def _artifact_status(stage: Stage) -> dict[str, bool]:
    return {artifact: (ROOT / artifact).exists() for artifact in stage.expected_artifacts}


def _stage_payload(stage: Stage) -> dict[str, object]:
    payload = asdict(stage)
    payload["artifact_status"] = _artifact_status(stage)
    return payload


def _print_stage(stage: Stage) -> None:
    print(f"Step {stage.number}: {stage.title} [{stage.kind}]")
    print(stage.description)
    for command in stage.commands:
        print(f"  {' '.join(command)}")
    if stage.expected_artifacts:
        print("  expected artifacts:")
        for artifact in stage.expected_artifacts:
            print(f"  - {artifact}")


def _run_stage(stage: Stage, dry_run: bool) -> int:
    _print_stage(stage)
    if dry_run:
        return 0
    for command in stage.commands:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stages = build_stages(args.env_url, args.model, args.results_path)

    if args.step is not None:
        stage = next((item for item in stages if item.number == args.step), None)
        if stage is None:
            raise SystemExit(f"Unknown step number: {args.step}")
        if args.json:
            print(json.dumps(_stage_payload(stage), indent=2))
            return 0
        return _run_stage(stage, args.dry_run)

    if args.status:
        payload = [_stage_payload(stage) for stage in stages]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for stage in stages:
                status = _artifact_status(stage)
                completed = all(status.values()) if status else False
                print(f"Step {stage.number}: {'done' if completed else 'pending'}")
                if status:
                    for artifact, exists in status.items():
                        print(f"  - {artifact}: {'present' if exists else 'missing'}")
        return 0

    if args.list or args.json or args.step is None:
        payload = [_stage_payload(stage) for stage in stages]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for stage in stages:
                _print_stage(stage)
                print()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
