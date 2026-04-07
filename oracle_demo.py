"""Run near-perfect oracle trajectories for AegisDesk tasks."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

try:
    from .oracle_tools import generate_trajectory_report, oracle_task_ids, write_report_files
except ImportError:
    from oracle_tools import generate_trajectory_report, oracle_task_ids, write_report_files


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", help="Run a single task id instead of a whole pack.")
    parser.add_argument(
        "--pack",
        choices=("core", "extended", "all"),
        default="core",
        help="Which task pack to run when --task-id is not provided.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed to use.")
    parser.add_argument(
        "--env-url",
        help="Optional remote environment URL. Omit to run against the in-process local environment.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional JSON output path for a single-task run. Multi-task runs append task ids automatically.",
    )
    parser.add_argument(
        "--output-md",
        help="Optional markdown output path for a single-task run. Multi-task runs append task ids automatically.",
    )
    return parser


def derive_output_path(base_path: str | None, task_id: str) -> str | None:
    """Create per-task output paths for multi-task runs."""

    if not base_path:
        return None
    if "{task_id}" in base_path:
        return base_path.format(task_id=task_id)

    if "." not in base_path:
        return f"{base_path}-{task_id}"

    stem, suffix = base_path.rsplit(".", 1)
    return f"{stem}-{task_id}.{suffix}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run oracle trajectories and print a compact summary."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    task_ids = [args.task_id] if args.task_id else oracle_task_ids(args.pack)
    reports = []
    for task_id in task_ids:
        report = generate_trajectory_report(task_id, seed=args.seed, env_url=args.env_url)
        write_report_files(
            report,
            output_json=derive_output_path(args.output_json, task_id),
            output_md=derive_output_path(args.output_md, task_id),
        )
        reports.append(report)

    summary = {
        "pack": args.pack if not args.task_id else "single",
        "seed": args.seed,
        "reports": [
            {
                "task_id": report["task_id"],
                "track": report["track"],
                "difficulty": report["difficulty"],
                "step_count": report["step_count"],
                "final_score": report["final_score"],
                "success": report["success"],
            }
            for report in reports
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
