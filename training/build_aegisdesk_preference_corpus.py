"""Build a project-focused DPO corpus by mixing harvested AegisDesk pairs with a base preference set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.dpo_pair_generator import generate_pairs, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wins-file", default=None, help="Path to harvested *_wins.jsonl. Defaults to latest.")
    parser.add_argument("--fails-file", default=None, help="Path to harvested *_fails.jsonl. Defaults beside wins file.")
    parser.add_argument("--base-pref", default="training/data/support_pref.jsonl")
    parser.add_argument("--output", default="training/data/aegisdesk_pref.jsonl")
    parser.add_argument("--max-pairs-per-task", type=int, default=80)
    parser.add_argument("--upsample-project-pairs", type=int, default=4)
    parser.add_argument("--max-base-rows", type=int, default=2500)
    return parser.parse_args()


def _latest_harvest_file(suffix: str) -> Path:
    candidates = sorted(Path("training/data").glob(f"harvest_*_{suffix}.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No training/data/harvest_*_{suffix}.jsonl files found")
    return candidates[-1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_preference_row(pair: dict[str, Any]) -> dict[str, Any]:
    chosen_score = float(pair.get("chosen_score", 0.0) or 0.0)
    rejected_score = float(pair.get("rejected_score", 0.0) or 0.0)
    return {
        "source": "aegisdesk-harvested-dpo",
        "license": "MIT",
        "split_role": "preference_train",
        "task_id": pair.get("task_id"),
        "fixture_id": pair.get("fixture_id"),
        "safety_slice": pair.get("safety_slice", "general"),
        "step": pair.get("step", 0),
        "prompt": pair.get("prompt", ""),
        "chosen": pair.get("chosen", ""),
        "rejected": pair.get("rejected", ""),
        "chosen_action": pair.get("chosen_action", {}),
        "rejected_action": pair.get("rejected_action", {}),
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "score_gap": chosen_score - rejected_score,
    }


def main() -> int:
    args = parse_args()

    wins_file = Path(args.wins_file) if args.wins_file else _latest_harvest_file("wins")
    fails_file = Path(args.fails_file) if args.fails_file else Path(str(wins_file).replace("_wins.jsonl", "_fails.jsonl"))
    if not fails_file.exists():
        raise FileNotFoundError(f"Fails file not found: {fails_file}")

    wins = load_jsonl(wins_file)
    fails = load_jsonl(fails_file)
    project_pairs = generate_pairs(wins, fails, max_pairs_per_task=args.max_pairs_per_task)
    project_rows = [_to_preference_row(pair) for pair in project_pairs]

    base_rows = _read_jsonl(Path(args.base_pref))
    trimmed_base_rows = base_rows[: max(0, args.max_base_rows)]

    output_rows: list[dict[str, Any]] = []
    output_rows.extend(trimmed_base_rows)
    for row in project_rows:
        output_rows.extend([row] * max(1, args.upsample_project_pairs))

    _write_jsonl(Path(args.output), output_rows)
    print(
        f"Built {args.output} with {len(output_rows)} rows "
        f"({len(trimmed_base_rows)} base + {len(project_rows)} project x{args.upsample_project_pairs})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
