"""DPOPairGenerator — creates (chosen, rejected) trajectory pairs for offline DPO.

Reads winning and failing JSONL files produced by trajectory_harvester.py and
generates DPO pairs where chosen = a winning trajectory step and rejected = a
failing step at the same decision point on the same task.

Usage:
    python training/dpo_pair_generator.py --wins-file training/data/harvest_X_wins.jsonl
                                           --fails-file training/data/harvest_X_fails.jsonl
                                           --output-dir training/data
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dpo_pair(
    win_episode: dict[str, Any],
    fail_episode: dict[str, Any],
    step_idx: int,
) -> dict[str, Any] | None:
    """Build one DPO pair from a winning and failing episode at the same step."""
    win_steps = win_episode.get("trajectory", [])
    fail_steps = fail_episode.get("trajectory", [])

    if step_idx >= len(win_steps) or step_idx >= len(fail_steps):
        return None

    win_step = win_steps[step_idx]
    fail_step = fail_steps[step_idx]

    if win_step.get("prompt") != fail_step.get("prompt"):
        return None

    return {
        "task_id": win_episode["task_id"],
        "step": step_idx,
        "prompt": win_step["prompt"],
        "chosen": win_step.get("raw_output", ""),
        "chosen_action": win_step.get("action", {}),
        "chosen_score": win_episode["final_score"],
        "rejected": fail_step.get("raw_output", ""),
        "rejected_action": fail_step.get("action", {}),
        "rejected_score": fail_episode["final_score"],
    }


def generate_pairs(
    wins: list[dict[str, Any]],
    fails: list[dict[str, Any]],
    max_pairs_per_task: int = 50,
) -> list[dict[str, Any]]:
    """Generate DPO pairs by task, matching wins with fails."""
    by_task_wins: dict[str, list] = defaultdict(list)
    by_task_fails: dict[str, list] = defaultdict(list)

    for ep in wins:
        by_task_wins[ep["task_id"]].append(ep)
    for ep in fails:
        by_task_fails[ep["task_id"]].append(ep)

    pairs: list[dict[str, Any]] = []
    for task_id in sorted(set(by_task_wins) & set(by_task_fails)):
        task_wins = by_task_wins[task_id]
        task_fails = by_task_fails[task_id]
        task_pairs = 0

        for win_ep in task_wins:
            for fail_ep in task_fails:
                if task_pairs >= max_pairs_per_task:
                    break
                min_steps = min(len(win_ep.get("trajectory", [])), len(fail_ep.get("trajectory", [])))
                for step_idx in range(min(min_steps, 6)):
                    pair = build_dpo_pair(win_ep, fail_ep, step_idx)
                    if pair is not None:
                        pairs.append(pair)
                        task_pairs += 1
                        break
            if task_pairs >= max_pairs_per_task:
                break

        print(f"  {task_id}: {task_pairs} pairs generated")

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO pairs from trajectory files")
    parser.add_argument("--wins-file", required=True)
    parser.add_argument("--fails-file", required=True)
    parser.add_argument("--output-dir", default="training/data")
    parser.add_argument("--max-pairs-per-task", type=int, default=50)
    args = parser.parse_args()

    wins = load_jsonl(Path(args.wins_file))
    fails = load_jsonl(Path(args.fails_file))
    print(f"Loaded {len(wins)} wins, {len(fails)} fails")

    pairs = generate_pairs(wins, fails, max_pairs_per_task=args.max_pairs_per_task)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dpo_pairs_{int(time.time())}.jsonl"

    with out_path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
