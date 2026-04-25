"""SelfImproveCLI — end-to-end self-improvement pipeline for AegisDesk.

Runs benchmark -> harvest trajectories -> generate DPO pairs -> fine-tune (GRPO)
-> re-evaluate -> report score delta.

Usage:
    python training/self_improve.py --rounds 3
    python training/self_improve.py --rounds 1 --dry-run
    python training/self_improve.py --rounds 2 --env-url https://i4mgr00t-meta.hf.space
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.fixtures import (
    benchmark_task_ids,
    canonical_benchmark_task_ids,
    generalization_fixture_ids,
    get_fixture,
    security_slice_fixture_ids,
    task_track,
)


DATA_DIR = Path("training/data")
BENCHMARK_RESULTS_PATH = Path("training/benchmark_results.json")
TAUBENCH_SFT_FILE = DATA_DIR / "taubench_sft.jsonl"

TRAINING_FIXTURE_IDS = canonical_benchmark_task_ids()
JUDGED_FIXTURE_IDS = benchmark_task_ids()
HELD_OUT_FIXTURE_IDS = generalization_fixture_ids()
SECURITY_FIXTURE_IDS = security_slice_fixture_ids()


def _make_inference_client() -> tuple[Any, str]:
    from openai import OpenAI

    from inference import resolve_inference_config

    config = resolve_inference_config()
    return OpenAI(api_key=config.api_key, base_url=config.api_base_url), config.model_name


def _safety_slice(task_id: str) -> str:
    if task_id in {"suspicious_admin_request", "api_partner_access_audit"}:
        return "security"
    return "general"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _run_rollout_episode(
    client: Any,
    model_name: str,
    env_url: str,
    fixture_id: str,
    seed: int,
) -> dict[str, Any]:
    from inference import (
        MAX_TOKENS,
        SYSTEM_PROMPT,
        TEMPERATURE,
        build_user_prompt,
        fallback_action,
        format_action_str,
        parse_model_action,
    )
    from models import SupportObservation

    reset_resp = requests.post(
        f"{env_url}/reset",
        json={"fixture_id": fixture_id, "seed": seed},
        timeout=30,
    )
    reset_resp.raise_for_status()
    reset_payload = reset_resp.json()
    observation = SupportObservation.model_validate(reset_payload["observation"])

    history: list[str] = []
    invalid_action_count = 0
    total_steps = max(observation.step_count + observation.remaining_steps, 1)

    for step_index in range(1, total_steps + 1):
        prompt = build_user_prompt(step_index, observation, history)
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
            action = parse_model_action(response_text)
        except Exception:
            action = fallback_action(observation)

        step_resp = requests.post(
            f"{env_url}/step",
            json={"action": action.model_dump(mode="json")},
            timeout=30,
        )
        step_resp.raise_for_status()
        step_payload = step_resp.json()
        observation = SupportObservation.model_validate(step_payload["observation"])
        reward_value = float(step_payload.get("reward", 0.0) or 0.0)
        if observation.last_action_error:
            invalid_action_count += 1
        history.append(
            f"step={step_index} action={format_action_str(action)} "
            f"reward={reward_value:+.2f} error={observation.last_action_error or 'null'}"
        )
        if step_payload.get("done"):
            break

    state_resp = requests.get(f"{env_url}/state", timeout=10)
    state_resp.raise_for_status()
    state = state_resp.json()
    resolved_fixture_id = observation.fixture_id or fixture_id
    task_id = observation.task_id or get_fixture(fixture_id=fixture_id).task_id
    final_score = float(state.get("final_score") or state.get("rubric_progress", 0.0))
    unsafe_actions = state.get("unsafe_actions", []) or []
    return {
        "fixture_id": resolved_fixture_id,
        "task_id": task_id,
        "track": task_track(resolved_fixture_id),
        "safety_slice": _safety_slice(task_id),
        "seed": seed,
        "score": final_score,
        "solved": final_score >= 0.95,
        "steps": len(history),
        "invalid_action_count": invalid_action_count,
        "invalid_action_rate": invalid_action_count / max(len(history), 1),
        "forbidden_action_count": len(unsafe_actions),
        "forbidden_action_hit": bool(unsafe_actions),
    }


def run_sft_warmup(sft_file: Path, dry_run: bool) -> bool:
    """Optional SFT warm-start from tau-bench retail data before GRPO rounds."""

    print("\n[0/4] SFT warm-start from tau-bench retail data...")
    if dry_run:
        print(f"  DRY RUN: would load SFT data from {sft_file}")
        if sft_file.exists():
            with sft_file.open(encoding="utf-8") as handle:
                count = sum(1 for _ in handle)
            print(f"  DRY RUN: {count} SFT rows available")
        return True
    if not sft_file.exists():
        print(f"  SKIP: {sft_file} not found — run scripts/fetch_real_datasets.py first")
        return True
    with sft_file.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    print(f"  Loaded {len(rows)} SFT rows from {sft_file}")
    print("  NOTE: SFT warm-start requires a separate SFTTrainer run (see training/README.md)")
    return True


def run_benchmark(env_url: str, fixture_ids: list[str], seeds: int, dry_run: bool) -> list[dict[str, Any]]:
    """Run the benchmark and return episode-level rollout metrics."""

    print("\n[1/4] Running benchmark evaluation...")
    if dry_run:
        print("  DRY RUN: skipping actual benchmark call")
        dry_rows = []
        for fixture_id in fixture_ids:
            fixture = get_fixture(fixture_id=fixture_id)
            for seed in range(seeds):
                dry_rows.append(
                    {
                        "fixture_id": fixture_id,
                        "task_id": fixture.task_id,
                        "track": task_track(fixture_id),
                        "safety_slice": _safety_slice(fixture.task_id),
                        "seed": seed,
                        "score": 0.27,
                        "solved": False,
                        "steps": 8,
                        "invalid_action_count": 0,
                        "invalid_action_rate": 0.0,
                        "forbidden_action_count": 0,
                        "forbidden_action_hit": False,
                    }
                )
        return dry_rows

    client, model_name = _make_inference_client()
    episodes: list[dict[str, Any]] = []
    for fixture_id in fixture_ids:
        for seed in range(seeds):
            try:
                result = _run_rollout_episode(client, model_name, env_url, fixture_id, seed)
                episodes.append(result)
                print(
                    f"  {fixture_id} seed={seed}: score={result['score']:.3f} "
                    f"steps={result['steps']} invalid={result['invalid_action_count']} "
                    f"forbidden={int(result['forbidden_action_hit'])}"
                )
            except Exception as exc:
                print(f"  WARNING: {fixture_id} seed={seed} failed: {exc}")
    return episodes


def harvest_trajectories(env_url: str, fixture_ids: list[str], seeds: int, dry_run: bool) -> Path | None:
    """Run trajectory harvester and return path to wins JSONL."""

    print("\n[2/4] Harvesting trajectories...")
    if dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dummy_path = DATA_DIR / f"harvest_{int(time.time())}_wins.jsonl"
        with dummy_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "fixture_id": "billing_seat_adjustment",
                        "task_id": "billing_seat_adjustment",
                        "seed": 0,
                        "final_score": 0.85,
                        "steps": 8,
                        "trajectory": [{"step": 0, "prompt": "test", "raw_output": "{}", "action": {}}],
                        "outcome": "win",
                    }
                )
                + "\n"
            )
        print(f"  DRY RUN: wrote dummy wins file to {dummy_path}")
        return dummy_path

    cmd = [
        sys.executable,
        "training/trajectory_harvester.py",
        "--env-url",
        env_url,
        "--tasks",
        ",".join(fixture_ids),
        "--seeds",
        str(seeds),
        "--output-dir",
        str(DATA_DIR),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None

    wins_files = sorted(DATA_DIR.glob("*_wins.jsonl"), key=lambda path: path.stat().st_mtime)
    return wins_files[-1] if wins_files else None


def generate_dpo_pairs(wins_file: Path, dry_run: bool) -> Path | None:
    """Generate DPO pairs and return path to pairs JSONL."""

    print("\n[3/4] Generating DPO pairs...")
    fails_file = Path(str(wins_file).replace("_wins.jsonl", "_fails.jsonl"))

    if dry_run:
        dummy_path = DATA_DIR / f"dpo_pairs_{int(time.time())}.jsonl"
        with dummy_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "task_id": "billing_seat_adjustment",
                        "fixture_id": "billing_seat_adjustment",
                        "safety_slice": "general",
                        "step": 0,
                        "prompt": "test",
                        "chosen": "good action",
                        "chosen_score": 0.85,
                        "rejected": "bad action",
                        "rejected_score": 0.10,
                    }
                )
                + "\n"
            )
        print(f"  DRY RUN: wrote dummy DPO pairs to {dummy_path}")
        return dummy_path

    if not fails_file.exists():
        print(f"  SKIP: no fails file found at {fails_file}")
        return None

    cmd = [
        sys.executable,
        "training/dpo_pair_generator.py",
        "--wins-file",
        str(wins_file),
        "--fails-file",
        str(fails_file),
        "--output-dir",
        str(DATA_DIR),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None

    pairs_files = sorted(DATA_DIR.glob("dpo_pairs_*.jsonl"), key=lambda path: path.stat().st_mtime)
    return pairs_files[-1] if pairs_files else None


def run_training(dry_run: bool) -> bool:
    """Run GRPO training step."""

    print("\n[4/4 (training)] Running GRPO training...")
    if dry_run:
        print("  DRY RUN: skipping actual training call")
        return True

    cmd = [
        sys.executable,
        "training/train_grpo_aegisdesk.py",
        "--all-tasks",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return result.returncode == 0


def _mean_score_by_fixture(episodes: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for episode in episodes:
        grouped.setdefault(episode["fixture_id"], []).append(float(episode["score"]))
    return {fixture_id: _mean(scores) for fixture_id, scores in grouped.items()}


def report_delta(before_episodes: list[dict[str, Any]], after_episodes: list[dict[str, Any]]) -> None:
    before = _mean_score_by_fixture(before_episodes)
    after = _mean_score_by_fixture(after_episodes)
    print("\n=== Score Delta Report ===")
    print(f"{'Fixture':<40} {'Before':>8} {'After':>8} {'Delta':>8}")
    print("-" * 70)
    for fixture_id in sorted(before):
        before_score = before.get(fixture_id, 0.0)
        after_score = after.get(fixture_id, 0.0)
        delta = after_score - before_score
        marker = " +" if delta > 0.02 else (" -" if delta < -0.02 else "  ")
        print(f"{fixture_id:<40} {before_score:>8.3f} {after_score:>8.3f} {delta:>+8.3f}{marker}")
    print("-" * 70)
    print(
        f"{'MEAN':<40} "
        f"{_mean(list(before.values())):>8.3f} "
        f"{_mean(list(after.values())):>8.3f} "
        f"{(_mean(list(after.values())) - _mean(list(before.values()))):>+8.3f}"
    )


def _slice_summary(episodes: list[dict[str, Any]], fixture_ids: list[str]) -> dict[str, float | int]:
    filtered = [episode for episode in episodes if episode["fixture_id"] in fixture_ids]
    if not filtered:
        return {
            "episodes": 0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "solve_rate": 0.0,
            "invalid_action_rate": 0.0,
            "forbidden_action_hit_rate": 0.0,
        }

    scores = [float(episode["score"]) for episode in filtered]
    steps = sum(int(episode["steps"]) for episode in filtered)
    invalid_actions = sum(int(episode["invalid_action_count"]) for episode in filtered)
    forbidden_hits = sum(1 for episode in filtered if episode["forbidden_action_hit"])
    solves = sum(1 for episode in filtered if episode["solved"])
    return {
        "episodes": len(filtered),
        "mean_score": round(_mean(scores), 4),
        "std_score": round(_stddev(scores), 4),
        "solve_rate": round(solves / len(filtered), 4),
        "invalid_action_rate": round(invalid_actions / max(steps, 1), 4),
        "forbidden_action_hit_rate": round(forbidden_hits / len(filtered), 4),
    }


def _track_summary(episodes: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(episode["track"], []).append(episode)
    return {
        track: _slice_summary(track_episodes, [episode["fixture_id"] for episode in track_episodes])
        for track, track_episodes in sorted(grouped.items())
    }


def _fixture_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(episode["fixture_id"], []).append(episode)

    rows: list[dict[str, Any]] = []
    for fixture_id in JUDGED_FIXTURE_IDS:
        fixture = get_fixture(fixture_id=fixture_id)
        fixture_episodes = grouped.get(fixture_id, [])
        scores = [float(episode["score"]) for episode in fixture_episodes]
        steps = sum(int(episode["steps"]) for episode in fixture_episodes)
        invalid_actions = sum(int(episode["invalid_action_count"]) for episode in fixture_episodes)
        forbidden_hits = sum(1 for episode in fixture_episodes if episode["forbidden_action_hit"])
        solves = sum(1 for episode in fixture_episodes if episode["solved"])
        rows.append(
            {
                "fixture_id": fixture_id,
                "task_id": fixture.task_id,
                "track": task_track(fixture_id),
                "difficulty": fixture.difficulty.value,
                "safety_slice": _safety_slice(fixture.task_id),
                "seed_scores": [round(score, 4) for score in scores],
                "mean_score": round(_mean(scores), 4),
                "std_score": round(_stddev(scores), 4),
                "solve_rate": round(solves / len(fixture_episodes), 4) if fixture_episodes else 0.0,
                "invalid_action_rate": round(invalid_actions / max(steps, 1), 4),
                "forbidden_action_hit_rate": round(forbidden_hits / len(fixture_episodes), 4) if fixture_episodes else 0.0,
            }
        )
    return rows


def _summary_payload(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _slice_summary(episodes, [episode["fixture_id"] for episode in episodes]),
        "canonical": _slice_summary(episodes, TRAINING_FIXTURE_IDS),
        "held_out": _slice_summary(episodes, HELD_OUT_FIXTURE_IDS),
        "security_slice": _slice_summary(episodes, SECURITY_FIXTURE_IDS),
        "by_track": _track_summary(episodes),
    }


def write_benchmark_results(
    *,
    before_episodes: list[dict[str, Any]],
    after_episodes: list[dict[str, Any]],
    seeds: int,
    env_url: str,
    output_path: Path,
    dry_run: bool,
) -> None:
    """Write a machine-readable benchmark comparison report."""

    before_fixtures = {row["fixture_id"]: row for row in _fixture_rows(before_episodes)}
    after_fixtures = {row["fixture_id"]: row for row in _fixture_rows(after_episodes)}

    fixture_rows = []
    for fixture_id in JUDGED_FIXTURE_IDS:
        before_row = before_fixtures[fixture_id]
        after_row = after_fixtures[fixture_id]
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "task_id": before_row["task_id"],
                "track": before_row["track"],
                "difficulty": before_row["difficulty"],
                "safety_slice": before_row["safety_slice"],
                "baseline": before_row,
                "champion": after_row,
                "delta": {
                    "mean_score": round(after_row["mean_score"] - before_row["mean_score"], 4),
                    "solve_rate": round(after_row["solve_rate"] - before_row["solve_rate"], 4),
                    "invalid_action_rate": round(
                        after_row["invalid_action_rate"] - before_row["invalid_action_rate"],
                        4,
                    ),
                    "forbidden_action_hit_rate": round(
                        after_row["forbidden_action_hit_rate"] - before_row["forbidden_action_hit_rate"],
                        4,
                    ),
                },
            }
        )

    payload = {
        "benchmark": "AegisDesk",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "env_url": env_url,
        "seeds_per_fixture": seeds,
        "packs": {
            "training": TRAINING_FIXTURE_IDS,
            "judged": JUDGED_FIXTURE_IDS,
            "held_out_generalization": HELD_OUT_FIXTURE_IDS,
            "security_slice": SECURITY_FIXTURE_IDS,
        },
        "summary": {
            "baseline": _summary_payload(before_episodes),
            "champion": _summary_payload(after_episodes),
            "delta": {
                "overall_mean_score": round(
                    _summary_payload(after_episodes)["overall"]["mean_score"]
                    - _summary_payload(before_episodes)["overall"]["mean_score"],
                    4,
                ),
                "canonical_mean_score": round(
                    _summary_payload(after_episodes)["canonical"]["mean_score"]
                    - _summary_payload(before_episodes)["canonical"]["mean_score"],
                    4,
                ),
                "held_out_mean_score": round(
                    _summary_payload(after_episodes)["held_out"]["mean_score"]
                    - _summary_payload(before_episodes)["held_out"]["mean_score"],
                    4,
                ),
                "security_slice_mean_score": round(
                    _summary_payload(after_episodes)["security_slice"]["mean_score"]
                    - _summary_payload(before_episodes)["security_slice"]["mean_score"],
                    4,
                ),
            },
        },
        "fixtures": fixture_rows,
        "episodes": {
            "baseline": before_episodes,
            "champion": after_episodes,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AegisDesk self-improvement pipeline")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--results-path", default=str(BENCHMARK_RESULTS_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("AegisDesk Self-Improvement Pipeline")
    print(f"Rounds: {args.rounds}  Seeds: {args.seeds}  Env: {args.env_url}")
    print(f"Dry run: {args.dry_run}\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    run_sft_warmup(TAUBENCH_SFT_FILE, args.dry_run)
    baseline_episodes = run_benchmark(args.env_url, JUDGED_FIXTURE_IDS, args.seeds, args.dry_run)
    print(f"  Baseline mean: {_mean(list(_mean_score_by_fixture(baseline_episodes).values())):.3f}")

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"Round {round_num}/{args.rounds}")
        print(f"{'=' * 60}")

        wins_file = harvest_trajectories(args.env_url, TRAINING_FIXTURE_IDS, args.seeds, args.dry_run)
        if wins_file is None:
            print("  ERROR: Harvest failed. Stopping.")
            break

        dpo_file = generate_dpo_pairs(wins_file, args.dry_run)
        if dpo_file:
            print(f"  DPO pairs ready: {dpo_file}")
        else:
            print("  WARNING: DPO pair generation skipped or failed.")

        ok = run_training(args.dry_run)
        if not ok:
            print("  WARNING: Training step failed. Continuing to evaluation.")

    final_episodes = run_benchmark(args.env_url, JUDGED_FIXTURE_IDS, args.seeds, args.dry_run)
    report_delta(baseline_episodes, final_episodes)
    if not args.dry_run:
        write_benchmark_results(
            before_episodes=baseline_episodes,
            after_episodes=final_episodes,
            seeds=args.seeds,
            env_url=args.env_url,
            output_path=Path(args.results_path),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
