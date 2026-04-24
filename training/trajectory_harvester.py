"""TrajectoryHarvester — collects (prompt, action, score) triples from benchmark runs.

Saves winning trajectories (score >= 0.7) and failing trajectories (score < 0.3)
to separate JSONL files for downstream DPO pair generation and analysis.

Usage:
    python training/trajectory_harvester.py --tasks all --seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI


WIN_THRESHOLD = 0.7
FAIL_THRESHOLD = 0.3
DEFAULT_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
DEFAULT_API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "hf_dummy")

SUPPORT_TASKS = [
    "billing_seat_adjustment",
    "login_incident_triage",
    "suspicious_admin_request",
    "customer_escalation_chain",
    "multi_tier_billing_dispute",
    "data_breach_response_lifecycle",
    "contract_renewal_negotiation",
    "service_reinstatement_review",
    "api_partner_access_audit",
]


def run_episode(client: OpenAI, env_url: str, task_id: str, seed: int) -> dict[str, Any]:
    """Run one episode and return trajectory dict."""
    import requests

    resp = requests.post(f"{env_url}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    obs = resp.json()

    trajectory: list[dict[str, Any]] = []
    final_score = 0.0
    step = 0
    max_steps = 30

    while step < max_steps:
        prompt = _build_prompt(obs)
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = completion.choices[0].message.content or ""
        action = _parse_action(raw)

        trajectory.append({"step": step, "prompt": prompt, "raw_output": raw, "action": action})

        step_resp = requests.post(f"{env_url}/step", json={"action": action}, timeout=30)
        obs = step_resp.json()
        step += 1

        if obs.get("done"):
            final_score = float(obs.get("reward", 0.0))
            break

    state_resp = requests.get(f"{env_url}/state", timeout=10)
    state = state_resp.json()
    final_score = float(state.get("rubric_progress", final_score))

    return {
        "task_id": task_id,
        "seed": seed,
        "final_score": final_score,
        "steps": step,
        "trajectory": trajectory,
        "outcome": "win" if final_score >= WIN_THRESHOLD else ("fail" if final_score < FAIL_THRESHOLD else "partial"),
    }


def _build_prompt(obs: dict[str, Any]) -> str:
    return (
        f"Task: {obs.get('task_brief', '')}\n"
        f"Inbox: {json.dumps(obs.get('inbox', []), indent=2)}\n"
        f"Available records: {obs.get('available_record_ids', [])}\n"
        f"Step: {obs.get('step_count', 0)} / {obs.get('step_count', 0) + obs.get('remaining_steps', 0)}\n"
        "Reply with a JSON action object."
    )


def _parse_action(raw: str) -> dict[str, Any]:
    import re
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"action_type": "search_kb", "query": "help"}


def harvest(
    env_url: str,
    tasks: list[str],
    seeds: int,
    output_dir: Path,
    client: OpenAI,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"harvest_{int(time.time())}"
    wins_path = output_dir / f"{run_id}_wins.jsonl"
    fails_path = output_dir / f"{run_id}_fails.jsonl"
    all_path = output_dir / f"{run_id}_all.jsonl"

    print(f"Harvesting {len(tasks)} tasks x {seeds} seeds = {len(tasks) * seeds} episodes")
    print(f"Output: {output_dir}")

    totals = {"win": 0, "fail": 0, "partial": 0}

    with wins_path.open("w") as wf, fails_path.open("w") as ff, all_path.open("w") as af:
        for task_id in tasks:
            for seed in range(seeds):
                try:
                    result = run_episode(client, env_url, task_id, seed)
                    af.write(json.dumps(result) + "\n")
                    totals[result["outcome"]] += 1
                    if result["outcome"] == "win":
                        wf.write(json.dumps(result) + "\n")
                    elif result["outcome"] == "fail":
                        ff.write(json.dumps(result) + "\n")
                    print(
                        f"  {task_id} seed={seed}: score={result['final_score']:.3f} [{result['outcome']}]"
                    )
                except Exception as exc:
                    print(f"  ERROR {task_id} seed={seed}: {exc}")

    print(f"\nDone. Wins: {totals['win']}  Partial: {totals['partial']}  Fails: {totals['fail']}")
    print(f"Wins file: {wins_path}")
    print(f"Fails file: {fails_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest trajectories from AegisDesk benchmark")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--tasks", default="all", help="Comma-separated task IDs or 'all'")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output-dir", default="training/data")
    args = parser.parse_args()

    tasks = SUPPORT_TASKS if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_API_BASE)

    harvest(
        env_url=args.env_url,
        tasks=tasks,
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        client=client,
    )


if __name__ == "__main__":
    main()
