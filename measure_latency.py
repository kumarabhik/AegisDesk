"""Compare AegisDesk startup and first-hit latency with and without prewarming.

This script launches the local FastAPI app twice:
1. with startup prewarming enabled
2. with startup prewarming disabled

It then measures:
- startup time until `/health` becomes ready
- first `/tasks` latency
- first `/benchmark-card` latency
- first `/reset` latency
- first `/trajectory-report` latency
- repeated `/trajectory-report` latency after caching
- combined time-to-first-reset (startup + first reset)

This makes it easy to quantify the benefit of the startup prewarm path.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

from verify_space import _request_json


DEFAULT_BASE_HOST = "http://127.0.0.1"


def _pick_free_port() -> int:
    """Return a predictable free local port for a temporary benchmark server."""

    for candidate in range(8810, 8899):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _health_ready(base_url: str) -> bool:
    """Return True once the local benchmark server reports healthy."""

    try:
        payload = _request_json(base_url, "/health")
    except Exception:  # noqa: BLE001
        return False
    return payload.get("status") in {"ok", "healthy"}


def _read_process_output(process: subprocess.Popen[str]) -> str:
    """Return any captured process output without raising if unavailable."""

    try:
        output = process.stdout.read() if process.stdout is not None else ""
    except Exception:  # noqa: BLE001
        output = ""
    return output.strip()


def _wait_for_health(
    base_url: str,
    timeout_seconds: int,
    process: subprocess.Popen[str],
) -> float:
    """Wait for server health and return the elapsed startup time in seconds."""

    start = time.perf_counter()
    deadline = start + timeout_seconds
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            output = _read_process_output(process)
            raise RuntimeError(
                f"Server exited before becoming healthy at {base_url}. "
                f"Output:\n{output or '<no output>'}"
            )
        if _health_ready(base_url):
            return time.perf_counter() - start
        time.sleep(0.05)
    output = _read_process_output(process) if process.poll() is not None else ""
    raise RuntimeError(
        f"Server at {base_url} did not become healthy within {timeout_seconds} seconds."
        + (f" Output:\n{output}" if output else "")
    )


def _timed_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return request latency in seconds plus the decoded JSON payload."""

    start = time.perf_counter()
    data = _request_json(base_url, path, method=method, payload=payload)
    return time.perf_counter() - start, data


def _run_once(*, prewarm: bool, timeout_seconds: int, seed: int) -> dict[str, float]:
    """Run one benchmark trial for either the prewarmed or non-prewarmed server."""

    port = _pick_free_port()
    base_url = f"{DEFAULT_BASE_HOST}:{port}"
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["AEGISDESK_PREWARM"] = "1" if prewarm else "0"

    process = subprocess.Popen(
        [sys.executable, "-m", "server.app"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        startup_seconds = _wait_for_health(base_url, timeout_seconds, process)
        tasks_seconds, _ = _timed_request(base_url, "/tasks")
        benchmark_card_seconds, _ = _timed_request(base_url, "/benchmark-card")
        reset_seconds, _ = _timed_request(
            base_url,
            "/reset",
            method="POST",
            payload={"task_id": "billing_seat_adjustment", "seed": seed},
        )
        trajectory_report_seconds, _ = _timed_request(
            base_url,
            f"/trajectory-report?task_id=billing_seat_adjustment&seed={seed}",
        )
        trajectory_report_cached_seconds, _ = _timed_request(
            base_url,
            f"/trajectory-report?task_id=billing_seat_adjustment&seed={seed}",
        )
        return {
            "startup_seconds": startup_seconds,
            "first_tasks_seconds": tasks_seconds,
            "first_benchmark_card_seconds": benchmark_card_seconds,
            "first_reset_seconds": reset_seconds,
            "first_trajectory_report_seconds": trajectory_report_seconds,
            "cached_trajectory_report_seconds": trajectory_report_cached_seconds,
            "time_to_first_reset_seconds": startup_seconds + reset_seconds,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _summarize(runs: list[dict[str, float]]) -> dict[str, float]:
    """Average benchmark metrics across repeated runs."""

    keys = runs[0].keys()
    return {key: statistics.mean(run[key] for run in runs) for key in keys}


def _delta_ms(before_seconds: float, after_seconds: float) -> float:
    """Return the improvement in milliseconds from before to after."""

    return round((before_seconds - after_seconds) * 1000.0, 2)


def _delta_pct(before_seconds: float, after_seconds: float) -> float:
    """Return percent improvement from before to after."""

    if before_seconds <= 0:
        return 0.0
    return round(((before_seconds - after_seconds) / before_seconds) * 100.0, 2)


def measure_latency(*, runs: int, timeout_seconds: int) -> dict[str, Any]:
    """Compare latency with and without startup prewarming."""

    cold_runs = [
        _run_once(prewarm=False, timeout_seconds=timeout_seconds, seed=index + 1)
        for index in range(runs)
    ]
    warm_runs = [
        _run_once(prewarm=True, timeout_seconds=timeout_seconds, seed=index + 1)
        for index in range(runs)
    ]

    cold = _summarize(cold_runs)
    warm = _summarize(warm_runs)

    comparison = {
        "first_tasks_latency_ms_saved": _delta_ms(
            cold["first_tasks_seconds"], warm["first_tasks_seconds"]
        ),
        "first_tasks_latency_pct_saved": _delta_pct(
            cold["first_tasks_seconds"], warm["first_tasks_seconds"]
        ),
        "first_reset_latency_ms_saved": _delta_ms(
            cold["first_reset_seconds"], warm["first_reset_seconds"]
        ),
        "first_reset_latency_pct_saved": _delta_pct(
            cold["first_reset_seconds"], warm["first_reset_seconds"]
        ),
        "benchmark_card_latency_ms_saved": _delta_ms(
            cold["first_benchmark_card_seconds"], warm["first_benchmark_card_seconds"]
        ),
        "benchmark_card_latency_pct_saved": _delta_pct(
            cold["first_benchmark_card_seconds"], warm["first_benchmark_card_seconds"]
        ),
        "first_trajectory_report_ms_saved": _delta_ms(
            cold["first_trajectory_report_seconds"], warm["first_trajectory_report_seconds"]
        ),
        "first_trajectory_report_pct_saved": _delta_pct(
            cold["first_trajectory_report_seconds"], warm["first_trajectory_report_seconds"]
        ),
        "cached_trajectory_report_ms_saved": _delta_ms(
            cold["cached_trajectory_report_seconds"], warm["cached_trajectory_report_seconds"]
        ),
        "cached_trajectory_report_pct_saved": _delta_pct(
            cold["cached_trajectory_report_seconds"], warm["cached_trajectory_report_seconds"]
        ),
        "startup_overhead_ms": round(
            (warm["startup_seconds"] - cold["startup_seconds"]) * 1000.0, 2
        ),
        "startup_overhead_pct": round(
            ((warm["startup_seconds"] - cold["startup_seconds"]) / cold["startup_seconds"])
            * 100.0,
            2,
        )
        if cold["startup_seconds"] > 0
        else 0.0,
        "time_to_first_reset_ms_saved": _delta_ms(
            cold["time_to_first_reset_seconds"], warm["time_to_first_reset_seconds"]
        ),
        "time_to_first_reset_pct_saved": _delta_pct(
            cold["time_to_first_reset_seconds"], warm["time_to_first_reset_seconds"]
        ),
    }

    return {
        "runs": runs,
        "cold_start_mode": cold,
        "prewarmed_mode": warm,
        "comparison": comparison,
        "notes": [
            "Positive *_ms_saved values mean prewarming reduced the measured latency.",
            "The benchmark-card and trajectory-report probes measure the new cached read-only endpoints.",
            "Positive startup_overhead_ms means startup took longer because work was shifted earlier.",
            "time_to_first_reset combines startup time and the first /reset call.",
        ],
    }


def main() -> int:
    """Parse CLI args, run the benchmark, and print JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark trials per mode.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Seconds to wait for each temporary server to become healthy.",
    )
    args = parser.parse_args()

    try:
        summary = measure_latency(
            runs=args.runs,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
