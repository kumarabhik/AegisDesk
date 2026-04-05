"""Launch a local support_ops_env server and verify it in one command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

from verify_space import _request_json
from verify_space import verify_space


DEFAULT_BASE_URL = "http://127.0.0.1:7860"


def _health_ready(base_url: str) -> bool:
    """Return True when the local service reports a healthy status."""

    try:
        payload = _request_json(base_url, "/health")
    except Exception:  # noqa: BLE001
        return False
    return payload.get("status") in {"ok", "healthy"}


def _wait_for_health(base_url: str, timeout_seconds: int) -> bool:
    """Poll the health endpoint until the service is ready or times out."""

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _health_ready(base_url):
            return True
        time.sleep(1)
    return False


def run_local_stack(
    *,
    base_url: str = DEFAULT_BASE_URL,
    port: int = 7860,
    task_id: str = "billing_seat_adjustment",
    seed: int = 1,
    startup_timeout: int = 30,
    keep_running: bool = False,
) -> dict[str, Any]:
    """Start the local API if needed, verify it, and optionally keep it alive."""

    started_server = False
    process: subprocess.Popen[str] | None = None

    if not _health_ready(base_url):
        env = os.environ.copy()
        env["PORT"] = str(port)
        process = subprocess.Popen(
            [sys.executable, "-m", "server.app"],
            env=env,
            text=True,
        )
        started_server = True
        if not _wait_for_health(base_url, startup_timeout):
            raise RuntimeError(
                f"Local server did not become healthy within {startup_timeout} seconds."
            )

    try:
        verification = verify_space(base_url, task_id, seed)
        summary: dict[str, Any] = {
            "base_url": base_url,
            "server_started": started_server,
            "verification": verification,
        }
        if process is not None and keep_running:
            summary["server_pid"] = process.pid
            summary["server_stopped"] = False
        else:
            summary["server_stopped"] = process is not None
        return summary
    finally:
        if process is not None and not keep_running:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main() -> int:
    """Parse arguments, run the helper, and print JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the local server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to use if the helper needs to start the server.",
    )
    parser.add_argument(
        "--task-id",
        default="billing_seat_adjustment",
        help="Task ID used for the verification reset call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deterministic seed used for verification.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=30,
        help="Seconds to wait for the local server to become healthy.",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Leave the started server running after verification.",
    )
    args = parser.parse_args()

    try:
        summary = run_local_stack(
            base_url=args.base_url,
            port=args.port,
            task_id=args.task_id,
            seed=args.seed,
            startup_timeout=args.startup_timeout,
            keep_running=args.keep_running,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
