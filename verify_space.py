"""Verify a deployed support_ops_env Space over HTTP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_space(base_url: str, task_id: str, seed: int) -> dict[str, Any]:
    """Run a small end-to-end verification against the deployed API."""

    root_payload = _request_json(base_url, "/")
    if root_payload.get("status") != "ok":
        raise RuntimeError(f"Root health check failed: {root_payload}")

    reset_payload = _request_json(
        base_url,
        "/reset",
        method="POST",
        payload={"task_id": task_id, "seed": seed},
    )
    observation = reset_payload.get("observation", {})
    inbox = observation.get("inbox", [])
    if not inbox:
        raise RuntimeError(f"Reset returned no inbox items: {reset_payload}")

    first_ticket_id = inbox[0]["ticket_id"]
    step_payload = _request_json(
        base_url,
        "/step",
        method="POST",
        payload={"action_type": "open_ticket", "ticket_id": first_ticket_id},
    )
    step_observation = step_payload.get("observation", {})
    if step_observation.get("active_ticket_id") != first_ticket_id:
        raise RuntimeError(f"Step failed to open the ticket: {step_payload}")

    state_payload = _request_json(base_url, "/state")
    if state_payload.get("selected_ticket_id") != first_ticket_id:
        raise RuntimeError(f"State did not reflect opened ticket: {state_payload}")

    return {
        "base_url": base_url,
        "root_status": root_payload.get("status"),
        "env_name": root_payload.get("env_name"),
        "task_id": state_payload.get("task_id"),
        "opened_ticket_id": first_ticket_id,
        "step_done": step_payload.get("done"),
        "selected_ticket_id": state_payload.get("selected_ticket_id"),
    }


def main() -> int:
    """Parse arguments and run the verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the running environment service.",
    )
    parser.add_argument(
        "--task-id",
        default="billing_seat_adjustment",
        help="Task ID to use for the reset call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deterministic seed for the reset call.",
    )
    args = parser.parse_args()

    try:
        summary = verify_space(args.base_url, args.task_id, args.seed)
    except urllib.error.HTTPError as exc:
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "error": f"HTTP {exc.code}: {exc.reason}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
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
