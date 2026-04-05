"""Run a compact submission-readiness audit for support_ops_env."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

from verify_space import verify_space


DEFAULT_SPACE_URL = "https://i4mgr00t-meta.hf.space"


def _run_command(command: list[str]) -> dict[str, Any]:
    """Run a command and return a compact structured result."""

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "ok": completed.returncode == 0,
    }


def run_audit(space_url: str) -> dict[str, Any]:
    """Run the local and live checks that matter for submission."""

    pytest_result = _run_command([sys.executable, "-m", "pytest"])
    validate_result = _run_command(["openenv", "validate"])
    remote_result = _run_command(["git", "ls-remote", "origin"])
    live_ok = True
    try:
        live_result: dict[str, Any] = verify_space(space_url, "billing_seat_adjustment", 1)
    except Exception as exc:  # noqa: BLE001
        live_ok = False
        live_result = {"base_url": space_url, "error": str(exc)}

    return {
        "space_url": space_url,
        "pytest": pytest_result,
        "openenv_validate": validate_result,
        "remote_head": remote_result,
        "live_verify": {
            "ok": live_ok,
            "result": live_result,
        },
        "overall_ok": all(
            [
                pytest_result["ok"],
                validate_result["ok"],
                remote_result["ok"],
                live_ok,
            ]
        ),
    }


def main() -> int:
    """Parse args and print the audit result as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space-url",
        default=os.getenv("ENV_BASE_URL", DEFAULT_SPACE_URL),
        help="Live Space URL to verify.",
    )
    args = parser.parse_args()

    audit = run_audit(args.space_url)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["overall_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
