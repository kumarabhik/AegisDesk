"""Inspect the local environment configuration for support_ops_env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from collections.abc import Sequence


DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_ENV_BASE_URL = "http://127.0.0.1:7860"


def inspect_environment(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Return a compact, non-secret summary of local configuration readiness."""

    values = dict(os.environ) if env is None else env

    hf_mode_missing = [
        name
        for name in ("HF_TOKEN", "MODEL_NAME")
        if not values.get(name)
    ]
    openai_mode_missing = [
        name
        for name in ("OPENAI_API_KEY", "MODEL_NAME")
        if not values.get(name)
    ]

    return {
        "env_base_url": values.get("ENV_BASE_URL", DEFAULT_ENV_BASE_URL),
        "hf_router_mode": {
            "ready": not hf_mode_missing,
            "base_url": values.get("API_BASE_URL", DEFAULT_HF_BASE_URL),
            "missing": hf_mode_missing,
        },
        "openai_compatible_mode": {
            "ready": not openai_mode_missing,
            "base_url": values.get("API_BASE_URL", ""),
            "missing": openai_mode_missing,
        },
        "local_server_expectation": {
            "port": values.get("PORT", "7860"),
            "default_base_url": DEFAULT_ENV_BASE_URL,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print a non-secret JSON summary of environment readiness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args([] if argv is None else list(argv))
    print(json.dumps(inspect_environment(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
