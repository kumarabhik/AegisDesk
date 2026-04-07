"""Fixture loading helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

try:
    from ..models import TaskFixture
except ImportError:
    from models import TaskFixture


FIXTURE_DIR = Path(__file__).resolve().parent / "task_data"
CANONICAL_TASK_IDS = [
    "billing_seat_adjustment",
    "login_incident_triage",
    "suspicious_admin_request",
]


@lru_cache(maxsize=1)
def load_all_fixtures() -> dict[str, TaskFixture]:
    """Load and cache all task fixtures."""

    fixtures: dict[str, TaskFixture] = {}
    for path in sorted(FIXTURE_DIR.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        fixture = TaskFixture.model_validate(payload)
        fixtures[fixture.task_id] = fixture
    return fixtures


def get_fixture(task_id: str) -> TaskFixture:
    """Return one fixture by id."""

    fixtures = load_all_fixtures()
    if task_id not in fixtures:
        available = ", ".join(sorted(fixtures))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {available}")
    return fixtures[task_id]


def ordered_task_ids() -> list[str]:
    """Stable ordering for deterministic iteration."""

    fixtures = load_all_fixtures()
    return [task_id for task_id in CANONICAL_TASK_IDS if task_id in fixtures]


def extended_task_ids() -> list[str]:
    """Return the non-judged extended task pack in stable order."""

    fixtures = load_all_fixtures()
    return sorted(task_id for task_id in fixtures if task_id not in CANONICAL_TASK_IDS)


def all_task_ids() -> list[str]:
    """Return the full catalog with canonical tasks first, then the extended pack."""

    return ordered_task_ids() + extended_task_ids()


def task_track(task_id: str) -> str:
    """Classify tasks into the judged core or the optional extended pack."""

    return "core" if task_id in CANONICAL_TASK_IDS else "extended"
