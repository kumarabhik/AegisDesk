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

V2_TASK_IDS = [
    "customer_escalation_chain",
    "multi_tier_billing_dispute",
    "data_breach_response_lifecycle",
    "contract_renewal_negotiation",
    "service_reinstatement_review",
    "api_partner_access_audit",
]

SHOWCASE_TASK_IDS = [
    "admin_role_transfer_verification",
    "api_rate_limit_escalation",
    "tax_exemption_credit_review",
]

GENERALIZATION_FIXTURE_IDS = [
    "billing_seat_adjustment_v1",
    "billing_seat_adjustment_v2",
    "login_incident_triage_v1",
    "login_incident_triage_v2",
    "suspicious_admin_request_v1",
    "suspicious_admin_request_v2",
    "customer_escalation_chain_v1",
    "customer_escalation_chain_v2",
    "multi_tier_billing_dispute_v1",
    "multi_tier_billing_dispute_v2",
    "data_breach_response_lifecycle_v1",
    "data_breach_response_lifecycle_v2",
    "contract_renewal_negotiation_v1",
    "contract_renewal_negotiation_v2",
    "service_reinstatement_review_v1",
    "service_reinstatement_review_v2",
    "api_partner_access_audit_v1",
    "api_partner_access_audit_v2",
]

LEGACY_EXTENDED_TASK_IDS = SHOWCASE_TASK_IDS

SECURITY_SLICE_FIXTURE_IDS = [
    "suspicious_admin_request",
    "suspicious_admin_request_v1",
    "suspicious_admin_request_v2",
    "api_partner_access_audit",
    "api_partner_access_audit_v1",
    "api_partner_access_audit_v2",
]


@lru_cache(maxsize=1)
def load_all_fixtures() -> dict[str, TaskFixture]:
    """Load and cache all fixtures keyed by fixture_id."""

    fixtures: dict[str, TaskFixture] = {}
    for path in sorted(FIXTURE_DIR.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["fixture_id"] = path.stem
        fixture = TaskFixture.model_validate(payload)
        if fixture.fixture_id in fixtures:
            raise KeyError(f"Duplicate fixture_id loaded from task_data: {fixture.fixture_id}")
        fixtures[fixture.fixture_id] = fixture
    return fixtures


@lru_cache(maxsize=1)
def canonical_fixture_ids_by_task() -> dict[str, str]:
    """Map a task family id to its canonical fixture id."""

    return {
        fixture.task_id: fixture.fixture_id
        for fixture in load_all_fixtures().values()
        if fixture.fixture_id == fixture.task_id
    }


def resolve_fixture_id(
    identifier: str | None = None,
    *,
    task_id: str | None = None,
    fixture_id: str | None = None,
) -> str:
    """Resolve a canonical task id or exact fixture id to a concrete fixture id."""

    fixtures = load_all_fixtures()
    candidate = fixture_id or identifier
    if candidate:
        if candidate in fixtures:
            return candidate
        canonical = canonical_fixture_ids_by_task().get(candidate)
        if canonical is not None:
            return canonical
    if task_id:
        if task_id in fixtures:
            return task_id
        canonical = canonical_fixture_ids_by_task().get(task_id)
        if canonical is not None:
            return canonical

    available = ", ".join(sorted(fixtures))
    raise KeyError(f"Unknown fixture identifier. Available fixture_ids: {available}")


def get_fixture(
    identifier: str | None = None,
    *,
    task_id: str | None = None,
    fixture_id: str | None = None,
) -> TaskFixture:
    """Return one fixture by exact fixture id or canonical task id."""

    resolved_fixture_id = resolve_fixture_id(
        identifier,
        task_id=task_id,
        fixture_id=fixture_id,
    )
    return load_all_fixtures()[resolved_fixture_id]


def ordered_task_ids() -> list[str]:
    """Return canonical core fixture ids in stable order."""

    fixtures = load_all_fixtures()
    return [task_id for task_id in CANONICAL_TASK_IDS if task_id in fixtures]


def v2_task_ids() -> list[str]:
    """Return canonical Round 2 fixture ids in stable order."""

    fixtures = load_all_fixtures()
    return [task_id for task_id in V2_TASK_IDS if task_id in fixtures]


def generalization_fixture_ids() -> list[str]:
    """Return the held-out judged generalization fixtures in stable order."""

    fixtures = load_all_fixtures()
    return [fixture_id for fixture_id in GENERALIZATION_FIXTURE_IDS if fixture_id in fixtures]


def showcase_fixture_ids() -> list[str]:
    """Return the legacy showcase fixtures in stable order."""

    fixtures = load_all_fixtures()
    return [task_id for task_id in SHOWCASE_TASK_IDS if task_id in fixtures]


def extended_task_ids() -> list[str]:
    """Backward-compatible alias for the showcase fixture pack."""

    return showcase_fixture_ids()


def canonical_benchmark_task_ids() -> list[str]:
    """Return the 9 canonical judged training tasks in stable order."""

    return ordered_task_ids() + v2_task_ids()


def benchmark_task_ids() -> list[str]:
    """Return the full judged benchmark: canonical tasks plus held-out variants."""

    return canonical_benchmark_task_ids() + generalization_fixture_ids()


def all_task_ids() -> list[str]:
    """Return the public surfaced catalog: judged fixtures plus showcase tasks."""

    return benchmark_task_ids() + showcase_fixture_ids()


def private_variant_fixture_ids() -> list[str]:
    """Return non-surfaced fixture variants reserved for training/curriculum."""

    surfaced = set(all_task_ids())
    return [
        fixture_id
        for fixture_id, fixture in load_all_fixtures().items()
        if fixture_id not in surfaced and fixture_id != fixture.task_id
    ]


def training_curriculum_fixture_ids() -> list[str]:
    """Return the canonical training pack plus non-judged private variants."""

    return canonical_benchmark_task_ids() + private_variant_fixture_ids()


def security_slice_fixture_ids() -> list[str]:
    """Return the benchmark's security-sensitive evaluation slice."""

    fixtures = load_all_fixtures()
    return [fixture_id for fixture_id in SECURITY_SLICE_FIXTURE_IDS if fixture_id in fixtures]


def task_track(identifier: str) -> str:
    """Classify a fixture identifier into the public benchmark taxonomy."""

    if identifier in GENERALIZATION_FIXTURE_IDS:
        return "generalization"
    if identifier in CANONICAL_TASK_IDS:
        return "core"
    if identifier in V2_TASK_IDS:
        return "v2"
    if identifier in SHOWCASE_TASK_IDS:
        return "showcase"
    return "training"
