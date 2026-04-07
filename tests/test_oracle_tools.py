"""Oracle trajectory tooling tests."""

from oracle_tools import generate_trajectory_report, oracle_task_ids
from server.fixtures import extended_task_ids, ordered_task_ids


def test_core_task_cycle_order_is_unchanged() -> None:
    assert ordered_task_ids() == [
        "billing_seat_adjustment",
        "login_incident_triage",
        "suspicious_admin_request",
    ]


def test_extended_pack_is_available_without_touching_core_order() -> None:
    assert extended_task_ids() == [
        "admin_role_transfer_verification",
        "api_rate_limit_escalation",
        "tax_exemption_credit_review",
    ]
    assert oracle_task_ids("all")[:3] == ordered_task_ids()


def test_oracle_report_scores_well_for_extended_task() -> None:
    report = generate_trajectory_report("api_rate_limit_escalation", seed=11)
    assert report["track"] == "extended"
    assert report["final_score"] >= 0.95
    assert report["success"] is True
    assert len(report["steps"]) >= 1
