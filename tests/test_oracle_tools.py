"""Oracle trajectory tooling tests."""

from oracle_tools import generate_trajectory_report, oracle_task_ids
from server.fixtures import generalization_fixture_ids, ordered_task_ids, showcase_fixture_ids, v2_task_ids


def test_core_task_cycle_order_is_unchanged() -> None:
    assert ordered_task_ids() == [
        "billing_seat_adjustment",
        "login_incident_triage",
        "suspicious_admin_request",
    ]


def test_task_packs_are_available_without_touching_core_order() -> None:
    assert v2_task_ids() == [
        "customer_escalation_chain",
        "multi_tier_billing_dispute",
        "data_breach_response_lifecycle",
        "contract_renewal_negotiation",
        "service_reinstatement_review",
        "api_partner_access_audit",
    ]
    assert generalization_fixture_ids() == [
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
    assert showcase_fixture_ids() == [
        "admin_role_transfer_verification",
        "api_rate_limit_escalation",
        "tax_exemption_credit_review",
    ]
    assert oracle_task_ids("core") == ordered_task_ids()
    assert oracle_task_ids("v2") == v2_task_ids()
    assert oracle_task_ids("benchmark") == ordered_task_ids() + v2_task_ids() + generalization_fixture_ids()
    assert oracle_task_ids("generalization") == generalization_fixture_ids()
    assert oracle_task_ids("showcase") == showcase_fixture_ids()
    assert oracle_task_ids("extended") == showcase_fixture_ids()
    assert oracle_task_ids("all") == ordered_task_ids() + v2_task_ids() + generalization_fixture_ids() + showcase_fixture_ids()


def test_oracle_report_scores_well_for_v2_task() -> None:
    report = generate_trajectory_report("api_partner_access_audit", seed=11)
    assert report["track"] == "v2"
    assert report["final_score"] >= 0.95
    assert report["success"] is True
    assert len(report["steps"]) >= 1


def test_oracle_report_scores_well_for_generalization_fixture() -> None:
    report = generate_trajectory_report("billing_seat_adjustment_v1", seed=11)
    assert report["fixture_id"] == "billing_seat_adjustment_v1"
    assert report["task_id"] == "billing_seat_adjustment"
    assert report["track"] == "generalization"
    assert report["final_score"] >= 0.95
    assert report["success"] is True
    assert len(report["steps"]) >= 1


def test_oracle_report_scores_well_for_showcase_task() -> None:
    report = generate_trajectory_report("api_rate_limit_escalation", seed=11)
    assert report["track"] == "showcase"
    assert report["final_score"] >= 0.95
    assert report["success"] is True
    assert len(report["steps"]) >= 1
