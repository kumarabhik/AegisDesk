"""CLI coverage for the oracle demo runner."""

from __future__ import annotations

import json

from oracle_demo import main


def test_oracle_demo_accepts_v2_pack(capsys) -> None:
    assert main(["--pack", "v2", "--seed", "11"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pack"] == "v2"
    assert [report["fixture_id"] for report in summary["reports"]] == [
        "customer_escalation_chain",
        "multi_tier_billing_dispute",
        "data_breach_response_lifecycle",
        "contract_renewal_negotiation",
        "service_reinstatement_review",
        "api_partner_access_audit",
    ]
    assert all(report["track"] == "v2" for report in summary["reports"])


def test_oracle_demo_accepts_generalization_pack(capsys) -> None:
    assert main(["--pack", "generalization", "--seed", "11"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pack"] == "generalization"
    assert [report["fixture_id"] for report in summary["reports"]] == [
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
    assert all(report["track"] == "generalization" for report in summary["reports"])


def test_oracle_demo_accepts_showcase_pack(capsys) -> None:
    assert main(["--pack", "showcase", "--seed", "11"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pack"] == "showcase"
    assert [report["fixture_id"] for report in summary["reports"]] == [
        "admin_role_transfer_verification",
        "api_rate_limit_escalation",
        "tax_exemption_credit_review",
    ]
    assert all(report["track"] == "showcase" for report in summary["reports"])
