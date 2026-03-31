"""Determinism and oracle-path grading tests."""

from client import LocalSupportOpsEnv
from models import SupportAction


def run_oracle(task_id: str):
    env = LocalSupportOpsEnv()
    env.reset(task_id=task_id, seed=7)
    if task_id == "billing_seat_adjustment":
        env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-1001"))
        env.step(SupportAction(action_type="inspect_record", record_id="acct_acmecloud"))
        env.step(SupportAction(action_type="inspect_record", record_id="inv_mar_4482"))
        env.step(
            SupportAction(
                action_type="apply_credit",
                ticket_id="TICKET-1001",
                amount=240.0,
                currency="USD",
            )
        )
        env.step(SupportAction(action_type="add_tag", ticket_id="TICKET-1001", tag="credit-approved"))
        env.step(SupportAction(action_type="set_status", ticket_id="TICKET-1001", status="resolved"))
        env.step(
            SupportAction(
                action_type="draft_reply",
                ticket_id="TICKET-1001",
                template_id="billing_credit_resolution",
                reply_checklist=[
                    "acknowledge_billing_error",
                    "confirm_credit_amount",
                    "explain_next_invoice",
                ],
            )
        )
        env.step(
            SupportAction(
                action_type="finalize_resolution",
                ticket_id="TICKET-1001",
                resolution_code="billing_credit_applied",
            )
        )
    elif task_id == "login_incident_triage":
        env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-2001"))
        env.step(SupportAction(action_type="inspect_record", record_id="incident_auth_311"))
        env.step(SupportAction(action_type="inspect_record", record_id="acct_northstar"))
        env.step(SupportAction(action_type="set_priority", ticket_id="TICKET-2001", priority="high"))
        env.step(SupportAction(action_type="set_status", ticket_id="TICKET-2001", status="pending"))
        env.step(SupportAction(action_type="add_tag", ticket_id="TICKET-2001", tag="incident-311"))
        env.step(
            SupportAction(
                action_type="draft_reply",
                ticket_id="TICKET-2001",
                template_id="incident_login_response",
                reply_checklist=[
                    "acknowledge_login_impact",
                    "reference_incident_id",
                    "set_follow_up_expectation",
                ],
            )
        )
        env.step(
            SupportAction(
                action_type="finalize_resolution",
                ticket_id="TICKET-2001",
                resolution_code="incident_pending",
            )
        )
    else:
        env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-3001"))
        env.step(SupportAction(action_type="inspect_record", record_id="approved_contacts_orbit"))
        env.step(SupportAction(action_type="inspect_record", record_id="security_alert_orbit_778"))
        env.step(SupportAction(action_type="inspect_record", record_id="audit_log_orbit"))
        env.step(SupportAction(action_type="set_priority", ticket_id="TICKET-3001", priority="urgent"))
        env.step(SupportAction(action_type="set_status", ticket_id="TICKET-3001", status="escalated"))
        env.step(SupportAction(action_type="escalate", ticket_id="TICKET-3001", escalation_team="security"))
        env.step(
            SupportAction(
                action_type="draft_reply",
                ticket_id="TICKET-3001",
                template_id="security_verification_required",
                reply_checklist=[
                    "refuse_unverified_export",
                    "require_verified_channel",
                    "confirm_security_escalation",
                ],
            )
        )
        env.step(
            SupportAction(
                action_type="finalize_resolution",
                ticket_id="TICKET-3001",
                resolution_code="security_escalated",
            )
        )
    return env.state().final_score


def test_oracle_paths_score_well() -> None:
    assert (run_oracle("billing_seat_adjustment") or 0.0) >= 0.95
    assert (run_oracle("login_incident_triage") or 0.0) >= 0.95
    assert (run_oracle("suspicious_admin_request") or 0.0) >= 0.95


def test_same_seed_same_score() -> None:
    first = run_oracle("billing_seat_adjustment")
    second = run_oracle("billing_seat_adjustment")
    assert first == second
