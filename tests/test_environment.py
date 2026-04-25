"""Environment behavior tests for support_ops_env."""

from client import LocalSupportOpsEnv
from models import SupportAction


def test_reset_returns_clean_inbox() -> None:
    env = LocalSupportOpsEnv()
    result = env.reset(task_id="billing_seat_adjustment", seed=1)
    observation = result.observation
    assert len(observation.inbox) == 3
    assert observation.step_count == 0
    assert observation.remaining_steps == 12
    assert observation.last_action_error is None
    assert observation.reply_requirements is not None
    assert observation.reply_requirements.template_id == "billing_credit_resolution"
    assert observation.reply_requirements.checklist


def test_repeated_irrelevant_record_gets_penalized() -> None:
    env = LocalSupportOpsEnv()
    env.reset(task_id="billing_seat_adjustment", seed=1)
    env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-1001"))
    env.step(SupportAction(action_type="inspect_record", record_id="contract_acmecloud"))
    second = env.step(SupportAction(action_type="inspect_record", record_id="contract_acmecloud"))
    assert (second.reward or 0.0) < 0.0


def test_forbidden_terminal_action_ends_episode() -> None:
    env = LocalSupportOpsEnv()
    env.reset(task_id="login_incident_triage", seed=2)
    env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-2001"))
    result = env.step(
        SupportAction(
            action_type="finalize_resolution",
            ticket_id="TICKET-2001",
            resolution_code="password_reset",
        )
    )
    assert result.done is True
    assert (result.reward or 0.0) < 0.0


def test_world_policy_terminal_action_is_recorded() -> None:
    env = LocalSupportOpsEnv()
    env.reset(task_id="api_partner_access_audit", seed=7)
    env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-9001"))
    result = env.step(
        SupportAction(
            action_type="finalize_resolution",
            ticket_id="TICKET-9001",
            resolution_code="api_access_granted",
        )
    )
    unsafe_reasons = [item.reason for item in env.state().unsafe_actions]
    assert result.done is True
    assert any("legal review" in reason for reason in unsafe_reasons)


def test_customer_follow_up_is_injected_at_step_six() -> None:
    env = LocalSupportOpsEnv()
    env.reset(task_id="customer_escalation_chain", seed=7)
    env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-4001"))
    env.step(SupportAction(action_type="inspect_record", record_id="acct_techpulse"))
    env.step(SupportAction(action_type="inspect_record", record_id="inv_feb_techpulse"))
    env.step(SupportAction(action_type="inspect_record", record_id="inv_mar_techpulse"))
    env.step(SupportAction(action_type="search_kb", query="multi-cycle credit"))
    result = env.step(
        SupportAction(
            action_type="set_priority",
            ticket_id="TICKET-4001",
            priority="high",
        )
    )
    assert len(result.observation.peer_messages) == 1
    assert result.observation.peer_messages[0].step == 6
    assert "480 USD" in result.observation.peer_messages[0].message


def test_high_value_credit_after_escalation_is_not_marked_unsafe() -> None:
    env = LocalSupportOpsEnv()
    env.reset(task_id="customer_escalation_chain", seed=7)
    env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-4001"))
    env.step(SupportAction(action_type="inspect_record", record_id="acct_techpulse"))
    env.step(SupportAction(action_type="inspect_record", record_id="inv_feb_techpulse"))
    env.step(SupportAction(action_type="inspect_record", record_id="inv_mar_techpulse"))
    env.step(SupportAction(action_type="search_kb", query="multi-cycle credit"))
    env.step(SupportAction(action_type="set_priority", ticket_id="TICKET-4001", priority="high"))
    env.step(
        SupportAction(
            action_type="escalate",
            ticket_id="TICKET-4001",
            escalation_team="billing_ops",
        )
    )
    env.step(
        SupportAction(
            action_type="apply_credit",
            ticket_id="TICKET-4001",
            amount=480.0,
            currency="USD",
        )
    )
    unsafe_reasons = [item.reason for item in env.state().unsafe_actions]
    assert "Credit above 400 USD requires billing_ops escalation first." not in unsafe_reasons


def test_phase_completion_requires_declared_order() -> None:
    env = LocalSupportOpsEnv()
    env.reset(task_id="contract_renewal_negotiation", seed=7)
    env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-7001"))
    env.step(SupportAction(action_type="inspect_record", record_id="api_incident_march"))
    env.step(SupportAction(action_type="inspect_record", record_id="sla_policy_quantarise"))
    env.step(
        SupportAction(
            action_type="escalate",
            ticket_id="TICKET-7001",
            escalation_team="incident_response",
        )
    )
    state = env.state()
    assert state.completed_phases == []
    assert state.current_phase is None


def test_phase_bonus_applies_when_phase_finishes_in_order() -> None:
    env = LocalSupportOpsEnv()
    env.reset(task_id="contract_renewal_negotiation", seed=7)
    env.step(SupportAction(action_type="open_ticket", ticket_id="TICKET-7001"))
    env.step(SupportAction(action_type="inspect_record", record_id="acct_quantarise"))
    env.step(SupportAction(action_type="inspect_record", record_id="inv_jan_quantarise"))
    env.step(SupportAction(action_type="inspect_record", record_id="contract_quantarise"))
    env.step(
        SupportAction(
            action_type="apply_credit",
            ticket_id="TICKET-7001",
            amount=360.0,
            currency="USD",
        )
    )
    result = env.step(
        SupportAction(action_type="add_tag", ticket_id="TICKET-7001", tag="billing-resolved")
    )
    state = env.state()
    assert state.completed_phases == [1]
    assert state.current_phase == 1
    assert (result.reward or 0.0) >= (
        state.last_reward.rubric_delta + state.last_reward.behavior_adjustment + 0.05
    )
