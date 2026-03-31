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
