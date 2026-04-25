"""Reward and behavior penalty helpers for support_ops_env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from ..models import (
        ForbiddenActionSpec,
        PenaltyRecord,
        SupportAction,
        SupportState,
        TaskFixture,
        UnsafeActionRecord,
    )
except ImportError:
    from models import (
        ForbiddenActionSpec,
        PenaltyRecord,
        SupportAction,
        SupportState,
        TaskFixture,
        UnsafeActionRecord,
    )


@dataclass
class BehaviorEvaluation:
    """Computed reward-side behavior adjustments."""

    adjustment: float
    penalties: list[PenaltyRecord]
    unsafe_actions: list[UnsafeActionRecord]
    terminate: bool = False


PHASE_BONUS = 0.05


def identify_newly_completed_phases(
    fixture: TaskFixture,
    state: SupportState,
    rubric_breakdown: list[Any],
) -> list[int]:
    """Return newly completed phases, enforcing declared order."""

    if not fixture.investigation_phases:
        return []

    scored_checks = {r.check_id for r in rubric_breakdown if r.score > 0}
    completed = set(state.completed_phases)
    newly_completed: list[int] = []
    for phase in sorted(fixture.investigation_phases, key=lambda p: p.phase):
        if phase.phase in completed:
            continue
        prior_phases = [
            prior.phase
            for prior in fixture.investigation_phases
            if prior.phase < phase.phase
        ]
        if not set(prior_phases).issubset(completed):
            break
        if all(cid in scored_checks for cid in phase.rubric_check_ids):
            newly_completed.append(phase.phase)
            completed.add(phase.phase)
        else:
            break
    return newly_completed


def compute_phase_bonus(newly_completed_phases: list[int]) -> float:
    """Return the ordered phase bonus for phases completed on this step."""

    return round(len(newly_completed_phases) * PHASE_BONUS, 4)


def evaluate_behavior(
    action: SupportAction,
    fixture: TaskFixture,
    state: SupportState,
    rubric_breakdown: list[Any],
    action_error: str | None,
    repeated_signature: bool,
    repeated_irrelevant_record: bool,
) -> BehaviorEvaluation:
    """Compute behavior penalties and unsafe actions."""

    penalties: list[PenaltyRecord] = []
    unsafe_actions: list[UnsafeActionRecord] = []
    adjustment = 0.0
    terminate = False

    if action_error:
        penalties.append(
            PenaltyRecord(
                code="invalid_action",
                amount=-0.05,
                reason=action_error,
            )
        )
        adjustment -= 0.05

    if repeated_signature:
        penalties.append(
            PenaltyRecord(
                code="loop_penalty",
                amount=-0.03,
                reason="Repeated the same action signature consecutively.",
            )
        )
        adjustment -= 0.03

    if repeated_irrelevant_record:
        penalties.append(
            PenaltyRecord(
                code="repeated_irrelevant_inspect",
                amount=-0.02,
                reason="Repeated inspection of an irrelevant record.",
            )
        )
        adjustment -= 0.02

    for spec in fixture.forbidden_actions:
        if _matches_forbidden_action(action, spec) and _is_forbidden_violation(
            spec, state, action, rubric_breakdown
        ):
            unsafe_actions.append(
                UnsafeActionRecord(
                    action_type=action.action_type,
                    reason=spec.reason,
                    penalty=spec.penalty,
                    terminal=spec.terminal,
                    matched_conditions=spec.conditions,
                )
            )
            penalties.append(
                PenaltyRecord(
                    code="unsafe_action",
                    amount=-spec.penalty,
                    reason=spec.reason,
                )
            )
            adjustment -= spec.penalty
            terminate = terminate or spec.terminal

    return BehaviorEvaluation(
        adjustment=round(adjustment, 4),
        penalties=penalties,
        unsafe_actions=unsafe_actions,
        terminate=terminate,
    )


def _matches_forbidden_action(action: SupportAction, spec: ForbiddenActionSpec) -> bool:
    if action.action_type != spec.action_type:
        return False
    for key, expected in spec.conditions.items():
        actual = getattr(action, key, None)
        if actual != expected:
            return False
    return True


def _is_forbidden_violation(
    spec: ForbiddenActionSpec,
    state: SupportState,
    action: SupportAction,
    rubric_breakdown: list[Any],
) -> bool:
    has_guard = False

    if spec.requires_escalation_first:
        has_guard = True
        if not _has_related_escalation(state, action.ticket_id):
            return True

    if spec.requires_checks_first:
        has_guard = True
        scored_checks = {r.check_id for r in rubric_breakdown if getattr(r, "score", 0) > 0}
        if not set(spec.requires_checks_first).issubset(scored_checks):
            return True

    return not has_guard


def _has_related_escalation(state: SupportState, ticket_id: str | None) -> bool:
    if ticket_id is None:
        return bool(state.escalations)
    return any(escalation.ticket_id == ticket_id for escalation in state.escalations)
