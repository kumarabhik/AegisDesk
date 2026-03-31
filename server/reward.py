"""Reward and behavior penalty helpers for support_ops_env."""

from __future__ import annotations

from dataclasses import dataclass

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


def evaluate_behavior(
    action: SupportAction,
    fixture: TaskFixture,
    state: SupportState,
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
        if _matches_forbidden_action(action, spec):
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
