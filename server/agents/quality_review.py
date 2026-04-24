"""QualityReviewAgent — async observer that scores support decisions post-step.

Applies a lightweight secondary rubric covering compliance, tone, and policy
adherence. The review_score contributes 15% to the final dense reward when
the task has multi-agent features enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReviewSignal:
    step: int
    action_type: str
    review_score: float
    flags: list[str] = field(default_factory=list)


class QualityReviewAgent:
    """Scores each action post-step on compliance and policy adherence.

    Rules are deterministic — no LLM required. Each rule maps an action
    pattern to a score adjustment in [-0.10, +0.05].
    """

    WEIGHT = 0.15

    def __init__(self) -> None:
        self._history: list[ReviewSignal] = []

    def review(self, step: int, action: dict[str, Any], state: dict[str, Any]) -> ReviewSignal:
        """Evaluate a completed action and return a ReviewSignal."""
        action_type = action.get("action_type", "")
        score = 1.0
        flags: list[str] = []

        score, flags = self._apply_rules(action_type, action, state, score, flags)
        score = max(0.0, min(1.0, score))

        signal = ReviewSignal(step=step, action_type=action_type, review_score=score, flags=flags)
        self._history.append(signal)
        return signal

    def _apply_rules(
        self,
        action_type: str,
        action: dict[str, Any],
        state: dict[str, Any],
        score: float,
        flags: list[str],
    ) -> tuple[float, list[str]]:
        unsafe_actions = state.get("unsafe_actions", [])

        if action_type == "apply_credit":
            amount = action.get("amount", 0)
            if amount and amount > 400 and not self._escalation_done(state):
                score -= 0.10
                flags.append("high_value_credit_without_escalation")

        if action_type == "finalize_resolution":
            evidence = state.get("records_viewed", [])
            if len(evidence) < 2:
                score -= 0.08
                flags.append("finalized_with_insufficient_evidence")

        if action_type in ("set_priority", "set_status") and self._is_security_task(state):
            if action.get("priority") == "urgent" or action.get("status") == "escalated":
                score += 0.03
                flags.append("correct_security_urgency_signal")

        if unsafe_actions:
            score -= 0.05 * len(unsafe_actions)
            flags.append("unsafe_actions_in_history")

        return score, flags

    def _escalation_done(self, state: dict[str, Any]) -> bool:
        return bool(state.get("escalations"))

    def _is_security_task(self, state: dict[str, Any]) -> bool:
        task_id = state.get("task_id", "")
        return "security" in task_id or "breach" in task_id or "suspicious" in task_id

    def mean_review_score(self) -> float:
        if not self._history:
            return 1.0
        return sum(s.review_score for s in self._history) / len(self._history)

    def reward_contribution(self) -> float:
        """Dense reward contribution: mean_review_score * WEIGHT."""
        return self.mean_review_score() * self.WEIGHT

    def reset(self) -> None:
        self._history.clear()
