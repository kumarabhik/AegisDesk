"""Deterministic rubric grader for support_ops_env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from ..models import DraftReplyState, RubricCheckResult, RubricRule, SupportState, TaskFixture
except ImportError:
    from models import DraftReplyState, RubricCheckResult, RubricRule, SupportState, TaskFixture


@dataclass
class RubricEvaluation:
    """Rubric evaluation bundle."""

    progress: float
    breakdown: list[RubricCheckResult]


class RubricEngine:
    """Evaluate task rubrics from environment state."""

    def evaluate(
        self,
        fixture: TaskFixture,
        state: SupportState,
        ticket_lookup: dict[str, dict[str, Any]],
    ) -> RubricEvaluation:
        breakdown: list[RubricCheckResult] = []
        total = 0.0
        for rule in fixture.rubric:
            score, details = self._score_rule(rule, state, ticket_lookup)
            weighted_score = round(score * rule.weight, 4)
            total += weighted_score
            breakdown.append(
                RubricCheckResult(
                    check_id=rule.check_id,
                    label=rule.label,
                    weight=rule.weight,
                    score=round(score, 4),
                    weighted_score=weighted_score,
                    details=details,
                )
            )
        return RubricEvaluation(progress=round(min(total, 1.0), 4), breakdown=breakdown)

    def _score_rule(
        self,
        rule: RubricRule,
        state: SupportState,
        ticket_lookup: dict[str, dict[str, Any]],
    ) -> tuple[float, str]:
        params = rule.params
        kind = rule.kind

        if kind == "selected_ticket":
            expected = params["ticket_id"]
            return (
                (1.0, "Primary ticket selected")
                if state.selected_ticket_id == expected
                else (0.0, "Primary ticket not yet selected")
            )

        if kind == "viewed_records_all":
            required = params.get("record_ids", [])
            if not required:
                return 1.0, "No records required"
            seen = set(state.records_viewed) | set(state.kb_articles_viewed)
            matched = sum(1 for item in required if item in seen)
            return matched / len(required), f"Viewed {matched}/{len(required)} required records"

        if kind == "ticket_priority":
            ticket_id = params["ticket_id"]
            actual = ticket_lookup[ticket_id]["priority"]
            expected = params["priority"]
            return (
                (1.0, f"Priority set to {expected}")
                if actual == expected
                else (0.0, f"Priority is {actual}, expected {expected}")
            )

        if kind == "ticket_status":
            ticket_id = params["ticket_id"]
            actual = ticket_lookup[ticket_id]["status"]
            expected = params["status"]
            return (
                (1.0, f"Status set to {expected}")
                if actual == expected
                else (0.0, f"Status is {actual}, expected {expected}")
            )

        if kind == "tag_added":
            ticket_id = params["ticket_id"]
            expected = params["tag"]
            tags = ticket_lookup[ticket_id]["tags"]
            return (
                (1.0, f"Tag {expected} present")
                if expected in tags
                else (0.0, f"Missing required tag {expected}")
            )

        if kind == "exact_credit":
            ticket_id = params["ticket_id"]
            amount = float(params["amount"])
            currency = params["currency"]
            for credit in state.credits_applied:
                if (
                    credit.ticket_id == ticket_id
                    and round(credit.amount, 2) == round(amount, 2)
                    and credit.currency == currency
                ):
                    return 1.0, "Exact credit applied"
            return 0.0, "Exact credit not yet applied"

        if kind == "escalation_team":
            expected = params["escalation_team"]
            ticket_id = params.get("ticket_id")
            for escalation in state.escalations:
                team_match = escalation.escalation_team == expected
                ticket_match = (ticket_id is None) or (escalation.ticket_id == ticket_id)
                if team_match and ticket_match:
                    return 1.0, f"Escalated to {expected}"
            return 0.0, f"Escalation to {expected} not found"

        if kind == "reply_template":
            return self._score_reply_template(state.draft_reply, params["template_id"])

        if kind == "reply_contains":
            return self._score_reply_items(state.draft_reply, params.get("checklist", []))

        if kind == "final_resolution_code":
            expected = params["resolution_code"]
            actual_resolution = ticket_lookup["_meta"].get("resolution_code", "")
            return (
                (1.0, f"Finalized with {expected}")
                if actual_resolution == expected
                else (0.0, f"Finalization code is {actual_resolution or 'unset'}")
            )

        if kind == "phase_complete":
            phase = int(params["phase"])
            if phase in state.completed_phases:
                return 1.0, f"Phase {phase} completed"
            return 0.0, f"Phase {phase} not yet completed"

        return 0.0, f"Unknown rubric kind: {kind}"

    def _score_reply_template(
        self, draft: DraftReplyState | None, expected: str
    ) -> tuple[float, str]:
        if draft is None:
            return 0.0, "Reply draft missing"
        if draft.template_id == expected:
            return 1.0, f"Reply template {expected} used"
        return 0.0, f"Reply template is {draft.template_id}, expected {expected}"

    def _score_reply_items(
        self, draft: DraftReplyState | None, expected_items: list[str]
    ) -> tuple[float, str]:
        if not expected_items:
            return 1.0, "No checklist items required"
        if draft is None:
            return 0.0, "Reply draft missing"
        actual = set(draft.reply_checklist)
        matched = sum(1 for item in expected_items if item in actual)
        return matched / len(expected_items), f"Reply includes {matched}/{len(expected_items)} items"
