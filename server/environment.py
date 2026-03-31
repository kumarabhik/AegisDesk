"""Core environment logic for support_ops_env."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from pydantic import ValidationError

try:
    from ..compat import Environment
    from ..models import (
        ActionTrace,
        ActionType,
        CreditRecord,
        DraftReplyState,
        EscalationRecord,
        FocusPanel,
        MutationRecord,
        SupportAction,
        SupportObservation,
        SupportReward,
        SupportState,
        TaskFixture,
        TicketSummary,
    )
    from .fixtures import get_fixture, ordered_task_ids
    from .grader import RubricEngine
    from .reward import evaluate_behavior
except ImportError:
    from compat import Environment
    from models import (
        ActionTrace,
        ActionType,
        CreditRecord,
        DraftReplyState,
        EscalationRecord,
        FocusPanel,
        MutationRecord,
        SupportAction,
        SupportObservation,
        SupportReward,
        SupportState,
        TaskFixture,
        TicketSummary,
    )
    from server.fixtures import get_fixture, ordered_task_ids
    from server.grader import RubricEngine
    from server.reward import evaluate_behavior


class SupportOpsEnvironment(Environment):
    """Fixture-backed B2B SaaS support operations environment."""

    _task_pointer = 0

    def __init__(self, task_id: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self._configured_task_id = task_id
        self._configured_seed = seed or 0
        self._grader = RubricEngine()
        self._fixture: TaskFixture | None = None
        self._ticket_lookup: dict[str, dict[str, Any]] = {}
        self._record_lookup: dict[str, dict[str, Any]] = {}
        self._kb_lookup: dict[str, dict[str, Any]] = {}
        self._focus_panel: FocusPanel | None = None
        self._last_action_signature: str | None = None
        self._resolution_code: str = ""
        self._action_history: list[ActionTrace] = []
        self._last_info: dict[str, Any] = {}
        self._state = SupportState(episode_id=str(uuid4()), step_count=0)
        self.reset(task_id=task_id, seed=seed)

    @property
    def state(self) -> SupportState:
        """Expose the current full environment state."""

        return self._state.model_copy(deep=True)

    @property
    def last_info(self) -> dict[str, Any]:
        """Structured info emitted by the last step/reset."""

        return deepcopy(self._last_info)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        """Reset the environment to a fresh episode."""

        task_to_load = task_id or self._configured_task_id or self._next_task_id()
        episode_seed = self._configured_seed if seed is None else seed or 0
        self._fixture = get_fixture(task_to_load)
        self._ticket_lookup = {
            ticket.ticket_id: ticket.model_dump(mode="python") for ticket in self._fixture.tickets
        }
        self._record_lookup = {
            record.record_id: record.model_dump(mode="python")
            for record in self._fixture.records
        }
        self._kb_lookup = {
            article.article_id: article.model_dump(mode="python")
            for article in self._fixture.kb_articles
        }
        self._focus_panel = None
        self._last_action_signature = None
        self._resolution_code = ""
        self._action_history = []
        self._state = SupportState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._fixture.task_id,
            seed=episode_seed,
            primary_ticket_id=self._fixture.primary_ticket_id,
            selected_ticket_id=None,
            active_ticket_id=None,
            done=False,
        )
        evaluation = self._grader.evaluate(self._fixture, self._state, self._ticket_lookup_with_meta())
        self._state.rubric_progress = evaluation.progress
        self._state.rubric_breakdown = evaluation.breakdown
        self._state.last_reward = SupportReward(score=evaluation.progress)
        observation = self._build_observation(reward=0.0, done=False)
        self._last_info = {
            "task_id": self._fixture.task_id,
            "reward_model": self._state.last_reward.model_dump(mode="json"),
        }
        return self._apply_transform(observation)

    def step(
        self,
        action: SupportAction | dict[str, Any],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        """Apply one action and return the updated observation."""

        if self._fixture is None:
            raise RuntimeError("Environment must be reset before stepping.")

        if self._state.done:
            self._state.last_action_error = "Episode already completed."
            observation = self._build_observation(reward=0.0, done=True)
            self._last_info = {
                "task_id": self._fixture.task_id,
                "reward_model": self._state.last_reward.model_dump(mode="json"),
            }
            return self._apply_transform(observation)

        action_obj, validation_error = self._coerce_action(action)
        prev_progress = self._state.rubric_progress
        action_error = validation_error
        repeated_irrelevant_record = False
        repeated_signature = False
        action_summary = "Rejected invalid action payload"

        if action_obj is not None:
            repeated_signature = action_obj.signature() == self._last_action_signature
            action_summary, action_error, repeated_irrelevant_record = self._apply_action(action_obj)
            self._last_action_signature = action_obj.signature()
        else:
            action_obj = SupportAction(action_type=ActionType.SEARCH_KB, query="invalid-payload")
            self._last_action_signature = None

        self._state.step_count += 1

        behavior = evaluate_behavior(
            action=action_obj,
            fixture=self._fixture,
            state=self._state,
            action_error=action_error,
            repeated_signature=repeated_signature,
            repeated_irrelevant_record=repeated_irrelevant_record,
        )
        self._state.behavior_penalties.extend(behavior.penalties)
        self._state.unsafe_actions.extend(behavior.unsafe_actions)

        evaluation = self._grader.evaluate(self._fixture, self._state, self._ticket_lookup_with_meta())
        self._state.rubric_progress = evaluation.progress
        self._state.rubric_breakdown = evaluation.breakdown

        total_reward = round(
            (evaluation.progress - prev_progress) + behavior.adjustment,
            4,
        )
        self._state.last_reward = SupportReward(
            score=evaluation.progress,
            rubric_delta=round(evaluation.progress - prev_progress, 4),
            behavior_adjustment=behavior.adjustment,
            penalties=behavior.penalties,
            unsafe_actions=behavior.unsafe_actions,
        )
        self._state.last_action_error = action_error

        if self._state.step_count >= self._fixture.max_steps:
            self._state.done = True
        if behavior.terminate:
            self._state.done = True
        if self._resolution_code:
            self._state.done = True

        if self._state.done:
            self._state.final_score = self._state.rubric_progress

        trace = ActionTrace(
            step_index=self._state.step_count,
            action_type=action_obj.action_type,
            summary=action_summary,
            reward=total_reward,
            error=action_error,
        )
        self._action_history.append(trace)
        self._last_info = {
            "task_id": self._fixture.task_id,
            "reward_model": self._state.last_reward.model_dump(mode="json"),
            "final_score": self._state.final_score,
        }
        return self._apply_transform(
            self._build_observation(reward=total_reward, done=self._state.done)
        )

    def state_snapshot(self) -> SupportState:
        """Method alias for local callers that expect state() style access."""

        return self.state

    def close(self) -> None:
        """No-op close method for API parity."""

        return None

    def _next_task_id(self) -> str:
        task_ids = ordered_task_ids()
        task_id = task_ids[self.__class__._task_pointer % len(task_ids)]
        self.__class__._task_pointer += 1
        return task_id

    def _coerce_action(
        self, action: SupportAction | dict[str, Any]
    ) -> tuple[SupportAction | None, str | None]:
        if isinstance(action, SupportAction):
            return action, None
        try:
            return SupportAction.model_validate(action), None
        except ValidationError as exc:
            return None, str(exc.errors()[0]["msg"])

    def _apply_action(self, action: SupportAction) -> tuple[str, str | None, bool]:
        assert self._fixture is not None

        repeated_irrelevant_record = False

        if action.action_type == ActionType.OPEN_TICKET:
            if action.ticket_id not in self._ticket_lookup:
                return "Failed to open ticket", f"Unknown ticket_id {action.ticket_id}", False
            self._state.selected_ticket_id = action.ticket_id
            self._state.active_ticket_id = action.ticket_id
            ticket = self._ticket_lookup[action.ticket_id]
            self._focus_panel = FocusPanel(
                panel_type="ticket",
                title=ticket["subject"],
                body=ticket["summary"],
                data=deepcopy(ticket),
            )
            return f"Opened ticket {action.ticket_id}", None, False

        if action.action_type == ActionType.INSPECT_RECORD:
            if action.record_id in self._record_lookup:
                record = self._record_lookup[action.record_id]
                if action.record_id in self._state.records_viewed:
                    repeated_irrelevant_record = self._is_irrelevant_record(action.record_id)
                else:
                    self._state.records_viewed.append(action.record_id)
                self._focus_panel = FocusPanel(
                    panel_type=str(record["kind"]),
                    title=record["title"],
                    body=record["body"],
                    data=deepcopy(record),
                )
                return f"Inspected record {action.record_id}", None, repeated_irrelevant_record
            if action.record_id in self._kb_lookup:
                article = self._kb_lookup[action.record_id]
                if action.record_id not in self._state.kb_articles_viewed:
                    self._state.kb_articles_viewed.append(action.record_id)
                self._focus_panel = FocusPanel(
                    panel_type="kb_article",
                    title=article["title"],
                    body=article["content"],
                    data=deepcopy(article),
                )
                return f"Inspected article {action.record_id}", None, False
            return "Failed to inspect record", f"Unknown record_id {action.record_id}", False

        if action.action_type == ActionType.SEARCH_KB:
            matches = []
            query_terms = {
                term.strip().lower() for term in (action.query or "").split() if term.strip()
            }
            for article in self._fixture.kb_articles:
                article_terms = {item.lower() for item in article.keywords}
                score = len(query_terms & article_terms)
                if score > 0 or not query_terms:
                    matches.append(
                        {
                            "article_id": article.article_id,
                            "title": article.title,
                            "summary": article.summary,
                            "score": score,
                        }
                    )
            matches.sort(key=lambda item: (-item["score"], item["article_id"]))
            self._focus_panel = FocusPanel(
                panel_type="kb_search",
                title=f"KB results for: {action.query}",
                body=f"Found {len(matches)} matching article(s).",
                data={"results": matches},
            )
            return f"Searched KB for '{action.query}'", None, False

        if not action.ticket_id or action.ticket_id not in self._ticket_lookup:
            return "Action rejected", f"Unknown ticket_id {action.ticket_id}", False

        ticket = self._ticket_lookup[action.ticket_id]
        self._state.active_ticket_id = action.ticket_id

        if action.action_type == ActionType.SET_PRIORITY:
            ticket["priority"] = action.priority.value
            self._state.ticket_mutations.append(
                MutationRecord(
                    ticket_id=action.ticket_id,
                    field_name="priority",
                    value=action.priority.value,
                )
            )
            return f"Set priority on {action.ticket_id} to {action.priority.value}", None, False

        if action.action_type == ActionType.SET_STATUS:
            ticket["status"] = action.status.value
            self._state.ticket_mutations.append(
                MutationRecord(
                    ticket_id=action.ticket_id,
                    field_name="status",
                    value=action.status.value,
                )
            )
            return f"Set status on {action.ticket_id} to {action.status.value}", None, False

        if action.action_type == ActionType.ADD_TAG:
            if action.tag not in ticket["tags"]:
                ticket["tags"].append(action.tag)
                self._state.ticket_mutations.append(
                    MutationRecord(
                        ticket_id=action.ticket_id,
                        field_name="tag",
                        value=action.tag or "",
                    )
                )
                return f"Added tag {action.tag} to {action.ticket_id}", None, False
            return f"Tag {action.tag} already present on {action.ticket_id}", "Tag already present", False

        if action.action_type == ActionType.APPLY_CREDIT:
            self._state.credits_applied.append(
                CreditRecord(
                    ticket_id=action.ticket_id,
                    amount=float(action.amount or 0.0),
                    currency=action.currency or "USD",
                )
            )
            return (
                f"Applied credit {action.amount:.2f} {action.currency} on {action.ticket_id}",
                None,
                False,
            )

        if action.action_type == ActionType.ESCALATE:
            self._state.escalations.append(
                EscalationRecord(
                    ticket_id=action.ticket_id,
                    escalation_team=action.escalation_team or "",
                )
            )
            return f"Escalated {action.ticket_id} to {action.escalation_team}", None, False

        if action.action_type == ActionType.DRAFT_REPLY:
            self._state.draft_reply = DraftReplyState(
                ticket_id=action.ticket_id,
                template_id=action.template_id or "",
                reply_checklist=sorted(set(action.reply_checklist)),
                freeform_note=action.freeform_note,
            )
            return f"Drafted reply for {action.ticket_id}", None, False

        if action.action_type == ActionType.FINALIZE_RESOLUTION:
            self._resolution_code = action.resolution_code or ""
            return f"Finalized {action.ticket_id} as {self._resolution_code}", None, False

        return "Unsupported action", f"Unsupported action_type {action.action_type.value}", False

    def _is_irrelevant_record(self, record_id: str) -> bool:
        assert self._fixture is not None
        required = set()
        for rule in self._fixture.rubric:
            if rule.kind == "viewed_records_all":
                required.update(rule.params.get("record_ids", []))
        return record_id not in required

    def _ticket_lookup_with_meta(self) -> dict[str, dict[str, Any]]:
        lookup = deepcopy(self._ticket_lookup)
        lookup["_meta"] = {"resolution_code": self._resolution_code}
        return lookup

    def _build_observation(self, reward: float, done: bool) -> SupportObservation:
        assert self._fixture is not None

        inbox = [
            TicketSummary(
                ticket_id=ticket["ticket_id"],
                subject=ticket["subject"],
                from_contact=ticket["from_contact"],
                account_id=ticket["account_id"],
                summary=ticket["summary"],
                priority=ticket["priority"],
                status=ticket["status"],
                tags=list(ticket["tags"]),
            )
            for ticket in self._ticket_lookup.values()
        ]
        return SupportObservation(
            task_brief=self._fixture.task_brief,
            inbox=inbox,
            active_ticket_id=self._state.active_ticket_id,
            focus_panel=self._focus_panel,
            available_record_ids=sorted(
                list(self._record_lookup.keys()) + list(self._kb_lookup.keys())
            ),
            action_history=deepcopy(self._action_history),
            step_count=self._state.step_count,
            remaining_steps=max(self._fixture.max_steps - self._state.step_count, 0),
            last_action_error=self._state.last_action_error,
            reward=reward,
            done=done,
        )
