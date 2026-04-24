"""Typed models for the support_ops_env OpenEnv package."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from .compat import Action, Observation, State
except ImportError:
    from compat import Action, Observation, State


class Difficulty(str, Enum):
    EASY = "easy"
    EASY_MEDIUM = "easy_medium"
    MEDIUM = "medium"
    MEDIUM_HARD = "medium_hard"
    HARD = "hard"


class ActionType(str, Enum):
    OPEN_TICKET = "open_ticket"
    INSPECT_RECORD = "inspect_record"
    SEARCH_KB = "search_kb"
    SET_PRIORITY = "set_priority"
    SET_STATUS = "set_status"
    ADD_TAG = "add_tag"
    APPLY_CREDIT = "apply_credit"
    ESCALATE = "escalate"
    DRAFT_REPLY = "draft_reply"
    FINALIZE_RESOLUTION = "finalize_resolution"


class TicketPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(str, Enum):
    OPEN = "open"
    PENDING = "pending"
    WAITING_ON_CUSTOMER = "waiting_on_customer"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class RecordKind(str, Enum):
    ACCOUNT = "account"
    INVOICE = "invoice"
    CONTRACT = "contract"
    INCIDENT = "incident"
    AUDIT_LOG = "audit_log"
    APPROVED_CONTACTS = "approved_contacts"
    SECURITY_ALERT = "security_alert"
    KB_ARTICLE = "kb_article"


class PeerMessage(BaseModel):
    """A message injected by a peer agent (CustomerSimAgent) mid-episode."""

    model_config = ConfigDict(extra="forbid")

    step: int
    from_role: str
    message: str


class WorldContext(BaseModel):
    """Compact world state surfaced in agent observations."""

    model_config = ConfigDict(extra="forbid")

    active_incidents: list[str] = Field(default_factory=list)
    policy_window_name: Optional[str] = None
    policy_window_description: Optional[str] = None
    policy_window_active: bool = False
    region: str = "global"
    account_health_index: float = 1.0


class InvestigationPhase(BaseModel):
    """A named phase in a long-horizon task, with associated rubric check IDs."""

    model_config = ConfigDict(extra="ignore")

    phase: int
    label: str
    description: str = ""
    rubric_check_ids: list[str] = Field(default_factory=list)


class PeerInjectSpec(BaseModel):
    """Fixture spec for a peer agent message injection."""

    model_config = ConfigDict(extra="ignore")

    at_step: int
    from_: str = Field("customer", alias="from")
    message: str
    inject_into_observation: bool = True

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TicketSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    subject: str
    from_contact: str
    account_id: str
    summary: str
    priority: TicketPriority
    status: TicketStatus
    tags: list[str] = Field(default_factory=list)


class FocusPanel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    panel_type: str
    title: str
    body: str
    data: dict[str, Any] = Field(default_factory=dict)


class ActionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    action_type: ActionType
    summary: str
    reward: float
    error: Optional[str] = None


class PenaltyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    amount: float
    reason: str


class UnsafeActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    reason: str
    penalty: float
    terminal: bool = False
    matched_conditions: dict[str, Any] = Field(default_factory=dict)


class RubricCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    label: str
    weight: float
    score: float
    weighted_score: float
    details: str = ""


class MutationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    field_name: str
    value: str


class CreditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    amount: float
    currency: str


class EscalationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    escalation_team: str


class DraftReplyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    template_id: str
    reply_checklist: list[str] = Field(default_factory=list)
    freeform_note: Optional[str] = None


class SupportReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = 0.0
    rubric_delta: float = 0.0
    behavior_adjustment: float = 0.0
    penalties: list[PenaltyRecord] = Field(default_factory=list)
    unsafe_actions: list[UnsafeActionRecord] = Field(default_factory=list)

    @property
    def total(self) -> float:
        return self.rubric_delta + self.behavior_adjustment


class SupportAction(Action):
    action_type: ActionType
    ticket_id: Optional[str] = None
    record_id: Optional[str] = None
    query: Optional[str] = None
    priority: Optional[TicketPriority] = None
    status: Optional[TicketStatus] = None
    tag: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    escalation_team: Optional[str] = None
    template_id: Optional[str] = None
    reply_checklist: list[str] = Field(default_factory=list)
    resolution_code: Optional[str] = None
    freeform_note: Optional[str] = None

    @model_validator(mode="after")
    def validate_for_action_type(self) -> "SupportAction":
        required_fields: dict[ActionType, set[str]] = {
            ActionType.OPEN_TICKET: {"ticket_id"},
            ActionType.INSPECT_RECORD: {"record_id"},
            ActionType.SEARCH_KB: {"query"},
            ActionType.SET_PRIORITY: {"priority"},
            ActionType.SET_STATUS: {"status"},
            ActionType.ADD_TAG: {"tag"},
            ActionType.APPLY_CREDIT: {"amount", "currency"},
            ActionType.ESCALATE: {"escalation_team"},
            ActionType.DRAFT_REPLY: {"template_id"},
            ActionType.FINALIZE_RESOLUTION: {"resolution_code"},
        }

        for field_name in required_fields[self.action_type]:
            value = getattr(self, field_name)
            if value in (None, "", []):
                raise ValueError(
                    f"{field_name} is required for action_type={self.action_type.value}"
                )

        if self.action_type in {
            ActionType.SET_PRIORITY,
            ActionType.SET_STATUS,
            ActionType.ADD_TAG,
            ActionType.APPLY_CREDIT,
            ActionType.ESCALATE,
            ActionType.DRAFT_REPLY,
            ActionType.FINALIZE_RESOLUTION,
        } and not self.ticket_id:
            raise ValueError(
                f"ticket_id is required for action_type={self.action_type.value}"
            )

        if self.amount is not None and self.amount <= 0:
            raise ValueError("amount must be positive")

        if self.amount is not None and self.amount > 100000:
            raise ValueError("amount exceeds allowed maximum")

        return self

    def signature(self) -> str:
        return (
            f"{self.action_type.value}|{self.ticket_id}|{self.record_id}|{self.query}|"
            f"{self.priority}|{self.status}|{self.tag}|{self.amount}|{self.currency}|"
            f"{self.escalation_team}|{self.template_id}|{sorted(self.reply_checklist)}|"
            f"{self.resolution_code}"
        )


class SupportObservation(Observation):
    task_brief: str
    inbox: list[TicketSummary] = Field(default_factory=list)
    active_ticket_id: Optional[str] = None
    focus_panel: Optional[FocusPanel] = None
    available_record_ids: list[str] = Field(default_factory=list)
    action_history: list[ActionTrace] = Field(default_factory=list)
    step_count: int = 0
    remaining_steps: int = 0
    last_action_error: Optional[str] = None
    reply_requirements: Optional[ReplyRequirements] = None
    reward: Optional[float] = 0.0
    done: bool = False
    peer_messages: list[PeerMessage] = Field(default_factory=list)
    world_context: Optional[WorldContext] = None
    current_phase: Optional[int] = None


class SupportState(State):
    task_id: str = ""
    seed: int = 0
    primary_ticket_id: str = ""
    selected_ticket_id: Optional[str] = None
    active_ticket_id: Optional[str] = None
    records_viewed: list[str] = Field(default_factory=list)
    kb_articles_viewed: list[str] = Field(default_factory=list)
    ticket_mutations: list[MutationRecord] = Field(default_factory=list)
    credits_applied: list[CreditRecord] = Field(default_factory=list)
    escalations: list[EscalationRecord] = Field(default_factory=list)
    draft_reply: Optional[DraftReplyState] = None
    rubric_progress: float = 0.0
    rubric_breakdown: list[RubricCheckResult] = Field(default_factory=list)
    unsafe_actions: list[UnsafeActionRecord] = Field(default_factory=list)
    behavior_penalties: list[PenaltyRecord] = Field(default_factory=list)
    final_score: Optional[float] = None
    last_action_error: Optional[str] = None
    last_reward: SupportReward = Field(default_factory=SupportReward)
    done: bool = False
    peer_messages: list[PeerMessage] = Field(default_factory=list)
    completed_phases: list[int] = Field(default_factory=list)
    current_phase: Optional[int] = None


class TicketFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    subject: str
    from_contact: str
    account_id: str
    summary: str
    priority: TicketPriority
    status: TicketStatus
    tags: list[str] = Field(default_factory=list)
    allowed_actions: list[ActionType] = Field(default_factory=list)


class RecordFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    kind: RecordKind
    title: str
    body: str
    related_ticket_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class KnowledgeBaseArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_id: str
    title: str
    summary: str
    content: str
    keywords: list[str] = Field(default_factory=list)


class RubricRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    label: str
    weight: float
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class ForbiddenActionSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action_type: ActionType
    conditions: dict[str, Any] = Field(default_factory=dict)
    reason: str
    penalty: float
    terminal: bool = False


class ReplyRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    checklist: list[str] = Field(default_factory=list)


class TaskFixture(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    difficulty: Difficulty
    task_brief: str
    primary_ticket_id: str
    max_steps: int = 12
    tickets: list[TicketFixture]
    records: list[RecordFixture] = Field(default_factory=list)
    kb_articles: list[KnowledgeBaseArticle] = Field(default_factory=list)
    rubric: list[RubricRule] = Field(default_factory=list)
    forbidden_actions: list[ForbiddenActionSpec] = Field(default_factory=list)
    reply_requirements: ReplyRequirements
    oracle_reference_path: list[str] = Field(default_factory=list)
    investigation_phases: list[InvestigationPhase] = Field(default_factory=list)
    world_context: Optional[dict[str, Any]] = None
    peer_inject: list[PeerInjectSpec] = Field(default_factory=list)
