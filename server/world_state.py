"""WorldStateEngine — fixture-driven context for world-modeling tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyWindow:
    name: str
    description: str
    active: bool
    affects_tasks: list[str]


@dataclass
class WorldState:
    active_incidents: list[str]
    policy_window: PolicyWindow | None
    region: str
    account_health_index: float

    def is_policy_active(self, task_id: str) -> bool:
        if self.policy_window is None or not self.policy_window.active:
            return False
        return task_id in self.policy_window.affects_tasks

    def has_active_incident(self, incident_id: str) -> bool:
        return incident_id in self.active_incidents

    def to_observation_dict(self) -> dict[str, Any]:
        """Compact world context for agent observation — no internal grader truth."""
        return {
            "active_incidents": self.active_incidents,
            "policy_window": {
                "name": self.policy_window.name,
                "description": self.policy_window.description,
                "active": self.policy_window.active,
            } if self.policy_window else None,
            "region": self.region,
            "account_health_index": self.account_health_index,
        }


_EMPTY_WORLD = WorldState(
    active_incidents=[],
    policy_window=None,
    region="global",
    account_health_index=1.0,
)


def build_world_state(world_context: dict[str, Any] | None) -> WorldState:
    """Build a WorldState from a fixture's world_context block, or return empty state."""
    if not world_context:
        return _EMPTY_WORLD

    pw_raw = world_context.get("policy_window")
    policy_window = None
    if pw_raw:
        policy_window = PolicyWindow(
            name=pw_raw.get("name", ""),
            description=pw_raw.get("description", ""),
            active=pw_raw.get("active", False),
            affects_tasks=pw_raw.get("affects_tasks", []),
        )

    return WorldState(
        active_incidents=world_context.get("active_incidents", []),
        policy_window=policy_window,
        region=world_context.get("region", "global"),
        account_health_index=float(world_context.get("account_health_index", 1.0)),
    )
