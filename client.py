"""Client helpers for local and remote support_ops_env usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import requests

try:
    from .models import SupportAction, SupportObservation, SupportState
    from .server.environment import SupportOpsEnvironment
except ImportError:
    from models import SupportAction, SupportObservation, SupportState
    from server.environment import SupportOpsEnvironment


@dataclass
class ClientStepResult:
    """Simple sync result wrapper used by local tests and the baseline script."""

    observation: SupportObservation
    reward: float | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class SupportOpsEnv:
    """Remote HTTP client for a running support_ops_env server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(
        self, task_id: Optional[str] = None, seed: Optional[int] = None
    ) -> ClientStepResult:
        response = self._session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        response.raise_for_status()
        return self._parse_result(response.json())

    def step(self, action: SupportAction) -> ClientStepResult:
        response = self._session.post(
            f"{self.base_url}/step",
            json=action.model_dump(mode="json"),
            timeout=30,
        )
        response.raise_for_status()
        return self._parse_result(response.json())

    def state(self) -> SupportState:
        response = self._session.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return SupportState.model_validate(response.json())

    def close(self) -> None:
        self._session.close()

    def _parse_result(self, payload: dict[str, Any]) -> ClientStepResult:
        observation = SupportObservation.model_validate(payload["observation"])
        return ClientStepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )


class LocalSupportOpsEnv:
    """In-process environment runner used for tests and local inference."""

    def __init__(self, task_id: Optional[str] = None, seed: Optional[int] = None):
        self._environment = SupportOpsEnvironment(task_id=task_id, seed=seed)

    def reset(
        self, task_id: Optional[str] = None, seed: Optional[int] = None
    ) -> ClientStepResult:
        observation = self._environment.reset(task_id=task_id, seed=seed)
        return ClientStepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
            info=self._environment.last_info,
        )

    def step(self, action: SupportAction) -> ClientStepResult:
        observation = self._environment.step(action)
        return ClientStepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
            info=self._environment.last_info,
        )

    def state(self) -> SupportState:
        return self._environment.state

    def close(self) -> None:
        return None
