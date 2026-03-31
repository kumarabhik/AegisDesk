"""Compatibility helpers for OpenEnv runtime and local development."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

ObservationT = TypeVar("ObservationT")

try:
    from openenv.core.client_types import StepResult as OpenEnvStepResult
except ImportError:

    @dataclass
    class OpenEnvStepResult(Generic[ObservationT]):
        """Fallback step result used when openenv-core is unavailable."""

        observation: ObservationT
        reward: Optional[float] = None
        done: bool = False
        info: dict[str, Any] = field(default_factory=dict)


try:
    from openenv.core.env_server.types import Action, Observation, State
    from openenv.core.env_server.interfaces import Environment
except ImportError:

    class Action(BaseModel):
        """Fallback action base model."""

        model_config = ConfigDict(extra="forbid")

    class Observation(BaseModel):
        """Fallback observation base model."""

        model_config = ConfigDict(extra="forbid")
        done: bool = False
        reward: Optional[float] = None

    class State(BaseModel):
        """Fallback state base model."""

        model_config = ConfigDict(extra="forbid")
        episode_id: Optional[str] = None
        step_count: int = 0

    class Environment:
        """Fallback environment base class."""

        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def reset_async(self, *args: Any, **kwargs: Any):
            return self.reset(*args, **kwargs)

        async def step_async(self, *args: Any, **kwargs: Any):
            return self.step(*args, **kwargs)


StepResult = OpenEnvStepResult
