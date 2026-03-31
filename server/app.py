"""FastAPI application entrypoint for support_ops_env."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pydantic import ValidationError
import uvicorn

try:
    from openenv.core.env_server import create_app
except ImportError:
    create_app = None

try:
    from ..models import SupportAction, SupportObservation
    from .environment import SupportOpsEnvironment
except ImportError:
    from models import SupportAction, SupportObservation
    from server.environment import SupportOpsEnvironment


_shared_env: SupportOpsEnvironment | None = None


def create_environment() -> SupportOpsEnvironment:
    """Factory used by OpenEnv and the custom HTTP routes."""

    global _shared_env
    if _shared_env is None:
        task_id = os.getenv("DEFAULT_TASK_ID")
        _shared_env = SupportOpsEnvironment(task_id=task_id)
    return _shared_env


class ResetPayload(BaseModel):
    task_id: str | None = None
    seed: int | None = None


if create_app is not None:
    app = create_app(
        create_environment,
        SupportAction,
        SupportObservation,
        env_name="support_ops_env",
    )
    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) not in {"/reset", "/step", "/state"}
    ]
else:
    app = FastAPI(title="support_ops_env")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetPayload | None = None) -> dict[str, Any]:
    env = create_environment()
    payload = payload or ResetPayload()
    observation = env.reset(task_id=payload.task_id, seed=payload.seed)
    return {
        "observation": observation.model_dump(mode="json"),
        "reward": observation.reward,
        "done": observation.done,
        "info": env.last_info,
    }


@app.post("/step")
async def step(request: Request) -> dict[str, Any]:
    env = create_environment()
    payload = await request.json()
    action_payload = payload.get("action", payload) if isinstance(payload, dict) else payload
    try:
        action = SupportAction.model_validate(action_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    observation = env.step(action)
    return {
        "observation": observation.model_dump(mode="json"),
        "reward": observation.reward,
        "done": observation.done,
        "info": env.last_info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    env = create_environment()
    return env.state.model_dump(mode="json")


@app.get("/")
def root() -> dict[str, str]:
    """Basic 200 endpoint for Space health probes."""

    return {"status": "ok", "env_name": "support_ops_env"}


def main() -> None:
    """Run the app with uvicorn."""

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
