"""Baseline inference runner for support_ops_env."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

try:
    from client import LocalSupportOpsEnv, SupportOpsEnv
    from models import SupportAction, SupportObservation
except ImportError:
    from .client import LocalSupportOpsEnv, SupportOpsEnv
    from .models import SupportAction, SupportObservation


TASK_IDS = [
    "billing_seat_adjustment",
    "login_incident_triage",
    "suspicious_admin_request",
]
MAX_STEPS = 12
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
SYSTEM_PROMPT = """
You are operating a structured SaaS support console benchmark.
Always return a single JSON object that matches the SupportAction schema.
Choose only one action at a time.
Use the visible inbox, focus panel, and available record ids.
Do not wrap JSON in markdown.
""".strip()
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


@dataclass(frozen=True)
class InferenceConfig:
    """Resolved provider configuration for the baseline runner."""

    api_key: str
    model_name: str
    api_base_url: str | None
    credential_source: str


def _first_env(*names: str) -> tuple[str | None, str | None]:
    """Return the first non-empty environment value and its variable name."""

    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), name
    return None, None


def resolve_inference_config() -> InferenceConfig:
    """Resolve credentials for OpenAI-compatible providers.

    Preferred order:
    1. Hackathon path via `HF_TOKEN` and the Hugging Face router
    2. Generic OpenAI-compatible path via `OPENAI_API_KEY`
    3. Local compatibility aliases such as Groq or xAI
    """

    api_key, api_key_name = _first_env(
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
    )
    model_name, _ = _first_env(
        "MODEL_NAME",
        "GROQ_MODEL",
        "XAI_MODEL",
        "GROK_MODEL",
    )
    if api_key_name == "HF_TOKEN":
        api_base_url = os.getenv("API_BASE_URL", "").strip() or HF_ROUTER_BASE_URL
    else:
        api_base_url, _ = _first_env(
            "API_BASE_URL",
            "GROQ_BASE_URL",
            "XAI_BASE_URL",
            "GROK_BASE_URL",
        )

    if not api_key:
        raise RuntimeError(
            "HF_TOKEN or OPENAI_API_KEY is required to run inference.py. "
            "For compatible local providers, GROQ_API_KEY, XAI_API_KEY, or "
            "GROK_API_KEY may also be used."
        )
    if not model_name:
        raise RuntimeError(
            "MODEL_NAME is required to run inference.py. "
            "For OpenAI-compatible providers, GROQ_MODEL, XAI_MODEL, or "
            "GROK_MODEL may also be used."
        )

    credential_source = api_key_name or "HF_TOKEN"
    return InferenceConfig(
        api_key=api_key,
        model_name=model_name,
        api_base_url=api_base_url,
        credential_source=credential_source,
    )


def make_openai_client():
    """Create an OpenAI-compatible client lazily."""

    from openai import OpenAI

    config = resolve_inference_config()
    return (
        OpenAI(api_key=config.api_key, base_url=config.api_base_url),
        config.model_name,
    )


def build_user_prompt(
    step_index: int, observation: SupportObservation, history: list[str]
) -> str:
    """Render a compact observation prompt."""

    inbox_lines = [
        (
            f"- {ticket.ticket_id} | {ticket.priority.value}/{ticket.status.value} | "
            f"{ticket.subject} | tags={','.join(ticket.tags)}"
        )
        for ticket in observation.inbox
    ]
    history_text = "\n".join(history[-6:]) if history else "No prior actions."
    focus_text = "None"
    if observation.focus_panel is not None:
        focus_text = (
            f"{observation.focus_panel.panel_type}: {observation.focus_panel.title}\n"
            f"{observation.focus_panel.body}"
        )

    return f"""
Task brief:
{observation.task_brief}

Step:
{step_index}

Inbox:
{chr(10).join(inbox_lines)}

Active ticket:
{observation.active_ticket_id}

Available record ids:
{", ".join(observation.available_record_ids)}

Focus panel:
{focus_text}

Recent history:
{history_text}

Return one JSON object with fields matching SupportAction.
""".strip()


def parse_model_action(response_text: str) -> SupportAction:
    """Parse model output into a SupportAction."""

    text = response_text.strip()
    if not text:
        raise ValueError("Empty model response")
    try:
        payload = json.loads(text)
        return SupportAction.model_validate(payload)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response did not contain JSON")
        payload = json.loads(text[start : end + 1])
        return SupportAction.model_validate(payload)


def fallback_action(observation: SupportObservation) -> SupportAction:
    """Safe fallback if the model call fails."""

    if observation.active_ticket_id is None and observation.inbox:
        return SupportAction(
            action_type="open_ticket",
            ticket_id=observation.inbox[0].ticket_id,
        )
    if observation.active_ticket_id and observation.available_record_ids:
        return SupportAction(
            action_type="inspect_record",
            record_id=observation.available_record_ids[0],
        )
    return SupportAction(action_type="search_kb", query="policy")


def make_environment():
    """Create either a remote or local environment client."""

    env_base_url = os.getenv("ENV_BASE_URL")
    if env_base_url:
        return SupportOpsEnv(env_base_url)
    return LocalSupportOpsEnv()


def _json_line(tag: str, payload: dict[str, Any]) -> str:
    """Return a stable tagged JSON log line."""

    return f"[{tag}] {json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}"


def emit_start_log(
    *,
    task_id: str,
    seed: int,
    model_name: str,
    api_base_url: str | None,
    env_base_url: str,
    max_steps: int,
) -> None:
    """Print a structured task-start log line."""

    print(
        _json_line(
            "START",
            {
                "task_id": task_id,
                "seed": seed,
                "model_name": model_name,
                "api_base_url": api_base_url or "",
                "env_base_url": env_base_url,
                "max_steps": max_steps,
            },
        )
    )


def emit_step_log(
    *,
    task_id: str,
    seed: int,
    step: int,
    action: SupportAction,
    reward: float | None,
    done: bool,
    last_action_error: str | None,
    fallback_used: bool,
) -> None:
    """Print a structured per-step log line."""

    print(
        _json_line(
            "STEP",
            {
                "task_id": task_id,
                "seed": seed,
                "step": step,
                "action_type": action.action_type.value,
                "action": action.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_defaults=True,
                ),
                "reward": reward if reward is not None else 0.0,
                "done": done,
                "last_action_error": last_action_error or "",
                "fallback_used": fallback_used,
            },
        )
    )


def emit_end_log(
    *,
    task_id: str,
    seed: int,
    final_score: float,
    steps_taken: int,
    completed: bool,
) -> None:
    """Print a structured task-end log line."""

    print(
        _json_line(
            "END",
            {
                "task_id": task_id,
                "seed": seed,
                "final_score": round(final_score, 4),
                "steps_taken": steps_taken,
                "completed": completed,
            },
        )
    )


def run_task(task_id: str, task_seed: int, config: InferenceConfig) -> float:
    """Run one task and return its final score."""

    env = make_environment()
    client, model_name = make_openai_client()
    history: list[str] = []
    env_base_url = os.getenv("ENV_BASE_URL", "local_in_process")
    emit_start_log(
        task_id=task_id,
        seed=task_seed,
        model_name=model_name,
        api_base_url=config.api_base_url,
        env_base_url=env_base_url,
        max_steps=MAX_STEPS,
    )
    try:
        result = env.reset(task_id=task_id, seed=task_seed)
        observation = result.observation
        final_score = 0.0
        completed = False
        steps_taken = 0
        for step_index in range(1, MAX_STEPS + 1):
            prompt = build_user_prompt(step_index, observation, history)
            fallback_used = False
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
                action = parse_model_action(response_text)
            except Exception:
                action = fallback_action(observation)
                fallback_used = True

            result = env.step(action)
            observation = result.observation
            steps_taken = step_index
            history.append(
                f"step={step_index} action={action.action_type.value} "
                f"reward={result.reward} error={observation.last_action_error}"
            )
            emit_step_log(
                task_id=task_id,
                seed=task_seed,
                step=step_index,
                action=action,
                reward=result.reward,
                done=result.done,
                last_action_error=observation.last_action_error,
                fallback_used=fallback_used,
            )
            if result.done:
                state = env.state()
                final_score = float(state.final_score or 0.0)
                completed = True
                break
        else:
            state = env.state()
            final_score = float(state.final_score or state.rubric_progress or 0.0)

        emit_end_log(
            task_id=task_id,
            seed=task_seed,
            final_score=final_score,
            steps_taken=steps_taken,
            completed=completed,
        )
        return final_score
    finally:
        env.close()


def main() -> None:
    """Run all canonical tasks and print a reproducible summary."""

    config = resolve_inference_config()
    scores: dict[str, float] = {}
    for index, task_id in enumerate(TASK_IDS, start=1):
        scores[task_id] = run_task(task_id, task_seed=index, config=config)

    mean_score = round(sum(scores.values()) / len(scores), 4)
    print(
        _json_line(
            "END",
            {
                "run_summary": True,
                "task_count": len(TASK_IDS),
                "mean_score": mean_score,
                "scores": scores,
                "model_name": config.model_name,
                "api_base_url": config.api_base_url or "",
                "credential_source": config.credential_source,
            },
        )
    )


if __name__ == "__main__":
    main()
