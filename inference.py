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

    The submission contract still prefers `OPENAI_API_KEY`, `MODEL_NAME`, and
    `API_BASE_URL`, but local runs may reuse compatible provider aliases such as
    Groq or xAI.
    """

    api_key, api_key_name = _first_env(
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
    api_base_url, _ = _first_env(
        "API_BASE_URL",
        "GROQ_BASE_URL",
        "XAI_BASE_URL",
        "GROK_BASE_URL",
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to run inference.py. "
            "For OpenAI-compatible providers, GROQ_API_KEY, XAI_API_KEY, or "
            "GROK_API_KEY may also be used."
        )
    if not model_name:
        raise RuntimeError(
            "MODEL_NAME is required to run inference.py. "
            "For OpenAI-compatible providers, GROQ_MODEL, XAI_MODEL, or "
            "GROK_MODEL may also be used."
        )

    credential_source = api_key_name or "OPENAI_API_KEY"
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


def run_task(task_id: str, task_seed: int) -> float:
    """Run one task and return its final score."""

    client, model_name = make_openai_client()
    env = make_environment()
    history: list[str] = []
    try:
        result = env.reset(task_id=task_id, seed=task_seed)
        observation = result.observation
        final_score = 0.0
        for step_index in range(1, MAX_STEPS + 1):
            prompt = build_user_prompt(step_index, observation, history)
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

            result = env.step(action)
            observation = result.observation
            history.append(
                f"step={step_index} action={action.action_type.value} "
                f"reward={result.reward} error={observation.last_action_error}"
            )
            if result.done:
                state = env.state()
                final_score = float(state.final_score or 0.0)
                break
        else:
            state = env.state()
            final_score = float(state.final_score or state.rubric_progress or 0.0)
        print(f"{task_id}: {final_score:.4f}")
        return final_score
    finally:
        env.close()


def main() -> None:
    """Run all canonical tasks and print a reproducible summary."""

    config = resolve_inference_config()
    print(
        json.dumps(
            {
                "api_base_url": config.api_base_url,
                "credential_source": config.credential_source,
                "model_name": config.model_name,
            },
            indent=2,
            sort_keys=True,
        )
    )
    scores: dict[str, float] = {}
    for index, task_id in enumerate(TASK_IDS, start=1):
        scores[task_id] = run_task(task_id, task_seed=index)
    mean_score = sum(scores.values()) / len(scores)
    payload: dict[str, Any] = {
        "scores": scores,
        "mean_score": round(mean_score, 4),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
