"""Helpers for the Kaggle GRPO notebook."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import httpx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


TASK_ORDER = [
    "billing_seat_adjustment",
    "login_incident_triage",
    "suspicious_admin_request",
    "customer_escalation_chain",
    "multi_tier_billing_dispute",
    "data_breach_response_lifecycle",
    "contract_renewal_negotiation",
    "service_reinstatement_review",
    "api_partner_access_audit",
]

VALID_ACTIONS = {
    "open_ticket",
    "inspect_record",
    "search_kb",
    "set_priority",
    "set_status",
    "add_tag",
    "apply_credit",
    "escalate",
    "draft_reply",
    "finalize_resolution",
}

TICKET_ACTIONS = {
    "open_ticket",
    "set_priority",
    "set_status",
    "add_tag",
    "apply_credit",
    "escalate",
    "draft_reply",
    "finalize_resolution",
}

ACTION_REQUIREMENTS = {
    "open_ticket": ("ticket_id",),
    "inspect_record": ("record_id",),
    "search_kb": ("query",),
    "set_priority": ("ticket_id", "priority"),
    "set_status": ("ticket_id", "status"),
    "add_tag": ("ticket_id", "tag"),
    "apply_credit": ("ticket_id", "amount", "currency"),
    "escalate": ("ticket_id", "escalation_team"),
    "draft_reply": ("ticket_id", "template_id"),
    "finalize_resolution": ("ticket_id", "resolution_code"),
}

EARLY_MUTATION_ACTIONS = {
    "set_priority",
    "set_status",
    "add_tag",
    "apply_credit",
    "escalate",
    "draft_reply",
    "finalize_resolution",
}


def load_qwen_kaggle_model(
    model_name: str,
    hf_token: str,
    *,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Load a Qwen model for Kaggle T4 training using 4-bit LoRA."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    return tokenizer, model


def completion_to_text(completion: Any) -> str:
    """Normalize GRPO completions into a plain text string for parsing.

    TRL may hand reward functions either raw strings or conversational
    structures like [{"role": "assistant", "content": "..."}].
    """

    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, bytes):
        return completion.decode("utf-8", errors="ignore")
    if isinstance(completion, dict):
        if "content" in completion:
            return completion_to_text(completion["content"])
        if "text" in completion:
            return completion_to_text(completion["text"])
        return json.dumps(completion, ensure_ascii=False)
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            text = completion_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(completion)


def strip_reasoning(text: Any) -> str:
    """Remove think blocks and markdown fences before JSON parsing."""

    normalized = completion_to_text(text)
    cleaned = re.sub(r"<think>.*?</think>", "", normalized or "", flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("```json", "```").replace("```JSON", "```")
    cleaned = cleaned.replace("```", " ")
    return cleaned.strip()


def parse_action_text(text: Any, *, fallback_ticket: str = "fallback") -> dict[str, Any]:
    """Parse an action JSON object from a raw model completion."""

    cleaned = strip_reasoning(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except Exception:
            pass

    return {
        "action_type": "finalize_resolution",
        "ticket_id": fallback_ticket,
        "resolution_code": "no_action",
    }


def force_disable_qwen_thinking(tokenizer: Any) -> bool:
    """Patch the tokenizer chat template so Qwen3 does not emit think blocks."""

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template or "enable_thinking" not in chat_template:
        return False

    patched = chat_template.replace("enable_thinking=True", "enable_thinking=False")
    if patched == chat_template:
        return False

    tokenizer.chat_template = patched
    return True


def parse_action_candidate(text: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Parse a candidate action and return diagnostics instead of a fake fallback action."""

    raw_text = completion_to_text(text)
    cleaned_text = strip_reasoning(text)
    metadata = {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "parse_mode": "invalid",
        "had_reasoning": raw_text != cleaned_text,
        "had_markdown_fence": "```" in raw_text,
    }

    try:
        action = json.loads(cleaned_text)
        metadata["parse_mode"] = "full_json"
        return action, metadata
    except Exception:
        pass

    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            action = json.loads(cleaned_text[start : end + 1])
            metadata["parse_mode"] = "substring_json"
            return action, metadata
        except Exception:
            pass

    return None, metadata


def choose_safe_follow_up_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Pick one low-risk follow-up action to reduce one-step reward ties."""

    available_records = [
        record_id
        for record_id in observation.get("available_record_ids", []) or []
        if isinstance(record_id, str) and record_id
    ]
    if available_records:
        return {
            "action_type": "inspect_record",
            "record_id": available_records[0],
        }

    inbox = observation.get("inbox", []) or []
    for ticket in inbox:
        if isinstance(ticket, dict) and ticket.get("ticket_id"):
            return {
                "action_type": "open_ticket",
                "ticket_id": ticket["ticket_id"],
            }

    active_ticket_id = observation.get("active_ticket_id")
    if active_ticket_id:
        return {
            "action_type": "search_kb",
            "query": f"policy checklist for {active_ticket_id}",
        }
    return None


def build_kaggle_grpo_reward_fn(
    env_url: str,
    *,
    debug_path: str | Path | None = None,
    http_timeout: int = 30,
    max_logged_groups: int = 12,
):
    """Build a GRPO reward function with stronger ranking signal and tie diagnostics."""

    debug_path = Path(debug_path) if debug_path else None
    debug_state = {
        "parse_failures": 0,
        "reset_failures": 0,
        "step_failures": 0,
        "identical_groups": 0,
        "logged_groups": 0,
        "logged_env_warning": False,
        "logged_group_overview": False,
    }

    def _safe_json(response: Any) -> dict[str, Any]:
        try:
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _extract_observation(payload: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        observation = payload.get("observation", payload) if isinstance(payload, dict) else {}
        if isinstance(observation, dict):
            return observation
        return fallback or {}

    def _state_score(payload: dict[str, Any], fallback: float = 0.0) -> float:
        if not isinstance(payload, dict):
            return fallback
        return float(payload.get("final_score") or payload.get("rubric_progress") or fallback or 0.0)

    def _prompt_group_key(prompt: Any, task_id: str, seed: int) -> str:
        prompt_text = completion_to_text(prompt)
        digest = hashlib.sha1(prompt_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{task_id}|{seed}|{digest}"

    def _append_debug(record: dict[str, Any]) -> None:
        if debug_path is None:
            return
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _format_score(meta: dict[str, Any]) -> tuple[float, dict[str, float]]:
        score = 0.0
        components: dict[str, float] = {}
        if meta["had_reasoning"]:
            components["reasoning_penalty"] = -0.08
            score -= 0.08
        if meta["had_markdown_fence"]:
            components["markdown_penalty"] = -0.04
            score -= 0.04
        if meta["cleaned_text"].startswith("{") and meta["cleaned_text"].endswith("}"):
            components["json_shape_bonus"] = 0.04
            score += 0.04
        elif meta["cleaned_text"]:
            components["json_shape_penalty"] = -0.03
            score -= 0.03
        return score, components

    def _shape_score(action: dict[str, Any], observation: dict[str, Any]) -> tuple[float, dict[str, float]]:
        action_type = str(action.get("action_type") or "").strip()
        components: dict[str, float] = {}
        score = 0.0

        if action_type in VALID_ACTIONS:
            components["valid_action_bonus"] = 0.10
            score += 0.10
        else:
            components["invalid_action_penalty"] = -0.35
            return score - 0.35, components

        required_fields = ACTION_REQUIREMENTS.get(action_type, ())
        missing_fields = [
            field
            for field in required_fields
            if not str(action.get(field, "")).strip()
        ]
        if missing_fields:
            components["missing_field_penalty"] = -0.28
            score -= 0.28
        else:
            components["required_fields_bonus"] = 0.08
            score += 0.08

        inbox = observation.get("inbox", []) or []
        inbox_ids = [
            ticket.get("ticket_id")
            for ticket in inbox
            if isinstance(ticket, dict) and ticket.get("ticket_id")
        ]
        inbox_set = set(inbox_ids)
        available_records = [
            record_id
            for record_id in observation.get("available_record_ids", []) or []
            if isinstance(record_id, str) and record_id
        ]
        record_set = set(available_records)
        active_ticket_id = observation.get("active_ticket_id")
        focus_panel = observation.get("focus_panel") or {}
        focus_text = (
            f"{focus_panel.get('title', '')}\n{focus_panel.get('body', '')}"
            if isinstance(focus_panel, dict)
            else ""
        ).lower()

        if action_type == "inspect_record":
            record_id = str(action.get("record_id") or "")
            if record_id in record_set:
                components["record_membership_bonus"] = 0.22
                score += 0.22
                if available_records and record_id == available_records[0]:
                    components["first_record_bonus"] = 0.04
                    score += 0.04
                if record_id.lower() in focus_text:
                    components["focus_record_bonus"] = 0.04
                    score += 0.04
            else:
                components["bad_record_penalty"] = -0.18
                score -= 0.18
            if not active_ticket_id and available_records:
                components["early_inspect_bonus"] = 0.06
                score += 0.06
        elif action_type == "open_ticket":
            ticket_id = str(action.get("ticket_id") or "")
            if ticket_id in inbox_set:
                components["ticket_membership_bonus"] = 0.18
                score += 0.18
                if inbox_ids and ticket_id == inbox_ids[0]:
                    components["first_ticket_bonus"] = 0.03
                    score += 0.03
            else:
                components["bad_ticket_penalty"] = -0.12
                score -= 0.12
            if active_ticket_id:
                components["reopen_penalty"] = -0.08
                score -= 0.08
        elif action_type == "search_kb":
            query = str(action.get("query") or "").strip()
            if query:
                components["query_bonus"] = 0.06
                score += 0.06
            if available_records and not active_ticket_id:
                components["skip_record_penalty"] = -0.05
                score -= 0.05
        elif action_type in TICKET_ACTIONS:
            ticket_id = str(action.get("ticket_id") or "")
            if ticket_id and (ticket_id in inbox_set or ticket_id == active_ticket_id):
                components["ticket_target_bonus"] = 0.10
                score += 0.10
            else:
                components["ticket_target_penalty"] = -0.10
                score -= 0.10
            if action_type in EARLY_MUTATION_ACTIONS and not active_ticket_id and available_records:
                components["early_mutation_penalty"] = -0.18
                score -= 0.18
            if action_type == "draft_reply" and active_ticket_id:
                components["reply_context_bonus"] = 0.03
                score += 0.03
            if action_type == "finalize_resolution" and available_records:
                components["premature_finalize_penalty"] = -0.08
                score -= 0.08

        return score, components

    def _normalize_sequence(values: Any, count: int, default: Any) -> list[Any]:
        if values is None:
            return [default] * count
        if isinstance(values, (str, int)):
            return [values] * count
        normalized = list(values)
        if not normalized:
            return [default] * count
        if len(normalized) < count:
            normalized.extend([normalized[-1]] * (count - len(normalized)))
        return normalized[:count]

    def reward_fn(completions, prompts=None, task_id=None, seed=None, **kwargs) -> list[float]:
        count = len(completions)
        if task_id is None:
            raise RuntimeError(
                "GRPO reward_fn expected the dataset column `task_id`, but TRL did not forward it. "
                "Keep `trl>=0.21`, keep `remove_unused_columns=False`, and verify the train dataset still carries `task_id`."
            )
        if seed is None:
            raise RuntimeError(
                "GRPO reward_fn expected the dataset column `seed`, but TRL did not forward it. "
                "Without per-row seeds the training mix silently collapses."
            )
        if prompts is None:
            raise RuntimeError(
                "GRPO reward_fn expected `prompts` from TRL, but they were missing. "
                "Without prompts we cannot group completions correctly for tie diagnostics."
            )

        task_ids = _normalize_sequence(task_id, count, "billing_seat_adjustment")
        seeds = [int(value) for value in _normalize_sequence(seed, count, 42)]
        prompts = _normalize_sequence(prompts, count, None)

        rows: list[dict[str, Any]] = []

        if not debug_state["logged_env_warning"]:
            debug_state["logged_env_warning"] = True
            print(
                "Reward fn note: the remote AegisDesk env is a shared singleton process. "
                "Run one training worker and avoid concurrent browser/demo requests during GRPO."
            )

        with httpx.Client(timeout=http_timeout, follow_redirects=True) as client:
            for index, completion in enumerate(completions):
                meta_action, meta = parse_action_candidate(completion)
                row = {
                    "index": index,
                    "task_id": task_ids[index],
                    "seed": seeds[index],
                    "group_key": _prompt_group_key(prompts[index], str(task_ids[index]), seeds[index]),
                    "raw_preview": meta["raw_text"][:300],
                    "clean_preview": meta["cleaned_text"][:300],
                    "parse_mode": meta["parse_mode"],
                    "action_type": None,
                    "reward": 0.0,
                    "components": {},
                }

                reward, format_components = _format_score(meta)
                row["components"].update(format_components)

                if meta_action is None:
                    debug_state["parse_failures"] += 1
                    reward -= 0.90
                    row["components"]["parse_failure_penalty"] = -0.90
                    row["reward"] = round(float(reward), 4)
                    rows.append(row)
                    continue

                row["action_type"] = meta_action.get("action_type")

                try:
                    reset_payload = _safe_json(
                        client.post(
                            f"{env_url}/reset",
                            json={"task_id": task_ids[index], "seed": seeds[index]},
                        )
                    )
                    observation = _extract_observation(reset_payload)
                    if not observation:
                        debug_state["reset_failures"] += 1
                        reward -= 0.35
                        row["components"]["reset_penalty"] = -0.35

                    state_before = _safe_json(client.get(f"{env_url}/state"))
                    pre_progress = _state_score(state_before, fallback=0.0)

                    shape_score, shape_components = _shape_score(meta_action, observation)
                    reward += shape_score
                    row["components"].update(shape_components)

                    step_payload = _safe_json(client.post(f"{env_url}/step", json=meta_action))
                    step_observation = _extract_observation(step_payload, fallback=observation)
                    step_reward = float(step_payload.get("reward", 0.0) or 0.0)
                    state_after = _safe_json(client.get(f"{env_url}/state"))
                    post_progress = _state_score(state_after, fallback=pre_progress)
                    progress_delta = post_progress - pre_progress

                    reward += 0.55 * step_reward
                    reward += 0.45 * progress_delta
                    reward += 0.15 * post_progress
                    row["components"]["step_reward_term"] = round(0.55 * step_reward, 4)
                    row["components"]["progress_delta_term"] = round(0.45 * progress_delta, 4)
                    row["components"]["progress_level_term"] = round(0.15 * post_progress, 4)

                    if step_observation.get("last_action_error"):
                        reward -= 0.14
                        row["components"]["step_error_penalty"] = -0.14

                    if not bool(step_payload.get("done")) and not step_observation.get("last_action_error"):
                        follow_up = choose_safe_follow_up_action(step_observation)
                        if follow_up is not None and follow_up != meta_action:
                            follow_payload = _safe_json(client.post(f"{env_url}/step", json=follow_up))
                            follow_state = _safe_json(client.get(f"{env_url}/state"))
                            follow_progress = _state_score(follow_state, fallback=post_progress)
                            follow_delta = follow_progress - post_progress
                            reward += 0.20 * follow_delta
                            row["follow_up"] = follow_up
                            row["components"]["follow_up_delta_term"] = round(0.20 * follow_delta, 4)
                            follow_obs = _extract_observation(follow_payload, fallback=step_observation)
                            if follow_obs.get("last_action_error"):
                                reward -= 0.03
                                row["components"]["follow_up_error_penalty"] = -0.03
                except Exception as exc:
                    debug_state["step_failures"] += 1
                    reward -= 0.45
                    row["components"]["exception_penalty"] = -0.45
                    row["exception"] = f"{type(exc).__name__}: {exc}"

                row["reward"] = round(float(reward), 4)
                rows.append(row)

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["group_key"]].append(row)

        if not debug_state["logged_group_overview"]:
            debug_state["logged_group_overview"] = True
            group_sizes = [len(group_rows) for group_rows in grouped.values()]
            print(
                "Reward fn grouping:",
                f"completions={count}",
                f"prompt_groups={len(grouped)}",
                f"group_sizes={group_sizes[:8]}",
            )
            if len(grouped) == 1 and count > 1:
                print(
                    "WARNING: only one prompt group was observed. "
                    "If this persists, verify TRL is forwarding prompt content and dataset columns correctly."
                )

        for group_key, group_rows in grouped.items():
            if len(group_rows) < 2:
                continue
            distinct_rewards = {round(float(row["reward"]), 4) for row in group_rows}
            if len(distinct_rewards) <= 1:
                debug_state["identical_groups"] += 1
                debug_record = {
                    "group_key": group_key,
                    "task_id": group_rows[0]["task_id"],
                    "seed": group_rows[0]["seed"],
                    "rewards": [row["reward"] for row in group_rows],
                    "action_types": [row["action_type"] for row in group_rows],
                    "parse_modes": [row["parse_mode"] for row in group_rows],
                    "raw_previews": [row["raw_preview"] for row in group_rows],
                    "components": [row["components"] for row in group_rows],
                }
                if debug_state["logged_groups"] < max_logged_groups:
                    debug_state["logged_groups"] += 1
                    print(
                        "Identical GRPO reward group:",
                        group_rows[0]["task_id"],
                        "seed=",
                        group_rows[0]["seed"],
                        "rewards=",
                        [row["reward"] for row in group_rows],
                        "actions=",
                        [row["action_type"] for row in group_rows],
                    )
                    _append_debug(debug_record)

        return [float(row["reward"]) for row in rows]

    return reward_fn


def build_chat_inputs(tokenizer, messages: list[dict[str, str]], device: str):
    """Build model inputs while disabling Qwen3 thinking mode when supported."""

    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        inputs = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        inputs = tokenizer.apply_chat_template(messages, **kwargs)
    return inputs.to(device)


def save_training_plots(
    output_dir: str | Path,
    *,
    baseline_score: float = 0.325,
    workdir: str | Path = "/kaggle/working",
) -> dict[str, Any]:
    """Render reward/loss plots from trainer outputs."""

    output_dir = Path(output_dir)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "reward_log.jsonl"
    state_path = output_dir / "trainer_state.json"

    reward_records: list[dict[str, Any]] = []
    loss_records: list[dict[str, Any]] = []

    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("global_step", row.get("step", 0))
            reward_value = row.get("reward", row.get("mean_reward", row.get("rewards/mean")))
            loss_value = row.get("loss", row.get("train_loss"))
            if reward_value is not None:
                reward_records.append(
                    {
                        "task_id": row.get("task_id", "all"),
                        "step": step,
                        "reward": float(reward_value),
                    }
                )
            if loss_value is not None:
                loss_records.append({"step": step, "loss": float(loss_value)})

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for entry in state.get("log_history", []):
            step = entry.get("step", entry.get("global_step", 0))
            reward_value = entry.get("rewards/mean", entry.get("reward", entry.get("mean_reward")))
            loss_value = entry.get("loss", entry.get("train_loss"))
            if reward_value is not None:
                reward_records.append(
                    {
                        "task_id": entry.get("task_id", "all"),
                        "step": step,
                        "reward": float(reward_value),
                    }
                )
            if loss_value is not None:
                loss_records.append({"step": step, "loss": float(loss_value)})

    outputs: list[str] = []

    if reward_records:
        task_rewards: dict[str, list[float]] = defaultdict(list)
        task_steps: dict[str, list[int]] = defaultdict(list)
        overall_r: list[float] = []
        overall_s: list[int] = []

        for record in reward_records:
            task_id = record.get("task_id", "all")
            step = int(record.get("step", 0))
            reward = float(record.get("reward", 0.0))
            task_rewards[task_id].append(reward)
            task_steps[task_id].append(step)
            overall_r.append(reward)
            overall_s.append(step)

        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle("AegisDesk v2 - GRPO Per-Task Reward", fontsize=14, fontweight="bold")
        for ax, task in zip(axes.flat, TASK_ORDER):
            xs = task_steps.get(task, [])
            ys = task_rewards.get(task, [])
            if xs:
                ax.plot(xs, ys, linewidth=1.2, alpha=0.6)
                if len(ys) >= 5:
                    window = 5
                    rolling = [sum(ys[max(0, i - window) : i + 1]) / min(window, i + 1) for i in range(len(ys))]
                    ax.plot(xs, rolling, linewidth=2, linestyle="--", label="rolling mean")
            ax.set_title(task.replace("_", "\n"), fontsize=7)
            ax.set_ylim(-0.2, 1.05)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.grid(alpha=0.3)
        plt.tight_layout()
        per_task_path = workdir / "reward_curves_per_task.png"
        plt.savefig(per_task_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(per_task_path))

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(overall_s, overall_r, alpha=0.35, linewidth=1, label="per-step reward")
        if len(overall_r) >= 10:
            window = 10
            rolling = [sum(overall_r[max(0, i - window) : i + 1]) / min(window, i + 1) for i in range(len(overall_r))]
            ax2.plot(overall_s, rolling, linewidth=2, label="rolling mean (10)")
        ax2.axhline(y=baseline_score, color="red", linestyle=":", linewidth=1.5, label=f"baseline ({baseline_score:.3f})")
        ax2.set_title("AegisDesk v2 - Overall Mean Reward", fontweight="bold")
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Reward")
        ax2.grid(alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        overall_reward_path = workdir / "reward_curves_overall.png"
        plt.savefig(overall_reward_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        outputs.append(str(overall_reward_path))

    if loss_records:
        xs = [int(record["step"]) for record in loss_records]
        ys = [float(record["loss"]) for record in loss_records]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(xs, ys, linewidth=1.5, color="#1d4ed8")
        ax3.set_title("AegisDesk v2 - Training Loss", fontweight="bold")
        ax3.set_xlabel("Training step")
        ax3.set_ylabel("Loss")
        ax3.grid(alpha=0.3)
        plt.tight_layout()
        loss_path = workdir / "loss_curve.png"
        plt.savefig(loss_path, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        outputs.append(str(loss_path))

    return {
        "reward_records": len(reward_records),
        "loss_records": len(loss_records),
        "outputs": outputs,
        "checked_paths": [str(log_path), str(state_path)],
    }


def evaluate_local_model_on_env(
    *,
    model,
    tokenizer,
    env_url: str,
    model_name: str,
    tasks: list[str] | None = None,
    seed: int = 42,
    baseline_score: float = 0.325,
    max_steps: int = 15,
) -> tuple[dict[str, float], float]:
    """Evaluate the local fine-tuned model against the live AegisDesk environment."""

    model.eval()
    tasks = tasks or list(TASK_ORDER)
    device = getattr(model, "device", "cuda")

    system_prompt = (
        "You are an expert B2B SaaS support operator working inside AegisDesk.\n"
        "Your goals: choose the correct ticket, inspect records before acting, avoid unsafe shortcuts,\n"
        "and finalize the case with the highest possible score.\n"
        "Be careful, grounded, and conservative. If a situation looks security-sensitive,\n"
        "prioritize verification and escalation over direct fulfillment."
    )
    action_types = [
        "open_ticket",
        "inspect_record",
        "search_kb",
        "set_priority",
        "set_status",
        "add_tag",
        "apply_credit",
        "escalate",
        "draft_reply",
        "finalize_resolution",
    ]

    def format_obs_brief(obs: dict[str, Any]) -> str:
        inbox = obs.get("inbox", []) or []
        inbox_lines = [
            f"  {t.get('ticket_id', '?')}: {t.get('subject', '?')} | {t.get('priority', '?')} | {t.get('status', '?')}"
            for t in inbox
            if isinstance(t, dict)
        ]
        focus_panel = obs.get("focus_panel") or {}
        focus = (
            f"\nFocus: {focus_panel.get('title', '')}\n{(focus_panel.get('body') or '')[:500]}"
            if focus_panel
            else ""
        )
        return (
            f"Task: {obs.get('task_brief', '')}\n"
            f"Step: {obs.get('step_count', 0)} / {obs.get('remaining_steps', 0)} remaining\n"
            f"Active ticket: {obs.get('active_ticket_id', 'none')}\n"
            f"Records: {', '.join(obs.get('available_record_ids', [])) or 'none'}\n"
            f"Error: {obs.get('last_action_error') or 'none'}\n"
            f"Inbox:\n" + "\n".join(inbox_lines) + focus
        )

    @torch.inference_mode()
    def generate_action(obs_text: str) -> dict[str, Any]:
        action_schema = (
            "Respond ONLY with a JSON object. Valid action_type values: "
            + ", ".join(action_types)
            + ". Required fields per type: open_ticket->ticket_id, inspect_record->record_id, "
              "search_kb->query, set_priority->ticket_id+priority, set_status->ticket_id+status, "
              "add_tag->ticket_id+tag, apply_credit->ticket_id+amount+currency, "
              "escalate->ticket_id+escalation_team, draft_reply->ticket_id+template_id, "
              "finalize_resolution->ticket_id+resolution_code."
        )
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + action_schema},
            {"role": "user", "content": obs_text},
        ]
        inputs = build_chat_inputs(tokenizer, messages, device)
        outputs = model.generate(
            inputs,
            max_new_tokens=160,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs.shape[-1] :]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        return parse_action_text(decoded)

    def _safe_json(response):
        try:
            return response.json()
        except Exception:
            return {}

    print(f"Running benchmark evaluation for {model_name} (max {max_steps} steps per task)...")
    scores: dict[str, float] = {}

    with httpx.Client(timeout=60, follow_redirects=True) as http:
        for task_id in tasks:
            try:
                reset_payload = _safe_json(http.post(
                    f"{env_url}/reset",
                    json={"task_id": task_id, "seed": seed},
                ))
                obs = reset_payload.get("observation", reset_payload) if isinstance(reset_payload, dict) else {}
                if not isinstance(obs, dict):
                    obs = {}
                for _ in range(max_steps):
                    action = generate_action(format_obs_brief(obs))
                    result = _safe_json(http.post(f"{env_url}/step", json=action))
                    if isinstance(result, dict):
                        obs = result.get("observation", obs) or obs
                        if result.get("done"):
                            break
                state = _safe_json(http.get(f"{env_url}/state"))
                score = float(state.get("final_score") or state.get("rubric_progress") or 0.0) if isinstance(state, dict) else 0.0
                scores[task_id] = score
                print(f"  {task_id:<42}  {score:.3f}")
            except Exception as exc:
                scores[task_id] = 0.0
                print(f"  {task_id:<42}  ERROR: {type(exc).__name__}: {exc!r}")

    mean_score = sum(scores.values()) / len(scores)
    print(f"\n  {'Mean':<42}  {mean_score:.3f}")
    print(f"  {'Reference baseline':<42}  {baseline_score:.3f}")
    print(f"  {'Delta':<42}  {mean_score - baseline_score:+.3f}")
    return scores, mean_score
