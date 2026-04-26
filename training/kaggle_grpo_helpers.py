"""Helpers for the Kaggle GRPO notebook."""

from __future__ import annotations

from collections import defaultdict
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
        inbox = obs.get("inbox", [])
        inbox_lines = [
            f"  {ticket['ticket_id']}: {ticket['subject']} | {ticket['priority']} | {ticket['status']}"
            for ticket in inbox
        ]
        focus_panel = obs.get("focus_panel")
        focus = f"\nFocus: {focus_panel['title']}\n{focus_panel['body'][:500]}" if focus_panel else ""
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

    print(f"Running benchmark evaluation for {model_name} (max {max_steps} steps per task)...")
    scores: dict[str, float] = {}

    with httpx.Client(timeout=60, follow_redirects=True) as http:
        for task_id in tasks:
            try:
                reset_payload = http.post(
                    f"{env_url}/reset",
                    json={"task_id": task_id, "seed": seed},
                ).json()
                obs = reset_payload.get("observation", reset_payload)
                for _ in range(max_steps):
                    action = generate_action(format_obs_brief(obs))
                    result = http.post(f"{env_url}/step", json=action).json()
                    obs = result.get("observation", obs)
                    if result.get("done"):
                        break
                state = http.get(f"{env_url}/state").json()
                score = float(state.get("final_score") or state.get("rubric_progress") or 0.0)
                scores[task_id] = score
                print(f"  {task_id:<42}  {score:.3f}")
            except Exception as exc:
                scores[task_id] = 0.0
                print(f"  {task_id:<42}  ERROR: {exc}")

    mean_score = sum(scores.values()) / len(scores)
    print(f"\n  {'Mean':<42}  {mean_score:.3f}")
    print(f"  {'Reference baseline':<42}  {baseline_score:.3f}")
    print(f"  {'Delta':<42}  {mean_score - baseline_score:+.3f}")
    return scores, mean_score
