"""Starter GRPO training script for AegisDesk using TRL's OpenEnv integration.

This script is intentionally lightweight and does not change the benchmark or
submission path. It is an optional training starter that wraps the existing
AegisDesk environment as a TRL-compatible tool environment.

Install:
    pip install -e .
    pip install accelerate datasets transformers trl

Run:
    accelerate launch training/train_grpo_aegisdesk.py
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from client import SupportOpsEnv
from models import SupportAction, SupportObservation
from server.fixtures import canonical_benchmark_task_ids, private_variant_fixture_ids

ENV_URL = "https://i4mgr00t-meta.hf.space"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert B2B SaaS support operator working inside AegisDesk.

    Your goals are:
    - choose the correct ticket from the inbox
    - inspect the right records before acting
    - avoid unsafe or policy-violating shortcuts
    - use the provided tools precisely
    - finalize the case with the highest possible score

    Be careful, grounded, and conservative. If a situation looks security-sensitive,
    prioritize verification and escalation over direct fulfillment.
    """
).strip()

TASK_PROMPTS = {
    "billing_seat_adjustment": (
        "Resolve the billing issue carefully. Identify the right ticket, inspect the "
        "correct records, apply the exact credit if justified, and close the case safely."
    ),
    "login_incident_triage": (
        "Handle the login failure ticket during an active incident. Inspect incident "
        "context before acting and avoid unsafe account changes."
    ),
    "suspicious_admin_request": (
        "Work the inbox carefully. Detect the suspicious request, inspect verification "
        "and security context, refuse unsafe fulfillment, and escalate appropriately."
    ),
    "customer_escalation_chain": (
        "A billing dispute involves multiple billing cycles. A customer follow-up message "
        "will arrive mid-episode with corrected figures. Handle it, escalate for approval, "
        "and apply the correct total credit."
    ),
    "multi_tier_billing_dispute": (
        "Two parties have submitted conflicting seat counts. Find the authoritative signed "
        "document and apply the correct pro-rata credit. Do not act on verbal claims alone."
    ),
    "data_breach_response_lifecycle": (
        "A potential data breach has been detected. Follow the five-phase response protocol "
        "in order: detect, contain, assess, notify, resolve. Do not skip phases."
    ),
    "contract_renewal_negotiation": (
        "An enterprise renewal is blocked by two open issues: a billing dispute and an API "
        "incident. Resolve both sub-cases before finalizing the renewal."
    ),
    "service_reinstatement_review": (
        "A suspended account has paid. Verify payment and check the policy window. If the "
        "grace period is active, reinstate immediately without escalation."
    ),
    "api_partner_access_audit": (
        "A partner wants extended API access. A legal review window is active. Audit usage, "
        "check the contract, and route to billing_ops — do not self-approve."
    ),
}


def format_observation(observation: SupportObservation) -> str:
    """Render a compact text view of the environment state for the model."""

    inbox_lines = [
        (
            f"- {ticket.ticket_id}: {ticket.subject} | priority={ticket.priority.value} "
            f"| status={ticket.status.value} | tags={','.join(ticket.tags) or 'none'}"
        )
        for ticket in observation.inbox
    ]
    focus_block = "None"
    if observation.focus_panel is not None:
        focus_block = (
            f"{observation.focus_panel.title}\n"
            f"{observation.focus_panel.body}\n"
            f"panel_type={observation.focus_panel.panel_type}"
        )

    return textwrap.dedent(
        f"""
        Task brief:
        {observation.task_brief}

        Active ticket: {observation.active_ticket_id or 'none'}
        Available record ids: {', '.join(observation.available_record_ids) or 'none'}
        Step count: {observation.step_count}
        Remaining steps: {observation.remaining_steps}
        Last action error: {observation.last_action_error or 'none'}

        Inbox:
        {chr(10).join(inbox_lines)}

        Focus panel:
        {focus_block}
        """
    ).strip()


class AegisDeskToolEnv:
    """TRL environment wrapper that exposes meaningful support operations tools."""

    def __init__(self):
        self.client = SupportOpsEnv(base_url=ENV_URL)
        self.score = 0.0
        self.done = False
        self.task_id = ""
        self.fixture_id = ""

    def reset(
        self,
        task_id: str | None = "billing_seat_adjustment",
        fixture_id: str | None = None,
        seed: int = 1,
        **kwargs: Any,
    ) -> str:
        """Start a fresh support episode and return the initial observation text."""

        self.score = 0.0
        self.done = False
        self.fixture_id = fixture_id or task_id or "billing_seat_adjustment"
        self.task_id = task_id or self.fixture_id
        result = self.client.reset(task_id=task_id, fixture_id=fixture_id, seed=seed)
        self.fixture_id = result.observation.fixture_id or self.fixture_id
        self.task_id = result.observation.task_id or self.task_id
        return format_observation(result.observation)

    def open_ticket(self, ticket_id: str) -> str:
        """
        Open a support ticket and move focus to it.

        Args:
            ticket_id: The ticket id to open.

        Returns:
            The updated observation text after opening the ticket.
        """

        return self._run_action(
            SupportAction(action_type="open_ticket", ticket_id=ticket_id)
        )

    def inspect_record(self, record_id: str) -> str:
        """
        Inspect a record or knowledge base article by id.

        Args:
            record_id: The record id or article id to inspect.

        Returns:
            The updated observation text after inspecting the record.
        """

        return self._run_action(
            SupportAction(action_type="inspect_record", record_id=record_id)
        )

    def search_kb(self, query: str) -> str:
        """
        Search the knowledge base for relevant policy or troubleshooting information.

        Args:
            query: The search query to use for the KB lookup.

        Returns:
            The updated observation text after searching the knowledge base.
        """

        return self._run_action(SupportAction(action_type="search_kb", query=query))

    def set_priority(self, ticket_id: str, priority: str) -> str:
        """
        Update the priority of a ticket.

        Args:
            ticket_id: The ticket id to mutate.
            priority: One of low, normal, high, or urgent.

        Returns:
            The updated observation text after changing the ticket priority.
        """

        return self._run_action(
            SupportAction(
                action_type="set_priority",
                ticket_id=ticket_id,
                priority=priority,
            )
        )

    def set_status(self, ticket_id: str, status: str) -> str:
        """
        Update the workflow status of a ticket.

        Args:
            ticket_id: The ticket id to mutate.
            status: One of open, pending, waiting_on_customer, resolved, or escalated.

        Returns:
            The updated observation text after changing the ticket status.
        """

        return self._run_action(
            SupportAction(
                action_type="set_status",
                ticket_id=ticket_id,
                status=status,
            )
        )

    def add_tag(self, ticket_id: str, tag: str) -> str:
        """
        Add a support tag to a ticket.

        Args:
            ticket_id: The ticket id to mutate.
            tag: The tag to add to the ticket.

        Returns:
            The updated observation text after adding the tag.
        """

        return self._run_action(
            SupportAction(action_type="add_tag", ticket_id=ticket_id, tag=tag)
        )

    def apply_credit(self, ticket_id: str, amount: float, currency: str) -> str:
        """
        Apply a billing credit to a ticket when the case justifies it.

        Args:
            ticket_id: The billing ticket to mutate.
            amount: The credit amount to apply.
            currency: The billing currency, for example USD.

        Returns:
            The updated observation text after applying the credit.
        """

        return self._run_action(
            SupportAction(
                action_type="apply_credit",
                ticket_id=ticket_id,
                amount=amount,
                currency=currency,
            )
        )

    def escalate(self, ticket_id: str, escalation_team: str) -> str:
        """
        Escalate a ticket to a specialized team.

        Args:
            ticket_id: The ticket id to escalate.
            escalation_team: One of billing_ops, incident_response, or security.

        Returns:
            The updated observation text after escalating the ticket.
        """

        return self._run_action(
            SupportAction(
                action_type="escalate",
                ticket_id=ticket_id,
                escalation_team=escalation_team,
            )
        )

    def draft_reply(
        self,
        ticket_id: str,
        template_id: str,
        reply_checklist_csv: str,
        freeform_note: str = "",
    ) -> str:
        """
        Draft a structured customer reply using a template and checklist.

        Args:
            ticket_id: The ticket id the reply belongs to.
            template_id: The structured reply template id to use.
            reply_checklist_csv: A comma-separated list of checklist items to include.
            freeform_note: Optional freeform note for realism.

        Returns:
            The updated observation text after drafting the reply.
        """

        reply_checklist = [
            item.strip()
            for item in reply_checklist_csv.split(",")
            if item.strip()
        ]
        return self._run_action(
            SupportAction(
                action_type="draft_reply",
                ticket_id=ticket_id,
                template_id=template_id,
                reply_checklist=reply_checklist,
                freeform_note=freeform_note or None,
            )
        )

    def finalize_resolution(self, ticket_id: str, resolution_code: str) -> str:
        """
        Finalize the current case and request terminal scoring from the environment.

        Args:
            ticket_id: The ticket id being finalized.
            resolution_code: The final resolution code to submit.

        Returns:
            The final observation text, including any terminal score context.
        """

        return self._run_action(
            SupportAction(
                action_type="finalize_resolution",
                ticket_id=ticket_id,
                resolution_code=resolution_code,
            )
        )

    def _run_action(self, action: SupportAction) -> str:
        if self.done:
            raise ValueError(
                "Episode already completed. Call reset() to start a new support case."
            )

        result = self.client.step(action)
        state = self.client.state()
        self.done = result.done
        self.score = float(state.final_score or state.rubric_progress or 0.0)
        observation_text = format_observation(result.observation)
        if result.done:
            observation_text += (
                f"\n\nEpisode complete for fixture={self.fixture_id} task={self.task_id}. "
                f"Current score={self.score:.4f}."
            )
        return observation_text


def reward_func(environments, **kwargs) -> list[float]:
    """Return the current benchmark score for each running environment instance."""

    return [float(getattr(env, "score", 0.0)) for env in environments]


def load_rl_manifest(path: str | None) -> dict[str, Any] | None:
    """Load an optional RL manifest describing canonical and curriculum fixtures."""

    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_dataset(
    repeat_count: int,
    all_tasks: bool = False,
    rl_manifest_path: str | None = None,
):
    """Build a prompt dataset cycling through fixtures with optional curriculum support."""

    from datasets import Dataset

    manifest = load_rl_manifest(rl_manifest_path)
    if manifest:
        if all_tasks:
            fixture_ids = manifest.get(
                "allowed_grpo_fixture_ids",
                manifest.get("canonical_train_fixture_ids", canonical_benchmark_task_ids()),
            )
        else:
            fixture_ids = manifest.get("core_fixture_ids", canonical_benchmark_task_ids()[:3])
    else:
        fixture_ids = (
            canonical_benchmark_task_ids() + private_variant_fixture_ids()
            if all_tasks
            else canonical_benchmark_task_ids()[:3]
        )

    rows: list[dict[str, Any]] = []
    for seed in range(1, repeat_count + 1):
        for fixture_id in fixture_ids:
            task_id = fixture_id.split("_v", 1)[0] if "_v" in fixture_id else fixture_id
            task_prompt = TASK_PROMPTS[task_id]
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task_prompt},
                    ],
                    "fixture_id": fixture_id,
                    "task_id": task_id,
                    "seed": seed,
                }
            )
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training starter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Base model to train with GRPO.",
    )
    parser.add_argument(
        "--phase",
        choices=("stabilize", "champion"),
        default="champion",
        help="stabilize = canonical 9 only, champion = canonical 9 plus private curriculum fixtures.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Train on the canonical 9-task pack plus any private curriculum fixtures listed in the RL manifest.",
    )
    parser.add_argument(
        "--rl-manifest",
        default="training/support_rl_manifest.json",
        help="Optional RL manifest describing canonical, held-out, and curriculum fixture packs.",
    )
    parser.add_argument(
        "--env-url",
        default=ENV_URL,
        help="Base URL of the running AegisDesk environment.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/aegisdesk-grpo",
        help="Output directory for trainer artifacts.",
    )
    parser.add_argument(
        "--hub-model-id",
        default=None,
        help="Optional Hub repo id for pushing trainer outputs.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push outputs to the Hugging Face Hub when hub-model-id is provided.",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help="Trainer reporting backend, for example 'none' or 'trackio'.",
    )
    parser.add_argument(
        "--run-name",
        default="aegisdesk-grpo",
        help="Optional descriptive run name for trainer tracking.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=16,
        help="How many times to repeat tasks in the training dataset.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of GRPO training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for GRPO.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help="Number of completions generated per prompt.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=1024,
        help="Maximum total completion length across the tool-calling episode.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Logging frequency for the trainer.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the optional GRPO training starter."""

    global ENV_URL

    args = parse_args()
    ENV_URL = args.env_url

    from trl import GRPOConfig, GRPOTrainer

    use_all_tasks = args.all_tasks or args.phase == "champion"
    dataset = build_dataset(
        args.repeat_count,
        all_tasks=use_all_tasks,
        rl_manifest_path=args.rl_manifest,
    )
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=AegisDeskToolEnv,
        args=GRPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            logging_steps=args.logging_steps,
            remove_unused_columns=False,
            report_to=args.report_to,
            run_name=args.run_name,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            temperature=0.7,
            log_completions=True,
            chat_template_kwargs={"enable_thinking": False},
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
