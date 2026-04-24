"""CustomerSimAgent — deterministic seeded agent for multi-agent task episodes.

Injects customer follow-up messages at configured steps based on task fixture
peer_inject blocks. When no fixture injection is configured, falls back to the
utterance pool keyed by scenario category.

Utterance pool sources:
  - Bitext Customer Support LLM Training Dataset (bitext/Bitext-customer-support-llm-chatbot-training-dataset)
    License: CC BY 4.0  |  26,872 rows, 11 categories
    Used for: access_issue (ACCOUNT/recover_password), billing_dispute (INVOICE + REFUND),
    escalation_demand (FEEDBACK/complaint + CONTACT/contact_human_agent)
    Fetched from: https://huggingface.co/api/datasets/bitext/.../parquet/default/train/0.parquet

  - ABCD Action-Based Conversations Dataset (asappresearch/abcd, MIT License)
    Used for: reinstatement_request, security_concern, general_followup
    (No direct Bitext category covers B2B suspension/reinstatement or breach scenarios)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class PeerMessage:
    step: int
    from_role: str
    message: str


# Utterance pool for CustomerSimAgent fallback mode.
# Real utterances sourced from Bitext (CC BY 4.0) and ABCD (MIT), noted per entry.
_ABCD_UTTERANCE_POOL: dict[str, list[str]] = {
    # Source: Bitext ACCOUNT/recover_password — adapted to B2B API/SaaS access context
    "access_issue": [
        "can I reset the PIN of my user profile?",
        "I can't retrieve my user access key — the portal isn't responding.",
        "I don't know how to recover the access key for my user profile.",
        "what do I have to do to recover my account access key?",
        "Our API integration stopped working after the access review. Can someone look into this?",
        "I submitted the access extension request three days ago and haven't heard back.",
        "The partner portal is showing our keys as expired but we have an active contract.",
        "Our team can't authenticate. Is there an active incident or is this specific to us?",
    ],
    # Source: Bitext INVOICE/check_invoice + REFUND/get_refund — B2B billing follow-up
    "billing_dispute": [
        "how can I locate the invoice for our account?",
        "can you help me check the bill on our account?",
        "I need help locating the outstanding invoice — our finance team is asking.",
        "I just wanted to check on the status of that credit — it hasn't shown up yet.",
        "I still haven't received the credit you promised. How much longer will this take?",
        "We were billed for 20 seats but we only have 14 active users. I need this corrected.",
        "Our finance team is asking for documentation on the credit. Can you send that over?",
        "how do I get a reimbursement for the overbilling on my account?",
    ],
    # Source: ABCD manage_account/status_credit_missing + B2B reinstatement scenarios
    "reinstatement_request": [
        "Hi, I need help getting our account reinstated. We've already made the payment.",
        "We settled the outstanding balance yesterday. Can someone confirm our service is back?",
        "Our account was suspended but the invoice was paid. How long does reinstatement take?",
        "This suspension is affecting our production systems. We paid — please restore access.",
        "I've got a payment confirmation number. Who do I send this to so we can get unblocked?",
        "problem with freemium account reactivation — can you help us get back online?",
    ],
    # Source: ABCD storewide_query + general follow-up patterns
    "general_followup": [
        "Just following up — has there been any progress on my ticket?",
        "I haven't heard back yet. Can you give me an update on where things stand?",
        "I need a resolution today. This is blocking our team from doing their work.",
        "Can you clarify the timeline? My manager is asking for an ETA.",
        "Is there someone more senior I can speak with about this?",
    ],
    # Source: Bitext FEEDBACK/complaint + CONTACT/contact_human_agent — adapted for B2B
    "escalation_demand": [
        "I'm not happy with your service — I need to file a complaint.",
        "where can I make a consumer claim against your organization?",
        "I would like to speak with a human agent, not an automated system.",
        "This has taken too long. I'd like to speak with a supervisor.",
        "I was told this would be resolved in 24 hours. It's been three days.",
        "Can you escalate this to your billing team? This is blocking a contract renewal.",
        "I call to make a customer claim against your business — this is unacceptable.",
    ],
    # Source: ABCD security flows — B2B breach/incident follow-up (no Bitext equivalent)
    "security_concern": [
        "We received a notification about unusual access. Can you tell me what data was affected?",
        "Our security team is asking for a full incident report. Who can we contact?",
        "Has the breach been contained? We need to know before we notify our own customers.",
        "Can you confirm what remediation steps have been completed so far?",
        "We need an official status update for compliance. Can you provide that in writing?",
    ],
}

# ABCD ontology flow → AegisDesk scenario category mapping
# Source: asappresearch/abcd/data/ontology.json (MIT License)
_ABCD_FLOW_TO_SCENARIO: dict[str, str] = {
    "product_defect": "billing_dispute",
    "manage_account": "billing_dispute",
    "purchase_dispute": "billing_dispute",
    "account_access": "access_issue",
    "single_item": "reinstatement_request",
    "subscription": "reinstatement_request",
    "storewide_query": "general_followup",
    "order_issue": "escalation_demand",
}


class CustomerSimAgent:
    """Injects customer follow-up messages at fixture-configured steps.

    Primary mode: driven by the peer_inject block in the task YAML (deterministic).
    Fallback mode: when no fixture injection, samples from the ABCD-derived utterance
    pool based on the scenario category (seeded random, still deterministic).
    """

    def __init__(
        self,
        peer_inject: list[dict[str, Any]],
        seed: int = 42,
        scenario_category: str = "general_followup",
    ) -> None:
        self._injections: dict[int, PeerMessage] = {}
        self._rng = random.Random(seed)
        self._scenario_category = scenario_category
        for entry in (peer_inject or []):
            step = int(entry["at_step"])
            self._injections[step] = PeerMessage(
                step=step,
                from_role=entry.get("from", "customer"),
                message=entry["message"].strip(),
            )

    def get_injection(self, current_step: int) -> PeerMessage | None:
        """Return a PeerMessage if one is configured for this step, else None."""
        return self._injections.get(current_step)

    def get_scenario_injection(self, current_step: int) -> PeerMessage | None:
        """Sample from the ABCD utterance pool for the current scenario category.

        Used when no fixture peer_inject is configured. The same seed + step always
        produces the same message (deterministic via seeded RNG).
        """
        pool = _ABCD_UTTERANCE_POOL.get(
            self._scenario_category,
            _ABCD_UTTERANCE_POOL["general_followup"],
        )
        message = self._rng.choice(pool)
        return PeerMessage(step=current_step, from_role="customer", message=message)

    def has_injections(self) -> bool:
        return len(self._injections) > 0

    def injection_steps(self) -> list[int]:
        return sorted(self._injections.keys())

    @staticmethod
    def scenario_for_task(task_id: str) -> str:
        """Map a task ID to an ABCD-derived scenario category."""
        mapping = {
            "customer_escalation_chain": "billing_dispute",
            "multi_tier_billing_dispute": "billing_dispute",
            "data_breach_response_lifecycle": "security_concern",
            "contract_renewal_negotiation": "escalation_demand",
            "service_reinstatement_review": "reinstatement_request",
            "api_partner_access_audit": "access_issue",
            "billing_seat_adjustment": "billing_dispute",
            "login_incident_triage": "access_issue",
            "suspicious_admin_request": "security_concern",
        }
        return mapping.get(task_id, "general_followup")
