"""
Fixture generator for AegisDesk v2.

Calls any OpenAI-compatible API (default: HF router) to generate variant episodes
for every existing task fixture. Turns 9 tasks into ~50 rich episodes.

Usage:
    # dry-run: show what would be generated, no API calls
    python scripts/generate_fixtures.py --dry-run

    # generate 4 variants per task (uses HF_TOKEN + HF router by default)
    python scripts/generate_fixtures.py --variants 4

    # use a specific model
    python scripts/generate_fixtures.py --variants 3 --model Qwen/Qwen3-4B

    # point at a different endpoint (e.g. local Ollama)
    python scripts/generate_fixtures.py --variants 2 --base-url http://localhost:11434/v1 --api-key ollama

Important:
    Fixture identity is now keyed by `fixture_id`, so generated variants can coexist safely with
    canonical fixtures. Public surfacing is controlled separately through `GENERALIZATION_FIXTURE_IDS`
    in `server/fixtures.py`; generated files are private by default until promoted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml  # pip install pyyaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
TASK_DATA_DIR = REPO_ROOT / "server" / "task_data"
OUTPUT_DIR = TASK_DATA_DIR  # generated variants go in the same directory

CORE_TASKS = [
    "billing_seat_adjustment",
    "login_incident_triage",
    "suspicious_admin_request",
]
V2_TASKS = [
    "customer_escalation_chain",
    "multi_tier_billing_dispute",
    "data_breach_response_lifecycle",
    "contract_renewal_negotiation",
    "service_reinstatement_review",
    "api_partner_access_audit",
]
ALL_TASKS = CORE_TASKS + V2_TASKS

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"

# ---------------------------------------------------------------------------
# Company / contact name pools for variation
# ---------------------------------------------------------------------------

COMPANY_POOL = [
    ("Orbix Technologies", "orbix"),
    ("Vantara Systems", "vantara"),
    ("ClearPath SaaS", "clearpath"),
    ("Nexflow Analytics", "nexflow"),
    ("Stackform Inc", "stackform"),
    ("Luminary Cloud", "luminary"),
    ("Driftline Software", "driftline"),
    ("Meridian Data", "meridian"),
    ("Irongate Solutions", "irongate"),
    ("Skyward Platforms", "skyward"),
    ("Crestline Tech", "crestline"),
    ("Bluestone SaaS", "bluestone"),
    ("Polygon Systems", "polygon"),
    ("Helios Analytics", "helios"),
    ("Vortex Cloud", "vortex"),
]

CONTACT_POOL = [
    ("Marcus Webb", "marcus"),
    ("Priya Nair", "priya"),
    ("Leon Fischer", "leon"),
    ("Amara Osei", "amara"),
    ("Chen Wei", "chen"),
    ("Sofia Reyes", "sofia"),
    ("Tariq Mansoor", "tariq"),
    ("Ingrid Larsson", "ingrid"),
    ("Kwame Asante", "kwame"),
    ("Yuki Tanaka", "yuki"),
    ("Fatima Al-Rashidi", "fatima"),
    ("Dmitri Volkov", "dmitri"),
    ("Chiara Ricci", "chiara"),
    ("Bao Nguyen", "bao"),
    ("Zara Ahmed", "zara"),
]

# ---------------------------------------------------------------------------
# Generator prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a fixture generator for AegisDesk, a B2B SaaS support operations benchmark.
Your job is to produce a YAML fixture file that is a realistic variation of an existing task.

Rules:
- Keep the EXACT same YAML schema and field names as the reference fixture
- Change: company names, contact names, ticket amounts/dates, account IDs, scenario specifics
- Keep: the same rubric structure, same check_id names, same forbidden_action logic, same world_context structure
- Make the scenario feel real: use plausible company names, realistic dollar amounts, real-sounding email addresses
- The oracle_reference_path must still be the correct solution for the new scenario
- Do NOT change the task_id, difficulty, max_steps, or world_context.policy_window.active value
- Vary the overcharge amounts, seat counts, ticket IDs (use TICKET-XXXX format with new numbers), record IDs, etc.
- Output ONLY valid YAML, no markdown fences, no explanation text"""

USER_TEMPLATE = """Here is the reference fixture for task `{task_id}`:

---REFERENCE FIXTURE---
{reference_yaml}
---END REFERENCE FIXTURE---

Generate variant #{variant_num} for this task.
- Company name: {company_name} (account slug: acct_{company_slug})
- Contact name: {contact_name}
- Variant seed: {seed}
- Change the ticket IDs, amounts, dates, and company details
- Keep all rubric check_ids and weights identical
- Keep all forbidden_action logic identical
- Output only the YAML, nothing else."""

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def call_llm(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4096,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/{retries}] error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def extract_yaml(raw: str) -> str:
    """Strip markdown fences if the model wrapped the output."""
    # Remove ```yaml ... ``` or ``` ... ``` wrappers
    clean = re.sub(r"^```(?:yaml)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean.strip())
    return clean.strip()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_fixture(data: dict, reference: dict, task_id: str) -> list[str]:
    """Return a list of validation errors (empty = OK)."""
    errors = []
    if data.get("task_id") != task_id:
        errors.append(f"task_id changed: got {data.get('task_id')!r}, expected {task_id!r}")

    ref_check_ids = {c["check_id"] for c in reference.get("rubric", [])}
    gen_check_ids = {c["check_id"] for c in data.get("rubric", [])}
    missing = ref_check_ids - gen_check_ids
    if missing:
        errors.append(f"rubric check_ids missing: {missing}")
    extra = gen_check_ids - ref_check_ids
    if extra:
        errors.append(f"rubric check_ids added (not allowed): {extra}")

    ref_weights = sum(c.get("weight", 0) for c in reference.get("rubric", []))
    gen_weights = sum(c.get("weight", 0) for c in data.get("rubric", []))
    if abs(ref_weights - gen_weights) > 0.01:
        errors.append(f"rubric weight sum changed: {ref_weights:.2f} -> {gen_weights:.2f}")

    if "tickets" not in data or len(data["tickets"]) == 0:
        errors.append("no tickets in generated fixture")
    if "records" not in data or len(data["records"]) == 0:
        errors.append("no records in generated fixture")
    if "rubric" not in data:
        errors.append("no rubric in generated fixture")

    return errors


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_variants(
    task_id: str,
    n_variants: int,
    client: Any,
    model: str,
    dry_run: bool,
    rng: random.Random,
) -> list[Path]:
    reference_path = TASK_DATA_DIR / f"{task_id}.yaml"
    if not reference_path.exists():
        print(f"  [skip] {task_id}: reference fixture not found at {reference_path}")
        return []

    reference_yaml = reference_path.read_text(encoding="utf-8")
    try:
        reference_data = yaml.safe_load(reference_yaml)
    except yaml.YAMLError as e:
        print(f"  [skip] {task_id}: could not parse reference YAML: {e}")
        return []

    generated_paths = []
    companies = rng.sample(COMPANY_POOL, min(n_variants, len(COMPANY_POOL)))
    contacts = rng.sample(CONTACT_POOL, min(n_variants, len(CONTACT_POOL)))

    for i in range(n_variants):
        company_name, company_slug = companies[i % len(companies)]
        contact_name, _ = contacts[i % len(contacts)]
        seed = rng.randint(1000, 9999)
        variant_id = f"{task_id}_v{i+1}"
        output_path = OUTPUT_DIR / f"{variant_id}.yaml"

        if output_path.exists():
            print(f"  [skip] {variant_id}: already exists")
            generated_paths.append(output_path)
            continue

        print(f"  [{i+1}/{n_variants}] generating {variant_id} "
              f"({company_name} / {contact_name})...", end=" ", flush=True)

        if dry_run:
            print("DRY-RUN (skipped)")
            continue

        user_prompt = USER_TEMPLATE.format(
            task_id=task_id,
            reference_yaml=reference_yaml,
            variant_num=i + 1,
            company_name=company_name,
            company_slug=company_slug,
            contact_name=contact_name,
            seed=seed,
        )

        try:
            raw = call_llm(client, model, SYSTEM_PROMPT, user_prompt)
            yaml_text = extract_yaml(raw)

            # Parse and validate
            generated_data = yaml.safe_load(yaml_text)
            if not isinstance(generated_data, dict):
                print(f"INVALID (not a dict)")
                continue

            # Force task_id to match (model sometimes changes it)
            generated_data["task_id"] = task_id

            errors = validate_fixture(generated_data, reference_data, task_id)
            if errors:
                print(f"VALIDATION ERRORS:")
                for err in errors:
                    print(f"      - {err}")
                # Save anyway with a .rejected suffix for manual review
                rejected_path = output_path.with_suffix(".rejected.yaml")
                rejected_path.write_text(
                    yaml.dump(generated_data, allow_unicode=True, default_flow_style=False),
                    encoding="utf-8",
                )
                print(f"      saved to {rejected_path.name} for review")
                continue

            # Write the validated fixture
            output_path.write_text(
                yaml.dump(generated_data, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )
            generated_paths.append(output_path)
            print(f"OK -> {output_path.name}")

        except Exception as e:
            print(f"ERROR: {e}")

    return generated_paths


# ---------------------------------------------------------------------------
# fixtures.py registration
# ---------------------------------------------------------------------------


def register_variants_in_fixtures(new_task_ids: list[str]) -> None:
    """Explain how generated variants are surfaced under the fixture-id model."""

    print("[register] No automatic public-pack mutation is required.")
    print("[register] Generated fixtures already load by filename stem as fixture_id values.")
    if new_task_ids:
        print("[register] Promote any held-out fixtures manually by editing GENERALIZATION_FIXTURE_IDS:")
        for tid in new_task_ids:
            print(f"  - {tid}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixture variants for AegisDesk tasks using an LLM."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=ALL_TASKS,
        metavar="TASK_ID",
        help="Tasks to generate variants for (default: all 9 tasks)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=4,
        help="Number of variants to generate per task (default: 4)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: reads HF_TOKEN env var, then OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for company/contact selection (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without making API calls",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip the post-generation reminder about promoting fixtures into public benchmark packs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve API key
    api_key = (
        args.api_key
        or os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key and not args.dry_run:
        print("ERROR: No API key found. Set HF_TOKEN or OPENAI_API_KEY, or pass --api-key.")
        sys.exit(1)

    # Build OpenAI client
    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or "dummy", base_url=args.base_url)
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)

    rng = random.Random(args.seed)

    print(f"AegisDesk Fixture Generator")
    print(f"  Tasks     : {len(args.tasks)}")
    print(f"  Variants  : {args.variants} per task -> up to {len(args.tasks) * args.variants} new fixtures")
    print(f"  Model     : {args.model}")
    print(f"  Base URL  : {args.base_url}")
    print(f"  Dry run   : {args.dry_run}")
    print()

    all_generated_ids: list[str] = []

    for task_id in args.tasks:
        print(f"Task: {task_id}")
        generated_paths = generate_variants(
            task_id=task_id,
            n_variants=args.variants,
            client=client,
            model=args.model,
            dry_run=args.dry_run,
            rng=rng,
        )
        for p in generated_paths:
            # Derive the task_id variant registered name from stem
            # e.g. billing_seat_adjustment_v1
            all_generated_ids.append(p.stem)
        print()

    if not args.dry_run and not args.no_register and all_generated_ids:
        register_variants_in_fixtures(all_generated_ids)

    total = len(all_generated_ids)
    print(f"\nDone. {total} variant fixture(s) generated.")
    if args.dry_run:
        print("(dry-run — no files written, no API calls made)")


if __name__ == "__main__":
    main()
