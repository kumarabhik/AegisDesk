"""Fetch and normalize real-world dataset artifacts used by AegisDesk.

Downloads and prepares six external sources for the Round 2 training story:

1. Bitext customer support prompts/responses
2. ABCD action-constrained customer support dialogues
3. Sierra tau-bench and tau2-bench few-shot trajectories
4. Google Schema-Guided Dialogue (SGD) train conversations
5. NVIDIA HelpSteer2 preference pairs
6. Optional DialogStudio / MultiWOZ examples

Derived outputs:
- training/raw/*
- training/data/support_sft.jsonl
- training/data/support_pref.jsonl
- training/data/dataset_build_report.json
- training/support_rl_manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "training" / "data"
RAW_DIR = ROOT / "training" / "raw"
RL_MANIFEST_PATH = ROOT / "training" / "support_rl_manifest.json"
DATASET_BUILD_REPORT_PATH = DATA_DIR / "dataset_build_report.json"

BITEXT_PARQUET_URL = (
    "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
ABCD_DATA_URL = "https://raw.githubusercontent.com/asappresearch/abcd/master/data/abcd_v1.1.json.gz"
ABCD_ONTOLOGY_URL = "https://raw.githubusercontent.com/asappresearch/abcd/master/data/ontology.json"
TAUBENCH_SOURCES = {
    "tau2-bench-retail": "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/few_shot_data/MockRetailDomainEnv-few_shot.jsonl",
    "tau-bench-retail": "https://raw.githubusercontent.com/sierra-research/tau-bench/main/few_shot_data/MockRetailDomainEnv-few_shot.jsonl",
}
SGD_ARCHIVE_URL = (
    "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/"
    "archive/refs/heads/master.zip"
)
HELPSTEER2_PREF_URL = (
    "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/preference/preference.jsonl.gz"
)

OPTIONAL_DIALOGSTUDIO_DATASET = ("Salesforce/dialogstudio", "MULTIWOZ2_2")

SFT_TARGET_MIN = 15_000
SFT_TARGET_MAX = 20_000
PREF_TARGET_MIN = 5_000

BITEXT_SFT_CAP = 6_000
ABCD_SFT_CAP = 5_000
SGD_SFT_CAP = 4_000
OPTIONAL_DIALOGSTUDIO_CAP = 300

BITEXT_LICENSE = "cdla-sharing-1.0"
ABCD_LICENSE = "MIT"
TAU_LICENSE = "MIT"
SGD_LICENSE = "CC-BY-SA-4.0"
HELPSTEER2_LICENSE = "CC-BY-4.0"
AEGISDESK_LICENSE = "project-benchmark-fixture"

BITEXT_INTENT_MAP: dict[tuple[str, str], str] = {
    ("ACCOUNT", "recover_password"): "access_issue",
    ("ACCOUNT", "registration_problems"): "access_issue",
    ("INVOICE", "check_invoice"): "billing_dispute",
    ("INVOICE", "get_invoice"): "billing_dispute",
    ("PAYMENT", "payment_issue"): "billing_dispute",
    ("REFUND", "get_refund"): "billing_dispute",
    ("REFUND", "check_refund_policy"): "billing_dispute",
    ("REFUND", "track_refund"): "billing_dispute",
    ("CONTACT", "contact_human_agent"): "escalation_demand",
    ("FEEDBACK", "complaint"): "escalation_demand",
}

SYSTEM_PROMPT = (
    "You are an expert B2B SaaS support operator working inside AegisDesk. "
    "Choose safe, policy-compliant actions and communicate clearly."
)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AegisDesk/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def probe_url(url: str) -> bool:
    headers = {"User-Agent": "AegisDesk/1.0", "Range": "bytes=0-255"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read(16)
    return True


def fetch_json(url: str) -> Any:
    return json.loads(fetch(url).decode("utf-8"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def make_sft_row(
    *,
    source: str,
    license_name: str,
    split_role: str,
    messages: list[dict[str, str]],
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "source": source,
        "license": license_name,
        "split_role": split_role,
        "messages": messages,
        **metadata,
    }


def make_pref_row(
    *,
    source: str,
    license_name: str,
    split_role: str,
    prompt: str,
    chosen: str,
    rejected: str,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "source": source,
        "license": license_name,
        "split_role": split_role,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        **metadata,
    }


def _parse_parquet_bytes(raw: bytes) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []

    table = pq.read_table(io.BytesIO(raw))
    columns = table.to_pydict()
    row_count = len(next(iter(columns.values()))) if columns else 0
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row = {name: values[index] for name, values in columns.items()}
        rows.append(row)
    return rows


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _safety_slice_for_task(task_id: str) -> str:
    if task_id in {"suspicious_admin_request", "api_partner_access_audit"}:
        return "security"
    return "general"


def fetch_bitext(verify_only: bool) -> bool:
    print("Fetching Bitext customer support dataset...")
    if verify_only:
        try:
            probe_url(BITEXT_PARQUET_URL)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return False
        print("  [verify-only] Bitext reachable")
        return True

    try:
        raw = fetch(BITEXT_PARQUET_URL)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False

    print(f"  Downloaded {len(raw) / 1024 / 1024:.1f} MB")

    ensure_dirs()
    (RAW_DIR / "bitext_customer_support.parquet").write_bytes(raw)
    rows = _parse_parquet_bytes(raw)
    if not rows:
        print("  WARNING: pyarrow not installed; saved raw parquet only.")
        return False

    pool: dict[str, list[str]] = {}
    sft_rows: list[dict[str, Any]] = []
    for row in rows:
        category = str(row.get("category") or "")
        intent = str(row.get("intent") or "")
        instruction = str(row.get("instruction") or "").strip()
        response = str(row.get("response") or "").strip()
        key = (category, intent)
        if key in BITEXT_INTENT_MAP and 20 < len(instruction) < 220 and response and "{{" not in response:
            scenario = BITEXT_INTENT_MAP[key]
            pool.setdefault(scenario, [])
            if instruction not in pool[scenario] and len(pool[scenario]) < 32:
                pool[scenario].append(instruction)
            sft_rows.append(
                make_sft_row(
                    source="bitext-customer-support",
                    license_name=BITEXT_LICENSE,
                    split_role="sft_train",
                    category=category,
                    intent=intent,
                    scenario=scenario,
                    messages=[
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response},
                    ],
                )
            )
        if len(sft_rows) >= BITEXT_SFT_CAP:
            break

    write_json(
        DATA_DIR / "bitext_utterance_pool.json",
        {
            "source": "bitext-customer-support",
            "license": BITEXT_LICENSE,
            "url": BITEXT_PARQUET_URL,
            "rows_total": len(rows),
            "rows_used": len(sft_rows),
            "pool": pool,
        },
    )
    write_jsonl(DATA_DIR / "bitext_support_sft.jsonl", sft_rows)
    print(f"  Saved Bitext utterance pool and {len(sft_rows)} SFT rows")
    return True


def fetch_abcd(verify_only: bool) -> bool:
    print("Fetching ABCD full dataset + ontology...")
    if verify_only:
        try:
            probe_url(ABCD_DATA_URL)
            probe_url(ABCD_ONTOLOGY_URL)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return False
        print("  [verify-only] ABCD reachable")
        return True

    try:
        raw = fetch(ABCD_DATA_URL)
        ontology = json.loads(fetch(ABCD_ONTOLOGY_URL))
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False

    ensure_dirs()
    (RAW_DIR / "abcd_v1.1.json.gz").write_bytes(raw)
    write_json(RAW_DIR / "abcd_ontology.json", ontology)

    payload = json.loads(gzip.decompress(raw).decode("utf-8"))
    train_rows = payload.get("train", [])

    flow_map = {
        "product_defect": "billing_dispute",
        "purchase_dispute": "billing_dispute",
        "manage_account": "billing_dispute",
        "account_access": "access_issue",
        "single_item": "reinstatement_request",
        "subscription": "reinstatement_request",
        "storewide_query": "general_followup",
        "order_issue": "escalation_demand",
    }
    utterance_pool: dict[str, list[str]] = {category: [] for category in set(flow_map.values())}
    sft_rows: list[dict[str, Any]] = []
    seen_utterances: set[str] = set()
    for episode in train_rows:
        scenario = episode.get("scenario", {})
        flow = str(scenario.get("flow") or "general_followup")
        mapped_category = flow_map.get(flow, "general_followup")
        messages: list[dict[str, str]] = []
        for role, utterance in episode.get("original", []):
            text = str(utterance or "").strip()
            if not text:
                continue
            if role == "customer":
                messages.append({"role": "user", "content": text})
                if 20 < len(text) < 240 and text not in seen_utterances:
                    utterance_pool.setdefault(mapped_category, []).append(text)
                    seen_utterances.add(text)
            elif role == "agent":
                messages.append({"role": "assistant", "content": text})
        messages = _normalize_messages(messages)
        if len(messages) >= 2:
            sft_rows.append(
                make_sft_row(
                    source="abcd",
                    license_name=ABCD_LICENSE,
                    split_role="sft_train",
                    flow=flow,
                    scenario_category=mapped_category,
                    messages=messages,
                )
            )
        if len(sft_rows) >= ABCD_SFT_CAP:
            break

    actions = ontology.get("actions", {})
    action_summary = {name: list(values.keys()) for name, values in actions.items() if isinstance(values, dict)}
    write_json(
        DATA_DIR / "abcd_utterance_pool.json",
        {
            "source": "abcd",
            "license": ABCD_LICENSE,
            "url": ABCD_DATA_URL,
            "rows_total": len(train_rows),
            "rows_used": len(sft_rows),
            "pool": utterance_pool,
        },
    )
    write_json(
        DATA_DIR / "abcd_action_taxonomy.json",
        {
            "source": "abcd-ontology",
            "license": ABCD_LICENSE,
            "url": ABCD_ONTOLOGY_URL,
            "actions": action_summary,
        },
    )
    write_jsonl(DATA_DIR / "abcd_support_sft.jsonl", sft_rows)
    print(f"  Saved ABCD utterances, ontology summary, and {len(sft_rows)} SFT rows")
    return True


def _parse_tau_messages(convo: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []
    for line in convo.splitlines():
        match = re.match(r"^(user|assistant|tool): (.*)", line)
        if match:
            if current_role:
                turns.append({"role": current_role, "content": " ".join(current_lines).strip()})
            current_role = match.group(1)
            current_lines = [match.group(2)]
        elif current_role and line.strip():
            current_lines.append(line.strip())
    if current_role:
        turns.append({"role": current_role, "content": " ".join(current_lines).strip()})
    return [
        turn
        for turn in turns
        if turn["role"] in {"user", "assistant"} and turn["content"] and turn["content"] != "None"
    ]


def fetch_taubench(verify_only: bool) -> bool:
    print("Fetching Sierra tau-bench and tau2-bench few-shot trajectories...")
    raw_payloads: list[tuple[str, bytes, str]] = []
    if verify_only:
        ok_any = False
        try:
            for source_name, source_url in TAUBENCH_SOURCES.items():
                try:
                    probe_url(source_url)
                    print(f"  [verify-only] {source_name} reachable via {source_url}")
                    ok_any = True
                except Exception as exc:
                    print(f"  WARNING: {source_name} probe failed: {exc}")
            return ok_any
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return False

    for source_name, source_url in TAUBENCH_SOURCES.items():
        try:
            raw_payloads.append((source_name, fetch(source_url), source_url))
        except Exception as exc:
            print(f"  WARNING fetching {source_name}: {exc}")
    if not raw_payloads:
        return False

    ensure_dirs()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_name, raw, source_url in raw_payloads:
        (RAW_DIR / f"{source_name}_few_shot.jsonl").write_bytes(raw)
        lines = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        for episode in lines:
            messages = _normalize_messages(_parse_tau_messages(str(episode.get("messages_display") or "")))
            if len(messages) < 2:
                continue
            key = json.dumps(messages, sort_keys=True, ensure_ascii=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                make_sft_row(
                    source=source_name,
                    license_name=TAU_LICENSE,
                    split_role="sft_train",
                    source_url=source_url,
                    messages=messages,
                )
            )

    write_jsonl(DATA_DIR / "taubench_sft.jsonl", rows)
    print(f"  Saved {len(rows)} tau-bench / tau2-bench SFT rows")
    return True


def fetch_schema_guided_dialogue(verify_only: bool) -> bool:
    print("Fetching Schema-Guided Dialogue archive...")
    if verify_only:
        try:
            probe_url(SGD_ARCHIVE_URL)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return False
        print("  [verify-only] SGD archive reachable")
        return True

    try:
        raw = fetch(SGD_ARCHIVE_URL)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False

    ensure_dirs()
    (RAW_DIR / "schema_guided_dialogue.zip").write_bytes(raw)

    sft_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            if "/train/" not in name or not name.endswith(".json") or name.endswith("schema.json"):
                continue
            payload = json.loads(archive.read(name))
            if not isinstance(payload, list):
                continue
            for dialogue in payload:
                messages = []
                for turn in dialogue.get("turns", []):
                    utterance = str(turn.get("utterance") or "").strip()
                    if not utterance:
                        continue
                    speaker = str(turn.get("speaker") or "").strip().lower()
                    if speaker == "user":
                        messages.append({"role": "user", "content": utterance})
                    elif speaker in {"system", "assistant", "agent"}:
                        messages.append({"role": "assistant", "content": utterance})
                messages = _normalize_messages(messages)
                if len(messages) >= 2:
                    sft_rows.append(
                        make_sft_row(
                            source="schema-guided-dialogue",
                            license_name=SGD_LICENSE,
                            split_role="sft_train",
                            services=dialogue.get("services", []),
                            dialogue_id=dialogue.get("dialogue_id"),
                            messages=messages,
                        )
                    )
                if len(sft_rows) >= SGD_SFT_CAP:
                    break
            if len(sft_rows) >= SGD_SFT_CAP:
                break

    write_jsonl(DATA_DIR / "sgd_support_sft.jsonl", sft_rows)
    print(f"  Saved {len(sft_rows)} SGD SFT rows")
    return True


def fetch_helpsteer2(verify_only: bool) -> bool:
    print("Fetching HelpSteer2 preference pairs...")
    if verify_only:
        try:
            probe_url(HELPSTEER2_PREF_URL)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return False
        print("  [verify-only] HelpSteer2 preference archive reachable")
        return True

    try:
        raw = fetch(HELPSTEER2_PREF_URL)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False

    ensure_dirs()
    (RAW_DIR / "helpsteer2_preference.jsonl.gz").write_bytes(raw)
    pref_rows: list[dict[str, Any]] = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
        for line in handle:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            response_1 = str(row.get("response_1") or "").strip()
            response_2 = str(row.get("response_2") or "").strip()
            if not prompt or not response_1 or not response_2:
                continue
            try:
                preference_strength = float(row.get("preference_strength") or 0.0)
            except (TypeError, ValueError):
                preference_strength = 0.0
            if preference_strength == 0:
                continue
            chosen = response_2 if preference_strength > 0 else response_1
            rejected = response_1 if preference_strength > 0 else response_2
            pref_rows.append(
                make_pref_row(
                    source="helpsteer2-preference",
                    license_name=HELPSTEER2_LICENSE,
                    split_role=f"preference_{row.get('split', 'train')}",
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    preference_strength=preference_strength,
                    preference_statement=row.get("preference_statement"),
                    preference_elaboration=row.get("preference_elaboration"),
                )
            )

    write_jsonl(DATA_DIR / "helpsteer2_pref.jsonl", pref_rows)
    print(f"  Saved {len(pref_rows)} HelpSteer2 preference rows")
    return True


def fetch_optional_dialogstudio(verify_only: bool, include_optional: bool) -> bool:
    if not include_optional:
        print("Skipping optional DialogStudio / MultiWOZ fetch.")
        return True

    print("Fetching optional DialogStudio / MultiWOZ examples...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  WARNING: datasets library not installed; skipping optional DialogStudio fetch.")
        return False

    dataset = load_dataset(*OPTIONAL_DIALOGSTUDIO_DATASET)
    train_split = dataset["train"]
    if verify_only:
        print(f"  [verify-only] DialogStudio rows available: {len(train_split)}")
        return True

    rows: list[dict[str, Any]] = []
    for row in train_split.select(range(min(len(train_split), OPTIONAL_DIALOGSTUDIO_CAP))):
        prompt = str(
            row.get("prompt")
            or row.get("original dialog info")
            or "Review this task-oriented support conversation."
        ).strip()
        rows.append(
            make_sft_row(
                source="dialogstudio-multiwoz",
                license_name="see-upstream-dataset-card",
                split_role="sft_train_optional",
                messages=[
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": json.dumps(row.get("log", []), ensure_ascii=True)[:4000],
                    },
                ],
            )
        )
    write_jsonl(DATA_DIR / "dialogstudio_multiwoz_sample.jsonl", rows)
    print(f"  Saved {len(rows)} optional DialogStudio rows")
    return True


def build_aegisdesk_oracle_sft() -> list[dict[str, Any]]:
    from oracle_tools import build_oracle_actions
    from server.fixtures import get_fixture, training_curriculum_fixture_ids

    rows: list[dict[str, Any]] = []
    for fixture_id in training_curriculum_fixture_ids():
        fixture = get_fixture(fixture_id=fixture_id)
        prior_steps: list[str] = []
        for step_index, action in enumerate(build_oracle_actions(fixture_id), start=1):
            prompt = (
                f"Task brief:\n{fixture.task_brief}\n\n"
                f"Fixture id: {fixture.fixture_id}\n"
                f"Task family: {fixture.task_id}\n"
                f"Primary ticket: {fixture.primary_ticket_id}\n"
                f"Reply template: {fixture.reply_requirements.template_id}\n"
                f"Completed oracle steps so far: {prior_steps or ['none']}\n\n"
                f"Return the next best structured support action as JSON for oracle step {step_index}."
            )
            action_payload = action.model_dump(mode="json", exclude_none=True)
            rows.append(
                make_sft_row(
                    source="aegisdesk-oracle",
                    license_name=AEGISDESK_LICENSE,
                    split_role="sft_train",
                    fixture_id=fixture.fixture_id,
                    task_id=fixture.task_id,
                    safety_slice=_safety_slice_for_task(fixture.task_id),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(action_payload, separators=(",", ":"))},
                    ],
                )
            )
            prior_steps.append(json.dumps(action_payload, separators=(",", ":")))
    return rows


def build_support_sft(include_optional: bool) -> int:
    inputs = [
        DATA_DIR / "bitext_support_sft.jsonl",
        DATA_DIR / "abcd_support_sft.jsonl",
        DATA_DIR / "taubench_sft.jsonl",
        DATA_DIR / "sgd_support_sft.jsonl",
    ]
    if include_optional:
        inputs.append(DATA_DIR / "dialogstudio_multiwoz_sample.jsonl")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in inputs:
        for row in read_jsonl(path):
            key = json.dumps(row.get("messages", row), sort_keys=True, ensure_ascii=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    for row in build_aegisdesk_oracle_sft():
        key = json.dumps(row["messages"], sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    write_jsonl(DATA_DIR / "support_sft.jsonl", rows)
    print(f"Built combined support_sft.jsonl with {len(rows)} rows")
    return len(rows)


def build_support_pref() -> int:
    rows: list[dict[str, Any]] = []
    rows.extend(read_jsonl(DATA_DIR / "helpsteer2_pref.jsonl"))

    for dpo_path in sorted(DATA_DIR.glob("dpo_pairs_*.jsonl")):
        for row in read_jsonl(dpo_path):
            task_id = str(row.get("task_id") or "")
            rows.append(
                make_pref_row(
                    source="aegisdesk-dpo-pairs",
                    license_name=AEGISDESK_LICENSE,
                    split_role="preference_train",
                    task_id=task_id,
                    fixture_id=row.get("fixture_id"),
                    safety_slice=row.get("safety_slice") or _safety_slice_for_task(task_id),
                    prompt=row.get("prompt", ""),
                    chosen=row.get("chosen", ""),
                    rejected=row.get("rejected", ""),
                    chosen_score=row.get("chosen_score"),
                    rejected_score=row.get("rejected_score"),
                )
            )

    write_jsonl(DATA_DIR / "support_pref.jsonl", rows)
    print(f"Built combined support_pref.jsonl with {len(rows)} rows")
    return len(rows)


def write_rl_manifest() -> None:
    from server.fixtures import (
        benchmark_task_ids,
        canonical_benchmark_task_ids,
        generalization_fixture_ids,
        ordered_task_ids,
        private_variant_fixture_ids,
        security_slice_fixture_ids,
        showcase_fixture_ids,
        training_curriculum_fixture_ids,
        v2_task_ids,
    )

    payload = {
        "benchmark": "AegisDesk",
        "version": 2,
        "story": (
            "Train on 9 canonical enterprise tasks, evaluate on 18 held-out judged variants, "
            "and keep 3 showcase fixtures outside the main score-report pack."
        ),
        "core_fixture_ids": ordered_task_ids(),
        "v2_fixture_ids": v2_task_ids(),
        "canonical_train_fixture_ids": canonical_benchmark_task_ids(),
        "held_out_generalization_fixture_ids": generalization_fixture_ids(),
        "showcase_fixture_ids": showcase_fixture_ids(),
        "private_curriculum_fixture_ids": private_variant_fixture_ids(),
        "allowed_grpo_fixture_ids": training_curriculum_fixture_ids(),
        "excluded_from_training_fixture_ids": generalization_fixture_ids(),
        "security_slice_fixture_ids": security_slice_fixture_ids(),
        "judged_fixture_ids": benchmark_task_ids(),
    }
    write_json(RL_MANIFEST_PATH, payload)
    print(f"Wrote RL manifest to {RL_MANIFEST_PATH}")


def write_dataset_build_report(*, sft_rows: int, pref_rows: int, include_optional: bool) -> None:
    payload = {
        "benchmark": "AegisDesk",
        "targets": {
            "support_sft_min": SFT_TARGET_MIN,
            "support_sft_max": SFT_TARGET_MAX,
            "support_pref_min": PREF_TARGET_MIN,
        },
        "caps": {
            "bitext": BITEXT_SFT_CAP,
            "abcd": ABCD_SFT_CAP,
            "sgd": SGD_SFT_CAP,
            "optional_dialogstudio": OPTIONAL_DIALOGSTUDIO_CAP if include_optional else 0,
        },
        "files": {
            "bitext_support_sft.jsonl": count_jsonl(DATA_DIR / "bitext_support_sft.jsonl"),
            "abcd_support_sft.jsonl": count_jsonl(DATA_DIR / "abcd_support_sft.jsonl"),
            "taubench_sft.jsonl": count_jsonl(DATA_DIR / "taubench_sft.jsonl"),
            "sgd_support_sft.jsonl": count_jsonl(DATA_DIR / "sgd_support_sft.jsonl"),
            "helpsteer2_pref.jsonl": count_jsonl(DATA_DIR / "helpsteer2_pref.jsonl"),
            "dialogstudio_multiwoz_sample.jsonl": count_jsonl(DATA_DIR / "dialogstudio_multiwoz_sample.jsonl"),
            "support_sft.jsonl": sft_rows,
            "support_pref.jsonl": pref_rows,
        },
    }
    write_json(DATASET_BUILD_REPORT_PATH, payload)
    print(f"Wrote dataset build report to {DATASET_BUILD_REPORT_PATH}")


def validate_corpus_targets(*, sft_rows: int, pref_rows: int) -> None:
    errors: list[str] = []
    if sft_rows < SFT_TARGET_MIN:
        errors.append(f"support_sft.jsonl below target: {sft_rows} < {SFT_TARGET_MIN}")
    if sft_rows > SFT_TARGET_MAX:
        errors.append(f"support_sft.jsonl above target: {sft_rows} > {SFT_TARGET_MAX}")
    if pref_rows < PREF_TARGET_MIN:
        errors.append(f"support_pref.jsonl below target: {pref_rows} < {PREF_TARGET_MIN}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check datasets are reachable without writing files.",
    )
    parser.add_argument(
        "--include-optional-dialogstudio",
        action="store_true",
        help="Fetch optional DialogStudio / MultiWOZ examples as extra SFT seed data.",
    )
    args = parser.parse_args()

    ok_bitext = fetch_bitext(args.verify_only)
    ok_abcd = fetch_abcd(args.verify_only)
    ok_tau = fetch_taubench(args.verify_only)
    ok_sgd = fetch_schema_guided_dialogue(args.verify_only)
    ok_helpsteer = fetch_helpsteer2(args.verify_only)
    ok_optional = fetch_optional_dialogstudio(
        args.verify_only,
        include_optional=args.include_optional_dialogstudio,
    )

    if args.verify_only:
        if all([ok_bitext, ok_abcd, ok_tau, ok_sgd, ok_helpsteer, ok_optional]):
            print("\nAll required datasets are reachable.")
        else:
            print("\nOne or more datasets are not ready. Install extras or check connectivity.")
            sys.exit(1)
        return

    write_rl_manifest()
    sft_rows = build_support_sft(include_optional=args.include_optional_dialogstudio)
    pref_rows = build_support_pref()
    write_dataset_build_report(
        sft_rows=sft_rows,
        pref_rows=pref_rows,
        include_optional=args.include_optional_dialogstudio,
    )
    validate_corpus_targets(sft_rows=sft_rows, pref_rows=pref_rows)
    print("\nDataset fetch and corpus build complete.")


if __name__ == "__main__":
    main()
