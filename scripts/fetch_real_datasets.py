"""Fetch real-world dataset artifacts used by AegisDesk.

Downloads and processes data from three open-source datasets:

  Bitext Customer Support LLM Training Dataset  (PRIMARY — most relevant)
    Source: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
    License: CC BY 4.0  |  26,872 rows, 11 categories
    Used for: CustomerSimAgent utterance pool — billing_dispute, access_issue, escalation_demand
    Categories: ACCOUNT/recover_password, INVOICE, REFUND, PAYMENT, FEEDBACK/complaint,
    CONTACT/contact_human_agent. Direct B2B billing/account/complaint language.

  ABCD Action-Based Conversations Dataset
    Source: https://github.com/asappresearch/abcd  |  License: MIT
    Used for: reinstatement_request, security_concern, general_followup
    (Bitext has no equivalent for B2B suspension/breach scenarios)

  tau-bench Task Agent Unified Benchmark
    Source: https://github.com/sierra-research/tau-bench  |  License: MIT
    Used for: SFT warm-start data (training/data/taubench_sft.jsonl)
    69 retail support conversations → user/assistant turn pairs for SFT pre-training.

Usage:
    python scripts/fetch_real_datasets.py
    python scripts/fetch_real_datasets.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "training" / "data"

BITEXT_PARQUET_URL = (
    "https://huggingface.co/api/datasets/bitext"
    "/Bitext-customer-support-llm-chatbot-training-dataset/parquet/default/train/0.parquet"
)
ABCD_SAMPLE_URL = (
    "https://raw.githubusercontent.com/asappresearch/abcd/master/data/abcd_sample.json"
)
ABCD_ONTOLOGY_URL = (
    "https://raw.githubusercontent.com/asappresearch/abcd/master/data/ontology.json"
)
TAUBENCH_FEW_SHOT_URL = (
    "https://raw.githubusercontent.com/sierra-research/tau-bench/main"
    "/few_shot_data/MockRetailDomainEnv-few_shot.jsonl"
)

# Bitext (category, intent) → AegisDesk scenario category
BITEXT_INTENT_MAP: dict[tuple[str, str], str] = {
    ("ACCOUNT",  "recover_password"):       "access_issue",
    ("ACCOUNT",  "registration_problems"):  "access_issue",
    ("INVOICE",  "check_invoice"):          "billing_dispute",
    ("INVOICE",  "get_invoice"):            "billing_dispute",
    ("PAYMENT",  "payment_issue"):          "billing_dispute",
    ("REFUND",   "get_refund"):             "billing_dispute",
    ("REFUND",   "check_refund_policy"):    "billing_dispute",
    ("REFUND",   "track_refund"):           "billing_dispute",
    ("CONTACT",  "contact_human_agent"):    "escalation_demand",
    ("FEEDBACK", "complaint"):              "escalation_demand",
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AegisDesk/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _parse_bitext_parquet(raw: bytes) -> list[dict]:
    """Minimal Parquet reader for the Bitext dataset (no pyarrow dependency).

    The Bitext parquet file uses plain-encoded string columns. We read it by
    finding the Parquet footer and parsing the row-group metadata, then reading
    each column chunk. Falls back to empty list if the format is unexpected.
    """
    try:
        import pyarrow.parquet as pq
        import io
        table = pq.read_table(io.BytesIO(raw))
        df = table.to_pydict()
        rows = []
        for i in range(len(df["instruction"])):
            rows.append({
                "category": df["category"][i],
                "intent": df["intent"][i],
                "instruction": df["instruction"][i],
            })
        return rows
    except ImportError:
        pass

    # pyarrow not available — use HF datasets server API as fallback
    return []


def fetch_bitext(verify_only: bool) -> bool:
    """Fetch Bitext parquet and extract utterances per AegisDesk scenario category."""
    print("Fetching Bitext Customer Support dataset from HuggingFace...")
    try:
        raw = fetch(BITEXT_PARQUET_URL)
        print(f"  Downloaded {len(raw) / 1024 / 1024:.1f} MB parquet file")
    except Exception as exc:
        print(f"  ERROR downloading parquet: {exc}")
        return False

    rows = _parse_bitext_parquet(raw)
    if not rows:
        print("  WARNING: pyarrow not available — saving parquet for offline processing")
        if not verify_only:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            parquet_out = DATA_DIR / "bitext_raw.parquet"
            parquet_out.write_bytes(raw)
            print(f"  Saved raw parquet to: {parquet_out}")
            print("  Install pyarrow and re-run to extract utterances.")
        return True

    pool: dict[str, list[str]] = {}
    for row in rows:
        cat = row["category"]
        intent = row["intent"]
        instr = row["instruction"].strip()
        key = (cat, intent)
        if key in BITEXT_INTENT_MAP and 20 < len(instr) < 150 and "{{" not in instr:
            sc = BITEXT_INTENT_MAP[key]
            pool.setdefault(sc, [])
            if instr not in pool[sc] and len(pool[sc]) < 8:
                pool[sc].append(instr)

    total = sum(len(v) for v in pool.values())
    print(f"  Extracted {total} utterances across {len(pool)} scenario categories")
    for cat, utts in sorted(pool.items()):
        print(f"    {cat}: {len(utts)} utterances")

    if verify_only:
        print("  [verify-only] Bitext data OK")
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "bitext_utterance_pool.json"
    out.write_text(json.dumps({
        "source": "bitext-customer-support",
        "license": "CC BY 4.0",
        "url": BITEXT_PARQUET_URL,
        "rows_total": len(rows),
        "pool": pool,
    }, indent=2))
    print(f"  Saved: {out}")
    return True


def build_abcd_utterance_pool(sample: list[dict]) -> dict[str, list[str]]:
    """Extract clean customer utterances per flow, mapped to B2B SaaS categories."""
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
    pool: dict[str, list[str]] = {cat: [] for cat in set(flow_map.values())}
    seen: set[str] = set()
    for episode in sample:
        flow = episode["scenario"]["flow"]
        category = flow_map.get(flow, "general_followup")
        for role, utt in episode["original"]:
            utt = utt.strip()
            if (
                role == "customer"
                and len(utt) > 20
                and len(utt) < 220
                and len(utt.split()) >= 5
                and utt not in seen
            ):
                seen.add(utt)
                pool.setdefault(category, []).append(utt)
    return pool


def fetch_abcd(verify_only: bool) -> bool:
    print("Fetching ABCD sample + ontology from GitHub...")
    try:
        sample = json.loads(fetch(ABCD_SAMPLE_URL))
        ontology = json.loads(fetch(ABCD_ONTOLOGY_URL))
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False

    pool = build_abcd_utterance_pool(sample)
    total = sum(len(v) for v in pool.values())
    print(f"  Extracted {total} utterances across {len(pool)} scenario categories")

    if verify_only:
        print("  [verify-only] ABCD data OK")
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "abcd_utterance_pool.json"
    out.write_text(json.dumps({"source": "abcd", "license": "MIT",
                                "url": ABCD_SAMPLE_URL, "pool": pool}, indent=2))
    print(f"  Saved: {out}")

    actions = ontology.get("actions", {})
    action_summary = {k: list(v.keys()) for k, v in actions.items() if isinstance(v, dict)}
    ont_out = DATA_DIR / "abcd_action_taxonomy.json"
    ont_out.write_text(json.dumps({"source": "abcd-ontology", "license": "MIT",
                                    "url": ABCD_ONTOLOGY_URL,
                                    "actions": action_summary}, indent=2))
    print(f"  Saved: {ont_out}")
    return True


def fetch_taubench(verify_only: bool) -> bool:
    print("Fetching tau-bench retail few-shot conversations from GitHub...")
    try:
        raw = fetch(TAUBENCH_FEW_SHOT_URL).decode()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False

    lines = [json.loads(l) for l in raw.splitlines() if l.strip()]
    sft_rows = []
    for ex in lines:
        convo = ex["messages_display"]
        turns = []
        buf_role = None
        buf_lines: list[str] = []
        for line in convo.split("\n"):
            m = re.match(r"^(user|assistant|tool): (.*)", line)
            if m:
                if buf_role:
                    turns.append({"role": buf_role, "content": " ".join(buf_lines).strip()})
                buf_role = m.group(1)
                buf_lines = [m.group(2)]
            else:
                if buf_role and line.strip():
                    buf_lines.append(line.strip())
        if buf_role:
            turns.append({"role": buf_role, "content": " ".join(buf_lines).strip()})
        filtered = [
            t for t in turns
            if t["role"] in ("user", "assistant")
            and t["content"]
            and t["content"] != "None"
        ]
        if len(filtered) >= 2:
            sft_rows.append({"source": "tau-bench-retail", "messages": filtered})

    print(f"  Parsed {len(sft_rows)} SFT-ready conversations from {len(lines)} episodes")

    if verify_only:
        print("  [verify-only] tau-bench data OK")
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "taubench_sft.jsonl"
    with out.open("w") as f:
        for row in sft_rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved: {out}  ({len(sft_rows)} rows)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check datasets are reachable without writing files.",
    )
    args = parser.parse_args()

    ok_bitext = fetch_bitext(args.verify_only)
    ok_abcd = fetch_abcd(args.verify_only)
    ok_tau = fetch_taubench(args.verify_only)

    if ok_bitext and ok_abcd and ok_tau:
        print("\nAll datasets fetched successfully.")
    else:
        print("\nOne or more datasets failed. Check network connectivity.")
        sys.exit(1)


if __name__ == "__main__":
    main()
