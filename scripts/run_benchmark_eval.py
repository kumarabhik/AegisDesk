"""
Run a full multi-step benchmark eval using the HF Inference API.
No GPU needed — just Python + HTTP.

Usage:
    python scripts/run_benchmark_eval.py

Reads HF_TOKEN and ENV_BASE_URL from .env (already in project root).
Saves training/benchmark_results.json when done.
"""

import os, json, re, time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN    = os.environ["HF_TOKEN"]
ENV_URL     = os.environ.get("ENV_BASE_URL", "https://i4mgr00t-meta.hf.space")
API_BASE    = os.environ.get("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME  = os.environ.get("MODEL_NAME",    "Qwen/Qwen3-4B")

try:
    from openai import OpenAI
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE)
except ImportError:
    raise SystemExit("Install openai: pip install openai")

try:
    import httpx
except ImportError:
    raise SystemExit("Install httpx: pip install httpx")

# ── Tasks ─────────────────────────────────────────────────────────────────────

ALL_TASKS = [
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

SYSTEM_PROMPT = """\
You are an expert B2B SaaS support operator working inside AegisDesk.

Rules:
- Open the correct ticket FIRST before inspecting any records.
- Inspect ALL relevant records before taking any mutating action.
- Follow policy: do not apply credits without verifying billing records.
- Escalate security-sensitive requests; never fulfil them directly.
- Finalize the resolution only after completing all investigation steps.

You will receive the current console state as JSON. Respond ONLY with a \
single JSON object that matches one of these action schemas:

{"action_type": "open_ticket",          "ticket_id": "..."}
{"action_type": "inspect_record",       "record_id": "..."}
{"action_type": "search_kb",            "query": "..."}
{"action_type": "set_priority",         "ticket_id": "...", "priority": "low|medium|high|critical"}
{"action_type": "set_status",           "ticket_id": "...", "status": "open|in_progress|resolved|closed"}
{"action_type": "add_tag",              "ticket_id": "...", "tag": "..."}
{"action_type": "apply_credit",         "ticket_id": "...", "amount": 0.0, "currency": "USD"}
{"action_type": "escalate",             "ticket_id": "...", "escalation_team": "..."}
{"action_type": "draft_reply",          "ticket_id": "...", "template_id": "..."}
{"action_type": "finalize_resolution",  "ticket_id": "...", "resolution_code": "..."}

Output ONLY the JSON object — no explanation, no markdown, no extra text.\
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def obs_to_text(obs: dict) -> str:
    inbox = obs.get("inbox", [])
    inbox_lines = "\n".join(
        f"  {t['ticket_id']}: {t['subject']} | {t['priority']} | {t['status']}"
        for t in inbox
    )
    records = ", ".join(obs.get("available_record_ids", [])) or "none"
    fp = obs.get("focus_panel")
    focus = f"\n\nFocus panel — {fp['title']}:\n{fp['body'][:600]}" if fp else ""
    return (
        f"Task brief: {obs.get('task_brief', '')}\n"
        f"Step: {obs.get('step_count', 0)} | Remaining steps: {obs.get('remaining_steps', 0)}\n"
        f"Active ticket: {obs.get('active_ticket_id') or 'none'}\n"
        f"Available records: {records}\n"
        f"Last action error: {obs.get('last_action_error') or 'none'}\n"
        f"Inbox:\n{inbox_lines}"
        f"{focus}"
    )

def parse_action(text: str) -> dict:
    text = text.strip()
    # strip thinking tags if model outputs them
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"action_type": "finalize_resolution",
            "ticket_id": "fallback", "resolution_code": "no_action"}

def call_model(obs_text: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": obs_text},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    raw = resp.choices[0].message.content or ""
    return parse_action(raw)

# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = 42, max_steps: int = 15) -> float:
    with httpx.Client(timeout=60, follow_redirects=True) as http:
        obs = http.post(f"{ENV_URL}/reset",
                        json={"task_id": task_id, "seed": seed}).json()
        for step in range(max_steps):
            action = call_model(obs_to_text(obs))
            result = http.post(f"{ENV_URL}/step", json=action).json()
            obs = result.get("observation", obs)
            if result.get("done"):
                return float(result.get("info", {}).get("final_score",
                       result.get("reward", 0.0)))
        state = http.post(f"{ENV_URL}/state").json()
        return float(state.get("final_score") or state.get("rubric_progress") or 0.0)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Model   : {MODEL_NAME}")
    print(f"Env URL : {ENV_URL}")
    print(f"Tasks   : {len(ALL_TASKS)}\n")

    scores = {}
    for task_id in ALL_TASKS:
        t0 = time.time()
        try:
            score = run_episode(task_id, seed=42, max_steps=15)
            scores[task_id] = score
            print(f"  {task_id:<45}  {score:.3f}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            scores[task_id] = 0.0
            print(f"  {task_id:<45}  ERROR: {e}")

    mean = sum(scores.values()) / len(scores)
    print(f"\n  {'Mean':<45}  {mean:.3f}")
    print(f"  {'Baseline (Qwen2.5-72B zero-shot)':<45}  0.270")
    print(f"  {'Delta':<45}  {mean - 0.27:+.3f}")

    out = {
        "timestamp":   datetime.utcnow().isoformat(),
        "model":       MODEL_NAME,
        "env_url":     ENV_URL,
        "eval_type":   "zero_shot_multistep",
        "task_scores": scores,
        "mean_score":  mean,
        "v1_baseline": 0.27,
        "delta":       mean - 0.27,
    }
    out_path = Path(__file__).parent.parent / "training" / "benchmark_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    print("Run `git add training/benchmark_results.json && git commit -m 'Add benchmark results' && git push github main`")

if __name__ == "__main__":
    main()
