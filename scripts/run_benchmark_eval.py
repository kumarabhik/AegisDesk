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

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

HF_TOKEN    = os.environ["HF_TOKEN"]
ENV_URL     = os.environ.get("ENV_BASE_URL", "https://i4mgr00t-meta.hf.space")
API_BASE    = os.environ.get("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME  = os.environ.get("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")

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

Rules (follow in order):
1. Open the correct ticket FIRST (look at inbox — pick the ticket matching the task).
2. Inspect ALL available records before taking any mutating action.
3. Apply credits, tags, or status changes only after inspecting records.
4. Escalate security-sensitive requests; never fulfil them directly.
5. Draft a reply, then finalize the resolution last.
6. NEVER repeat the same action twice — if you already opened a ticket, move to inspect_record next.

You will receive the current console state. Use active_ticket_id and available_record_ids \
to decide your next action. Respond ONLY with a single JSON object:

{"action_type": "open_ticket",          "ticket_id": "<id from inbox>"}
{"action_type": "inspect_record",       "record_id": "<id from available_record_ids>"}
{"action_type": "search_kb",            "query": "<search terms>"}
{"action_type": "set_priority",         "ticket_id": "<id>", "priority": "low|medium|high|critical"}
{"action_type": "set_status",           "ticket_id": "<id>", "status": "open|in_progress|resolved|closed"}
{"action_type": "add_tag",              "ticket_id": "<id>", "tag": "<tag>"}
{"action_type": "apply_credit",         "ticket_id": "<id>", "amount": 0.0, "currency": "USD"}
{"action_type": "escalate",             "ticket_id": "<id>", "escalation_team": "<team>"}
{"action_type": "draft_reply",          "ticket_id": "<id>", "template_id": "<template>"}
{"action_type": "finalize_resolution",  "ticket_id": "<id>", "resolution_code": "<code>"}

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

def run_episode(task_id: str, seed: int = 42, max_steps: int = 15,
                debug: bool = False) -> float:
    with httpx.Client(timeout=60, follow_redirects=True) as http:
        reset_result = http.post(f"{ENV_URL}/reset",
                                 json={"task_id": task_id, "seed": seed}).json()
        # /reset returns {"observation": {...}, "reward": ..., "done": ..., "info": ...}
        obs = reset_result.get("observation", reset_result)
        if debug:
            print(f"\n  [reset] outer keys={list(reset_result.keys())}")
            print(f"  [reset] obs keys={list(obs.keys())}")
            print(f"  [reset] task_brief={str(obs.get('task_brief','MISSING'))[:80]}")
            print(f"  [reset] inbox={obs.get('inbox','MISSING')}")
            print(f"  [reset] available_record_ids={obs.get('available_record_ids','MISSING')}")

        for step in range(max_steps):
            obs_text = obs_to_text(obs)
            raw_resp  = None
            # get raw response before parsing
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": obs_text},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            raw_resp = resp.choices[0].message.content or ""
            action = parse_action(raw_resp)

            if debug and step == 0:
                print(f"  [step 0 raw] {repr(raw_resp[:200])}")
                print(f"  [step 0 action] {action}")

            result = http.post(f"{ENV_URL}/step", json=action).json()
            if debug and step == 0:
                print(f"  [step 0 result] reward={result.get('reward')} done={result.get('done')} keys={list(result.keys())}")

            obs = result.get("observation", obs)
            if result.get("done"):
                final = float(result.get("info", {}).get("final_score",
                              result.get("reward", 0.0)))
                if debug:
                    print(f"  [done at step {step}] final={final}")
                return final

        # /state is GET, not POST
        state = http.get(f"{ENV_URL}/state").json()
        if debug:
            print(f"  [state] keys={list(state.keys())}")
            print(f"  [state] {state}")
        # explicit None check — 0.0 is falsy so "or" would skip it
        for key in ("final_score", "rubric_progress", "score", "current_score"):
            v = state.get(key)
            if v is not None:
                return float(v)
        return 0.0

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
