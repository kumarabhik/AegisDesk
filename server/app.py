"""FastAPI application entrypoint for support_ops_env."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pydantic import ValidationError
import uvicorn

try:
    from openenv.core.env_server import create_app
except ImportError:
    create_app = None

try:
    from ..models import SupportAction, SupportObservation
    from .environment import SupportOpsEnvironment
    from .fixtures import load_all_fixtures
except ImportError:
    from models import SupportAction, SupportObservation
    from server.environment import SupportOpsEnvironment
    from server.fixtures import load_all_fixtures


_shared_env: SupportOpsEnvironment | None = None

CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AegisDesk Console</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --panel: #fffaf2;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9cbb3;
      --accent: #0b6e4f;
      --accent-2: #8b5e34;
      --danger: #a61b1b;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      --sans: "Segoe UI", "Aptos", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(11,110,79,0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(139,94,52,0.10), transparent 22%),
        linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    .hero {
      display: grid;
      gap: 10px;
      margin-bottom: 24px;
    }
    .eyebrow {
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 12px;
      color: var(--accent-2);
      font-weight: 700;
    }
    h1 {
      margin: 0;
      font-size: clamp(32px, 5vw, 56px);
      line-height: 0.95;
      font-weight: 800;
    }
    .sub {
      max-width: 860px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
    }
    .panel {
      background: color-mix(in srgb, var(--panel) 86%, white 14%);
      border: 1px solid color-mix(in srgb, var(--line) 80%, white 20%);
      border-radius: 20px;
      box-shadow: 0 20px 50px rgba(31, 41, 51, 0.07);
      overflow: hidden;
    }
    .panel-head {
      padding: 16px 18px;
      border-bottom: 1px solid rgba(82, 96, 109, 0.12);
      background: linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,250,242,0.92));
    }
    .panel-head h2 {
      margin: 0;
      font-size: 15px;
    }
    .panel-body {
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    .field {
      display: grid;
      gap: 6px;
    }
    .field label {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    input, select, textarea, button {
      font: inherit;
    }
    input, select, textarea {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(82, 96, 109, 0.24);
      background: rgba(255,255,255,0.84);
      padding: 11px 12px;
      color: var(--ink);
    }
    textarea {
      min-height: 110px;
      resize: vertical;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.45;
    }
    .btn-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      cursor: pointer;
      font-weight: 700;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: progress; transform: none; }
    .btn-primary { background: var(--accent); color: white; }
    .btn-secondary { background: #ebe2d0; color: var(--ink); }
    .btn-danger { background: #f5d4d4; color: var(--danger); }
    .meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .meta-card {
      border: 1px solid rgba(82, 96, 109, 0.14);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.7);
    }
    .meta-card strong {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .ticket-list, .record-list {
      display: grid;
      gap: 10px;
    }
    .ticket {
      border: 1px solid rgba(82, 96, 109, 0.14);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.72);
    }
    .ticket-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }
    .ticket-id {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--accent-2);
      font-weight: 700;
    }
    .ticket p {
      margin: 4px 0 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #efe6d8;
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 700;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #16212b;
      color: #e6eef5;
      border-radius: 16px;
      padding: 14px;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.55;
      max-height: 360px;
      overflow: auto;
    }
    .hint {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.5;
    }
    .status {
      padding: 12px 14px;
      border-radius: 14px;
      font-size: 13px;
      line-height: 1.5;
      background: #e7f3ec;
      color: var(--accent);
      border: 1px solid rgba(11,110,79,0.12);
    }
    .status.error {
      background: #fdeaea;
      color: var(--danger);
      border-color: rgba(166,27,27,0.16);
    }
    .status.warn {
      background: #fff4dc;
      color: #8a6116;
      border-color: rgba(138,97,22,0.16);
    }
    @media (max-width: 1040px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Interactive Benchmark Console</div>
      <h1>AegisDesk Console</h1>
      <div class="sub">
        Explore the benchmark manually without touching the judged API contract.
        Reset into any task, inspect the inbox, fire structured actions, and watch
        rewards, state, and focus context update in real time.
      </div>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="panel-head"><h2>Episode Controls</h2></div>
        <div class="panel-body">
          <div id="status" class="status">Loading task catalog...</div>

          <div class="field">
            <label for="taskSelect">Task</label>
            <select id="taskSelect"></select>
          </div>

          <div class="field">
            <label for="seedInput">Seed</label>
            <input id="seedInput" type="number" value="1" min="1" step="1">
          </div>

          <div class="btn-row">
            <button id="resetBtn" class="btn-primary">Reset Episode</button>
            <button id="stateBtn" class="btn-secondary">Refresh State</button>
          </div>

          <div class="field">
            <label for="actionType">Action Type</label>
            <select id="actionType"></select>
          </div>

          <div id="actionFields"></div>

          <div class="btn-row">
            <button id="stepBtn" class="btn-primary">Send Action</button>
            <button id="fillBtn" class="btn-secondary">Fill From Active Context</button>
            <button id="clearBtn" class="btn-danger">Clear Fields</button>
          </div>

          <div class="field">
            <label for="actionPreview">Action Preview</label>
            <textarea id="actionPreview" readonly></textarea>
          </div>

          <div class="hint">
            Tip: the console keeps the official API untouched. It simply calls
            <code>/tasks</code>, <code>/reset</code>, <code>/step</code>, and <code>/state</code>
            for you.
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Observation and State</h2></div>
        <div class="panel-body">
          <div class="meta">
            <div class="meta-card"><strong>Task</strong><span id="metaTask">-</span></div>
            <div class="meta-card"><strong>Reward</strong><span id="metaReward">-</span></div>
            <div class="meta-card"><strong>Done</strong><span id="metaDone">-</span></div>
          </div>

          <div class="field">
            <label>Task Brief</label>
            <textarea id="taskBrief" readonly></textarea>
          </div>

          <div class="field">
            <label>Inbox</label>
            <div id="ticketList" class="ticket-list"></div>
          </div>

          <div class="field">
            <label>Available Records</label>
            <div id="recordList" class="record-list"></div>
          </div>

          <div class="field">
            <label>Focus Panel</label>
            <pre id="focusPanel">No panel selected yet.</pre>
          </div>

          <div class="field">
            <label>State Snapshot</label>
            <pre id="statePanel">No state loaded yet.</pre>
          </div>

          <div class="field">
            <label>Last Response</label>
            <pre id="rawResponse">No API response yet.</pre>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const actionSchemas = {
      open_ticket: ["ticket_id"],
      inspect_record: ["record_id"],
      search_kb: ["query"],
      set_priority: ["ticket_id", "priority"],
      set_status: ["ticket_id", "status"],
      add_tag: ["ticket_id", "tag"],
      apply_credit: ["ticket_id", "amount", "currency"],
      escalate: ["ticket_id", "escalation_team"],
      draft_reply: ["ticket_id", "template_id", "reply_checklist", "freeform_note"],
      finalize_resolution: ["ticket_id", "resolution_code"],
    };

    const fieldLabels = {
      ticket_id: "Ticket ID",
      record_id: "Record ID",
      query: "Knowledge Base Query",
      priority: "Priority",
      status: "Status",
      tag: "Tag",
      amount: "Credit Amount",
      currency: "Currency",
      escalation_team: "Escalation Team",
      template_id: "Reply Template ID",
      reply_checklist: "Reply Checklist (comma-separated)",
      freeform_note: "Freeform Note",
      resolution_code: "Resolution Code",
    };

    const fieldDefaults = {
      priority: ["low", "normal", "high", "urgent"],
      status: ["open", "pending", "waiting_on_customer", "resolved", "escalated"],
      escalation_team: ["billing_ops", "incident_response", "security"],
      currency: ["USD", "EUR", "GBP"],
    };

    let latestObservation = null;
    let latestState = null;

    const taskSelect = document.getElementById("taskSelect");
    const actionType = document.getElementById("actionType");
    const actionFields = document.getElementById("actionFields");
    const actionPreview = document.getElementById("actionPreview");
    const statusBox = document.getElementById("status");

    function setStatus(message, kind = "ok") {
      statusBox.textContent = message;
      statusBox.className = "status" + (kind === "ok" ? "" : " " + kind);
    }

    function toJson(value) {
      return JSON.stringify(value, null, 2);
    }

    function buildField(name) {
      const wrapper = document.createElement("div");
      wrapper.className = "field";
      const label = document.createElement("label");
      label.textContent = fieldLabels[name] || name;
      label.setAttribute("for", "field-" + name);
      wrapper.appendChild(label);

      if (fieldDefaults[name]) {
        const select = document.createElement("select");
        select.id = "field-" + name;
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = "Select...";
        select.appendChild(empty);
        fieldDefaults[name].forEach((optionValue) => {
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = optionValue;
          select.appendChild(option);
        });
        select.addEventListener("change", updateActionPreview);
        wrapper.appendChild(select);
        return wrapper;
      }

      const input = name === "freeform_note" ? document.createElement("textarea") : document.createElement("input");
      input.id = "field-" + name;
      if (name === "amount") {
        input.type = "number";
        input.step = "0.01";
      } else {
        input.type = "text";
      }
      input.addEventListener("input", updateActionPreview);
      wrapper.appendChild(input);
      return wrapper;
    }

    function renderActionFields() {
      actionFields.innerHTML = "";
      const fields = actionSchemas[actionType.value] || [];
      fields.forEach((fieldName) => actionFields.appendChild(buildField(fieldName)));
      updateActionPreview();
    }

    function collectActionPayload() {
      const payload = { action_type: actionType.value };
      (actionSchemas[actionType.value] || []).forEach((fieldName) => {
        const element = document.getElementById("field-" + fieldName);
        if (!element) return;
        let value = element.value;
        if (value === "") return;
        if (fieldName === "amount") {
          value = Number(value);
        }
        if (fieldName === "reply_checklist") {
          value = value.split(",").map((item) => item.trim()).filter(Boolean);
        }
        payload[fieldName] = value;
      });
      return payload;
    }

    function updateActionPreview() {
      actionPreview.value = toJson(collectActionPayload());
    }

    function fillFromContext() {
      const activeTicket = latestObservation?.active_ticket_id || latestState?.active_ticket_id || latestState?.selected_ticket_id || "";
      const firstRecord = latestObservation?.available_record_ids?.[0] || "";

      const mapping = {
        ticket_id: activeTicket,
        record_id: firstRecord,
      };

      Object.entries(mapping).forEach(([fieldName, value]) => {
        const element = document.getElementById("field-" + fieldName);
        if (element && !element.value && value) {
          element.value = value;
        }
      });
      updateActionPreview();
    }

    function clearFields() {
      (actionSchemas[actionType.value] || []).forEach((fieldName) => {
        const element = document.getElementById("field-" + fieldName);
        if (element) {
          element.value = "";
        }
      });
      updateActionPreview();
    }

    function renderInbox(inbox) {
      const container = document.getElementById("ticketList");
      container.innerHTML = "";
      if (!inbox?.length) {
        container.innerHTML = "<div class='hint'>No inbox items yet.</div>";
        return;
      }

      inbox.forEach((ticket) => {
        const el = document.createElement("div");
        el.className = "ticket";
        el.innerHTML = `
          <div class="ticket-top">
            <div>
              <div class="ticket-id">${ticket.ticket_id}</div>
              <strong>${ticket.subject}</strong>
            </div>
            <button class="btn-secondary" type="button" data-ticket="${ticket.ticket_id}">Open</button>
          </div>
          <p>${ticket.summary}</p>
          <div class="chips">
            <span class="chip">priority: ${ticket.priority}</span>
            <span class="chip">status: ${ticket.status}</span>
            ${(ticket.tags || []).map((tag) => `<span class="chip">${tag}</span>`).join("")}
          </div>
        `;
        el.querySelector("button").addEventListener("click", async () => {
          actionType.value = "open_ticket";
          renderActionFields();
          document.getElementById("field-ticket_id").value = ticket.ticket_id;
          updateActionPreview();
          await sendAction();
        });
        container.appendChild(el);
      });
    }

    function renderRecords(recordIds) {
      const container = document.getElementById("recordList");
      container.innerHTML = "";
      if (!recordIds?.length) {
        container.innerHTML = "<div class='hint'>No records available yet.</div>";
        return;
      }
      recordIds.forEach((recordId) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn-secondary";
        button.textContent = recordId;
        button.addEventListener("click", async () => {
          actionType.value = "inspect_record";
          renderActionFields();
          document.getElementById("field-record_id").value = recordId;
          updateActionPreview();
          await sendAction();
        });
        container.appendChild(button);
      });
    }

    function renderObservation(payload) {
      const observation = payload?.observation || payload || {};
      latestObservation = observation;

      document.getElementById("metaTask").textContent = latestState?.task_id || taskSelect.value || "-";
      document.getElementById("metaReward").textContent = payload?.reward ?? observation?.reward ?? "-";
      document.getElementById("metaDone").textContent = String(payload?.done ?? observation?.done ?? false);
      document.getElementById("taskBrief").value = observation.task_brief || "";
      document.getElementById("focusPanel").textContent = observation.focus_panel ? toJson(observation.focus_panel) : "No panel selected yet.";
      document.getElementById("rawResponse").textContent = toJson(payload);

      renderInbox(observation.inbox || []);
      renderRecords(observation.available_record_ids || []);
    }

    function renderState(state) {
      latestState = state;
      document.getElementById("statePanel").textContent = toJson(state);
      document.getElementById("metaTask").textContent = state.task_id || taskSelect.value || "-";
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || ("Request failed with status " + response.status));
      }
      return response.json();
    }

    async function loadTasks() {
      const payload = await api("/tasks");
      taskSelect.innerHTML = "";
      payload.tasks.forEach((task) => {
        const option = document.createElement("option");
        option.value = task.task_id;
        option.textContent = `${task.task_id} (${task.difficulty})`;
        taskSelect.appendChild(option);
      });
      setStatus("Task catalog loaded. Reset an episode to begin.");
    }

    async function resetEpisode() {
      const resetBtn = document.getElementById("resetBtn");
      resetBtn.disabled = true;
      try {
        setStatus("Resetting episode...", "warn");
        const payload = await api("/reset", {
          method: "POST",
          body: JSON.stringify({
            task_id: taskSelect.value,
            seed: Number(document.getElementById("seedInput").value || 1),
          }),
        });
        renderObservation(payload);
        const state = await api("/state");
        renderState(state);
        fillFromContext();
        setStatus("Episode reset successfully.");
      } catch (error) {
        setStatus(String(error), "error");
      } finally {
        resetBtn.disabled = false;
      }
    }

    async function refreshState() {
      try {
        setStatus("Refreshing state...", "warn");
        const state = await api("/state");
        renderState(state);
        setStatus("State refreshed.");
      } catch (error) {
        setStatus(String(error), "error");
      }
    }

    async function sendAction() {
      const stepBtn = document.getElementById("stepBtn");
      stepBtn.disabled = true;
      try {
        setStatus("Sending action...", "warn");
        const payload = await api("/step", {
          method: "POST",
          body: JSON.stringify(collectActionPayload()),
        });
        renderObservation(payload);
        const state = await api("/state");
        renderState(state);
        fillFromContext();
        setStatus(payload.done ? "Episode complete." : "Action applied successfully.");
      } catch (error) {
        setStatus(String(error), "error");
      } finally {
        stepBtn.disabled = false;
      }
    }

    document.getElementById("resetBtn").addEventListener("click", resetEpisode);
    document.getElementById("stateBtn").addEventListener("click", refreshState);
    document.getElementById("stepBtn").addEventListener("click", sendAction);
    document.getElementById("fillBtn").addEventListener("click", fillFromContext);
    document.getElementById("clearBtn").addEventListener("click", clearFields);

    Object.keys(actionSchemas).forEach((actionName) => {
      const option = document.createElement("option");
      option.value = actionName;
      option.textContent = actionName;
      actionType.appendChild(option);
    });
    actionType.addEventListener("change", renderActionFields);
    actionType.value = "open_ticket";
    renderActionFields();
    loadTasks().catch((error) => setStatus(String(error), "error"));
  </script>
</body>
</html>
"""


def create_environment() -> SupportOpsEnvironment:
    """Factory used by OpenEnv and the custom HTTP routes."""

    global _shared_env
    if _shared_env is None:
        task_id = os.getenv("DEFAULT_TASK_ID")
        _shared_env = SupportOpsEnvironment(task_id=task_id)
    return _shared_env


class ResetPayload(BaseModel):
    task_id: str | None = None
    seed: int | None = None


def _prewarm_runtime() -> None:
    """Warm fixture cache and the shared environment to reduce first-hit latency."""

    load_all_fixtures()
    create_environment()


def _prewarm_enabled() -> bool:
    """Return whether startup prewarming is enabled for this process."""

    value = os.getenv("AEGISDESK_PREWARM", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    """Warm caches before serving requests so the first interactive hit is faster."""

    if _prewarm_enabled():
        _prewarm_runtime()
    yield


if create_app is not None:
    app = create_app(
        create_environment,
        SupportAction,
        SupportObservation,
        env_name="support_ops_env",
    )
    app.router.lifespan_context = app_lifespan
    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) not in {"/reset", "/step", "/state"}
    ]
else:
    app = FastAPI(title="support_ops_env", lifespan=app_lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetPayload | None = None) -> dict[str, Any]:
    env = create_environment()
    payload = payload or ResetPayload()
    observation = env.reset(task_id=payload.task_id, seed=payload.seed)
    return {
        "observation": observation.model_dump(mode="json"),
        "reward": observation.reward,
        "done": observation.done,
        "info": env.last_info,
    }


@app.post("/step")
async def step(request: Request) -> dict[str, Any]:
    env = create_environment()
    payload = await request.json()
    action_payload = payload.get("action", payload) if isinstance(payload, dict) else payload
    try:
        action = SupportAction.model_validate(action_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    observation = env.step(action)
    return {
        "observation": observation.model_dump(mode="json"),
        "reward": observation.reward,
        "done": observation.done,
        "info": env.last_info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    env = create_environment()
    return env.state.model_dump(mode="json")


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    """Return a compact catalog of available benchmark tasks."""

    fixtures = load_all_fixtures()
    return {
        "tasks": [
            {
                "task_id": fixture.task_id,
                "difficulty": fixture.difficulty.value,
                "task_brief": fixture.task_brief,
                "max_steps": fixture.max_steps,
                "reply_template_id": fixture.reply_requirements.template_id,
                "reply_checklist": fixture.reply_requirements.checklist,
            }
            for fixture in fixtures.values()
        ]
    }


@app.get("/")
def root() -> dict[str, str]:
    """Basic 200 endpoint for Space health probes."""

    return {"status": "ok", "env_name": "support_ops_env"}


@app.get("/health")
def health() -> dict[str, str]:
    """Dedicated health endpoint for container and uptime checks."""

    return {"status": "ok", "env_name": "support_ops_env"}


@app.get("/console", response_class=HTMLResponse)
def console() -> str:
    """Serve a lightweight operator console for manual benchmark exploration."""

    return CONSOLE_HTML


def main() -> None:
    """Run the app with uvicorn."""

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
