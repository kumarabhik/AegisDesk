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
    from ..oracle_tools import generate_trajectory_report
    from .environment import SupportOpsEnvironment
    from .fixtures import all_task_ids, load_all_fixtures, task_track
except ImportError:
    from models import SupportAction, SupportObservation
    from oracle_tools import generate_trajectory_report
    from server.environment import SupportOpsEnvironment
    from server.fixtures import all_task_ids, load_all_fixtures, task_track


_shared_env: SupportOpsEnvironment | None = None

LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AegisDesk</title>
  <style>
    :root {
      --bg: #f4efe5;
      --panel: rgba(255, 250, 242, 0.92);
      --ink: #1d2a39;
      --muted: #52606d;
      --line: rgba(140, 108, 70, 0.18);
      --accent: #0f7a5a;
      --accent-2: #8b5e34;
      --dark: #18232e;
      --sans: "Segoe UI", "Aptos", system-ui, sans-serif;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(15,122,90,0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(139,94,52,0.13), transparent 22%),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
    }
    .wrap {
      max-width: 1240px;
      margin: 0 auto;
      padding: 34px 20px 50px;
      display: grid;
      gap: 24px;
    }
    .hero {
      display: grid;
      gap: 14px;
      padding: 8px 0 4px;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      font-weight: 700;
      color: var(--accent-2);
    }
    h1 {
      margin: 0;
      font-size: clamp(40px, 7vw, 82px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .sub {
      max-width: 880px;
      font-size: 18px;
      line-height: 1.65;
      color: var(--muted);
    }
    .hero-actions, .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    a.btn, .chip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      text-decoration: none;
      border-radius: 999px;
      font-weight: 700;
    }
    a.btn {
      padding: 12px 18px;
      border: 1px solid transparent;
    }
    a.btn.primary {
      background: var(--accent);
      color: white;
      box-shadow: 0 14px 32px rgba(15,122,90,0.18);
    }
    a.btn.secondary {
      background: rgba(255,255,255,0.72);
      color: var(--ink);
      border-color: var(--line);
    }
    .chip {
      padding: 8px 12px;
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
      color: var(--accent-2);
      font-size: 13px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 22px 50px rgba(29, 42, 57, 0.08);
      overflow: hidden;
    }
    .panel-head {
      padding: 18px 20px;
      border-bottom: 1px solid rgba(82,96,109,0.10);
      background: linear-gradient(180deg, rgba(255,255,255,0.75), rgba(255,250,242,0.92));
    }
    .panel-head h2 {
      margin: 0;
      font-size: 16px;
    }
    .panel-body {
      padding: 20px;
      display: grid;
      gap: 16px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .metric {
      padding: 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.75);
      border: 1px solid rgba(82,96,109,0.10);
    }
    .metric strong {
      display: block;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .metric span {
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }
    .list, .route-list {
      display: grid;
      gap: 12px;
    }
    .item {
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(82,96,109,0.10);
      background: rgba(255,255,255,0.72);
    }
    .item strong {
      display: block;
      margin-bottom: 5px;
      font-size: 14px;
    }
    .item p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }
    .route {
      display: grid;
      gap: 6px;
      padding: 13px 14px;
      border-radius: 16px;
      background: var(--dark);
      color: #e8eff5;
    }
    .route code {
      font-family: var(--mono);
      font-size: 13px;
      color: #b8f1d3;
    }
    .route span {
      color: #a9bac7;
      font-size: 13px;
      line-height: 1.45;
    }
    .foot {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      .wrap { padding: 24px 14px 36px; }
      .metric-grid { grid-template-columns: 1fr; }
      .sub { font-size: 16px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">OpenEnv Benchmark • Live Space</div>
      <h1>AegisDesk</h1>
      <div class="sub">
        A real-world agent benchmark for B2B SaaS support operations. Agents must triage a
        live-looking support inbox, inspect the right records, follow policy, avoid unsafe
        shortcuts, and finalize a deterministic, gradable resolution.
      </div>
      <div class="hero-actions">
        <a class="btn primary" href="/console">Open Interactive Console</a>
        <a class="btn secondary" href="/trajectory-viewer">Open Trajectory Viewer</a>
        <a class="btn secondary" href="/benchmark-card">View Benchmark Card</a>
      </div>
      <div class="chips">
        <div class="chip">3 judged core tasks</div>
        <div class="chip">3 extended demo tasks</div>
        <div class="chip">Deterministic scores in [0, 1]</div>
        <div class="chip">OpenAI-client inference path</div>
      </div>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="panel-head"><h2>Why This Project Stands Out</h2></div>
        <div class="panel-body">
          <div class="metric-grid">
            <div class="metric"><strong>Core Tasks</strong><span>3</span></div>
            <div class="metric"><strong>Extended Tasks</strong><span>3</span></div>
            <div class="metric"><strong>Difficulty Bands</strong><span>Easy→Hard</span></div>
            <div class="metric"><strong>Scoring</strong><span>Deterministic</span></div>
          </div>
          <div class="list">
            <div class="item">
              <strong>Real workflow, not a toy</strong>
              <p>Each episode models support-operations judgment: ticket selection, evidence gathering, safe escalation, and structured customer communication.</p>
            </div>
            <div class="item">
              <strong>Judge-friendly by design</strong>
              <p>The benchmark includes an interactive console, an oracle trajectory viewer, a reproducible inference script, and captured validation results.</p>
            </div>
            <div class="item">
              <strong>Dense rewards with safety penalties</strong>
              <p>Agents receive partial credit for meaningful progress and get penalized for loops, irrelevant inspection, or unsafe direct actions.</p>
            </div>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Live Routes</h2></div>
        <div class="panel-body">
          <div class="route-list">
            <div class="route">
              <code>/console</code>
              <span>Manual benchmark UI for resetting episodes, sending structured actions, and inspecting observation/state updates in real time.</span>
            </div>
            <div class="route">
              <code>/trajectory-viewer</code>
              <span>Judge-friendly oracle viewer with per-step rewards, rubric progress, penalties, and final score breakdown.</span>
            </div>
            <div class="route">
              <code>/benchmark-card</code>
              <span>Compact machine-readable summary of task counts, validation posture, and public benchmark routes.</span>
            </div>
            <div class="route">
              <code>/tasks</code>
              <span>Task catalog containing the judged core pack plus the optional extended pack used for demos and inspection.</span>
            </div>
          </div>
        </div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>Core Benchmark Tracks</h2></div>
      <div class="panel-body">
        <div class="list">
          <div class="item">
            <strong>billing_seat_adjustment</strong>
            <p>Resolve a real overbilling case by inspecting account and invoice records, applying the exact credit, updating ticket metadata, and sending the correct structured reply.</p>
          </div>
          <div class="item">
            <strong>login_incident_triage</strong>
            <p>Handle a VIP login issue during an active incident without resorting to unsafe account-level shortcuts.</p>
          </div>
          <div class="item">
            <strong>suspicious_admin_request</strong>
            <p>Catch a likely account-takeover scenario, inspect verification evidence, escalate to security, and refuse unsafe fulfillment.</p>
          </div>
        </div>
      </div>
    </section>

    <div class="foot">
      API clients and validators still receive the standard JSON health response from <code>/</code> unless they request HTML.
      This landing page is a human-facing view layered on top of the same judged contract.
    </div>
  </div>
</body>
</html>
"""

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

TRAJECTORY_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AegisDesk Trajectory Viewer</title>
  <style>
    :root {
      --bg: #f4f0e6;
      --panel: #fffaf1;
      --ink: #17212b;
      --muted: #546170;
      --line: #d8cab3;
      --accent: #0f766e;
      --accent-soft: #daf0ec;
      --danger: #9f1239;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      --sans: "Segoe UI", "Aptos", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(159,18,57,0.06), transparent 20%),
        linear-gradient(180deg, #f8f4ec, var(--bg));
    }
    .wrap {
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 18px 40px;
      display: grid;
      gap: 20px;
    }
    .hero {
      display: grid;
      gap: 8px;
    }
    .eyebrow {
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 12px;
      font-weight: 700;
      color: var(--accent);
    }
    h1 {
      margin: 0;
      font-size: clamp(30px, 5vw, 54px);
      line-height: 0.95;
    }
    .sub {
      color: var(--muted);
      max-width: 840px;
      line-height: 1.6;
    }
    .grid {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 18px;
    }
    .panel {
      background: rgba(255,250,241,0.88);
      border: 1px solid rgba(84,97,112,0.16);
      border-radius: 20px;
      box-shadow: 0 18px 48px rgba(23,33,43,0.07);
      overflow: hidden;
    }
    .panel-head {
      padding: 16px 18px;
      border-bottom: 1px solid rgba(84,97,112,0.12);
      background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,250,241,0.95));
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
    select, input, button {
      font: inherit;
    }
    select, input {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(84,97,112,0.24);
      background: rgba(255,255,255,0.84);
      padding: 11px 12px;
      color: var(--ink);
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      font-weight: 700;
      cursor: pointer;
      background: var(--accent);
      color: white;
    }
    button:disabled {
      opacity: 0.6;
      cursor: progress;
    }
    .status {
      padding: 12px 14px;
      border-radius: 14px;
      font-size: 13px;
      line-height: 1.5;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid rgba(15,118,110,0.12);
    }
    .status.error {
      background: #fde8ef;
      border-color: rgba(159,18,57,0.16);
      color: var(--danger);
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }
    .meta-card {
      border: 1px solid rgba(84,97,112,0.14);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.72);
    }
    .meta-card strong {
      display: block;
      font-size: 12px;
      margin-bottom: 6px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .step-list {
      display: grid;
      gap: 12px;
    }
    details {
      border: 1px solid rgba(84,97,112,0.14);
      border-radius: 16px;
      background: rgba(255,255,255,0.78);
      overflow: hidden;
    }
    summary {
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      font-weight: 700;
    }
    summary::-webkit-details-marker { display: none; }
    .step-body {
      padding: 0 16px 16px;
      display: grid;
      gap: 12px;
    }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      background: #e9e4d8;
      color: #5d4b37;
      font-size: 12px;
      font-weight: 700;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px;
      border-bottom: 1px solid rgba(84,97,112,0.12);
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #152330;
      color: #e8f0f6;
      border-radius: 14px;
      padding: 14px;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.55;
      overflow: auto;
    }
    @media (max-width: 1040px) {
      .grid { grid-template-columns: 1fr; }
      .meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Oracle Demo Viewer</div>
      <h1>AegisDesk Trajectory Viewer</h1>
      <div class="sub">
        Inspect the near-perfect oracle trajectory for any core or extended task. Each
        step shows the exact action, reward, rubric progress, penalties, and full rubric
        breakdown so judges can understand how the benchmark scores a trajectory.
      </div>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="panel-head"><h2>Report Controls</h2></div>
        <div class="panel-body">
          <div id="status" class="status">Loading task catalog...</div>

          <div class="field">
            <label for="taskSelect">Task</label>
            <select id="taskSelect"></select>
          </div>

          <div class="field">
            <label for="seedInput">Seed</label>
            <input id="seedInput" type="number" value="7" min="1" step="1">
          </div>

          <button id="loadBtn" type="button">Load Oracle Trajectory</button>

          <div class="field">
            <label>Oracle Reference Path</label>
            <pre id="oraclePath">No report loaded yet.</pre>
          </div>

          <div class="field">
            <label>Raw Report</label>
            <pre id="rawReport">No report loaded yet.</pre>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Trajectory Summary</h2></div>
        <div class="panel-body">
          <div class="meta">
            <div class="meta-card"><strong>Task</strong><span id="metaTask">-</span></div>
            <div class="meta-card"><strong>Track</strong><span id="metaTrack">-</span></div>
            <div class="meta-card"><strong>Difficulty</strong><span id="metaDifficulty">-</span></div>
            <div class="meta-card"><strong>Final Score</strong><span id="metaScore">-</span></div>
            <div class="meta-card"><strong>Steps</strong><span id="metaSteps">-</span></div>
          </div>

          <div class="field">
            <label>Task Brief</label>
            <pre id="taskBrief">No report loaded yet.</pre>
          </div>

          <div class="field">
            <label>Step Trace</label>
            <div id="stepList" class="step-list"></div>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const taskSelect = document.getElementById("taskSelect");
    const statusBox = document.getElementById("status");

    function setStatus(message, kind = "ok") {
      statusBox.textContent = message;
      statusBox.className = "status" + (kind === "ok" ? "" : " " + kind);
    }

    function pretty(value) {
      return JSON.stringify(value, null, 2);
    }

    async function api(path) {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(await response.text() || ("Request failed with status " + response.status));
      }
      return response.json();
    }

    function renderSummary(report) {
      document.getElementById("metaTask").textContent = report.task_id;
      document.getElementById("metaTrack").textContent = report.track;
      document.getElementById("metaDifficulty").textContent = report.difficulty;
      document.getElementById("metaScore").textContent = report.final_score.toFixed(2);
      document.getElementById("metaSteps").textContent = String(report.step_count);
      document.getElementById("taskBrief").textContent = report.task_brief;
      document.getElementById("oraclePath").textContent = report.oracle_reference_path.join("\\n");
      document.getElementById("rawReport").textContent = pretty(report);

      const stepList = document.getElementById("stepList");
      stepList.innerHTML = "";
      report.steps.forEach((step) => {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.innerHTML = `<span>Step ${step.step}</span><span>reward=${step.reward.toFixed(2)} | progress=${step.rubric_progress.toFixed(2)} | done=${step.done}</span>`;
        details.appendChild(summary);

        const body = document.createElement("div");
        body.className = "step-body";
        body.innerHTML = `
          <div class="pill-row">
            <span class="pill">active ticket: ${step.active_ticket_id || "-"}</span>
            <span class="pill">error: ${step.last_action_error || "null"}</span>
            <span class="pill">final score snapshot: ${step.final_score ?? "-"}</span>
          </div>
          <div>
            <strong>Action</strong>
            <pre>${pretty(step.action)}</pre>
          </div>
          <div>
            <strong>Focus Panel</strong>
            <pre>${pretty(step.focus_panel)}</pre>
          </div>
        `;

        const table = document.createElement("table");
        table.innerHTML = `
          <thead>
            <tr>
              <th>Check</th>
              <th>Score</th>
              <th>Weighted</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            ${step.rubric_breakdown.map((item) => `
              <tr>
                <td>${item.check_id}</td>
                <td>${item.score}</td>
                <td>${item.weighted_score}</td>
                <td>${item.details}</td>
              </tr>
            `).join("")}
          </tbody>
        `;
        body.appendChild(table);
        details.appendChild(body);
        stepList.appendChild(details);
      });
    }

    async function loadTasks() {
      const payload = await api("/tasks");
      taskSelect.innerHTML = "";
      payload.tasks.forEach((task) => {
        const option = document.createElement("option");
        option.value = task.task_id;
        option.textContent = `${task.task_id} (${task.track} / ${task.difficulty})`;
        taskSelect.appendChild(option);
      });
      setStatus("Task catalog loaded. Choose a task and load the oracle trajectory.");
    }

    async function loadReport() {
      const button = document.getElementById("loadBtn");
      button.disabled = true;
      try {
        setStatus("Generating oracle trajectory...", "ok");
        const seed = Number(document.getElementById("seedInput").value || 7);
        const report = await api(`/trajectory-report?task_id=${encodeURIComponent(taskSelect.value)}&seed=${seed}`);
        renderSummary(report);
        setStatus("Oracle trajectory loaded.");
      } catch (error) {
        setStatus(String(error), "error");
      } finally {
        button.disabled = false;
      }
    }

    document.getElementById("loadBtn").addEventListener("click", loadReport);
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
                "track": task_track(fixture.task_id),
                "difficulty": fixture.difficulty.value,
                "task_brief": fixture.task_brief,
                "max_steps": fixture.max_steps,
                "reply_template_id": fixture.reply_requirements.template_id,
                "reply_checklist": fixture.reply_requirements.checklist,
                "oracle_available": True,
            }
            for fixture in (fixtures[task_id] for task_id in all_task_ids())
        ]
    }


def benchmark_card_payload() -> dict[str, Any]:
    """Return a compact public summary of the benchmark."""

    fixtures = load_all_fixtures()
    catalog = [
        {
            "task_id": fixture.task_id,
            "track": task_track(fixture.task_id),
            "difficulty": fixture.difficulty.value,
            "max_steps": fixture.max_steps,
        }
        for fixture in (fixtures[task_id] for task_id in all_task_ids())
    ]
    core_count = sum(1 for task in catalog if task["track"] == "core")
    extended_count = sum(1 for task in catalog if task["track"] == "extended")
    return {
        "name": "AegisDesk",
        "env_name": "support_ops_env",
        "status": "ok",
        "summary": "Deterministic OpenEnv benchmark for B2B SaaS support operations.",
        "task_counts": {
            "core": core_count,
            "extended": extended_count,
            "total": len(catalog),
        },
        "features": [
            "typed action and observation models",
            "deterministic rubric grading",
            "dense reward shaping with penalties",
            "interactive console",
            "oracle trajectory viewer",
            "OpenAI-client baseline inference",
        ],
        "routes": {
            "console": "/console",
            "trajectory_viewer": "/trajectory-viewer",
            "benchmark_card": "/benchmark-card",
            "tasks": "/tasks",
            "health": "/health",
        },
        "tasks": catalog,
    }


@app.get("/benchmark-card")
def benchmark_card() -> dict[str, Any]:
    """Return a machine-readable benchmark summary for judges and demos."""

    return benchmark_card_payload()


@app.get("/home", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
def home() -> str:
    """Serve the human-facing landing page."""

    return LANDING_HTML


@app.get("/", response_model=None)
def root(request: Request) -> Any:
    """Return JSON for validators and HTML for browser clients."""

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(LANDING_HTML)
    return {"status": "ok", "env_name": "support_ops_env"}


@app.get("/health")
def health() -> dict[str, str]:
    """Dedicated health endpoint for container and uptime checks."""

    return {"status": "ok", "env_name": "support_ops_env"}


@app.get("/console", response_class=HTMLResponse)
def console() -> str:
    """Serve a lightweight operator console for manual benchmark exploration."""

    return CONSOLE_HTML


@app.get("/trajectory-report")
def trajectory_report(task_id: str, seed: int = 7) -> dict[str, Any]:
    """Return a step-by-step oracle trajectory report for one task."""

    try:
        return generate_trajectory_report(task_id, seed=seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/trajectory-viewer", response_class=HTMLResponse)
def trajectory_viewer() -> str:
    """Serve a read-only oracle trajectory inspection UI."""

    return TRAJECTORY_VIEWER_HTML


def main() -> None:
    """Run the app with uvicorn."""

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
