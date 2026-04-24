# AegisDesk Roadmap

## Status Legend
- `[x]` Done
- `[~]` Partially done / in progress
- `[ ]` Not started

## Summary
This roadmap turns `AegisDesk` from an empty repo into a complete OpenEnv project for B2B SaaS support operations. The underlying technical environment id remains `support_ops_env`, while the public-facing benchmark name is `AegisDesk`. The work is organized to reduce validator risk early, then layer in task content, dense rewards, inference reproducibility, hackathon-compliant HF-router inference, and deployment readiness.

## [x] Phase 0: Planning and project definition
Deliverables:
- Define the environment direction, task shape, reward philosophy, and OpenEnv-compatible architecture.
- Create the implementation guide for the project.

Exit criteria:
- The environment concept and task set are locked.
- The roadmap is ready to guide implementation.

## [x] Phase 1: Scaffold and project shape
Deliverables:
- Create the OpenEnv-compatible repository structure.
- Add root package metadata, `openenv.yaml`, `client.py`, `models.py`, `server/app.py`, `server/environment.py`, and Docker entrypoints.
- Decide the internal module boundaries for fixtures, grading, and rewards.

Implementation notes:
- Prefer the standard OpenEnv template shape so validation is straightforward.
- Keep the first pass minimal but structurally correct.

Exit criteria:
- Repository layout matches the expected OpenEnv structure.
- Stub environment can be imported without errors.
- `openenv validate` is expected to pass once implementation stubs are completed.

## [x] Phase 2: Models and fixture schema
Deliverables:
- Define `SupportAction`, `SupportObservation`, `SupportState`, and nested models such as ticket summaries, records, mutations, and rubric entries.
- Create a fixture schema for tasks, records, rubrics, forbidden actions, and reply requirements.
- Add validation rules for action-specific payload requirements.

Implementation notes:
- Keep action shape strict enough for deterministic server behavior.
- Keep observation shape compact and state shape rich enough for debugging and grading.

Exit criteria:
- Fixtures load successfully into typed models.
- Invalid payloads are rejected predictably.
- Observation and state boundaries are clear and documented.

## [x] Phase 3: Core environment logic
Deliverables:
- Implement `reset()`, `step()`, and `state()` with in-memory state transitions.
- Add support for ticket focus changes, record inspection, KB search, ticket metadata mutation, credits, escalations, structured reply drafting, and finalization.
- Build observation projection so only opened content is exposed.

Implementation notes:
- Track every mutation in append-only logs for transparency.
- Make step outcomes explicit so grading and reward logic can be computed from state.

Exit criteria:
- Manual smoke runs show correct state reset and transitions.
- Max-step handling works.
- Illegal actions do not crash the environment.

## [x] Phase 4: Graders and reward shaping
Deliverables:
- Implement a shared rubric engine for terminal grading and dense rewards.
- Add penalty logic for invalid actions, looping, irrelevant repeated inspection, and unsafe operations.
- Add fixture-driven catastrophic unsafe action handling.

Implementation notes:
- Recompute rubric after every action from normalized state rather than applying ad hoc incremental mutations.
- Keep all scorer behavior deterministic and auditable.

Exit criteria:
- Each task emits partial reward over the trajectory.
- Final scores are deterministic and clamped to `[0.0, 1.0]`.
- Unsafe actions visibly affect reward and score.

## [x] Phase 5: Canonical task content
Deliverables:
- Author the 3 official tasks:
  - `billing_seat_adjustment`
  - `login_incident_triage`
  - `suspicious_admin_request`
- Include 2-3 tickets per episode with realistic distractors.
- Add linked records, expected reply checklist items, rubric weights, and forbidden actions.

Implementation notes:
- Make the easy task direct and instructional.
- Make the medium task test incident awareness and restraint.
- Make the hard task test security judgment and escalation behavior.

Exit criteria:
- Oracle action paths score near `1.0`.
- Common failure paths score materially lower.
- Difficulty increases cleanly from easy to hard.

## [x] Phase 6: Baseline and documentation
Deliverables:
- Implement root `inference.py` using the OpenAI client and environment variables.
- Add fixed prompting, JSON action parsing, fallback behavior, and deterministic task order.
- Write `README.md` with environment motivation, action and observation spaces, task descriptions, setup, validation, deployment, and baseline scores.
- Keep the public documentation aligned with the actual implementation.

Implementation notes:
- Use `temperature=0` by default.
- Print per-task scores plus an overall mean in a reproducible format.

Exit criteria:
- Baseline completes under 20 minutes on constrained hardware.
- Documentation matches the live code paths.
- Local users can run the environment and the baseline from the README alone.

Current status:
- `inference.py` is implemented with OpenAI-env defaults plus Groq/xAI-compatible aliases.
- README is implemented and now includes verified local validation status plus recorded baseline scores.
- Historical exact-score baseline runs completed successfully in under 2 minutes with a mean score of `0.2667`.
- `RESULTS.md` now captures the latest rerun outputs, including the current rounded live baseline summary.

## [x] Phase 7: Validation and deployment
Deliverables:
- Add unit tests and integration smoke tests.
- Run `openenv validate`.
- Verify local `docker build` and `docker run`.
- Prepare the repo for Docker-based Hugging Face Spaces deployment on port `7860`.

Implementation notes:
- Record actual baseline scores only after the environment logic is final.
- Treat validator compatibility and health checks as release blockers.

Exit criteria:
- Container responds successfully to health checks and `reset()`.
- Validation passes.
- The project is ready to push to a Hugging Face Space tagged `openenv`.

Current status:
- `python -m pytest` passes locally with 33 tests.
- `openenv validate` passes locally.
- `docker build` succeeds locally.
- `docker run` succeeds locally.
- live container checks for `/`, `/reset`, `/step`, and `/state` succeed.
- Hugging Face Space deployment is live and the public `.hf.space` root returns `200`.

## [x] Phase 8: Hackathon submission compliance
Deliverables:
- Update `inference.py` so the preferred submission path uses `HF_TOKEN` with the Hugging Face router.
- Update the public docs so the documented setup matches the hackathon guidance.
- Push the repo to the Hugging Face Space and configure the required variables/secrets.
- Verify the live Space returns `200` and supports `/`, `/reset`, `/step`, and `/state`.
- Record the final submission-ready baseline path and final deployment status.

Implementation notes:
- Keep using the OpenAI Python client, but point it at `https://router.huggingface.co/v1` for the official hackathon path.
- Treat Groq/xAI aliases as local development fallbacks rather than the default documented path.
- The final submission should rely on `HF_TOKEN` rather than a paid OpenAI key.

Exit criteria:
- `inference.py` works with `HF_TOKEN` and the Hugging Face router.
- The Hugging Face Space is live and reachable.
- README instructions are fully aligned with what the judges will run.
- The latest pushed submission is the intended final candidate.

Current status:
- Core environment implementation is complete and locally verified.
- `inference.py` now prefers `HF_TOKEN` plus `https://router.huggingface.co/v1` and this path is covered by tests.
- `python -m pytest` now passes locally with 33 tests after the latest app and tooling updates.
- HF CLI authentication is configured locally and the Space variables/secrets were set through the HF API.
- The Hugging Face Space remote now points to the project commit and the live `.hf.space` URL returns `200`.
- Live checks for `/`, `/reset`, `/step`, and `/state` succeed.
- `inference.py` now emits tagged `[START]`, `[STEP]`, and `[END]` log lines during live HF-router runs.
- Local pre-validation mirrors now exist at `validate-submission.sh` and `validate-submission.ps1`.
- The exact official hackathon pre-validation script now passes locally against the live Space.

## [x] Phase 9: Final hardening and submission checks
Deliverables:
- Add a reusable verification helper for local and live HTTP endpoint checks.
- Document a concise final submission checklist in the README.
- Sync the retained public docs with the final deployed state so the repo is self-describing.
- Re-run validation and live checks after the hardening pass.

Exit criteria:
- A single command can verify the root, reset, step, and state endpoints of the running service.
- The README contains a final checklist that can be followed without chat context.
- The retained public docs reflect the final live deployment status and baseline path.

Current status:
- `verify_space.py` is the reusable live/local verification helper for `/`, `/reset`, `/step`, and `/state`.
- README includes a final submission checklist and live verification command.
- `RESULTS.md` now captures the latest live deployment and baseline evidence.

## [x] Phase 10: Submission audit and executive polish
Deliverables:
- Add a single-command submission audit that covers tests, validator status, remote head, and live Space health.
- Surface the new audit path in the README alongside the existing verification helper.
- Refresh roadmap metrics so the tracked counts and deployment state match the live repo.

Exit criteria:
- `submission-audit --space-url https://i4mgr00t-meta.hf.space` succeeds.
- README exposes both the quick verifier and the full submission audit.
- Roadmap status reflects the actual final state of the project.

Current status:
- `submission_audit.py` provides a compact JSON readiness report for local and live checks.
- `tests/test_submission_audit.py` covers the audit helper logic.
- The project now has 33 passing tests and a reusable final verification workflow.

## [x] Phase 11: Narrative and operator documentation
Deliverables:
- Add polished reader-facing documentation that explains the benchmark's real-world value and what makes it distinctive.
- Add a checked-in `.env.example` so local setup is easier without ever committing secrets.

Exit criteria:
- A new reader can understand the project, run it, and verify it without relying on chat history.
- The repo includes enough public documentation for operators and judges.
- Local setup variables are documented in a non-secret example file.

Current status:
- README, `design_doc.md`, `SUBMISSION_OVERVIEW.md`, and `RESULTS.md` provide the retained public benchmark narrative and verification flow.
- `.env.example` documents the expected environment variables without storing secrets.

## [x] Phase 12: Final evaluator compliance
Deliverables:
- Update `inference.py` so stdout strictly follows the required `[START]`, `[STEP]`, and `[END]` format from the latest sample script.
- Run the official hackathon pre-validation script rather than relying only on the local `submission_audit.py` helper.
- Sync the final docs after the inference-format update so reported test counts and submission instructions stay accurate.

Exit criteria:
- `inference.py` emits the exact required structured log format.
- The official pre-validation script passes in this workspace.
- The final docs no longer overstate submission readiness.

Current status:
- The environment itself is complete and deployable.
- The evaluator-facing inference log format is now implemented and covered by tests.
- A PowerShell-friendly local pre-validation script now fails correctly when Docker is unavailable instead of masking the error.
- The official pre-validation script passes locally against `https://i4mgr00t-meta.hf.space`.

## [x] Phase 13: Optional training extras
Deliverables:
- Add an optional TRL + OpenEnv training starter so the benchmark can be used beyond evaluation.
- Surface those extras in the README without changing the judged environment path.

Exit criteria:
- A working training starter script and training notes exist in the repo.
- The core submission wiring remains unchanged.

Current status:
- `training/README.md` and `training/train_grpo_aegisdesk.py` now exist as post-submission training extras.
- The environment id, Space URL, validator path, and baseline submission flow remain unchanged.

## [x] Phase 14: Interactive console and latency polish
Deliverables:
- Add a browser-based console for manual task exploration without changing the judged endpoints.
- Add a task catalog endpoint to support UI discovery and external tooling.
- Reduce first-hit latency by prewarming the fixture cache and shared environment on startup.

Exit criteria:
- A manual user can explore the benchmark from the browser.
- The server exposes a lightweight task discovery endpoint.
- The first-request penalty is reduced through eager startup warming.

Current status:
- `server/app.py` now exposes `/console` and `/tasks`.
- Startup prewarming now loads fixtures and creates the shared environment before the first interactive request.
- `measure_latency.py` now captures before/after startup and first-hit timings for prewarming.
- App smoke tests now cover the new routes and the repo passes with 33 tests.

## [x] Phase 15: Results capture and doc sync
Deliverables:
- Capture the latest exact validator output in a checked-in report.
- Record the latest verification, latency, and baseline results in one place.
- Sync the markdown docs so they all point to the same current evidence.

Exit criteria:
- A single markdown file captures the latest measured results.
- The root docs no longer refer to stale pending validator work.
- Public repo and Space links are documented consistently.

Current status:
- `RESULTS.md` now captures the latest exact official validator pass, live verification output, live inference baseline, and latency benchmark.
- The public GitHub repo and Hugging Face Space links are now surfaced consistently across the docs.

## [ ] Phase 16: Multi-Agent Interactions (Round 2 — Theme 1)
Deliverables:
- Add `CustomerSimAgent` that injects deterministic customer follow-up messages mid-episode.
- Add `QualityReviewAgent` that scores support decisions for compliance, tone, and policy adherence.
- Author 2 new multi-agent tasks: `customer_escalation_chain` and `multi_tier_billing_dispute`.
- Extend `SupportObservation` with a `peer_messages` field for inter-agent communication.
- Dataset source: ABCD (Action-Based Conversations Dataset) — 55 action types, policy-graded outcomes.

New files:
- `server/agents/customer_sim.py`
- `server/agents/quality_review.py`
- `server/task_data/customer_escalation_chain.yaml`
- `server/task_data/multi_tier_billing_dispute.yaml`

Modified files:
- `models.py` — add `peer_messages: list[PeerMessage]` to `SupportObservation`
- `server/environment.py` — agent orchestration hooks

Exit criteria:
- Both new tasks run end-to-end with deterministic scores.
- `CustomerSimAgent` injects a follow-up at step 6 and the rubric rewards correct handling.
- `QualityReviewAgent` review score feeds into dense reward at 15% weight.
- Existing 3 tasks unaffected; all prior tests pass.

## [ ] Phase 17: Long-Horizon Planning & Instruction Following (Round 2 — Theme 2)
Deliverables:
- Support per-task `max_steps` override in fixture schema (default 12, complex tasks 25–30).
- Add `investigation_phases` to fixture schema; each phase has a mini-rubric with partial credit.
- Dense reward emits a `phase_bonus` (+0.05) when phases complete in declared order.
- Author 2 new long-horizon tasks: `data_breach_response_lifecycle` (30 steps, 5 phases) and `contract_renewal_negotiation` (25 steps, 2 sub-cases).
- Dataset source: tau-bench (τ-bench) multi-intent retail patterns + ABCD security escalation flows.

New files:
- `server/task_data/data_breach_response_lifecycle.yaml`
- `server/task_data/contract_renewal_negotiation.yaml`

Modified files:
- `server/fixtures.py` — parse `investigation_phases` and per-task `max_steps`
- `server/grader.py` — add `phase_complete` rubric check kind
- `server/reward.py` — add `phase_bonus` to dense reward
- `models.py` — add `current_phase: int | None` to `SupportObservation`

Exit criteria:
- Oracle path on `data_breach_response_lifecycle` scores ≥ 0.85 with all 5 phases completed.
- Skipping a phase reduces score by ≥ 0.10 vs. in-order completion.
- `max_steps` override works without breaking existing 12-step tasks.

## [ ] Phase 18: World Modeling (Round 2 — Theme 3)
Deliverables:
- Add `WorldStateEngine` (`server/world_state.py`) tracking: active incidents, policy calendar, account health, regional outage map — all fixture-driven and deterministic.
- Fixture YAML gets a new `world_context` block: `active_incidents`, `policy_window`, `region`.
- Grader reads world state for conditional rubric checks (e.g., "do not resolve during active outage").
- Author 3 new world-modeling tasks: `service_reinstatement_review`, `api_partner_access_audit`, `consumer_account_recovery`.
- Dataset source: SGD (Schema-Guided Dialogue) 20-domain world-state templates + tau-bench personal domain patterns.

New files:
- `server/world_state.py`
- `server/task_data/service_reinstatement_review.yaml`
- `server/task_data/api_partner_access_audit.yaml`
- `server/task_data/consumer_account_recovery.yaml`

Modified files:
- `server/environment.py` — integrate `WorldStateEngine` into `reset()` and `step()`
- `server/fixtures.py` — parse `world_context` block
- `server/grader.py` — world-state-conditional rubric checks
- `models.py` — add `world_context: WorldContext | None` to `SupportObservation`

Exit criteria:
- World context appears in observation for world-modeling tasks.
- Grader correctly fails world-conditional checks when world state is violated.
- All 3 new tasks score deterministically.

## [ ] Phase 19: Self-Improving Agent System (Round 2 — Theme 4)
Deliverables:
- `TrajectoryHarvester` collects (prompt, action, score) triples from benchmark runs; splits into winning (≥0.7) and failing (<0.3) trajectories.
- `DPOPairGenerator` creates (chosen, rejected) trajectory pairs from same-task wins vs. failures.
- `AdaptiveDifficultyScheduler` adjusts per-task training weights based on rolling score history (curriculum learning).
- `SelfImproveCLI` runs the full loop: benchmark → harvest → DPO pairs → fine-tune → re-evaluate → delta report.
- Upgrade default GRPO base model from `Qwen3-0.6B` → `Qwen3-4B`.
- Extend GRPO training to cover all 9 tasks (up from 3).

New files:
- `training/trajectory_harvester.py`
- `training/dpo_pair_generator.py`
- `training/adaptive_scheduler.py`
- `training/self_improve.py`

Modified files:
- `training/train_grpo_aegisdesk.py` — Qwen3-4B, adaptive scheduler, all 9 tasks, trajectory logging
- `training/README.md` — full self-improvement pipeline documentation

Exit criteria:
- `python training/self_improve.py --rounds 1 --dry-run` completes without errors.
- Trajectory harvester produces valid JSONL with winning/failing split.
- DPO pair generator outputs valid pairs for at least 2 tasks.
- GRPO config loads all 9 tasks and runs 1 epoch without crash.

---

## Round 2 Submission Definition

### Problem Statement
AegisDesk v2 evaluates AI agents on real-world B2B SaaS support operations — a domain structured enough to score deterministically but complex enough to require genuine judgment. Agents triage competing tickets, investigate records, follow policy, communicate correctly, and avoid unsafe shortcuts across 9 tasks spanning billing, incidents, security, multi-agent coordination, long-horizon planning, and world-aware decision making.

### Environment
- OpenEnv-compliant FastAPI server with 9 fixture-backed tasks
- Multi-agent episodes: `CustomerSimAgent` + primary support agent + `QualityReviewAgent`
- `WorldStateEngine` provides dynamic context (incidents, policies, account health)
- Step limits: 12 (standard), 25–30 (long-horizon tasks)
- Docker + HF Spaces deployment at `https://i4mgr00t-meta.hf.space`

### Agent Capabilities
- 10 action types: inspection, mutation, escalation, communication, finalization
- Structured observations: inbox, active ticket, opened records, world context, peer messages, current phase
- Operates within per-task step limits with deterministic rubric feedback at every step

### Task Roster
| Task | Difficulty | Theme |
|---|---|---|
| `billing_seat_adjustment` | Easy | Baseline |
| `login_incident_triage` | Medium | World Modeling |
| `suspicious_admin_request` | Hard | Baseline |
| `customer_escalation_chain` | Medium-Hard | Multi-Agent |
| `multi_tier_billing_dispute` | Medium | Multi-Agent |
| `data_breach_response_lifecycle` | Hard | Long-Horizon |
| `contract_renewal_negotiation` | Medium-Hard | Long-Horizon |
| `service_reinstatement_review` | Easy-Medium | World Modeling |
| `api_partner_access_audit` | Medium | World Modeling |

### Reward Model / Evaluation Logic
- **Dense reward**: rubric progress delta + behavior adjustments + phase completion bonuses (+0.05/phase)
- **Terminal score**: deterministic rubric `[0.0, 1.0]`, no LLM judge, no free-text scoring
- **Multi-agent scoring**: QualityReviewAgent review score weighted at 15% of final
- **Safety enforcement**: forbidden actions trigger penalties; catastrophic violations terminate episode immediately

### Post-Training / Self-Improvement Strategy
- GRPO fine-tuning on all 9 tasks with `AdaptiveDifficultyScheduler` (curriculum learning)
- DPO pair generation from the benchmark's own trajectory data (self-supervised signal)
- 3-round self-improvement loop: benchmark → harvest → DPO → fine-tune → re-evaluate
- Base model: `Qwen3-4B` (strong instruction following, fits HF compute budget)
- Recommended dataset: **ABCD + tau-bench** patterns used to author the new fixture tasks

### Dataset Choice Rationale
| Dataset | Role |
|---|---|
| ABCD (10K dialogues, 55 action types) | Template for multi-agent and billing task fixtures |
| tau-bench (retail + airline, policy grading) | Template for long-horizon and world-modeling tasks |
| SGD (20-domain world-state) | WorldStateEngine context design |
| MultiWOZ 2.4 (multi-turn, multi-intent) | Complex distractor ticket design |

---

## Cross-cutting quality gates
These checks apply throughout the project:
- deterministic grading only
- no external runtime dependencies for task execution
- no observation leakage of hidden grader truth
- task count: 3 for v1, 9 for v2
- reward shaping remains dense but auditable
- project stays within `vcpu=2`, `memory=8gb`, and baseline runtime under 20 minutes

## Test milestones
Tests should be added incrementally rather than all at the end.

Milestone coverage:
- Phase 2: model validation and fixture parsing tests
- Phase 3: reset/step/state behavior tests
- Phase 4: reward and grader determinism tests
- Phase 5: oracle-path and failure-path task tests
- Phase 6: inference contract and parsing tests
- Phase 7: Docker and validator smoke checks
- Phase 16: multi-agent task oracle paths and CustomerSimAgent injection tests
- Phase 17: phase_complete rubric, phase_bonus reward, long-horizon task oracle tests
- Phase 18: WorldStateEngine fixture loading, world-conditional grader tests
- Phase 19: trajectory harvester JSONL format, DPO pair validity tests

## Risks and mitigations
Risk: the action schema becomes too flexible and makes grading ambiguous.
Mitigation: keep reply and resolution actions structured and enum-driven.

Risk: reward shaping drifts away from final grading.
Mitigation: derive both from the same rubric engine.

Risk: hard task becomes unfair or under-specified.
Mitigation: anchor it in explicit fixture rules, approved-contact checks, and security escalation paths.

Risk: OpenEnv validator integration fails late.
Mitigation: scaffold the standard file layout first and validate continuously.

Risk: multi-agent injection breaks determinism.
Mitigation: CustomerSimAgent uses seeded deterministic policy; no LLM required.

Risk: long-horizon tasks exceed compute budget.
Mitigation: 30-step tasks still run under 5 minutes on inference; GRPO training uses gradient accumulation.

## Definition of done
`support_ops_env` v2 is done when:
- the environment implements the full OpenEnv interface
- all 9 tasks are present with deterministic graders
- dense rewards reflect real progress including phase bonuses
- multi-agent and world-modeling observations are wired end-to-end
- self-improvement pipeline completes a dry-run without errors
- `inference.py` produces reproducible scores across all 9 tasks
- Docker builds cleanly
- the project is ready for Hugging Face Spaces deployment
