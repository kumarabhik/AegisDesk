# support_ops_env Roadmap

## Status Legend
- `[x]` Done
- `[~]` Partially done / in progress
- `[ ]` Not started

## Summary
This roadmap turns `support_ops_env` from an empty repo into a complete OpenEnv project for B2B SaaS support operations. The work is organized to reduce validator risk early, then layer in task content, dense rewards, inference reproducibility, hackathon-compliant HF-router inference, and deployment readiness.

## [x] Phase 0: Planning and project definition
Deliverables:
- Define the environment direction, task shape, reward philosophy, and OpenEnv-compatible architecture.
- Create `design_doc.md` and `roadmap.md` as the implementation guide for the project.

Exit criteria:
- The environment concept and task set are locked.
- The design and roadmap documents exist and are ready to guide implementation.

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
- Keep `design_doc.md` and `roadmap.md` aligned with the actual implementation.

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
- A real baseline run completed successfully in under 2 minutes with a mean score of `0.2667`.
- A hackathon-path baseline run also completed successfully through the HF router with the same mean score of `0.2667`.

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
- `python -m pytest` passes locally with 14 tests.
- `openenv validate` passes locally.
- `docker build` succeeds locally.
- `docker run` succeeds locally.
- live container checks for `/`, `/reset`, `/step`, and `/state` succeed.
- Hugging Face Space deployment is live and the public `.hf.space` root returns `200`.

## [x] Phase 8: Hackathon submission compliance
Deliverables:
- Update `inference.py` so the preferred submission path uses `HF_TOKEN` with the Hugging Face router.
- Update `README.md`, `design_doc.md`, and `roadmap.md` so the documented setup matches the hackathon guidance.
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
- `python -m pytest` passes locally with 14 tests after the HF-router updates.
- HF CLI authentication is configured locally and the Space variables/secrets were set through the HF API.
- The Hugging Face Space remote now points to the project commit and the live `.hf.space` URL returns `200`.
- Live checks for `/`, `/reset`, `/step`, and `/state` succeed.

## [x] Phase 9: Final hardening and submission checks
Deliverables:
- Add a reusable verification helper for local and live HTTP endpoint checks.
- Document a concise final submission checklist in the README.
- Sync the design doc with the final deployed state so the repo is self-describing.
- Re-run validation and live checks after the hardening pass.

Exit criteria:
- A single command can verify the root, reset, step, and state endpoints of the running service.
- The README contains a final checklist that can be followed without chat context.
- The design doc reflects the final live deployment status and baseline path.

Current status:
- `verify_space.py` is the reusable live/local verification helper for `/`, `/reset`, `/step`, and `/state`.
- README includes a final submission checklist and live verification command.
- The design doc now includes a deployment status snapshot with the live Space URL and baseline scores.

## Cross-cutting quality gates
These checks apply throughout the project:
- deterministic grading only
- no external runtime dependencies for task execution
- no observation leakage of hidden grader truth
- fixed task count of 3 for v1
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

## Risks and mitigations
Risk: the action schema becomes too flexible and makes grading ambiguous.
Mitigation: keep reply and resolution actions structured and enum-driven.

Risk: reward shaping drifts away from final grading.
Mitigation: derive both from the same rubric engine.

Risk: hard task becomes unfair or under-specified.
Mitigation: anchor it in explicit fixture rules, approved-contact checks, and security escalation paths.

Risk: OpenEnv validator integration fails late.
Mitigation: scaffold the standard file layout first and validate continuously.

## Definition of done
`support_ops_env` is done when:
- the environment implements the full OpenEnv interface
- all 3 tasks are present with deterministic graders
- dense rewards reflect real progress
- `inference.py` produces reproducible scores
- Docker builds cleanly
- the project is ready for Hugging Face Spaces deployment
