# AegisDesk: A Practical Walkthrough

`AegisDesk` is the public-facing name for `support_ops_env`, a real-world OpenEnv environment built around a workflow that people actually perform in modern software companies: support operations. Instead of asking an agent to play a game, sort abstract symbols, or solve an artificial benchmark, this project asks an agent to behave like a support operator working inside a SaaS company. The agent sees a small inbox, chooses which ticket deserves attention, opens the right internal records, makes safe operational decisions, drafts a structured reply, and closes or escalates the case. That makes the environment useful not only as a hackathon submission, but also as a more serious benchmark for evaluating whether an agent can navigate realistic business workflows without taking unsafe shortcuts.

The core idea is simple, but the design is intentionally disciplined. Every episode contains a small inbox with two or three tickets, one of which is the true target and the others acting as distractors. The agent is not rewarded just for finishing. It is rewarded for making progress in the right direction. That means the environment does not wait until the very last step to decide whether the model did well. Instead, it evaluates the path the agent takes. If the agent opens the correct ticket, inspects the right records, applies the right tags, avoids dangerous actions, and drafts an appropriate structured response, the score improves gradually. If the agent loops, performs irrelevant actions, or takes a policy-violating shortcut, the reward drops. This gives the benchmark the kind of dense feedback that reinforcement learning systems learn from more effectively than sparse pass-or-fail scoring.

The project currently ships with three canonical tasks that form an intentional progression. The easy task, `billing_seat_adjustment`, checks whether the agent can resolve a routine but structured billing correction. The medium task, `login_incident_triage`, tests whether the agent can recognize that a user problem is actually part of a wider incident and avoid taking reckless remediation steps. The hard task, `suspicious_admin_request`, is designed to test security judgment, because the agent has to detect an unverified and potentially malicious request, inspect the right security context, and escalate rather than comply. In all three cases, the environment models a realistic support judgment problem rather than a toy objective.

From a technical point of view, the project is organized around the OpenEnv contract. The typed action, observation, and state models live in `models.py`. The environment logic itself lives in `server/environment.py`, with the API surface exposed by `server/app.py`. The grading logic and dense reward shaping are factored into `server/grader.py` and `server/reward.py`, while the task fixtures live under `server/task_data`. The project also includes `client.py` for interacting with the environment, `inference.py` for running a baseline model against the tasks, `verify_space.py` for checking the running service over HTTP, `run_local_stack.py` for starting and checking a local server in one command, `env_doctor.py` for checking environment-variable readiness without printing secrets, and `submission_audit.py` for running an end-to-end readiness audit that combines local test status, validator status, remote repository visibility, and live Space verification into a single report.

What makes the environment particularly practical is that it stays deterministic while still feeling operationally realistic. The records the agent can inspect include account snapshots, invoices, incident information, approved contacts, and security alerts. The agent can open tickets, inspect records, search the knowledge base, set priority, set status, add tags, apply credits, escalate, draft a structured reply, and finalize a resolution. Because the reply step is structured rather than free-form graded, the environment avoids fuzzy semantic scoring and stays reproducible. This is important in a benchmark setting, because deterministic evaluation is far easier to trust and debug than model-judged free text.

Running the project locally is straightforward. If you want to treat it like a normal Python package, start in the repository root and install the project in editable mode. The simplest commands are:

```bash
pip install -e .
python -m server.app
```

Once the server is running, it listens on port `7860` by default. If you prefer `uvicorn` directly, that works too:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

If you want to verify that the environment is actually behaving correctly rather than just trusting that the server started, the fastest path is to use the verification helper. Against a local server, run:

```bash
python verify_space.py --base-url http://127.0.0.1:7860
```

That command checks the root endpoint, performs a `reset`, takes one valid `step`, and then calls `state` to confirm that the environment is truly stateful. It is a better check than simply hitting the homepage because it verifies the actual OpenEnv-style interaction pattern. If you want an even stronger one-command confidence check, use the submission audit helper instead:

```bash
python submission_audit.py --space-url http://127.0.0.1:7860
```

That audit command runs the test suite, runs `openenv validate`, checks the Git remote visibility, and performs a live API verification. In other words, it gives you a compact JSON view of whether the project is truly ready, not just whether one endpoint happens to respond.

If you do not want to manage multiple terminals manually, the repository now includes `run_local_stack.py`. That helper starts the local API if it is not already running, waits for `/health`, performs the same end-to-end HTTP verification as `verify_space.py`, and then shuts the process down again unless you tell it to stay alive. In practice, that turns the local operator flow into a single command:

```bash
python run_local_stack.py
```

If your goal is to run the baseline model rather than just host the environment, the project supports two main inference paths. The official hackathon path uses the OpenAI client pointed at the Hugging Face router. In that setup, you provide `HF_TOKEN`, optionally provide `API_BASE_URL` if you want to be explicit, set `MODEL_NAME`, and point `ENV_BASE_URL` at either your local server or the deployed Space. Before you do that, you can inspect your local setup without printing any secrets by running `python env_doctor.py`. A working example looks like this:

```bash
HF_TOKEN=...
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-1M
ENV_BASE_URL=https://i4mgr00t-meta.hf.space
python inference.py
```

The project also supports fallback OpenAI-compatible providers for local experimentation, but the preferred benchmark path is the Hugging Face router because it aligns with the hackathon's expected setup. When you run `inference.py`, it evaluates all three canonical tasks in a fixed order and prints a reproducible score report. In the current verified runs, both the compatibility path and the hackathon router path produce the same mean score of `0.2667`, with task scores of `0.2750`, `0.2750`, and `0.2500`.

Verifying the deployed Hugging Face Space works almost exactly the same way as verifying locally. The project is live at `https://i4mgr00t-meta.hf.space/`, and the quickest operational check is:

```bash
python verify_space.py --base-url https://i4mgr00t-meta.hf.space
```

If you want the full final audit instead of just the API check, run:

```bash
python submission_audit.py --space-url https://i4mgr00t-meta.hf.space
```

That gives you a realistic submission-readiness signal because it confirms not only that the Space is alive, but that the local project still passes tests and validator checks too. This is especially useful near submission time, because it reduces the chance that you rely on stale assumptions from an earlier run.

The most important thing to understand about what this project does is that it is not simply a FastAPI app with some JSON endpoints. It is a deliberately structured evaluation environment for agent behavior. The reason the small inbox design works well is that it introduces ambiguity without becoming chaotic. The agent must decide which ticket matters, which records matter, which actions are safe, and whether the correct response is resolution, waiting, or escalation. That creates a richer training and evaluation surface than a single linear task, while still being deterministic enough to validate automatically and cheaply.

There is also a subtle but useful design choice in the way the reward system works. Many environments only care whether the agent got the final answer right. This one cares about the trajectory. That means the benchmark can distinguish between an agent that arrives at a correct resolution through grounded reasoning and one that blindly stumbles into the right answer or takes a harmful shortcut. In real support workflows, that difference matters. A frontline support system that resolves one ticket correctly but violates policy or ignores security signals is not a strong agent. This benchmark was designed to capture that distinction.

If you are reading this document to understand the project as a whole, the best mental model is to think of it as a compact support-operations simulator. It is small enough to run quickly and deterministically, but rich enough to stress prioritization, evidence gathering, safety, escalation judgment, and structured communication. If you are reading it because you want to operate the project, the best practical workflow is to run `python run_local_stack.py`, inspect `python env_doctor.py`, run `submission_audit.py`, and then run `inference.py` with the proper environment variables. That gives you confidence at every level: the server is alive, the environment state is correct, the repository passes validation, and the model-facing path works end to end.

At this point, the project is in strong shape as both a benchmark and a submission artifact. The environment logic is complete, the deployment is live, the baseline path is verified, and the verification tooling is now good enough that another person can pick up the repository and understand not only what it is, but how to trust that it is working.
