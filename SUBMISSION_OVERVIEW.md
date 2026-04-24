# AegisDesk Submission Overview

`AegisDesk` is the public-facing name for `support_ops_env`, and this document is the judge-facing narrative for the project as a benchmark, not just as a code repository. The goal of this document is to explain, in a more polished and professional tone, why the environment exists, what makes it useful, and what is distinctive about it compared with more conventional toy-style reinforcement learning environments.

The project models support operations inside a fictional B2B SaaS company. That is an important design decision because it places the benchmark in a workflow that is both common and consequential. Modern software companies spend enormous effort handling billing corrections, incident triage, access verification, and suspicious account activity. These tasks are repetitive enough to be structured, but sensitive enough that a bad agent can do real harm. That tension is exactly what makes them valuable for evaluating model behavior. A benchmark in this space can test not just whether an agent can "finish a task," but whether it can choose the right task, gather evidence before acting, avoid unsafe shortcuts, and communicate the outcome in a structured and policy-compliant way.

What makes this environment distinctive is the way it balances realism and determinism. Many realistic environments become messy because they depend on a browser, a live toolchain, or hidden evaluation logic that is difficult to debug. Many deterministic benchmarks, on the other hand, become too abstract to say anything useful about real-world agent behavior. `AegisDesk` tries to hold the middle ground. It uses fixture-backed episodes so the environment is fast, reproducible, and easy to validate, but it still presents the agent with enough ambiguity to force meaningful judgment. Each episode contains a small inbox with one true target ticket and one or two distractors. That means the agent must first decide what matters, not just how to act after the right object has already been selected for it.

The task design is intentionally progressive. The easy task, `billing_seat_adjustment`, teaches and evaluates grounded workflow behavior in a familiar support case: inspect the relevant account and invoice records, apply the correct credit, update the metadata, and finalize the resolution cleanly. The medium task, `login_incident_triage`, introduces the need for restraint. The agent must recognize that a customer problem is part of an active incident and resist taking the kind of risky direct action that might feel superficially helpful but would be operationally wrong. The hard task, `suspicious_admin_request`, pushes the benchmark into security-sensitive territory. Here the agent must detect that a request is both high-risk and under-verified, inspect the correct security context, escalate appropriately, and refuse unsafe fulfillment. Together, these tasks form a compact but meaningful capability ladder.

Another strong point of the project is the reward design. Instead of using a sparse success metric that only triggers at the end, the environment computes dense reward from rubric progress plus explicit behavior adjustments. This matters because it gives reinforcement learning systems a richer signal over the whole trajectory. The agent is rewarded for getting closer to the correct operational behavior and penalized for things like invalid payloads, repeated irrelevant inspection, loops, or unsafe actions. That makes the environment more informative than a plain pass-fail judge while still remaining deterministic. In a benchmark setting, this is especially useful because it lets a researcher inspect not only the final score, but also how the score evolved as the agent interacted with the environment.

The environment also makes a deliberate choice not to rely on model-graded free text for scoring. The reply action is structured around a template identifier and a checklist of reply intents rather than fuzzy semantic evaluation. That may seem restrictive at first glance, but it is actually a strength in a benchmark intended for reproducible evaluation. A benchmark is more trustworthy when another person can replay the same trajectory and get the same result. By keeping grading deterministic and state-based, the environment is easier to audit, easier to validate, and harder to exploit through prompt-sensitive evaluator behavior.

Operationally, the project is mature. It includes a typed OpenEnv-compatible environment, a live Hugging Face Space deployment, a working Docker path, a baseline inference runner, a local and remote verification helper, and a higher-level submission audit tool. The project is not just code that "ought" to work. It has already been exercised locally and against the deployed Space. The Hugging Face router path using `HF_TOKEN` has been verified, the live Space endpoints return healthy results, and the baseline path has been run end to end. That means the repository is not merely descriptive; it is backed by a tested operational workflow.

That operational maturity is now backed by a checked-in evidence trail as well. The exact official pre-validation script passes locally against the live Space, and the latest validator output, verification output, latency benchmark, and baseline run are captured in `RESULTS.md`. The public delivery endpoints are:
- GitHub: `https://github.com/kumarabhik/AegisDesk`
- Hugging Face Space: `https://huggingface.co/spaces/I4mGr00T/Meta`
- Live app: `https://i4mgr00t-meta.hf.space`

From a submission point of view, the strongest story this project tells is that it evaluates the kind of judgment that real operators and real agent products need. It does not reduce the problem to classification or syntax. The agent must prioritize, investigate, mutate state carefully, escalate when necessary, and communicate cleanly. Those are exactly the capabilities that become important when agents move from isolated demos into production-facing business workflows. That is why `support_ops_env` is not only a valid hackathon environment, but also a useful benchmark direction in its own right.

---

## Round 2 Extension — Scaler Hackathon

### Problem Statement

AegisDesk v2 extends the benchmark to cover all four Round 2 themes: **Multi-Agent Interactions**, **Long-Horizon Planning & Instruction Following**, **World Modeling**, and **Self-Improving Agent Systems**. The core domain — B2B SaaS support operations — is preserved because it remains one of the richest natural evaluation surfaces for real-world agent judgment.

### Environment

The v2 environment is an evolution of the same OpenEnv-compliant FastAPI server, now with nine fixture-backed tasks, a `WorldStateEngine` that provides dynamic operational context, multi-agent injection hooks, and long-horizon episode support. All evaluation remains deterministic — no LLM judges, no live data.

### Multi-Agent Interactions

Two new agents are added. The `CustomerSimAgent` injects deterministic customer follow-up messages at configured steps during an episode, simulating realistic mid-case escalations derived from the ABCD customer service dataset. The `QualityReviewAgent` scores each support decision post-step for compliance and policy adherence, contributing 15% of the dense reward signal. Two new tasks, `customer_escalation_chain` and `multi_tier_billing_dispute`, are designed around these multi-party interaction patterns.

### Long-Horizon Planning & Instruction Following

Episodes can now run up to 30 steps. Two new tasks require multi-phase completion. `data_breach_response_lifecycle` follows a five-phase security incident protocol — Detection, Containment, Assessment, Notification, Resolution — where phases must be completed in order. `contract_renewal_negotiation` requires resolving two independent sub-cases (a billing dispute and an API incident) before finalizing an enterprise renewal. The grader rewards each phase completion with a +0.05 bonus, making the dense reward more informative over long trajectories.

### World Modeling

The `WorldStateEngine` maintains a fixture-driven context for each task: active incidents, a policy window, regional state, and account health. World-modeling tasks (`service_reinstatement_review`, `api_partner_access_audit`, and the upgraded `login_incident_triage`) require agents to consult this context before acting. An agent that ignores the active policy window and self-approves an extended API grant during a legal review will trigger a terminal forbidden action — the same kind of real-world constraint that makes the environment genuinely useful for evaluation. Dataset patterns from the Schema-Guided Dialogue (SGD) corpus informed the world-state design.

### Self-Improving Agent Systems

The self-improvement pipeline turns the benchmark into its own training data source. After each evaluation run, the `TrajectoryHarvester` separates winning episodes (score ≥ 0.7) from failing ones (score < 0.3). The `DPOPairGenerator` creates (chosen, rejected) training pairs from the same task, where the contrast is a critical early action that separates successful from unsuccessful trajectories. The `AdaptiveDifficultyScheduler` adjusts per-task training weights in GRPO using a rolling score history — tasks where the agent is already performing well get lower weight; tasks where the agent is struggling get higher weight. The end-to-end pipeline is exposed as `training/self_improve.py`, which runs the full loop in a single command. The base model is upgraded from `Qwen3-0.6B` to `Qwen3-4B` for better instruction following across the expanded nine-task curriculum.

### Dataset Sources

The v2 task fixtures are not raw data ingestion — all episodes remain fixture-backed and in-memory. Instead, the ABCD and tau-bench datasets informed the design of new tasks. ABCD's 55-action customer service taxonomy shaped the multi-agent tasks. tau-bench's multi-intent retail patterns shaped the long-horizon tasks. SGD's 20-domain world-state structure informed the `WorldStateEngine` design. This approach keeps the benchmark deterministic and reproducible while grounding the task content in realistic agent behavior patterns documented in the research literature.

### Evaluation Summary

| Dimension | v1 | v2 |
|---|---|---|
| Tasks | 3 | 9 |
| Max episode length | 12 steps | 30 steps |
| Multi-agent | No | CustomerSimAgent + QualityReviewAgent |
| World context | No | WorldStateEngine (fixture-driven) |
| Phase structure | No | Up to 5 phases with completion bonuses |
| Self-improvement | Manual GRPO only | Full harvester → DPO → GRPO pipeline |
| Base training model | Qwen3-0.6B | Qwen3-4B |
| Expected post-training mean | ~0.35 | ~0.55+ (3 rounds) |
