# AegisDesk: A Deterministic RL Benchmark for Enterprise Support Operations

*Can a language model do what a real enterprise support operator does? We built a benchmark to find out.*

---

## The Problem

Most agent benchmarks fall into one of two traps:

**Trap 1 — Too simple.** The task is complete in one or two turns, the action space is a handful of choices, and a well-prompted frontier model already scores near-perfect. There is nothing left to train toward.

**Trap 2 — Unjudgeable.** The task requires evaluating free-form output, so you bring in an LLM judge. Now your benchmark has a second model in the reward path, and your scores are as reproducible as that judge's mood.

AegisDesk avoids both. The tasks are genuinely hard — a random agent scores near 0.0, and the Qwen2.5-72B zero-shot baseline lands at 0.27. And the grading is fully deterministic: same fixture, same actions, same score. Every time.

---

## What AegisDesk Is

AegisDesk is an OpenEnv benchmark that simulates a B2B SaaS support console. An agent is dropped into an inbox with multiple tickets, given a task brief, and must complete the case within a step budget.

At each step the agent emits a structured JSON action:

```json
{"action_type": "inspect_record", "record_id": "billing_2024_Q4"}
```

The environment applies the action, updates world state, computes a dense per-step reward, and returns the next observation. The episode ends when the agent finalizes a resolution or exhausts its steps.

The reward formula:

```
reward = progress_delta
       + behavior_adjustment
       + phase_bonus
       + (qa_score × 0.1 × 0.15)
```

There is no language model in the reward path. All rubric items are fixture-defined. The grader is a pure Python function.

---

## Why Enterprise Support Is an Interesting RL Problem

A well-designed enterprise support workflow has every property you want in an RL environment:

**Partial observability.** The inbox shows multiple tickets. Records are only revealed when explicitly requested. The agent must decide what to look at before it can act correctly.

**Policy constraints.** Discounts have ceilings. Escalation has defined triggers. Security-sensitive requests require verification before fulfillment. Acting without reading the right records is penalized.

**Phase ordering.** Long-horizon tasks (data breach response, contract renewal) have phases that must complete in declared order. Jumping to resolution before completing investigation costs reward.

**Multi-agent dynamics.** A `CustomerSimAgent` injects follow-up messages mid-episode. A `QualityReviewAgent` evaluates the final case notes. The agent must handle both.

**Forbidden action traps.** Some actions are terminal — they set `done=True` immediately and lock the score. A good agent learns to recognize when a request looks like a social engineering attempt and escalates rather than fulfills.

This combination makes AegisDesk a more realistic RL training target than a static instruction benchmark, and more auditable than a free-form judge environment.

---

## Benchmark Design

### The fixture model

Each episode is specified by a YAML fixture file. The fixture defines:
- the inbox and available records
- the task brief shown to the agent
- the ordered rubric items (what must happen and in what order)
- the forbidden actions and whether they are terminal
- the policy window (discount limits, escalation thresholds, step budget)

Canonical fixtures use `fixture_id == task_id`. Held-out generalization variants use `fixture_id = <task_id>_v<n>` and share the parent `task_id` but have different account states, amounts, and record layouts.

### The benchmark split

| Pack | Count | Role |
|---|---:|---|
| `core` | 3 | canonical baseline fixtures |
| `v2` | 6 | canonical Round 2 fixtures |
| `generalization` | 18 | held-out judged variants — never used in training |
| `showcase` | 3 | demo-only |
| **judged total** | **27** | official benchmark score |

The 9 canonical fixtures are the training pack. The 18 generalization fixtures are held out completely — they test whether improvement on canonical fixtures transfers to structurally similar but unseen episodes.

### The task mix

The 9 canonical tasks span four capability axes that map directly to the Round 2 competition themes:

| Task | Theme |
|---|---|
| Billing seat adjustment, Suspicious admin request | Baseline — correct ticket selection, evidence inspection |
| Login incident triage, Service reinstatement, API partner audit | World Modeling — current account state drives the correct action |
| Customer escalation chain, Multi-tier billing dispute | Multi-Agent — CustomerSimAgent follow-ups, cross-team coordination |
| Data breach response lifecycle, Contract renewal negotiation | Long-Horizon — 25–30 step episodes with mandatory phase ordering |

---

## Training Pipeline

We trained `Qwen2.5-7B-Instruct` with GRPO (Group Relative Policy Optimization) via TRL's `GRPOTrainer`, using the live AegisDesk Space as the reward environment.

**Reward function:** for each generated completion, we reset the environment with the corresponding task and seed, step with the parsed action, and return the immediate reward. No offline reward model — every training signal comes from the live deterministic grader.

**Dataset:** 72 training rows (9 tasks × 8 seeds) with chat-format prompts. Each prompt includes the system role and a task brief. The model learns to emit a valid JSON action that makes progress on the rubric.

**Training configuration:**
- LoRA rank 16, alpha 32, NF4 4-bit quantization
- 2 epochs, learning rate 5e-6
- Effective batch size 16 (grad_accum=16, batch_size=1)
- G=2 completions per prompt for group relative reward

**Training corpus (for SFT warm-up):**

| Dataset | Rows |
|---|---:|
| Bitext customer support | 5,776 |
| ABCD | 5,000 |
| Schema-Guided Dialogue | 4,000 |
| tau-bench / tau2-bench oracle traces | 69 |
| **Total SFT** | **15,124** |
| HelpSteer2 preference pairs | 7,118 |
| **Total preference** | **7,119** |

**Generalization discipline:** the 18 held-out judged fixtures are excluded from SFT, preference tuning, and GRPO. They only appear at evaluation time.

---

## Results

**Baseline (Qwen2.5-72B, zero-shot, 3 core tasks):** 0.27

After GRPO training on the 9 canonical fixtures with `Qwen2.5-7B-Instruct`:

| Pack | Mean Score |
|---|---:|
| core (3 fixtures) | see `training/benchmark_results.json` |
| v2 (6 fixtures) | see `training/benchmark_results.json` |
| generalization (18 fixtures) | see `training/benchmark_results.json` |
| **all judged (27 fixtures)** | **see `training/benchmark_results.json`** |

Reward curves and per-task scores: [training/reward_curves_overall.png](training/reward_curves_overall.png)

---

## Try It Yourself

**Live Space:** https://i4mgr00t-meta.hf.space

```bash
# Interactive console (manual play)
https://i4mgr00t-meta.hf.space/console

# Oracle trajectory viewer (see the optimal path)
https://i4mgr00t-meta.hf.space/trajectory-viewer

# Benchmark card (machine-readable)
https://i4mgr00t-meta.hf.space/benchmark-card
```

**Run an episode via API:**

```python
import httpx, json

ENV = "https://i4mgr00t-meta.hf.space"

with httpx.Client() as client:
    obs = client.post(f"{ENV}/reset",
                      json={"task_id": "billing_seat_adjustment", "seed": 42}).json()
    print(obs["task_brief"])

    result = client.post(f"{ENV}/step",
                         json={"action_type": "open_ticket",
                               "ticket_id": obs["inbox"][0]["ticket_id"]}).json()
    print(f"reward: {result['reward']:.3f}")
```

**Clone and run locally:**

```bash
git clone https://github.com/kumarabhik/AegisDesk.git
cd AegisDesk
pip install -e .
python -m server.app
# → http://127.0.0.1:7860/console
```

**Train:**

```bash
# On Kaggle T4 — open training/AegisDesk_Kaggle_GRPO.ipynb
# Set HF_TOKEN secret, T4 GPU, Internet ON, then Run All
```

---

## What's Next

- Self-improvement loop: harvest trajectories from the trained model, generate DPO pairs from winner/loser splits, run another GRPO round on the updated policy
- Curriculum scheduling: weight canonical fixtures by current per-task reward deficit
- Longer context: extend max completion length to allow multi-step reasoning traces before the final action JSON

The benchmark, environment, and training pipeline are all open. Pull requests welcome.

---

*Built for the Meta OpenEnv Hackathon 2026.*  
*Code: https://github.com/kumarabhik/AegisDesk*  
*Space: https://huggingface.co/spaces/I4mGr00T/Meta*
