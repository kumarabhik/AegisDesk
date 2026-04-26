# AegisDesk: Training Support Agents to Think Before They Act

*A deterministic OpenEnv benchmark for B2B SaaS support operations.*

Most agent benchmarks are easy to demo and hard to trust. They either collapse into short toy tasks that strong models already solve, or they rely on free-form judging where another model quietly sits in the reward loop and decides what "good" means. That makes the score look impressive, but not always reproducible.

AegisDesk was built to solve that problem from the ground up. The goal was not to make another chatbot benchmark. The goal was to build an environment where an agent has to behave like a real enterprise support operator: pick the right ticket from a noisy inbox, inspect the right records before touching account state, follow policy windows, react safely to suspicious requests, and close the case under a real step budget. The more we worked on it, the clearer the challenge became: this is not just language generation. It is sequential decision making with memory, caution, and consequences.

At a glance, AegisDesk is built around four ideas:

- deterministic grading instead of LLM judges
- enterprise support workflows instead of toy chat tasks
- held-out generalization variants instead of a single small pack
- a public training path that judges can inspect and rerun

## Situation

Enterprise support is a surprisingly strong testbed for reinforcement learning. In a real SaaS operations queue, the right action is rarely obvious from the first user message. Two tickets may look similar but require opposite outcomes once you inspect the account state. A refund can be allowed in one case and forbidden in another. An admin request can be routine, or it can be a social engineering trap. Good support work is not about sounding helpful. It is about reading the right evidence before acting.

That is exactly the gap we wanted to capture. We needed a benchmark that felt realistic enough to matter, but deterministic enough to grade without an LLM judge. If the same trajectory could get a different score on a different day, the benchmark would fail its purpose.

## Task

So the task for AegisDesk was very specific: build an OpenEnv environment that is hard enough to train on, strict enough to audit, and broad enough to measure generalization instead of memorization.

We also wanted the benchmark story to be stronger than "here are nine tasks." A serious RL benchmark should not only ask whether a model can improve on the same episodes it sees during training. It should ask whether that improvement transfers to unseen but structurally similar cases.

That is why AegisDesk is organized around **27 judged fixtures**. Nine are canonical training fixtures. Eighteen are held-out generalization variants that share the same business logic, but change the account state, record layout, amounts, and context. On top of that, there are three extra showcase fixtures used for demos, not for the official score. In other words, the environment is designed to answer a harder question: *does the model actually learn the workflow, or does it just learn the examples?*

The benchmark split is simple, but important:

- **9 canonical fixtures** for training and the main development loop
- **18 held-out judged variants** for real generalization checks
- **3 showcase fixtures** for demos, not for the official score

## Action

AegisDesk drops the agent into a simulated support console built on OpenEnv. Every step is a structured JSON action such as opening a ticket, inspecting a record, searching the knowledge base, escalating, applying credit, or finalizing a resolution. The action space is simple to inspect, but the decision process behind each action is not.

What makes the environment interesting is the combination of constraints. The inbox contains distractor tickets. Records are hidden until the agent explicitly inspects them. Several tasks contain policy gates that only become clear after reading the evidence. Some requests are security-sensitive and should be escalated rather than fulfilled. Long-horizon tasks such as breach response and contract renewal have mandatory phase ordering, so an agent that jumps straight to resolution loses reward even if the final answer sounds plausible.

The environment also includes dynamic behavior. A `CustomerSimAgent` can inject follow-up messages mid-episode, which forces the policy to respond to changing context instead of replaying a canned plan. A `QualityReviewAgent` contributes a small deterministic quality component at the end, but the reward path itself does not depend on another model's opinion. The core grader is still fixture-defined and reproducible.

That determinism is the heart of the project. Every fixture encodes its rubric directly: what records must be inspected, what actions are forbidden, what order the phases must happen in, and what the policy windows are. The same fixture plus the same sequence of actions produces the same score every time. That makes AegisDesk much more suitable for RL than a benchmark whose reward changes with phrasing.

The training story follows the same principle. We created a public Kaggle notebook and helper stack around GRPO using Hugging Face TRL, with the live AegisDesk Space acting as the reward environment. We also built supporting corpora from customer-support and task-oriented dialogue datasets so the agent can be warmed up before RL. The important discipline is that the held-out generalization fixtures stay out of training. They are reserved for evaluation only.

In practice, the agent is forced to learn several different behaviors at once:

- choose the right ticket from a distractor-filled inbox
- inspect records before making state-changing actions
- follow policy windows such as limits, approvals, and escalation rules
- survive long-horizon workflows where phases must happen in order
- react safely when a request looks suspicious or incomplete

## Result

The most important result so far is not a single headline score. It is that AegisDesk already shows real benchmark headroom while remaining fully deterministic.

In the latest public reference run, `Qwen/Qwen2.5-72B-Instruct` reached a mean score of **0.325** across the nine canonical tasks in zero-shot multi-step evaluation. That is meaningfully above the earlier `0.27` reference baseline, but still far from saturation. In practice, this is exactly what we want from the benchmark: strong models can make progress, but they do not solve it by default.

Just as important, the environment is not a toy. The benchmark now exposes **30 surfaced fixtures** in total, with **27 judged fixtures** used for the official score. The nine canonical tasks cover baseline execution, world modeling, multi-agent follow-up, and long-horizon phase ordering. The eighteen held-out variants test whether improvements transfer to unseen but structurally similar support episodes. That makes the score much harder to game and much more interesting to interpret.

The project also meets the hackathon requirement that the environment be discoverable and runnable. The Hugging Face Space is live, the benchmark card is machine-readable, the oracle viewer is public, and the GRPO training notebook is available for reruns. In other words, this is not just a paper design. It is a working environment with a public interface and a training path judges can inspect themselves.

For a judge, that means AegisDesk is easy to validate in three different ways:

- you can manually play the console and feel the task difficulty
- you can inspect deterministic oracle traces and benchmark metadata
- you can rerun the public training notebook and inspect reward and loss behavior

## Why This Matters

What makes AegisDesk exciting to us is that it sits in a sweet spot many benchmarks miss. It is realistic enough to feel like work, not trivia. It is strict enough to reward cautious behavior instead of eloquent guessing. And it is reproducible enough to support serious RL experimentation.

That combination matters because enterprise agents will eventually be judged on more than tone. They will be judged on whether they read the right evidence, whether they follow policy, whether they resist unsafe shortcuts, and whether they can recover from ambiguity without human-style handholding. AegisDesk was designed around exactly those behaviors.

If the benchmark does its job well, the result is bigger than one competition submission. It becomes a compact but meaningful environment for studying how operational agents learn to act safely under uncertainty.

That is the real bet behind the project. We are not claiming that enterprise support is the only useful agent benchmark. We are claiming that it is one of the clearest places where RL should matter, because the difference between a weak agent and a strong one is not style. It is whether the model learns to inspect, verify, and act with discipline.

## Try AegisDesk

The live environment is available on Hugging Face Space: [https://i4mgr00t-meta.hf.space](https://i4mgr00t-meta.hf.space)

The GitHub repository is here: [https://github.com/kumarabhik/AegisDesk](https://github.com/kumarabhik/AegisDesk)

The public Kaggle-style training notebook lives here: [training/AegisDesk_Kaggle_GRPO.ipynb](training/AegisDesk_Kaggle_GRPO.ipynb)

The latest public benchmark reference run is here: [training/benchmark_results.json](training/benchmark_results.json)

If you want the shortest version of the story, it is this:

- train on 9 canonical support workflows
- test on 18 held-out variants
- grade everything deterministically
- measure whether the policy learns to think before it acts

Built for the Meta OpenEnv India Hackathon 2026.
