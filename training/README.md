# Training AegisDesk with TRL + OpenEnv

This directory contains optional, post-submission extras for training experiments. It does not change the judged environment or the validator-facing submission path. The goal here is to make `AegisDesk` usable not only as an evaluated benchmark, but also as a starting point for GRPO-based reinforcement learning with TRL.

Related links:
- GitHub repo: `https://github.com/kumarabhik/AegisDesk`
- Hugging Face Space: `https://huggingface.co/spaces/I4mGr00T/Meta`
- Live app: `https://i4mgr00t-meta.hf.space`
- Latest captured validator and benchmark outputs: `RESULTS.md`

The implementation in this folder follows the current TRL OpenEnv integration pattern documented by Hugging Face. In that pattern, you do not write a custom rollout loop by hand unless you have to. Instead, you define an environment class with a `reset()` method and a set of public tool methods. TRL discovers those methods automatically, exposes them as function-calling tools, and handles the multi-turn loop internally. That is the approach used in `train_grpo_aegisdesk.py`.

The training wrapper is intentionally conservative. It does not replace the benchmark's own reward shaping or grader logic. Instead, it talks to the deployed environment through the existing client, stores the current score on the environment instance, and returns that score through a simple reward function. This makes the setup easy to reason about and keeps the training example close to the real benchmark behavior.

## What this starter does

The starter script:

- connects to a running AegisDesk environment, usually the live HF Space
- resets into one of the three canonical tasks
- exposes environment actions as meaningful tool methods such as `open_ticket`, `inspect_record`, `apply_credit`, and `finalize_resolution`
- lets TRL handle the multi-turn tool-calling conversation
- reads the latest benchmark score from the environment instance and uses it as the training reward

This is a small but useful bridge from the benchmark into RL experimentation.

## Install

First install the benchmark itself:

```bash
pip install -e .
```

Then install the training stack:

```bash
pip install accelerate datasets transformers trl
```

If you want a more isolated setup, create a fresh virtual environment first.

## Run

The easiest way to start is:

```bash
accelerate launch training/train_grpo_aegisdesk.py
```

If you want to point at a different environment host or choose a different base model:

```bash
accelerate launch training/train_grpo_aegisdesk.py \
  --env-url https://i4mgr00t-meta.hf.space \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/aegisdesk-grpo
```

The default script is intentionally lightweight. It is meant to be a starter, not a "one click best model" recipe.

## What to tune next

If you want to push this further, the most important knobs are:

- `num_generations`
- `max_completion_length`
- `gradient_accumulation_steps`
- `num_train_epochs`
- `repeat-count`

You can also experiment with different reward definitions. The current starter uses the benchmark's latest score directly, which is the safest and simplest baseline. Later you could compare this against binary success rewards or mixed reward shaping if you want to study training behavior more systematically.

## Why this is optional

The Round 1 submission requirements are already satisfied without this directory. These files are here because the surrounding OpenEnv tutorials also emphasize the training story, and AegisDesk is a much more useful project if it can serve as both a benchmark and a learning environment.
