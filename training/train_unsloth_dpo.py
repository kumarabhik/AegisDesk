"""Unsloth DPO trainer for the AegisDesk support preference corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", default="training/data/support_pref.jsonl")
    parser.add_argument("--output-dir", "--output", dest="output_dir", default="outputs/aegisdesk-dpo")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="aegisdesk-qwen25-15b-dpo")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", "--epochs", dest="num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--require-project-pairs", action="store_true")
    return parser.parse_args()


def _validate_row(row: dict[str, Any], min_score_gap: float) -> bool:
    prompt = str(row.get("prompt") or "").strip()
    chosen = str(row.get("chosen") or "").strip()
    rejected = str(row.get("rejected") or "").strip()
    if not prompt or not chosen or not rejected:
        return False
    if chosen == rejected:
        return False

    chosen_score = row.get("chosen_score")
    rejected_score = row.get("rejected_score")
    if chosen_score is not None and rejected_score is not None:
        try:
            gap = float(chosen_score) - float(rejected_score)
        except (TypeError, ValueError):
            gap = 0.0
        if gap < min_score_gap:
            return False
    return True


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import DPOTrainer
    from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

    PatchDPOTrainer()
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    original_count = len(dataset)

    def _row_ok(row: dict[str, Any]) -> bool:
        if args.require_project_pairs and row.get("source") == "helpsteer2-preference":
            return False
        return _validate_row(row, min_score_gap=args.min_score_gap)

    dataset = dataset.filter(_row_ok)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError(
            f"No usable DPO rows remained after filtering {Path(args.dataset)} "
            f"(original={original_count}, min_score_gap={args.min_score_gap})."
        )
    print(
        f"Loaded DPO dataset: {len(dataset)} / {original_count} usable rows "
        f"from {Path(args.dataset)}"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            optim="adamw_8bit",
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            report_to=args.report_to,
            run_name=args.run_name,
            seed=3407,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
