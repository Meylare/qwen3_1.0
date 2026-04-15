"""
Supervised pairwise A/B training from direct choice logits for Qwen3.5-9B-Base.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from pairwise_ab import compute_choice_metrics, get_choice_token_ids, label_to_class_id
from prompting import build_answer_only_prompt
from train_gspo_qwen35_smoke import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    configure_wandb,
    ensure_chat_template,
    ensure_output_dir,
    ensure_smoke_dataset,
    load_preprocess_metrics,
    maybe_probe_sample,
    maybe_wandb_update_summary,
    resolve_model_path,
    sanitize_nested,
)


class PairwisePromptDataset:
    """Wrap a Hugging Face dataset and rebuild prompts in answer-only mode."""

    def __init__(self, base_dataset: Any, fps: float = 2.0):
        self.base_dataset = base_dataset
        self.fps = fps

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = sanitize_nested(self.base_dataset[index])
        if "label" not in row and {"views_a", "views_b"} <= row.keys():
            row["label"] = "A" if row["views_a"] > row["views_b"] else "B"
        if {"video_a", "video_b"} <= row.keys():
            row["prompt"] = build_answer_only_prompt(row, fps=self.fps)
        return row


class PairwiseABCollator:
    def __init__(self, processor: Any, chat_template: str | None = None):
        self.processor = processor
        self.chat_template = chat_template

    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [feature["prompt"] for feature in features]
        encoded = self.processor.apply_chat_template(
            prompts,
            chat_template=self.chat_template,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        batch = {key: value for key, value in encoded.items()}
        batch["labels"] = torch.tensor(
            [label_to_class_id(feature["label"]) for feature in features],
            dtype=torch.long,
        )
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise A/B logits on Qwen3.5-9B-Base.")
    parser.add_argument("--model_name", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--train_dataset", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    parser.add_argument("--eval_dataset", type=Path, default=DEFAULT_DATA_DIR / "eval.jsonl")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "qwen35_9b_base_pairwise_ab",
    )
    parser.add_argument("--preprocess_metrics", type=Path, default=DEFAULT_DATA_DIR / "preprocess_metrics.json")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--finetune_mlp_modules", action="store_true", default=False)
    parser.add_argument("--no_finetune_mlp_modules", action="store_false", dest="finetune_mlp_modules")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--probe_sample", action="store_true", default=True)
    parser.add_argument("--no_probe_sample", action="store_false", dest="probe_sample")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_tags", default=os.environ.get("WANDB_TAGS", ""))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def determine_eval_strategy(args: argparse.Namespace, has_eval_dataset: bool) -> str:
    if not has_eval_dataset:
        return "no"
    if args.eval_steps and args.eval_steps > 0:
        return "steps"
    return "no"


def compute_last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected [batch, seq] attention mask, got {attention_mask.shape}")
    flipped = attention_mask.to(dtype=torch.long).flip(dims=[1])
    distance_from_end = torch.argmax(flipped, dim=1)
    return attention_mask.shape[1] - 1 - distance_from_end


def build_pairwise_trainer_cls(trainer_cls):
    class PairwiseABTrainer(trainer_cls):
        def __init__(self, *args, choice_token_ids: Dict[str, int], **kwargs):
            super().__init__(*args, **kwargs)
            self.choice_token_ids = choice_token_ids
            self._supports_logits_to_keep: bool | None = None

        def _forward_model(self, model: Any, model_inputs: Dict[str, Any]) -> Any:
            if self._supports_logits_to_keep is False:
                return model(**model_inputs)
            try:
                outputs = model(**model_inputs, logits_to_keep=1)
                self._supports_logits_to_keep = True
                return outputs
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                self._supports_logits_to_keep = False
                return model(**model_inputs)

        def _extract_choice_logits(self, model: Any, model_inputs: Dict[str, Any]) -> torch.Tensor:
            outputs = self._forward_model(model, model_inputs)
            logits = outputs.logits
            if logits.ndim != 3:
                raise ValueError(f"Expected logits shaped [batch, seq, vocab], got {logits.shape}")
            if logits.shape[1] == 1:
                next_token_logits = logits[:, 0, :]
            else:
                attention_mask = model_inputs.get("attention_mask")
                if attention_mask is None:
                    next_token_logits = logits[:, -1, :]
                else:
                    last_indices = compute_last_nonpad_indices(attention_mask)
                    batch_indices = torch.arange(logits.shape[0], device=logits.device)
                    next_token_logits = logits[batch_indices, last_indices, :]
            return torch.stack(
                (
                    next_token_logits[:, self.choice_token_ids["A"]],
                    next_token_logits[:, self.choice_token_ids["B"]],
                ),
                dim=-1,
            )

        def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **_: Any):
            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            choice_logits = self._extract_choice_logits(model, model_inputs)
            loss = F.cross_entropy(choice_logits.float(), labels)
            if return_outputs:
                return loss, {"logits": choice_logits}
            return loss

        def prediction_step(
            self,
            model: Any,
            inputs: Dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: list[str] | None = None,
        ):
            del ignore_keys
            inputs = self._prepare_inputs(inputs)
            labels = inputs.get("labels")
            with torch.no_grad(), self.compute_loss_context_manager():
                model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                choice_logits = self._extract_choice_logits(model, model_inputs)
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(choice_logits.float(), labels)
            if prediction_loss_only:
                return (loss.detach() if loss is not None else None, None, None)
            return (
                loss.detach() if loss is not None else None,
                choice_logits.detach(),
                labels.detach() if labels is not None else None,
            )

    return PairwiseABTrainer


def build_runtime_summary(
    args: argparse.Namespace,
    train_dataset: Any,
    eval_dataset: Any,
    preprocess_metrics: Dict[str, Any],
    probe_metrics: Dict[str, Any],
    choice_token_ids: Dict[str, int],
    train_metrics: Dict[str, Any],
    eval_metrics: Dict[str, Any],
    train_duration_sec: float,
) -> Dict[str, Any]:
    return {
        "task": "pairwise_ab_logits",
        "model_name": args.model_name,
        "finetuning_mode": "qlora_4bit" if args.load_in_4bit else "lora_bf16",
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "fps": args.fps,
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "finetune_mlp_modules": bool(args.finetune_mlp_modules),
        "choice_token_ids": choice_token_ids,
        "probe_metrics": probe_metrics,
        "train_duration_sec": round(train_duration_sec, 4),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "preprocess_metrics": preprocess_metrics,
    }


def maybe_probe_label_flip(dataset_row: Dict[str, Any]) -> Dict[str, Any]:
    views_a = int(dataset_row["views_a"])
    views_b = int(dataset_row["views_b"])
    original = "A" if views_a > views_b else "B"
    swapped = "A" if views_b > views_a else "B"
    return {
        "views_a": views_a,
        "views_b": views_b,
        "original_label": original,
        "swapped_label": swapped,
        "swap_inverts_label": original != swapped,
    }


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    wandb_is_enabled = configure_wandb(args)

    ensure_smoke_dataset(args.train_dataset, args.eval_dataset, args.preprocess_metrics)
    preprocess_metrics = load_preprocess_metrics(args.preprocess_metrics)
    resolved_model_name = resolve_model_path(args.model_name)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    from datasets import load_dataset
    from transformers import TrainingArguments
    from transformers.trainer import Trainer
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=resolved_model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
        fast_inference=False,
    )

    chat_template = ensure_chat_template(processor, resolved_model_name)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.finetune_mlp_modules:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",
        target_modules=target_modules,
    )

    if hasattr(processor, "tokenizer"):
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"
    choice_token_ids = get_choice_token_ids(processor.tokenizer)

    dataset_files = {"train": str(args.train_dataset)}
    if args.eval_dataset.exists():
        dataset_files["eval"] = str(args.eval_dataset)
    raw_dataset = load_dataset("json", data_files=dataset_files)
    train_dataset = PairwisePromptDataset(raw_dataset["train"], fps=args.fps)
    eval_dataset = PairwisePromptDataset(raw_dataset["eval"], fps=args.fps) if "eval" in raw_dataset else None

    probe_metrics: Dict[str, Any] = {}
    if args.probe_sample and len(train_dataset) > 0:
        probe_metrics = maybe_probe_sample(processor, train_dataset[0], chat_template=chat_template)
        probe_metrics["choice_token_ids"] = choice_token_ids
        probe_metrics["label_flip_probe"] = maybe_probe_label_flip(train_dataset[0])
        (args.output_dir / "batch_probe.json").write_text(
            json.dumps(probe_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy=determine_eval_strategy(args, eval_dataset is not None),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.0 if args.warmup_steps > 0 else args.warmup_ratio,
        warmup_steps=max(0, args.warmup_steps),
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        report_to="wandb" if wandb_is_enabled else "none",
        run_name=args.wandb_run_name or args.output_dir.name,
        bf16=True,
        tf32=True,
        optim="adamw_8bit",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_pin_memory=False,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    collator = PairwiseABCollator(processor=processor, chat_template=chat_template)
    PairwiseABTrainer = build_pairwise_trainer_cls(Trainer)
    trainer = PairwiseABTrainer(
        choice_token_ids=choice_token_ids,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_pred: compute_choice_metrics(eval_pred.predictions, eval_pred.label_ids),
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_duration_sec = time.perf_counter() - train_start
    train_metrics = dict(train_result.metrics)
    train_metrics["train_runtime_stage1_sec"] = round(train_duration_sec, 4)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics: Dict[str, Any] = {}
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    final_dir = args.output_dir / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(final_dir))

    summary = build_runtime_summary(
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess_metrics=preprocess_metrics,
        probe_metrics=probe_metrics,
        choice_token_ids=choice_token_ids,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        train_duration_sec=train_duration_sec,
    )
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    maybe_wandb_update_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
