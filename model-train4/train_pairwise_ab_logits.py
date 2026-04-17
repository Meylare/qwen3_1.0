"""
Supervised pairwise A/B training from direct choice logits for Qwen3.5-9B-Base.
"""

from __future__ import annotations

import argparse
import csv
import math
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from pairwise_ab import compute_choice_metrics, get_choice_token_ids, label_to_class_id
from prompting import build_answer_only_prompt
from train_gspo_qwen35_smoke import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    TelegramNotifier,
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

    def __init__(self, base_dataset: Any, fps: float = 2.0, split_name: str = "train"):
        self.base_dataset = base_dataset
        self.fps = fps
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = sanitize_nested(self.base_dataset[index])
        if "label" not in row and {"views_a", "views_b"} <= row.keys():
            row["label"] = "A" if row["views_a"] > row["views_b"] else "B"
        if {"video_a", "video_b"} <= row.keys():
            row["prompt"] = build_answer_only_prompt(row, fps=self.fps)
        row["sample_id"] = f"{self.split_name}:{index}"
        return row


class PairwiseABCollator:
    def __init__(
        self,
        processor: Any,
        chat_template: str | None = None,
        cache_processed_samples: bool = True,
    ):
        self.processor = processor
        self.chat_template = chat_template
        self.cache_processed_samples = bool(cache_processed_samples)
        self._cached_samples: Dict[str, Dict[str, Any]] = {}
        self._pad_token_id = 0
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            self._pad_token_id = int(tokenizer.pad_token_id)

    def _encode_single_feature(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        encoded = self.processor.apply_chat_template(
            [feature["prompt"]],
            chat_template=self.chat_template,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        sample: Dict[str, Any] = {
            "input_ids": encoded["input_ids"][0].detach().cpu().clone(),
            "attention_mask": encoded["attention_mask"][0].detach().cpu().clone(),
            "labels": torch.tensor(label_to_class_id(feature["label"]), dtype=torch.long),
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            value = encoded.get(key)
            if value is not None:
                sample[key] = value.detach().cpu().clone()
        return sample

    def _get_cached_sample(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(feature.get("sample_id", "__uncached__"))
        cached = self._cached_samples.get(sample_id)
        if cached is None:
            cached = self._encode_single_feature(feature)
            if sample_id != "__uncached__":
                self._cached_samples[sample_id] = cached
        return {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in cached.items()
        }

    def _collate_cached_features(self, features: list[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [self._get_cached_sample(feature) for feature in features]
        batch: Dict[str, Any] = {
            "input_ids": pad_sequence(
                [sample["input_ids"] for sample in samples],
                batch_first=True,
                padding_value=self._pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [sample["attention_mask"] for sample in samples],
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.stack([sample["labels"] for sample in samples]),
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [sample[key] for sample in samples if key in sample]
            if not values:
                continue
            if len(values) != len(samples):
                raise ValueError(f"Inconsistent cached batch for key {key!r}")
            batch[key] = torch.cat(values, dim=0)
        return batch

    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, Any]:
        if self.cache_processed_samples:
            return self._collate_cached_features(features)
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
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--training_backend", choices=["manual", "trainer"], default="manual")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--auto_expand_max_seq_length", action="store_true", default=True)
    parser.add_argument("--no_auto_expand_max_seq_length", action="store_false", dest="auto_expand_max_seq_length")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing_mode", choices=["hf", "unsloth", "off"], default="unsloth")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--disable_unsloth_compile", action="store_true", default=True)
    parser.add_argument("--enable_unsloth_compile", action="store_false", dest="disable_unsloth_compile")
    parser.add_argument("--finetune_mlp_modules", action="store_true", default=False)
    parser.add_argument("--no_finetune_mlp_modules", action="store_false", dest="finetune_mlp_modules")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--probe_sample", action="store_true", default=True)
    parser.add_argument("--no_probe_sample", action="store_false", dest="probe_sample")
    parser.add_argument("--cache_processed_samples", action="store_true", default=True)
    parser.add_argument("--no_cache_processed_samples", action="store_false", dest="cache_processed_samples")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--telegram_bot_token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument("--telegram_chat_id", default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    parser.add_argument("--telegram_progress_every", type=int, default=0)
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
            # Unsloth patches Trainer.get_batch_samples assuming token-level labels.
            # Our stage-1 objective uses scalar class labels (A/B), so we opt out.
            self.model_accepts_loss_kwargs = False

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            del device
            batch_samples = []
            for _ in range(num_batches):
                try:
                    batch_samples.append(next(epoch_iterator))
                except StopIteration:
                    break
            return batch_samples, None

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
    train_stage_breakdown: Dict[str, Any] | None = None,
    train_stage_metrics_csv: str | None = None,
) -> Dict[str, Any]:
    summary = {
        "task": "pairwise_ab_logits",
        "model_name": args.model_name,
        "finetuning_mode": "qlora_4bit" if args.load_in_4bit else "lora_bf16",
        "training_backend": args.training_backend,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "fps": args.fps,
        "max_seq_length": args.max_seq_length,
        "auto_expand_max_seq_length": bool(args.auto_expand_max_seq_length),
        "gradient_checkpointing_mode": args.gradient_checkpointing_mode,
        "attn_implementation": args.attn_implementation,
        "disable_unsloth_compile": bool(args.disable_unsloth_compile),
        "cache_processed_samples": bool(args.cache_processed_samples),
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "telegram_progress_every": int(args.telegram_progress_every),
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
    if train_stage_breakdown:
        summary["train_stage_breakdown"] = train_stage_breakdown
    if train_stage_metrics_csv:
        summary["train_stage_metrics_csv"] = train_stage_metrics_csv
    return summary


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


def probe_collated_batch(collator: PairwiseABCollator, dataset_row: Dict[str, Any]) -> Dict[str, Any]:
    batch = collator([dataset_row])
    probe: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            probe[f"{key}_shape"] = list(value.shape)
    if "input_ids_shape" in probe:
        probe["effective_seq_length"] = probe["input_ids_shape"][-1]
    return probe


def round_up_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def resolve_gradient_checkpointing_mode(mode: str) -> bool | str:
    if mode == "off":
        return False
    if mode == "unsloth":
        return "unsloth"
    return True


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: (value.to(device, non_blocking=True) if hasattr(value, "to") else value)
        for key, value in batch.items()
    }


def build_autocast_context() -> Any:
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_choice_logits(
    model: Any,
    model_inputs: Dict[str, Any],
    choice_token_ids: Dict[str, int],
    supports_logits_to_keep: bool | None,
) -> tuple[torch.Tensor, bool]:
    if supports_logits_to_keep is False:
        outputs = model(**model_inputs)
    else:
        try:
            outputs = model(**model_inputs, logits_to_keep=1)
            supports_logits_to_keep = True
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            outputs = model(**model_inputs)
            supports_logits_to_keep = False

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
    choice_logits = torch.stack(
        (
            next_token_logits[:, choice_token_ids["A"]],
            next_token_logits[:, choice_token_ids["B"]],
        ),
        dim=-1,
    )
    return choice_logits, bool(supports_logits_to_keep)


def compute_total_training_steps(args: argparse.Namespace, train_loader_len: int) -> int:
    if train_loader_len <= 0:
        return 0
    updates_per_epoch = math.ceil(train_loader_len / max(1, args.gradient_accumulation_steps))
    if args.max_steps > 0:
        return args.max_steps
    return max(1, math.ceil(args.num_train_epochs * updates_per_epoch))


def save_metrics_file(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def save_step_profiles_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_step_profiles(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    mean_fields = [
        "update_step_sec",
        "batch_fetch_sec",
        "move_batch_to_device_sec",
        "forward_loss_sec",
        "backward_sec",
        "optimizer_step_sec",
        "peak_vram_mib",
        "input_ids_tokens",
        "video_rows",
    ]
    summary = {
        field: round(sum(float(row.get(field, 0.0) or 0.0) for row in rows) / len(rows), 6)
        for field in mean_fields
    }
    summary["peak_vram_mib_max"] = round(max(float(row.get("peak_vram_mib", 0.0) or 0.0) for row in rows), 6)
    return summary


def run_manual_evaluation(
    model: Any,
    eval_dataset: Any,
    collator: PairwiseABCollator,
    args: argparse.Namespace,
    choice_token_ids: Dict[str, int],
    supports_logits_to_keep: bool | None,
) -> tuple[Dict[str, Any], bool | None]:
    if eval_dataset is None or len(eval_dataset) == 0:
        return {}, supports_logits_to_keep

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    model.eval()
    losses: list[float] = []
    logits_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = move_batch_to_device(batch, model.device)
            labels = batch.pop("labels")
            with build_autocast_context():
                choice_logits, supports_logits_to_keep = forward_choice_logits(
                    model=model,
                    model_inputs=batch,
                    choice_token_ids=choice_token_ids,
                    supports_logits_to_keep=supports_logits_to_keep,
                )
                loss = F.cross_entropy(choice_logits.float(), labels)
            losses.append(float(loss.detach().item()))
            logits_rows.append(choice_logits.detach().float().cpu())
            label_rows.append(labels.detach().cpu())

    model.train()
    logits_np = torch.cat(logits_rows, dim=0).numpy() if logits_rows else []
    labels_np = torch.cat(label_rows, dim=0).numpy() if label_rows else []
    metrics = compute_choice_metrics(logits_np, labels_np)
    if losses:
        metrics["eval_loss"] = round(sum(losses) / len(losses), 6)
    return metrics, supports_logits_to_keep


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    wandb_is_enabled = configure_wandb(args)
    notifier = TelegramNotifier(args.telegram_bot_token, args.telegram_chat_id)

    ensure_smoke_dataset(args.train_dataset, args.eval_dataset, args.preprocess_metrics)
    preprocess_metrics = load_preprocess_metrics(args.preprocess_metrics)
    resolved_model_name = resolve_model_path(args.model_name)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    if args.disable_unsloth_compile:
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    else:
        os.environ.pop("UNSLOTH_COMPILE_DISABLE", None)

    from datasets import load_dataset
    from unsloth import FastVisionModel
    from transformers import AutoProcessor, TrainingArguments, get_scheduler
    from transformers.trainer import Trainer

    dataset_files = {"train": str(args.train_dataset)}
    if args.eval_dataset.exists():
        dataset_files["eval"] = str(args.eval_dataset)
    raw_dataset = load_dataset("json", data_files=dataset_files)
    train_dataset = PairwisePromptDataset(raw_dataset["train"], fps=args.fps, split_name="train")
    eval_dataset = PairwisePromptDataset(raw_dataset["eval"], fps=args.fps, split_name="eval") if "eval" in raw_dataset else None

    probe_processor = AutoProcessor.from_pretrained(resolved_model_name, trust_remote_code=True)
    probe_chat_template = ensure_chat_template(probe_processor, resolved_model_name)
    requested_max_seq_length = args.max_seq_length
    effective_max_seq_length = args.max_seq_length
    pre_model_probe: Dict[str, Any] = {}
    if args.auto_expand_max_seq_length and len(train_dataset) > 0:
        pre_model_probe = maybe_probe_sample(probe_processor, train_dataset[0], chat_template=probe_chat_template)
        observed_length = int(pre_model_probe.get("input_ids_shape", [0, requested_max_seq_length])[-1])
        effective_max_seq_length = max(requested_max_seq_length, round_up_to_multiple(observed_length, 256))
    args.max_seq_length = effective_max_seq_length

    model_load_kwargs = dict(
        model_name=resolved_model_name,
        max_seq_length=effective_max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
        fast_inference=False,
        fullgraph=False,
    )
    if args.attn_implementation:
        model_load_kwargs["attn_implementation"] = args.attn_implementation
    model, processor = FastVisionModel.from_pretrained(**model_load_kwargs)

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
        use_gradient_checkpointing=resolve_gradient_checkpointing_mode(args.gradient_checkpointing_mode),
        target_modules=target_modules,
    )

    if hasattr(processor, "tokenizer"):
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"
    choice_token_ids = get_choice_token_ids(processor.tokenizer)
    collator = PairwiseABCollator(
        processor=processor,
        chat_template=chat_template,
        cache_processed_samples=args.cache_processed_samples,
    )

    probe_metrics: Dict[str, Any] = {}
    if args.probe_sample and len(train_dataset) > 0:
        probe_metrics = maybe_probe_sample(processor, train_dataset[0], chat_template=chat_template)
        probe_metrics["max_seq_length_resolution"] = {
            "requested_max_seq_length": requested_max_seq_length,
            "effective_max_seq_length": effective_max_seq_length,
            "auto_expanded": effective_max_seq_length != requested_max_seq_length,
        }
        if pre_model_probe:
            probe_metrics["pre_model_probe"] = pre_model_probe
        probe_metrics["collated_batch_probe"] = probe_collated_batch(collator, train_dataset[0])
        probe_metrics["choice_token_ids"] = choice_token_ids
        probe_metrics["label_flip_probe"] = maybe_probe_label_flip(train_dataset[0])
        (args.output_dir / "batch_probe.json").write_text(
            json.dumps(probe_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    train_metrics: Dict[str, Any] = {}
    eval_metrics: Dict[str, Any] = {}
    train_stage_profiles: list[Dict[str, Any]] = []
    train_stage_breakdown: Dict[str, Any] = {}
    train_stage_metrics_csv: str | None = None

    if args.training_backend == "manual":
        if args.resume_from_checkpoint:
            raise ValueError("resume_from_checkpoint is not supported with training_backend=manual.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            num_workers=args.dataloader_num_workers,
            persistent_workers=args.dataloader_num_workers > 0,
        )
        total_training_steps = compute_total_training_steps(args, len(train_loader))
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=max(0, args.warmup_steps),
            num_training_steps=max(1, total_training_steps),
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        supports_logits_to_keep: bool | None = None
        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        update_losses: list[float] = []
        model.train()
        train_start = time.perf_counter()
        epoch_index = 0
        update_profile: Dict[str, Any] | None = None
        update_window_start = 0.0
        if notifier.enabled:
            notifier.send(
                (
                    f"Pairwise AB train started\n"
                    f"output_dir={args.output_dir}\n"
                    f"steps={total_training_steps}\n"
                    f"train_rows={len(train_dataset)}\n"
                    f"eval_rows={len(eval_dataset) if eval_dataset is not None else 0}\n"
                    f"progress_every={args.telegram_progress_every}"
                )
            )

        while global_step < total_training_steps:
            epoch_index += 1
            epoch_iterator = iter(train_loader)
            batch_index = 0
            while global_step < total_training_steps:
                fetch_start = time.perf_counter()
                try:
                    batch = next(epoch_iterator)
                except StopIteration:
                    break
                batch_fetch_sec = time.perf_counter() - fetch_start
                batch_index += 1
                if update_profile is None:
                    update_window_start = time.perf_counter()
                    update_profile = {
                        "step_index": global_step + 1,
                        "epoch_index": epoch_index,
                        "micro_batches": 0,
                        "batch_fetch_sec": 0.0,
                        "move_batch_to_device_sec": 0.0,
                        "forward_loss_sec": 0.0,
                        "backward_sec": 0.0,
                        "optimizer_step_sec": 0.0,
                        "input_ids_tokens": 0,
                        "video_rows": 0,
                    }
                update_profile["micro_batches"] += 1
                update_profile["batch_fetch_sec"] += batch_fetch_sec
                input_ids = batch.get("input_ids")
                if hasattr(input_ids, "numel"):
                    update_profile["input_ids_tokens"] += int(input_ids.numel())
                pixel_values_videos = batch.get("pixel_values_videos")
                if hasattr(pixel_values_videos, "shape") and len(pixel_values_videos.shape) > 0:
                    update_profile["video_rows"] += int(pixel_values_videos.shape[0])

                move_start = time.perf_counter()
                batch = move_batch_to_device(batch, model.device)
                update_profile["move_batch_to_device_sec"] += time.perf_counter() - move_start
                labels = batch.pop("labels")
                forward_start = time.perf_counter()
                with build_autocast_context():
                    choice_logits, supports_logits_to_keep = forward_choice_logits(
                        model=model,
                        model_inputs=batch,
                        choice_token_ids=choice_token_ids,
                        supports_logits_to_keep=supports_logits_to_keep,
                    )
                    loss = F.cross_entropy(choice_logits.float(), labels)
                    scaled_loss = loss / max(1, args.gradient_accumulation_steps)
                update_profile["forward_loss_sec"] += time.perf_counter() - forward_start

                backward_start = time.perf_counter()
                scaled_loss.backward()
                update_profile["backward_sec"] += time.perf_counter() - backward_start
                accumulated_loss += float(loss.detach().item())
                is_update_step = batch_index % max(1, args.gradient_accumulation_steps) == 0
                is_last_batch = batch_index == len(train_loader)
                if not (is_update_step or is_last_batch):
                    continue

                optimizer_start = time.perf_counter()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_profile["optimizer_step_sec"] += time.perf_counter() - optimizer_start

                global_step += 1
                mean_update_loss = accumulated_loss / max(1, args.gradient_accumulation_steps)
                update_losses.append(mean_update_loss)
                peak_vram_mib = 0.0
                if torch.cuda.is_available():
                    peak_vram_mib = torch.cuda.max_memory_reserved() / 1024**2
                update_profile["update_step_sec"] = time.perf_counter() - update_window_start
                update_profile["peak_vram_mib"] = round(float(peak_vram_mib), 2)
                profile_row = {
                    key: (
                        round(value, 6)
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in update_profile.items()
                }
                train_stage_profiles.append(profile_row)
                if global_step % args.logging_steps == 0:
                    log_payload = {
                        "step": global_step,
                        "loss": round(mean_update_loss, 6),
                        "lr": round(float(scheduler.get_last_lr()[0]), 10),
                        "batch_fetch_sec": profile_row["batch_fetch_sec"],
                        "forward_loss_sec": profile_row["forward_loss_sec"],
                        "backward_sec": profile_row["backward_sec"],
                        "optimizer_step_sec": profile_row["optimizer_step_sec"],
                    }
                    print(json.dumps(log_payload, ensure_ascii=False), flush=True)
                if (
                    notifier.enabled
                    and args.telegram_progress_every > 0
                    and global_step % args.telegram_progress_every == 0
                ):
                    elapsed_sec = time.perf_counter() - train_start
                    peak_vram_mib = 0.0
                    if torch.cuda.is_available():
                        peak_vram_mib = torch.cuda.max_memory_reserved() / 1024**2
                    notifier.send(
                        (
                            f"Pairwise AB progress: step {global_step}/{total_training_steps}\n"
                            f"loss={mean_update_loss:.6f}\n"
                            f"elapsed_sec={elapsed_sec:.2f}\n"
                            f"peak_vram_mib={peak_vram_mib:.1f}"
                        )
                    )
                accumulated_loss = 0.0
                update_profile = None
                if global_step >= total_training_steps:
                    break

        train_duration_sec = time.perf_counter() - train_start
        peak_vram_mib = 0.0
        if torch.cuda.is_available():
            peak_vram_mib = torch.cuda.max_memory_reserved() / 1024**2
        train_metrics = {
            "global_step": global_step,
            "train_loss": round(sum(update_losses) / len(update_losses), 6) if update_losses else 0.0,
            "train_runtime": round(train_duration_sec, 4),
            "train_runtime_stage1_sec": round(train_duration_sec, 4),
            "train_steps_per_second": round(global_step / train_duration_sec, 6) if train_duration_sec > 0 else 0.0,
            "peak_vram_mib": round(float(peak_vram_mib), 2),
        }
        train_stage_metrics_csv = str(args.output_dir / "train_stage_metrics.csv")
        save_step_profiles_csv(Path(train_stage_metrics_csv), train_stage_profiles)
        train_stage_breakdown = summarize_step_profiles(train_stage_profiles)
        save_metrics_file(args.output_dir / "train_results.json", train_metrics)
        save_metrics_file(
            args.output_dir / "trainer_state.json",
            {
                "global_step": global_step,
                "max_steps": total_training_steps,
                "training_backend": args.training_backend,
            },
        )
        eval_metrics, _ = run_manual_evaluation(
            model=model,
            eval_dataset=eval_dataset,
            collator=collator,
            args=args,
            choice_token_ids=choice_token_ids,
            supports_logits_to_keep=supports_logits_to_keep,
        )
        if eval_metrics:
            save_metrics_file(args.output_dir / "eval_results.json", eval_metrics)
        if notifier.enabled:
            notifier.send(
                (
                    f"Pairwise AB train finished\n"
                    f"output_dir={args.output_dir}\n"
                    f"steps={global_step}\n"
                    f"train_loss={train_metrics['train_loss']:.6f}\n"
                    f"eval_accuracy={float(eval_metrics.get('accuracy', 0.0)):.6f}"
                )
            )
    else:
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

        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

    final_dir = args.output_dir / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
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
        train_stage_breakdown=train_stage_breakdown,
        train_stage_metrics_csv=train_stage_metrics_csv,
    )
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    maybe_wandb_update_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
