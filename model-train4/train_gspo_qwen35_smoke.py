"""
Realistic GSPO throughput smoke training for Qwen3.5-9B-Base.

This script intentionally keeps the input regime close to the target workload:
two raw videos per sample at 2 fps / 45 seconds, plus full transcript and audio summary.
The smoke simplification only happens in the reward and the dataset size.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from reward_functions import smoke_reward_func


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3.5-9B-Base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSPO smoke training on Qwen3.5-9B-Base.")
    parser.add_argument("--model_name", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--train_dataset", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    parser.add_argument("--eval_dataset", type=Path, default=DEFAULT_DATA_DIR / "eval.jsonl")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "output" / "qwen35_9b_base_gspo_smoke")
    parser.add_argument("--preprocess_metrics", type=Path, default=DEFAULT_DATA_DIR / "preprocess_metrics.json")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--max_completion_length", type=int, default=768)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--probe_sample", action="store_true", default=True)
    parser.add_argument("--no_probe_sample", action="store_false", dest="probe_sample")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path not found: {model_path}. "
            f"Download the base model into {DEFAULT_MODEL_DIR} or pass --model_name explicitly."
        )
    return str(model_path.resolve())


def _patch_instrumentation(GRPOTrainerCls):
    class InstrumentedGRPOTrainer(GRPOTrainerCls):
        pass

    if hasattr(GRPOTrainerCls, "_generate_and_score_completions"):
        original_generate = getattr(GRPOTrainerCls, "_generate_and_score_completions")

        def _generate_and_score_completions(self, *args, **kwargs):
            start = time.perf_counter()
            out = original_generate(self, *args, **kwargs)
            self._last_rollout_sec = time.perf_counter() - start
            return out

        InstrumentedGRPOTrainer._generate_and_score_completions = _generate_and_score_completions

    if hasattr(GRPOTrainerCls, "training_step"):
        original_training_step = getattr(GRPOTrainerCls, "training_step")

        def training_step(self, *args, **kwargs):
            start = time.perf_counter()
            loss = original_training_step(self, *args, **kwargs)
            step_sec = time.perf_counter() - start
            peak_vram_mib = 0.0
            if torch.cuda.is_available():
                peak_vram_mib = torch.cuda.max_memory_reserved() / 1024**2
            self._step_records.append(
                {
                    "step_index": len(self._step_records) + 1,
                    "step_sec": round(step_sec, 6),
                    "rollout_sec": round(float(getattr(self, "_last_rollout_sec", 0.0) or 0.0), 6),
                    "peak_vram_mib": round(float(peak_vram_mib), 2),
                }
            )
            self._last_rollout_sec = 0.0
            return loss

        InstrumentedGRPOTrainer.training_step = training_step

    return InstrumentedGRPOTrainer


def maybe_probe_sample(processor, dataset_row: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    encoded = processor.apply_chat_template(
        dataset_row["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    probe = {
        "probe_sec": round(time.perf_counter() - start, 4),
        "input_ids_shape": list(encoded["input_ids"].shape),
        "attention_mask_shape": list(encoded["attention_mask"].shape),
    }
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
        if key in encoded:
            probe[f"{key}_shape"] = list(encoded[key].shape)
    return probe


def load_preprocess_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_training(
    step_records: List[Dict[str, Any]],
    preprocess_metrics: Dict[str, Any],
    args: argparse.Namespace,
    resolved_model_name: str,
    train_duration_sec: float,
) -> Dict[str, Any]:
    step_times = [row["step_sec"] for row in step_records]
    rollout_times = [row["rollout_sec"] for row in step_records if row["rollout_sec"] > 0]
    peak_vram = max((row["peak_vram_mib"] for row in step_records), default=0.0)

    avg_step_sec = statistics.mean(step_times) if step_times else 0.0
    avg_rollout_sec = statistics.mean(rollout_times) if rollout_times else 0.0
    samples_per_hour = (3600.0 / avg_step_sec) if avg_step_sec > 0 else 0.0

    projections: Dict[str, Any] = {}
    avg_pair_preprocess_sec = float(preprocess_metrics.get("avg_pair_preprocess_sec", 0.0) or 0.0)
    for n_pairs in (500, 1000, 7000):
        training_hours = (n_pairs * avg_step_sec) / 3600.0 if avg_step_sec > 0 else None
        preprocessing_hours = (n_pairs * avg_pair_preprocess_sec) / 3600.0 if avg_pair_preprocess_sec > 0 else None
        total_hours = None
        if training_hours is not None:
            total_hours = training_hours + (preprocessing_hours or 0.0)
        projections[str(n_pairs)] = {
            "training_hours": round(training_hours, 4) if training_hours is not None else None,
            "preprocessing_hours": round(preprocessing_hours, 4) if preprocessing_hours is not None else None,
            "combined_hours": round(total_hours, 4) if total_hours is not None else None,
        }

    return {
        "model_name": resolved_model_name,
        "finetuning_mode": "qlora_4bit" if args.load_in_4bit else "lora_bf16",
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "load_in_4bit": args.load_in_4bit,
        "train_duration_sec": round(train_duration_sec, 4),
        "avg_step_sec": round(avg_step_sec, 4) if avg_step_sec else None,
        "avg_rollout_sec": round(avg_rollout_sec, 4) if avg_rollout_sec else None,
        "samples_per_hour": round(samples_per_hour, 2) if samples_per_hour else None,
        "peak_vram_mib": round(peak_vram, 2),
        "preprocess_metrics": preprocess_metrics,
        "projections": projections,
    }


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    resolved_model_name = resolve_model_path(args.model_name)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    from datasets import load_dataset
    from unsloth import FastVisionModel, PatchFastRL

    PatchFastRL("GRPO", FastVisionModel)

    from trl import GRPOConfig, GRPOTrainer

    InstrumentedGRPOTrainer = _patch_instrumentation(GRPOTrainer)

    model, processor = FastVisionModel.from_pretrained(
        model_name=resolved_model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
        fast_inference=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    if hasattr(processor, "tokenizer"):
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

    dataset_files = {"train": str(args.train_dataset)}
    if args.eval_dataset.exists():
        dataset_files["eval"] = str(args.eval_dataset)
    dataset = load_dataset("json", data_files=dataset_files)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("eval")

    probe_metrics: Dict[str, Any] = {}
    if args.probe_sample and len(train_dataset) > 0:
        probe_metrics = maybe_probe_sample(processor, train_dataset[0])
        (args.output_dir / "batch_probe.json").write_text(
            json.dumps(probe_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        bf16=True,
        tf32=True,
        optim="adamw_8bit",
        max_prompt_length=None,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        log_completions=True,
        seed=args.seed,
        importance_sampling_level="sequence",
    )

    trainer = InstrumentedGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=smoke_reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer._step_records = []
    trainer._last_rollout_sec = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_start = time.perf_counter()
    trainer.train()
    train_duration_sec = time.perf_counter() - train_start

    final_dir = args.output_dir / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(final_dir))

    step_csv = args.output_dir / "step_metrics.csv"
    write_csv(step_csv, trainer._step_records)

    preprocess_metrics = load_preprocess_metrics(args.preprocess_metrics)
    summary = summarize_training(
        step_records=trainer._step_records,
        preprocess_metrics=preprocess_metrics,
        args=args,
        resolved_model_name=resolved_model_name,
        train_duration_sec=train_duration_sec,
    )
    if probe_metrics:
        summary["batch_probe"] = probe_metrics

    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
