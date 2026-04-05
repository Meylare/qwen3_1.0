"""
Realistic GSPO throughput smoke training for Qwen3.5-9B-Base.

This script intentionally keeps the input regime close to the target workload:
two raw videos per sample at 2 fps / 45 seconds, plus full transcript and audio summary.
The smoke simplification only happens in the reward and the dataset size.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import statistics
import subprocess
import threading
import sys
import time
import importlib.util
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import torch

from prompting import build_prompt
from reward_functions import REWARD_FUNCS, get_last_reward_debug_batch


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3.5-9B-Base"
DEFAULT_RAW_DATASET = Path(__file__).resolve().parent / "data" / "raw" / "train_fixed.jsonl"


class SanitizedPromptDataset:
    """Wrap a Hugging Face dataset and strip null multimodal keys on access."""

    def __init__(self, base_dataset: Any, fps: float = 2.0, max_think_tokens: int = 2500):
        self.base_dataset = base_dataset
        self.fps = fps
        self.max_think_tokens = max_think_tokens

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = sanitize_nested(self.base_dataset[index])
        if {
            "video_a",
            "video_b",
            "author_context",
            "transcript_a",
            "transcript_b",
            "audio_summary_a",
            "audio_summary_b",
        } <= row.keys():
            row["prompt"] = build_prompt(row, fps=self.fps, max_think_tokens=self.max_think_tokens)
        return row


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
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--max_completion_length", type=int, default=4000)
    parser.add_argument("--max_think_tokens", type=int, default=2500)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--resource_sample_interval_sec", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)
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
    parser.add_argument("--telegram_bot_token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument("--telegram_chat_id", default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    parser.add_argument("--telegram_progress_every", type=int, default=0)
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_tags", default=os.environ.get("WANDB_TAGS", ""))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wandb_available() -> bool:
    return importlib.util.find_spec("wandb") is not None


def wandb_enabled(args: argparse.Namespace) -> bool:
    return bool((args.wandb_project or "").strip()) and wandb_available()


def configure_wandb(args: argparse.Namespace) -> bool:
    if not wandb_enabled(args):
        return False
    os.environ["WANDB_PROJECT"] = args.wandb_project.strip()
    if (args.wandb_entity or "").strip():
        os.environ["WANDB_ENTITY"] = args.wandb_entity.strip()
    if (args.wandb_tags or "").strip():
        os.environ["WANDB_TAGS"] = args.wandb_tags.strip()
    if (args.wandb_mode or "").strip():
        os.environ["WANDB_MODE"] = args.wandb_mode.strip()
    return True


def maybe_wandb_log(metrics: Dict[str, Any], step: int | None = None) -> None:
    if not wandb_available():
        return
    try:
        import wandb

        if wandb.run is None:
            return
        payload = {key: value for key, value in metrics.items() if value is not None}
        if not payload:
            return
        if step is None:
            wandb.log(payload)
        else:
            wandb.log(payload, step=step)
    except Exception:
        return


def maybe_wandb_update_summary(summary: Dict[str, Any]) -> None:
    if not wandb_available():
        return
    try:
        import wandb

        if wandb.run is None:
            return
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                wandb.run.summary[key] = value
    except Exception:
        return


def sanitize_nested(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: sanitize_nested(nested_value)
            for key, nested_value in value.items()
            if nested_value is not None
        }
        return cleaned
    if isinstance(value, list):
        return [sanitize_nested(item) for item in value]
    return value


def ensure_smoke_dataset(
    train_dataset: Path,
    eval_dataset: Path,
    preprocess_metrics: Path,
) -> None:
    if train_dataset.exists() and eval_dataset.exists() and preprocess_metrics.exists():
        return

    prepare_script = Path(__file__).resolve().parent / "prepare_smoke_dataset.py"
    if not DEFAULT_RAW_DATASET.exists():
        raise FileNotFoundError(
            f"Smoke dataset is missing and raw source dataset was not found: {DEFAULT_RAW_DATASET}"
        )

    train_dataset.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(prepare_script),
        "--input",
        str(DEFAULT_RAW_DATASET),
        "--output_dir",
        str(train_dataset.parent),
    ]
    print(f">>> Smoke dataset missing, preparing it via: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path not found: {model_path}. "
            f"Download the base model into {DEFAULT_MODEL_DIR} or pass --model_name explicitly."
        )
    return str(model_path.resolve())


def ensure_chat_template(processor: Any, model_path: str) -> str | None:
    current_template = getattr(processor, "chat_template", None)
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None

    chat_template = current_template or tokenizer_template
    if not chat_template:
        tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            chat_template = tokenizer_config.get("chat_template")

    if not chat_template:
        return None

    if tokenizer is not None and getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = chat_template
    if getattr(processor, "chat_template", None) is None:
        processor.chat_template = chat_template
    return chat_template


def _patch_instrumentation(GRPOTrainerCls):
    class InstrumentedGRPOTrainer(GRPOTrainerCls):
        def _reset_stage_profile(self) -> None:
            self._current_update_profile = {
                "prepare_inputs_sec": 0.0,
                "rollout_generate_sec": 0.0,
                "reward_scoring_sec": 0.0,
                "compute_loss_sec": 0.0,
                "optimizer_step_sec": 0.0,
                "training_step_sec": 0.0,
            }

        def create_optimizer(self):
            out = super().create_optimizer()
            if getattr(self, "_optimizer_step_wrapped", False):
                return out
            original_step = self.optimizer.step

            def timed_step(*args, **kwargs):
                start = time.perf_counter()
                result = original_step(*args, **kwargs)
                self._current_update_profile["optimizer_step_sec"] = (
                    self._current_update_profile.get("optimizer_step_sec", 0.0)
                    + (time.perf_counter() - start)
                )
                return result

            self.optimizer.step = timed_step
            self._optimizer_step_wrapped = True
            return out

    if hasattr(GRPOTrainerCls, "_generate_and_score_completions"):
        original_generate = getattr(GRPOTrainerCls, "_generate_and_score_completions")

        def _generate_and_score_completions(self, *args, **kwargs):
            if not hasattr(self, "_completion_records"):
                self._completion_records = []
            self._last_reward_capture = None
            current_logs = getattr(self, "_logs", {}) or {}
            start_offsets = {
                "prompt": _log_len(current_logs, "prompt"),
                "completion": _log_len(current_logs, "completion"),
                "advantages": _log_len(current_logs, "advantages"),
                "rewards": {
                    name: _reward_log_len(current_logs, name)
                    for name in sorted(((current_logs.get("rewards", {}) or {}).keys()))
                },
            }
            step_index = int(getattr(getattr(self, "state", None), "global_step", 0) or 0) + 1
            start = time.perf_counter()
            out = original_generate(self, *args, **kwargs)
            self._last_rollout_sec = time.perf_counter() - start
            step_advantages = None
            if isinstance(out, dict) and out.get("advantages") is not None:
                step_advantages = out["advantages"].detach().float().cpu().tolist()
            self._completion_records.extend(
                capture_new_completion_rows(
                    self,
                    step_index,
                    start_offsets,
                    raw_capture=getattr(self, "_last_reward_capture", None),
                    step_advantages=step_advantages,
                )
            )
            return out

        InstrumentedGRPOTrainer._generate_and_score_completions = _generate_and_score_completions

    if hasattr(GRPOTrainerCls, "_generate"):
        original_rollout_generate = getattr(GRPOTrainerCls, "_generate")

        def _generate(self, *args, **kwargs):
            start = time.perf_counter()
            out = original_rollout_generate(self, *args, **kwargs)
            self._current_update_profile["rollout_generate_sec"] = (
                self._current_update_profile.get("rollout_generate_sec", 0.0)
                + (time.perf_counter() - start)
            )
            return out

        InstrumentedGRPOTrainer._generate = _generate

    if hasattr(GRPOTrainerCls, "_calculate_rewards"):
        original_calculate_rewards = getattr(GRPOTrainerCls, "_calculate_rewards")

        def _calculate_rewards(self, *args, **kwargs):
            start = time.perf_counter()
            out = original_calculate_rewards(self, *args, **kwargs)
            self._current_update_profile["reward_scoring_sec"] = (
                self._current_update_profile.get("reward_scoring_sec", 0.0)
                + (time.perf_counter() - start)
            )
            prompts = args[1] if len(args) > 1 else kwargs.get("prompts")
            completions = args[2] if len(args) > 2 else kwargs.get("completions")
            self._last_reward_capture = {
                "prompts": list(prompts or []),
                "completions": list(completions or []),
                "reward_debug_batch": get_last_reward_debug_batch(),
            }
            return out

        InstrumentedGRPOTrainer._calculate_rewards = _calculate_rewards

    if hasattr(GRPOTrainerCls, "_prepare_inputs"):
        original_prepare_inputs = getattr(GRPOTrainerCls, "_prepare_inputs")

        def _prepare_inputs(self, *args, **kwargs):
            start = time.perf_counter()
            out = original_prepare_inputs(self, *args, **kwargs)
            self._current_update_profile["prepare_inputs_sec"] = (
                self._current_update_profile.get("prepare_inputs_sec", 0.0)
                + (time.perf_counter() - start)
            )
            return out

        InstrumentedGRPOTrainer._prepare_inputs = _prepare_inputs

    if hasattr(GRPOTrainerCls, "_compute_loss"):
        original_compute_loss = getattr(GRPOTrainerCls, "_compute_loss")

        def _compute_loss(self, *args, **kwargs):
            start = time.perf_counter()
            out = original_compute_loss(self, *args, **kwargs)
            self._current_update_profile["compute_loss_sec"] = (
                self._current_update_profile.get("compute_loss_sec", 0.0)
                + (time.perf_counter() - start)
            )
            return out

        InstrumentedGRPOTrainer._compute_loss = _compute_loss

    if hasattr(GRPOTrainerCls, "training_step"):
        original_training_step = getattr(GRPOTrainerCls, "training_step")

        def training_step(self, *args, **kwargs):
            self._reset_stage_profile()
            start = time.perf_counter()
            loss = original_training_step(self, *args, **kwargs)
            step_sec = time.perf_counter() - start
            self._current_update_profile["training_step_sec"] = step_sec
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


def maybe_probe_sample(processor, dataset_row: Dict[str, Any], chat_template: str | None = None) -> Dict[str, Any]:
    start = time.perf_counter()
    encoded = processor.apply_chat_template(
        dataset_row["prompt"],
        chat_template=chat_template,
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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log_len(logs: Dict[str, Any], key: str) -> int:
    values = logs.get(key, [])
    return len(values) if values is not None else 0


def _reward_log_len(logs: Dict[str, Any], reward_name: str) -> int:
    reward_logs = logs.get("rewards", {}) or {}
    values = reward_logs.get(reward_name, [])
    return len(values) if values is not None else 0


def capture_new_completion_rows(
    trainer: Any,
    step_index: int,
    start_offsets: Dict[str, Any],
    raw_capture: Dict[str, Any] | None = None,
    step_advantages: List[float] | None = None,
) -> List[Dict[str, Any]]:
    if raw_capture:
        prompts = list(raw_capture.get("prompts", []) or [])
        completions = list(raw_capture.get("completions", []) or [])
        reward_debug_batch = list(raw_capture.get("reward_debug_batch", []) or [])
        total = max(len(prompts), len(completions), len(reward_debug_batch), len(step_advantages or []), 0)
        if total == 0:
            return []

        num_generations = int(getattr(trainer, "num_generations", 1) or 1)
        rows: List[Dict[str, Any]] = []
        for offset in range(total):
            debug = reward_debug_batch[offset] if offset < len(reward_debug_batch) else {}
            row: Dict[str, Any] = {
                "index": len(getattr(trainer, "_completion_records", [])) + len(rows),
                "step_index": step_index,
                "pair_index_in_step": offset // num_generations,
                "generation_index": offset % num_generations,
                "num_generations": num_generations,
                "prompt": prompts[offset] if offset < len(prompts) else None,
                "completion": completions[offset] if offset < len(completions) else None,
                "advantage": step_advantages[offset] if step_advantages and offset < len(step_advantages) else None,
                "reward_accuracy": debug.get("reward_accuracy"),
                "reward_format": debug.get("reward_format"),
                "reward_language": debug.get("reward_language"),
                "reward_reasoning": debug.get("reward_reasoning"),
                "reward_length": debug.get("reward_length"),
                "reward_total": debug.get("reward_total"),
                "ground_truth_winner": debug.get("ground_truth_winner"),
                "parsed_winner": debug.get("parsed_winner"),
                "target_language": debug.get("target_language"),
                "advice_language": debug.get("advice_language"),
                "format_status": debug.get("format_status"),
                "generic_advice_penalty_applied": debug.get("generic_advice_penalty_applied"),
                "near_cap_penalty_applied": debug.get("near_cap_penalty_applied"),
                "unfinished_near_cap_penalty_applied": debug.get("unfinished_near_cap_penalty_applied"),
                "completion_char_length": debug.get("completion_char_length"),
                "reward_max_completion_length": debug.get("max_completion_length"),
            }
            rows.append(row)
        return rows

    
    
    
    logs = getattr(trainer, "_logs", None)
    if not logs:
        return []

    prompts = list(logs.get("prompt", []))
    completions = list(logs.get("completion", []))
    advantages = list(logs.get("advantages", []))
    reward_logs = logs.get("rewards", {}) or {}
    reward_names = sorted(reward_logs.keys())

    prompt_start = int(start_offsets.get("prompt", 0))
    completion_start = int(start_offsets.get("completion", 0))
    advantage_start = int(start_offsets.get("advantages", 0))

    prompt_count = max(0, len(prompts) - prompt_start)
    completion_count = max(0, len(completions) - completion_start)
    advantage_count = max(0, len(advantages) - advantage_start)
    reward_counts = {
        name: max(0, len(reward_logs.get(name, [])) - int((start_offsets.get("rewards", {}) or {}).get(name, 0)))
        for name in reward_names
    }
    total = max(prompt_count, completion_count, advantage_count, *(reward_counts.values() or [0]))
    if total == 0:
        return []

    num_generations = int(getattr(trainer, "num_generations", 1) or 1)
    reward_debug_batch = get_last_reward_debug_batch()
    rows: List[Dict[str, Any]] = []
    for offset in range(total):
        row: Dict[str, Any] = {
            "index": len(getattr(trainer, "_completion_records", [])) + len(rows),
            "step_index": step_index,
            "pair_index_in_step": offset // num_generations,
            "generation_index": offset % num_generations,
            "num_generations": num_generations,
            "prompt": prompts[prompt_start + offset] if prompt_start + offset < len(prompts) else None,
            "completion": completions[completion_start + offset] if completion_start + offset < len(completions) else None,
            "advantage": advantages[advantage_start + offset] if advantage_start + offset < len(advantages) else None,
        }
        reward_total = 0.0
        reward_seen = False
        for name in reward_names:
            values = reward_logs.get(name, [])
            start = int((start_offsets.get("rewards", {}) or {}).get(name, 0))
            value = values[start + offset] if start + offset < len(values) else None
            row[f"reward_{name}"] = value
            if value is not None:
                reward_total += float(value)
                reward_seen = True
        row["reward_total"] = round(reward_total, 6) if reward_seen else None
        if offset < len(reward_debug_batch):
            debug = reward_debug_batch[offset]
            for key in (
                "ground_truth_winner",
                "parsed_winner",
                "target_language",
                "advice_language",
                "format_status",
                "generic_advice_penalty_applied",
            ):
                row[key] = debug.get(key)
        rows.append(row)
    return rows


def extract_completion_rows(trainer: Any) -> List[Dict[str, Any]]:
    captured_rows = list(getattr(trainer, "_completion_records", []) or [])
    if captured_rows:
        return captured_rows

    logs = getattr(trainer, "_logs", None)
    if not logs:
        return []

    prompts = list(logs.get("prompt", []))
    completions = list(logs.get("completion", []))
    advantages = list(logs.get("advantages", []))
    rewards_by_func = logs.get("rewards", {}) or {}
    reward_names = sorted(rewards_by_func.keys())
    total = max(
        len(prompts),
        len(completions),
        len(advantages),
        *(len(rewards_by_func[name]) for name in reward_names),
        0,
    )

    rows: List[Dict[str, Any]] = []
    for idx in range(total):
        row: Dict[str, Any] = {
            "index": idx,
            "prompt": prompts[idx] if idx < len(prompts) else None,
            "completion": completions[idx] if idx < len(completions) else None,
            "advantage": advantages[idx] if idx < len(advantages) else None,
        }
        reward_total = 0.0
        reward_seen = False
        for name in reward_names:
            values = rewards_by_func.get(name, [])
            value = values[idx] if idx < len(values) else None
            row[f"reward_{name}"] = value
            if value is not None:
                reward_total += float(value)
                reward_seen = True
        row["reward_total"] = round(reward_total, 6) if reward_seen else None
        rows.append(row)
    return rows


class StepProfilingCallback:
    def __init__(self, trainer_ref, notifier=None, progress_every: int = 0, wandb_enabled: bool = False):
        self.trainer_ref = trainer_ref
        self._step_started_at: float | None = None
        self.notifier = notifier
        self.progress_every = max(0, int(progress_every))
        self.wandb_enabled = bool(wandb_enabled)

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda args, state, control, **kwargs: control
        raise AttributeError(name)

    def on_step_begin(self, args, state, control, **kwargs):
        trainer = self.trainer_ref()
        if trainer is not None:
            trainer._current_update_profile = {}
        self._step_started_at = time.perf_counter()
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        trainer = self.trainer_ref()
        if trainer is not None:
            trainer._current_update_profile["pre_optimizer_at_sec"] = round(time.perf_counter(), 6)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref()
        if trainer is None or self._step_started_at is None:
            return control

        update_step_sec = time.perf_counter() - self._step_started_at
        accum = dict(getattr(trainer, "_current_update_profile", {}))
        prepare_inputs_sec = float(accum.get("prepare_inputs_sec", 0.0) or 0.0)
        rollout_generate_sec = float(accum.get("rollout_generate_sec", 0.0) or 0.0)
        reward_scoring_sec = float(accum.get("reward_scoring_sec", 0.0) or 0.0)
        compute_loss_sec = float(accum.get("compute_loss_sec", 0.0) or 0.0)
        optimizer_step_sec = float(accum.get("optimizer_step_sec", 0.0) or 0.0)
        backward_sec = max(0.0, float(accum.get("training_step_sec", 0.0) or 0.0) - prepare_inputs_sec - compute_loss_sec)
        prepare_overhead_sec = max(0.0, prepare_inputs_sec - rollout_generate_sec - reward_scoring_sec)
        trainer_overhead_sec = max(
            0.0,
            update_step_sec
            - prepare_inputs_sec
            - compute_loss_sec
            - backward_sec
            - optimizer_step_sec,
        )

        peak_vram_mib = 0.0
        if torch.cuda.is_available():
            peak_vram_mib = torch.cuda.max_memory_reserved() / 1024**2

        trainer._update_step_records.append(
            {
                "step_index": len(trainer._update_step_records) + 1,
                "update_step_sec": round(update_step_sec, 6),
                "prepare_inputs_sec": round(prepare_inputs_sec, 6),
                "rollout_generate_sec": round(rollout_generate_sec, 6),
                "reward_scoring_sec": round(reward_scoring_sec, 6),
                "prepare_overhead_sec": round(prepare_overhead_sec, 6),
                "compute_loss_sec": round(compute_loss_sec, 6),
                "backward_sec": round(backward_sec, 6),
                "optimizer_step_sec": round(optimizer_step_sec, 6),
                "trainer_overhead_sec": round(trainer_overhead_sec, 6),
                "peak_vram_mib": round(float(peak_vram_mib), 2),
            }
        )
        if self.wandb_enabled:
            maybe_wandb_log(
                {
                    "timings/update_step_sec": round(update_step_sec, 6),
                    "timings/prepare_inputs_sec": round(prepare_inputs_sec, 6),
                    "timings/rollout_generate_sec": round(rollout_generate_sec, 6),
                    "timings/reward_scoring_sec": round(reward_scoring_sec, 6),
                    "timings/prepare_overhead_sec": round(prepare_overhead_sec, 6),
                    "timings/compute_loss_sec": round(compute_loss_sec, 6),
                    "timings/backward_sec": round(backward_sec, 6),
                    "timings/optimizer_step_sec": round(optimizer_step_sec, 6),
                    "timings/trainer_overhead_sec": round(trainer_overhead_sec, 6),
                    "system/peak_vram_mib": round(float(peak_vram_mib), 2),
                },
                step=state.global_step,
            )
        if self.notifier is not None and self.progress_every > 0 and state.global_step % self.progress_every == 0:
            self.notifier.send(
                (
                    f"GSPO train progress: step {state.global_step}/{state.max_steps}\n"
                    f"update_step_sec={update_step_sec:.2f}\n"
                    f"rollout_sec={rollout_generate_sec:.2f}\n"
                    f"peak_vram_mib={peak_vram_mib:.1f}"
                )
            )
        return control


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = urlencode(
                {
                    "chat_id": self.chat_id,
                    "text": text[:4000],
                    "disable_web_page_preview": "true",
                }
            ).encode("utf-8")
            req = Request(api_url, data=payload, method="POST")
            with contextlib.closing(urlopen(req, timeout=20)) as resp:
                return 200 <= getattr(resp, "status", 200) < 300
        except Exception:
            return False


def _read_proc_stat() -> Dict[str, List[int]]:
    stats: Dict[str, List[int]] = {}
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            stats[parts[0]] = [int(value) for value in parts[1:]]
    return stats


def _cpu_percentages(prev: Dict[str, List[int]], curr: Dict[str, List[int]]) -> Dict[str, float]:
    percentages: Dict[str, float] = {}
    for key, prev_vals in prev.items():
        curr_vals = curr.get(key)
        if not curr_vals:
            continue
        prev_total = sum(prev_vals)
        curr_total = sum(curr_vals)
        total_delta = curr_total - prev_total
        idle_delta = (curr_vals[3] + curr_vals[4]) - (prev_vals[3] + prev_vals[4])
        if total_delta <= 0:
            percentages[key] = 0.0
            continue
        percentages[key] = round(100.0 * (1.0 - (idle_delta / total_delta)), 2)
    return percentages


def _read_process_rss_mib() -> float:
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                parts = line.split()
                return round(int(parts[1]) / 1024.0, 2)
    return 0.0


def _query_gpu_metrics() -> Dict[str, float | int | None]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = proc.stdout.strip().splitlines()[0]
        util_gpu, util_mem, mem_used, mem_total = [part.strip() for part in line.split(",")]
        return {
            "gpu_util_pct": float(util_gpu),
            "gpu_mem_util_pct": float(util_mem),
            "gpu_mem_used_mib": int(float(mem_used)),
            "gpu_mem_total_mib": int(float(mem_total)),
        }
    except Exception:
        return {
            "gpu_util_pct": None,
            "gpu_mem_util_pct": None,
            "gpu_mem_used_mib": None,
            "gpu_mem_total_mib": None,
        }


class ResourceMonitor:
    def __init__(self, sample_interval_sec: float = 1.0):
        self.sample_interval_sec = max(0.2, float(sample_interval_sec))
        self.rows: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ts: float | None = None
        self.phase: str = "init"

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def start(self) -> None:
        if self._thread is not None:
            return
        self._start_ts = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.sample_interval_sec * 2)
        self._thread = None

    def _run(self) -> None:
        prev_cpu = _read_proc_stat()
        time.sleep(self.sample_interval_sec)
        while not self._stop_event.is_set():
            now = time.perf_counter()
            curr_cpu = _read_proc_stat()
            cpu_pct = _cpu_percentages(prev_cpu, curr_cpu)
            prev_cpu = curr_cpu
            gpu = _query_gpu_metrics()
            row: Dict[str, Any] = {
                "t_sec": round(now - float(self._start_ts or now), 3),
                "phase": self.phase,
                "process_rss_mib": _read_process_rss_mib(),
                "cpu_total_pct": cpu_pct.get("cpu"),
                "gpu_util_pct": gpu["gpu_util_pct"],
                "gpu_mem_util_pct": gpu["gpu_mem_util_pct"],
                "gpu_mem_used_mib": gpu["gpu_mem_used_mib"],
                "gpu_mem_total_mib": gpu["gpu_mem_total_mib"],
            }
            for key, value in cpu_pct.items():
                if key == "cpu":
                    continue
                row[f"{key}_pct"] = value
            self.rows.append(row)
            time.sleep(self.sample_interval_sec)


def summarize_training(
    step_records: List[Dict[str, Any]],
    update_step_records: List[Dict[str, Any]],
    preprocess_metrics: Dict[str, Any],
    args: argparse.Namespace,
    resolved_model_name: str,
    train_duration_sec: float,
) -> Dict[str, Any]:
    step_times = [row["step_sec"] for row in step_records]
    rollout_times = [row["rollout_sec"] for row in step_records if row["rollout_sec"] > 0]
    peak_vram = max((row["peak_vram_mib"] for row in step_records), default=0.0)
    update_step_times = [row["update_step_sec"] for row in update_step_records]

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
        "finetune_mlp_modules": bool(args.finetune_mlp_modules),
        "lora_target_modules": (
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if args.finetune_mlp_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "max_seq_length": args.max_seq_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "repetition_penalty": args.repetition_penalty,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "load_in_4bit": args.load_in_4bit,
        "train_duration_sec": round(train_duration_sec, 4),
        "avg_step_sec": round(avg_step_sec, 4) if avg_step_sec else None,
        "avg_update_step_sec": round(statistics.mean(update_step_times), 4) if update_step_times else None,
        "avg_rollout_sec": round(avg_rollout_sec, 4) if avg_rollout_sec else None,
        "samples_per_hour": round(samples_per_hour, 2) if samples_per_hour else None,
        "peak_vram_mib": round(peak_vram, 2),
        "preprocess_metrics": preprocess_metrics,
        "projections": projections,
    }


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    wandb_is_enabled = configure_wandb(args)
    notifier = TelegramNotifier(args.telegram_bot_token, args.telegram_chat_id)
    os.environ["GSPO_MAX_COMPLETION_LENGTH"] = str(args.max_completion_length)
    stage_timings: Dict[str, float] = {}
    stage_start = time.perf_counter()
    ensure_smoke_dataset(args.train_dataset, args.eval_dataset, args.preprocess_metrics)
    stage_timings["ensure_smoke_dataset_sec"] = round(time.perf_counter() - stage_start, 4)

    monitor = ResourceMonitor(sample_interval_sec=args.resource_sample_interval_sec)
    monitor.start()
    resolved_model_name = resolve_model_path(args.model_name)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    monitor.set_phase("imports")
    stage_start = time.perf_counter()
    from datasets import load_dataset
    from unsloth import FastVisionModel, PatchFastRL

    PatchFastRL("GRPO", FastVisionModel)

    from trl import GRPOConfig, GRPOTrainer
    stage_timings["imports_sec"] = round(time.perf_counter() - stage_start, 4)

    InstrumentedGRPOTrainer = _patch_instrumentation(GRPOTrainer)

    monitor.set_phase("model_load")
    stage_start = time.perf_counter()
    model, processor = FastVisionModel.from_pretrained(
        model_name=resolved_model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
        fast_inference=False,
    )
    stage_timings["model_load_sec"] = round(time.perf_counter() - stage_start, 4)

    monitor.set_phase("chat_template")
    stage_start = time.perf_counter()
    chat_template = ensure_chat_template(processor, resolved_model_name)
    stage_timings["chat_template_sec"] = round(time.perf_counter() - stage_start, 4)

    monitor.set_phase("peft_setup")
    stage_start = time.perf_counter()
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
    stage_timings["peft_setup_sec"] = round(time.perf_counter() - stage_start, 4)

    if hasattr(processor, "tokenizer"):
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

    monitor.set_phase("dataset_load")
    stage_start = time.perf_counter()
    dataset_files = {"train": str(args.train_dataset)}
    if args.eval_dataset.exists():
        dataset_files["eval"] = str(args.eval_dataset)
    dataset = load_dataset("json", data_files=dataset_files)
    train_dataset = SanitizedPromptDataset(dataset["train"], fps=args.fps, max_think_tokens=args.max_think_tokens)
    eval_dataset = SanitizedPromptDataset(dataset["eval"], fps=args.fps, max_think_tokens=args.max_think_tokens) if "eval" in dataset else None
    stage_timings["dataset_load_sec"] = round(time.perf_counter() - stage_start, 4)

    probe_metrics: Dict[str, Any] = {}
    if args.probe_sample and len(train_dataset) > 0:
        monitor.set_phase("probe")
        stage_start = time.perf_counter()
        probe_metrics = maybe_probe_sample(processor, train_dataset[0], chat_template=chat_template)
        (args.output_dir / "batch_probe.json").write_text(
            json.dumps(probe_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stage_timings["probe_sec"] = round(time.perf_counter() - stage_start, 4)

    monitor.set_phase("trainer_setup")
    stage_start = time.perf_counter()
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.0 if args.warmup_steps > 0 else args.warmup_ratio,
        warmup_steps=max(0, args.warmup_steps),
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="wandb" if wandb_is_enabled else "none",
        run_name=args.wandb_run_name or args.output_dir.name,
        bf16=True,
        tf32=True,
        optim="adamw_8bit",
        max_prompt_length=None,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        log_completions=True,
        seed=args.seed,
        importance_sampling_level="sequence",
    )

    trainer = InstrumentedGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer._step_records = []
    trainer._last_rollout_sec = 0.0
    trainer._update_step_records = []
    trainer._current_update_profile = {}
    trainer._optimizer_step_wrapped = False
    trainer._completion_records = []
    trainer.add_callback(
        StepProfilingCallback(
            lambda: trainer,
            notifier=notifier,
            progress_every=args.telegram_progress_every,
            wandb_enabled=wandb_is_enabled,
        )
    )
    stage_timings["trainer_setup_sec"] = round(time.perf_counter() - stage_start, 4)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    monitor.set_phase("train")
    train_start = time.perf_counter()
    if notifier.enabled:
        notifier.send(
            (
                f"GSPO train started\n"
                f"output_dir={args.output_dir}\n"
                f"steps={args.max_steps}\n"
                f"batch_size={args.per_device_train_batch_size}\n"
                f"num_generations={args.num_generations}\n"
                f"repetition_penalty={args.repetition_penalty}\n"
                f"lr_scheduler_type={args.lr_scheduler_type}\n"
                f"warmup_steps={args.warmup_steps}"
            )
        )
    trainer.train()
    train_duration_sec = time.perf_counter() - train_start
    stage_timings["train_sec"] = round(train_duration_sec, 4)

    monitor.set_phase("save")
    stage_start = time.perf_counter()
    final_dir = args.output_dir / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(final_dir))
    stage_timings["save_sec"] = round(time.perf_counter() - stage_start, 4)

    monitor.set_phase("write_outputs")
    stage_start = time.perf_counter()
    step_csv = args.output_dir / "step_metrics.csv"
    write_csv(step_csv, trainer._step_records)
    update_step_csv = args.output_dir / "train_stage_metrics.csv"
    write_csv(update_step_csv, trainer._update_step_records)
    completion_rows = extract_completion_rows(trainer)
    completion_jsonl = args.output_dir / "sample_completions.jsonl"
    write_jsonl(completion_jsonl, completion_rows)

    preprocess_metrics = load_preprocess_metrics(args.preprocess_metrics)
    summary = summarize_training(
        step_records=trainer._step_records,
        update_step_records=trainer._update_step_records,
        preprocess_metrics=preprocess_metrics,
        args=args,
        resolved_model_name=resolved_model_name,
        train_duration_sec=train_duration_sec,
    )
    if trainer._update_step_records:
        last = trainer._update_step_records[-1]
        summary["train_stage_breakdown"] = {
            key: value
            for key, value in last.items()
            if key != "step_index"
        }
    if probe_metrics:
        summary["batch_probe"] = probe_metrics

    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    stage_timings["write_outputs_sec"] = round(time.perf_counter() - stage_start, 4)

    monitor.stop()
    resource_timeline_path = args.output_dir / "resource_timeline.csv"
    write_csv(resource_timeline_path, monitor.rows)
    stage_timings["resource_samples"] = len(monitor.rows)
    stage_timings["resource_sample_interval_sec"] = args.resource_sample_interval_sec
    (args.output_dir / "stage_timings.json").write_text(json.dumps(stage_timings, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["resource_timeline_csv"] = str(resource_timeline_path)
    summary["stage_timings_json"] = str((args.output_dir / "stage_timings.json"))
    summary["train_stage_metrics_csv"] = str(update_step_csv)
    if completion_rows:
        summary["sample_completions_jsonl"] = str(completion_jsonl)
        summary["logged_completions"] = len(completion_rows)
    maybe_wandb_update_summary(summary)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if notifier.enabled:
        notifier.send(
            (
                f"GSPO train finished\n"
                f"output_dir={args.output_dir}\n"
                f"train_duration_sec={summary.get('train_duration_sec')}\n"
                f"avg_step_sec={summary.get('avg_step_sec')}\n"
                f"peak_vram_mib={summary.get('peak_vram_mib')}"
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
