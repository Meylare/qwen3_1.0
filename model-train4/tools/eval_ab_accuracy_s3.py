"""
Evaluate base Qwen3.5 A/B accuracy on S3-backed video pairs without training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOME_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_smoke_dataset import build_whisper_model, extract_video_side_info
from pairwise_ab import get_choice_token_ids
from prompting import build_answer_only_prompt, build_prompt


DEFAULT_MODEL_DIR = HOME_ROOT / "models" / "Qwen3.5-9B-Base"
DEFAULT_INPUT_JSONL = PROJECT_ROOT / "data" / "manifests" / "train_test.jsonl"
DEFAULT_S3_CACHE_DIR = PROJECT_ROOT / "s3_cache"
DEFAULT_SIDEINFO_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "eval_100_base_s3"
DEFAULT_MUSIC_GENRE_MODEL = "dima806/music_genres_classification"

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
DIRECT_ANSWER_RE = re.compile(r"^\s*([AB])\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base Qwen3.5 A/B accuracy on S3 dataset.")
    parser.add_argument("--input_jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--base_model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--s3_cache_dir", type=Path, default=DEFAULT_S3_CACHE_DIR)
    parser.add_argument("--sideinfo_cache_dir", type=Path, default=DEFAULT_SIDEINFO_CACHE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_video_duration", type=float, default=45.0)
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--whisper_device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--speech_segmentation_mode", choices=("off", "inaspeech"), default="inaspeech")
    parser.add_argument("--speech_min_segment_sec", type=float, default=0.35)
    parser.add_argument("--speech_merge_gap_sec", type=float, default=0.25)
    parser.add_argument("--speech_keep_leading_trailing_pad_sec", type=float, default=0.10)
    parser.add_argument("--segmentation_fallback_to_full_audio", action="store_true", default=True)
    parser.add_argument("--no_segmentation_fallback_to_full_audio", action="store_false", dest="segmentation_fallback_to_full_audio")
    parser.add_argument("--music_genre_model", default=DEFAULT_MUSIC_GENRE_MODEL)
    parser.add_argument("--music_genre_device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_think_tokens", type=int, default=2000)
    parser.add_argument("--max_answer_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=2200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--response_mode", choices=("xml_reasoned", "answer_only"), default="xml_reasoned")
    parser.add_argument("--balance_labels", action="store_true")
    parser.add_argument("--balance_seed", type=int, default=0)
    parser.add_argument("--aws_profile", default=None)
    parser.add_argument("--aws_region", default=None)
    parser.add_argument("--telegram_bot_token", default=os.getenv("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--telegram_chat_id", default=os.getenv("TELEGRAM_CHAT_ID"))
    parser.add_argument("--telegram_progress_every", type=int, default=20)
    parser.add_argument("--telegram_timeout_sec", type=float, default=10.0)
    parser.add_argument("--telegram_notify_errors", action="store_true", default=True)
    parser.add_argument("--no_telegram_notify_errors", action="store_false", dest="telegram_notify_errors")
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_ab_accuracy_s3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")
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


def append_system_instruction(messages: List[Dict[str, Any]], instruction: str) -> List[Dict[str, Any]]:
    enhanced = copy.deepcopy(messages)
    if not enhanced:
        return [{"role": "system", "content": [{"type": "text", "text": instruction}]}]

    first = enhanced[0]
    if first.get("role") == "system":
        content = first.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = f"{item.get('text', '').rstrip()}\n\n{instruction}"
                    return enhanced
            content.append({"type": "text", "text": instruction})
            first["content"] = content
            return enhanced
        if isinstance(content, str):
            first["content"] = f"{content.rstrip()}\n\n{instruction}"
            return enhanced

    enhanced.insert(0, {"role": "system", "content": [{"type": "text", "text": instruction}]})
    return enhanced


def add_output_limits(messages: List[Dict[str, Any]], max_think_tokens: int, max_answer_tokens: int) -> List[Dict[str, Any]]:
    instruction = (
        "Additional hard output limits:\n"
        f"- Think inside exactly one <think>...</think> block, capped at {max_think_tokens} tokens.\n"
        f"- Keep the final <answer> block at or below {max_answer_tokens} tokens.\n"
        "- The final output must be exactly: <think>...</think><answer>A</answer> or <think>...</think><answer>B</answer>.\n"
        "- Do not output any extra text before <think> or after </answer>."
    )
    return append_system_instruction(messages, instruction)


def parse_answer(text: str, response_mode: str) -> str | None:
    if response_mode == "answer_only":
        match = DIRECT_ANSWER_RE.match(text.strip())
        if not match:
            return None
        return match.group(1).upper()

    match = ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).upper()


def maybe_swap_pair(raw: Dict[str, Any], pair_index: int, balance_labels: bool, balance_seed: int) -> Tuple[Dict[str, Any], bool]:
    pair = dict(raw)
    if not balance_labels:
        return pair, False

    rng = random.Random(balance_seed + pair_index)
    should_swap = rng.random() < 0.5
    if not should_swap:
        return pair, False

    pair["video_a"], pair["video_b"] = raw["video_b"], raw["video_a"]
    pair["views_a"], pair["views_b"] = raw["views_b"], raw["views_a"]
    return pair, True


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def query_gpu_snapshot() -> Dict[str, float | int | None]:
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


def get_s3_client(aws_profile: str | None, aws_region: str | None):
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for S3 video access. Install it in .venv5.") from exc

    session = boto3.session.Session(profile_name=aws_profile, region_name=aws_region)
    return session.client("s3")


def ensure_local_video_path(
    raw_path: str,
    s3_cache_dir: Path,
    s3_client: Any,
) -> Tuple[Path, bool]:
    if raw_path.startswith("s3://"):
        bucket, key = parse_s3_uri(raw_path)
        target = s3_cache_dir / bucket / key
        if target.exists() and target.stat().st_size > 0:
            return target.resolve(), True
        target.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(target))
        return target.resolve(), False

    local_path = Path(raw_path).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(f"Local video path does not exist: {raw_path}")
    return local_path.resolve(), True


def write_predictions_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "pair_index",
        "video_a",
        "video_b",
        "views_a",
        "views_b",
        "swapped_ab",
        "label",
        "prediction",
        "is_correct",
        "error_stage",
        "error_message",
        "s3_cache_hit_a",
        "s3_cache_hit_b",
        "sideinfo_cache_hit_a",
        "sideinfo_cache_hit_b",
        "context_tokens",
        "generated_tokens",
        "latency_sec",
        "ttft_sec",
        "tokens_per_sec",
        "torch_peak_allocated_mib",
        "torch_peak_reserved_mib",
        "gpu_util_before_pct",
        "gpu_util_after_pct",
        "gpu_mem_util_before_pct",
        "gpu_mem_util_after_pct",
        "gpu_mem_used_before_mib",
        "gpu_mem_used_after_mib",
        "gpu_mem_total_mib",
        "choice_logit_a",
        "choice_logit_b",
        "total_pair_sec",
        "answer_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str | None,
        chat_id: str | None,
        timeout_sec: float,
        logger: logging.Logger,
    ):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.timeout_sec = float(timeout_sec)
        self.logger = logger

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
            with urlopen(req, timeout=self.timeout_sec) as response:
                if response.status != 200:
                    self.logger.warning("Telegram notify failed with status=%s", response.status)
                    return False
            return True
        except Exception as exc:
            self.logger.warning("Telegram notify failed: %s", exc)
            return False


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.output_dir)
    notifier = TelegramNotifier(
        bot_token=args.telegram_bot_token,
        chat_id=args.telegram_chat_id,
        timeout_sec=args.telegram_timeout_sec,
        logger=logger,
    )

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input_jsonl}")
    if args.num_pairs <= 0:
        raise ValueError("--num_pairs must be positive")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")

    resolved_model = resolve_model_path(args.base_model)
    args.s3_cache_dir.mkdir(parents=True, exist_ok=True)
    args.sideinfo_cache_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.input_jsonl)
    selected = rows[args.start_index : args.start_index + args.num_pairs]
    if len(selected) < args.num_pairs:
        raise ValueError(
            f"Requested {args.num_pairs} pairs from index {args.start_index}, "
            f"but only {len(selected)} are available."
        )

    logger.info("Loading models and dependencies")
    if notifier.enabled:
        notifier.send(
            (
                "🚀 eval_ab_accuracy_s3 started\n"
                f"pairs: {args.start_index}..{args.start_index + args.num_pairs - 1} ({args.num_pairs})\n"
                f"input: {args.input_jsonl}\n"
                f"model: {args.base_model}\n"
                f"progress_every: {args.telegram_progress_every}"
            )
        )
    s3_client = get_s3_client(args.aws_profile, args.aws_region)

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    compute_type = "float16" if args.whisper_device.startswith("cuda") else "int8"
    whisper_model = build_whisper_model(args.whisper_model, args.whisper_device, compute_type)

    processor = AutoProcessor.from_pretrained(resolved_model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    chat_template = ensure_chat_template(processor, resolved_model)
    choice_token_ids = get_choice_token_ids(processor.tokenizer) if args.response_mode == "answer_only" else None

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        resolved_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    predictions: List[Dict[str, Any]] = []
    run_start = time.perf_counter()

    for offset, raw in enumerate(selected):
        pair_index = args.start_index + offset
        pair_start = time.perf_counter()

        record: Dict[str, Any] = {
            "pair_index": pair_index,
            "video_a": raw.get("video_a"),
            "video_b": raw.get("video_b"),
            "views_a": raw.get("views_a"),
            "views_b": raw.get("views_b"),
            "swapped_ab": False,
            "label": None,
            "prediction": None,
            "is_correct": None,
            "error_stage": None,
            "error_message": None,
            "s3_cache_hit_a": None,
            "s3_cache_hit_b": None,
            "sideinfo_cache_hit_a": None,
            "sideinfo_cache_hit_b": None,
            "context_tokens": None,
            "generated_tokens": None,
            "latency_sec": None,
            "ttft_sec": None,
            "tokens_per_sec": None,
            "torch_peak_allocated_mib": None,
            "torch_peak_reserved_mib": None,
            "gpu_util_before_pct": None,
            "gpu_util_after_pct": None,
            "gpu_mem_util_before_pct": None,
            "gpu_mem_util_after_pct": None,
            "gpu_mem_used_before_mib": None,
            "gpu_mem_used_after_mib": None,
            "gpu_mem_total_mib": None,
            "choice_logit_a": None,
            "choice_logit_b": None,
            "total_pair_sec": None,
            "answer_text": None,
        }

        pair_raw, was_swapped = maybe_swap_pair(raw, pair_index, args.balance_labels, args.balance_seed)
        record["video_a"] = pair_raw.get("video_a")
        record["video_b"] = pair_raw.get("video_b")
        record["views_a"] = pair_raw.get("views_a")
        record["views_b"] = pair_raw.get("views_b")
        record["swapped_ab"] = was_swapped

        try:
            views_a = int(pair_raw["views_a"])
            views_b = int(pair_raw["views_b"])
            label = "A" if views_a > views_b else "B"
            record["label"] = label
        except Exception as exc:
            record["error_stage"] = "label"
            record["error_message"] = str(exc)
            record["total_pair_sec"] = round(time.perf_counter() - pair_start, 4)
            predictions.append(record)
            logger.error("pair %s failed at label stage: %s", pair_index, exc)
            if args.telegram_notify_errors:
                notifier.send(f"⚠️ eval pair={pair_index} failed at label stage: {exc}")
            processed = len(predictions)
            if notifier.enabled and args.telegram_progress_every > 0 and processed % args.telegram_progress_every == 0:
                valid_so_far = sum(1 for row in predictions if row.get("prediction") in {"A", "B"})
                correct_so_far = sum(1 for row in predictions if row.get("is_correct") is True)
                acc_so_far = (correct_so_far / valid_so_far) if valid_so_far else 0.0
                elapsed_so_far = time.perf_counter() - run_start
                notifier.send(
                    (
                        f"📈 eval progress: {processed}/{args.num_pairs}\n"
                        f"valid: {valid_so_far}, correct: {correct_so_far}, acc_valid: {acc_so_far:.4f}\n"
                        f"elapsed_sec: {elapsed_so_far:.1f}"
                    )
                )
            continue

        try:
            local_a, s3_hit_a = ensure_local_video_path(pair_raw["video_a"], args.s3_cache_dir, s3_client)
            local_b, s3_hit_b = ensure_local_video_path(pair_raw["video_b"], args.s3_cache_dir, s3_client)
            record["s3_cache_hit_a"] = s3_hit_a
            record["s3_cache_hit_b"] = s3_hit_b
        except Exception as exc:
            record["error_stage"] = "s3_download"
            record["error_message"] = str(exc)
            record["total_pair_sec"] = round(time.perf_counter() - pair_start, 4)
            predictions.append(record)
            logger.error("pair %s failed at s3 stage: %s", pair_index, exc)
            if args.telegram_notify_errors:
                notifier.send(f"⚠️ eval pair={pair_index} failed at s3_download: {exc}")
            processed = len(predictions)
            if notifier.enabled and args.telegram_progress_every > 0 and processed % args.telegram_progress_every == 0:
                valid_so_far = sum(1 for row in predictions if row.get("prediction") in {"A", "B"})
                correct_so_far = sum(1 for row in predictions if row.get("is_correct") is True)
                acc_so_far = (correct_so_far / valid_so_far) if valid_so_far else 0.0
                elapsed_so_far = time.perf_counter() - run_start
                notifier.send(
                    (
                        f"📈 eval progress: {processed}/{args.num_pairs}\n"
                        f"valid: {valid_so_far}, correct: {correct_so_far}, acc_valid: {acc_so_far:.4f}\n"
                        f"elapsed_sec: {elapsed_so_far:.1f}"
                    )
                )
            continue

        try:
            side_a = extract_video_side_info(
                video_path=local_a,
                cache_dir=args.sideinfo_cache_dir,
                max_duration=args.max_video_duration,
                whisper_model_name=args.whisper_model,
                whisper_model=whisper_model,
                speech_segmentation_mode=args.speech_segmentation_mode,
                speech_min_segment_sec=args.speech_min_segment_sec,
                speech_merge_gap_sec=args.speech_merge_gap_sec,
                speech_keep_leading_trailing_pad_sec=args.speech_keep_leading_trailing_pad_sec,
                segmentation_fallback_to_full_audio=args.segmentation_fallback_to_full_audio,
                music_genre_model=args.music_genre_model,
                music_genre_device=args.music_genre_device,
            )
            side_b = extract_video_side_info(
                video_path=local_b,
                cache_dir=args.sideinfo_cache_dir,
                max_duration=args.max_video_duration,
                whisper_model_name=args.whisper_model,
                whisper_model=whisper_model,
                speech_segmentation_mode=args.speech_segmentation_mode,
                speech_min_segment_sec=args.speech_min_segment_sec,
                speech_merge_gap_sec=args.speech_merge_gap_sec,
                speech_keep_leading_trailing_pad_sec=args.speech_keep_leading_trailing_pad_sec,
                segmentation_fallback_to_full_audio=args.segmentation_fallback_to_full_audio,
                music_genre_model=args.music_genre_model,
                music_genre_device=args.music_genre_device,
            )
            record["sideinfo_cache_hit_a"] = bool(side_a.get("cache_hit", False))
            record["sideinfo_cache_hit_b"] = bool(side_b.get("cache_hit", False))
        except Exception as exc:
            record["error_stage"] = "sideinfo"
            record["error_message"] = str(exc)
            record["total_pair_sec"] = round(time.perf_counter() - pair_start, 4)
            predictions.append(record)
            logger.error("pair %s failed at sideinfo stage: %s", pair_index, exc)
            if args.telegram_notify_errors:
                notifier.send(f"⚠️ eval pair={pair_index} failed at sideinfo: {exc}")
            processed = len(predictions)
            if notifier.enabled and args.telegram_progress_every > 0 and processed % args.telegram_progress_every == 0:
                valid_so_far = sum(1 for row in predictions if row.get("prediction") in {"A", "B"})
                correct_so_far = sum(1 for row in predictions if row.get("is_correct") is True)
                acc_so_far = (correct_so_far / valid_so_far) if valid_so_far else 0.0
                elapsed_so_far = time.perf_counter() - run_start
                notifier.send(
                    (
                        f"📈 eval progress: {processed}/{args.num_pairs}\n"
                        f"valid: {valid_so_far}, correct: {correct_so_far}, acc_valid: {acc_so_far:.4f}\n"
                        f"elapsed_sec: {elapsed_so_far:.1f}"
                    )
                )
            continue

        prompt_item = {
            "video_a": str(local_a),
            "video_b": str(local_b),
            "author_context": pair_raw.get("author_context", ""),
            "transcript_a": side_a.get("transcript", ""),
            "transcript_b": side_b.get("transcript", ""),
            "lyrics_a": side_a.get("lyrics_transcript", ""),
            "lyrics_b": side_b.get("lyrics_transcript", ""),
            "audio_summary_a": side_a.get("audio_summary", ""),
            "audio_summary_b": side_b.get("audio_summary", ""),
        }
        if args.response_mode == "answer_only":
            prompt = build_answer_only_prompt(prompt_item, fps=args.fps)
        else:
            prompt = build_prompt(prompt_item, fps=args.fps, max_think_tokens=args.max_think_tokens)
            prompt = add_output_limits(prompt, args.max_think_tokens, args.max_answer_tokens)

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            gpu_before = query_gpu_snapshot()
            record["gpu_util_before_pct"] = gpu_before["gpu_util_pct"]
            record["gpu_mem_util_before_pct"] = gpu_before["gpu_mem_util_pct"]
            record["gpu_mem_used_before_mib"] = gpu_before["gpu_mem_used_mib"]
            record["gpu_mem_total_mib"] = gpu_before["gpu_mem_total_mib"]

            encoded = processor.apply_chat_template(
                prompt,
                chat_template=chat_template,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            context_tokens = int(encoded["input_ids"].shape[1])
            model_inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }

            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": min(args.max_new_tokens, 8) if args.response_mode == "answer_only" else args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                generate_kwargs["temperature"] = args.temperature

            if args.response_mode == "answer_only":
                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model(**model_inputs, logits_to_keep=1)
                elapsed = time.perf_counter() - start
                next_token_logits = outputs.logits[:, -1, :][0]
                logit_a = float(next_token_logits[choice_token_ids["A"]].item())
                logit_b = float(next_token_logits[choice_token_ids["B"]].item())
                prediction = "A" if logit_a >= logit_b else "B"
                answer_text = prediction
                generated_tokens = 1
                record["choice_logit_a"] = round(logit_a, 6)
                record["choice_logit_b"] = round(logit_b, 6)
            else:
                start = time.perf_counter()
                generated = model.generate(**model_inputs, **generate_kwargs)
                elapsed = time.perf_counter() - start

                prompt_len = model_inputs["input_ids"].shape[1]
                trimmed = generated[:, prompt_len:]
                answer_text = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                generated_tokens = int(trimmed.shape[1])
                prediction = parse_answer(answer_text, args.response_mode)

            record["prediction"] = prediction
            record["is_correct"] = bool(prediction == label) if prediction is not None else False
            record["context_tokens"] = context_tokens
            record["generated_tokens"] = generated_tokens
            record["latency_sec"] = round(elapsed, 4)
            record["ttft_sec"] = round(elapsed, 4)
            record["tokens_per_sec"] = round(generated_tokens / elapsed, 4) if elapsed > 0 else 0.0
            record["answer_text"] = answer_text
        except Exception as exc:
            record["error_stage"] = "inference"
            record["error_message"] = str(exc)
            logger.error("pair %s failed at inference stage: %s", pair_index, exc)
            if args.telegram_notify_errors:
                notifier.send(f"⚠️ eval pair={pair_index} failed at inference: {exc}")
        finally:
            if torch.cuda.is_available():
                record["torch_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
                record["torch_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
            gpu_after = query_gpu_snapshot()
            record["gpu_util_after_pct"] = gpu_after["gpu_util_pct"]
            record["gpu_mem_util_after_pct"] = gpu_after["gpu_mem_util_pct"]
            record["gpu_mem_used_after_mib"] = gpu_after["gpu_mem_used_mib"]
            if record.get("gpu_mem_total_mib") is None:
                record["gpu_mem_total_mib"] = gpu_after["gpu_mem_total_mib"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        record["total_pair_sec"] = round(time.perf_counter() - pair_start, 4)
        predictions.append(record)

        logger.info(
            "pair=%s label=%s pred=%s correct=%s s3_cache=(%s,%s) side_cache=(%s,%s) context=%s gen=%s ttft=%s latency=%s peak_alloc_mib=%s peak_reserved_mib=%s gpu_util_after=%s gpu_mem_after_mib=%s",
            pair_index,
            record.get("label"),
            record.get("prediction"),
            record.get("is_correct"),
            record.get("s3_cache_hit_a"),
            record.get("s3_cache_hit_b"),
            record.get("sideinfo_cache_hit_a"),
            record.get("sideinfo_cache_hit_b"),
            record.get("context_tokens"),
            record.get("generated_tokens"),
            record.get("ttft_sec"),
            record.get("latency_sec"),
            record.get("torch_peak_allocated_mib"),
            record.get("torch_peak_reserved_mib"),
            record.get("gpu_util_after_pct"),
            record.get("gpu_mem_used_after_mib"),
        )

        processed = len(predictions)
        if notifier.enabled and args.telegram_progress_every > 0 and processed % args.telegram_progress_every == 0:
            valid_so_far = sum(1 for row in predictions if row.get("prediction") in {"A", "B"})
            correct_so_far = sum(1 for row in predictions if row.get("is_correct") is True)
            acc_so_far = (correct_so_far / valid_so_far) if valid_so_far else 0.0
            elapsed_so_far = time.perf_counter() - run_start
            notifier.send(
                (
                    f"📈 eval progress: {processed}/{args.num_pairs}\n"
                    f"valid: {valid_so_far}, correct: {correct_so_far}, acc_valid: {acc_so_far:.4f}\n"
                    f"elapsed_sec: {elapsed_so_far:.1f}"
                )
            )

    total_runtime_sec = time.perf_counter() - run_start
    total = len(predictions)
    valid_predictions = [row for row in predictions if row.get("prediction") in {"A", "B"}]
    valid_count = len(valid_predictions)
    invalid_count = total - valid_count
    correct_count = sum(1 for row in predictions if row.get("is_correct") is True)
    incorrect_count = sum(
        1
        for row in predictions
        if row.get("prediction") in {"A", "B"} and row.get("is_correct") is False
    )

    latency_vals = [float(row["latency_sec"]) for row in predictions if row.get("latency_sec") is not None]
    ttft_vals = [float(row["ttft_sec"]) for row in predictions if row.get("ttft_sec") is not None]
    context_vals = [int(row["context_tokens"]) for row in predictions if row.get("context_tokens") is not None]
    speed_vals = [float(row["tokens_per_sec"]) for row in predictions if row.get("tokens_per_sec") is not None]
    peak_alloc_vals = [float(row["torch_peak_allocated_mib"]) for row in predictions if row.get("torch_peak_allocated_mib") is not None]
    peak_reserved_vals = [float(row["torch_peak_reserved_mib"]) for row in predictions if row.get("torch_peak_reserved_mib") is not None]
    gpu_util_after_vals = [float(row["gpu_util_after_pct"]) for row in predictions if row.get("gpu_util_after_pct") is not None]
    gpu_mem_after_vals = [float(row["gpu_mem_used_after_mib"]) for row in predictions if row.get("gpu_mem_used_after_mib") is not None]

    summary = {
        "input_jsonl": str(args.input_jsonl.resolve()),
        "base_model": resolved_model,
        "num_pairs_requested": args.num_pairs,
        "num_pairs_evaluated": total,
        "start_index": args.start_index,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "valid_predictions": valid_count,
        "invalid_predictions": invalid_count,
        "accuracy_overall": round(correct_count / total, 6) if total > 0 else None,
        "accuracy_valid_only": round(correct_count / valid_count, 6) if valid_count > 0 else None,
        "avg_latency_sec": round(statistics.mean(latency_vals), 4) if latency_vals else None,
        "avg_ttft_sec": round(statistics.mean(ttft_vals), 4) if ttft_vals else None,
        "avg_context_tokens": round(statistics.mean(context_vals), 2) if context_vals else None,
        "avg_tokens_per_sec": round(statistics.mean(speed_vals), 4) if speed_vals else None,
        "max_torch_peak_allocated_mib": round(max(peak_alloc_vals), 2) if peak_alloc_vals else None,
        "max_torch_peak_reserved_mib": round(max(peak_reserved_vals), 2) if peak_reserved_vals else None,
        "avg_gpu_util_after_pct": round(statistics.mean(gpu_util_after_vals), 2) if gpu_util_after_vals else None,
        "max_gpu_util_after_pct": round(max(gpu_util_after_vals), 2) if gpu_util_after_vals else None,
        "max_gpu_mem_used_after_mib": round(max(gpu_mem_after_vals), 2) if gpu_mem_after_vals else None,
        "total_runtime_sec": round(total_runtime_sec, 4),
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "speech_segmentation_mode": args.speech_segmentation_mode,
        "music_genre_model": args.music_genre_model,
        "music_genre_device": args.music_genre_device,
        "speech_min_segment_sec": args.speech_min_segment_sec,
        "speech_merge_gap_sec": args.speech_merge_gap_sec,
        "speech_keep_leading_trailing_pad_sec": args.speech_keep_leading_trailing_pad_sec,
        "segmentation_fallback_to_full_audio": args.segmentation_fallback_to_full_audio,
        "max_video_duration": args.max_video_duration,
        "max_think_tokens": args.max_think_tokens,
        "max_answer_tokens": args.max_answer_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "response_mode": args.response_mode,
        "balance_labels": args.balance_labels,
        "balance_seed": args.balance_seed,
        "s3_cache_dir": str(args.s3_cache_dir.resolve()),
        "sideinfo_cache_dir": str(args.sideinfo_cache_dir.resolve()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    predictions_json_path = args.output_dir / "predictions.json"
    predictions_csv_path = args.output_dir / "predictions.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_json_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    write_predictions_csv(predictions_csv_path, predictions)

    logger.info("Finished. Summary: %s", json.dumps(summary, ensure_ascii=False))
    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved predictions JSON to %s", predictions_json_path)
    logger.info("Saved predictions CSV to %s", predictions_csv_path)
    if notifier.enabled:
        notifier.send(
            (
                "✅ eval_ab_accuracy_s3 finished\n"
                f"pairs: {summary['num_pairs_evaluated']}\n"
                f"accuracy_overall: {summary['accuracy_overall']}\n"
                f"accuracy_valid_only: {summary['accuracy_valid_only']}\n"
                f"valid_predictions: {summary['valid_predictions']}\n"
                f"invalid_predictions: {summary['invalid_predictions']}\n"
                f"total_runtime_sec: {summary['total_runtime_sec']}\n"
                f"summary: {summary_path}"
            )
        )


if __name__ == "__main__":
    main()
