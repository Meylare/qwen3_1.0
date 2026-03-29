"""
Build a realistic pairwise GSPO smoke dataset with cached transcript/audio features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np

from prompting import build_prompt, deterministic_flip


TRANSCRIPT_VERSION = "v1"
AUDIO_SUMMARY_VERSION = "v1"
DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "train.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ffmpeg_extract_audio(video_path: Path, wav_path: Path, max_duration: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-t",
        str(max_duration),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {proc.stderr.strip()}")


def format_transcript(segments: Iterable[Any]) -> str:
    parts: List[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        parts.append(f"[{seg.start:.2f}-{seg.end:.2f}] {text}")
    return "\n".join(parts)


def describe_level(value: float, low: float, high: float, labels: Tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value < high:
        return labels[1]
    return labels[2]


def build_audio_summary(
    wav_path: Path,
    transcript_text: str,
    transcript_segments: Iterable[Any],
    detected_language: Optional[str],
) -> str:
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if y.size == 0:
        return "No usable audio was extracted from the clip."

    duration = len(y) / sr
    hop_length = 512

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zero_crossings = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo_arr = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tempo = float(tempo_arr[0]) if np.size(tempo_arr) else 0.0

    silence_threshold = float(np.percentile(rms, 25)) if rms.size else 0.0
    silence_ratio = float(np.mean(rms <= silence_threshold)) if rms.size else 0.0
    dynamic_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)) if rms_db.size else 0.0
    loudness_db = float(np.mean(rms_db)) if rms_db.size else -80.0
    centroid_hz = float(np.mean(spectral_centroid)) if spectral_centroid.size else 0.0
    zcr = float(np.mean(zero_crossings)) if zero_crossings.size else 0.0

    speech_seconds = 0.0
    for seg in transcript_segments:
        start = max(0.0, float(getattr(seg, "start", 0.0)))
        end = max(start, float(getattr(seg, "end", start)))
        speech_seconds += max(0.0, end - start)
    speech_ratio = min(1.0, speech_seconds / duration) if duration > 0 else 0.0

    transcript_words = len(re.findall(r"\w+", transcript_text))
    speech_profile = describe_level(speech_ratio, 0.2, 0.6, ("music/noise-heavy", "mixed speech-and-sound", "speech-dominant"))
    energy_profile = describe_level(loudness_db, -28.0, -18.0, ("low-energy", "medium-energy", "high-energy"))
    rhythm_profile = describe_level(tempo, 80.0, 130.0, ("slow", "moderate", "fast"))
    brightness_profile = describe_level(centroid_hz, 1200.0, 2600.0, ("warm/dark", "balanced", "bright"))

    language_text = detected_language or "unknown"
    return (
        f"Audio duration {duration:.1f}s. Detected language: {language_text}. "
        f"Transcript length: {transcript_words} words. Speech coverage is about {speech_ratio:.2f} of the clip, "
        f"so the soundtrack is {speech_profile}. Estimated rhythm is {rhythm_profile} at about {tempo:.0f} BPM. "
        f"Average loudness is {loudness_db:.1f} dBFS with dynamic range {dynamic_range_db:.1f} dB, giving a "
        f"{energy_profile} feel. Silence ratio is {silence_ratio:.2f}. Spectral brightness is {brightness_profile} "
        f"(centroid {centroid_hz:.0f} Hz, zero-crossing rate {zcr:.3f})."
    )


def get_cache_key(video_path: Path, max_duration: float, whisper_model: str) -> str:
    stat = video_path.stat()
    signature = {
        "path": str(video_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "max_duration": max_duration,
        "transcript_version": TRANSCRIPT_VERSION,
        "audio_summary_version": AUDIO_SUMMARY_VERSION,
        "whisper_model": whisper_model,
    }
    return sha1_text(json.dumps(signature, sort_keys=True))


def maybe_load_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_cache(cache_path: Path, payload: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_whisper_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def extract_video_side_info(
    video_path: Path,
    cache_dir: Path,
    max_duration: float,
    whisper_model_name: str,
    whisper_model,
) -> Dict[str, Any]:
    cache_key = get_cache_key(video_path, max_duration, whisper_model_name)
    cache_path = cache_dir / f"{cache_key}.json"
    cached = maybe_load_cache(cache_path)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    total_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qwen35_audio_") as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"

        audio_start = time.perf_counter()
        ffmpeg_extract_audio(video_path, wav_path, max_duration=max_duration)
        audio_extract_sec = time.perf_counter() - audio_start

        transcribe_start = time.perf_counter()
        segments, info = whisper_model.transcribe(str(wav_path), word_timestamps=False, vad_filter=True)
        segment_list = list(segments)
        transcript_text = format_transcript(segment_list)
        transcription_sec = time.perf_counter() - transcribe_start

        summary_start = time.perf_counter()
        audio_summary = build_audio_summary(
            wav_path=wav_path,
            transcript_text=transcript_text,
            transcript_segments=segment_list,
            detected_language=getattr(info, "language", None),
        )
        audio_summary_sec = time.perf_counter() - summary_start

    payload = {
        "video_path": str(video_path.resolve()),
        "transcript": transcript_text,
        "audio_summary": audio_summary,
        "detected_language": getattr(info, "language", None),
        "audio_extract_sec": round(audio_extract_sec, 4),
        "transcription_sec": round(transcription_sec, 4),
        "audio_summary_sec": round(audio_summary_sec, 4),
        "total_sec": round(time.perf_counter() - total_start, 4),
        "cache_hit": False,
    }
    save_cache(cache_path, payload)
    return payload


def split_by_author(items: List[Dict[str, Any]], eval_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    author_to_items: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        author = item.get("author_context", "").strip() or "unknown_author"
        author_to_items.setdefault(author, []).append(item)

    authors = sorted(author_to_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(authors)

    if len(authors) <= 1:
        cutoff = max(1, math.ceil(len(items) * eval_ratio))
        return items[cutoff:], items[:cutoff]

    n_eval_authors = max(1, int(round(len(authors) * eval_ratio)))
    eval_authors = set(authors[:n_eval_authors])

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for author, author_items in author_to_items.items():
        if author in eval_authors:
            eval_rows.extend(author_items)
        else:
            train_rows.extend(author_items)
    return train_rows, eval_rows


def convert_rows(
    rows: List[Dict[str, Any]],
    video_fps: float,
) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for idx, raw_item in enumerate(rows):
        item = deterministic_flip(raw_item, idx)
        label = "A" if item["views_a"] > item["views_b"] else "B"
        converted.append(
            {
                "prompt": build_prompt(item, fps=video_fps),
                "label": label,
                "video_a": item["video_a"],
                "video_b": item["video_b"],
                "views_a": item["views_a"],
                "views_b": item["views_b"],
                "author_context": item.get("author_context", ""),
                "transcript_a": item.get("transcript_a", ""),
                "transcript_b": item.get("transcript_b", ""),
                "audio_summary_a": item.get("audio_summary_a", ""),
                "audio_summary_b": item.get("audio_summary_b", ""),
            }
        )
    return converted


def choose_smoke_subset(rows: List[Dict[str, Any]], smoke_pairs: int, seed: int) -> List[Dict[str, Any]]:
    if smoke_pairs >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=smoke_pairs, replace=False).tolist())
    return [rows[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute transcript/audio summaries for GSPO smoke training.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--cache_dir", type=Path, default=Path(__file__).resolve().parent / "cache")
    parser.add_argument("--smoke_pairs", type=int, default=10)
    parser.add_argument("--max_video_duration", type=float, default=45.0)
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whisper_model", default="small")
    parser.add_argument("--whisper_device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
    parser.add_argument("--whisper_compute_type", default=None)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    compute_type = args.whisper_compute_type
    if compute_type is None:
        compute_type = "float16" if args.whisper_device.startswith("cuda") else "int8"

    raw_rows = load_jsonl(args.input)
    subset = choose_smoke_subset(raw_rows, args.smoke_pairs, args.seed)
    whisper_model = build_whisper_model(args.whisper_model, args.whisper_device, compute_type)

    processed_rows: List[Dict[str, Any]] = []
    pair_timings: List[float] = []
    video_timings: List[float] = []
    cache_hits = 0

    for idx, row in enumerate(subset):
        pair_start = time.perf_counter()
        video_a = Path(row["video_a"])
        video_b = Path(row["video_b"])

        meta_a = extract_video_side_info(
            video_path=video_a,
            cache_dir=args.cache_dir,
            max_duration=args.max_video_duration,
            whisper_model_name=args.whisper_model,
            whisper_model=whisper_model,
        )
        meta_b = extract_video_side_info(
            video_path=video_b,
            cache_dir=args.cache_dir,
            max_duration=args.max_video_duration,
            whisper_model_name=args.whisper_model,
            whisper_model=whisper_model,
        )
        cache_hits += int(meta_a.get("cache_hit", False)) + int(meta_b.get("cache_hit", False))
        video_timings.extend([meta_a["total_sec"], meta_b["total_sec"]])

        enriched = dict(row)
        enriched["transcript_a"] = meta_a["transcript"]
        enriched["transcript_b"] = meta_b["transcript"]
        enriched["audio_summary_a"] = meta_a["audio_summary"]
        enriched["audio_summary_b"] = meta_b["audio_summary"]
        processed_rows.append(enriched)
        pair_timings.append(time.perf_counter() - pair_start)

        print(
            f"[{idx + 1}/{len(subset)}] processed pair | "
            f"A cache={meta_a.get('cache_hit', False)} | "
            f"B cache={meta_b.get('cache_hit', False)} | "
            f"pair_sec={pair_timings[-1]:.2f}"
        )

    train_raw, eval_raw = split_by_author(processed_rows, args.eval_ratio, args.seed)
    train_rows = convert_rows(train_raw, video_fps=args.video_fps)
    eval_rows = convert_rows(eval_raw, video_fps=args.video_fps)

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    metrics_path = output_dir / "preprocess_metrics.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    metrics = {
        "input_path": str(args.input.resolve()),
        "smoke_pairs_requested": args.smoke_pairs,
        "pairs_processed": len(processed_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "max_video_duration": args.max_video_duration,
        "video_fps": args.video_fps,
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "whisper_compute_type": compute_type,
        "avg_pair_preprocess_sec": round(statistics.mean(pair_timings), 4) if pair_timings else 0.0,
        "avg_video_preprocess_sec": round(statistics.mean(video_timings), 4) if video_timings else 0.0,
        "cache_hits": cache_hits,
        "cache_misses": max(0, len(video_timings) - cache_hits),
        "train_path": str(train_path.resolve()),
        "eval_path": str(eval_path.resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
