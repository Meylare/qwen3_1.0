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
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import torch

from prompting import build_prompt, deterministic_flip


TRANSCRIPT_VERSION = "v2"
AUDIO_SUMMARY_VERSION = "v4"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "train_fixed.jsonl"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "pair_vid"
SEGMENTER_INSTANCE: Any | None = None
DEFAULT_MUSIC_GENRE_MODEL = "dima806/music_genres_classification"
MUSIC_GENRE_EXTRACTOR: Any | None = None
MUSIC_GENRE_MODEL_INSTANCE: Any | None = None
MUSIC_GENRE_MODEL_KEY: Tuple[str, str] | None = None


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


def resolve_video_path(raw_path: str, video_dir: Path = DEFAULT_VIDEO_DIR) -> Path:
    """Map Windows or relative dataset paths to the local pair_vid directory."""
    normalized = raw_path.replace("\\", "/").strip()
    candidate = Path(normalized).expanduser()
    if candidate.exists():
        return candidate.resolve()

    basename = Path(normalized).name
    if basename:
        local_candidate = video_dir / basename
        if local_candidate.exists():
            return local_candidate.resolve()

    raise FileNotFoundError(
        f"Video file not found for path '{raw_path}'. "
        f"Checked '{candidate}' and '{video_dir / basename}'."
    )


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


def ffprobe_has_audio_stream(video_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def format_transcript(segments: Iterable[Any]) -> str:
    parts: List[str] = []
    for seg in segments:
        text = get_segment_text(seg).strip()
        if not text:
            continue
        parts.append(f"[{get_segment_start(seg):.2f}-{get_segment_end(seg):.2f}] {text}")
    return "\n".join(parts)


def get_segment_start(seg: Any) -> float:
    if isinstance(seg, dict):
        return float(seg.get("start", 0.0))
    return float(getattr(seg, "start", 0.0))


def get_segment_end(seg: Any) -> float:
    if isinstance(seg, dict):
        return float(seg.get("end", 0.0))
    return float(getattr(seg, "end", 0.0))


def get_segment_text(seg: Any) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text", "") or "")
    return str(getattr(seg, "text", "") or "")

def build_music_summary(
    music_wav_path: Optional[Path],
    segmentation_info: Optional[Dict[str, Any]] = None,
    music_genre_model: str = DEFAULT_MUSIC_GENRE_MODEL,
    music_genre_device: str = "cpu",
) -> str:
    if music_wav_path is None or not music_wav_path.exists():
        return "Music not detected."

    y, sr = librosa.load(str(music_wav_path), sr=16000, mono=True)
    if y.size == 0:
        return "Music not detected."

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo_arr = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tempo = float(tempo_arr[0]) if np.size(tempo_arr) else 0.0
    genre_label, genre_confidence = classify_music_genre(
        wav_path=music_wav_path,
        model_name=music_genre_model,
        device=music_genre_device,
    )

    parts = []
    if genre_label:
        if genre_confidence is not None:
            parts.append(f"Music genre: {genre_label} ({genre_confidence:.2f} confidence).")
        else:
            parts.append(f"Music genre: {genre_label}.")
    else:
        parts.append("Music genre: unknown.")

    if tempo > 0:
        parts.append(f"Music tempo: ~{tempo:.0f} BPM.")
    else:
        parts.append("Music tempo: unknown.")
    return " ".join(parts)


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


def get_cache_key_with_segmentation(
    video_path: Path,
    max_duration: float,
    whisper_model: str,
    speech_segmentation_mode: str,
    speech_min_segment_sec: float,
    speech_merge_gap_sec: float,
    speech_keep_leading_trailing_pad_sec: float,
    segmentation_fallback_to_full_audio: bool,
    music_genre_model: str,
) -> str:
    stat = video_path.stat()
    signature = {
        "path": str(video_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "max_duration": max_duration,
        "transcript_version": TRANSCRIPT_VERSION,
        "audio_summary_version": AUDIO_SUMMARY_VERSION,
        "whisper_model": whisper_model,
        "speech_segmentation_mode": speech_segmentation_mode,
        "speech_min_segment_sec": speech_min_segment_sec,
        "speech_merge_gap_sec": speech_merge_gap_sec,
        "speech_keep_leading_trailing_pad_sec": speech_keep_leading_trailing_pad_sec,
        "segmentation_fallback_to_full_audio": segmentation_fallback_to_full_audio,
        "music_genre_model": music_genre_model,
    }
    return sha1_text(json.dumps(signature, sort_keys=True))


def build_speech_segmenter():
    global SEGMENTER_INSTANCE
    if SEGMENTER_INSTANCE is not None:
        return SEGMENTER_INSTANCE
    from inaSpeechSegmenter import Segmenter

    SEGMENTER_INSTANCE = Segmenter()
    return SEGMENTER_INSTANCE


def is_inaspeech_speech_label(label: str) -> bool:
    normalized = (label or "").strip().lower()
    return normalized in {"speech", "male", "female"}


def is_inaspeech_music_label(label: str) -> bool:
    normalized = (label or "").strip().lower()
    return "music" in normalized


def postprocess_labeled_segments(
    segments: List[Dict[str, Any]],
    total_duration_sec: float,
    min_segment_sec: float,
    merge_gap_sec: float,
    keep_pad_sec: float,
    label_predicate,
    output_label: str,
) -> List[Dict[str, Any]]:
    filtered_segments: List[Dict[str, Any]] = []
    for seg in segments:
        if not label_predicate(str(seg.get("label", ""))):
            continue
        start = max(0.0, float(seg["start_sec"]) - keep_pad_sec)
        end = min(total_duration_sec, float(seg["end_sec"]) + keep_pad_sec)
        if end - start < min_segment_sec:
            continue
        filtered_segments.append(
            {
                "label": output_label,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": max(0.0, end - start),
            }
        )

    if not filtered_segments:
        return []

    filtered_segments.sort(key=lambda x: x["start_sec"])
    merged = [filtered_segments[0].copy()]
    for seg in filtered_segments[1:]:
        prev = merged[-1]
        if float(seg["start_sec"]) - float(prev["end_sec"]) <= merge_gap_sec:
            prev["end_sec"] = max(float(prev["end_sec"]), float(seg["end_sec"]))
            prev["duration_sec"] = max(0.0, float(prev["end_sec"]) - float(prev["start_sec"]))
        else:
            merged.append(seg.copy())
    return merged


def segment_audio_with_inaspeech(wav_path: Path) -> List[Dict[str, Any]]:
    segmenter = build_speech_segmenter()
    raw_segments = segmenter(str(wav_path))
    normalized: List[Dict[str, Any]] = []
    for item in raw_segments:
        if len(item) < 3:
            continue
        label = str(item[0]).strip().lower()
        start = float(item[1])
        end = float(item[2])
        normalized.append(
            {
                "label": label,
                "start_sec": max(0.0, start),
                "end_sec": max(start, end),
                "duration_sec": max(0.0, end - start),
            }
        )
    return normalized


def postprocess_speech_segments(
    segments: List[Dict[str, Any]],
    total_duration_sec: float,
    min_segment_sec: float,
    merge_gap_sec: float,
    keep_pad_sec: float,
) -> List[Dict[str, Any]]:
    return postprocess_labeled_segments(
        segments=segments,
        total_duration_sec=total_duration_sec,
        min_segment_sec=min_segment_sec,
        merge_gap_sec=merge_gap_sec,
        keep_pad_sec=keep_pad_sec,
        label_predicate=is_inaspeech_speech_label,
        output_label="speech",
    )


def postprocess_music_segments(
    segments: List[Dict[str, Any]],
    total_duration_sec: float,
    min_segment_sec: float = 1.0,
    merge_gap_sec: float = 0.5,
) -> List[Dict[str, Any]]:
    return postprocess_labeled_segments(
        segments=segments,
        total_duration_sec=total_duration_sec,
        min_segment_sec=min_segment_sec,
        merge_gap_sec=merge_gap_sec,
        keep_pad_sec=0.0,
        label_predicate=is_inaspeech_music_label,
        output_label="music",
    )


def render_segments_wav(src_wav: Path, segments: List[Dict[str, Any]], dst_wav: Path) -> Tuple[List[Dict[str, float]], float]:
    if not segments:
        return [], 0.0

    with wave.open(str(src_wav), "rb") as reader:
        nchannels = reader.getnchannels()
        sampwidth = reader.getsampwidth()
        framerate = reader.getframerate()
        nframes = reader.getnframes()
        frames = reader.readframes(nframes)

    if nchannels != 1 or sampwidth != 2:
        raise RuntimeError(f"Expected mono 16-bit wav, got channels={nchannels}, sampwidth={sampwidth}")

    pcm = np.frombuffer(frames, dtype=np.int16)
    pieces: List[np.ndarray] = []
    mapping: List[Dict[str, float]] = []
    cursor_sec = 0.0

    for seg in segments:
        orig_start = max(0.0, float(seg["start_sec"]))
        orig_end = max(orig_start, float(seg["end_sec"]))
        start_idx = min(len(pcm), int(round(orig_start * framerate)))
        end_idx = min(len(pcm), int(round(orig_end * framerate)))
        if end_idx <= start_idx:
            continue
        piece = pcm[start_idx:end_idx]
        pieces.append(piece)
        duration_sec = len(piece) / float(framerate)
        mapping.append(
            {
                "concat_start_sec": cursor_sec,
                "concat_end_sec": cursor_sec + duration_sec,
                "orig_start_sec": orig_start,
                "orig_end_sec": orig_start + duration_sec,
            }
        )
        cursor_sec += duration_sec

    if not pieces:
        return [], 0.0

    out = np.concatenate(pieces)
    with wave.open(str(dst_wav), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(framerate)
        writer.writeframes(out.astype(np.int16).tobytes())

    return mapping, len(out) / float(framerate)


def render_speech_only_wav(src_wav: Path, speech_segments: List[Dict[str, Any]], dst_wav: Path) -> Tuple[List[Dict[str, float]], float]:
    return render_segments_wav(src_wav=src_wav, segments=speech_segments, dst_wav=dst_wav)


def remap_time_to_original(value: float, mapping_table: List[Dict[str, float]]) -> float:
    if not mapping_table:
        return value
    for item in mapping_table:
        concat_start = float(item["concat_start_sec"])
        concat_end = float(item["concat_end_sec"])
        if concat_start <= value <= concat_end:
            return float(item["orig_start_sec"]) + (value - concat_start)
    if value < float(mapping_table[0]["concat_start_sec"]):
        return float(mapping_table[0]["orig_start_sec"])
    last = mapping_table[-1]
    overflow = value - float(last["concat_end_sec"])
    return float(last["orig_end_sec"]) + max(0.0, overflow)


def remap_whisper_segments_to_original(segments: Iterable[Any], mapping_table: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    remapped: List[Dict[str, Any]] = []
    for seg in segments:
        start = remap_time_to_original(get_segment_start(seg), mapping_table)
        end = remap_time_to_original(get_segment_end(seg), mapping_table)
        remapped.append(
            {
                "start": round(start, 4),
                "end": round(max(start, end), 4),
                "text": get_segment_text(seg),
            }
        )
    return remapped


def build_segmentation_summary(segmentation_info: Dict[str, Any]) -> str:
    return (
        f"Segmentation mode {segmentation_info.get('mode', 'off')} | "
        f"speech={float(segmentation_info.get('speech_sec', 0.0) or 0.0):.1f}s | "
        f"music={float(segmentation_info.get('music_sec', 0.0) or 0.0):.1f}s | "
        f"noise={float(segmentation_info.get('noise_sec', 0.0) or 0.0):.1f}s | "
        f"speech_ratio={float(segmentation_info.get('speech_ratio', 0.0) or 0.0):.2f} | "
        f"speech_segments={int(segmentation_info.get('num_speech_segments', 0) or 0)}"
    )


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


def build_music_genre_classifier(model_name: str, device: str):
    global MUSIC_GENRE_EXTRACTOR, MUSIC_GENRE_MODEL_INSTANCE, MUSIC_GENRE_MODEL_KEY

    model_key = (model_name, device)
    if MUSIC_GENRE_MODEL_INSTANCE is not None and MUSIC_GENRE_EXTRACTOR is not None and MUSIC_GENRE_MODEL_KEY == model_key:
        return MUSIC_GENRE_EXTRACTOR, MUSIC_GENRE_MODEL_INSTANCE

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(torch_device)
    model.eval()

    MUSIC_GENRE_EXTRACTOR = extractor
    MUSIC_GENRE_MODEL_INSTANCE = model
    MUSIC_GENRE_MODEL_KEY = model_key
    return extractor, model


def classify_music_genre(wav_path: Path, model_name: str, device: str) -> Tuple[Optional[str], Optional[float]]:
    try:
        extractor, model = build_music_genre_classifier(model_name, device)
    except Exception:
        return None, None

    sampling_rate = int(getattr(extractor, "sampling_rate", 16000) or 16000)
    y, _ = librosa.load(str(wav_path), sr=sampling_rate, mono=True)
    if y.size == 0:
        return None, None

    max_seconds = 30
    max_samples = sampling_rate * max_seconds
    if len(y) > max_samples:
        y = y[:max_samples]

    inputs = extractor(y, sampling_rate=sampling_rate, return_tensors="pt")
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        logits = model(**model_inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        best_idx = int(torch.argmax(probs).item())
        confidence = float(probs[best_idx].item())

    label = model.config.id2label.get(best_idx)
    return (str(label).strip().lower() if label is not None else None, confidence)


def transcribe_audio_path(
    whisper_model,
    wav_path: Path,
    vad_filter: bool,
    mapping_table: Optional[List[Dict[str, float]]] = None,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    segments, info = whisper_model.transcribe(
        str(wav_path),
        word_timestamps=False,
        vad_filter=vad_filter,
    )
    raw_segments = list(segments)
    if mapping_table:
        segment_list = remap_whisper_segments_to_original(raw_segments, mapping_table)
    else:
        segment_list = [
            {
                "start": round(get_segment_start(seg), 4),
                "end": round(get_segment_end(seg), 4),
                "text": get_segment_text(seg),
            }
            for seg in raw_segments
        ]
    return format_transcript(segment_list), segment_list, getattr(info, "language", None)


def extract_video_side_info(
    video_path: Path,
    cache_dir: Path,
    max_duration: float,
    whisper_model_name: str,
    whisper_model,
    speech_segmentation_mode: str = "inaspeech",
    speech_min_segment_sec: float = 0.35,
    speech_merge_gap_sec: float = 0.25,
    speech_keep_leading_trailing_pad_sec: float = 0.10,
    segmentation_fallback_to_full_audio: bool = True,
    music_genre_model: str = DEFAULT_MUSIC_GENRE_MODEL,
    music_genre_device: str = "cpu",
) -> Dict[str, Any]:
    cache_key = get_cache_key_with_segmentation(
        video_path=video_path,
        max_duration=max_duration,
        whisper_model=whisper_model_name,
        speech_segmentation_mode=speech_segmentation_mode,
        speech_min_segment_sec=speech_min_segment_sec,
        speech_merge_gap_sec=speech_merge_gap_sec,
        speech_keep_leading_trailing_pad_sec=speech_keep_leading_trailing_pad_sec,
        segmentation_fallback_to_full_audio=segmentation_fallback_to_full_audio,
        music_genre_model=music_genre_model,
    )
    cache_path = cache_dir / f"{cache_key}.json"
    cached = maybe_load_cache(cache_path)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    total_start = time.perf_counter()

    if not ffprobe_has_audio_stream(video_path):
        payload = {
            "video_path": str(video_path.resolve()),
            "transcript": "",
            "transcript_segments": [],
            "lyrics_transcript": "",
            "lyrics_transcript_segments": [],
            "audio_summary": "Music not detected.",
            "detected_language": None,
            "lyrics_detected_language": None,
            "speech_segmentation_mode": speech_segmentation_mode,
            "segmentation_applied": False,
            "segmentation_fallback": False,
            "transcript_source": "no_audio",
            "segmentation_total_audio_sec": 0.0,
            "segmentation_speech_sec": 0.0,
            "segmentation_music_sec": 0.0,
            "segmentation_noise_sec": 0.0,
            "segmentation_speech_ratio": 0.0,
            "segmentation_num_speech_segments": 0,
            "speech_only_audio_sec": 0.0,
            "music_only_audio_sec": 0.0,
            "segmentation_summary": "No audio stream detected in the clip.",
            "segmentation_sec": 0.0,
            "audio_extract_sec": 0.0,
            "transcription_sec": 0.0,
            "lyrics_transcription_sec": 0.0,
            "audio_summary_sec": 0.0,
            "total_sec": round(time.perf_counter() - total_start, 4),
            "cache_hit": False,
        }
        save_cache(cache_path, payload)
        return payload

    with tempfile.TemporaryDirectory(prefix="qwen35_audio_") as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        speech_only_wav_path = Path(tmpdir) / "speech_only.wav"
        music_only_wav_path = Path(tmpdir) / "music_only.wav"

        audio_start = time.perf_counter()
        ffmpeg_extract_audio(video_path, wav_path, max_duration=max_duration)
        audio_extract_sec = time.perf_counter() - audio_start

        y_full, sr_full = librosa.load(str(wav_path), sr=16000, mono=True)
        total_audio_sec = float(len(y_full) / sr_full) if y_full.size else 0.0
        segmentation_applied = speech_segmentation_mode == "inaspeech"
        segmentation_fallback = False
        transcript_source = "full_audio"
        segmentation_sec = 0.0
        speech_only_audio_sec = 0.0
        music_only_audio_sec = 0.0
        segmentation_summary = ""
        transcript_segments_for_summary: List[Any] = []
        lyrics_segment_list: List[Dict[str, Any]] = []
        lyrics_text = ""
        lyrics_detected_language: Optional[str] = None
        segmentation_info = {
            "mode": speech_segmentation_mode,
            "total_audio_sec": round(total_audio_sec, 4),
            "speech_sec": 0.0,
            "music_sec": 0.0,
            "noise_sec": 0.0,
            "speech_ratio": 0.0,
            "num_speech_segments": 0,
        }
        transcription_target_wav = wav_path
        speech_mapping_table: List[Dict[str, float]] = []
        music_mapping_table: List[Dict[str, float]] = []
        music_summary = "Music not detected."

        if speech_segmentation_mode == "inaspeech":
            seg_start = time.perf_counter()
            try:
                raw_segments = segment_audio_with_inaspeech(wav_path)
                processed_speech_segments = postprocess_speech_segments(
                    raw_segments,
                    total_duration_sec=total_audio_sec,
                    min_segment_sec=speech_min_segment_sec,
                    merge_gap_sec=speech_merge_gap_sec,
                    keep_pad_sec=speech_keep_leading_trailing_pad_sec,
                )
                processed_music_segments = postprocess_music_segments(
                    raw_segments,
                    total_duration_sec=total_audio_sec,
                )
                label_durations = {"speech": 0.0, "music": 0.0, "noise": 0.0}
                for seg in raw_segments:
                    label = str(seg.get("label", "")).lower()
                    duration = float(seg.get("duration_sec", 0.0) or 0.0)
                    if is_inaspeech_speech_label(label):
                        label_durations["speech"] += duration
                    elif is_inaspeech_music_label(label):
                        label_durations["music"] += duration
                    else:
                        label_durations["noise"] += duration

                speech_mapping_table, speech_only_audio_sec = render_speech_only_wav(
                    src_wav=wav_path,
                    speech_segments=processed_speech_segments,
                    dst_wav=speech_only_wav_path,
                )
                music_mapping_table, music_only_audio_sec = render_segments_wav(
                    src_wav=wav_path,
                    segments=processed_music_segments,
                    dst_wav=music_only_wav_path,
                )
                segmentation_info = {
                    "mode": speech_segmentation_mode,
                    "total_audio_sec": round(total_audio_sec, 4),
                    "speech_sec": round(sum(float(s["duration_sec"]) for s in processed_speech_segments), 4),
                    "music_sec": round(label_durations["music"], 4),
                    "noise_sec": round(max(0.0, total_audio_sec - label_durations["music"] - label_durations["speech"]), 4),
                    "speech_ratio": round(
                        (sum(float(s["duration_sec"]) for s in processed_speech_segments) / total_audio_sec)
                        if total_audio_sec > 0
                        else 0.0,
                        4,
                    ),
                    "num_speech_segments": len(processed_speech_segments),
                }
                transcript_source = "speech_only"
                segmentation_summary = build_segmentation_summary(segmentation_info)
                if speech_only_audio_sec > 0 and speech_only_wav_path.exists():
                    transcription_target_wav = speech_only_wav_path
                else:
                    transcript_source = "speech_only"
            except Exception:
                segmentation_fallback = True
                segmentation_summary = "inaSpeech segmentation failed; fell back to full-audio Whisper."
                if not segmentation_fallback_to_full_audio:
                    raise
                transcript_source = "full_audio"
                transcription_target_wav = wav_path
            finally:
                segmentation_sec = time.perf_counter() - seg_start

        transcribe_start = time.perf_counter()
        if transcript_source == "speech_only" and not segmentation_fallback and speech_only_audio_sec <= 0:
            segment_list = []
            info = type("WhisperInfo", (), {"language": None})()
            transcript_text = ""
            transcription_sec = time.perf_counter() - transcribe_start
        else:
            transcript_text, segment_list, detected_language = transcribe_audio_path(
                whisper_model=whisper_model,
                wav_path=transcription_target_wav,
                vad_filter=True,
                mapping_table=speech_mapping_table if transcript_source == "speech_only" else None,
            )
            info = type("WhisperInfo", (), {"language": detected_language})()
            transcription_sec = time.perf_counter() - transcribe_start
        transcript_segments_for_summary = segment_list

        lyrics_start = time.perf_counter()
        if (
            speech_segmentation_mode == "inaspeech"
            and not segmentation_fallback
            and music_only_audio_sec > 0
            and music_only_wav_path.exists()
        ):
            lyrics_text, lyrics_segment_list, lyrics_detected_language = transcribe_audio_path(
                whisper_model=whisper_model,
                wav_path=music_only_wav_path,
                vad_filter=False,
                mapping_table=music_mapping_table,
            )
        lyrics_transcription_sec = time.perf_counter() - lyrics_start

        summary_start = time.perf_counter()
        audio_summary = build_music_summary(
            music_wav_path=music_only_wav_path if music_only_audio_sec > 0 and music_only_wav_path.exists() else None,
            segmentation_info=segmentation_info if segmentation_applied else None,
            music_genre_model=music_genre_model,
            music_genre_device=music_genre_device,
        )
        audio_summary_sec = time.perf_counter() - summary_start

    payload = {
        "video_path": str(video_path.resolve()),
        "transcript": transcript_text,
        "transcript_segments": transcript_segments_for_summary,
        "lyrics_transcript": lyrics_text,
        "lyrics_transcript_segments": lyrics_segment_list,
        "audio_summary": audio_summary,
        "detected_language": getattr(info, "language", None),
        "lyrics_detected_language": lyrics_detected_language,
        "speech_segmentation_mode": speech_segmentation_mode,
        "segmentation_applied": segmentation_applied,
        "segmentation_fallback": segmentation_fallback,
        "transcript_source": transcript_source,
        "segmentation_total_audio_sec": round(float(segmentation_info.get("total_audio_sec", 0.0) or 0.0), 4),
        "segmentation_speech_sec": round(float(segmentation_info.get("speech_sec", 0.0) or 0.0), 4),
        "segmentation_music_sec": round(float(segmentation_info.get("music_sec", 0.0) or 0.0), 4),
        "segmentation_noise_sec": round(float(segmentation_info.get("noise_sec", 0.0) or 0.0), 4),
        "segmentation_speech_ratio": round(float(segmentation_info.get("speech_ratio", 0.0) or 0.0), 4),
        "segmentation_num_speech_segments": int(segmentation_info.get("num_speech_segments", 0) or 0),
        "speech_only_audio_sec": round(float(speech_only_audio_sec), 4),
        "music_only_audio_sec": round(float(music_only_audio_sec), 4),
        "segmentation_summary": segmentation_summary,
        "segmentation_sec": round(segmentation_sec, 4),
        "audio_extract_sec": round(audio_extract_sec, 4),
        "transcription_sec": round(transcription_sec, 4),
        "lyrics_transcription_sec": round(lyrics_transcription_sec, 4),
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
                "transcript_segments_a": item.get("transcript_segments_a", []),
                "transcript_segments_b": item.get("transcript_segments_b", []),
                "lyrics_a": item.get("lyrics_a", ""),
                "lyrics_b": item.get("lyrics_b", ""),
                "lyrics_segments_a": item.get("lyrics_segments_a", []),
                "lyrics_segments_b": item.get("lyrics_segments_b", []),
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
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--whisper_device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--whisper_compute_type", default=None)
    parser.add_argument("--speech_segmentation_mode", choices=("off", "inaspeech"), default="inaspeech")
    parser.add_argument("--speech_min_segment_sec", type=float, default=0.35)
    parser.add_argument("--speech_merge_gap_sec", type=float, default=0.25)
    parser.add_argument("--speech_keep_leading_trailing_pad_sec", type=float, default=0.10)
    parser.add_argument("--segmentation_fallback_to_full_audio", action="store_true", default=True)
    parser.add_argument("--no_segmentation_fallback_to_full_audio", action="store_false", dest="segmentation_fallback_to_full_audio")
    parser.add_argument("--music_genre_model", default=DEFAULT_MUSIC_GENRE_MODEL)
    parser.add_argument("--music_genre_device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    segmentation_applied_count = 0
    segmentation_fallback_count = 0
    empty_transcript_count = 0
    segmentation_times: List[float] = []
    transcription_times_after_segmentation: List[float] = []
    speech_ratios: List[float] = []

    for idx, row in enumerate(subset):
        pair_start = time.perf_counter()
        video_a = resolve_video_path(row["video_a"])
        video_b = resolve_video_path(row["video_b"])

        meta_a = extract_video_side_info(
            video_path=video_a,
            cache_dir=args.cache_dir,
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
        meta_b = extract_video_side_info(
            video_path=video_b,
            cache_dir=args.cache_dir,
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
        cache_hits += int(meta_a.get("cache_hit", False)) + int(meta_b.get("cache_hit", False))
        video_timings.extend([meta_a["total_sec"], meta_b["total_sec"]])
        for meta in (meta_a, meta_b):
            segmentation_applied_count += int(bool(meta.get("segmentation_applied", False)))
            segmentation_fallback_count += int(bool(meta.get("segmentation_fallback", False)))
            empty_transcript_count += int(not bool(str(meta.get("transcript", "")).strip()))
            segmentation_times.append(float(meta.get("segmentation_sec", 0.0) or 0.0))
            transcription_times_after_segmentation.append(float(meta.get("transcription_sec", 0.0) or 0.0))
            speech_ratios.append(float(meta.get("segmentation_speech_ratio", 0.0) or 0.0))

        enriched = dict(row)
        enriched["transcript_a"] = meta_a["transcript"]
        enriched["transcript_b"] = meta_b["transcript"]
        enriched["transcript_segments_a"] = meta_a.get("transcript_segments", [])
        enriched["transcript_segments_b"] = meta_b.get("transcript_segments", [])
        enriched["lyrics_a"] = meta_a.get("lyrics_transcript", "")
        enriched["lyrics_b"] = meta_b.get("lyrics_transcript", "")
        enriched["lyrics_segments_a"] = meta_a.get("lyrics_transcript_segments", [])
        enriched["lyrics_segments_b"] = meta_b.get("lyrics_transcript_segments", [])
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
        "speech_segmentation_mode": args.speech_segmentation_mode,
        "music_genre_model": args.music_genre_model,
        "music_genre_device": args.music_genre_device,
        "speech_min_segment_sec": args.speech_min_segment_sec,
        "speech_merge_gap_sec": args.speech_merge_gap_sec,
        "speech_keep_leading_trailing_pad_sec": args.speech_keep_leading_trailing_pad_sec,
        "segmentation_fallback_to_full_audio": args.segmentation_fallback_to_full_audio,
        "avg_pair_preprocess_sec": round(statistics.mean(pair_timings), 4) if pair_timings else 0.0,
        "avg_video_preprocess_sec": round(statistics.mean(video_timings), 4) if video_timings else 0.0,
        "segmentation_videos": segmentation_applied_count,
        "segmentation_fallback_videos": segmentation_fallback_count,
        "empty_transcript_videos": empty_transcript_count,
        "avg_segmentation_sec": round(statistics.mean(segmentation_times), 4) if segmentation_times else 0.0,
        "avg_transcription_after_segmentation_sec": round(statistics.mean(transcription_times_after_segmentation), 4)
        if transcription_times_after_segmentation
        else 0.0,
        "avg_segmentation_speech_ratio": round(statistics.mean(speech_ratios), 4) if speech_ratios else 0.0,
        "cache_hits": cache_hits,
        "cache_misses": max(0, len(video_timings) - cache_hits),
        "train_path": str(train_path.resolve()),
        "eval_path": str(eval_path.resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
