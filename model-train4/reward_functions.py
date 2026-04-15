"""
Reward functions for pairwise virality training with winner + advice outputs.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - graceful fallback if dependency is unavailable
    DetectorFactory = None
    LangDetectException = Exception
    detect_langs = None


WINNER_RE = re.compile(r"<winner>\s*([AB])\s*</winner>", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
THINK_PREFILL_RE = re.compile(r"^(.*?)</think>", re.DOTALL | re.IGNORECASE)
ADVICE_RE = re.compile(r"<advice>(.*?)</advice>", re.DOTALL | re.IGNORECASE)
LANG_RE = re.compile(r"detected language:\s*([a-z]{2,3}(?:-[a-z]{2,3})?)", re.IGNORECASE)
CREATOR_CONTEXT_RE = re.compile(r"creator context:\s*(.*?)\n\s*video a metadata:", re.IGNORECASE | re.DOTALL)
SEGMENTATION_SPEECH_RE = re.compile(
    r"(?:segmentation found|inaspeech breakdown:)\s*speech=\s*([0-9]*\.?[0-9]+)s",
    re.IGNORECASE,
)

GENERIC_ADVICE_PATTERNS = (
    "make it more engaging",
    "make it better",
    "improve the video",
    "improve quality",
    "make it more interesting",
    "make it more dynamic",
    "сделайте интереснее",
    "сделайте лучше",
    "улучшите видео",
    "улучшите качество",
    "сделайте динамичнее",
    "сделайте более вовлекающим",
)

CONCRETE_HINT_PATTERNS = (
    "first 2 seconds",
    "first 3 seconds",
    "first second",
    "opening shot",
    "start with",
    "show the",
    "cut the",
    "add text",
    "subtitle",
    "caption",
    "hook",
    "cta",
    "price",
    "product",
    "close-up",
    "voice",
    "visual",
    "montage",
    "edit",
    "pacing",
    "в начале",
    "в первые",
    "первые 2",
    "первые 3",
    "первая секунда",
    "покажите",
    "сократите",
    "добавьте текст",
    "субтитр",
    "подпись",
    "хук",
    "цта",
    "цена",
    "товар",
    "крупный план",
    "озвуч",
    "визуал",
    "монтаж",
    "темп",
    "ритм",
    "голос",
)

MIN_ADVICE_LANG_CHARS = 20
MIN_REASONING_CHARS = 24
DEFAULT_MAX_COMPLETION_LENGTH = 2500

_LAST_REWARD_DEBUG_BATCH: List[Dict[str, Any]] = []


def coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (coerce_text(item) for item in value) if part)
    if isinstance(value, dict):
        if "content" in value:
            return coerce_text(value["content"])
        if isinstance(value.get("text"), str):
            return value["text"]
        return "\n".join(part for part in (coerce_text(item) for item in value.values()) if part)
    return ""


def parse_winner_block(text: Any) -> Optional[str]:
    normalized = coerce_text(text)
    normalized = THINK_RE.sub("", normalized)
    match = WINNER_RE.search(normalized)
    if match:
        return match.group(1).upper()
    match = ANSWER_RE.search(normalized)
    return match.group(1).upper() if match else None


def extract_think(text: Any) -> str:
    normalized = coerce_text(text)
    match = THINK_RE.search(normalized)
    if match:
        return match.group(1).strip()
    prefill_match = THINK_PREFILL_RE.search(normalized)
    return prefill_match.group(1).strip() if prefill_match else ""


def extract_advice(text: Any) -> str:
    normalized = coerce_text(text)
    match = ADVICE_RE.search(normalized)
    return match.group(1).strip() if match else ""


def has_valid_full_structure(text: Any) -> bool:
    return bool(extract_think(text)) and bool(parse_winner_block(text)) and bool(extract_advice(text))


def normalize_lang_code(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().lower().replace("_", "-")
    if not value:
        return None
    if value in {"unknown", "unk", "und", "none", "null", "n/a"}:
        return None
    if value == "ru-ru":
        return "ru"
    if value == "en-us":
        return "en"
    if value == "en-gb":
        return "en"
    if "-" in value:
        value = value.split("-", 1)[0]
    if not re.fullmatch(r"[a-z]{2,3}", value):
        return None
    return value


def extract_language_from_audio_summary(text: Any) -> Optional[str]:
    normalized = coerce_text(text)
    match = LANG_RE.search(normalized)
    return normalize_lang_code(match.group(1)) if match else None


def detect_language(text: Any, min_chars: int = MIN_ADVICE_LANG_CHARS) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", coerce_text(text)).strip()
    if len(normalized) < min_chars or detect_langs is None:
        return None
    try:
        candidates = detect_langs(normalized)
    except LangDetectException:
        return None
    if not candidates:
        return None
    best = candidates[0]
    if getattr(best, "prob", 0.0) < 0.70:
        return None
    return normalize_lang_code(getattr(best, "lang", None))


def infer_prompt_language(prompt: Any) -> Optional[str]:
    text = coerce_text(prompt)
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return None


def extract_creator_context(prompt: Any) -> str:
    text = coerce_text(prompt)
    match = CREATOR_CONTEXT_RE.search(text)
    return match.group(1).strip() if match else ""


def infer_creator_context_language(prompt: Any) -> Optional[str]:
    context = extract_creator_context(prompt)
    if not context:
        return None
    if re.search(r"[А-Яа-яЁё]", context):
        return "ru"
    if re.search(r"[A-Za-z]", context):
        return "en"
    return None


def is_music_only_case(audio_summary: Any, transcript: Any) -> bool:
    summary = coerce_text(audio_summary)
    transcript_text = coerce_text(transcript).strip()

    speech_match = SEGMENTATION_SPEECH_RE.search(summary)
    if speech_match:
        try:
            speech_seconds = float(speech_match.group(1))
            if speech_seconds <= 0.01 and not transcript_text:
                return True
        except ValueError:
            pass

    lowered = summary.lower()
    if "author speech was not detected after segmentation" in lowered and not transcript_text:
        return True
    if "music not detected" not in lowered and not transcript_text:
        return True
    return False


def is_generic_advice(advice: str) -> bool:
    advice_lower = advice.lower()
    return any(pattern in advice_lower for pattern in GENERIC_ADVICE_PATTERNS)


def has_concrete_advice_signal(advice: str) -> bool:
    advice_lower = advice.lower()
    if re.search(r"\d", advice_lower):
        return True
    return any(pattern in advice_lower for pattern in CONCRETE_HINT_PATTERNS)


def format_status(completion: Any) -> str:
    winner = parse_winner_block(completion)
    think = extract_think(completion)
    advice = extract_advice(completion)
    if not winner:
        return "missing_winner"
    if think and advice:
        return "full"
    if not think and not advice:
        return "winner_only"
    if not think:
        return "missing_think"
    return "missing_advice"


def determine_target_language(
    gt: str,
    prompt: Any = None,
    transcript_a: Any = None,
    transcript_b: Any = None,
    lyrics_a: Any = None,
    lyrics_b: Any = None,
    audio_summary_a: Any = None,
    audio_summary_b: Any = None,
) -> Optional[str]:
    loser_label = "B" if gt == "A" else "A"
    loser_transcript = transcript_b if loser_label == "B" else transcript_a
    loser_lyrics = lyrics_b if loser_label == "B" else lyrics_a

    lang = detect_language(loser_transcript)
    if lang:
        return lang

    lang = detect_language(loser_lyrics)
    if lang:
        return lang

    loser_audio_summary = audio_summary_b if loser_label == "B" else audio_summary_a
    lang = extract_language_from_audio_summary(loser_audio_summary)
    if lang:
        return lang

    lang = infer_creator_context_language(prompt)
    if lang:
        return lang

    return infer_prompt_language(prompt)


def reasoning_signal(think_text: str) -> float:
    normalized = re.sub(r"\s+", " ", think_text).strip()
    if len(normalized) < MIN_REASONING_CHARS:
        return -0.05
    return 0.05


def resolve_max_completion_length(context: Dict[str, Any]) -> int:
    for key in ("max_completion_length", "reward_max_completion_length"):
        value = context.get(key)
        if value is not None:
            try:
                parsed = int(value)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass
    env_value = os.environ.get("GSPO_MAX_COMPLETION_LENGTH")
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_MAX_COMPLETION_LENGTH


def has_unfinished_structure(text: Any) -> bool:
    normalized = coerce_text(text)
    lowered = normalized.lower()
    if "<advice>" in lowered and "</advice>" not in lowered:
        return True
    if "<winner>" in lowered and "</winner>" not in lowered:
        return True
    if "<think>" in lowered and "</think>" not in lowered:
        return True
    status = format_status(normalized)
    return status in {"missing_winner", "missing_advice", "missing_think", "winner_only"}


def length_signal(completion: Any, max_completion_length: int) -> tuple[float, bool, bool, int]:
    completion_text = coerce_text(completion)
    completion_char_length = len(completion_text)
    capped = max(1, int(max_completion_length))
    soft_limit = int(capped * 4)
    ratio = completion_char_length / soft_limit

    penalty = 0.0
    near_cap_penalty_applied = False
    unfinished_near_cap_penalty_applied = False

    if ratio >= 0.97:
        penalty -= 0.20
        near_cap_penalty_applied = True
    elif ratio >= 0.90:
        penalty -= 0.10
        near_cap_penalty_applied = True

    if ratio >= 0.90 and has_unfinished_structure(completion):
        penalty -= 0.50
        unfinished_near_cap_penalty_applied = True

    return round(penalty, 4), near_cap_penalty_applied, unfinished_near_cap_penalty_applied, completion_char_length


def language_signal(advice_text: str, target_language: Optional[str]) -> tuple[float, Optional[str]]:
    advice_language = detect_language(advice_text)
    if not target_language or not advice_language:
        return 0.0, advice_language
    if advice_language == target_language:
        return 0.1, advice_language
    return -0.2, advice_language


def build_reward_debug_entry(completion: Any, gt: str, **context: Any) -> Dict[str, Any]:
    predicted = parse_winner_block(completion)
    think_text = extract_think(completion)
    advice_text = extract_advice(completion)
    status = format_status(completion)

    if predicted is None:
        reward_accuracy = -1.5
    elif predicted == gt:
        reward_accuracy = 1.0
    else:
        reward_accuracy = -1.0

    if status == "full":
        reward_format = 0.2
    elif status == "missing_winner":
        reward_format = -0.5
    else:
        reward_format = -0.1

    target_language = determine_target_language(
        gt,
        prompt=context.get("prompt"),
        transcript_a=context.get("transcript_a"),
        transcript_b=context.get("transcript_b"),
        lyrics_a=context.get("lyrics_a"),
        lyrics_b=context.get("lyrics_b"),
        audio_summary_a=context.get("audio_summary_a"),
        audio_summary_b=context.get("audio_summary_b"),
    )
    reward_language, advice_language = language_signal(advice_text, target_language)

    reward_reasoning = reasoning_signal(think_text)
    generic_penalty_applied = False
    if advice_text and is_generic_advice(advice_text) and not has_concrete_advice_signal(advice_text):
        reward_reasoning -= 0.05
        generic_penalty_applied = True

    max_completion_length = resolve_max_completion_length(context)
    reward_length, near_cap_penalty_applied, unfinished_near_cap_penalty_applied, completion_char_length = length_signal(
        completion,
        max_completion_length=max_completion_length,
    )

    return {
        "ground_truth_winner": gt,
        "parsed_winner": predicted,
        "target_language": target_language,
        "advice_language": advice_language,
        "format_status": status,
        "generic_advice_penalty_applied": generic_penalty_applied,
        "near_cap_penalty_applied": near_cap_penalty_applied,
        "unfinished_near_cap_penalty_applied": unfinished_near_cap_penalty_applied,
        "completion_char_length": completion_char_length,
        "max_completion_length": max_completion_length,
        "reward_accuracy": round(reward_accuracy, 4),
        "reward_format": round(reward_format, 4),
        "reward_language": round(reward_language, 4),
        "reward_reasoning": round(reward_reasoning, 4),
        "reward_length": round(reward_length, 4),
        "reward_total": round(reward_accuracy + reward_format + reward_language + reward_reasoning + reward_length, 4),
    }


def _value_at(values: Any, index: int) -> Any:
    if isinstance(values, list):
        return values[index] if index < len(values) else None
    return values


def build_reward_debug_batch(completions: List[str], label: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
    global _LAST_REWARD_DEBUG_BATCH

    batch: List[Dict[str, Any]] = []
    for idx, (completion, gt) in enumerate(zip(completions, label)):
        context = {
            "prompt": _value_at(kwargs.get("prompt"), idx),
            "transcript_a": _value_at(kwargs.get("transcript_a"), idx),
            "transcript_b": _value_at(kwargs.get("transcript_b"), idx),
            "lyrics_a": _value_at(kwargs.get("lyrics_a"), idx),
            "lyrics_b": _value_at(kwargs.get("lyrics_b"), idx),
            "audio_summary_a": _value_at(kwargs.get("audio_summary_a"), idx),
            "audio_summary_b": _value_at(kwargs.get("audio_summary_b"), idx),
            "max_completion_length": _value_at(kwargs.get("max_completion_length"), idx),
        }
        batch.append(build_reward_debug_entry(completion, gt, **context))

    _LAST_REWARD_DEBUG_BATCH = batch
    return batch


def get_last_reward_debug_batch() -> List[Dict[str, Any]]:
    return [dict(row) for row in _LAST_REWARD_DEBUG_BATCH]


def accuracy_reward(completions: List[str], label: List[str], **kwargs: Any) -> List[float]:
    batch = build_reward_debug_batch(completions, label, **kwargs)
    return [row["reward_accuracy"] for row in batch]


def format_reward(completions: List[str], label: List[str], **kwargs: Any) -> List[float]:
    batch = build_reward_debug_batch(completions, label, **kwargs)
    return [row["reward_format"] for row in batch]


def language_reward(completions: List[str], label: List[str], **kwargs: Any) -> List[float]:
    batch = build_reward_debug_batch(completions, label, **kwargs)
    return [row["reward_language"] for row in batch]


def reasoning_reward(completions: List[str], label: List[str], **kwargs: Any) -> List[float]:
    batch = build_reward_debug_batch(completions, label, **kwargs)
    return [row["reward_reasoning"] for row in batch]


def length_reward(completions: List[str], label: List[str], **kwargs: Any) -> List[float]:
    batch = build_reward_debug_batch(completions, label, **kwargs)
    return [row["reward_length"] for row in batch]


accuracy_reward.__name__ = "accuracy"
format_reward.__name__ = "format"
language_reward.__name__ = "language"
reasoning_reward.__name__ = "reasoning"
length_reward.__name__ = "length"

REWARD_FUNCS = [accuracy_reward, format_reward, language_reward, reasoning_reward, length_reward]


def smoke_reward_func(completions: List[str], label: List[str], **kwargs: Any) -> List[float]:
    batch = build_reward_debug_batch(completions, label, **kwargs)
    return [row["reward_total"] for row in batch]
