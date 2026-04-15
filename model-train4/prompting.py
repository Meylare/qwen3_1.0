"""
Shared prompt construction helpers for the Qwen3.5 GSPO smoke stack.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


def build_system_prompt(max_think_tokens: int = 2500) -> str:
    return f"""You are Slon Producer, an AI assistant for content creators. Your task is to look at two Instagram videos from the same creator, understand which one most likely got more views, and then help the weaker video become stronger.

Work like a strong producer, not like a generic critic. Look at how the video captures attention, how information flows through it, how interest is held, where trust appears, how clearly the message is understood, and how the visuals and audio guide the viewer from the first seconds to the end. The videos themselves are the main source of truth. The transcripts, lyrics, and music summary are supporting context that help you read what is happening more precisely.

The reply has three parts, and it should feel like one continuous answer that changes form at the right moment:
<think>Grounded comparative reasoning about which video got more views, why, and what the losing video should improve</think>
<winner>A</winner>
or
<winner>B</winner>
<advice>The advice itself</advice>

Think of it like this: first the line of thought lives inside <think>. When that thought is complete, </think> is closed, and only after that does the final choice appear inside <winner>. After that, the answer naturally moves into <advice>. The closing </advice> token should be the final token of the whole reply.

Keep all reasoning and comparison inside <think>. In <winner>, write only A or B. In <advice>, speak politely and directly to the creator of the losing video and make the advice practical. The advice must be in the same language as the losing video's speech transcript. If there is no speech transcript, use the language of the creator context.

Use only the evidence in the prompt. Do not invent scenes, dialogue, lyrics, brands, or outcomes. Do not switch into hypothetical examples or placeholder reasoning. Do not say the videos or transcripts were not provided when they are present. Keep the thinking detailed enough to be useful, but grounded. The whole reply should stay under about 3900 tokens total, and the thinking portion should stay comfortably below the technical response limit of about {max_think_tokens} tokens.
"""


def build_answer_only_system_prompt() -> str:
    return """You are Slon Producer, an AI assistant for content creators. You will compare two Instagram videos from the same creator and decide which one most likely received more views.

Use the videos as the main evidence. The transcripts, lyrics, and music summaries are supporting context that can help you read the videos more accurately.

Return exactly one uppercase letter: A or B.
Do not output XML tags, punctuation, whitespace-only lines, explanations, analysis, or any other text.
Your entire response must be exactly one token-long choice: A or B.
"""


def deterministic_flip(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Match the existing training behavior: A/B is deterministically flipped by index."""
    rng = random.Random(idx)
    flipped = dict(item)
    if rng.random() < 0.5:
        flipped["video_a"], flipped["video_b"] = item["video_b"], item["video_a"]
        flipped["views_a"], flipped["views_b"] = item["views_b"], item["views_a"]
        flipped["transcript_a"], flipped["transcript_b"] = item["transcript_b"], item["transcript_a"]
        flipped["transcript_segments_a"], flipped["transcript_segments_b"] = (
            item.get("transcript_segments_b", []),
            item.get("transcript_segments_a", []),
        )
        flipped["lyrics_a"], flipped["lyrics_b"] = item.get("lyrics_b", ""), item.get("lyrics_a", "")
        flipped["lyrics_segments_a"], flipped["lyrics_segments_b"] = (
            item.get("lyrics_segments_b", []),
            item.get("lyrics_segments_a", []),
        )
        flipped["audio_summary_a"], flipped["audio_summary_b"] = item["audio_summary_b"], item["audio_summary_a"]
    return flipped


def build_user_text(item: Dict[str, Any]) -> str:
    author_context = item.get("author_context", "").strip() or "Контекст автора неизвестен."

    transcript_a = item.get("transcript_a", "").strip() or "Речь не обнаружена."
    transcript_b = item.get("transcript_b", "").strip() or "Речь не обнаружена."
    lyrics_a = item.get("lyrics_a", "").strip() or "Текст песни не обнаружен."
    lyrics_b = item.get("lyrics_b", "").strip() or "Текст песни не обнаружен."
    audio_summary_a = item.get("audio_summary_a", "").strip() or "Музыка не обнаружена."
    audio_summary_b = item.get("audio_summary_b", "").strip() or "Музыка не обнаружена."

    return f"""The creator asks: "How can I improve my video and get more views?"

You are comparing two actual candidate videos from the same creator. One of them received significantly more views than the other.

Creator context:
{author_context}

Video A metadata:
Speech transcript A:
{transcript_a}

Song lyrics A:
{lyrics_a}

Music summary A:
{audio_summary_a}

Video B metadata:
Speech transcript B:
{transcript_b}

Song lyrics B:
{lyrics_b}

Music summary B:
{audio_summary_b}
"""


def build_prompt(item: Dict[str, Any], fps: float = 2.0, max_think_tokens: int = 2500) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": build_system_prompt(max_think_tokens=max_think_tokens)}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": item["video_a"], "fps": fps},
                {"type": "video", "video": item["video_b"], "fps": fps},
                {"type": "text", "text": build_user_text(item)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<think>"}],
        },
    ]


def build_answer_only_prompt(item: Dict[str, Any], fps: float = 2.0) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": build_answer_only_system_prompt()}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": item["video_a"], "fps": fps},
                {"type": "video", "video": item["video_b"], "fps": fps},
                {"type": "text", "text": build_user_text(item)},
            ],
        },
    ]
