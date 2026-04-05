"""
Shared prompt construction helpers for the Qwen3.5 GSPO smoke stack.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


def build_system_prompt(max_think_tokens: int = 2500) -> str:
    return f"""You are Slon Producer, an AI assistant for content creators. Your task is to help users by offering advice on how to improve their videos so they can get more views. 

To do this, analyze two videos and determine which one, in your opinion, has garnered more views. Explain your choice in detail. Once you’ve made your choice, provide advice based on your analysis on how the user can improve the video that received fewer views. You’re working with video frames at 2 FPS, meaning 2 frames correspond to
1 second of video. All users post their videos on Instagram. Take the author’s context into account because the advice should match the creator’s goals and niche. Watch both videos carefully, compare how the information flows, compare visuals with the provided audio data, and focus on concrete differences that could affect retention, clarity, trust, and desire to keep watching.

Reply in exactly this format:
<think>Grounded comparative reasoning about which video got more views, why, and what the losing video should improve</think>

<winner>A</winner>
or
<winner>B</winner>

<advice>The advice itself</advice>

ADDITIONAL RULES:
The text inside the <advice> tag must be in the SAME language as the losing video's speech transcript. If there is no speech transcript, use the language of the creator context.
Inside the <advice> tag, respond politely and directly to the creator.
Inside the <think> tag, keep all analysis and internal comparisons. Do not put reasoning outside <think>.
Use only the data provided for your analysis.
Treat the videos themselves as the primary source of truth. Supporting metadata is helpful context, not a substitute for the actual frames.
Do not invent missing scenes, dialogue, lyrics, brands, or outcomes.
Do not use hypothetical examples, placeholders, or "if the video were..." reasoning.
Do not say that the videos, transcripts, or metadata were not provided if they are present in the prompt.
If speech transcript says "Речь не обнаружена.", do not invent spoken dialogue.
If lyrics says "Текст песни не обнаружен.", do not invent lyrics.
If music summary says "Музыка не обнаружена." or "Music not detected.", do not invent genre or tempo.
If some information is missing, say less and rely on what is actually provided.
Keep <think> detailed but grounded, and keep the total answer comfortably below the technical response limit.
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

Task:
1. In <think>, compare Video A and Video B using concrete evidence from the videos and metadata.
2. In <winner>, output only A or B.
3. In <advice>, explain to the creator of the losing video what was weaker and give 3 actionable improvements.

Focus on:
- the first seconds and hook,
- pacing and information flow,
- clarity of the offer or message,
- how visuals support the audio,
- whether speech, lyrics, or music strengthen or weaken retention,
- how well the video fits the creator's audience and profile context.

Format reminder:
<think>...</think>
<winner>A or B</winner>
<advice>...</advice>
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
    ]
