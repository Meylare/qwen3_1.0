"""
Shared prompt construction helpers for the Qwen3.5 GSPO smoke stack.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


SYSTEM_PROMPT = """You are an expert at predicting short-form video virality.

You will see two videos from the same creator, plus creator context, full speech transcripts,
and an audio summary for each clip. Compare the two videos as complete packages:
visual hook, pacing, editing rhythm, clarity of message, speech delivery, soundtrack, emotional pull,
and the fit between the content and the creator's audience.

Reason carefully and comparatively. Do not give generic social-media tips. Focus on what is actually
observable in the provided videos and side information.

Always respond exactly in this format:
<think>
...
</think>
<answer>A</answer>
or
<answer>B</answer>"""


def deterministic_flip(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Match the existing training behavior: A/B is deterministically flipped by index."""
    rng = random.Random(idx)
    flipped = dict(item)
    if rng.random() < 0.5:
        flipped["video_a"], flipped["video_b"] = item["video_b"], item["video_a"]
        flipped["views_a"], flipped["views_b"] = item["views_b"], item["views_a"]
        flipped["transcript_a"], flipped["transcript_b"] = item["transcript_b"], item["transcript_a"]
        flipped["audio_summary_a"], flipped["audio_summary_b"] = item["audio_summary_b"], item["audio_summary_a"]
    return flipped


def build_user_text(item: Dict[str, Any]) -> str:
    author_context = item.get("author_context", "").strip() or "Unknown creator context."

    transcript_a = item.get("transcript_a", "").strip() or "No speech transcript available."
    transcript_b = item.get("transcript_b", "").strip() or "No speech transcript available."
    audio_summary_a = item.get("audio_summary_a", "").strip() or "No audio summary available."
    audio_summary_b = item.get("audio_summary_b", "").strip() or "No audio summary available."

    return (
        f"Creator context:\n{author_context}\n\n"
        "Video A metadata:\n"
        f"Transcript A:\n{transcript_a}\n\n"
        f"Audio summary A:\n{audio_summary_a}\n\n"
        "Video B metadata:\n"
        f"Transcript B:\n{transcript_b}\n\n"
        f"Audio summary B:\n{audio_summary_b}\n\n"
        "Task:\n"
        "Compare Video A and Video B from the same creator and decide which one is more likely "
        "to have received significantly more views. Base the answer on the actual comparative strengths "
        "and weaknesses of the two videos. End with a single final choice."
    )


def build_prompt(item: Dict[str, Any], fps: float = 2.0) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": item["video_a"], "fps": fps},
                {"type": "video", "video": item["video_b"], "fps": fps},
                {"type": "text", "text": build_user_text(item)},
            ],
        },
    ]
