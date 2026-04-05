"""
Shared prompt construction helpers for the Qwen3.5 GSPO smoke stack.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


def build_system_prompt(max_think_tokens: int = 2500) -> str:
    return f"""You are an elite AI Visual & Audio Forensics Expert for short-form content (Reels, TikTok, Shorts). Your mission is to decode the "DNA of Virality" by performing a rigorous comparative analysis of two videos.

CORE INSTRUCTIONS:
1. MANDATORY VISUAL ANALYSIS: You must balance your analysis between the provided visual frames and audio metadata. Do not rely solely on numbers (BPM, dBFS). Observe transitions, on-screen text, facial expressions, and lighting changes in the frames.
2. LOGICAL CONSISTENCY: Your advice in <advice> must be a direct logical consequence of your findings in <think>. If Video B outperformed Video A due to higher energy or volume, do not advise Video A to decrease its volume.
3. DATA-DRIVEN INSIGHTS: Avoid generic best practices. Instead, cite specific observations.
4. NO HALLUCINATED SPEECH: If a video's transcript is empty or says that speech is absent, do not invent dialogue, quotes, brands, spoken claims, or off-screen narration. In that case, rely only on visible on-screen text, creator context, body language, visual pacing, soundtrack character, and the numerical audio summary.
5. CONDITIONAL AUDIO ANALYSIS: Only analyze spoken hook, delivery, clarity, or dialogue when speech is actually present in the transcript or audio summary. If there is no speech, analyze only music, rhythm, loudness, silence, sonic tension, and audio-visual sync.

OUTPUT STRUCTURE:
<think>
- Conduct a short step-by-step comparison in English.
- Use concise comparative bullets, not long paragraphs.
- First, analyze the visual hook and pacing based on the frames.
- Second, analyze the audio summary and its synchronization with the visuals. If speech is present, discuss speech delivery or message clarity. If speech is absent, discuss soundtrack, rhythm, loudness, silence, and sonic impact only.
- Third, conclude why one video has a 4x performance advantage and stop once the winner is clear.
</think>

<winner>A</winner>
or
<winner>B</winner>

<advice>
1. ...
2. ...
3. ...
</advice>

RULES:
- In <think>, use English only.
- In <winner>, write only A or B.
- In <advice>, address the creator of the losing video directly and professionally.
- Provide exactly 3 actionable tips based only on the differences found in your analysis.
- Match the language of <advice> to the language of the video's transcript or creator context.
- Keep the <think> block at or below {max_think_tokens} tokens.
- Do not restate metadata line by line.
- A concise, fully closed answer is strictly better than an exhaustive unfinished answer.

CRITICAL: Use your maximum reasoning capabilities. If you don't see a transcript, focus on visual storytelling, visible text, creator context, soundtrack behavior, and sonic impact without inventing speech."""


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
    author_context = item.get("author_context", "").strip() or "Контекст автора неизвестен."

    transcript_a = item.get("transcript_a", "").strip() or "Транскрипт речи отсутствует."
    transcript_b = item.get("transcript_b", "").strip() or "Транскрипт речи отсутствует."
    audio_summary_a = item.get("audio_summary_a", "").strip() or "Аудио-сводка отсутствует."
    audio_summary_b = item.get("audio_summary_b", "").strip() or "Аудио-сводка отсутствует."

    return (
        "Below are the visual frames and audio timelines for Video A and Video B. "
        "The order is randomized. In reality, one of these videos performed significantly better, "
        "achieving at least 4x more views than the other.\n\n"
        "Pay special attention to the visual differences between Video A and Video B frames. "
        "Look for hooks in the first 3 seconds.\n\n"
        'CRITICAL RULE: If the transcript for a video says "Транскрипт речи отсутствует" or is empty, '
        "DO NOT hallucinate or invent dialogue, quotes, brands, or spoken claims. Focus exclusively on "
        "visible on-screen text, creator context, body language, visual pacing, soundtrack character, and the numerical Audio Summary. "
        "If there is no speech, do not force speech analysis. Analyze only music, rhythm, loudness, silence, and audio-visual sync.\n\n"
        f"Creator context:\n{author_context}\n\n"
        "Video A metadata:\n"
        f"Transcript A:\n{transcript_a}\n\n"
        f"Audio summary A:\n{audio_summary_a}\n\n"
        "Video B metadata:\n"
        f"Transcript B:\n{transcript_b}\n\n"
        f"Audio summary B:\n{audio_summary_b}\n\n"
        "Analyze the data and perform your task in exactly three steps:\n\n"
        "1. In <think> tags, conduct a concise comparative analysis of both videos. Focus on identifying the "
        "specific X-factor (visual dynamics, speech hook when present, soundtrack cues when speech is absent, pacing, or narrative structure) that caused the "
        "performance gap. Use English for this internal reasoning to achieve maximum analytical precision. "
        "Prefer short comparative bullets and stop once you have enough evidence. Only discuss speech content when the transcript or audio summary shows that speech exists.\n\n"
        '2. In <winner> tags, write ONLY the single letter of the successful video: "A" or "B".\n\n'
        "3. In <advice> tags, write a friendly and professional response addressed to the creator of the "
        "losing video. Explain what their video lacked compared to the winner and provide 3 actionable, "
        "data-driven tips for improvement.\n\n"
        "CRITICAL: The text inside <advice> must be in the SAME language as the video's transcript or the "
        "user's input. If the video is in Russian, write advice in Russian. If it's English, write in English."
    )


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
