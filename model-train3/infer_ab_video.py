#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


def require_file(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return str(p.resolve())


def main() -> None:
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-27B")

    video_a = require_file(
        os.getenv("VIDEO_A", "/home/ubuntu/model-train/data/pair_vid/9e24c4a58a0964a5.mp4")
    )
    video_b = require_file(
        os.getenv("VIDEO_B", "/home/ubuntu/model-train/data/pair_vid/9e129b58b17e2d25.mp4")
    )

    prompt_text = os.getenv(
        "PROMPT_TEXT",
        "Контекст канала: Сеть пиццерий в Казахстане. "
        "Видео A и Видео B от одного автора. "
        "Какое видео наберет больше просмотров и почему? "
        "Сначала краткий анализ, затем финал строго в формате: ВЫБОР: A или ВЫБОР: B.",
    )

    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("MAX_TOKENS", "512"))

    client = OpenAI(base_url=base_url, api_key=api_key)

    content = [
        {"type": "video_url", "video_url": {"url": f"file://{video_a}"}},
        {"type": "video_url", "video_url": {"url": f"file://{video_b}"}},
        {"type": "text", "text": prompt_text},
    ]

    started_at = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_s = time.perf_counter() - started_at

    text = response.choices[0].message.content if response.choices else ""
    usage = response.usage.model_dump() if response.usage else {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    completion_tps = (
        round(completion_tokens / latency_s, 3)
        if isinstance(completion_tokens, int) and latency_s > 0
        else None
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "video_a": video_a,
        "video_b": video_b,
        "prompt": prompt_text,
        "answer": text,
        "usage": usage or None,
        "metrics": {
            "latency_s": round(latency_s, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "completion_tokens_per_s": completion_tps,
        },
    }

    output_path = Path(os.getenv("OUTPUT_JSON", "./output/infer_ab_video_result.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(text)
    print(f"\nSaved result to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
