"""
dataset_converter.py — конвертация датасета из нашего формата в ms-swift формат.

Наш формат (train_ready.jsonl):
{
    "video_a": "/path/to/video_a.mp4",
    "video_b": "/path/to/video_b.mp4",
    "views_a": 1500000,
    "views_b": 42000,
    "author_context": "фитнес блогер, ..."
}

ms-swift формат для мультимодального GRPO:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": [
            {"type": "video", "video": "/path/to/video_a.mp4"},
            {"type": "video", "video": "/path/to/video_b.mp4"},
            {"type": "text", "text": "Creator profile: ...\n\nVideo A vs Video B..."}
        ]}
    ],
    "label": "A"   ← extra колонка для reward функции
}

Запуск:
    python dataset_converter.py \
        --input ./data/train_ready.jsonl \
        --output ./data/train_swift.jsonl \
        --split train

    python dataset_converter.py \
        --input ./data/train_ready.jsonl \
        --output ./data/eval_swift.jsonl \
        --split eval \
        --eval_ratio 0.1
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

SYSTEM_PROMPT = (
    "You are an expert at predicting social media virality on Instagram. "
    "You will see two short videos from the same creator and must decide which one "
    "received significantly more views. Analyze visual quality, hook strength, "
    "editing pace, audio, content clarity, and emotional resonance. "
    "Be concise in your reasoning — focus only on the factors that truly drive views. "
    "Always respond in the format:\n"
    "<think>\n[your analysis]\n</think>\n"
    "<answer>A</answer>  or  <answer>B</answer>"
)


def convert_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Конвертирует один элемент из нашего формата в ms-swift формат.

    Включает детерминированный A/B флип по idx — тот же что в ViralityDataset.
    """
    import random as _random

    # Детерминированный A/B флип (идентично ViralityDataset.__getitem__)
    rng = _random.Random(idx)
    video_a = item["video_a"]
    video_b = item["video_b"]
    views_a = item["views_a"]
    views_b = item["views_b"]

    if rng.random() < 0.5:
        video_a, video_b = video_b, video_a
        views_a, views_b = views_b, views_a

    label = "A" if views_a > views_b else "B"
    author_context = item.get("author_context", "")

    user_text = (
        f"Creator profile: {author_context}\n\n"
        "Video A (first video above) vs Video B (second video above) — "
        "both from the same creator. Which received significantly more views?\n\n"
        "Think through the key factors, then give your answer."
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    # Видео идут ПЕРЕД текстом — требование Qwen3-Omni
                    {"type": "video", "video": video_a},
                    {"type": "video", "video": video_b},
                    {"type": "text",  "text": user_text},
                ],
            },
        ],
        # Extra колонки — передаются в reward функцию через kwargs
        "label": label,
        "views_a": views_a,
        "views_b": views_b,
        "author_context": author_context,
    }


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: List[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_by_author(
    items: List[Dict],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Разбивает датасет на train/eval без утечки авторов.
    Авторы из eval не должны быть в train (data leakage через стиль автора).
    """
    # Группируем по автору через author_context как прокси
    author_to_indices = {}
    for i, item in enumerate(items):
        author = item.get("author_context", f"unknown_{i}")
        if author not in author_to_indices:
            author_to_indices[author] = []
        author_to_indices[author].append(i)

    authors = list(author_to_indices.keys())
    random.seed(seed)
    random.shuffle(authors)

    n_eval_authors = max(1, int(len(authors) * eval_ratio))
    eval_authors = set(authors[:n_eval_authors])

    train_items = [item for item in items if item.get("author_context", "") not in eval_authors]
    eval_items  = [item for item in items if item.get("author_context", "") in eval_authors]

    return train_items, eval_items


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to ms-swift format")
    parser.add_argument("--input",  required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--split",  choices=["train", "eval", "both"], default="both",
                        help="Which split to output")
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                        help="Fraction of authors for eval split")
    parser.add_argument("--eval_output", default=None,
                        help="Output path for eval split (only used with --split both)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    raw_items = load_jsonl(args.input)
    print(f"Loaded {len(raw_items)} items")

    if args.split == "both":
        train_raw, eval_raw = split_by_author(raw_items, args.eval_ratio, args.seed)
        print(f"Train: {len(train_raw)} items, Eval: {len(eval_raw)} items")

        train_converted = [convert_item(item, idx) for idx, item in enumerate(train_raw)]
        eval_converted  = [convert_item(item, idx) for idx, item in enumerate(eval_raw)]

        write_jsonl(args.output, train_converted)
        print(f"Written train to {args.output}")

        eval_path = args.eval_output or args.output.replace(".jsonl", "_eval.jsonl")
        write_jsonl(eval_path, eval_converted)
        print(f"Written eval to {eval_path}")

    else:
        if args.split == "train":
            train_raw, _ = split_by_author(raw_items, args.eval_ratio, args.seed)
            items_to_convert = train_raw
        else:
            _, eval_raw = split_by_author(raw_items, args.eval_ratio, args.seed)
            items_to_convert = eval_raw

        converted = [convert_item(item, idx) for idx, item in enumerate(items_to_convert)]
        write_jsonl(args.output, converted)
        print(f"Written {len(converted)} items to {args.output}")


if __name__ == "__main__":
    main()