"""
prepare_dataset.py — подготовка датасета пар видео для GSPO-обучения.

Вход  : JSONL с полями video_a, video_b, views_a, views_b, author_context
Выход : data/train.jsonl и data/val.jsonl в формате ms-swift

Использование:
    python prepare_dataset.py \
        --input raw_pairs.jsonl \
        --output_dir data/ \
        --val_ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

# ---------------------------------------------------------------------------
# System-prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Ты эксперт по анализу виральности коротких видео.

Тебе дают два видео от одного автора и контекст о его канале.
Твоя задача — определить, какое видео наберёт больше просмотров, и объяснить почему.

Порядок ответа:
1. Внутри <think>...</think> детально разбери оба видео по критериям:
   - Крючок (первые 1–3 секунды): насколько цепляет?
   - Динамика и монтаж: темп, переходы, ритм
   - Эмоциональный отклик: вызывает ли видео реакцию?
   - Соответствие аудитории автора и контексту канала
   - Технические характеристики: качество, звук, субтитры
   - Потенциал удержания (retention): хочется ли досмотреть?

2. После </think> дай финальный ответ строго в формате:
   ВЫБОР: A
   или
   ВЫБОР: B

Никакого другого текста после </think>, только строка ВЫБОР: A или ВЫБОР: B."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_path(p: str) -> str:
    """Нормализует Windows-пути в Unix для кросс-платформенности."""
    return p.replace("\\", "/")


def _confidence_label(views_a: int, views_b: int) -> str:
    """Метка уверенности для отладки — не используется в обучении."""
    hi, lo = max(views_a, views_b), min(views_a, views_b)
    ratio = hi / lo if lo > 0 else float("inf")
    if ratio >= 10: return "high"
    if ratio >= 3:  return "medium"
    return "low"


def _make_example(row: dict, swap: bool = False) -> dict:
    """
    Формирует один пример датасета.

    swap=True: меняем A и B местами для аугментации
    (позиционное смещение — модель не должна всегда выбирать A).
    """
    va = _fix_path(row["video_a"])
    vb = _fix_path(row["video_b"])
    views_a: int = row["views_a"]
    views_b: int = row["views_b"]
    context: str = row.get("author_context", "")

    if swap:
        va, vb = vb, va
        views_a, views_b = views_b, views_a

    # Ground truth: какое видео победило?
    winner = "A" if views_a > views_b else "B"

    user_text = (
        f"Контекст канала: {context}\n\n"
        f"Видео A: <video>{va}</video>\n"
        f"Видео B: <video>{vb}</video>\n"
        f"Какое видео наберёт больше просмотров — A или B?"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    return {
        "messages": messages,
        "solution": winner,
        "task_type": "video_virality",
        "views_a": views_a,                          # нужны virality_calibration reward
        "views_b": views_b,                          # нужны virality_calibration reward
        "_confidence": _confidence_label(views_a, views_b),   # только для аналитики
        "_swapped": swap,                                      # только для аналитики
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(
    input_path: str,
    output_dir: str,
    val_ratio: float = 0.1,
    augment_swap: bool = True,
    seed: int = 42,
) -> None:
    base_dir = pathlib.Path(__file__).resolve().parent
    input_path_p = pathlib.Path(input_path)
    if not input_path_p.is_absolute():
        input_path_p = base_dir / input_path_p
    output_dir_p = pathlib.Path(output_dir)
    if not output_dir_p.is_absolute():
        output_dir_p = base_dir / output_dir_p

    random.seed(seed)

    raw: list[dict] = []
    with open(input_path_p) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                raw.append(row)
            except json.JSONDecodeError as e:
                print(f"  [пропуск строки {i}] {e}")

    print(f"Загружено: {len(raw)} пар")

    # Формирование примеров
    examples: list[dict] = []
    for row in raw:
        examples.append(_make_example(row, swap=False))
        if augment_swap:
            examples.append(_make_example(row, swap=True))

    # Перемешивание
    random.shuffle(examples)

    # Разбивка
    n_val = max(1, int(len(examples) * val_ratio))
    val_examples   = examples[:n_val]
    train_examples = examples[n_val:]

    # Статистика по уверенности
    for split_name, split in [("train", train_examples), ("val", val_examples)]:
        from collections import Counter
        conf_counts = Counter(e["_confidence"] for e in split)
        print(f"{split_name}: {len(split)} примеров | confidence: {dict(conf_counts)}")

    # Запись
    output_dir_p.mkdir(parents=True, exist_ok=True)

    def _write(path: pathlib.Path, items: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                # удаляем мета-поля перед записью (ms-swift их игнорирует, но чище без них)
                clean = {k: v for k, v in item.items() if not k.startswith("_")}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        print(f"  Записано: {path}  ({len(items)} строк)")

    _write(output_dir_p / "train.jsonl", train_examples)
    _write(output_dir_p / "val.jsonl",   val_examples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",        default="raw_pairs.jsonl")
    ap.add_argument("--output_dir",   default="data/")
    ap.add_argument("--val_ratio",    type=float, default=0.1)
    ap.add_argument("--no_swap",      action="store_true",
                    help="Отключить аугментацию зеркальными парами")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    build_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        augment_swap=not args.no_swap,
        seed=args.seed,
    )