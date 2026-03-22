"""
reward_functions.py — reward-функции для задачи предсказания виральности видео.

Подключение в скрипте обучения:
    --reward_funcs_path reward_functions.py
    --reward_funcs virality_accuracy virality_format virality_calibration
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_RE   = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_CHOICE_RE  = re.compile(r"ВЫБОР\s*:\s*([AB])", re.IGNORECASE)

# Ключевые аспекты разбора, которые хотим видеть в reasoning
_QUALITY_SIGNALS = [
    r"крючок|hook|первые\s*\d+\s*сек",        # разбор хука
    r"монтаж|темп|ритм|переход",               # динамика
    r"эмоц|реакц|вовлечен",                    # эмоциональность
    r"удержан|retention|досмотр",              # retention
    r"аудитор|канал|контекст|автор",           # соответствие каналу
]


def _extract_thinking(text: str) -> str | None:
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_choice(text: str) -> str | None:
    """Извлекает финальный выбор A или B после </think>."""
    after_think = _THINK_RE.sub("", text).strip()
    m = _CHOICE_RE.search(after_think)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# 1. Accuracy — правильно ли выбрано видео
# ---------------------------------------------------------------------------

def virality_accuracy(
    completions: list[str],
    solution: list[str],
    **kwargs: Any,
) -> list[float]:
    """
    1.0 — модель выбрала правильное видео
    0.0 — неправильное или не смогла распознать ответ
    """
    rewards: list[float] = []
    for completion, gt in zip(completions, solution):
        choice = _extract_choice(completion)
        if choice is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if choice == gt.upper() else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# 2. Format — структура ответа
# ---------------------------------------------------------------------------

def virality_format(
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """
    Проверяет корректность структуры ответа:
      +0.3  наличие непустого <think>...</think>
      +0.4  наличие строки "ВЫБОР: A" или "ВЫБОР: B" после </think>
      +0.3  ничего лишнего после строки ВЫБОР (модель не "разговорчива")
    """
    rewards: list[float] = []
    for completion in completions:
        score = 0.0
        thinking = _extract_thinking(completion)

        if thinking and len(thinking) >= 50:
            score += 0.3

        after_think = _THINK_RE.sub("", completion).strip()
        choice_match = _CHOICE_RE.search(after_think)
        if choice_match:
            score += 0.4
            tail = after_think[choice_match.end():].strip()
            if len(tail) < 20:
                score += 0.3

        rewards.append(round(score, 3))
    return rewards


# ---------------------------------------------------------------------------
# 3. Calibration — глубина и соразмерность анализа
# ---------------------------------------------------------------------------

def virality_calibration(
    completions: list[str],
    views_a: list[int] | None = None,
    views_b: list[int] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Оценивает качество reasoning по двум осям:

    A. Покрытие аспектов (0.0–0.6):
       Сколько из 5 ключевых аспектов (хук, монтаж, эмоции,
       retention, аудитория) упомянуто в <think>.
       Каждый аспект = +0.12

    B. Калибровка уверенности (0.0–0.4):
       Если разрыв в просмотрах большой (ratio >= 5x) — ожидаем
       короткое уверенное рассуждение.
       Если разрыв маленький (ratio < 2x) — ожидаем длинный анализ.
       Штрафует "уверенность без оснований" и "сомнения на ровном месте".

    views_a/views_b передаются ms-swift автоматически из одноимённых
    колонок датасета.
    """
    rewards: list[float] = []

    for i, completion in enumerate(completions):
        thinking = _extract_thinking(completion)
        if not thinking:
            rewards.append(0.0)
            continue

        # --- A. Покрытие аспектов ---
        coverage_score = 0.0
        text_lower = thinking.lower()
        for pattern in _QUALITY_SIGNALS:
            if re.search(pattern, text_lower):
                coverage_score += 0.12
        coverage_score = min(coverage_score, 0.6)

        # --- B. Калибровка уверенности ---
        calibration_score = 0.2   # нейтральный дефолт
        va = (views_a[i] if views_a else None)
        vb = (views_b[i] if views_b else None)

        if va is not None and vb is not None and va > 0 and vb > 0:
            hi, lo = max(va, vb), min(va, vb)
            ratio = hi / lo
            think_len = len(thinking.split())

            if ratio >= 5.0:
                calibration_score = 0.4 if think_len < 400 else 0.1
            elif ratio >= 2.0:
                calibration_score = 0.4 if 100 < think_len < 600 else 0.2
            else:
                calibration_score = 0.4 if think_len > 200 else 0.1

        rewards.append(round(coverage_score + calibration_score, 3))

    return rewards