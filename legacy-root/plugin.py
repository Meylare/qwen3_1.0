"""
plugin.py — reward функция для ms-swift GRPO.

Регистрируется через:
    --external_plugins ./plugin.py
    --reward_funcs virality

Логика наград (идентична оригинальному reward.py):
    +1.0   — правильно выбрала вирусное видео
    -1.0   — выбрала не то видео
    -1.0   — не дала ответ в правильном формате
    +0.10  — бонус за наличие обоих тегов <think> и <answer>
    -0..0.25 — мягкий штраф за слишком длинный thinking
"""
import math
import re
import logging
from typing import List, Optional

from swift.plugin import ORM, orms

logger = logging.getLogger(__name__)

# ── Паттерны парсинга ─────────────────────────────────────────────────────────
ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
THINK_RE  = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def parse_answer(completion: str) -> Optional[str]:
    """
    Извлекает 'A' или 'B' из <answer> тега.
    Сначала вырезаем <think> блок — иначе можем найти <answer> внутри него.
    """
    text_without_think = THINK_RE.sub("", completion)
    match = ANSWER_RE.search(text_without_think)
    return match.group(1).upper() if match else None


def has_proper_format(completion: str) -> bool:
    """Проверяем наличие обоих тегов <think> и <answer>."""
    return bool(THINK_RE.search(completion)) and bool(ANSWER_RE.search(completion))


def get_thinking_token_count(completion: str) -> int:
    """
    Подсчёт токенов в <think> блоке.
    Fallback: 1 токен ≈ 4 символа (для английского текста).
    """
    match = THINK_RE.search(completion)
    if not match:
        return 0
    return len(match.group(1)) // 4


def compute_length_penalty(
    thinking_tokens: int,
    max_tokens: int = 400,
    penalty_max: float = 0.25,
) -> float:
    """Мягкий нарастающий штраф за длинный thinking через tanh."""
    if thinking_tokens <= max_tokens:
        return 0.0
    excess = thinking_tokens - max_tokens
    return penalty_max * math.tanh(excess / max_tokens)


class ViralityReward(ORM):
    """
    Reward функция для задачи предсказания виральности видео.

    ms-swift передаёт в kwargs все колонки датасета.
    Нам нужна колонка 'label' — 'A' или 'B' (ground truth).

    Параметры reward можно настраивать через атрибуты класса.
    """

    # Конфиг наград — менять здесь
    CORRECT_REWARD:      float = 1.0
    WRONG_REWARD:        float = -1.0
    NO_ANSWER_PENALTY:   float = -1.0
    FORMAT_BONUS:        float = 0.1
    THINKING_MAX_TOKENS: int   = 400
    LENGTH_PENALTY_MAX:  float = 0.25

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Args:
            completions: список сгенерированных текстов (len = batch * G)
            **kwargs:    все колонки датасета, включая 'label'

        Returns:
            Список float наград, один на completion.
        """
        # ms-swift передаёт колонки датасета как списки той же длины что completions
        labels = kwargs.get("label", [])

        if not labels:
            logger.warning(
                "ViralityReward: 'label' column not found in kwargs. "
                "Check dataset has 'label' column with values 'A' or 'B'."
            )
            return [0.0] * len(completions)

        rewards = []
        for completion, gt in zip(completions, labels):
            predicted = parse_answer(completion)

            # 1. Правильность
            if predicted is None:
                reward = self.NO_ANSWER_PENALTY
            elif predicted == gt:
                reward = self.CORRECT_REWARD
            else:
                reward = self.WRONG_REWARD

            # 2. Бонус за формат (только если ответ есть)
            if predicted is not None and has_proper_format(completion):
                reward += self.FORMAT_BONUS

            # 3. Штраф за длинный thinking
            thinking_len = get_thinking_token_count(completion)
            penalty = compute_length_penalty(
                thinking_len,
                max_tokens=self.THINKING_MAX_TOKENS,
                penalty_max=self.LENGTH_PENALTY_MAX,
            )
            reward -= penalty

            rewards.append(float(reward))

        return rewards


# Регистрируем под именем 'virality' — используется в --reward_funcs virality
orms["virality"] = ViralityReward