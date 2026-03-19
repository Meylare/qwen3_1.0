"""
reward.py — функции награды для GRPO.

Логика награды:
  +1.0   — правильно выбрала вирусное видео
  -1.0   — выбрала не то видео
  -1.0   — не дала ответ в правильном формате
  +0.10  — бонус за наличие обоих тегов <think> и <answer>
  -0..0.25 — мягкий штраф за слишком длинный thinking

Философия: штраф за длину мягкий и нарастающий (не обрубаем мысль),
но модель должна понять, что краткость ценится.
"""
import math
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Паттерны для парсинга
ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def parse_answer(completion: str) -> Optional[str]:
    """
    Извлекает 'A' или 'B' из <answer> тега.

    Только строгий match по тегу — без fallback на свободный текст.

    ВАЖНО: сначала вырезаем <think>...</think> блок, затем ищем <answer>.
    Без этого re.search находит первый <answer> в тексте — который может
    стоять внутри <think> (например модель рассуждает "если бы ответ был
    <answer>A</answer>..."). Это давало неверный результат.
    """
    # Вырезаем think блок перед поиском answer
    text_without_think = THINK_RE.sub("", completion)
    match = ANSWER_RE.search(text_without_think)
    if match:
        return match.group(1).upper()
    return None


def get_thinking_token_count(completion: str, tokenizer=None) -> int:
    """
    Подсчёт токенов в <think> блоке.

    Если передан tokenizer — используем его (точный подсчёт, работает для кириллицы).
    Fallback: 1 токен ≈ 4 символа (только для английского текста — занижает в 2-3 раза
    для кириллицы, что приводит к отсутствию штрафа за длинный thinking на русском).
    """
    match = THINK_RE.search(completion)
    if not match:
        return 0
    text = match.group(1)
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text) // 4  # fallback: только для en-текста


def has_proper_format(completion: str) -> bool:
    """Проверяем наличие обоих тегов."""
    has_think = bool(THINK_RE.search(completion))
    has_answer = bool(ANSWER_RE.search(completion))
    return has_think and has_answer


def compute_length_penalty(
    thinking_tokens: int,
    max_tokens: int = 400,
    penalty_max: float = 0.25,
) -> float:
    """
    Мягкий нарастающий штраф за длинный thinking.

    При thinking_tokens <= max_tokens: штрафа нет.
    При thinking_tokens = 2 * max_tokens: штраф ≈ 0.19  (penalty_max * tanh(1) ≈ 0.762)
    При thinking_tokens = 5 * max_tokens: штраф ≈ 0.25  (penalty_max * tanh(4) ≈ 0.999)

    Функция: penalty = penalty_max * tanh(excess / max_tokens)

    tanh асимптотически приближается к penalty_max, но никогда его не достигает
    на конечном excess — "максимальный штраф" наступает практически при ~5x overflow.
    """
    if thinking_tokens <= max_tokens:
        return 0.0
    excess = thinking_tokens - max_tokens
    # tanh плавно насыщается → не обрубает мысль резко
    penalty = penalty_max * math.tanh(excess / max_tokens)
    return penalty


def virality_reward_fn(
    completions: List[str],
    # Поля из датасета (TRL передаёт как kwargs если они есть в батче)
    labels_text: List[str],
    # Конфиги (передаются через functools.partial в train.py)
    correct_reward: float = 1.0,
    wrong_reward: float = -1.0,
    no_answer_penalty: float = -1.0,
    format_bonus: float = 0.1,
    thinking_max_tokens: int = 400,
    length_penalty_max: float = 0.25,
    tokenizer=None,   # передаётся из train.py через partial для точного подсчёта токенов
    **kwargs,  # поглощаем лишние kwargs от TRL
) -> List[float]:
    """
    Основная функция награды для GRPO.

    Args:
        completions: список сгенерированных текстов (len = batch_size * G)
        labels_text: ground truth ['A', 'B', ...] (len = batch_size * G,
                     TRL повторяет labels для каждого из G сэмплов)

    Returns:
        Список float наград, один на completion.
    """
    rewards = []

    for completion, gt in zip(completions, labels_text):
        predicted = parse_answer(completion)

        # ── 1. Правильность ──────────────────────────────────
        if predicted is None:
            # Нет ответа — максимальный штраф
            reward = no_answer_penalty
            logger.debug(f"No answer found in completion (len={len(completion)})")
        elif predicted == gt:
            reward = correct_reward
        else:
            reward = wrong_reward

        # ── 2. Бонус за формат ───────────────────────────────
        # Даём бонус только если ответ присутствует
        # (чтобы не поощрять формат без смысла)
        if predicted is not None and has_proper_format(completion):
            reward += format_bonus

        # ── 3. Штраф за длинный thinking ─────────────────────
        thinking_len = get_thinking_token_count(completion, tokenizer=tokenizer)
        penalty = compute_length_penalty(
            thinking_len,
            max_tokens=thinking_max_tokens,
            penalty_max=length_penalty_max,
        )
        reward -= penalty

        rewards.append(float(reward))

    return rewards