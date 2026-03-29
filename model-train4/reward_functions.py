"""
Minimal smoke reward for pairwise virality training.
"""

from __future__ import annotations

import re
from typing import List, Optional


ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def parse_answer(text: str) -> Optional[str]:
    text_wo_think = THINK_RE.sub("", text)
    match = ANSWER_RE.search(text_wo_think)
    return match.group(1).upper() if match else None


def has_required_format(text: str) -> bool:
    return bool(THINK_RE.search(text)) and bool(ANSWER_RE.search(text))


def smoke_reward_func(completions: List[str], label: List[str], **_: object) -> List[float]:
    rewards: List[float] = []
    for completion, gt in zip(completions, label):
        predicted = parse_answer(completion)
        if predicted is None:
            reward = -1.0
        elif predicted == gt:
            reward = 1.0
        else:
            reward = -1.0

        if predicted is not None and has_required_format(completion):
            reward += 0.1

        rewards.append(float(reward))
    return rewards
