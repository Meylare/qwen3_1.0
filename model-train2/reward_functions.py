from __future__ import annotations
import re
from typing import Any, List
from swift.rewards.orm import ORM

_THINK_RE  = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_CHOICE_RE = re.compile(r"ВЫБОР\s*:\s*([AB])", re.IGNORECASE)
_QUALITY_SIGNALS = [
    r"крючок|hook|первые\s*\d+\s*сек",
    r"монтаж|темп|ритм|переход",
    r"эмоц|реакц|вовлечен",
    r"удержан|retention|досмотр",
    r"аудитор|канал|контекст|автор",
]

def _extract_thinking(text):
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else None

def _extract_choice(text):
    after_think = _THINK_RE.sub("", text).strip()
    m = _CHOICE_RE.search(after_think)
    return m.group(1).upper() if m else None

class ViralityAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs):
        rewards = []
        for completion, gt in zip(completions, solution):
            choice = _extract_choice(completion)
            rewards.append(1.0 if choice and choice == gt.upper() else 0.0)
        return rewards

class ViralityFormat(ORM):
    def __call__(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            score = 0.0
            thinking = _extract_thinking(completion)
            if thinking and len(thinking) >= 50:
                score += 0.3
            after_think = _THINK_RE.sub("", completion).strip()
            m = _CHOICE_RE.search(after_think)
            if m:
                score += 0.4
                if len(after_think[m.end():].strip()) < 20:
                    score += 0.3
            rewards.append(round(score, 3))
        return rewards

class ViralityCalibration(ORM):
    def __call__(self, completions, views_a=None, views_b=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            thinking = _extract_thinking(completion)
            if not thinking:
                rewards.append(0.0)
                continue
            coverage = sum(0.12 for p in _QUALITY_SIGNALS if re.search(p, thinking.lower()))
            coverage = min(coverage, 0.6)
            cal = 0.2
            va = views_a[i] if views_a else None
            vb = views_b[i] if views_b else None
            if va and vb and va > 0 and vb > 0:
                ratio = max(va, vb) / min(va, vb)
                n = len(thinking.split())
                if ratio >= 5.0:   cal = 0.4 if n < 400 else 0.1
                elif ratio >= 2.0: cal = 0.4 if 100 < n < 600 else 0.2
                else:              cal = 0.4 if n > 200 else 0.1
            rewards.append(round(coverage + cal, 3))
        return rewards
