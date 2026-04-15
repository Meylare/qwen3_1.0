"""
Shared helpers for pairwise A/B choice training and evaluation.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


LABEL_TO_ID = {"A": 0, "B": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def normalize_choice_label(value: Any) -> str:
    label = str(value).strip().upper()
    if label not in LABEL_TO_ID:
        raise ValueError(f"Expected choice label 'A' or 'B', got {value!r}")
    return label


def label_to_class_id(value: Any) -> int:
    return LABEL_TO_ID[normalize_choice_label(value)]


def class_id_to_label(value: int) -> str:
    if int(value) not in ID_TO_LABEL:
        raise ValueError(f"Expected class id 0 or 1, got {value!r}")
    return ID_TO_LABEL[int(value)]


def get_choice_token_ids(tokenizer: Any) -> Dict[str, int]:
    choice_ids: Dict[str, int] = {}
    for label in ("A", "B"):
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Expected {label!r} to map to exactly one token, got {token_ids}")
        choice_ids[label] = int(token_ids[0])
    return choice_ids


def compute_choice_metrics(logits: Any, labels: Any) -> Dict[str, float]:
    logits_np = np.asarray(logits, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits_np.ndim == 3 and logits_np.shape[1] == 1:
        logits_np = logits_np[:, 0, :]
    if logits_np.ndim != 2 or logits_np.shape[-1] != 2:
        raise ValueError(f"Expected logits shaped [batch, 2], got {logits_np.shape}")
    if labels_np.ndim != 1:
        raise ValueError(f"Expected labels shaped [batch], got {labels_np.shape}")
    if len(logits_np) != len(labels_np):
        raise ValueError(f"Logits/labels length mismatch: {len(logits_np)} vs {len(labels_np)}")
    if len(labels_np) == 0:
        return {"accuracy": 0.0, "mean_correct_probability": 0.0, "nll": 0.0}

    shifted = logits_np - np.max(logits_np, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=-1, keepdims=True)

    predictions = np.argmax(logits_np, axis=-1)
    accuracy = float(np.mean(predictions == labels_np))
    correct_probs = probs[np.arange(len(labels_np)), labels_np]
    nll = float(np.mean(-np.log(np.clip(correct_probs, 1e-12, 1.0))))

    return {
        "accuracy": round(accuracy, 6),
        "mean_correct_probability": round(float(np.mean(correct_probs)), 6),
        "nll": round(nll, 6),
    }
