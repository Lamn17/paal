"""Evaluation metrics and label-efficiency summaries."""

from __future__ import annotations

import numpy as np


def top_k_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean((top == np.asarray(y)[:, None]).any(axis=1)))


def per_class_min_queries(labels: np.ndarray, classes: np.ndarray) -> int:
    """Fewest labeled samples of any class in ``classes`` (Soltani et al., Fig. 6)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(np.max(classes)) + 1)
    return int(counts[np.asarray(classes)].min())


def labels_to_reach(n_labels: np.ndarray, accuracy: np.ndarray, target: float) -> float:
    """First labeled-set size at which ``accuracy`` reaches ``target`` (inf if never)."""
    hit = np.flatnonzero(np.asarray(accuracy) >= target)
    return float(np.asarray(n_labels)[hit[0]]) if len(hit) else float("inf")
