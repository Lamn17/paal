"""Numerical helpers shared by the query strategies."""

from __future__ import annotations

import numpy as np


def top_b(scores: np.ndarray, b: int) -> np.ndarray:
    """Positions of the ``b`` largest scores (ties broken by position)."""
    return np.argsort(-np.asarray(scores), kind="stable")[: min(int(b), len(scores))]


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=axis)


def sq_dist_to(x: np.ndarray, centers: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Squared Euclidean distance from every row of ``x`` to its nearest row of ``centers``."""
    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    out = np.full(len(x), np.inf, dtype=np.float32)
    x_sq = np.einsum("ij,ij->i", x, x)
    for start in range(0, len(centers), chunk):
        block = centers[start : start + chunk]
        d = x_sq[:, None] + np.einsum("ij,ij->i", block, block)[None, :] - 2.0 * (x @ block.T)
        np.minimum(out, d.min(axis=1), out=out)
    return np.maximum(out, 0.0)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    """Lloyd's k-means with k-means++ seeding; returns the cluster index of every row."""
    x = np.asarray(x, dtype=np.float32)
    k = max(1, min(int(k), len(x)))
    centers = [x[rng.integers(len(x))]]
    dist = sq_dist_to(x, np.asarray(centers))
    for _ in range(1, k):
        total = float(dist.sum())
        pick = rng.choice(len(x), p=dist / total) if total > 0 else int(rng.integers(len(x)))
        centers.append(x[pick])
        dist = np.minimum(dist, sq_dist_to(x, x[pick][None]))
    centers = np.asarray(centers)
    assign = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        assign_new = nearest_center(x, centers)
        if np.array_equal(assign_new, assign) and _ > 0:
            break
        assign = assign_new
        sums = np.zeros_like(centers)
        np.add.at(sums, assign, x)
        counts = np.bincount(assign, minlength=k)[:, None]
        centers = np.where(counts > 0, sums / np.maximum(counts, 1), centers)
    return assign


def nearest_center(x: np.ndarray, centers: np.ndarray, chunk: int = 2048) -> np.ndarray:
    x_sq = np.einsum("ij,ij->i", x, x)
    c_sq = np.einsum("ij,ij->i", centers, centers)
    out = np.empty(len(x), dtype=np.int64)
    for start in range(0, len(x), chunk):
        block = x[start : start + chunk]
        d = x_sq[start : start + chunk, None] + c_sq[None, :] - 2.0 * (block @ centers.T)
        out[start : start + chunk] = d.argmin(axis=1)
    return out
