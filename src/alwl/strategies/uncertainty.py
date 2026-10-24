"""Random sampling and the classic uncertainty-sampling rules (Settles, 2009)."""

from __future__ import annotations

import numpy as np

from ._common import entropy, top_b
from .base import Context


class Random:
    """Uniform sampling without replacement (equivalent to classical, passive training)."""

    name = "random"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        return ctx.rng.choice(len(ctx.pool_ids), min(int(b), len(ctx.pool_ids)), replace=False)


class LeastConfidence:
    """Lowest top-1 probability; the "confidence sampling" baseline of Soltani et al."""

    name = "least_conf"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        return top_b(1.0 - ctx.probs.max(axis=1), b)


class Margin:
    """Smallest gap between the two most probable classes."""

    name = "margin"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        top2 = np.partition(ctx.probs, -2, axis=1)[:, -2:]
        return top_b(-(top2[:, 1] - top2[:, 0]), b)


class Entropy:
    """Highest predictive entropy."""

    name = "entropy"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        return top_b(entropy(ctx.probs), b)
