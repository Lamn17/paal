"""BALD (Houlsby et al., 2011; Gal et al., ICML 2017) with MC dropout on the classifier head."""

from __future__ import annotations

import numpy as np

from ._common import entropy, top_b
from .base import Context


class Bald:
    """Mutual information between the label and the weights, I = H[E p] - E H[p].

    The backbones carry no dropout, so the posterior is approximated by Monte-Carlo
    dropout (rate ``dropout``) on the penultimate features feeding the trained linear
    head, averaged over ``mc_samples`` masks.
    """

    name = "bald"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        rate = float(ctx.params.get("dropout", 0.5))
        samples = int(ctx.params.get("mc_samples", 20))
        h = np.asarray(ctx.emb, dtype=np.float32)
        W, bias = ctx.head_weight, ctx.head_bias
        mean_p = np.zeros((len(h), W.shape[0]), dtype=np.float64)
        mean_entropy = np.zeros(len(h), dtype=np.float64)
        for _ in range(samples):
            mask = ctx.rng.random(h.shape, dtype=np.float32) >= rate
            logits = (h * mask / (1.0 - rate)) @ W.T + bias
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            mean_p += p
            mean_entropy += entropy(p)
        mean_p /= samples
        mean_entropy /= samples
        return top_b(entropy(mean_p) - mean_entropy, b)
