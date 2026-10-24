"""BADGE (Ash et al., ICLR 2020): k-means++ seeding on last-layer gradient embeddings."""

from __future__ import annotations

import numpy as np

from .base import Context


class Badge:
    """The gradient embedding of x under its predicted label is g = a (x) h with
    a = p - onehot(argmax p) and h the penultimate features. Distances are computed exactly
    from ||g_x - g_c||^2 = |a_x|^2|h_x|^2 + |a_c|^2|h_c|^2 - 2 (a_x . a_c)(h_x . h_c),
    so the K*D-dimensional embeddings are never materialised.
    """

    name = "badge"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        p = np.asarray(ctx.probs, dtype=np.float64)
        h = np.asarray(ctx.emb, dtype=np.float64)
        a = p.copy()
        a[np.arange(len(p)), p.argmax(axis=1)] -= 1.0
        norm_sq = (a * a).sum(axis=1) * (h * h).sum(axis=1)
        count = min(int(b), len(p))

        def dist_to(c: int) -> np.ndarray:
            return np.maximum(norm_sq + norm_sq[c] - 2.0 * (a @ a[c]) * (h @ h[c]), 0.0)

        # As in the reference implementation, start from the largest gradient.
        chosen = [int(np.argmax(norm_sq))]
        dist = dist_to(chosen[0])
        dist[chosen[0]] = 0.0
        while len(chosen) < count:
            total = dist.sum()
            if total > 0:
                pos = int(ctx.rng.choice(len(p), p=dist / total))
            else:
                rest = np.setdiff1d(np.arange(len(p)), chosen)
                pos = int(ctx.rng.choice(rest))
            chosen.append(pos)
            np.minimum(dist, dist_to(pos), out=dist)
            dist[chosen] = 0.0
        return np.asarray(chosen, dtype=np.int64)
