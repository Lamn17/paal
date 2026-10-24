"""Core-set (Sener & Savarese, ICLR 2018): greedy k-center cover of the embedding space."""

from __future__ import annotations

import numpy as np

from ._common import sq_dist_to
from .base import Context


class Coreset:
    name = "coreset"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        emb = np.asarray(ctx.emb, dtype=np.float32)
        count = min(int(b), len(emb))
        if len(ctx.labeled_emb):
            dist = sq_dist_to(emb, ctx.labeled_emb)
        else:
            dist = ((emb - emb.mean(axis=0)) ** 2).sum(axis=1)
        chosen = np.empty(count, dtype=np.int64)
        for i in range(count):
            pos = int(np.argmax(dist))
            chosen[i] = pos
            np.minimum(dist, ((emb - emb[pos]) ** 2).sum(axis=1), out=dist)
            dist[chosen[: i + 1]] = -np.inf
        return chosen
