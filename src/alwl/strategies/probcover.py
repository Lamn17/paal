"""ProbCover (Yehuda, Dekel, Hacohen & Weinshall, NeurIPS 2022): greedy max coverage."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ._common import l2_normalize
from .base import Context


def _ball_graph(x: np.ndarray, knn: int, chunk: int = 2048) -> sparse.csr_matrix:
    """Adjacency of the delta-ball graph; delta is the median distance to the ``knn``-th
    nearest neighbour, so a typical ball holds about ``knn`` points."""
    sq = np.einsum("ij,ij->i", x, x)
    kth = np.empty(len(x), dtype=np.float32)
    for s in range(0, len(x), chunk):
        d = sq[s : s + chunk, None] + sq[None, :] - 2.0 * (x[s : s + chunk] @ x.T)
        kth[s : s + chunk] = np.partition(d, min(knn, len(x) - 1), axis=1)[:, min(knn, len(x) - 1)]
    delta_sq = float(np.median(np.maximum(kth, 0.0)))
    rows, cols = [], []
    for s in range(0, len(x), chunk):
        d = sq[s : s + chunk, None] + sq[None, :] - 2.0 * (x[s : s + chunk] @ x.T)
        r, c = np.nonzero(d <= delta_sq)
        rows.append(r + s)
        cols.append(c)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(x),) * 2)


class ProbCover:
    """Build a delta-ball graph over labeled + unlabeled embeddings (L2-normalised); points
    within delta of a labeled sample are covered. Repeatedly query the unlabeled sample
    whose ball covers the most uncovered points. The paper picks delta from self-supervised
    cluster purity; here delta is set from the ``knn``-th neighbour distance of the current
    model's embeddings. When everything is covered the rest of the batch is random.
    """

    name = "probcover"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        n_pool = len(ctx.pool_ids)
        b = min(int(b), n_pool)
        x = l2_normalize(np.concatenate([ctx.emb, ctx.labeled_emb]))
        graph = _ball_graph(x, int(ctx.params.get("knn", 20)))
        covered = np.zeros(len(x), dtype=bool)
        if len(x) > n_pool:
            covered[graph[n_pool:].indices] = True
        degree = graph @ (~covered).astype(np.float32)
        chosen: list[int] = []
        taken = np.zeros(n_pool, dtype=bool)
        while len(chosen) < b:
            score = np.where(taken, -1.0, degree[:n_pool])
            pos = int(np.argmax(score))
            if score[pos] <= 0:
                rest = np.flatnonzero(~taken)
                extra = ctx.rng.choice(rest, b - len(chosen), replace=False)
                chosen.extend(int(e) for e in extra)
                break
            chosen.append(pos)
            taken[pos] = True
            newly = graph[pos].indices[~covered[graph[pos].indices]]
            covered[newly] = True
            if len(newly):
                indicator = np.zeros(len(x), dtype=np.float32)
                indicator[newly] = 1.0
                degree -= graph @ indicator
        return np.asarray(chosen, dtype=np.int64)
