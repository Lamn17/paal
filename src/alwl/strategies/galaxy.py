"""GALAXY (Zhang, Katz-Samuels & Nowak, ICML 2022): graph-based AL for extreme imbalance."""

from __future__ import annotations

import numpy as np

from .base import Context
from .uncertainty import LeastConfidence


class Galaxy:
    """One linear graph per class k orders labeled and unlabeled samples by the one-vs-rest
    margin p_k - max_{j != k} p_j. Within a batch GALAXY queries one sample at a time:

    1. among all graphs, take the shortest *bisectable* segment -- consecutive labeled
       nodes of opposite class (k vs. not k) with only unlabeled nodes between them -- and
       query its midpoint; the revealed label immediately updates every graph;
    2. once no graph has a bisectable segment, query the unlabeled node closest to an
       already identified cut, which may open new bisectable segments (back to step 1).

    Requires ``ctx.oracle``; every queried sample is part of the returned batch.
    """

    name = "galaxy"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        if ctx.oracle is None or len(ctx.labeled_ids) == 0:
            return LeastConfidence().select(ctx, b)
        n_pool = len(ctx.pool_ids)
        b = min(int(b), n_pool)
        probs = np.concatenate([ctx.probs, ctx.labeled_probs]).astype(np.float64)
        K = probs.shape[1]
        # node index: 0..n_pool-1 pool, then labeled samples
        label = np.full(len(probs), -1, dtype=np.int64)
        label[n_pool:] = np.asarray(ctx.labels, dtype=np.int64)
        orders, ranks = [], []
        for k in range(K):
            rest = probs.copy()
            rest[:, k] = -np.inf
            order = np.argsort(probs[:, k] - rest.max(axis=1), kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            orders.append(order)
            ranks.append(rank)
        is_labeled = [label[order] >= 0 for order in orders]
        is_k = [label[order] == k for k, order in enumerate(orders)]

        chosen: list[int] = []
        for _ in range(b):
            node = self._bisect(is_labeled, is_k, orders)
            if node is None:
                node = self._near_cut(is_labeled, is_k, orders)
            if node is None:
                break
            y = int(ctx.oracle(int(ctx.pool_ids[node])))
            label[node] = y
            chosen.append(node)
            for k in range(K):
                r = ranks[k][node]
                is_labeled[k][r] = True
                is_k[k][r] = y == k
        if len(chosen) < b:  # no cut anywhere yet: fall back to least confidence
            taken = set(chosen)
            fill = [p for p in LeastConfidence().select(ctx, n_pool) if p not in taken]
            chosen.extend(fill[: b - len(chosen)])
        return np.asarray(chosen, dtype=np.int64)

    @staticmethod
    def _bisect(is_labeled, is_k, orders) -> int | None:
        best_len, best_node = None, None
        for lab, pos_k, order in zip(is_labeled, is_k, orders):
            lp = np.flatnonzero(lab)
            if len(lp) < 2:
                continue
            side = pos_k[lp]
            gap = np.diff(lp)
            ok = (side[1:] != side[:-1]) & (gap > 1)
            if not ok.any():
                continue
            i = np.flatnonzero(ok)[np.argmin(gap[ok])]
            if best_len is None or gap[i] < best_len:
                best_len, best_node = gap[i], int(order[lp[i] + gap[i] // 2])
        return best_node

    @staticmethod
    def _near_cut(is_labeled, is_k, orders) -> int | None:
        best_dist, best_node = None, None
        for lab, pos_k, order in zip(is_labeled, is_k, orders):
            lp = np.flatnonzero(lab)
            if len(lp) < 2:
                continue
            side = pos_k[lp]
            cuts = (lp[:-1][side[1:] != side[:-1]] + lp[1:][side[1:] != side[:-1]]) / 2.0
            unlabeled = np.flatnonzero(~lab)
            if not len(cuts) or not len(unlabeled):
                continue
            j = np.searchsorted(cuts, unlabeled)
            left = np.abs(unlabeled - cuts[np.clip(j - 1, 0, len(cuts) - 1)])
            right = np.abs(unlabeled - cuts[np.clip(j, 0, len(cuts) - 1)])
            dist = np.minimum(left, right)
            i = int(np.argmin(dist))
            if best_dist is None or dist[i] < best_dist:
                best_dist, best_node = dist[i], int(order[unlabeled[i]])
        return best_node
