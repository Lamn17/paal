"""PAAL (Power-Aware Active Learning): acquisition from the full beam sweep.

The exhaustive-search oracle measures the power of every beam, not just the best one.
PAAL uses those sweeps of the labeled samples to

  (2) score each unlabeled sample by its expected power loss (dB) under the beam the model
      would pick, EPL(x) = sum_k p(k|x) D(k, a_hat(x)), where D(k, a) is the mean power gap
      gamma_k - gamma_a over labeled samples whose best beam is k; and
  (3) build the batch by EPL-weighted k-center on the embeddings: each step takes the sample
      maximising EPL(x) times its distance to the labeled set and the batch so far, so the
      batch covers high-loss regions without redundant queries.

Ablations: without (1) = PAAL trained on hard labels; without (2) = Core-set (unweighted
k-center) with soft labels; without (3) = ``epl``, the b highest-EPL samples.

Component (1), training on power-aware soft labels, is set in ``model.soft_labels``.
"""

from __future__ import annotations

import numpy as np

from ._common import sq_dist_to, top_b
from .base import Context


def power_loss_matrix(gamma: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """D (K, K) in dB: mean of gamma_k - gamma_a over labeled samples whose best beam is k.

    Rows of beams with no labeled sample take the mean of the observed rows.
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    D = np.full((K, K), np.nan)
    for k in np.unique(labels):
        rows = gamma[labels == k]
        D[k] = (rows[:, k][:, None] - rows).mean(axis=0)
    seen = ~np.isnan(D[:, 0])
    if seen.any() and not seen.all():
        D[~seen] = D[seen].mean(axis=0)
    D = np.nan_to_num(np.maximum(D, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


def expected_power_loss(probs: np.ndarray, D: np.ndarray) -> np.ndarray:
    """EPL(x) = sum_k p(k|x) D(k, argmax p(.|x))."""
    a_hat = probs.argmax(axis=1)
    return (probs * D[:, a_hat].T).sum(axis=1)


def _k_center(emb: np.ndarray, labeled_emb: np.ndarray, b: int,
              weight: np.ndarray | None = None) -> np.ndarray:
    """Greedy k-center over ``emb``; with ``weight`` each step takes argmax weight * distance."""
    emb = np.asarray(emb, dtype=np.float32)
    if len(labeled_emb):
        dist = sq_dist_to(emb, labeled_emb)
    else:
        dist = ((emb - emb.mean(axis=0)) ** 2).sum(axis=1)
    w = np.ones(len(emb), np.float32) if weight is None else np.asarray(weight, np.float32)
    chosen = np.empty(min(int(b), len(emb)), dtype=np.int64)
    for i in range(len(chosen)):
        pos = int(np.argmax(w * dist))
        chosen[i] = pos
        np.minimum(dist, ((emb - emb[pos]) ** 2).sum(axis=1), out=dist)
        dist[chosen[: i + 1]] = -np.inf
    return chosen


class Paal:
    name = "paal"

    def _epl(self, ctx: Context) -> np.ndarray:
        if ctx.labeled_gamma is None:
            raise ValueError("PAAL needs the labeled power sweeps (ctx.labeled_gamma)")
        D = power_loss_matrix(ctx.labeled_gamma, ctx.labels, ctx.num_classes)
        return expected_power_loss(np.asarray(ctx.probs, dtype=np.float64), D)

    def select(self, ctx: Context, b: int) -> np.ndarray:
        return _k_center(ctx.emb, ctx.labeled_emb, b, weight=self._epl(ctx) + 1e-6)


class Epl(Paal):
    """Ablation: the b highest-EPL samples, without step (3)."""

    name = "epl"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        return top_b(self._epl(ctx), min(int(b), len(ctx.pool_ids)))
