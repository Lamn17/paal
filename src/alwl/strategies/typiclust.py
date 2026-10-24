"""TypiClust (Hacohen, Dekel & Weinshall, ICML 2022): typical samples of uncovered clusters."""

from __future__ import annotations

import numpy as np

from ._common import kmeans, l2_normalize
from .base import Context


def _pca(x: np.ndarray, dims: int) -> np.ndarray:
    if x.shape[1] <= dims:
        return x
    centered = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centered[:: max(1, len(centered) // 20000)], full_matrices=False)
    return centered @ vt[:dims].T


def _typicality(pts: np.ndarray, nn: int, chunk: int = 2048) -> np.ndarray:
    """Inverse mean Euclidean distance of each point to its ``nn`` nearest other points."""
    sq = np.einsum("ij,ij->i", pts, pts)
    out = np.empty(len(pts), dtype=np.float64)
    for s in range(0, len(pts), chunk):
        d = sq[s : s + chunk, None] + sq[None, :] - 2.0 * (pts[s : s + chunk] @ pts.T)
        d = np.sqrt(np.maximum(np.partition(d, nn, axis=1)[:, 1 : nn + 1], 0.0))
        out[s : s + chunk] = 1.0 / (d.mean(axis=1) + 1e-12)
    return out


class TypiClust:
    """Cluster labeled + unlabeled embeddings into min(|L| + b, max_clusters) groups; then,
    b times, pick the cluster holding the fewest labeled samples (ties: the largest) and
    query its most *typical* unlabeled member, typicality being the inverse mean distance
    to its ``knn`` nearest neighbours in the cluster. Clusters smaller than
    ``min_cluster_size`` are skipped. The paper clusters self-supervised features; here
    the current model's embeddings (reduced to ``pca_dims``) stand in for them.
    """

    name = "typiclust"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        n_pool = len(ctx.pool_ids)
        b = min(int(b), n_pool)
        x = l2_normalize(np.concatenate([ctx.emb, ctx.labeled_emb]))
        x = _pca(x, int(ctx.params.get("pca_dims", 64))).astype(np.float32)
        k = min(len(ctx.labeled_ids) + b, int(ctx.params.get("max_clusters", 500)))
        assign = kmeans(x, k, ctx.rng)
        knn = int(ctx.params.get("knn", 20))
        min_size = int(ctx.params.get("min_cluster_size", 5))
        labeled = np.zeros(len(x), dtype=bool)
        labeled[n_pool:] = True
        sizes = np.bincount(assign, minlength=k)
        covered = np.bincount(assign[labeled], minlength=k)
        chosen: list[int] = []
        while len(chosen) < b:
            eligible = (sizes >= min_size) & (np.bincount(assign[~labeled], minlength=k) > 0)
            if not eligible.any():
                eligible = np.bincount(assign[~labeled], minlength=k) > 0
            key = np.where(eligible, covered * (len(x) + 1) - sizes, np.iinfo(np.int64).max)
            cluster = int(np.argmin(key))
            members = np.flatnonzero(assign == cluster)
            pts = x[members]
            nn = min(knn, len(members) - 1)
            if nn > 0:
                typicality = _typicality(pts, nn)
            else:
                typicality = np.ones(len(members))
            typicality[labeled[members]] = -np.inf
            node = int(members[int(np.argmax(typicality))])
            chosen.append(node)
            labeled[node] = True
            covered[cluster] += 1
        return np.asarray(chosen, dtype=np.int64)
