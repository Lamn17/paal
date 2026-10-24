"""BAIT (Ash et al., NeurIPS 2021): greedy Fisher-information trace minimisation."""

from __future__ import annotations

import numpy as np
import torch

from .base import Context


def _project(h_all: np.ndarray, r: int) -> np.ndarray:
    """Top-``r`` principal directions of the embeddings (columns), for a low-rank Fisher."""
    centered = h_all - h_all.mean(axis=0)
    _, _, vt = np.linalg.svd(centered[:: max(1, len(centered) // 20000)], full_matrices=False)
    return vt[:r].T.astype(np.float32)


def _fisher_embedding(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """V_x (n, K*r, K): column k is sqrt(p_k) (e_k - p) (x) z, so V V^T is the last-layer Fisher."""
    K = p.shape[1]
    a = torch.eye(K, device=p.device)[None] - p[:, :, None]  # a[n, i, k] = delta_ik - p_i
    a = a * p.sqrt()[:, None, :]
    return torch.einsum("nik,nj->nijk", a, z).reshape(len(p), -1, K)


class Bait:
    """Select S to minimise tr((sum_{x in L u S} V_x V_x^T + lambda I)^{-1} F), F the pool Fisher.

    Forward-greedy adds ``oversample * b`` samples, backward-greedy prunes back to ``b``
    (Algorithm 1 of the paper). For tractability the penultimate features are projected
    onto their top ``rank // K`` principal directions and the forward/backward search runs
    over a random subset of ``candidates`` pool samples.
    """

    name = "bait"

    def select(self, ctx: Context, b: int) -> np.ndarray:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        K = ctx.num_classes
        lam = float(ctx.params.get("lamb", 1.0))
        r = max(1, int(ctx.params.get("rank", 256)) // K)
        n_cand = int(ctx.params.get("candidates", 4000))
        b = min(int(b), len(ctx.pool_ids))
        oversample = max(b, min(int(float(ctx.params.get("oversample", 2.0)) * b), len(ctx.pool_ids)))

        h_all = np.concatenate([ctx.emb, ctx.labeled_emb]).astype(np.float32)
        proj = torch.from_numpy(_project(h_all, r)).to(dev)

        def embed(p, h, chunk=2048):
            for s in range(0, len(p), chunk):
                pt = torch.from_numpy(np.asarray(p[s : s + chunk], dtype=np.float32)).to(dev)
                zt = torch.from_numpy(np.asarray(h[s : s + chunk], dtype=np.float32)).to(dev) @ proj
                yield _fisher_embedding(pt, zt)

        d = K * proj.shape[1]
        fisher = torch.zeros(d, d, device=dev)
        for v in embed(np.concatenate([ctx.probs, ctx.labeled_probs]), h_all):
            fisher += torch.einsum("ndk,nek->de", v, v)
        fisher /= len(h_all)
        M = lam * torch.eye(d, device=dev)
        for v in embed(ctx.labeled_probs, ctx.labeled_emb):
            M += torch.einsum("ndk,nek->de", v, v)

        cand = np.arange(len(ctx.pool_ids))
        if len(cand) > max(n_cand, oversample):
            cand = np.sort(ctx.rng.choice(cand, max(n_cand, oversample), replace=False))
        V = torch.cat(list(embed(ctx.probs[cand], ctx.emb[cand])))  # (n, d, K)
        eye_k = torch.eye(K, device=dev)

        def inv(m):
            return torch.linalg.inv(m.double()).float()

        chosen: list[int] = []
        available = torch.ones(len(cand), dtype=torch.bool, device=dev)
        for _ in range(oversample):  # forward: largest decrease of the trace
            Minv = inv(M)
            T = Minv @ fisher @ Minv
            MV = torch.einsum("de,nek->ndk", Minv, V)
            B = eye_k + torch.einsum("ndk,ndl->nkl", V, MV)
            A = torch.einsum("ndk,ndl->nkl", V, torch.einsum("de,nel->ndl", T, V))
            gain = torch.linalg.solve(B, A).diagonal(dim1=1, dim2=2).sum(dim=1)
            gain[~available] = -torch.inf
            pos = int(torch.argmax(gain))
            chosen.append(pos)
            available[pos] = False
            M += V[pos] @ V[pos].T
        while len(chosen) > b:  # backward: smallest increase when removed
            Minv = inv(M)
            T = Minv @ fisher @ Minv
            Vs = V[chosen]
            MV = torch.einsum("de,nek->ndk", Minv, Vs)
            B = eye_k - torch.einsum("ndk,ndl->nkl", Vs, MV)
            A = torch.einsum("ndk,ndl->nkl", Vs, torch.einsum("de,nel->ndl", T, Vs))
            loss = torch.linalg.solve(B, A).diagonal(dim1=1, dim2=2).sum(dim=1)
            drop = int(torch.argmin(loss))
            M -= Vs[drop] @ Vs[drop].T
            chosen.pop(drop)
        return cand[np.asarray(chosen, dtype=np.int64)]
