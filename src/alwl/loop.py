"""Pool-based active learning (Soltani et al., 2024, Section II-A).

(i) fix the per-iteration batch; (ii) label a random initial batch; (iii) train the network
from scratch on the labeled set; (iv) predict on the whole unlabeled pool; (v) let the query
strategy pick the next batch; (vi) reveal its labels (exhaustive-search oracle);
(vii) repeat from (iii) until the labeling budget is spent.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from alwl.data import InputStore, image_size, load_samples, load_split, soft_targets
from alwl.metrics import per_class_min_queries, top_k_accuracy
from alwl.strategies import Context, get_strategy
from alwl.train import head_parameters, predict, train_model

LOG = logging.getLogger(__name__)


def run_active_learning(cfg: dict, strategy_name: str, seed: int, out_dir: str | Path) -> Path:
    """Run one strategy for one seed; write ``log.jsonl`` (one row per iteration) and
    ``queried.npz`` (sample ids in query order, with their iteration) to ``out_dir``."""
    al = cfg["al"]
    K = int(cfg["dataset"]["K"])
    strategy = get_strategy(strategy_name)
    params = (cfg.get("strategies") or {}).get(strategy.name, {}) or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    log_path.unlink(missing_ok=True)

    frame = load_samples(cfg)
    pool_ids, test_ids = load_split(cfg, seed)
    y_of = dict(zip(frame["sample_id"].to_numpy(dtype=np.int64), frame["y"].to_numpy()))
    gamma = np.asarray(frame["gamma"].tolist(), dtype=np.float32)
    gamma_of = dict(zip(frame["sample_id"].to_numpy(dtype=np.int64), gamma))
    soft_cfg = cfg["model"].get("soft_labels") or {}
    if soft_cfg:
        soft_of = dict(zip(frame["sample_id"].to_numpy(dtype=np.int64),
                           soft_targets(gamma, float(soft_cfg.get("temperature", 1.0)))))
    store = InputStore.load(frame, np.concatenate([pool_ids, test_ids]), cfg["model"]["modality"],
                            image_size=image_size(cfg))
    y_test = np.asarray([y_of[int(i)] for i in test_ids])
    pool_classes = np.unique([y_of[int(i)] for i in pool_ids])

    rng = np.random.default_rng(seed)
    # Sizes are absolute (``initial``/``batch``) or fractions of the pool (``*_fraction``).
    n_pool = len(pool_ids)
    batch = int(al["batch"]) if "batch" in al else int(round(float(al["batch_fraction"]) * n_pool))
    initial = (
        int(round(float(al["initial_fraction"]) * n_pool))
        if "initial_fraction" in al
        else int(al.get("initial", batch))
    )
    rounds = int(al.get("rounds", np.ceil((len(pool_ids) - initial) / batch)))
    labeled = list(rng.choice(pool_ids, min(initial, len(pool_ids)), replace=False))
    queried_round = [0] * len(labeled)

    for it in range(rounds + 1):
        labels = np.asarray([y_of[int(i)] for i in labeled])
        t0 = time.time()
        soft = np.stack([soft_of[int(i)] for i in labeled]) if soft_cfg else None
        model, epochs = train_model(cfg, store, labeled, labels, seed=seed * 10_000 + it, soft=soft)
        t_train = time.time() - t0
        probs_test, _ = predict(model, cfg, store, test_ids)
        row = {
            "round": it,
            "strategy": strategy.name,
            "seed": seed,
            "modality": cfg["model"]["modality"],
            "n_labels": len(labeled),
            "top1": top_k_accuracy(probs_test, y_test, 1),
            "top5": top_k_accuracy(probs_test, y_test, 5),
            "per_class_min_q": per_class_min_queries(labels, pool_classes),
            "n_classes_covered": int(len(np.unique(labels))),
            "epochs": epochs,
            "train_s": round(t_train, 1),
        }
        unlabeled = np.setdiff1d(pool_ids, labeled)
        if it < rounds and len(unlabeled):
            t0 = time.time()
            probs, emb = predict(model, cfg, store, unlabeled)
            labeled_probs, labeled_emb = predict(model, cfg, store, labeled)
            weight, bias = head_parameters(model)
            revealed: set[int] = set()

            def oracle(sample_id: int) -> int:
                revealed.add(int(sample_id))
                return int(y_of[int(sample_id)])

            ctx = Context(
                pool_ids=unlabeled,
                probs=probs,
                emb=emb,
                labeled_ids=np.asarray(labeled, dtype=np.int64),
                labeled_probs=labeled_probs,
                labeled_emb=labeled_emb,
                labels=labels,
                head_weight=weight,
                head_bias=bias,
                num_classes=K,
                rng=np.random.default_rng([seed, it]),
                params=params,
                oracle=oracle,
                labeled_gamma=np.stack([gamma_of[int(i)] for i in labeled]),
            )
            t_query = time.time()
            pos = np.asarray(strategy.select(ctx, min(batch, len(unlabeled))), dtype=np.int64)
            row["query_s"] = round(time.time() - t_query, 3)  # select() alone, no inference
            picked = unlabeled[pos]
            if len(np.unique(picked)) != len(picked) or len(picked) != min(batch, len(unlabeled)):
                raise RuntimeError(f"{strategy.name} returned an invalid batch")
            if not revealed <= set(map(int, picked)):
                raise RuntimeError(f"{strategy.name} queried labels it did not select")
            labeled.extend(int(i) for i in picked)
            queried_round.extend([it + 1] * len(picked))
            row["select_s"] = round(time.time() - t0, 1)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        LOG.info("%s seed %d round %d: n=%d top1=%.4f", strategy.name, seed, it,
                 row["n_labels"], row["top1"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not len(unlabeled):
            break
    np.savez(out / "queried.npz", ids=np.asarray(labeled, dtype=np.int64),
             round=np.asarray(queried_round, dtype=np.int64))
    return log_path
