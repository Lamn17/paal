"""Classical training on the whole labeled pool: the accuracy active learning aims to match.

With ``model.soft_labels`` set (as for the soft-label active-learning runs) the pool is
trained on power-aware soft labels too, the matching reference for those runs:

  python -m scripts.train_full_pool --config configs/flash.yaml --seed 1 --modality lidar \
    --override run_dir=runs/flash_soft --override "model.soft_labels={temperature: 1.0, weight: 0.5}"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from alwl.config import load_config
from alwl.data import InputStore, image_size, load_samples, load_split, soft_targets
from alwl.metrics import top_k_accuracy
from alwl.train import predict, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--modality", choices=["image", "lidar"])
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.override)
    if args.modality:
        cfg["model"]["modality"] = args.modality
    frame = load_samples(cfg)
    pool_ids, test_ids = load_split(cfg, args.seed)
    y_of = dict(zip(frame["sample_id"], frame["y"]))
    store = InputStore.load(frame, np.concatenate([pool_ids, test_ids]), cfg["model"]["modality"],
                            image_size=image_size(cfg))
    labels = np.asarray([y_of[int(i)] for i in pool_ids])
    soft_cfg = cfg["model"].get("soft_labels") or {}
    soft = None
    if soft_cfg:
        gamma_of = dict(zip(frame["sample_id"], frame["gamma"]))
        gamma = np.asarray([gamma_of[int(i)] for i in pool_ids], dtype=np.float32)
        soft = soft_targets(gamma, float(soft_cfg.get("temperature", 1.0)))
    model, epochs = train_model(cfg, store, pool_ids, labels, seed=args.seed, soft=soft)
    probs, _ = predict(model, cfg, store, test_ids)
    y_test = np.asarray([y_of[int(i)] for i in test_ids])
    result = {
        "seed": args.seed,
        "modality": cfg["model"]["modality"],
        "n_train": len(pool_ids),
        "n_test": len(test_ids),
        "epochs": epochs,
        "soft_labels": soft_cfg or None,
        "top1": top_k_accuracy(probs, y_test, 1),
        "top5": top_k_accuracy(probs, y_test, 5),
    }
    out = Path(cfg["run_dir"]) / cfg["model"]["modality"] / f"full_supervised_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
