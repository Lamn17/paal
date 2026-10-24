"""Run one query strategy for one seed and write its log to <run_dir>/<modality>/<strategy>_seed<seed>/."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alwl.config import load_config
from alwl.loop import run_active_learning


def run_dir(cfg: dict, strategy: str, seed: int) -> Path:
    return Path(cfg["run_dir"]) / cfg["model"]["modality"] / f"{strategy}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--modality", choices=["image", "lidar"])
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.override)
    if args.modality:
        cfg["model"]["modality"] = args.modality
    run_active_learning(cfg, args.strategy, args.seed, run_dir(cfg, args.strategy, args.seed))


if __name__ == "__main__":
    main()
