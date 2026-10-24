"""End-to-end smoke test on a tiny synthetic LiDAR dataset."""

import json

import numpy as np
import pandas as pd
import pytest

from alwl.data import make_split
from alwl.loop import run_active_learning


@pytest.fixture
def tiny_dataset(tmp_path):
    rng = np.random.default_rng(0)
    K, n = 3, 60
    rows = []
    for i in range(n):
        y = i % K
        grid = (rng.random((20, 20, 20)) < 0.02).astype(np.int8)
        grid[y * 5 : y * 5 + 5] = 1  # class-dependent occupancy
        path = tmp_path / f"lidar_{i}.npy"
        np.save(path, grid)
        gamma = rng.normal(size=K).astype(np.float32)
        gamma[y] += 5
        rows.append({"sample_id": i, "episode_id": i // 4, "gamma": gamma, "lidar_path": str(path),
                     "image_path": ""})
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "samples.parquet")
    frame["y"] = np.asarray(frame["gamma"].tolist()).argmax(axis=1)
    pool, test = make_split(frame, 3, "sample", 1)
    (tmp_path / "splits").mkdir()
    np.savez(tmp_path / "splits" / "paper_seed1.npz", pool_ids=pool, test_ids=test)
    return {
        "dataset": {"processed": str(tmp_path), "K": K, "split_name": "paper"},
        "al": {"initial": 6, "batch": 6, "rounds": 2, "seeds": [1]},
        "model": {"modality": "lidar", "epochs": 2, "lr": 1e-3, "batch": 8, "num_workers": 0},
        "strategies": {"bait": {"candidates": 30}},
    }


@pytest.mark.parametrize("strategy", ["random", "galaxy", "badge", "bait", "probcover"])
def test_loop_runs_and_logs_every_round(tiny_dataset, tmp_path, strategy):
    log = run_active_learning(tiny_dataset, strategy, 1, tmp_path / strategy)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["round"] for r in rows] == [0, 1, 2]
    assert [r["n_labels"] for r in rows] == [6, 12, 18]
    assert all(0.0 <= r["top1"] <= 1.0 for r in rows)
    queried = np.load(tmp_path / strategy / "queried.npz")
    assert len(np.unique(queried["ids"])) == 18
    assert queried["round"].tolist() == [0] * 6 + [1] * 6 + [2] * 6


def test_sizes_as_pool_fractions(tiny_dataset, tmp_path):
    cfg = {**tiny_dataset, "al": {"initial_fraction": 0.2, "batch_fraction": 0.1, "rounds": 1}}
    log = run_active_learning(cfg, "random", 1, tmp_path / "fraction")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["n_labels"] for r in rows] == [10, 15]  # pool of 51 samples


def test_soft_labels_train_and_targets_are_distributions(tiny_dataset, tmp_path):
    from alwl.data import soft_targets

    q = soft_targets(np.array([[-80.0, -62.0, -60.0, -61.0, -85.0]]), 1.0)
    assert np.isclose(q.sum(), 1.0) and q.argmax() == 2 and q[0, 3] > q[0, 1] > q[0, 0]
    cfg = {**tiny_dataset, "model": {**tiny_dataset["model"],
                                     "soft_labels": {"temperature": 1.0, "weight": 0.5}}}
    log = run_active_learning(cfg, "entropy", 1, tmp_path / "soft")
    assert len(log.read_text().splitlines()) == 3
