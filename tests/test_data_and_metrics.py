import numpy as np
import pandas as pd

from alwl.config import load_config
from alwl.data import make_split
from alwl.metrics import labels_to_reach, per_class_min_queries, top_k_accuracy


def _frame():
    y = np.array([0] * 30 + [1] * 30 + [2] * 6)
    return pd.DataFrame(
        {"sample_id": np.arange(len(y)) + 100, "episode_id": np.arange(len(y)) // 3, "y": y}
    )


def test_sample_split_is_class_balanced_and_disjoint():
    frame = _frame()
    pool, test = make_split(frame, 4, "sample", 0)
    by_id = frame.set_index("sample_id")
    assert set(pool).isdisjoint(test)
    assert len(pool) + len(test) == len(frame)
    assert np.bincount(by_id.loc[test, "y"]).tolist() == [4, 4, 4]


def test_max_test_fraction_keeps_rare_class_in_pool():
    frame = _frame()
    _, test = make_split(frame, 10, "sample", 0)
    assert np.bincount(frame.set_index("sample_id").loc[test, "y"])[2] == 6  # all of class 2
    pool, test = make_split(frame, 10, "sample", 0, max_test_fraction=0.5)
    by_id = frame.set_index("sample_id")
    assert np.bincount(by_id.loc[test, "y"]).tolist() == [10, 10, 3]
    assert (by_id.loc[pool, "y"] == 2).sum() == 3


def test_episode_split_keeps_episodes_whole():
    frame = _frame()
    pool, test = make_split(frame, 2, "episode", 1)
    by_id = frame.set_index("sample_id")
    assert set(by_id.loc[pool, "episode_id"]).isdisjoint(by_id.loc[test, "episode_id"])


def test_split_depends_only_on_seed():
    frame = _frame()
    assert np.array_equal(make_split(frame, 4, "sample", 7)[1], make_split(frame, 4, "sample", 7)[1])
    assert not np.array_equal(make_split(frame, 4, "sample", 7)[1], make_split(frame, 4, "sample", 8)[1])


def test_metrics():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
    y = np.array([0, 1, 1])
    assert top_k_accuracy(probs, y, 1) == 1 / 3
    assert top_k_accuracy(probs, y, 2) == 1.0
    assert per_class_min_queries(np.array([0, 0, 2]), np.array([0, 1, 2])) == 0
    assert per_class_min_queries(np.array([0, 1, 2, 2]), np.array([0, 1, 2])) == 1
    assert labels_to_reach(np.array([10, 20, 30]), np.array([0.1, 0.5, 0.7]), 0.5) == 20
    assert labels_to_reach(np.array([10, 20]), np.array([0.1, 0.2]), 0.5) == float("inf")


def test_configs_load_and_list_all_baselines():
    from alwl.strategies import STRATEGIES

    for path in ["configs/flash.yaml", "configs/deepsense.yaml"]:
        cfg = load_config(path, ["al.rounds=2"])
        assert cfg["al"]["rounds"] == 2
        assert set(cfg["benchmark"]["strategies"]) <= set(STRATEGIES)
        assert len(cfg["benchmark"]["strategies"]) >= 11
        assert 0 < cfg["al"]["initial_fraction"] <= 1 and 0 < cfg["al"]["batch_fraction"] <= 1
