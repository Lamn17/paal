import numpy as np
import pytest

from alwl.strategies import STRATEGIES, Context, get_strategy


def _context(n_pool=120, n_lab=30, K=6, D=8, seed=0, params=None):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(K, D)) * 3
    y_pool = rng.integers(K, size=n_pool)
    y_lab = np.arange(n_lab) % K
    emb = (centers[y_pool] + rng.normal(size=(n_pool, D))).astype(np.float32)
    lab_emb = (centers[y_lab] + rng.normal(size=(n_lab, D))).astype(np.float32)
    W = rng.normal(size=(K, D)).astype(np.float32) * 0.3
    bias = np.zeros(K, np.float32)

    def softmax(h):
        z = h @ W.T
        z -= z.max(axis=1, keepdims=True)
        return (np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)).astype(np.float32)

    gamma_lab = rng.normal(size=(n_lab, K)).astype(np.float32)
    gamma_lab[np.arange(n_lab), y_lab] += 6.0  # the labeled best beam is the strongest
    pool_ids = np.arange(1000, 1000 + n_pool)
    truth = dict(zip(pool_ids, y_pool))
    return Context(
        pool_ids=pool_ids,
        probs=softmax(emb),
        emb=emb,
        labeled_ids=np.arange(n_lab),
        labeled_probs=softmax(lab_emb),
        labeled_emb=lab_emb,
        labels=y_lab,
        head_weight=W,
        head_bias=bias,
        num_classes=K,
        rng=np.random.default_rng(seed),
        params=params or {},
        oracle=lambda i: int(truth[int(i)]),
        labeled_gamma=gamma_lab,
    )


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_returns_b_distinct_pool_positions(name):
    ctx = _context()
    pos = np.asarray(get_strategy(name).select(ctx, 17))
    assert len(pos) == 17
    assert len(np.unique(pos)) == 17
    assert pos.min() >= 0 and pos.max() < len(ctx.pool_ids)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_batch_larger_than_pool_returns_whole_pool(name):
    ctx = _context(n_pool=10)
    pos = np.asarray(get_strategy(name).select(ctx, 50))
    assert sorted(pos) == list(range(10))


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_deterministic_given_rng(name):
    a = get_strategy(name).select(_context(seed=3), 9)
    b = get_strategy(name).select(_context(seed=3), 9)
    assert list(a) == list(b)


def test_least_confidence_margin_entropy_rank_by_uncertainty():
    ctx = _context()
    ctx.probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.99, 0.01]], np.float32)
    ctx.pool_ids = np.arange(4)
    for name in ["least_conf", "margin", "entropy"]:
        assert list(get_strategy(name).select(ctx, 2)) == [1, 2]
    assert get_strategy("confidence").name == "least_conf"


def test_coreset_picks_farthest_point_first():
    ctx = _context()
    ctx.emb = np.array([[0.0], [1.0], [10.0]], np.float32)
    ctx.pool_ids = np.arange(3)
    ctx.labeled_emb = np.array([[0.0]], np.float32)
    assert list(get_strategy("coreset").select(ctx, 2)) == [2, 1]


def test_badge_distance_matches_explicit_gradient_embedding():
    from alwl.strategies.badge import Badge

    ctx = _context(n_pool=40, K=4, D=3)
    pos = Badge().select(ctx, 1)
    a = ctx.probs.astype(np.float64).copy()
    a[np.arange(40), a.argmax(axis=1)] -= 1
    g = (a[:, :, None] * ctx.emb[:, None, :]).reshape(40, -1)
    assert pos[0] == int(np.argmax((g**2).sum(axis=1)))


def test_bald_prefers_samples_whose_prediction_flips_under_dropout():
    from alwl.strategies.bald import Bald

    ctx = _context(K=2, D=2, params={"dropout": 0.5, "mc_samples": 200})
    ctx.head_weight = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    ctx.head_bias = np.zeros(2, np.float32)
    # Sample 0 relies on two opposing features (dropout flips it); sample 1 is unambiguous
    # under any mask, and sample 2 has no signal at all (uncertain but not epistemic).
    ctx.emb = np.array([[6.0, 6.0], [6.0, 0.0], [0.0, 0.0]], np.float32)
    ctx.pool_ids = np.arange(3)
    assert Bald().select(ctx, 1)[0] == 0


def test_galaxy_queries_only_what_it_returns_and_bisects_first():
    ctx = _context()
    asked = []
    oracle = ctx.oracle
    ctx.oracle = lambda i: asked.append(int(i)) or oracle(i)
    pos = get_strategy("galaxy").select(ctx, 12)
    assert set(asked) <= set(ctx.pool_ids[pos].tolist())


def test_galaxy_queries_midpoint_of_shortest_bisectable_segment():
    from alwl.strategies.galaxy import Galaxy

    # One graph per class; nodes sorted by margin. Labeled nodes of opposite class at
    # ranks 0 and 4 leave a bisectable segment whose midpoint is rank 2.
    lab = np.array([True, False, False, False, True])
    is_k = np.array([False, False, False, False, True])
    order = np.arange(5)
    assert Galaxy._bisect([lab], [is_k], [order]) == 2


def test_probcover_covers_dense_regions_first():
    ctx = _context(n_lab=0)
    ctx.labeled_emb = np.zeros((0, ctx.emb.shape[1]), np.float32)
    ctx.labeled_probs = np.zeros((0, ctx.num_classes), np.float32)
    ctx.labeled_ids = np.zeros(0, np.int64)
    ctx.labels = np.zeros(0, np.int64)
    pos = get_strategy("probcover").select(ctx, 6)
    assert len(np.unique(pos)) == 6


def test_typiclust_spreads_over_clusters():
    ctx = _context(n_lab=0, K=6)
    ctx.labeled_emb = np.zeros((0, ctx.emb.shape[1]), np.float32)
    ctx.labeled_probs = np.zeros((0, ctx.num_classes), np.float32)
    ctx.labeled_ids = np.zeros(0, np.int64)
    ctx.labels = np.zeros(0, np.int64)
    ctx.params = {"max_clusters": 6, "min_cluster_size": 1}
    pos = get_strategy("typiclust").select(ctx, 6)
    assert len(np.unique(pos)) == 6


def test_power_loss_matrix_and_expected_power_loss():
    from alwl.strategies.paal import expected_power_loss, power_loss_matrix

    gamma = np.array([[0.0, -1.0, -20.0], [0.0, -3.0, -10.0], [-5.0, 0.0, -2.0]])
    D = power_loss_matrix(gamma, np.array([0, 0, 1]), 3)
    assert np.allclose(D[0], [0.0, 2.0, 15.0])  # mean gap of the two class-0 sweeps
    assert np.allclose(D[1], [5.0, 0.0, 2.0])
    assert np.allclose(D[2], (D[0] + D[1]) / 2 * np.array([1, 1, 0]))  # unseen row, zero diagonal
    probs = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])
    # both pick beam 0; sample 1 risks beam 1 (5 dB gap), sample 2 risks beam 2 (2.5 dB)
    assert np.allclose(expected_power_loss(probs, D), [2.5, 1.25])


def test_epl_takes_the_highest_expected_power_loss():
    from alwl.strategies.paal import Epl, expected_power_loss, power_loss_matrix

    ctx = _context(n_pool=60)
    D = power_loss_matrix(ctx.labeled_gamma, ctx.labels, ctx.num_classes)
    epl = expected_power_loss(ctx.probs.astype(np.float64), D)
    assert list(Epl().select(ctx, 5)) == list(np.argsort(-epl, kind="stable")[:5])


def test_paal_prefers_far_high_loss_samples():
    from alwl.strategies.paal import Paal

    ctx = _context(K=2, D=1, n_lab=2)
    ctx.labels = np.array([0, 1])
    ctx.labeled_gamma = np.array([[10.0, 0.0], [0.0, 10.0]], np.float32)
    ctx.labeled_emb = np.array([[0.0]], np.float32)
    ctx.emb = np.array([[1.0], [10.0], [10.5]], np.float32)
    ctx.probs = np.array([[0.5, 0.5], [0.99, 0.01], [0.6, 0.4]], np.float32)
    ctx.pool_ids = np.arange(3)
    # sample 1 is far but certain; sample 2 is as far and risks the 10 dB beam
    assert Paal().select(ctx, 1).tolist() == [2]
