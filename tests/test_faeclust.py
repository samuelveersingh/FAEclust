"""Unit tests and a light end-to-end test for FAEclust."""
import numpy as np
import pytest
import tensorflow as tf

from FAEclust import (
    standardize_functional, smoothing_features, bspline_basis,
    TimeSeriesDistance, NearestNeighborsOpt, ConvexClustering,
    fista_solve, ManifoldReadout, DatasetGenerator, FunctionalAutoencoder,
)
from FAEclust.fista import _edges_from_neighbors
from sklearn.metrics import adjusted_rand_score


def test_functional_standardization():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 50)
    # both features must vary across samples so the standard deviation used as
    # the divisor is not zero
    X = np.stack([np.stack([2 + 3 * np.sin(2 * np.pi * t) + 0.1 * rng.standard_normal(50),
                            -1 + 0.5 * np.cos(2 * np.pi * t) + 0.1 * rng.standard_normal(50)])
                  for _ in range(30)])
    Xs = standardize_functional(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-2)


def test_fisher_rao_reparam_invariance():
    t = np.linspace(0, 1, 80)
    y0 = np.stack([np.sin(2 * np.pi * t) + 0.3 * t, np.cos(2 * np.pi * t)])
    g = t ** 1.7
    g = (g - g[0]) / (g[-1] - g[0])
    yw = np.stack([np.interp(g, t, y0[f]) for f in range(2)])
    yd = np.stack([np.sin(4 * np.pi * t), np.cos(2 * np.pi * t) + 0.5])
    X = np.stack([y0, yw, yd])
    D = TimeSeriesDistance(X, metric='elastic', n_jobs=1).compute_distances()
    # a time-warped copy of a curve should be much closer to it than a
    # genuinely different curve
    assert D[0, 1] < 0.2 * D[0, 2]


def test_elastic_jit_matches_pure_reference():
    """The compiled (JIT) elastic distance must match the plain NumPy version
    up to tiny floating-point differences; the compilation only speeds it up
    and must not change the result."""
    from FAEclust.srvf import (
        _srvf_optimum_reparam_distance,
        _srvf_optimum_reparam_distance_pure,
        TimeSeriesDistance as _TSD,
    )
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 60)
    y0 = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    y1 = np.stack([np.sin(2 * np.pi * t + 0.5), np.cos(3 * np.pi * t)])
    X = np.stack([y0, y1])
    tsd = _TSD(X, metric='elastic', n_jobs=1)
    Q = [tsd._curve_to_srvf(tsd.X[i], tsd.times) for i in range(2)]
    d_jit = _srvf_optimum_reparam_distance(Q[0], Q[1], tsd.times, band_radius=-1)
    d_pure = _srvf_optimum_reparam_distance_pure(Q[0], Q[1], tsd.times)
    assert abs(d_jit - d_pure) < 1e-10


def test_elastic_banded_matches_unbanded_when_band_wide():
    """Restricting the time-warp search to a band of width T // 4 is wide
    enough to contain the best warp for these smooth signals, so the banded
    and unbanded distances must agree."""
    from FAEclust.srvf import (
        _srvf_optimum_reparam_distance, TimeSeriesDistance as _TSD,
    )
    t = np.linspace(0, 1, 80)
    y0 = np.stack([np.sin(2 * np.pi * t) + 0.3 * t, np.cos(2 * np.pi * t)])
    g = t ** 1.4
    g = (g - g[0]) / (g[-1] - g[0])
    y1 = np.stack([np.interp(g, t, y0[f]) for f in range(2)])
    X = np.stack([y0, y1])
    tsd = _TSD(X, metric='elastic', n_jobs=1)
    Q = [tsd._curve_to_srvf(tsd.X[i], tsd.times) for i in range(2)]
    d_full = _srvf_optimum_reparam_distance(Q[0], Q[1], tsd.times, band_radius=-1)
    d_band = _srvf_optimum_reparam_distance(Q[0], Q[1], tsd.times, band_radius=80 // 4)
    assert abs(d_band - d_full) < 1e-8


def test_elastic_fast_metric_is_upper_bound_and_close():
    """The 'elastic-fast' metric searches for the best time warp only within a
    band around a rough first estimate, so its distance can only be greater
    than or equal to the exact (unbanded) distance, and should be close on
    well-behaved inputs."""
    t = np.linspace(0, 1, 200)
    y0 = np.stack([np.sin(2 * np.pi * t) + 0.3 * t, np.cos(2 * np.pi * t)])
    g = t ** 1.4
    g = (g - g[0]) / (g[-1] - g[0])
    y1 = np.stack([np.interp(g, t, y0[f]) for f in range(2)])
    y2 = np.stack([np.sin(4 * np.pi * t), np.cos(2 * np.pi * t) + 0.5])
    X = np.stack([y0, y1, y2])
    # Exact distance: band_radius=-1 turns off the band entirely.
    D_full = TimeSeriesDistance(X, metric='elastic', n_jobs=1,
                                band_radius=-1).compute_distances()
    D_fast = TimeSeriesDistance(X, metric='elastic-fast', n_jobs=1).compute_distances()
    # a restricted search can only over-estimate the true minimum distance
    assert (D_fast + 1e-6 >= D_full).all()
    # ... and stays within 5% for these well-behaved warps
    assert np.max(np.abs(D_fast - D_full) / (D_full + 1e-6)) < 0.05


def test_fista_matches_cvxpy():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal([0, 0], 0.3, (6, 2)),
                        rng.normal([4, 4], 0.3, (6, 2))])
    n = len(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    o = NearestNeighborsOpt(D)
    nd = o.get_nearest_neighbors(opt_m=4)
    S = o.compute_similarity(nd, 'neighbors')
    ei, ej, w = _edges_from_neighbors(nd, S, n)
    lam = 0.3
    U = cp.Variable((n, 2))
    pen = sum(w[e] * cp.norm1(U[ei[e]] - U[ej[e]]) for e in range(len(w)))
    cp.Problem(cp.Minimize((1.0 / n) * cp.sum_squares(X - U) + lam * pen)).solve()
    Uf = fista_solve(X, ei, ej, w, lam, n=n, n_iter=4000, tol=1e-10)
    assert np.abs(Uf - U.value).max() < 1e-2


def test_homotopy_recovers_two_clusters():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal([0, 0], 0.25, (12, 2)),
                        rng.normal([6, 6], 0.25, (12, 2))])
    truth = np.array([0] * 12 + [1] * 12)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    o = NearestNeighborsOpt(D)
    nd = o.get_nearest_neighbors(opt_m=4)
    S = o.compute_similarity(nd, 'neighbors')
    for crit in ('silhouette', 'davies_bouldin'):
        labels = ConvexClustering(X, nd, S, criterion=crit).fit()
        assert adjusted_rand_score(truth, labels) == 1.0


@pytest.mark.parametrize("kind", ["sphere", "poincare"])
def test_manifold_exp_log_identity(kind):
    g = DatasetGenerator(10, 3 if kind == "sphere" else 2, 30, 2)
    X, _ = (g.generate_hypersphere() if kind == "sphere"
            else g.generate_hyperbolic())
    ro = ManifoldReadout(kind)
    if kind == "sphere":
        ro.fit_anchor(X)
    Y = tf.constant(X, dtype=tf.float32)
    rec = ro(ro.to_tangent(Y))
    assert float(tf.reduce_max(tf.abs(rec - Y))) < 1e-4


def test_euclidean_readout_is_identity():
    z = tf.random.normal([2, 3, 7])
    assert bool(tf.reduce_all(ManifoldReadout(None)(z) == z))


def test_network_end_to_end_runs():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 36)
    npc = 10

    def mk(f, ph):
        return np.stack([np.sin(2 * np.pi * f * t + ph), np.cos(2 * np.pi * f * t + ph)])

    X = np.stack([mk(1.0, rng.uniform(0, .3)) for _ in range(npc)] +
                 [mk(2.0, rng.uniform(0, .3)) for _ in range(npc)])
    X += 0.03 * rng.standard_normal(X.shape)
    y = np.array([0] * npc + [1] * npc)
    co, _, bs = smoothing_features(X, m=8, fit='bspline', standardize=True)
    bi = bspline_basis(6)
    D = TimeSeriesDistance(X, metric='fastdtw', n_jobs=1).compute_distances()
    o = NearestNeighborsOpt(D)
    nd = o.get_nearest_neighbors(opt_m=5)
    S = o.compute_similarity(nd, 'neighbors')
    fae = FunctionalAutoencoder(2, [12, 8, 3, 8, 12, 12, 12], l=6, m=8,
                                basis_smoothing=bs, basis_input=bi,
                                lambda_e=0.5, lambda_d=0.05, lambda_c=0.5,
                                t=t, sim_matrix=S, tau=0.9, manifold=None)
    fae.train_model(co, epochs=8, learning_rate=1e-3, batch_size=8,
                    neighbors_dict=nd, sim_matrix=S, pretrain_epochs=3)
    Z, lab = fae.predict(co)
    assert Z.shape == (2 * npc, 3)
    assert len(lab) == 2 * npc
    assert 1 <= len(set(lab)) <= 2 * npc
