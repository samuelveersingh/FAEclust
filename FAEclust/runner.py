"""One-call experiment runner for quick dataset / hyperparameter testing.

    from FAEclust import run_experiment
    res = run_experiment('pendulum', epochs=80, metric='fastdtw', device=None)
    print(res['ami'], res['ari'])

Runs the whole pipeline (generate data -> compute distances -> build the
nearest-neighbor graph -> smooth -> train -> cluster -> score) with sensible
default settings, automatic GPU use when available, and a `fast=True` preset
for a quick check.
"""

import time
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

from .utils import smoothing_features, bspline_basis, rescale
from .manifolds import DatasetGenerator
from .srvf import TimeSeriesDistance
from .mnn import NearestNeighborsOpt
from .FAE import FunctionalAutoencoder, pick_device

# name -> (n_samples, n_features, n_steps, n_clusters, manifold_kind)
MANIFOLD_DATASETS = {
    'hypersphere': (100, 3, 100, 2, 'sphere'),
    'hyperbolic':  (200, 2, 50,  2, 'poincare'),
    'swiss_roll':  (300, 2, 200, 4, None),
    'lorenz':      (100, 3, 100, 3, None),
    'pendulum':    (200, 2, 100, 4, None),
}


def run_experiment(dataset='pendulum', *, metric='fastdtw', m_basis=30,
                   l_basis=30, dis_p=200, latent=4, epochs=120,
                   pretrain_epochs=40, lr=1e-3, batch_size=32, beta=0.9,
                   lambda_e=0.5, lambda_d=0.05, lambda_c=0.5, tau=0.9,
                   use_bn=True, criterion='silhouette', use_manifold_readout=True,
                   cluster_every=5, device=None, seed=0, fast=False,
                   m=None, verbose=True):
    """
    Run the full FAEclust pipeline on a built-in synthetic dataset.

    Returns a dict: {dataset, ami, ari, n_pred, n_true, labels, Z,
                      epoch_loss, train_time, device, model}.

    `fast=True` shrinks the number of epochs and grid points for a quick check.
    Any hyperparameter can be overridden by keyword — useful for sweeps.
    """
    if dataset not in MANIFOLD_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"choose from {list(MANIFOLD_DATASETS)}")
    ns, nf, nstp, nc, kind = MANIFOLD_DATASETS[dataset]
    if not use_manifold_readout:
        kind = None
    if fast:
        epochs, pretrain_epochs, m_basis, l_basis, dis_p = 20, 6, 16, 16, 80

    dev = pick_device(device)
    if verbose:
        print(f"[{dataset}] device={dev}  (samples={ns}, dims={nf}, "
              f"steps={nstp}, true K={nc}, readout={kind})")

    np.random.seed(seed)
    gen = DatasetGenerator(ns, nf, nstp, nc)
    X, y = getattr(gen, f"generate_{dataset}")()
    X, y = rescale(X, y, dataset)

    t0 = time.time()
    D = TimeSeriesDistance(X, metric=metric, n_jobs=-1).compute_distances()
    opt = NearestNeighborsOpt(D)
    # Choose how many nearest neighbors to use in the graph. By default, pick
    # the smallest number that keeps the whole graph connected, then cap it so
    # the graph stays sparse. A user-supplied `m` overrides this. (The
    # alternative 'avg_distance' rule tends to pick too many neighbors here,
    # producing an almost fully connected graph that makes clustering both
    # slow and prone to collapsing into too few clusters.)
    if m is None:
        m_con = opt.estimate_optimal_m(method='connectivity',
                                       max_m=X.shape[0] - 1)
        m = int(max(5, min(m_con, X.shape[0] // 5)))
    if verbose:
        print(f"[{dataset}] nearest-neighbor graph size m={m}")
    neighbors_dict = opt.get_nearest_neighbors(opt_m=m)
    sim_matrix = opt.compute_similarity(neighbors_dict)
    t_grid = np.linspace(0, 1, dis_p)
    coeffs, _, basis_smoothing = smoothing_features(
        X, m=m_basis, dis_p=dis_p, fit='bspline', standardize=True)
    prep_time = time.time() - t0

    model = FunctionalAutoencoder(
        p=nf, layers=[16, 8, latent, 8, 16, 16, 16], l=l_basis, m=m_basis,
        basis_smoothing=basis_smoothing, basis_input=bspline_basis(l_basis),
        lambda_e=lambda_e, lambda_d=lambda_d, lambda_c=lambda_c,
        t=t_grid, sim_matrix=sim_matrix, tau=tau, use_bn=use_bn,
        manifold=kind, seed=seed)

    t0 = time.time()
    model.train_model(coeffs, epochs=epochs, learning_rate=lr,
                       batch_size=batch_size, neighbors_dict=neighbors_dict,
                       sim_matrix=sim_matrix, beta=beta,
                       pretrain_epochs=pretrain_epochs, criterion=criterion,
                       verbose=verbose, device=dev, cluster_every=cluster_every)
    train_time = time.time() - t0

    Z, labels = model.predict(coeffs, batch_size=batch_size, criterion=criterion)
    ami = adjusted_mutual_info_score(y, labels)
    ari = adjusted_rand_score(y, labels)
    if verbose:
        print(f"[{dataset}] AMI={ami:.4f} ARI={ari:.4f} "
              f"K_pred={len(np.unique(labels))}/{nc} "
              f"| prep={prep_time:.1f}s train={train_time:.1f}s")
    return dict(dataset=dataset, ami=ami, ari=ari,
                n_pred=int(len(np.unique(labels))), n_true=nc,
                labels=labels, Z=Z, epoch_loss=model.epoch_loss,
                train_time=train_time, prep_time=prep_time,
                device=str(dev), model=model)

## ------------------------------------------------------------------------- ##
