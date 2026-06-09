"""Optional hyperparameter tuning with Optuna.

This is an optional helper for users and is not part of the core training
loop. Optuna is an optional dependency:

    pip install optuna

By default it searches for the hyperparameters that give the lowest training
loss after a short training run. You can instead pass a custom `score_fn`
(for example, the silhouette score of the predicted clustering) to optimize.
"""

import numpy as np


def optimize_hyperparameters(
    coeffs, basis_smoothing, basis_input, t, neighbors_dict, sim_matrix, p,
    n_trials=30, search_space=None, score_fn=None, epochs=60, pretrain_epochs=20,
    manifold=None, seed=0,
):
    """
    Search over learning rate, batch size, tau, latent size, and the
    regularization weights (lambda_e, lambda_d, lambda_c) using Optuna's
    Tree-structured Parzen Estimator sampler.

    Parameters
    ----------
    coeffs, basis_smoothing, basis_input, t, neighbors_dict, sim_matrix, p
        Same objects passed to `FunctionalAutoencoder`.
    n_trials : int
    search_space : dict or None
        Optional overrides, e.g. {'lr': (1e-4, 1e-2), 'tau': (0.5, 1.0),
        'lambda_e': (1e-3, 1.0), 'lambda_d': (1e-4, 0.5),
        'lambda_c': (1e-2, 1.0), 'batch_size': [8,16,32],
        'latent': [2,3,4,8]}.
    score_fn : callable(model, coeffs) -> float (higher is better) or None
        If None, uses the negative of the final training loss.

    Returns
    -------
    best_params : dict
    study : optuna.Study
    """
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Optuna is required for FAEclust.tuning. "
            "Install it with `pip install optuna`."
        ) from e

    from .FAE import FunctionalAutoencoder

    sp = {
        'lr': (1e-4, 1e-2),
        'tau': (0.5, 1.0),
        'lambda_e': (1e-3, 1.0),
        'lambda_d': (1e-4, 0.5),
        'lambda_c': (1e-2, 1.0),
        'batch_size': [8, 16, 32],
        'latent': [2, 3, 4, 8],
    }
    if search_space:
        sp.update(search_space)

    def objective(trial):
        lr = trial.suggest_float('lr', *sp['lr'], log=True)
        tau = trial.suggest_float('tau', *sp['tau'])
        le = trial.suggest_float('lambda_e', *sp['lambda_e'], log=True)
        ld = trial.suggest_float('lambda_d', *sp['lambda_d'], log=True)
        lc = trial.suggest_float('lambda_c', *sp['lambda_c'], log=True)
        bs = trial.suggest_categorical('batch_size', sp['batch_size'])
        latent = trial.suggest_categorical('latent', sp['latent'])
        layers = [16, 8, latent, 8, 16, 16, 16]

        model = FunctionalAutoencoder(
            p, layers, l=len(basis_input), m=coeffs.shape[-1],
            basis_smoothing=basis_smoothing, basis_input=basis_input,
            lambda_e=le, lambda_d=ld, lambda_c=lc, t=t, sim_matrix=sim_matrix,
            tau=tau, manifold=manifold, seed=seed,
        )
        model.train_model(
            coeffs, epochs=epochs, learning_rate=lr, batch_size=bs,
            neighbors_dict=neighbors_dict, sim_matrix=sim_matrix,
            pretrain_epochs=pretrain_epochs,
        )
        if score_fn is not None:
            return -float(score_fn(model, coeffs))   # negate: Optuna minimizes
        return float(np.mean(model.epoch_loss[-5:]))

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study

## ------------------------------------------------------------------------- ##
