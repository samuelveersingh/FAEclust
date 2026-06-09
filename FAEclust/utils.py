import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.interpolate import BSpline
from .smoothing import Smoothing
import matplotlib.pyplot as plt


## ------------------------------------------------------------------------- ##
## Data Loading & Preprocessing from arff files
def load_and_preprocess(data_path, label_column=-1):
    """
    Load ARFF dataset, normalize features to [-1,1], and extract integer labels.

    Returns
    -------
    data : np.ndarray
        Preprocessed feature matrix.
    labels : np.ndarray, shape (n_samples,)
        Integer-encoded class labels.
    """
    data, meta = arff.loadarff(data_path)
    df = pd.DataFrame(data)

    labels = df.iloc[:, label_column]
    labels = pd.factorize(labels)[0].astype('int32')

    if df.shape[1] == 2:
        features = np.array(df.drop(df.columns[label_column], axis=1)).reshape(len(labels))
        a, b = 1, -1
        normalized_features = []
        for k in range(len(features[0])):
            feat = np.array([[features[i][k][j] for j in range(len(features[0][0]))]
                             for i in range(len(features))]).astype('float32')
            feat_min, feat_max = np.min(feat), np.max(feat)
            normalized = (feat - feat_min) * ((a - b) / (feat_max - feat_min)) + b
            normalized_features.append(normalized)
        data = np.stack(normalized_features, axis=-1)
    else:
        data = np.array(df)
        data = data[:, :-1].astype('float32')
        k = np.ones(data.shape)
        a, b = 1, -1
        data = (data - np.min(data) * k) * ((a - b) / (np.max(data) - np.min(data))) + b * k
    return data, labels


## ------------------------------------------------------------------------- ##
## rescale and preprocess X, y
def rescale(X, y, name="Dataset"):
    """
    Scale features to [-1, 1] and encode labels as zero-based integers.
    """
    a, b, k = 1, -1, np.ones(X.shape)
    X_min, X_max = np.min(X), np.max(X)
    X_scaled = (X - X_min * k) * ((a - b) / (X_max - X_min)) + b * k

    # Re-encode the labels as contiguous integers starting at 0, so they can be
    # safely used as array indices (e.g. for counting or one-hot encoding).
    y = np.asarray(y)
    y_encoded = np.unique(y, return_inverse=True)[1].astype(np.int32)

    print(f"{name}: Shape of X = {X_scaled.shape}")
    return X_scaled, y_encoded


## ------------------------------------------------------------------------- ##
## Standardize each feature curve across samples
def standardize_functional(data, eps=1e-8, return_stats=False):
    """
    Standardize each feature (component) curve across all samples.

    For each feature d and each time point t, subtract the mean value across
    samples and divide by the standard deviation across samples:

        y_i^d(t)  ->  (y_i^d(t) - mean^d(t)) / std^d(t).

    This puts every feature curve on a comparable scale, so that features with
    larger ranges or more variation do not dominate the distance-based
    clustering. Apply it to the raw sample curves BEFORE smoothing.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, p, T)
        Raw (rescaled) sample curves.
    eps : float
        Small value added to the standard deviation to avoid dividing by zero.
    return_stats : bool
        If True, also return the (mean, std) curves of shape (p, T).

    Returns
    -------
    data_std : np.ndarray, shape (n_samples, p, T)
    (mean_fn, std_fn) : tuple of np.ndarray, shape (p, T)  [only if return_stats]
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected (n_samples, p, T), got shape {data.shape}")
    mean_fn = data.mean(axis=0)                      # mean curve per feature, (p, T)
    std_fn = np.sqrt(data.var(axis=0)) + eps         # std curve per feature, (p, T)
    data_std = ((data - mean_fn[None]) / std_fn[None]).astype('float32')
    if return_stats:
        return data_std, (mean_fn.astype('float32'), std_fn.astype('float32'))
    return data_std


## ------------------------------------------------------------------------- ##
## Feature Processing Pipeline
def smoothing_features(data, m=20, dis_p=600, fit='bspline', wavelet_level=None,
                       standardize=False):
    """
    Smooth each feature dimension via chosen basis and return coefficients.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, p, T)
        Input functional data to be smoothed.
    m : int
        Number of basis terms (default=20). For Fourier, interpret m as 2n+1.
    dis_p : int
        Number of fine evaluation grid points (default=600).
    fit : {'bspline', 'fourier', <wavelet name like 'db4'>}
        Basis type passed to Smoothing.
    wavelet_level : int or None
        Wavelet decomposition level (auto via GCV if None).
    standardize : bool
        If True, standardize each feature curve across samples before
        smoothing. See `standardize_functional`.
    """
    data = np.asarray(data, dtype='float32')
    if standardize:
        data = standardize_functional(data)

    n_samples, p, T = data.shape
    t_coarse = np.linspace(0, 1, T)
    coeffs_list = []
    curves_list = []
    basis_smoothing = None

    for i in range(p):
        kwargs = dict(dis_p=dis_p, fit=fit, data=data[:, i, :])
        if fit == 'bspline':
            kwargs['terms'] = m
        elif fit == 'fourier':
            if m is not None:
                kwargs['n'] = max(1, (int(m) + 1) // 2)
        else:
            if wavelet_level is not None:
                kwargs['wavelet_level'] = wavelet_level

        smoothing = Smoothing(**kwargs)

        if fit == 'fourier':
            C = np.asarray(smoothing.coeffs).astype('float32')
            coeffs_list.append(C)
            basis_smoothing = smoothing.fourier_basis()
        elif fit == 'bspline':
            coeffs_list.append(np.asarray(smoothing.coeffs, dtype='float32'))
            basis_smoothing = smoothing.smoothing_basis
        else:
            if basis_smoothing is None:
                basis_smoothing = bspline_basis(m)
                Phi = np.stack([b(t_coarse) for b in basis_smoothing], axis=1)  # [T, m]
                G = Phi.T @ Phi
                G_inv = np.linalg.pinv(G)
                Phi_pinv = G_inv @ Phi.T       # [m, T]
            C = np.empty((n_samples, m), dtype='float32')
            for j, f in enumerate(smoothing.fn_s):
                y = f(t_coarse)
                C[j] = (Phi_pinv @ y).astype('float32')
            coeffs_list.append(C)

        curves_list.append(smoothing.fn_s)

    coeffs = np.stack(coeffs_list, axis=1).astype('float32')  # (n_samples, p, m_eff)
    return coeffs, curves_list, basis_smoothing


## ------------------------------------------------------------------------- ##
def plot_smooth_fit(features, curves, labels):
    if len(curves.shape) == 1:
        plt.figure(figsize=(10, 6))
        idx = [min([i for i, x in enumerate(labels) if x == label])
               for label in sorted(set(labels), key=int)]
        fine_t = np.linspace(0, 1, 500)
        cmap = plt.cm.viridis
        color = cmap(np.linspace(0, 1, len(idx)))
        for i in range(len(idx)):
            plt.plot(fine_t, curves[idx[i]](fine_t), c=color[i], label=labels[idx[i]])
            plt.plot(np.linspace(0, 1, features.shape[2]), features[idx[i], 0], '.', c=color[i])
        plt.title('Smooth fit', fontweight='bold')
        plt.xlim((0, 1)); plt.ylim((-1, 1))
        plt.xlabel('scaled observations #'); plt.ylabel('scaled features')
        plt.grid(); plt.legend(); plt.show()


## ------------------------------------------------------------------------- ##
def bspline_basis(num_basis, degree=3):
    """Create `num_basis` callable B-spline basis functions of given degree on [0,1]."""
    num_knots = num_basis + degree + 1
    t_t = np.concatenate(([0] * degree, np.linspace(0, 1, num_knots - 2 * degree), [1] * degree))
    basis_input = []
    for i in range(num_basis):
        basis_coefs = np.zeros(num_knots - degree - 1)
        basis_coefs[i] = 1
        basis_input.append(BSpline(t_t, basis_coefs, degree))
    return basis_input


## ------------------------------------------------------------------------- ##
def fourier_basis(num_basis):
    """Create `num_basis` callable Fourier basis functions (constant, sin/cos pairs)."""
    basis_input = [lambda x: np.ones_like(x)]
    k = 1
    while len(basis_input) < num_basis:
        basis_input.append(lambda x, k=k: np.sin(2 * np.pi * k * x))
        if len(basis_input) >= num_basis:
            break
        basis_input.append(lambda x, k=k: np.cos(2 * np.pi * k * x))
        k += 1
    return basis_input


## ------------------------------------------------------------------------- ##
def load_dataset(filepath, n_features, n_steps):
    """Load dataset from a single CSV file using pandas."""
    df = pd.read_csv(filepath)
    y = df['label'].astype(int).values
    flat_X = df.drop(columns=['label']).values
    n_samples = flat_X.shape[0]
    X = flat_X.reshape(n_samples, n_features, n_steps)
    return X, y

## ------------------------------------------------------------------------- ##
