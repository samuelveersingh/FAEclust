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

    Parameters
    ----------
    data_path : str
        Path to the .arff dataset file.
    label_column : int, optional (default=-1)
        Index of the label column in the loaded data.

    Returns
    -------
    data : np.ndarray, shape (n_samples, n_features) or (n_samples,)
        Preprocessed feature matrix (or vector for single-feature cases).
    labels : np.ndarray, shape (n_samples,)
        Integer-encoded class labels.
    """
    data, meta = arff.loadarff(data_path)
    
    # Convert structured array to DataFrame
    df = pd.DataFrame(data)
    
    # Extract features and labels
    labels = df.iloc[:, label_column]
    labels = pd.factorize(labels)[0].astype('int32')
    
    if df.shape[1] == 2:
        features = np.array(df.drop(df.columns[label_column], axis=1)).reshape(len(labels))
        
        # Normalize features
        a, b = 1, -1
        normalized_features = []
        for k in range(len(features[0])):
            feat = np.array([[features[i][k][j] for j in range(len(features[0][0]))] for i in range(len(features))]).astype('float32')
            feat_min, feat_max = np.min(feat), np.max(feat)
            normalized = (feat - feat_min) * ((a - b)/(feat_max - feat_min)) + b
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
## rescale and preproces X, y
def rescale(X, y, name="Dataset"):
    """
    Scale features to [-1, 1] and encode labels as zero-based integers.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features,)
        Input feature matrix.
    y : np.ndarray, shape (n_samples,)
        Input label vector. Can be integer dtype or any other (e.g., strings).
    name : str, optional (default="Dataset")
        Identifier for logging.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_samples, n_features,)
        Feature matrix after min–max scaling to [-1, 1].
    y_encoded : np.ndarray, shape (n_samples,)
        Labels converted to zero-based integer codes.
    """
    # target range endpoints
    a, b, k = 1, -1, np.ones(X.shape)
    # compute global min and max of X
    X_min, X_max = np.min(X), np.max(X)
    # scale X to [b, a]: ((X - X_min) / (X_max - X_min)) * (a - b) + b
    X_scaled = (X - X_min * k) * ((a - b) / (X_max - X_min)) + b * k

    # encode y to zero-based integers
    if y.dtype == np.int64:
        # assume labels start at 1 → cast to int32 and subtract 1
        y_encoded = y.astype(np.int32) - 1
    else:
        # map arbitrary string labels to 0…n_classes-1
        y_encoded = np.unique(y, return_inverse=True)[1]

    print(f"{name}: Shape of X = {X_scaled.shape}")
    return X_scaled, y_encoded

## ------------------------------------------------------------------------- ##
## Feature Processing Pipeline 
def smoothing_features(data, m=20, dis_p=600, fit='bspline', wavelet_level=None):
    """
    Smooth each feature dimension via chosen basis and return coefficients.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, p, T)
        Input functional data to be smoothed.
    m : int, optional
        Number of basis terms (default=20). For Fourier, interpret m as 2n+1.
    dis_p : int, optional
        Number of fine evaluation grid points (default=600).
    fit : {'bspline', 'fourier', <wavelet name like 'db4'>}
        Basis type passed to Smoothing. For wavelets pass the wavelet name string (e.g. 'db4').
    wavelet_level : int or None, optional
        Wavelet decomposition level. If None and fit is a wavelet, Smoothing may choose via GCV.
        
    """
    n_samples, p, T = data.shape
    t_coarse = np.linspace(0, 1, T)
    coeffs_list = []
    curves_list = []
    basis_smoothing = None

    for i in range(p):
        kwargs = dict(dis_p=dis_p, fit=fit, data=data[:, i, :])
        if fit == 'bspline':
            kwargs['terms'] = m  # explicit number of spline basis terms
        elif fit == 'fourier':
            if m is not None:
                kwargs['n'] = max(1, (int(m) + 1) // 2)
        else:
            # assume wavelet name string (e.g. 'db4')
            if wavelet_level is not None:
                kwargs['wavelet_level'] = wavelet_level

        smoothing = Smoothing(**kwargs)

        # --- Collect coefficients in a stackable (n_samples, m_eff) array ---
        if fit == 'fourier':
            C = smoothing.coeffs
            C = np.asarray(C).astype('float32')
            coeffs_list.append(C)
            basis_smoothing = smoothing.fourier_basis()

        elif fit == 'bspline':
            coeffs_list.append(np.asarray(smoothing.coeffs, dtype='float32'))
            basis_smoothing = smoothing.smoothing_basis

        else:
            # Wavelet: Smoothing stores thresholded coeff structures; to get fixed-size coeffs ->
            # project each smoothed curve onto a B-spline basis of size m.
            # Build projection basis once.
            if basis_smoothing is None:
                basis_smoothing = bspline_basis(m)
                Phi = np.stack([b(t_coarse) for b in basis_smoothing], axis=1)  # [T, m]
                G = Phi.T @ Phi  # [m, m]
                G_inv = np.linalg.pinv(G)
                Phi_pinv = G_inv @ Phi.T       # [m, T]
            C = np.empty((n_samples, m), dtype='float32')
            for j, f in enumerate(smoothing.fn_s):
                y = f(t_coarse)                 # [T]
                beta = Phi_pinv @ y             # [m]
                C[j] = beta.astype('float32')
            coeffs_list.append(C)

        # store the callable curves for this feature (list length n_samples)
        curves_list.append(smoothing.fn_s)

    # stack coeffs → list of p arrays [n_samples, m_eff] -> (n_samples, p, m_eff)
    coeffs = np.stack(coeffs_list, axis=1).astype('float32')  # (n_samples, p, m_eff)

    # curves: keep as list-of-lists of callables, shape [p][n_samples]
    all_curves = curves_list

    return coeffs, all_curves, basis_smoothing

## ------------------------------------------------------------------------- ##
## plotting the fitted curves
def plot_smooth_fit(features, curves, labels):
    if len(curves.shape) == 1:
        plt.figure(figsize=(10, 6))
        idx = [min([i for i, x in enumerate(labels) if x == label]) for label in sorted(set(labels), key=int)]
        fine_t = np.linspace(0, 1, 500)
        cmap = plt.cm.viridis  # Get the viridis colormap
        color = cmap(np.linspace(0, 1, len(idx)))
        for i in range(len(idx)):
            plt.plot(fine_t, curves[idx[i]](fine_t), c=color[i], label=labels[idx[i]])
            plt.plot(np.linspace(0, 1, features.shape[2]), features[idx[i],0], '.', c=color[i])
    
        plt.title('Smooth fit', fontweight='bold')
        plt.xlim((0,1))
        plt.ylim((-1,1))
        plt.xlabel('scaled observations #')
        plt.ylabel('scaled features')
        plt.grid()
        plt.legend()
        plt.show()

## ------------------------------------------------------------------------- ##
## setting the basis function for the input functional weights 
def bspline_basis(num_basis, degree=3):
    """
    Create B-spline basis functions of given degree.

    Parameters
    ----------
    num_basis : int
        Number of basis functions.
    degree : int, optional (default=3)
        Spline degree.

    Returns
    -------
    basis_input : list of BSpline
        List of callable basis functions.
    """
    # knot vector with multiplicity at the boundaries
    num_knots = num_basis + degree + 1
    t_t = np.concatenate(([0] * degree, np.linspace(0, 1, num_knots - 2 * degree), [1] * degree))
    basis_input = []
    # basis functions
    for i in range(num_basis):
        # coefficients: only one non-zero entry per basis function
        basis_coefs = np.zeros(num_knots - degree - 1)
        basis_coefs[i] = 1
        basis_input.append(BSpline(t_t, basis_coefs, degree))
    return basis_input

## ------------------------------------------------------------------------- ##
def fourier_basis(num_basis):
    """
    Create Fourier basis functions (constant, sines, and cosines).

    Parameters
    ----------
    num_basis : int
        Total number of basis functions to generate (including the constant term).
        num_basis has to be of form 2n+1.

    Returns
    -------
    basis_input : list of callables
        List of functions f(x) defined on [0, 1] that form the Fourier basis:
        the first function is the constant term, followed by sine and cosine pairs
        of increasing frequency until num_basis functions are produced.
    """
    basis_input = []
    # constant (zero-frequency) term
    basis_input.append(lambda x: np.ones_like(x))

    k = 1
    # generate sine and cosine pairs until we reach the desired number
    while len(basis_input) < num_basis:
        # sine term at frequency k
        basis_input.append(lambda x, k=k: np.sin(2 * np.pi * k * x))
        # check again in case num_basis is odd and we've just hit the limit
        if len(basis_input) >= num_basis:
            break
        # cosine term at frequency k
        basis_input.append(lambda x, k=k: np.cos(2 * np.pi * k * x))
        k += 1
    return basis_input


## ------------------------------------------------------------------------- ##
## load simulated dataset
def load_dataset(filepath, n_features, n_steps):
    """Load dataset from a single CSV file using pandas."""
    df = pd.read_csv(filepath)
    y = df['label'].astype(int).values
    flat_X = df.drop(columns=['label']).values
    n_samples = flat_X.shape[0]
    X = flat_X.reshape(n_samples, n_features, n_steps)
    return X, y

## ------------------------------------------------------------------------- ##