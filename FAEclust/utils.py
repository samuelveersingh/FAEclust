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
def smoothing_features(data, m=20, dis_p=600, fit='bspline'):
    """
    Smooth each feature dimension via chosen basis and return coefficients.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, p, T)
        Input functional data to be smoothed.
    m : int, optional
        Number of basis terms (default=20).
    dis_p : int, optional
        Number of fine evaluation grid points (default=600).
    fit : {'bspline', 'fourier', ...}
        Basis type passed to Smoothing.

    Returns
    -------
    coeffs : np.ndarray, shape (n_samples, p, m)
        Basis coefficients for each sample-feature.
    all_curves : np.ndarray, shape (T, p)
        Smoothed curves evaluated at coarse grid T.
    basis_smoothing : list
        List of basis functions used for smoothing.
    """
    n, p, T = data.shape
    coeffs_list = []
    curves_list = []

    for i in range(p):
        smoothing = Smoothing(
            dis_p=dis_p,
            fit=fit,
            terms=m,
            data=data[:, i, :]
        )
        coeffs_list.append(smoothing.coeffs)  # (n, m)
        curves_list.append(smoothing.fn_s)                      # callable over t

    # stack coeffs → (p, n, m) → transpose → (n, p, m)
    coeffs = np.stack(coeffs_list, axis=0).transpose(1, 0, 2).astype('float32')

    # curves_list: list of p callables fn_s(t). Evaluate each over grid t to get (T, p)
    all_curves = np.stack(curves_list, axis=0).T  # (T, p)

    basis_smoothing = smoothing.smoothing_basis
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