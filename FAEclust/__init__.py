"""FAEclust — clustering of multi-dimensional functional data (v1.0.0).

FAEclust uses a deep functional autoencoder to learn a shape-aware latent
representation of functional / time-series data, then clusters the data in that
latent space. The functional autoencoder (`FAE`) and the manifold readout
(`manifolds`) are built on TensorFlow/Keras; the remaining modules (`srvf`,
`mnn`, `smoothing`, `utils`, `fista`, `DatasetGenerator`) use plain
NumPy/Numba.
"""

# Quiet TensorFlow's C++ startup banner (oneDNN / device INFO logs) and Python
# warnings before TensorFlow is imported below. These are set as environment
# variables (rather than a Python logging call) so they also take effect in
# worker subprocesses that re-import this package during parallel computation,
# which otherwise repeat the banner many times. Set TF_CPP_MIN_LOG_LEVEL=0
# before importing to restore the verbose logs.
import os as _os
import warnings as _warnings
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
_os.environ.setdefault('PYTHONWARNINGS', 'ignore')
_warnings.filterwarnings('ignore')

from .utils import (
    smoothing_features,
    standardize_functional,
    bspline_basis,
    fourier_basis,
    rescale,
    load_dataset,
)
from .smoothing import Smoothing
from .srvf import TimeSeriesDistance
from .mnn import NearestNeighborsOpt
from .fista import ConvexClustering, fista_solve
from .manifolds import DatasetGenerator, ManifoldReadout
from .FAE import FunctionalAutoencoder, pick_device
from .runner import run_experiment, MANIFOLD_DATASETS

__version__ = "1.0.0"

__all__ = [
    "run_experiment",
    "MANIFOLD_DATASETS",
    "pick_device",
    "smoothing_features",
    "standardize_functional",
    "bspline_basis",
    "fourier_basis",
    "rescale",
    "load_dataset",
    "Smoothing",
    "TimeSeriesDistance",
    "NearestNeighborsOpt",
    "ConvexClustering",
    "fista_solve",
    "DatasetGenerator",
    "ManifoldReadout",
    "FunctionalAutoencoder",
]

# Optional Optuna tuning is imported lazily; see FAEclust.tuning
