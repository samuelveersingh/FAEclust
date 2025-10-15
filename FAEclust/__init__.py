# expose the main functions & classes at package level
from .utils     import smoothing_features, bspline_basis, load_dataset, rescale
from .manifolds import DatasetGenerator
from .srvf      import TimeSeriesDistance
from .mnn       import NearestNeighborsOpt
from .FAE       import FunctionalAutoencoder


__all__ = [
    "smoothing_features", "bspline_basis", "load_dataset", "rescale",
    "DatasetGenerator",
    "TimeSeriesDistance",
    "NearestNeighborsOpt",
    "FunctionalAutoencoder",
]
