import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

class TimeSeriesDistance:
    """
    Compute pairwise distances for multivariate time series using:
      - Plain multivariate DTW (fastdtw)
      - Elastic amplitude distance via SRVF + reparameterization

    Parameters
    ----------
    X : np.ndarray, shape=(n_samples, n_features, n_timesteps)
        Raw multivariate time series data.
    metric : {'fastdtw', 'elastic'}, default='elastic'
        Distance metric to use:
        - 'fastdtw': standard multivariate DTW via fastdtw
        - 'elastic': functional elastic amplitude distance using SRVF
    n_jobs : int, default=-1
        Number of parallel jobs for distance matrix computation.
    """
    def __init__(self,
                 X: np.ndarray,
                 metric: str = 'elastic',
                 n_jobs: int = -1):
        # Store raw and configuration
        self.X_raw = X
        self.metric = metric
        self.n_jobs = n_jobs
        # Unpack dimensions
        self.n_samples, self.n_features, self.n_timesteps = X.shape

        # Standardize each feature across all samples and timesteps
        eps = np.finfo(float).eps
        Xs = np.empty_like(X, dtype=float)
        for f in range(self.n_features):
            feat = X[:, f, :].ravel()
            mu, sigma = feat.mean(), feat.std() + eps
            Xs[:, f, :] = (X[:, f, :] - mu) / sigma
        self.X = Xs

        # uniform time grid [0,1]
        self.times = np.linspace(0, 1, self.n_timesteps)

        # placeholder for the distance matrix
        self.D = None

    @staticmethod
    def _curve_to_srvf(curve: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Convert a multivariate curve to its SRVF (Square-Root Velocity Function).

        q_f(t) = f'(t) / sqrt(||f'(t)||)

        Parameters
        ----------
        curve : np.ndarray, shape=(F, T)
            Multivariate curve with F features over T timesteps.
        t : np.ndarray, shape=(T,)
            Corresponding time grid.

        Returns
        -------
        q : np.ndarray, shape=(F, T)
            SRVF of the input curve.
        """
        # Compute derivative per feature along time
        dX = np.vstack([np.gradient(curve[f], t) for f in range(curve.shape[0])])
        eps = np.finfo(float).eps
        # Speed is the L2 norm across features at each timepoint
        speed = np.linalg.norm(dX, axis=0) + eps  # (T,)
        # Normalize velocity by sqrt(speed)
        return dX / np.sqrt(speed)               # (F×T)

    def _elastic_distance(self, Xi: np.ndarray, Xj: np.ndarray) -> float:
        """
        Compute elastic amplitude distance between two curves using SRVF + reparametrization.

        Steps:
          1. Map Xi, Xj → SRVF curves q1, q2
          2. Use fastdtw on q1, q2 to get optimal reparameterization path
          3. Build gamma(t): average matched times for each t
          4. Warp q2 by gamma and weight by sqrt(gamma')
          5. Compute L2 distance between q1 and warped q2 over time

        Returns
        -------
        Dy : float
            Elastic amplitude distance between Xi and Xj.
        """
        eps = np.finfo(float).eps

        # 1) Compute SRVF representations
        q1 = self._curve_to_srvf(Xi, self.times)  # F×T
        q2 = self._curve_to_srvf(Xj, self.times)  # F×T

        # 2) Alignment on the SRVF curves (transpose to time×feature)
        _, path = fastdtw(q1.T, q2.T, dist=euclidean)

        # 3) Build warping function gamma: for each time index i, average matched js
        gamma = np.zeros(self.n_timesteps)
        for i in range(self.n_timesteps):
            js = [j for (ii, j) in path if ii == i]
            gamma[i] = np.mean(self.times[js]) if js else self.times[i]
        # enforce monotonicity
        gamma = np.maximum.accumulate(gamma)
        # normalize gamma to [0, 1] 
        g0, g1 = float(gamma[0]), float(gamma[-1])
        if g1 - g0 < 1e-12:
            # Degenerate mapping; fall back to identity reparameterization
            gamma = self.times.copy()
        else:
            gamma = (gamma - g0) / (g1 - g0)
        # Ensure exact endpoints for stability with np.interp and gradient
        gamma[0] = 0.0
        gamma[-1] = 1.0

        # 4) Warp q2 by gamma and weight by sqrt(gamma')
        # Interpolate each feature of q2 at warped times
        qw = np.vstack([
            np.interp(gamma, self.times, q2[f])
            for f in range(self.n_features)
        ])  # F×T
        # Compute derivative of gamma wrt original times
        dgam = np.gradient(gamma, self.times)
        # Clip small negatives, add eps, then sqrt
        dgam_clipped = np.clip(dgam, 0, None)
        sqrt_dgam = np.sqrt(dgam_clipped + eps)
        # Weight warped SRVF by sqrt(gamma')
        qw *= sqrt_dgam  # broadcast

        # 5) Compute amplitude distance Dy = sqrt(integral ||q1 - qw||^2 dt)
        diff2 = np.sum((q1 - qw)**2, axis=0)
        Dy = np.sqrt(np.trapz(diff2, self.times))
        return Dy

    def _multivariate_dtw(self, Xi: np.ndarray, Xj: np.ndarray) -> float:
        """Plain multivariate DTW distance via fastdtw on feature vectors."""
        return fastdtw(Xi.T, Xj.T, dist=euclidean)[0]

    def _pairwise(self, i: int, j: int) -> float:
        """
        Compute the selected metric between series i and j.
        """
        Xi = self.X[i]
        Xj = self.X[j]
        if self.metric == 'fastdtw':
            return self._multivariate_dtw(Xi, Xj)
        elif self.metric == 'elastic':
            return self._elastic_distance(Xi, Xj)
        else:
            raise ValueError(f"Unknown metric {self.metric!r}")

    def compute_distances(self) -> np.ndarray:
        """
        Compute the full pairwise distance matrix D.

        Returns
        -------
        D : np.ndarray, shape=(n_samples, n_samples)
            Symmetric distance matrix with zeros on the diagonal.
        """
        def row(i):
            # Compute distances from sample i to all j>i
            r = np.zeros(self.n_samples)
            for j in range(i+1, self.n_samples):
                r[j] = self._pairwise(i, j)
            return r
        
        # Parallelize row computations and stack into upper-triangular matrix
        self.D = np.vstack(Parallel(n_jobs=self.n_jobs)(
            delayed(row)(i) for i in range(self.n_samples)
        ))
        # Symmetrize
        self.D_upper = self.D
        self.D = self.D_upper + self.D_upper.T
        np.fill_diagonal(self.D, 0.0)
        return self.D

    def most_similar(self):
        """
        Return the pair of series with the smallest non-zero distance.

        Returns
        -------
        (i, j, d) : (int, int, float)
            Indices and distance of the most similar pair.
        """
        if self.D is None:
            raise RuntimeError("Call compute_distances() first.")
        # Mask out diagonal and lower triangle
        mask = np.triu(np.ones_like(self.D, bool), k=1)
        idx = np.argmin(self.D[mask])
        coords = np.argwhere(mask)
        i, j = coords[idx]
        return i, j, float(self.D[i, j])

    def most_different(self):
        """
        Return the pair of series with the largest distance.

        Returns
        -------
        (i, j, d) : (int, int, float)
            Indices and distance of the most different pair.
        """
        if self.D is None:
            raise RuntimeError("Call compute_distances() first.")
        mask = np.triu(np.ones_like(self.D, bool), k=1)
        idx = np.argmax(self.D[mask])
        coords = np.argwhere(mask)
        i, j = coords[idx]
        return i, j, float(self.D[i, j])

    
    def plot_extremes(self):
        """
        Plot the most similar and most different time series pairs,
        using a custom colour cycle of length = self.n_features.
        """
        if self.D is None:
            raise RuntimeError("Call compute_distances() first.")
        # Identify indices
        i1, j1, _ = self.most_similar()
        i2, j2, _ = self.most_different()
        t = np.arange(self.n_timesteps)
        
        # choose a colormap: tab10 for up to 10 features, else tab20
        cmap_name = 'tab10' if self.n_features <= 10 else 'tab20'
        cmap = plt.get_cmap(cmap_name, self.n_features)
        # build exactly n_features colours
        colour_cycle = [cmap(i) for i in range(self.n_features)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # most similar
        for f in range(self.n_features):
            color = colour_cycle[f]
            ax1.plot(t, self.X_raw[i1, f],      color=color, label=f"feature {f+1}")
            ax1.plot(t, self.X_raw[j1, f], '--', color=color)
        ax1.set_title(f"Most similar: indices {i1} vs {j1}")
        ax1.legend(loc='best')
        
        # most distinct
        for f in range(self.n_features):
            color = colour_cycle[f]
            ax2.plot(t, self.X_raw[i2, f],      color=color, label=f"feature {f+1}")
            ax2.plot(t, self.X_raw[j2, f], '--', color=color)
        ax2.set_title(f"Most distinct: indices {i2} vs {j2}")
        ax2.legend(loc='best')
        
        # plt.tight_layout() 
        plt.show()

## ------------------------------------------------------------------------- ##
if __name__ == "__main__":
    # Example usage: load data, and compute distance matrix
    from aeon.datasets import load_classification
    import time 
    
    # Load classification dataset 'BME', get time-series X and labels y
    X, y = load_classification('BME')
    
    # Method 1: 
    st = time.time()
    tsd = TimeSeriesDistance(X, metric='elastic', n_jobs=-1)
    D = tsd.compute_distances()
    en = time.time()
    tsd.plot_extremes()
    print(f'time for elastic: {en -st}')
    
    # Method 2: 
    st = time.time()
    tsd = TimeSeriesDistance(X, metric='fastdtw', n_jobs=-1)
    D = tsd.compute_distances()
    en = time.time()
    tsd.plot_extremes()
    print(f'time for fast DTW: {en -st}')
    
## ------------------------------------------------------------------------- ##
    
    

