import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

## ------------------------------------------------------------------------- ##
class NearestNeighborsOpt:
    """
    Determine optimal number of nearest neighbors (m) using:
      1. Average m-th neighbor distance curve (sharp jump method)
      2. Graph connectivity

    Parameters
    ----------
    dist_matrix : array-like, shape (n_samples, n_samples)
        Symmetric matrix of pairwise distances.
    """
    """
    Determine an optimal number of nearest neighbors (m)
    using two complementary approaches:
      1. Average m-th neighbor distance curve (sharp jump detection)
      2. Graph connectivity (first m yielding a single connected component)

    Parameters
    ----------
    dist_matrix : array-like, shape (n_samples, n_samples)
        Symmetric matrix of pairwise distances between samples.
    """
    def __init__(self, dist_matrix):
        # Store distance matrix, ensuring it is square
        self.dist_matrix = np.asarray(dist_matrix)
        if self.dist_matrix.shape[0] != self.dist_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        self.n_samples = self.dist_matrix.shape[0]

    def average_mth_distances(self, max_m=None):
        """
        Compute the mean distance to the m-th nearest neighbor for m=1..max_m.

        Parameters
        ----------
        max_m : int, optional
            Maximum neighbor index to consider (default n_samples-1).

        Returns
        -------
        ks : np.ndarray, shape (max_m,)
            Values of m (1-based indices).
        avg_dists : np.ndarray, shape (max_m,)
            Mean of the m-th smallest distances across all samples.
        """
        if max_m is None:
            max_m = self.n_samples - 1
        # Avoid self-distance by setting diagonal to infinity
        D_temp = self.dist_matrix.copy()
        np.fill_diagonal(D_temp, np.inf)
        # Sort each row to find neighbor distances in ascending order
        sorted_dists = np.sort(D_temp, axis=1)
        ks = np.arange(1, max_m + 1)
        # Compute mean of the distance to the m-th neighbor for each m
        avg_dists = np.array([sorted_dists[:, m-1].mean() for m in ks])
        return ks, avg_dists

    def plot_average_mth_distances(self, max_m=None, **plot_kwargs):
        """
        Plot the average distance to the m-th neighbor against m.

        Useful for visual identification of a 'knee' or sharp jump.
        Additional matplotlib keyword args can be passed via plot_kwargs.
        """
        ks, avg_dists = self.average_mth_distances(max_m)
        plt.figure()
        plt.plot(ks, avg_dists, '-o', **plot_kwargs)
        plt.xlabel('m (neighbors)')
        plt.ylabel('Average m-th neighbor distance')
        plt.title('Average m-th Neighbor Distance vs m')
        plt.grid(True)
        plt.show()

    def connectivity(self, max_m=None):
        """
        Compute the number of connected components in the k-nearest neighbor graph
        for each m from 1..max_m.

        Parameters
        ----------
        max_m : int, optional
            Maximum neighbor count to consider (default n_samples-1).

        Returns
        -------
        ks : np.ndarray, shape (max_m,)
            Values of m.
        comps : list of int, length max_m
            Number of connected components in the graph using m neighbors.
        """
        if max_m is None:
            max_m = self.n_samples - 1
        # Prepare a matrix of sorted neighbor indices per sample
        D_temp = self.dist_matrix.copy()
        np.fill_diagonal(D_temp, np.inf)
        sorted_idx = np.argsort(D_temp, axis=1)
        ks = np.arange(1, max_m + 1)
        comps = []
        # For each m, build the graph and count connected components
        for m in ks:
            G = nx.Graph()
            G.add_nodes_from(range(self.n_samples))
            for i in range(self.n_samples):
                neighbors = sorted_idx[i, :m]   # m nearest neighbors of i
                for j in neighbors:
                    G.add_edge(i, j)
            comps.append(nx.number_connected_components(G))
        return ks, comps

    def plot_connectivity(self, max_m=None, **plot_kwargs):
        """
        Plot the number of connected components versus m.

        Useful to see at which m the graph becomes fully connected.
        """
        ks, comps = self.connectivity(max_m)
        plt.figure()
        plt.plot(ks, comps, '-o', **plot_kwargs)
        plt.xlabel('m (neighbors)')
        plt.ylabel('Number of connected components')
        plt.title('Graph Connectivity vs m')
        plt.grid(True)
        plt.show()

    def estimate_optimal_m(self, method='connectivity', max_m=None):
        """
        Estimate optimal neighbor count using one of two methods:

        Parameters
        ----------
        method : {'connectivity', 'avg_distance'}
            - 'connectivity': first m where graph is fully connected (1 component).
            - 'avg_distance': m at which the average-distance curve shows the largest jump.
        max_m : int, optional
            Upper limit for m (default n_samples-1).

        Returns
        -------
        opt_m : int
            Estimated optimal number of neighbors m.
        """
        if max_m is None:
            max_m = self.n_samples - 1

        if method == 'connectivity':
            # Use connectivity criterion
            ks, comps = self.connectivity(max_m)
            for m, c in zip(ks, comps):
                if c == 1:
                    return int(m)
            return int(ks[-1])

        elif method == 'avg_distance':
            # Use sharp jump in average-distance curve
            ks, avg_dists = self.average_mth_distances(max_m)
            # Identify the m where the increase (jump) in average distance is maximal
            diffs = np.diff(avg_dists)
            idx = np.argmax(diffs)      # index of largest jump
            ## or the first sharp jump
            # idx = np.where(diffs > 0.1*len(diffs))[0][0]
            # idx corresponds to jump from m=ks[idx] to ks[idx+1]
            return int(ks[idx+1])       # neighbor count where jump occurs

        else:
            raise ValueError("method must be 'connectivity' or 'avg_distance'")
    
    def get_nearest_neighbors(self, opt_m=None):
        """
        Build a neighbors_dict mapping each sample to its k nearest neighbors.

        Parameters
        ----------
        opt_m : int, optional
            Number of neighbors; if None, estimates via avg_distance.

        Returns
        -------
        neighbors_dict : dict
            Mapping i -> array of opt_m nearest neighbor indices.
        """
        if opt_m is None:
            # auto-estimate m via avg_distance method
            ks, avg_dists = self.average_mth_distances(max_m=150)
            diffs = np.diff(avg_dists)
            idx = np.argmax(diffs)
            opt_m = int(ks[idx+1])
            
        N = self.dist_matrix.shape[0]
        neighbors_dict = {}
        for i in range(N):
            # sort indices by distance, exclude self
            neighbors = np.argsort(self.dist_matrix[i])
            neighbors = neighbors[neighbors != i]
            neighbors_dict[i] = neighbors[:opt_m]
        return neighbors_dict
    
    def compute_similarity(self, neighbors_dict, method='neighbors'):
        """
        Compute a dense similarity matrix from distances.

        Two modes:
        - 'neighbors': only neighbor edges have non-zero similarity.
        - 'distance': all pairs weighted by a Gaussian kernel.

        Parameters
        ----------
        neighbors_dict : dict[int, Sequence[int]]
            Mapping i -> list of neighbor indices.
        method : {'neighbors', 'distance'}
            Similarity computation mode.

        Returns
        -------
        sim : np.ndarray, shape (n_samples, n_samples)
            Symmetric similarity matrix.
        """
        n = self.dist_matrix.shape[0]
        # Build boolean mask for neighbor relationships
        mask = np.zeros((n, n), dtype=bool)
        for i, nbrs in neighbors_dict.items():
            mask[i, nbrs] = True
        mask |= mask.T          # make symmetric

        # Compute Gaussian similarities for all pairs
        # sigma = np.median(self.dist_matrix)
        sigma = np.std(self.dist_matrix)
        sim = np.exp(- (self.dist_matrix ** 2) / (2 * sigma ** 2), out=np.empty_like(self.dist_matrix))
        # sim = np.exp(- (self.dist_matrix))
        if method == 'neighbors':
            # Zero out non-neighbor pairs
            sim[~mask] = 0
            return sim
        elif method == 'distance':
            # Keep all similarities, but zero out self-similarity
            sim[np.eye(n, dtype=bool)] = 0
            return sim
        else:
            raise ValueError("method must be 'neighbors' or 'distance'")
            
## ------------------------------------------------------------------------- ##

if __name__ == "__main__":
    # Example usage: load data, compute distance matrix, and find optimal m
    from aeon.datasets import load_classification
    from srvf import TimeSeriesDistance          # custom TS distance calculator
    
    # Load classification dataset 'BME', get time-series X and labels y
    X, y = load_classification('BME')
    # Compute pairwise distances via fastDTW
    tsd = TimeSeriesDistance(X, metric='fastdtw', n_jobs=-1)
    D = tsd.compute_distances()

    max_m = X.shape[0]-1
    opt = NearestNeighborsOpt(D)
    # Method 1: plot and estimate via average-distance jump
    opt.plot_average_mth_distances(max_m=max_m)
    k1 = opt.estimate_optimal_m(method='avg_distance', max_m=max_m)
    print(f"Optimal m (avg_distance): {k1}")
    neighbors_dict = opt.get_nearest_neighbors(opt_m=k1)
    
    # Method 2: plot and estimate via connectivity
    opt.plot_connectivity(max_m=max_m)
    k2 = opt.estimate_optimal_m(method='connectivity', max_m=max_m)
    print(f"Optimal m (connectivity): {k2}")

## ------------------------------------------------------------------------- ##