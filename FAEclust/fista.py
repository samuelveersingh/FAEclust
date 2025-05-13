import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score

## ------------------------------------------------------------------------- ##
class ConvexClustering:
    """
    Convex clustering with homotopy path construction and silhouette-based model selection.

    Parameters
    ----------
    X : array-like, shape (N, d)
        Data matrix where each row is a point in d-dimensional space.
    neighbors_dict : dict
        Mapping i -> list of neighbor indices for point i, defining edges in the graph.
    sim_matrix : array-like, shape (N, N)
        Symmetric matrix of pairwise similarity weights; used to weight edges.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs for computing threshold values across edges.
    verbose : bool, optional (default=False)
        If True, prints fusion steps with lambda, cluster count, silhouette score, etc.
    """
    def __init__(self, X, neighbors_dict, sim_matrix, n_jobs=-1, verbose=False):
        self.X = np.asarray(X)
        self.neighbors_dict = neighbors_dict
        self.sim_matrix = np.asarray(sim_matrix)
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Attributes to be set by fit()
        self.lambdas = None               # list of fusion thresholds
        self.cluster_labels_path = None   # cluster labels after each fusion
        self.silhouette_scores = None     # silhouette score per valid fusion
        self.valid_lambdas = None         # lambdas with >=2 clusters
        self.valid_labels = None          # corresponding labels
        self.best_lambda = None           # lambda with highest silhouette
        self.best_labels = None           # labels at best_lambda

    def _compute_homotopy_path(self):
        """
        Build the homotopy path by gradually fusing clusters using Kruskal-like union-find.

        Returns
        -------
        lambdas : list of float
            Sorted threshold values at which fusions occur.
        labels_path : list of list of int
            Cluster labels (0..K-1) after each fusion step, starting with each point separate.
        """
        N, _ = self.X.shape
        # 1. build unique undirected edge list + weights
        edges, weights, seen = [], [], set()
        for i, neighs in self.neighbors_dict.items():
            for j in neighs:
                if i == j: 
                    continue        # skip self-loops
                a, b = min(i, j), max(i, j)
                if (a, b) in seen:
                    continue        # avoid duplicates
                seen.add((a, b))
                edges.append((a, b))
                # weight: prefer sim_matrix[a,b], fallback to sim_matrix[b,a] or 1.0
                w = self.sim_matrix[a, b] or self.sim_matrix[b, a] or 1.0
                weights.append(w)
        edges = np.array(edges, dtype=int)
        weights = np.array(weights, dtype=float)

        # 2. Compute fusion thresholds λ_e = ||X_i - X_j|| / (2 w_e)
        def thr(idx):
            i, j = edges[idx]
            return np.linalg.norm(self.X[i] - self.X[j]) / (2.0 * weights[idx])

        thr_vals = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(thr)(idx) for idx in range(len(edges))
        )
        thr_vals = np.array(thr_vals)

        # 3. Sort edges by ascending threshold
        order = np.argsort(thr_vals)
        sorted_edges = edges[order]
        sorted_thr   = thr_vals[order]

        # 4. Union-Find to fuse clusters at each threshold
        parent = list(range(N))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        def union(u, v):
            ru, rv = find(u), find(v)
            if ru == rv:
                return False        # already in same set
            parent[rv] = ru
            return True

        lambdas = []
        labels_path = []
        current_count = N

        # Initial labels: each point in its own cluster
        labels_path.append(list(range(N)))

        # Fuse edges one by one in threshold order
        for lam, (i, j) in zip(sorted_thr, sorted_edges):
            if union(i, j):
                lambdas.append(lam)
                # Build raw parent labels, then relabel to 0..K-1
                raw = [find(u) for u in range(N)]
                remap, next_lbl = {}, 0
                for idx, r in enumerate(raw):
                    if r not in remap:
                        remap[r] = next_lbl
                        next_lbl += 1
                    raw[idx] = remap[r]
                labels_path.append(raw)
                current_count -= 1
                if current_count == 1:
                    break        # all fused into single cluster

        return lambdas, labels_path

    def fit(self):
        """
        Compute the homotopy path, score each clustering by silhouette,
        and select the best lambda (highest silhouette score).

        Returns
        -------
        best_lambda : float or None
        best_labels : list of int or None
            Cluster labels at the optimal lambda, or None if no valid split.
        """
        # 1. Build homotopy path of lambdas and labelings
        self.lambdas, self.cluster_labels_path = self._compute_homotopy_path()

        # 2. Evaluate silhouette for each fusion step with >=2 clusters
        sil_scores = []
        valid_lams = []
        valid_lbls = []
        for idx, lam in enumerate(self.lambdas):
            labels = self.cluster_labels_path[idx + 1]
            K = len(set(labels))
            if K >= 2:
                score = silhouette_score(self.X, labels)
                sil_scores.append(score)
                valid_lams.append(lam)
                valid_lbls.append(labels)

        self.silhouette_scores = sil_scores
        self.valid_lambdas     = valid_lams
        self.valid_labels      = valid_lbls
        
        # 3. Choose lambda with maximum silhouette score
        if self.valid_lambdas:
            best_idx = int(np.argmax(sil_scores))
            self.best_lambda = self.valid_lambdas[best_idx]
            self.best_labels = self.valid_labels[best_idx]
        else:
            self.best_lambda = None
            self.best_labels = None

        # 4. Optional verbose reporting
        if self.verbose:
            for idx, lam in enumerate(self.lambdas[:-1]):
                labels = self.cluster_labels_path[idx + 1]
                K = len(set(labels))
                score = self.silhouette_scores[idx]
                if score is not None:
                    print(f"λ = {lam:.6f} | clusters = {K} | silhouette = {score:.4f}")
                else:
                    print(f"λ = {lam:.6f} | clusters = {K} | silhouette = undefined")
                # print(f"  labels: {labels}\n")
            if self.best_lambda is not None:
                print(f"→ best λ = {self.best_lambda:.6f} with silhouette = "
                      f"{max(self.silhouette_scores):.4f}, clusters = {len(set(self.best_labels))}")
            else:
                print("No valid silhouette scores (never had ≥2 clusters).")

        return self.best_labels

## ------------------------------------------------------------------------- ##


