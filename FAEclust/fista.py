"""Convex clustering of latent points.

This groups points into clusters by giving each point its own moving
"center" and pulling nearby centers together. A strength setting (lambda)
controls how strongly centers are pulled: when it is small every point is its
own cluster, and as it grows centers merge until everything is one cluster.

For a fixed strength, each center solves this trade-off (one term keeps it
near its own point, the other pulls connected centers together):

    L_s(U, X) = (1/n) ||X - U||_F^2 + lambda * sum_{i<j} s(y_i,y_j) ||u_i-u_j||_1

* ``fista_solve``  — finds the center positions for a given strength using
  an iterative optimizer (FISTA).
* ``ConvexClustering`` — gradually increases the strength so clusters merge
  one at a time, recording the clustering at every merge, and then picks the
  clustering that scores best by a quality measure (silhouette or
  Davies-Bouldin). It tracks which points belong to each cluster, how strongly
  clusters are connected, and where each center moves as the strength grows, so
  it can compute exactly when the next two clusters will meet.
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


# --------------------------------------------------------------------------- #
def _edges_from_neighbors(neighbors_dict, sim_matrix, n):
    """Turn the nearest-neighbor graph into a list of unique edges (each pair
    counted once) with their similarity weights."""
    seen, ei, ej, w = set(), [], [], []
    S = np.asarray(sim_matrix)
    for i, nbrs in neighbors_dict.items():
        for j in np.asarray(nbrs, dtype=int):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            wij = S[a, b] if S[a, b] != 0 else S[b, a]
            if wij <= 0:
                continue
            ei.append(a); ej.append(b); w.append(float(wij))
    return np.array(ei, int), np.array(ej, int), np.array(w, float)


def fista_solve(X, edges_i, edges_j, w, lam, n=None, n_iter=500, tol=1e-7):
    """
    Find the center positions U that balance staying close to each point X
    against pulling connected centers together, for a given pull strength lam:

        min_U (1/n)||X-U||_F^2 + lam * sum_e w_e ||u_{i_e}-u_{j_e}||_1

    Solved with an iterative optimizer (FISTA).

    Parameters
    ----------
    X : (n, s) point positions in the latent space.
    edges_i, edges_j : (E,) endpoints of each edge.
    w : (E,) similarity weight of each edge.
    lam : float, pull strength (larger -> more merging).
    n : int, number of points (used in the 1/n scaling); defaults to X.shape[0].

    Returns
    -------
    U : (n, s) center positions.
    """
    X = np.asarray(X, float)
    N, s = X.shape
    n = N if n is None else n
    E = len(edges_i)
    if E == 0 or lam == 0:
        return X.copy()

    # Dmul: for each edge, the difference between its two endpoints -> (E, s).
    # Dtmul: the reverse, spreading edge values back onto the points -> (N, s).
    def Dmul(U):
        return U[edges_i] - U[edges_j]

    def Dtmul(Z):
        out = np.zeros((N, s))
        np.add.at(out, edges_i, Z)
        np.add.at(out, edges_j, -Z)
        return out

    # Instead of optimizing the centers directly, we optimize a helper variable
    # Z (one value per edge, kept within +/- lam*w per edge) and recover the
    # centers as U = X - (n/2) D^T Z. The optimizer needs a step size, which
    # depends on how strongly the edges interact (a quantity we estimate next).
    DX = Dmul(X)
    # Estimate that interaction strength by repeatedly applying the edge maps
    # and measuring how much the vector grows (power iteration).
    v = np.random.default_rng(0).standard_normal((E, s))
    nv = 1.0
    for _ in range(100):
        v = Dmul(Dtmul(v))
        nv = np.linalg.norm(v) + 1e-12
        v /= nv
    L = (n / 2.0) * nv + 1e-9
    step = 1.0 / L

    Z = np.zeros((E, s))
    Y = Z.copy()
    t = 1.0
    cap = lam * w[:, None]                            # per-edge limit on Z
    for _ in range(n_iter):
        grad = DX - (n / 2.0) * Dmul(Dtmul(Y))      # direction to move Z
        Z_new = np.clip(Y + step * grad, -cap, cap)  # step, then keep Z within its limits
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        Y = Z_new + ((t - 1.0) / t_new) * (Z_new - Z)
        if np.linalg.norm(Z_new - Z) <= tol * (np.linalg.norm(Z) + 1e-12):
            Z = Z_new
            break
        Z, t = Z_new, t_new
    U = X - (n / 2.0) * Dtmul(Z)
    return U


# --------------------------------------------------------------------------- #
class ConvexClustering:
    """
    Convex clustering that gradually merges clusters and keeps the best result.

    It starts with every point as its own cluster, slowly increases the pull
    strength so the closest clusters merge one at a time, and records the
    clustering after each merge. It then picks the clustering that scores best
    by a quality measure.

    Parameters
    ----------
    X : (n, s) point positions in the latent space.
    neighbors_dict : dict mapping each point index to its neighbor indices.
    sim_matrix : (n, n) similarity weights between points.
    verbose : bool
        Print the merge steps and their quality scores.
    criterion : {'silhouette', 'davies_bouldin'}
        Quality measure used to choose the number of clusters.
    """

    def __init__(self, X, neighbors_dict, sim_matrix, n_jobs=-1,
                 verbose=False, criterion='silhouette'):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X[:, None]
        self.neighbors_dict = neighbors_dict
        self.sim_matrix = np.asarray(sim_matrix, dtype=float)
        self.verbose = verbose
        self.criterion = criterion

        self.lambdas = None
        self.cluster_labels_path = None
        self.scores = None
        self.best_lambda = None
        self.best_labels = None

    # -- merge clusters step by step ----------------------------------------
    def _compute_homotopy_path(self):
        X = self.X
        N, s = X.shape
        ei, ej, w = _edges_from_neighbors(self.neighbors_dict, self.sim_matrix, N)

        # quick lookup of the similarity weight for each connected pair
        Wmap = {}
        for a, b, wab in zip(ei, ej, w):
            Wmap[(int(a), int(b))] = Wmap.get((int(a), int(b)), 0.0) + wab

        # cluster state: which points are in each cluster, and where each
        # cluster's center currently sits
        members = {k: [k] for k in range(N)}
        a = {k: X[k].copy() for k in range(N)}
        active = set(range(N))
        lam0 = 0.0       # current pull strength

        lambdas = []
        labels_path = [self._labels_from_members(members, N)]

        def affinity(k, v):
            # total similarity connecting two clusters, summed over the edges
            # between their members
            ssum = 0.0
            mk, mv = members[k], members[v]
            # loop over the smaller cluster for speed
            if len(mk) > len(mv):
                mk, mv = mv, mk
            mvset = set(mv)
            for i in mk:
                nb = self.neighbors_dict.get(i, [])
                for j in nb:
                    j = int(j)
                    if j in mvset:
                        p = (i, j) if i < j else (j, i)
                        ssum += Wmap.get(p, 0.0)
            return ssum

        while len(active) > 1:
            act = list(active)
            # similarity between every pair of current clusters that is connected
            pair_aff = {}
            for idx_k in range(len(act)):
                for idx_v in range(idx_k + 1, len(act)):
                    k, v = act[idx_k], act[idx_v]
                    skv = affinity(k, v)
                    if skv > 0:
                        pair_aff[(k, v)] = skv
            if not pair_aff:
                break  # nothing left is connected: remaining clusters never merge

            # velocity of each cluster's center: as the pull strength grows,
            # each center drifts toward the clusters it is connected to (faster
            # for smaller clusters and stronger connections)
            b = {k: np.zeros(s) for k in act}
            for (k, v), skv in pair_aff.items():
                diff = a[k] - a[v]
                nrm = np.linalg.norm(diff)
                direction = diff / nrm if nrm > 1e-12 else np.zeros(s)
                b[k] += -(N / (2.0 * len(members[k]))) * skv * direction
                b[v] += -(N / (2.0 * len(members[v]))) * skv * (-direction)

            # find how much more pull strength (tau) is needed before the next
            # pair of centers meets, and which pair meets first
            best_tau, best_pair = np.inf, None
            for (k, v) in pair_aff:
                d0 = a[k] - a[v]
                db = b[k] - b[v]
                den = float(db @ db)
                if den < 1e-18:
                    continue
                tau = -float(d0 @ db) / den
                if tau > 1e-12:
                    resid = np.linalg.norm(d0 + tau * db)
                    if resid < 1e-6 * (np.linalg.norm(d0) + 1e-9) and tau < best_tau:
                        best_tau, best_pair = tau, (k, v)
            if best_pair is None:
                # backup: just take the pair that meets soonest
                for (k, v) in pair_aff:
                    d0 = a[k] - a[v]; db = b[k] - b[v]
                    den = float(db @ db)
                    if den < 1e-18:
                        continue
                    tau = -float(d0 @ db) / den
                    if tau > 1e-12 and tau < best_tau:
                        best_tau, best_pair = tau, (k, v)
            if best_pair is None:
                break

            lam_new = lam0 + best_tau
            # move every active center forward to where the next merge happens
            for k in act:
                a[k] = a[k] + best_tau * b[k]
            # merge the two clusters that just met
            k, v = best_pair
            new_members = members[k] + members[v]
            wk, wv = len(members[k]), len(members[v])
            a[k] = (wk * a[k] + wv * a[v]) / (wk + wv)
            members[k] = new_members
            del members[v]; del a[v]
            active.discard(v)
            lam0 = lam_new

            lambdas.append(lam_new)
            labels_path.append(self._labels_from_members(members, N))

        return lambdas, labels_path

    @staticmethod
    def _labels_from_members(members, N):
        lab = np.empty(N, dtype=int)
        for new_id, (_, idxs) in enumerate(sorted(members.items())):
            for i in idxs:
                lab[i] = new_id
        return lab

    # -- run clustering and choose the best number of clusters --------------
    def fit(self):
        self.lambdas, self.cluster_labels_path = self._compute_homotopy_path()

        scores, valid = [], []
        n = len(self.X)
        for step, labels in enumerate(self.cluster_labels_path):
            labels = np.asarray(labels)
            K = len(set(labels))
            # Skip uninformative clusterings: require between 2 and n-1 clusters
            # and at least 2 points in every cluster. Clusterings with lone
            # single-point clusters make the quality scores meaningless, so they
            # are never chosen.
            min_size = np.min(np.bincount(labels))
            if 2 <= K < n and min_size >= 2:
                if self.criterion == 'davies_bouldin':
                    sc = -davies_bouldin_score(self.X, labels)  # negate so higher = better
                else:
                    sc = silhouette_score(self.X, labels)
                scores.append(sc)
                valid.append((step, labels))
        self.scores = scores

        if valid:
            best = int(np.argmax(scores))
            step, labels = valid[best]
            self.best_labels = labels
            self.best_lambda = (self.lambdas[step - 1] if 0 < step <= len(self.lambdas)
                                else 0.0)
        else:
            self.best_labels = self.cluster_labels_path[-1]
            self.best_lambda = self.lambdas[-1] if self.lambdas else 0.0

        if self.verbose:
            for (step, labels), sc in zip(valid, scores):
                print(f"step {step:3d} | clusters = {len(set(labels)):3d} "
                      f"| {self.criterion} = {sc:.4f}")
            print(f"-> best: {len(set(self.best_labels))} clusters "
                  f"(lambda ~ {self.best_lambda:.6g})")
        return self.best_labels

## ------------------------------------------------------------------------- ##
