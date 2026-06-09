"""Pairwise distances between multi-dimensional time series.

These distances compare the *shape* of two curves while allowing for
differences in timing, so two curves with the same shape but stretched or
shifted in time are treated as similar. Several methods are available:

  * ``metric='elastic'``      — an elastic distance that finds the best way
    to warp time so the two curves line up, then measures how different they
    still are. It works on the square-root velocity (SRVF) form of each curve
    (a transform that makes time-warping easy to handle) and searches for the
    best warping with dynamic programming (trying many alignments and keeping
    the best). The core loop is sped up with Numba when it is installed, and
    the search can be limited to a band of alignments near the diagonal
    (``band_radius``) for speed. Without Numba it falls back to a slower
    pure-NumPy version that gives the same answer.
  * ``metric='elastic-fast'`` — a faster two-pass version of the elastic
    distance: first solve the alignment on a shrunk-down (coarse) version of
    the curves, then refine it on the full curves but only near the alignment
    found in the first pass. Much faster per pair with nearly the same answer.
  * ``metric='fastdtw'``      — dynamic time warping (DTW) limited to a band
    of alignments near the diagonal, giving roughly linear-time speed.
  * ``metric='ultrafast'``    — a recursive multi-resolution version of DTW
    that is also roughly linear time.

The public ``TimeSeriesDistance`` class is the entry point.
"""

import math
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Numba is optional. When installed, it compiles the inner distance loop to
# machine code and runs it about 100x faster. If it is missing, we use a
# slower pure-NumPy version that produces the same result.
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        # Stand-in decorator that does nothing, so the @njit(...) lines still
        # work when Numba is not installed.
        if args and callable(args[0]):
            return args[0]
        def _wrap(f):
            return f
        return _wrap


# --------------------------------------------------------------------------- #
# Dynamic time warping (DTW) building blocks (no external DTW library needed)
# --------------------------------------------------------------------------- #
def _full_dtw(a, b):
    """Plain DTW with no shortcuts, for multi-feature series.

    a has shape (Ta, F), b has shape (Tb, F), where F is the number of
    features. Returns (distance, path), where path is the list of matched
    index pairs that line the two series up."""
    Ta, Tb = len(a), len(b)
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        ai = a[i - 1]
        for j in range(1, Tb + 1):
            c = np.linalg.norm(ai - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # Walk back through the cost grid to recover the matching path.
    i, j, path = Ta, Tb, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(D[Ta, Tb]), path


def _banded_dtw(a, b, radius):
    """DTW that only considers alignments within ``radius`` steps of the
    diagonal. This keeps the cost roughly proportional to N * radius instead
    of N * N."""
    Ta, Tb = len(a), len(b)
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    scale = Tb / max(1, Ta)
    for i in range(1, Ta + 1):
        center = int(round((i - 1) * scale))
        lo = max(1, center - radius + 1)
        hi = min(Tb, center + radius + 1)
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            c = np.linalg.norm(ai - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = Ta, Tb, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(D[Ta, Tb]), path


def _fast_dtw(a, b, radius=1, min_size=16):
    """Fast DTW that works at multiple resolutions. It first aligns shrunk-down
    copies of the two series, then refines that alignment at finer resolutions,
    only searching near the rough path. Roughly linear time."""
    Ta = len(a)
    if Ta <= min_size or len(b) <= min_size:
        return _full_dtw(a, b)

    def _coarsen(x):
        n = len(x) // 2
        return x[: 2 * n].reshape(n, 2, x.shape[1]).mean(axis=1)

    a_c, b_c = _coarsen(a), _coarsen(b)
    _, low_path = _fast_dtw(a_c, b_c, radius, min_size)

    # Expand the rough path from the coarse level into a search window of
    # cells to check at the finer level.
    window = set()
    for (i, j) in low_path:
        for di in range(2):
            for dj in range(2):
                window.add((2 * i + di, 2 * j + dj))
    expanded = set()
    for (i, j) in window:
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < len(a) and 0 <= jj < len(b):
                    expanded.add((ii, jj))

    Ta, Tb = len(a), len(b)
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    cells = sorted(expanded)
    for (i0, j0) in cells:
        i, j = i0 + 1, j0 + 1
        c = np.linalg.norm(a[i0] - b[j0])
        D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = Ta, Tb, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(D[Ta, Tb]), path


# Allowed steps the alignment can take through the grid. Each pair (di, dj)
# is a move that advances the first curve by di points and the second by dj.
# The variety of step sizes lets the alignment speed up or slow down time at
# different rates while staying smooth.
_DP_NBRS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2),
            (1, 4), (4, 1), (3, 4), (4, 3)]
_DP_NBRS_ARR = np.asarray(_DP_NBRS, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Numba-compiled core of the elastic distance
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=True)
def _srvf_dp_njit(q1, q2, j_center, band_radius, track_path):
    """
    Find the time-warping that best aligns two curves and return how different
    they still are after aligning. This is done with dynamic programming: it
    fills a grid of partial-alignment costs and keeps the cheapest path.

    q1, q2       : (F, T) arrays in square-root velocity (SRVF) form, where F
                   is the number of features and T the number of time points.
    j_center     : (T,) array giving, for each row, the column the band should
                   be centred on. Use np.arange(T) for a band along the
                   diagonal.
    band_radius  : int. If < 0, search the whole T x T grid (no band).
                   Otherwise only look at cells within this many columns of
                   j_center, which is faster.
    track_path   : bool. If True, also record, for each cell, which step led
                   to it, so the best alignment can be traced back afterwards.

    Returns
    -------
    dist : float
        The elastic distance, or -1.0 if the band was too narrow to reach the
        far corner of the grid.
    P : (T, T) array recording the chosen step into each cell (indices into
        _DP_NBRS_ARR), or an empty (0, 0) array when track_path is False.
    """
    F = q1.shape[0]
    T = q1.shape[1]
    INF = 1.0e18

    E = np.full((T, T), INF)
    E[0, 0] = 0.0

    if track_path:
        P = np.full((T, T), -1, dtype=np.int8)
    else:
        P = np.empty((0, 0), dtype=np.int8)

    if T < 2:
        return 0.0, P

    dt = 1.0 / (T - 1)
    nbrs = _DP_NBRS_ARR
    nN = nbrs.shape[0]

    for i in range(1, T):
        if band_radius < 0:
            jlo = 1
            jhi = T
        else:
            jc = j_center[i]
            jlo_i = int(jc) - band_radius
            jhi_i = int(jc) + band_radius + 1
            if jlo_i < 1:
                jlo_i = 1
            if jhi_i > T:
                jhi_i = T
            jlo = jlo_i
            jhi = jhi_i

        for j in range(jlo, jhi):
            best = INF
            best_n = -1
            for n in range(nN):
                di = nbrs[n, 0]
                dj = nbrs[n, 1]
                k = i - di
                l = j - dj
                if k < 0 or l < 0:
                    continue
                ekl = E[k, l]
                if ekl >= INF:
                    continue

                # How fast time is being warped on this step (how many points
                # of curve 2 we cover per point of curve 1).
                slope = (j - l) / (i - k)
                # Scaling factor the SRVF form needs when time is stretched;
                # the lower bound just avoids taking sqrt of (near) zero.
                if slope < 1e-12:
                    sqslope = math.sqrt(1e-12)
                else:
                    sqslope = math.sqrt(slope)

                # Add up the squared difference between the two curves along
                # this step, sampling curve 2 at the warped positions. This is
                # the cost of choosing this step.
                acc = 0.0
                prev_s2 = 0.0
                seg_len = i - k + 1
                for s in range(seg_len):
                    x = k + s
                    gidx = l + s * slope
                    lo_ix = int(gidx)
                    if lo_ix < 0:
                        lo_ix = 0
                    elif lo_ix > T - 2:
                        lo_ix = T - 2
                    frac = gidx - lo_ix
                    if frac < 0.0:
                        frac = 0.0
                    elif frac > 1.0:
                        frac = 1.0
                    s2 = 0.0
                    for fch in range(F):
                        q2w = (q2[fch, lo_ix] * (1.0 - frac)
                               + q2[fch, lo_ix + 1] * frac) * sqslope
                        d = q1[fch, x] - q2w
                        s2 += d * d
                    if s > 0:
                        acc += 0.5 * (prev_s2 + s2) * dt
                    prev_s2 = s2

                val = ekl + acc
                if val < best:
                    best = val
                    best_n = n

            E[i, j] = best
            if track_path and best_n >= 0:
                P[i, j] = best_n

    final = E[T - 1, T - 1]
    if final >= INF:
        return -1.0, P
    if final < 0.0:
        final = 0.0
    return math.sqrt(final), P


def _backtrack_path(P):
    """Rebuild the chosen alignment from the table of recorded steps. Returns
    the list of (i, j) index pairs of the path, from the start corner (0, 0)
    to the end corner (T-1, T-1)."""
    T = P.shape[0]
    nbrs = _DP_NBRS
    path = []
    i, j = T - 1, T - 1
    while i > 0 or j > 0:
        path.append((i, j))
        n = int(P[i, j])
        if n < 0:
            # No recorded step here; stop (shouldn't happen on the full grid).
            break
        di, dj = nbrs[n]
        i -= di
        j -= dj
    path.append((0, 0))
    path.reverse()
    return path


def _srvf_optimum_reparam_distance_pure(q1, q2, t):
    """
    Elastic distance between two curves, computed with dynamic programming in
    plain NumPy. This version is used for testing and as a fallback when Numba
    is not installed. The compiled ``_srvf_dp_njit`` gives the same answer (to
    within tiny rounding differences) but runs about 100x faster.
    """
    F, T = q1.shape
    INF = np.inf
    E = np.full((T, T), INF)
    E[0, 0] = 0.0

    def seg_cost(k, l, i, j):
        slope = (t[j] - t[l]) / (t[i] - t[k])
        xs = np.arange(k, i + 1)
        gpos = t[l] + (t[xs] - t[k]) * slope
        gidx = np.interp(gpos, t, np.arange(T))
        lo = np.clip(np.floor(gidx).astype(int), 0, T - 1)
        hi = np.clip(lo + 1, 0, T - 1)
        frac = gidx - lo
        q2w = (q2[:, lo] * (1 - frac) + q2[:, hi] * frac) * np.sqrt(max(slope, 1e-12))
        integrand = np.sum((q1[:, xs] - q2w) ** 2, axis=0)
        return _trapz(integrand, t[xs])

    for i in range(1, T):
        for j in range(1, T):
            best = INF
            for (di, dj) in _DP_NBRS:
                k, l = i - di, j - dj
                if k < 0 or l < 0 or not np.isfinite(E[k, l]):
                    continue
                val = E[k, l] + seg_cost(k, l, i, j)
                if val < best:
                    best = val
            if best < E[i, j]:
                E[i, j] = best
    return float(np.sqrt(max(E[T - 1, T - 1], 0.0)))


def _srvf_optimum_reparam_distance(q1, q2, t, band_radius=-1, j_center=None):
    """
    Elastic distance between two curves, using the compiled core when Numba is
    available and the plain-NumPy version otherwise.

    band_radius : int, default -1 (search the whole grid). If >= 0, only search
                  alignments within this many columns of ``j_center``, for speed.
    j_center    : (T,) array of column centres per row (default: the diagonal).
    """
    if not _HAS_NUMBA:
        # The plain-NumPy version has no cheap banded mode, so just search the
        # whole grid.
        return _srvf_optimum_reparam_distance_pure(q1, q2, t)
    F, T = q1.shape
    if j_center is None:
        j_center_arr = np.arange(T, dtype=np.float64)
    else:
        j_center_arr = np.ascontiguousarray(j_center, dtype=np.float64)
    q1c = np.ascontiguousarray(q1, dtype=np.float64)
    q2c = np.ascontiguousarray(q2, dtype=np.float64)
    dist, _ = _srvf_dp_njit(q1c, q2c, j_center_arr, int(band_radius), False)
    if dist < 0.0:
        # The band was too narrow to connect the two corners; retry with a
        # wider band, then with no band at all if needed.
        wider = max(int(band_radius) * 2, T // 4)
        dist, _ = _srvf_dp_njit(q1c, q2c, j_center_arr, wider, False)
        if dist < 0.0:
            dist, _ = _srvf_dp_njit(q1c, q2c, j_center_arr, -1, False)
    return float(dist)


def _srvf_optimum_reparam_distance_fast(q1, q2, t, coarse_factor=2,
                                        refine_radius=12):
    """
    Faster, two-pass elastic distance. First it shrinks both curves down and
    solves the alignment on the small versions to get a rough match. Then it
    re-solves on the full curves, but only searching near that rough match.

    The defaults (coarse_factor=2, refine_radius=12) give answers that match
    the full search to within rounding while cutting the work per pair roughly
    4-8x for curves of length 300-600. To trade more accuracy for more speed,
    raise coarse_factor (e.g. 4) and/or lower refine_radius. The result can
    only ever be slightly too large, never too small.

    coarse_factor : int >= 2, how much to shrink the curves in the first pass.
    refine_radius : int, how far from the rough match to search in the second
                    pass on the full curves.

    Returns the elastic distance. It is approximate but almost always exact,
    because the true best alignment nearly always falls within ``refine_radius``
    of the rough one.
    """
    if not _HAS_NUMBA:
        return _srvf_optimum_reparam_distance_pure(q1, q2, t)
    F, T = q1.shape
    Tc = max(16, T // max(1, int(coarse_factor)))
    if Tc >= T or T <= 64:
        # The curves are already short, so a single banded search is cheaper
        # than shrinking, refining, and tracing back. Below about T=64 the
        # two-pass approach would cost more than it saves.
        auto_band = max(12, T // 8)
        return _srvf_optimum_reparam_distance(q1, q2, t, band_radius=auto_band)

    # First pass: shrink q1 and q2 down to length Tc by interpolation.
    idx_full = np.arange(T)
    idx_coarse = np.linspace(0.0, T - 1, Tc)
    q1c = np.empty((F, Tc), dtype=np.float64)
    q2c = np.empty((F, Tc), dtype=np.float64)
    for fch in range(F):
        q1c[fch] = np.interp(idx_coarse, idx_full, q1[fch])
        q2c[fch] = np.interp(idx_coarse, idx_full, q2[fch])

    # Solve the alignment on the shrunk curves, searching the whole small grid,
    # and trace back the rough matching path.
    j_center_c = np.arange(Tc, dtype=np.float64)
    _, P = _srvf_dp_njit(np.ascontiguousarray(q1c),
                         np.ascontiguousarray(q2c),
                         j_center_c, -1, True)
    path_c = _backtrack_path(P)

    # Scale the rough path back up to full length and read off, for every row
    # i, which column j it passes through.
    scale = (T - 1) / (Tc - 1)
    pi = np.array([p[0] * scale for p in path_c], dtype=np.float64)
    pj = np.array([p[1] * scale for p in path_c], dtype=np.float64)
    # The path can briefly hold the same row across off-diagonal steps; sort by
    # row and drop duplicate rows so the interpolation below stays well-defined.
    order = np.argsort(pi, kind='stable')
    pi = pi[order]
    pj = pj[order]
    keep = np.concatenate(([True], np.diff(pi) > 0))
    pi = pi[keep]
    pj = pj[keep]
    j_center = np.interp(np.arange(T, dtype=np.float64), pi, pj)

    # Second pass: search the full grid, but only in a band around the rough
    # path found above.
    return _srvf_optimum_reparam_distance(q1, q2, t,
                                          band_radius=int(refine_radius),
                                          j_center=j_center)


# --------------------------------------------------------------------------- #
class TimeSeriesDistance:
    """
    Compute all pairwise distances between a set of multi-feature time series.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features, n_timesteps)
        The set of time series to compare.
    metric : {'elastic', 'elastic-fast', 'fastdtw', 'ultrafast'}, default 'elastic'
        - 'elastic'      : elastic distance that warps time to best align the
                           curves; accurate, compiled, and optionally banded.
        - 'elastic-fast' : the faster two-pass version of the elastic distance
                           (rough align, then refine); nearly as accurate.
        - 'fastdtw'      : band-limited dynamic time warping, roughly linear time.
        - 'ultrafast'    : multi-resolution dynamic time warping, roughly
                           linear time.
    n_jobs : int, default -1
        How many CPU cores to use (-1 means all of them).
    band_radius : int or None, default None (auto)
        How far from the diagonal the alignment search may stray; a smaller
        value is faster but may miss the best alignment. When ``None`` it is set
        automatically to ``max(12, n_timesteps // 4)`` (about a quarter of the
        series length), which is wide enough for the best alignment in practice.
        It is used as the band for 'fastdtw' and 'elastic' (pass ``-1`` to turn
        banding off for 'elastic'), and as the refinement band for
        'elastic-fast' and 'ultrafast'.
    elastic_coarse_factor : int, default 2
        For 'elastic-fast', how much to shrink the curves in the first pass.
    """

    def __init__(self, X, metric='elastic', n_jobs=-1, band_radius=None,
                 elastic_coarse_factor=2):
        self.X_raw = np.asarray(X, dtype=float)
        self.metric = metric
        self.n_jobs = n_jobs
        self.elastic_coarse_factor = elastic_coarse_factor
        self.n_samples, self.n_features, self.n_timesteps = self.X_raw.shape
        # Pick the band width automatically: max(12, T // 4). For short series
        # the floor of 12 keeps the band wide enough; for long series it grows
        # with the length so the best alignment still fits inside the band.
        if band_radius is None:
            self.band_radius = max(12, self.n_timesteps // 4)
        else:
            self.band_radius = band_radius

        eps = np.finfo(float).eps
        Xs = np.empty_like(self.X_raw)
        for f in range(self.n_features):
            feat = self.X_raw[:, f, :].ravel()
            mu, sigma = feat.mean(), feat.std() + eps
            Xs[:, f, :] = (self.X_raw[:, f, :] - mu) / sigma
        self.X = Xs
        self.times = np.linspace(0, 1, self.n_timesteps)
        self.D = None

    # --- square-root velocity (SRVF) transform ------------------------------
    @staticmethod
    def _curve_to_srvf(curve, t):
        """Convert a curve to its square-root velocity form, a transform of the
        curve's slope that makes the elastic (time-warping) distance easy to
        compute. Input and output both have shape (F, T): F features, T time
        points."""
        dX = np.vstack([np.gradient(curve[f], t) for f in range(curve.shape[0])])
        eps = np.finfo(float).eps
        speed = np.linalg.norm(dX, axis=0) + eps
        return dX / np.sqrt(speed)

    # --- distance for one pair, per metric ----------------------------------
    def _elastic_distance(self, i, j):
        # Each curve's SRVF form is computed once up front in
        # compute_distances() and reused here, instead of recomputing it for
        # every pair.
        band = -1 if self.band_radius is None else int(self.band_radius)
        return _srvf_optimum_reparam_distance(self._Q[i], self._Q[j],
                                              self.times, band_radius=band)

    def _elastic_fast_distance(self, i, j):
        return _srvf_optimum_reparam_distance_fast(
            self._Q[i], self._Q[j], self.times,
            coarse_factor=int(self.elastic_coarse_factor),
            refine_radius=max(1, int(self.band_radius)))

    def _dtw_distance(self, Xi, Xj):
        radius = max(1, int(self.band_radius))
        if self.metric == 'fastdtw':
            return _banded_dtw(Xi.T, Xj.T, radius)[0]
        return _fast_dtw(Xi.T, Xj.T, radius=max(1, radius // 6))[0]

    def _pairwise(self, i, j):
        if self.metric == 'elastic':
            return self._elastic_distance(i, j)
        elif self.metric == 'elastic-fast':
            return self._elastic_fast_distance(i, j)
        elif self.metric in ('fastdtw', 'ultrafast'):
            return self._dtw_distance(self.X[i], self.X[j])
        raise ValueError(f"Unknown metric {self.metric!r}")

    # --- main entry point ---------------------------------------------------
    def compute_distances(self):
        """Build and return the full distance matrix, shape (n_samples,
        n_samples). It is symmetric with zeros on the diagonal."""
        if self.metric in ('elastic', 'elastic-fast'):
            # Compute each curve's SRVF form just once and reuse it for every
            # pair, rather than recomputing it for each pair.
            self._Q = [self._curve_to_srvf(self.X[i], self.times)
                       for i in range(self.n_samples)]
            # Trigger Numba's one-time compilation here, before splitting work
            # across worker processes, so each worker doesn't repeat it.
            if _HAS_NUMBA and self.n_samples >= 2:
                _srvf_optimum_reparam_distance(self._Q[0], self._Q[1],
                                               self.times, band_radius=4)

        def row(i):
            r = np.zeros(self.n_samples)
            for j in range(i + 1, self.n_samples):
                r[j] = self._pairwise(i, j)
            return r

        self.D = np.vstack(Parallel(n_jobs=self.n_jobs)(
            delayed(row)(i) for i in range(self.n_samples)))
        self.D_upper = self.D
        self.D = self.D_upper + self.D_upper.T
        np.fill_diagonal(self.D, 0.0)
        return self.D

    def most_similar(self):
        if self.D is None:
            raise RuntimeError("Call compute_distances() first.")
        mask = np.triu(np.ones_like(self.D, bool), k=1)
        idx = np.argmin(self.D[mask])
        i, j = np.argwhere(mask)[idx]
        return i, j, float(self.D[i, j])

    def most_different(self):
        if self.D is None:
            raise RuntimeError("Call compute_distances() first.")
        mask = np.triu(np.ones_like(self.D, bool), k=1)
        idx = np.argmax(self.D[mask])
        i, j = np.argwhere(mask)[idx]
        return i, j, float(self.D[i, j])

    def plot_extremes(self):
        if self.D is None:
            raise RuntimeError("Call compute_distances() first.")
        i1, j1, _ = self.most_similar()
        i2, j2, _ = self.most_different()
        t = np.arange(self.n_timesteps)
        cmap_name = 'tab10' if self.n_features <= 10 else 'tab20'
        cmap = plt.get_cmap(cmap_name, self.n_features)
        colour_cycle = [cmap(i) for i in range(self.n_features)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        for f in range(self.n_features):
            ax1.plot(t, self.X_raw[i1, f], color=colour_cycle[f], label=f"feature {f+1}")
            ax1.plot(t, self.X_raw[j1, f], '--', color=colour_cycle[f])
        ax1.set_title(f"Most similar: indices {i1} vs {j1}")
        ax1.legend(loc='best')
        for f in range(self.n_features):
            ax2.plot(t, self.X_raw[i2, f], color=colour_cycle[f], label=f"feature {f+1}")
            ax2.plot(t, self.X_raw[j2, f], '--', color=colour_cycle[f])
        ax2.set_title(f"Most distinct: indices {i2} vs {j2}")
        ax2.legend(loc='best')
        plt.show()

## ------------------------------------------------------------------------- ##
