import numpy as np
import pywt
from scipy.interpolate import BSpline, LSQUnivariateSpline


## ------------------------------------------------------------------------- ##
class Smoothing:
    """
    Fit smooth curves to noisy, sampled data. Each row of the data is a series
    of measurements, and this turns it into a smooth function you can evaluate
    at any time. The smooth curve is built from simple building-block functions
    (the "basis"): smooth piecewise polynomials (B-splines), sines and cosines
    (Fourier), or wavelets.

    Parameters
    ----------
    dis_p : int
        How many points to evaluate the fitted curve at when sampling it
        finely (default=300).
    fit : {'bspline', 'fourier', 'wavelet'}
        Which kind of building-block functions to use.
    n : int
        Number of Fourier frequencies (giving 2n + 1 building blocks in total).
    smoothing_str : float
        How strongly to smooth a B-spline fit. When ``terms`` is None, it is
        used as the starting scale while the smoothing strength is chosen
        automatically.
    terms : int or None
        Number of building blocks (knots). If None, a good number is chosen
        automatically.
    wavelet_level : int
        How many levels of detail to use for a wavelet fit. If None, chosen
        automatically.
    data : array-like, shape (N_samples, N_timepoints)
        The data to fit: one time series per row.
    """
    def __init__(self, 
                 dis_p=300,
                 fit='bspline',
                 n=3,                     # number of Fourier frequencies (2n+1 building blocks)
                 smoothing_str=0.3,       # starting scale for the smoothing strength when chosen automatically
                 terms=None,
                 wavelet_level=4,         # level of detail for a wavelet fit
                 data=None,
                 ):
        ## -----------------------------------------------------------------
        # Store the settings.
        self.smoothing_str = smoothing_str  # also used as the starting scale when the smoothing strength is chosen automatically
        self.n = n              # number of Fourier frequencies (if None, chosen automatically)
        self.dis_p = dis_p      # how many points to sample the fitted curve at
        self.fit = fit
        self.wavelet_level = wavelet_level
        self.num_knots = terms
        self.data = data

        # Set up the time axes the data lives on.
        self.data_size = self.data.shape[1]
        self.t = np.linspace(0, 1, self.data_size)      # the times the data was measured at
        self.fine_t = np.linspace(0, 1, self.dis_p)     # a denser set of times to evaluate the fitted curve at

        ## -----------------------------------------------------------------
        # --- Build the fit, depending on which basis was chosen ---
        if self.fit == 'bspline':
            if self.num_knots is None:
                # Choose the smoothing strength and number of building blocks
                # automatically, then fit a smoothed B-spline.
                self.smoothing_str, self.num_knots = self._gcv_bspline_penalty_and_terms()
                self.fn_s = self._fit_bsplines_penalized()
            else:
                # A fixed number of building blocks was given: fit a B-spline
                # with that many evenly spaced knots by least squares.
                self.degree = 3
                knot_pos = np.linspace(0, 1, self.num_knots)
                knots = np.concatenate(([0] * self.degree, knot_pos[1:-1], [1] * self.degree))
                curves = []
                self.coeffs = []
                t0 = np.linspace(0, 1, data.shape[1])
                for row in data:
                    spline = LSQUnivariateSpline(t0, row, t=knots[self.degree + 1 : -self.degree - 1], k=self.degree)
                    curves.append(spline)
                    self.coeffs.append(np.array(spline.get_coeffs()))
                self.fn_s = curves
                # Also build the list of basis functions, for callers that need them.
                def bspline_basis(num_basis=None, degree=3):
                    num_knots = num_basis + degree + 1
                    t_t = np.concatenate(([0] * degree, np.linspace(0, 1, num_knots - 2 * degree), [1] * degree))
                    basis_input = []
                    for i in range(num_basis):
                        basis_coefs = np.zeros(num_knots - degree - 1)
                        basis_coefs[i] = 1
                        basis_input.append(BSpline(t_t, basis_coefs, degree))
                    return basis_input
                self.smoothing_basis = bspline_basis(num_basis=self.num_knots)

        elif self.fit == 'fourier':
            # If the number of Fourier frequencies wasn't given, pick a good
            # number automatically first.
            if self.n is None:
                lam, eff_n = self._gcv_fourier_penalty_and_terms()
                self.smoothing_str = lam
                self.n = max(1, int(eff_n))
            # Fit using that number of frequencies.
            self.fn_s = self._fit_curves(basis='fourier')

        else:
            # Wavelet fit. Here ``fit`` holds a wavelet name such as 'db4'.
            # If no level of detail was given, choose the level and the
            # noise-removal threshold automatically.
            if self.wavelet_level is None:
                self.wavelet_level, self.wavelet_threshold = self._gcv_wavelet_level_and_threshold()
            # Fit using the chosen (or given) level and threshold.
            self.fn_s = self._fit_curves(basis=self.fit)
    
    ## ---------------------------------------------------------------------
    # --- Helpers ---

    @staticmethod
    def _gcv_error(y_true, y_pred, df):
        """
        Score how well a fit matches the data while penalizing fits that use
        too many building blocks, so we can compare settings and pick the best.
        Lower is better. ``df`` measures roughly how many free parameters the
        fit used. (This is the generalized cross-validation, or GCV, score:
        n * sum-of-squared-errors / (n - df)^2.)
        """
        residual = y_true - y_pred
        n = len(y_true)
        rss = np.sum(residual**2)
        # Keep the denominator away from zero when df gets close to n.
        denom = max(1e-12, (n - df)) ** 2
        return n * rss / denom

    ## ---------------------------------------------------------------------
    # --- B-spline (smooth piecewise-polynomial fit) ---
    def _bspline_design(self, x, num_basis, degree=3):
        """Build the matrix of B-spline building-block functions evaluated at
        the points ``x``, using evenly spaced knots on [0, 1]. Returns that
        matrix and the list of building-block functions."""
        num_knots = num_basis + degree + 1
        t_t = np.concatenate(([0] * degree, np.linspace(0, 1, num_knots - 2 * degree), [1] * degree))
        basis = []
        for i in range(num_basis):
            coefs = np.zeros(num_knots - degree - 1)
            coefs[i] = 1.0
            basis.append(BSpline(t_t, coefs, degree))
        X = np.column_stack([b(x) for b in basis])
        return X, basis

    def _penalized_spline_fit(self, y, X, lam, order=2):
        """Fit the data ``y`` with the building blocks in ``X``, adding a
        penalty that discourages wiggly fits. ``lam`` controls how strong the
        penalty is (larger means smoother). The penalty measures how much
        neighbouring coefficients differ. Returns the fitted values and a
        rough count of how many free parameters were used."""
        p = X.shape[1]
        # Build the operator D that takes differences of the coefficients.
        D = np.zeros((max(0, p - order), p))
        for i in range(D.shape[0]):
            # Coefficients for a difference of the given order.
            coeff = np.zeros(order + 1)
            # Alternating-sign binomial coefficients.
            from math import comb
            for k in range(order + 1):
                coeff[k] = ((-1)**(order - k)) * comb(order, k)
            D[i, i:i+order+1] = coeff
        XtX = X.T @ X
        if D.size == 0:
            P = np.zeros_like(XtX)
        else:
            P = D.T @ D
        A = XtX + lam * P
        S = X @ np.linalg.solve(A, X.T)
        yhat = S @ y
        df = np.trace(S)
        return yhat, df

    def _gcv_bspline_penalty_and_terms(self):
        """Try several numbers of building blocks and several smoothing
        strengths, score each with the GCV measure, and return the smoothing
        strength and number of building blocks that scored best."""
        # Candidate numbers of building blocks (kept modest to avoid overfitting).
        nT = self.data_size
        candidates = np.unique(np.clip(np.array([6, 8, 10, 12, 15, 20, 25, 30]), 4, nT-1))
        # Range of smoothing strengths to try, spread across several orders of
        # magnitude around the given starting scale.
        base = max(1e-4, float(self.smoothing_str))
        lam_grid = np.unique(np.concatenate([
            10.0**np.linspace(-6, 2, 20),
            base * 10.0**np.linspace(-3, 3, 13),
        ]))

        best = (np.inf, None, None)  # best score so far, and its (smoothing strength, K)
        for K in candidates:
            X, basis = self._bspline_design(self.t, K, degree=3)
            # Compute these once per K to save time inside the inner loop.
            XtX = X.T @ X
            # Penalty matrix based on second differences of the coefficients.
            p = X.shape[1]
            D = np.zeros((max(0, p - 2), p))
            for i in range(D.shape[0]):
                coeff = np.array([1, -2, 1], dtype=float)
                D[i, i:i+3] = coeff
            P = D.T @ D if D.size else np.zeros_like(XtX)

            for lam in lam_grid:
                A = XtX + lam * P
                AinvXt = np.linalg.solve(A, X.T)
                S = X @ AinvXt
                df = np.trace(S)
                gcv_sum = 0.0
                for row in self.data:
                    yhat = S @ row
                    gcv_sum += self._gcv_error(row, yhat, df)
                gcv_avg = gcv_sum / len(self.data)
                if gcv_avg < best[0]:
                    best = (gcv_avg, lam, K)

        best_gcv, best_lam, best_K = best
        # Keep the chosen building blocks so they can be reused later.
        self._bspline_best_basis = self._bspline_design(self.t, int(best_K), degree=3)[1]
        return float(best_lam), int(best_K)

    def _fit_bsplines_penalized(self):
        """Fit a smoothed B-spline to each row using the chosen smoothing
        strength and number of building blocks. Returns one callable curve per
        row."""
        K = self.num_knots
        X, basis = self._bspline_design(self.t, K, degree=3)
        # Penalty based on second differences of the coefficients.
        p = X.shape[1]
        D = np.zeros((max(0, p - 2), p))
        for i in range(D.shape[0]):
            D[i, i:i+3] = np.array([1, -2, 1], dtype=float)
        P = D.T @ D if D.size else np.zeros((p, p))
        XtX = X.T @ X
        A = XtX + self.smoothing_str * P
        AinvXt = np.linalg.solve(A, X.T)    
        curves = []
        self.coeffs = []
        for row in self.data:
            beta = AinvXt @ row
            self.coeffs.append(beta.copy())
            yfine = np.column_stack([b(self.fine_t) for b in basis]) @ beta
            # Wrap the densely-sampled fitted curve in a spline object so it is
            # easy to evaluate later.
            spline = LSQUnivariateSpline(self.fine_t, yfine, t=np.linspace(0,1,max(2, K-4))[1:-1], k=3) if K>4 else \
                     LSQUnivariateSpline(self.fine_t, yfine, t=[], k=3)
            curves.append(spline)
        self.smoothing_basis = basis
        return curves
    
    ## ---------------------------------------------------------------------
    # --- Fourier (sums of sines and cosines) ---
    def _fourier_design(self, x, max_n):
        """Build the matrix of Fourier building blocks evaluated at the points
        ``x``: a constant column plus cos(2*pi*j*x) and sin(2*pi*j*x) for each
        frequency j from 1 to max_n."""
        cols = [np.ones_like(x)]
        for j in range(1, max_n + 1):
            cols.append(np.cos(2*np.pi*j*x))
            cols.append(np.sin(2*np.pi*j*x))
        X = np.column_stack(cols)
        return X

    def _gcv_fourier_penalty_and_terms(self, m=2):
        """
        Pick a good smoothing strength for a Fourier fit by trying several and
        scoring each with the GCV measure. Higher frequencies are penalized
        more, which favours smoother curves. Returns the chosen smoothing
        strength and a sensible number of frequencies to keep.
        """
        max_n = min(50, (self.data_size - 1)//2)  # cap the number of frequencies so the matrices stay small
        X = self._fourier_design(self.t, max_n)
        p = X.shape[1]

        # Penalty weights: none for the constant term, and a weight that grows
        # with frequency for each cosine/sine pair, so higher frequencies are
        # smoothed away more strongly.
        P = np.zeros((p, p))
        idx = 1
        for j in range(1, max_n + 1):
            w = (2*np.pi*j)**(2*m)
            P[idx, idx] = w       # cosine term
            P[idx+1, idx+1] = w   # sine term
            idx += 2

        XtX = X.T @ X
        base = max(1e-6, float(self.smoothing_str))
        lam_grid = np.unique(np.concatenate([10.0**np.linspace(-8, 4, 25), base * 10.0**np.linspace(-4, 4, 17)]))

        best = (np.inf, None, None)  # best score so far, and its (smoothing strength, df)
        for lam in lam_grid:
            A = XtX + lam * P
            AinvXt = np.linalg.solve(A, X.T)
            S = X @ AinvXt
            df = np.trace(S)
            gcv_sum = 0.0
            for row in self.data:
                yhat = S @ row
                gcv_sum += self._gcv_error(row, yhat, df)
            gcv_avg = gcv_sum / len(self.data)
            if gcv_avg < best[0]:
                best = (gcv_avg, lam, df)

        _, best_lam, best_df = best
        # Turn the effective number of parameters into a frequency count: one
        # for the constant term, then two (a cosine and a sine) per frequency.
        eff_n = max(1, int(round(max(0.0, best_df - 1) / 2.0)))
        return float(best_lam), eff_n

    def _fit_fourier(self, x, a0, an, bn):
        """Evaluate a Fourier curve at points ``x`` from its coefficients
        (the constant ``a0`` and the cosine/sine coefficients ``an``/``bn``)."""
        y = a0 * np.ones_like(x)
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            y += a * np.cos(2 * np.pi * n * x) + b * np.sin(2 * np.pi * n * x)
        return y

    def _fourier_coefficients(self, data, T, n):
        """Find the Fourier coefficients of one time series, keeping only the
        first ``n`` frequencies."""
        N = len(data)
        fft_output = np.fft.fft(data)
        a0 = fft_output[0].real / N
        # Convert the FFT's real and imaginary parts into cosine/sine coefficients.
        an = 2 * fft_output.real[1:N//2] / N
        bn = -2 * fft_output.imag[1:N//2] / N
        return a0, an[:n], bn[:n]

    def fourier_basis(self):
        """
        Build the list of Fourier building-block functions you can call:
        [constant, cos1, sin1, ..., cos n, sin n].
        """
        bases = [lambda x: np.ones_like(x)]
        for k in range(1, self.n + 1):
            bases.append(lambda x, k=k: np.cos(2*np.pi*k*x))
            bases.append(lambda x, k=k: np.sin(2*np.pi*k*x))
        return bases
    
    ## ---------------------------------------------------------------------
    # --- Wavelet (smoothing by zeroing out small detail coefficients) ---
    def _count_nonzero_coeffs(self, coeffs):
        count = 0
        for c in coeffs:
            if isinstance(c, tuple) or isinstance(c, list):
                # Detail coefficients can come grouped; turn them into one array.
                c = np.asarray(c)
            count += np.count_nonzero(c)
        return count

    def _threshold_coeffs(self, coeffs, tau, mode='soft'):
        # Shrink wavelet coefficients toward zero, dropping ones smaller than
        # the threshold ``tau``. This removes noise (small wiggles) and keeps
        # the larger, meaningful features.
        thr = []
        for c in coeffs:
            arr = np.asarray(c)
            thr.append(pywt.threshold(arr, tau, mode=mode))
        return thr

    def _recompose_like_input(self, coeffs, wavelet):
        # Rebuild a curve from its wavelet coefficients.
        rec = pywt.waverec(coeffs, wavelet)
        return rec

    def _gcv_wavelet_level_and_threshold(self, wavelet_name=None):
        """
        Try several levels of detail and several noise thresholds, score each
        with the GCV measure, and return the best level and threshold. Here the
        number of free parameters is the count of coefficients left after
        thresholding.
        """
        wavelet = wavelet_name or self.fit  # e.g. a name like 'db4'
        max_lev = min(self.wavelet_level if self.wavelet_level else 6, pywt.dwt_max_level(self.data_size, pywt.Wavelet(wavelet).dec_len))
        levels = range(1, max_lev + 1)
        # Estimate the noise level from the finest detail of the first series,
        # using a robust spread measure so a few outliers don't throw it off.
        sigma_ref = None
        best = (np.inf, None, None)
        for lev in levels:
            gbest = (np.inf, None)
            for i, row in enumerate(self.data):
                coeffs = pywt.wavedec(row, wavelet, level=lev)
                if sigma_ref is None and len(coeffs) > 1:
                    d1 = coeffs[-1]
                    sigma_ref = np.median(np.abs(d1 - np.median(d1))) / 0.6745 + 1e-12
                # Build a range of thresholds to try for this level.
            tau_grid = np.linspace(0.5, 3.5, 8) * (sigma_ref if sigma_ref is not None else 1.0)
            # Score each threshold, averaging over all the series.
            for tau in tau_grid:
                gcv_sum = 0.0
                df_sum = 0.0
                for row in self.data:
                    coeffs = pywt.wavedec(row, wavelet, level=lev)
                    thr = self._threshold_coeffs(coeffs, tau, mode='soft')
                    df = self._count_nonzero_coeffs(thr)
                    yhat = pywt.waverec(thr, wavelet)
                    # Rebuilt curve can be one point longer; trim to match.
                    yhat = yhat[:len(row)]
                    gcv_sum += self._gcv_error(row, yhat, df)
                    df_sum += df
                gcv_avg = gcv_sum / len(self.data)
                if gcv_avg < gbest[0]:
                    gbest = (gcv_avg, tau)
            # Remember the best threshold found at this level.
            if gbest[0] < best[0]:
                best = (gbest[0], lev, gbest[1])
        best_gcv, best_level, best_tau = best
        return int(best_level), float(best_tau)
    
    ## ---------------------------------------------------------------------
    # --- choose and run the right fitting routine ---
    def _fit_curves(self, basis=None):
        """
        Fit a curve to each row of the data using the chosen building blocks
        ('bspline', 'fourier', or a wavelet name).

        Returns
        -------
        fn_s : list
            One fitted curve (a callable function or spline object) per data row.
        """
        if basis == 'bspline':
            # Already fitted in __init__.
            return self.fn_s
        elif basis == 'fourier':
            fn_s = []; T = 1; coeff = []
            for i in range(len(self.data)):
                a0, an, bn = self._fourier_coefficients(self.data[i], T, self.n)
                # Store the coefficients in one flat array, ordered as
                # [constant, cos1, sin1, cos2, sin2, ...], length 2n+1.
                packed = np.empty(1 + 2*self.n, dtype=float)
                packed[0] = a0
                for k in range(self.n):
                    packed[1 + 2*k]     = an[k]
                    packed[1 + 2*k + 1] = bn[k]
                coeff.append(packed)
                # Make a function that evaluates this curve from its coefficients.
                def make_fn(p):
                    def f(x):
                        y = p[0] * np.ones_like(x)
                        for k in range(self.n):
                            a = p[1 + 2*k]
                            b = p[1 + 2*k + 1]
                            y += a * np.cos(2*np.pi*(k+1)*x) + b * np.sin(2*np.pi*(k+1)*x)
                        return y
                    return f
                fn_s.append(make_fn(packed))
            self.coeffs = np.vstack(coeff)
            # The Fourier building blocks, in the same order as the coefficients.
            self.smoothing_basis = self.fourier_basis()
            return fn_s
        else:
            # Wavelet fit: break each series into wavelet coefficients, drop the
            # small (noise) ones, and rebuild a smooth curve.
            wavelet = basis
            fn_s = []
            self.coeffs = []
            for row in self.data:
                coeffs = pywt.wavedec(row, wavelet, level=self.wavelet_level)
                if hasattr(self, "wavelet_threshold"):
                    thr = self._threshold_coeffs(coeffs, self.wavelet_threshold, mode='soft')
                else:
                    # No threshold was chosen, so estimate the noise level and
                    # use a mild threshold based on it.
                    d1 = coeffs[-1]
                    sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745 + 1e-12
                    thr = self._threshold_coeffs(coeffs, 2.0 * sigma, mode='soft')
                self.coeffs.append(thr)
                reconstructed = pywt.waverec(thr, wavelet)[:len(row)]
                # Wrap the rebuilt curve in a spline so it is easy to evaluate.
                t0 = np.linspace(0,1,len(reconstructed))
                spline = LSQUnivariateSpline(t0, reconstructed, t=np.linspace(0,1,max(2, len(reconstructed)//8))[1:-1], k=3) \
                         if len(reconstructed) > 12 else LSQUnivariateSpline(t0, reconstructed, t=[], k=3)
                fn_s.append(spline)
            return fn_s

## ------------------------------------------------------------------------- ##

