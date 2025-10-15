import numpy as np
import pywt
from scipy.interpolate import BSpline, LSQUnivariateSpline


## ------------------------------------------------------------------------- ##
class Smoothing:
    """
    Fit smooth functional representations to discrete data using different bases.

    Parameters
    ----------
    dis_p : int
        Number of evaluation points on the fine grid (default=300).
    fit : {'bspline', 'fourier', 'wavelet'}
        Basis type for smoothing.
    n : int
        Number of Fourier terms (2n + 1 total basis functions).
    smoothing_str : float
        Smoothing parameter for B-spline fitting (used as penalty lambda in GCV search when terms is None).
    terms : int or None
        Number of basis terms / knots (if None, optimized automatically via GCV).
    wavelet_level : int
        Decomposition level for wavelet basis (if None, optimized automatically via GCV).
    data : array-like, shape (N_samples, N_timepoints)
        Input data matrix to fit.
    """
    def __init__(self, 
                 dis_p=300,
                 fit='bspline',
                 n=3,                     # number of fourier terms (2n+1)
                 smoothing_str=0.3,       # used as initial scale for lambda grid when optimizing
                 terms=None,
                 wavelet_level=4,         # resolution of wavelet fit
                 data=None,
                 ):
        ## -----------------------------------------------------------------
        # Store input settings
        self.smoothing_str = smoothing_str  # treat this as a scale for lambda when GCV-optimizing
        self.n = n              # Number of sine and cosine terms (if None -> pick by GCV via penalty)
        self.dis_p = dis_p      # number of discrete points for evaluations
        self.fit = fit
        self.wavelet_level = wavelet_level
        self.num_knots = terms
        self.data = data
        
        # Data dimensions and grids
        self.data_size = self.data.shape[1]
        self.t = np.linspace(0, 1, self.data_size)      # coarse grid
        self.fine_t = np.linspace(0, 1, self.dis_p)     # fine grid for evaluation

        ## -----------------------------------------------------------------
        # --- Model selection by fit type ---
        if self.fit == 'bspline':
            if self.num_knots is None:
                # choose lambda by GCV and infer number of basis functions (terms)
                self.smoothing_str, self.num_knots = self._gcv_bspline_penalty_and_terms()
                # fit penalized bspline
                self.fn_s = self._fit_bsplines_penalized()
            else:
                # explicit number of terms => LSQ spline with those knots
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
                # also expose basis builder for parity with earlier API
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
            # If number of Fourier terms (n) was not provided, use GCV to pick a penalty and infer n.
            if self.n is None:
                lam, eff_n = self._gcv_fourier_penalty_and_terms()
                self.smoothing_str = lam
                self.n = max(1, int(eff_n))
            # In both cases, fit using the provided/inferred n WITHOUT GCV here.
            self.fn_s = self._fit_curves(basis='fourier')

        else:
            # wavelet fit (fit holds a wavelet name, e.g. 'db4')
            # If explicit wavelet_level is None -> use GCV to choose (level, threshold).
            if self.wavelet_level is None:
                self.wavelet_level, self.wavelet_threshold = self._gcv_wavelet_level_and_threshold()
            # Fit once with provided or chosen level (and threshold if set).
            self.fn_s = self._fit_curves(basis=self.fit)
    
    ## ---------------------------------------------------------------------
    # --- Utilities ---

    @staticmethod
    def _gcv_error(y_true, y_pred, df):
        """
        Compute Generalized Cross-Validation (GCV) error.
        Standard: GCV = n * RSS / (n - df)^2.
        """
        residual = y_true - y_pred
        n = len(y_true)
        rss = np.sum(residual**2)
        # guard for df close to n
        denom = max(1e-12, (n - df)) ** 2
        return n * rss / denom

    ## ---------------------------------------------------------------------
    # --- B-Spline (penalized regression spline) ---
    def _bspline_design(self, x, num_basis, degree=3):
        """Construct B-spline design matrix with equally-spaced internal knots on [0,1]."""
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
        """Solve min ||y - Xb||^2 + lam * ||D b||^2  with D = finite-diff of given order."""
        p = X.shape[1]
        # Build difference operator D (p - order rows)
        D = np.zeros((max(0, p - order), p))
        for i in range(D.shape[0]):
            # finite differences of given order
            coeff = np.zeros(order + 1)
            # binomial coefficients with alternating signs
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
        """Select lambda (penalty) by GCV for several basis sizes; pick best basis size (terms)."""
        # candidate number of basis functions (keep small to moderate to avoid overfitting)
        nT = self.data_size
        candidates = np.unique(np.clip(np.array([6, 8, 10, 12, 15, 20, 25, 30]), 4, nT-1))
        # lambda grid (log space around provided smoothing_str as scale)
        base = max(1e-4, float(self.smoothing_str))
        lam_grid = np.unique(np.concatenate([
            10.0**np.linspace(-6, 2, 20),
            base * 10.0**np.linspace(-3, 3, 13),
        ]))

        best = (np.inf, None, None)  # gcv, lambda, K
        for K in candidates:
            X, basis = self._bspline_design(self.t, K, degree=3)
            # precompute for speed
            XtX = X.T @ X
            # difference penalty matrix for order=2
            p = X.shape[1]
            D = np.zeros((max(0, p - 2), p))
            for i in range(D.shape[0]):
                coeff = np.array([1, -2, 1], dtype=float)
                D[i, i:i+3] = coeff
            P = D.T @ D if D.size else np.zeros_like(XtX)

            for lam in lam_grid:
                A = XtX + lam * P
                # Precompute factorization
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
        # store chosen basis for later reuse
        self._bspline_best_basis = self._bspline_design(self.t, int(best_K), degree=3)[1]
        return float(best_lam), int(best_K)

    def _fit_bsplines_penalized(self):
        """Fit penalized regression spline with chosen lambda and basis size; return callable splines per row."""
        K = self.num_knots
        X, basis = self._bspline_design(self.t, K, degree=3)
        # penalty (second differences)
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
            # represent as a BSpline by fitting LSQ spline to the fine grid for convenience
            spline = LSQUnivariateSpline(self.fine_t, yfine, t=np.linspace(0,1,max(2, K-4))[1:-1], k=3) if K>4 else \
                     LSQUnivariateSpline(self.fine_t, yfine, t=[], k=3)
            curves.append(spline)
        self.smoothing_basis = basis
        return curves
    
    ## ---------------------------------------------------------------------
    # --- Fourier (frequency-penalized ridge) ---
    def _fourier_design(self, x, max_n):
        """Design matrix with columns: 1, cos(2π j x), sin(2π j x) for j=1..max_n."""
        cols = [np.ones_like(x)]
        for j in range(1, max_n + 1):
            cols.append(np.cos(2*np.pi*j*x))
            cols.append(np.sin(2*np.pi*j*x))
        X = np.column_stack(cols)
        return X

    def _gcv_fourier_penalty_and_terms(self, m=2):
        """
        Choose lambda for Fourier ridge with frequency penalty w_j proportional to (2π j)^{2m}.
        Return (lambda, effective n terms) where effective terms ~ round((df-1)/2).
        """
        max_n = min(50, (self.data_size - 1)//2)  # cap to avoid huge designs
        X = self._fourier_design(self.t, max_n)
        p = X.shape[1]

        # Penalty matrix P: 0 for intercept, and same weight for cos/sin of frequency j.
        P = np.zeros((p, p))
        idx = 1
        for j in range(1, max_n + 1):
            w = (2*np.pi*j)**(2*m)
            P[idx, idx] = w       # cos
            P[idx+1, idx+1] = w   # sin
            idx += 2

        XtX = X.T @ X
        base = max(1e-6, float(self.smoothing_str))
        lam_grid = np.unique(np.concatenate([10.0**np.linspace(-8, 4, 25), base * 10.0**np.linspace(-4, 4, 17)]))

        best = (np.inf, None, None)  # gcv, lambda, df
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
        eff_n = max(1, int(round(max(0.0, best_df - 1) / 2.0)))  # subtract intercept df, 2 dof per frequency
        return float(best_lam), eff_n

    def _fit_fourier(self, x, a0, an, bn):
        """Reconstruct signal at points x using Fourier coefficients."""
        y = a0 * np.ones_like(x)
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            y += a * np.cos(2 * np.pi * n * x) + b * np.sin(2 * np.pi * n * x)
        return y

    def _fourier_coefficients(self, data, T, n):
        """Compute truncated Fourier series coefficients for one-dimensional data."""
        N = len(data)
        fft_output = np.fft.fft(data)
        a0 = fft_output[0].real / N
        # real and imag to cos/sin
        an = 2 * fft_output.real[1:N//2] / N
        bn = -2 * fft_output.imag[1:N//2] / N
        return a0, an[:n], bn[:n]
    
    def fourier_basis(self):
        """
        Generate callable Fourier basis functions [1, cos1, sin1, ..., cos n, sin n].
        """
        bases = [lambda x: np.ones_like(x)]
        for k in range(1, self.n + 1):
            bases.append(lambda x, k=k: np.cos(2*np.pi*k*x))
            bases.append(lambda x, k=k: np.sin(2*np.pi*k*x))
        return bases
    
    ## ---------------------------------------------------------------------
    # --- Wavelet (thresholding with GCV over tau & level) ---
    def _count_nonzero_coeffs(self, coeffs):
        count = 0
        for c in coeffs:
            if isinstance(c, tuple) or isinstance(c, list):
                # pywt return tuples of detail coeffs by axis; flatten
                c = np.asarray(c)
            count += np.count_nonzero(c)
        return count

    def _threshold_coeffs(self, coeffs, tau, mode='soft'):
        thr = []
        for c in coeffs:
            arr = np.asarray(c)
            thr.append(pywt.threshold(arr, tau, mode=mode))
        return thr

    def _recompose_like_input(self, coeffs, wavelet):
        rec = pywt.waverec(coeffs, wavelet)
        # If self.data has multiple rows, pywt.wavedec/rec expects per-row application.
        # We'll apply along axis=1 (time) for each row.
        return rec

    def _gcv_wavelet_level_and_threshold(self, wavelet_name=None):
        """
        Search over levels and thresholds; choose by GCV. 
        df = # nonzero coeffs after thresholding.
        """
        wavelet = wavelet_name or self.fit  # allow string like 'db4'
        max_lev = min(self.wavelet_level if self.wavelet_level else 6, pywt.dwt_max_level(self.data_size, pywt.Wavelet(wavelet).dec_len))
        levels = range(1, max_lev + 1)
        # robust noise scale estimate using MAD from finest detail of first sample
        sigma_ref = None
        best = (np.inf, None, None)
        for lev in levels:
            gbest = (np.inf, None)
            for i, row in enumerate(self.data):
                coeffs = pywt.wavedec(row, wavelet, level=lev)
                if sigma_ref is None and len(coeffs) > 1:
                    d1 = coeffs[-1]
                    sigma_ref = np.median(np.abs(d1 - np.median(d1))) / 0.6745 + 1e-12
                # build a reasonable tau grid per level
            tau_grid = np.linspace(0.5, 3.5, 8) * (sigma_ref if sigma_ref is not None else 1.0)
            # evaluate per tau averaging over samples
            for tau in tau_grid:
                gcv_sum = 0.0
                df_sum = 0.0
                for row in self.data:
                    coeffs = pywt.wavedec(row, wavelet, level=lev)
                    thr = self._threshold_coeffs(coeffs, tau, mode='soft')
                    df = self._count_nonzero_coeffs(thr)
                    yhat = pywt.waverec(thr, wavelet)
                    # length align (waverec may differ by 1 sample depending on boundary)
                    yhat = yhat[:len(row)]
                    gcv_sum += self._gcv_error(row, yhat, df)
                    df_sum += df
                gcv_avg = gcv_sum / len(self.data)
                if gcv_avg < gbest[0]:
                    gbest = (gcv_avg, tau)
            # keep best tau for this level
            if gbest[0] < best[0]:
                best = (gbest[0], lev, gbest[1])
        best_gcv, best_level, best_tau = best
        return int(best_level), float(best_tau)
    
    ## ---------------------------------------------------------------------
    # --- fit dispatcher ---
    def _fit_curves(self, basis=None):
        """
        Fit curves using specified basis type ('bspline', 'fourier', or wavelet name).

        Returns
        -------
        fn_s : list of callables or BSpline objects
            Fitted curve functions for each data row.
        """
        if basis == 'bspline':            
            # handled in __init__
            return self.fn_s
        elif basis == 'fourier':
            fn_s = []; T = 1; coeff = []
            for i in range(len(self.data)):
                a0, an, bn = self._fourier_coefficients(self.data[i], T, self.n)
                # pack coefficients as [a0, cos1, sin1, cos2, sin2, ...] length 2n+1
                packed = np.empty(1 + 2*self.n, dtype=float)
                packed[0] = a0
                for k in range(self.n):
                    packed[1 + 2*k]     = an[k]
                    packed[1 + 2*k + 1] = bn[k]
                coeff.append(packed)
                # closure for evaluation from packed coeffs
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
            # Fourier basis with the same ordering
            self.smoothing_basis = self.fourier_basis()
            return fn_s
        else:
            # wavelet path: use chosen level & (if found) threshold; df counts nonzero coeffs
            wavelet = basis
            fn_s = []
            self.coeffs = []
            for row in self.data:
                coeffs = pywt.wavedec(row, wavelet, level=self.wavelet_level)
                if hasattr(self, "wavelet_threshold"):
                    thr = self._threshold_coeffs(coeffs, self.wavelet_threshold, mode='soft')
                else:
                    # fallback: mild threshold based on MAD
                    d1 = coeffs[-1]
                    sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745 + 1e-12
                    thr = self._threshold_coeffs(coeffs, 2.0 * sigma, mode='soft')
                self.coeffs.append(thr)
                reconstructed = pywt.waverec(thr, wavelet)[:len(row)]
                # wrap into a smooth BSpline for easy evaluation
                t0 = np.linspace(0,1,len(reconstructed))
                spline = LSQUnivariateSpline(t0, reconstructed, t=np.linspace(0,1,max(2, len(reconstructed)//8))[1:-1], k=3) \
                         if len(reconstructed) > 12 else LSQUnivariateSpline(t0, reconstructed, t=[], k=3)
                fn_s.append(spline)
            return fn_s

## ------------------------------------------------------------------------- ##

if __name__ == "__main__":
    # --- Tests ---
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    N_samples = 10
    T = 256
    t = np.linspace(0, 1, T)

    # Ground-truth signal: smooth periodic + low-frequency trend + small kink
    def true_signal(x):
        return 0.6*np.sin(2*np.pi*2*x) + 0.3*np.cos(2*np.pi*3*x) + 0.5*(x-0.5)**2

    Y = np.vstack([true_signal(t) for _ in range(N_samples)])
    noise = 0.15 * rng.standard_normal(size=Y.shape)
    Y_noisy = Y + noise

    # Helper to plot one fit
    def plot_fit(ax, title, t, y_true, y_noisy, fn):
        ax.plot(t, y_noisy, lw=1, alpha=0.5, label="noisy")
        # evaluate on a fine grid for smooth curve
        fine_t = np.linspace(0, 1, 600)
        try:
            y_fit = fn(fine_t)
        except TypeError:
            # fn is a callable returning values, e.g., lambda x: ...
            y_fit = fn(fine_t)
        ax.plot(fine_t, y_fit, lw=2, label="fit")
        ax.plot(t, y_true, lw=1, ls="--", label="true")
        ax.set_title(title)
        ax.legend(loc="best")

    # --- B-spline ---
    # (A) terms=None -> choose lambda via GCV, infer number of basis terms, then fit
    sm_bs_gcv = Smoothing(fit='bspline', terms=None, smoothing_str=0.3, data=Y_noisy)
    # (B) explicit terms -> skip GCV
    sm_bs_exp = Smoothing(fit='bspline', terms=12, data=Y_noisy)

    # --- Fourier ---
    # (A) n=None -> choose frequency penalty via GCV, infer effective n, then fit
    sm_ft_gcv = Smoothing(fit='fourier', n=None, smoothing_str=0.1, data=Y_noisy)
    # (B) explicit n -> skip GCV
    sm_ft_exp = Smoothing(fit='fourier', n=5, data=Y_noisy)

    # --- Wavelet ---
    # Use a Daubechies wavelet name in 'fit'
    wave_name = 'db4'
    # (A) level=None -> choose (level, threshold) via GCV
    sm_wv_gcv = Smoothing(fit=wave_name, wavelet_level=None, data=Y_noisy)
    # (B) explicit level -> skip GCV (still applies a mild MAD-based soft threshold if no threshold stored)
    sm_wv_exp = Smoothing(fit=wave_name, wavelet_level=4, data=Y_noisy)

    # --- Plot results on first sample ---
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    i = 0  # sample index

    # B-spline
    plot_fit(axes[0,0], f"B-spline (GCV: terms={sm_bs_gcv.num_knots}, λ≈{sm_bs_gcv.smoothing_str:.3g})",
             t, Y[i], Y_noisy[i], sm_bs_gcv.fn_s[i])
    plot_fit(axes[0,1], "B-spline (explicit terms=12)",
             t, Y[i], Y_noisy[i], sm_bs_exp.fn_s[i])

    # Fourier
    # For plotting Fourier, create a closure that uses the chosen coefficients
    def fourier_callable(sm, idx):
        a0, an, bn = sm.coeffs[idx]
        return lambda x: sm._fit_fourier(x, a0, an, bn)

    plot_fit(axes[1,0], f"Fourier (GCV: n={sm_ft_gcv.n}, λ≈{sm_ft_gcv.smoothing_str:.3g})",
             t, Y[i], Y_noisy[i], fourier_callable(sm_ft_gcv, i))

    plot_fit(axes[1,1], "Fourier (explicit n=5)",
             t, Y[i], Y_noisy[i], fourier_callable(sm_ft_exp, i))

    # Wavelet
    plot_fit(axes[2,0], f"Wavelet {wave_name} (GCV: level={sm_wv_gcv.wavelet_level}, τ≈{getattr(sm_wv_gcv,'wavelet_threshold',np.nan):.3g})",
             t, Y[i], Y_noisy[i], sm_wv_gcv.fn_s[i])
    plot_fit(axes[2,1], f"Wavelet {wave_name} (explicit level=4)",
             t, Y[i], Y_noisy[i], sm_wv_exp.fn_s[i])

    for ax in axes[-1]:
        ax.set_xlabel("t")
    plt.suptitle("Smoothing tests: B-spline, Fourier, Wavelet (first sample)")
    plt.show()
    
## ------------------------------------------------------------------------- ##
