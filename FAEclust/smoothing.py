import numpy as np
import pywt
from scipy.interpolate import splrep, BSpline, LSQUnivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
plt.rc('font', size=13)
plt.rcParams['figure.constrained_layout.use'] = True

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
        Smoothing parameter s for B-spline fitting (if unspecified, optimized).
    terms : int or None
        Number of basis terms / knots (if None, optimized automatically).
    wavelet_level : int
        Decomposition level for wavelet basis.
    data : array-like, shape (N_samples, N_timepoints)
        Input data matrix to fit.
    """
    def __init__(self, 
                  dis_p=300,
                  fit='bspline',
                  n=3,                     # number of fourier terms (2n+1)
                  smoothing_str=0.3,       # bspline parameter
                  terms = None,
                  wavelet_level=5,         # resolution of wavelet fit
                  data = None,
                  # data_path='Research/FAEclass/datasets/canadian_temperature_daily.csv'
                  ):
        # Store input settings
        self.smoothing_str = smoothing_str
        self.n = n              # Number of sine and cosine terms
        self.dis_p = dis_p      # number of discrete points for evaluations
        self.fit = fit
        self.wavelet_level = wavelet_level
        self.num_knots = terms
        self.data = data
        
        # Data dimensions and grids
        self.data_size = self.data.shape[1]
        self.t = np.linspace(0, 1, self.data_size)      # coarse grid
        self.fine_t = np.linspace(0, 1, self.dis_p)     # fine grid for evaluation
        
        # If number of knots not provided, auto-select based on fit type
        if fit == 'bspline' and self.num_knots == None:
            # Optimize smoothing parameter via GCV
            self.smoothing_str = self._find_optimal_bspline_smoothing()
        elif fit == 'fourier' and self.num_knots == None:
            # Optimize number of Fourier terms via GCV
            self.n = self._find_optimal_fourier_terms()
        elif self.num_knots == None:
            # Optimize wavelet decomposition level via GCV
            self.wavelet_level = self._find_optimal_wavelet_level()
            # self.smoothing_basis = self.wavelet_basis(fit)
        
        # Fit curves using chosen basis or number of knots
        if self.num_knots == None:
            # Fit using basis method directly
            self.fn_s = self._fit_curves(basis=self.fit)
            
        if self.num_knots != None:
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
            
            def bspline_basis(num_basis=None, degree=3):
                num_knots = num_basis + degree + 1
                t_t = np.concatenate(([0] * degree, np.linspace(0, 1, num_knots - 2 * degree), [1] * degree))
                basis_input = []
                for i in range(num_basis):
                    # coefficients: only one non-zero entry per basis function
                    basis_coefs = np.zeros(num_knots - degree - 1)
                    basis_coefs[i] = 1
                    basis_input.append(BSpline(t_t, basis_coefs, degree))
                return basis_input
            self.smoothing_basis = bspline_basis(num_basis = self.num_knots)
    
    def _fit_bsplines(self):
        """
        Fit B-spline functions to each data row using scipy.splrep.

        Returns
        -------
        splines : list of BSpline
            Fitted spline objects per row.
        """
        splines = []
        for row in self.data:
            tck = splrep(self.t, row, s=self.smoothing_str)
            splines.append(BSpline(*tck))
        return splines
    
    def _fourier_coefficients(self, data, T, n):
        """
        Compute Fourier series coefficients for one-dimensional data.

        Returns
        -------
        a0 : float
            Zero-frequency term.
        an, bn : array-like
            Cosine and sine coefficients (length ≈ n//2).
        freq : array-like
            Corresponding frequencies.
        """
        N = len(data)
        frequency = np.fft.fftfreq(N, T/N)
        fft_output = np.fft.fft(data)
        a0 = fft_output[0].real / N
        an = 2 * fft_output.real[1:N//2] / N
        bn = -2 * fft_output.imag[1:N//2] / N
        return a0, an[:n//2], bn[:n//2], frequency[1:N//2]
    
    def _fit_fourier(self, x, a0, an, bn):
        """
        Reconstruct signal at points x using Fourier coefficients.
        """
        y = a0 * np.ones_like(x)
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            y += a * np.cos(2 * np.pi * n * x) + b * np.sin(2 * np.pi * n * x)
        return y
    
    def fourier_basis(self):
        """
        Generate callable Fourier basis functions up to order n.

        Returns
        -------
        bases : list of callables
            Each function maps x to basis value.
        """
        bases = []
        for i in range(self.n+1):
            if i == 0:
                bases.append(lambda x, idx=0: np.ones_like(x))
            else:
                bases.append(lambda x, idx=i: np.cos(2*np.pi*idx*x))
                bases.append(lambda x, idx=i: np.sin(2*np.pi*idx*x))
        return bases
    
    def _fit_curves(self, basis=None):
        """
        Fit curves using specified basis type ('bspline', 'fourier', or wavelet name).

        Returns
        -------
        fn_s : list of callables or BSpline objects
            Fitted curve functions for each data row.
        """
        if basis == 'bspline':
            return self._fit_bsplines()
        elif basis == 'fourier':
            fn_s = []; T = 1; coeff = []
            for i in range(len(self.data)):
                a0, an, bn, frequencies = self._fourier_coefficients(self.data[i], T, self.n*2)
                coeff.append([a0, an, bn])
                fn_s.append(lambda x, idx=i: self._fit_fourier(x, 
                                                    coeff[idx][0],coeff[idx][1],coeff[idx][2]))
            self.coeffs = coeff
            return fn_s
        else:
            coeffs = pywt.wavedec(self.data, basis, level=self.wavelet_level)
            coef = [np.where(np.abs(coeff) < 0.1, 0, coeff) for coeff in coeffs]
            self.coeffs = coef
            reconstructed = pywt.waverec(coef, basis)
            fn_s = []
            for row in reconstructed:
                tck = splrep(np.linspace(0,1,len(row)), row, s=0)
                fn_s.append(BSpline(*tck))
            return fn_s

    def _gcv_error(self, y_true, y_pred, df):
        """
        Compute Generalized Cross-Validation (GCV) error.
        """
        residual = y_true - y_pred
        n = len(y_true)
        rss = np.sum(residual**2)
        penalty = df  # to penalizes the degrees of freedom to avoid overfitting
        gcv = rss / (n - df)**2 + penalty
        return gcv
    
    def _find_optimal_bspline_smoothing(self):
        """
        Grid-search over s in [0, smoothing_str] to minimize GCV.
        """
        best_gcv = float('inf')
        best_s = None
        for s in np.linspace(0.0, self.smoothing_str, 10):
            total_gcv = 0
            for row in self.data:
                tck = splrep(self.t, row, s=s)
                spline = BSpline(*tck)
                y_pred = spline(self.t)
                df = tck[2] + len(tck[0]) - 1  # adjusted for the B-spline's degree
                gcv = self._gcv_error(row, y_pred, df)
                total_gcv += gcv
            avg_gcv = total_gcv / len(self.data)
            if avg_gcv < best_gcv:
                best_gcv = avg_gcv
                best_s = s
        return best_s

    def _find_optimal_fourier_terms(self):
        """
        Search number of Fourier terms n in [1..self.n] minimizing GCV.
        """
        best_gcv = float('inf')
        best_n = None
        for n in range(1, self.n+1):
            total_gcv = 0
            for row in self.data:
                a0, an, bn, frequencies = self._fourier_coefficients(row, 1, n*2)
                y_pred = self._fit_fourier(self.t, a0, an, bn)
                df = 1 / (2*n + 1)
                gcv = self._gcv_error(row, y_pred, df)
                total_gcv += gcv
            avg_gcv = total_gcv / len(self.data)
            if avg_gcv < best_gcv:
                best_gcv = avg_gcv
                best_n = n
        return best_n
    
    def _find_optimal_wavelet_level(self):
        """
        Find best wavelet decomposition level by minimizing GCV.
        """
        best_gcv = float('inf')
        best_level = None
        for level in range(1, 8):
            total_gcv = 0
            coeffs = pywt.wavedec(self.data, self.fit, level=level)
            reconstructed = pywt.waverec(coeffs, self.fit)
            for i, row in enumerate(self.data):
                y_pred = reconstructed[i]
                df = len(coeffs)
                gcv = self._gcv_error(row, y_pred, df)
                total_gcv += gcv / self.wavelet_level
            avg_gcv = total_gcv / len(self.data)
            if avg_gcv < best_gcv:
                best_gcv = avg_gcv
                best_level = level
        return best_level