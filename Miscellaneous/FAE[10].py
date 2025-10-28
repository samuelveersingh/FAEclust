import numpy as np
from aeon.datasets import load_classification
from scipy.interpolate import BSpline

import tensorflow as tf
from tensorflow.keras import Model, layers

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

## ------------------------------------------------------------------------- ##
class FunctionalEncoder(layers.Layer):
    """Functional encoder that maps B-spline coefficients to a latent vector."""
    def __init__(self, latent_dim, n_channels, n_basis, Gram):
        super().__init__()
        self.Gram = tf.constant(Gram.astype(np.float32))
        # Theta shape: (channels, latent_dim, basis)
        self.Theta = self.add_weight(
            shape=(n_channels, latent_dim, n_basis),
            initializer="random_normal",
            trainable=True,
            name="Theta"
        )

    def call(self, inputs):
        # inputs: (batch, channels, basis)
        # Multiply on basis dimension by Gram: xg[b, c, b’] = sum_b X[b,c,b]*G[b,b’]
        xg = tf.tensordot(inputs, self.Gram, axes=[[2], [0]])
        # Contract channels & basis with Theta -> (batch, latent_dim)
        z = tf.einsum("icb,clb->il", xg, self.Theta)
        return tf.nn.tanh(z)


class FunctionalDecoder(layers.Layer):
    """Functional decoder that maps latent vector back to time domain via B-spline basis."""
    def __init__(self, latent_dim, n_channels, n_basis, basis_mat):
        super().__init__()
        # (basis, time)
        self.basis = tf.constant(basis_mat.T.astype(np.float32))
        # Beta shape: (channels, latent_dim, basis)
        self.Beta = self.add_weight(
            shape=(n_channels, latent_dim, n_basis),
            initializer="random_normal",
            trainable=True,
            name="Beta"
        )

    def call(self, inputs):
        # inputs: (batch, latent_dim)
        # reconstruct coefficients per channel: (batch, channels, basis)
        recon_c = tf.einsum("il,clb->icb", inputs, self.Beta)
        # map coefficients to time domain: (batch, channels, time)
        x_hat = tf.tensordot(recon_c, self.basis, axes=[[2], [0]])
        return x_hat

## ------------------------------------------------------------------------- ##
## FAE network + k-means clustering 
class FAE:
    """
    Functional AutoEncoder (FAE) pipeline:
      - builds B-spline basis & Gram matrix
      - projects series onto basis
      - trains functional autoencoder
      - extracts latent embeddings
      - runs KMeans over a range of cluster counts and reports metrics
    """
    def __init__(
        self,
        latent_dim: int = 16,
        batch_size: int = 16,
        lr: float = 1e-3,
        epochs: int = 200,
        n_basis: int = 50,
        degree: int = 3,
        noc_range: tuple = (2, 10),  # [low, high)
        verbose: int = 0
    ):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.n_basis = n_basis
        self.degree = degree
        self.noc_range = noc_range  # interpreted as range(low, high)
        self.verbose = verbose

        # placeholders populated after fitting
        self.dataset_name = None
        self.X = None          # (n, C, T)
        self.y = None
        self.X_coeff = None    # (n, C, B)
        self.basis_matrix = None  # (T, B)
        self.Gram = None          # (B, B)
        self.model = None
        self.encoder = None
        self.latent = None     # (n, latent_dim)
    
    ## --------------------------------------------------------------------- ##
    ## utilities
    def _build_basis_and_gram(self, n_timepoints: int):
        """Build B-spline basis matrix (T,B) and Gram matrix (B,B)."""
        n_basis = self.n_basis
        degree = self.degree

        # internal knots
        n_internal = n_basis - (degree + 1)
        if n_internal > 0:
            interior = np.linspace(0, 1, n_internal + 2)[1:-1]
        else:
            interior = np.array([])
        knots = np.concatenate(([0.0] * (degree + 1), interior, [1.0] * (degree + 1)))

        # basis matrix
        t_pts = np.linspace(0, 1, n_timepoints)
        basis_matrix = np.zeros((n_timepoints, n_basis))
        for i in range(n_basis):
            coeff = np.zeros(n_basis)
            coeff[i] = 1.0
            spline = BSpline(knots, coeff, degree)
            basis_matrix[:, i] = spline(t_pts)

        # Gram (approx integral of product via Riemann sum)
        dt = 1.0 / (n_timepoints - 1)
        Gram = (basis_matrix.T @ basis_matrix) * dt

        self.basis_matrix = basis_matrix
        self.Gram = Gram

    def _project_to_coeffs(self, X: np.ndarray) -> np.ndarray:
        """Project each series (per channel) to B-spline coefficients."""
        n_samples, n_channels, _ = X.shape
        n_basis = self.n_basis
        X_coeff = np.zeros((n_samples, n_channels, n_basis))
        # Solve least squares: basis_matrix @ coeff ≈ series
        for i in range(n_samples):
            for j in range(n_channels):
                series = X[i, j, :]
                coeff, *_ = np.linalg.lstsq(self.basis_matrix, series, rcond=None)
                X_coeff[i, j, :] = coeff
        return X_coeff

    ## --------------------------------------------------------------------- ##
    ## Model pipeline
    def _build_model(self, n_channels: int, n_basis: int, n_timepoints: int):
        """Build and compile the functional autoencoder and its encoder head."""
        enc = FunctionalEncoder(self.latent_dim, n_channels, n_basis, self.Gram)
        dec = FunctionalDecoder(self.latent_dim, n_channels, n_basis, self.basis_matrix)

        inp = tf.keras.Input(shape=(n_channels, n_basis))
        z = enc(inp)
        out = dec(z)
        ae = Model(inp, out, name="FAE")
        ae.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="mse")

        # encoder model for inference
        encoder_model = Model(ae.input, enc(ae.input), name="FAE_Encoder")

        self.model = ae
        self.encoder = encoder_model

    ## --------------------------------------------------------------------- ##
    ## Load data 
    def load_data(self, dataset_name: str):
        """Load a univariate/multivariate time-series classification dataset from UCR via aeon."""
        self.dataset_name = dataset_name
        X, y = load_classification(dataset_name)
        X = np.array(X)  # (n, C, T)
        self.X, self.y = X, np.array(y)

        # Build basis + Gram for this time length
        _, _, n_timepoints = X.shape
        self._build_basis_and_gram(n_timepoints)

        # Pre-project to coeffs once
        self.X_coeff = self._project_to_coeffs(self.X)

        # Build model
        n_samples, n_channels, n_basis = self.X_coeff.shape
        self._build_model(n_channels, n_basis, n_timepoints)

        print(f"\n=== Dataset: {dataset_name} ===")
        print(f"Samples: {n_samples} | Channels: {n_channels} | Timepoints: {n_timepoints}")
    
    ## --------------------------------------------------------------------- ##
    ## Train
    def fit(self):
        """Train the autoencoder."""
        if self.model is None or self.X is None or self.X_coeff is None:
            raise RuntimeError("Call load_data(dataset_name) before fit().")
        self.model.fit(
            self.X_coeff.astype(np.float32),
            self.X.astype(np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
    
    ## embeddings 
    def embed(self) -> np.ndarray:
        """Get latent embeddings for all samples."""
        if self.encoder is None or self.X_coeff is None:
            raise RuntimeError("Model not built or data not loaded.")
        self.embeddings = self.encoder.predict(self.X_coeff.astype(np.float32), verbose=0)
        return self.embeddings
    
    ## --------------------------------------------------------------------- ##
    ## clustering
    def cluster(self):
        """
        Run KMeans for k in [noc_range[0], noc_range[1]) and compute:
          - silhouette score
          - adjusted mutual information (AMI)
          - adjusted rand index (ARI)
        Returns a dict with per-k metrics and the best (by silhouette).
        """
        if self.embeddings is None:
            self.embed()

        Z = self.embeddings
        y = self.y

        low, high = self.noc_range
        ks = list(range(low, min(high, len(Z))))  # end-exclusive, safeguard upper bound
        sil_scores, nks, amis, aris, labelings = [], [], [], [], []
        
        print("\nClustering:")
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10)
            lbls = km.fit_predict(Z)
            nk = len(np.unique(lbls))
            # metrics
            sil = silhouette_score(Z, lbls) if nk > 1 else -1.0
            ami = adjusted_mutual_info_score(y, lbls)
            ari = adjusted_rand_score(y, lbls)

            sil_scores.append(sil)
            nks.append(nk)
            amis.append(ami)
            aris.append(ari)
            labelings.append(lbls)

            print(f"k={k:2d} | clusters={nk:2d} | silhouette={sil:.4f} | AMI={ami:.4f} | ARI={ari:.4f}")

        # optimum NoC with silhouette
        best_idx = int(np.argmax(sil_scores))
        best_k = ks[best_idx]
        best = {
            "k": best_k,
            "silhouette": sil_scores[best_idx],
            "AMI": amis[best_idx],
            "ARI": aris[best_idx],
            "labels": labelings[best_idx],
        }

        print("-" * 28)
        print(f"Optimum number of clusters: {best_k}")
        print(f"AMI: {best['AMI']:.4f} | ARI: {best['ARI']:.4f}")
        return {
            "ks": ks,
            "silhouette": sil_scores,
            "AMI": amis,
            "ARI": aris,
            "labelings": labelings,
            "best": best
        }

    ## --------------------------------------------------------------------- ##
    def train(self, dataset_name: str):
        """Load -> train -> embed -> cluster. Returns clustering summary dict."""
        self.load_data(dataset_name)
        self.fit()
        self.embed()
        return self.cluster()


## ------------------------------------------------------------------------- ##
if __name__ == "__main__":
    # Plane dataset from UCR
    fae = FAE(
        latent_dim=16,
        batch_size=16,
        lr=1e-3,
        epochs=200,
        n_basis=50,
        degree=3,
        noc_range=(2, 10)   # tries k = 2..9 
    )
    
    dataset_name = "Plane"
    results = fae.train(dataset_name)
    embeddings = fae.embeddings               # (n_samples, latent_dim)

## ------------------------------------------------------------------------- ##