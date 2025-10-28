from string import ascii_lowercase
from typing import Callable, Optional, Union, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from aeon.datasets import load_classification

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score

## ------------------------------------------------------------------------- ##
## Utilities 
def define_basis(basis_type: str, n_functions: int, resolution: Union[int, Tuple[int, ...]]) -> tf.Tensor:
    """
    Define basis functions sampled on an equidistant grid.
    """
    if isinstance(resolution, int):
        resolution = (resolution,)

    if basis_type == 'Legendre' and len(resolution) == 1:
        # Legendre via scipy.special.legendre evaluation on [-1, 1]
        support = np.linspace(-1, 1, resolution[0], dtype=np.float64)
        from scipy.special import legendre
        basis = [legendre(i)(support) for i in range(n_functions)]
        basis = np.stack(basis, axis=-1).astype(np.float32)
        return tf.convert_to_tensor(basis, dtype=tf.float32)

    elif basis_type == 'Fourier' and len(resolution) == 1:
        support = tf.linspace(tf.constant(0.0, tf.float32), tf.constant(1.0, tf.float32), resolution[0])
        sqrt2 = tf.math.sqrt(tf.constant(2.0, tf.float32))
        pi32 = tf.constant(np.pi, tf.float32)

        def _cos(k):
            return tf.math.cos(pi32 * tf.cast(k, tf.float32) * support)

        def _sin(k):
            return tf.math.sin(pi32 * tf.cast(k, tf.float32) * support)

        basis_list = []
        for i in range(n_functions):
            if i == 0:
                basis_i = tf.ones_like(support, dtype=tf.float32)
            elif i % 2 == 1:
                # odd index -> sine with frequency (i+1)
                basis_i = sqrt2 * _sin(i + 1)
            else:
                # even index (i>0) -> cosine with frequency i
                basis_i = sqrt2 * _cos(i)
            basis_list.append(basis_i)

        basis = tf.stack(basis_list, axis=-1)  # (T, n_functions)
        return tf.cast(basis, tf.float32)

    else:
        raise NotImplementedError(f"Basis type {basis_type} not implemented for resolution={resolution}.")


def calculate_linear_combination(scalar_weights: tf.Tensor, basis: tf.Tensor) -> tf.Tensor:
    """
    Linear combination of scalar weights with basis functions.
    weights: (n_basis_functions, n_channels, n_filters)
    basis:   resolution + (n_basis_functions,)
    returns: resolution + (n_channels, n_filters)
    """
    if scalar_weights.dtype != basis.dtype:
        scalar_weights = tf.cast(scalar_weights, basis.dtype)

    n_dims = len(basis.shape) - 1
    dim_indices = ascii_lowercase[:n_dims]
    equation = f'xyz,{dim_indices}x->{dim_indices}yz'
    return tf.einsum(equation, scalar_weights, basis)

## ------------------------------------------------------------------------- ##
## Functional convolution layer 
class FunctionalConvolution(layers.Layer):
    def __init__(
        self,
        n_filters: int,
        basis_options: Dict,
        padding: str = 'VALID',
        pooling: bool = False,
        activation: Optional[Union[str, Callable]] = None,
        calculate_weights: Callable = calculate_linear_combination,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.basis_options = basis_options
        self.padding = padding
        self.pooling = pooling
        self.activation = tf.keras.activations.get(activation)
        self.calculate_weights = calculate_weights

        n_basis_functions = self.basis_options.get('n_functions', 1)
        resolution = self.basis_options.get('resolution', 1)
        basis_type = self.basis_options.get('basis_type', 'Fourier')

        # basis.shape = resolution + (n_functions,)
        self.basis = define_basis(basis_type, n_basis_functions, resolution)
        self.scalar_weights = None  # (n_functions, in_channels, n_filters)

    def build(self, input_shape: tuple):
        n_functions = self.basis.shape[-1]
        n_channels = input_shape[-1]
        self.scalar_weights = self.add_weight(
            name='scalar_weights',
            shape=(n_functions, n_channels, self.n_filters),
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Compute full filters in input space
        weights = self.calculate_weights(self.scalar_weights, self.basis)
        # tf.nn.convolution expects: input (N, *spatial, Cin), filters (*spatial, Cin, Cout)
        outputs = tf.nn.convolution(inputs, weights, padding=self.padding)

        if self.pooling:
            axes = list(range(1, len(self.basis.shape)))   # pool over spatial dims
            outputs = tf.math.reduce_mean(outputs, axis=axes)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

## ------------------------------------------------------------------------- ##
## Functional dense layer 
class FunctionalDense(layers.Layer):
    def __init__(
        self,
        n_neurons: int,
        basis_options: Dict,
        pooling: bool = False,
        activation: Optional[Union[str, Callable]] = None,
        calculate_weights: Callable = calculate_linear_combination,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.basis_options = basis_options
        self.pooling = pooling
        self.activation = tf.keras.activations.get(activation)
        self.calculate_weights = calculate_weights

        n_basis_functions = self.basis_options.get('n_functions', 1)
        resolution = self.basis_options.get('resolution', 1)
        basis_type = self.basis_options.get('basis_type', 'Fourier')
        self.basis = define_basis(basis_type, n_basis_functions, resolution)

        self.scalar_weights = None
        self.call_equation = None

    def build(self, input_shape: tuple):
        resolution = self.basis.shape[:-1]
        if input_shape[1:-1] != resolution:
            raise TypeError(f"Incompatible shapes: input {input_shape[1:-1]} vs basis {resolution}")
        n_functions = self.basis.shape[-1]
        n_channels = input_shape[-1]
        self.scalar_weights = self.add_weight(
            name='scalar_weights',
            shape=(n_functions, n_channels, self.n_neurons),
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32
        )
        n_dims = len(resolution)
        idx = ascii_lowercase[:n_dims]
        self.call_equation = f"x{idx}y,{idx}yz->x{idx}z"

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        weights = self.calculate_weights(self.scalar_weights, self.basis)  # res + (Cin, Cout)
        outputs = tf.einsum(self.call_equation, inputs, weights)
        if self.pooling:
            axes = list(range(1, len(self.basis.shape)))  # average over spatial dims
            outputs = tf.math.reduce_mean(outputs, axis=axes)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

## ------------------------------------------------------------------------- ##
## FNN network
class FNN:
    """
    Functional Neural Network (FNN) pipeline:
      - builds an autoencoder with FunctionalConvolution + FunctionalDense encoder
      - trains to reconstruct input
      - extracts latent embeddings
      - runs KMeans over a range of cluster counts and reports metrics
    """

    def __init__(
        self,
        conv_blocks: Optional[list[Dict]] = None,
        dense_blocks: Optional[list[Dict]] = None,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        noc_range: tuple = (2, 10),     # [low, high)
        verbose: int = 0
    ):
        self.conv_blocks = conv_blocks or []
        self.dense_blocks = dense_blocks or []
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.noc_range = noc_range  # interpreted as range(low, high)
        self.verbose = verbose

        self.dataset_name = None
        self.X = None           # (N, T, C)
        self.y = None           # numeric labels
        self.time_steps = None
        self.n_channels = None
        self.embedding_dim = None
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self.autoencoder: Optional[Model] = None
        self.embeddings = None
    
    ## --------------------------------------------------------------------- ##
    ## Load data 
    @staticmethod
    def _to_numeric_labels(y_raw: np.ndarray) -> np.ndarray:
        if y_raw.dtype.kind in {'U', 'S', 'O'}:
            _, y_num = np.unique(y_raw, return_inverse=True)
        else:
            y_num = y_raw.astype(int).ravel()
        return y_num

    def load(self, dataset_name: str = "Plane"):
        self.dataset_name = dataset_name
        X, y = load_classification(dataset_name)  # default (N, C, T) or (N, T)
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.ndim == 2:
            X = X[:, None, :]
        # to layout (N, T, C)
        X = X.transpose(0, 2, 1).astype(np.float32)

        self.X = X
        self.y = self._to_numeric_labels(y)
        self.time_steps, self.n_channels = X.shape[1], X.shape[2]
        
        print(f"\n=== Dataset: {dataset_name} ===")
        print(f"Samples: {X.shape[0]} | Channels: {self.n_channels} | Timepoints: {self.time_steps}")
        return self
    
    ## --------------------------------------------------------------------- ##
    ## Model pipeline
    def _build_encoder(self) -> Model:
        inp = tf.keras.Input(shape=(self.time_steps, self.n_channels))
        ## Normalize over time and channel dims 
        x = tf.keras.layers.LayerNormalization(axis=[1, 2], center=False, scale=False, epsilon=1e-10)(inp)

        # FunctionalConv blocks
        for i, fo in enumerate(self.conv_blocks):
            fo = dict(fo)
            fo['basis_options'] = dict(fo.get('basis_options', {}))
            fo['basis_options']['resolution'] = self.time_steps
            x = FunctionalConvolution(**fo, name=f"FuncConv_{i}")(x)

        # FunctionalDense blocks
        for i, lo in enumerate(self.dense_blocks):
            lo = dict(lo)
            lo['basis_options'] = dict(lo.get('basis_options', {}))
            lo['basis_options']['resolution'] = self.time_steps
            x = FunctionalDense(**lo, name=f"FuncDense_{i}")(x)

        self.embedding_dim = self.dense_blocks[-1]['n_neurons'] if self.dense_blocks else self.n_channels
        return Model(inp, x, name="encoder")

    def _build_decoder(self) -> Model:
        z = tf.keras.Input(shape=(self.embedding_dim,))
        x = tf.keras.layers.Dense(self.time_steps * self.n_channels, activation=None)(z)
        x = tf.keras.layers.Reshape((self.time_steps, self.n_channels))(x)
        return Model(z, x, name="decoder")

    def build(self):
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        inp = self.encoder.input
        out = self.decoder(self.encoder(inp))
        self.autoencoder = Model(inp, out, name="autoencoder")
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="mse")
        return self

    ## Train
    def fit(self):
        self.autoencoder.fit(
            self.X, self.X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=self.verbose
        )
        return self

    ## embeddings 
    def embed(self) -> np.ndarray:
        self.embeddings = self.encoder.predict(self.X, batch_size=self.batch_size, verbose=0)
        if self.embeddings.ndim == 3:
            self.embeddings = self.embeddings.mean(axis=1)
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
    def train(self, dataset_name: str = "Plane"):
        """Load -> train -> embed -> cluster. Returns clustering summary dict."""
        self.load(dataset_name).build().fit().embed()
        results = self.cluster()
        return results


## ------------------------------------------------------------------------- ##
if __name__ == "__main__":
    # Plane dataset from UCR
    conv_blocks = [
        {
            "n_filters": 16,
            "basis_options": {"n_functions": 5, "resolution": None, "basis_type": "Legendre"},
            "activation": "elu",
            "padding": "SAME",
            "pooling": False
        },
        {
            "n_filters": 16,
            "basis_options": {"n_functions": 5, "resolution": None, "basis_type": "Legendre"},
            "activation": "elu",
            "padding": "SAME",
            "pooling": False
        },
    ]
    dense_blocks = [
        {
            "n_neurons": 16,
            "basis_options": {"n_functions": 4, "resolution": None, "basis_type": "Fourier"},
            "activation": None,      # linear embedding
            "pooling": True          # global average over time → (N, d)
        }
    ]

    fnn = FNN(
        conv_blocks=conv_blocks,
        dense_blocks=dense_blocks,
        epochs=200,
        batch_size=16,
        lr=1e-3,
        noc_range=(2, 10)   # tries k = 2..9
    )
    
    dataset_name="Plane"
    results = fnn.train(dataset_name)
    embeddings = fnn.embeddings
## ------------------------------------------------------------------------- ##