
# FAEclust: Cluster Analysis of Multi-Dimensional Functional Data through Non-linear Representation Learning

We introduce **FAEclust**, a functional autoencoder framework for clustering multi-dimensional functional data (vector‐valued random functions). Our key contributions include:

* A **functional encoder** capturing complex nonlinear dependencies across component functions.
* A **universal‐approximator decoder** reconstructing both Euclidean and manifold‐valued functional data.
* **Regularization** strategies on functional weights/biases for stability and robustness.
* A **clustering loss** integrated into training to shape latent representations for clustering, with a **shape‐informed** objective to resist phase variations and time warping.
* A theoretical proof of the decoder’s universal approximation property, and extensive empirical validation, are provided in the main paper.

---

<!--
## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/_/FAEclust.git
   cd FAEclust
   ```
2. Create a Python environment and install dependencies:

   ```bash
   python3 -m venv FAE
   source FAE/bin/activate
   pip install -r requirements.txt
   ```

---
-->

## 🛠️ Core API

FAEclust exposes four main stages:

1. **Smoothing & Basis Construction**
2. **Distance Computation**
3. **Graph Construction**
4. **Functional Autoencoder Training & Clustering**

All modules are available at the package level:

```python
from FAEclust import (
    smoothing_features,    # functional smoothing
    bspline_basis,         # B-spline basis functions
    rescale,               # min–max scaling & label encoding
    TimeSeriesDistance,    # DTW / elastic distance
    NearestNeighborsOpt,   # m-NN graph & similarity
    FunctionalAutoencoder  # the joint FAE + clustering model
)
```

---

### 1. Smoothing & Basis Construction (preprocessing)

* **`smoothing_features(data, m, dis_p, fit='bspline')`**
  Smooth each feature dimension using the specified basis and return:

  * `coeffs`: array of shape `(n_samples, p, m)`—basis coefficients
  * `curves`: smoothed curves evaluated on the coarse grid
  * `basis_smoothing`: list of smoothing basis functions


* **`bspline_basis(num_basis, degree=3)`**
  Generate a list of `BSpline` basis callables for functional‐weight inputs.


* **`rescale(X, y, name)`**
  Standardize `X` and encode labels `y` as zero‐based integers.

---

### 2. Distance Computation

* **`TimeSeriesDistance(X, metric='elastic', n_jobs=-1)`**
  Compute pairwise distances for multivariate time series:

  * `'fastdtw'`: standard DTW
  * `'elastic'`: SRVF-based elastic amplitude distance
    Methods:
  * `compute_distances()` → returns `(n_samples, n_samples)` distance matrix


---

### 3. Graph Construction

* **`NearestNeighborsOpt(dist_matrix)`**
  Determine an optimal nearest neighbor count `m` via:

  1. **Connectivity**: first `m` yielding one connected component
  2. **Avg-distance knee**: knee in mean m-th neighbor distance curve
     Methods:

  * `estimate_optimal_m(method='avg_distance' or 'connectivity')`
  * `get_nearest_neighbors(opt_m)` → dict of neighbor indices
  * `compute_similarity(neighbors_dict)` → dense similarity matrix


---

### 4. Functional Autoencoder & Clustering

#### `FunctionalAutoencoder` constructor parameters :

| Argument              | Description                                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `p` (int)             | Input functional dimension.                                                                 |
| `layers` (list\[int]) | `[q₁, …, s, …, Q₁, Z₁, Z₂]` specifying: first encoder and last three decoder functional‐weight counts (`q₁`,`Q₁`,`Z₁`,`Z₂`), with MLP  layer sizes in‐between (` …, s, …`). |
| `l_basis` (int)             | Number of basis functions for **encoder** expansion.                                                                           |
| `m_basis` (int)             | Number of basis functions for **smoothing** expansion.                                                                         |
| `basis_smoothing`     | List of smoothing basis callables from `smoothing_features()`.                                                                 |
| `basis_input`         | List of input basis callables from `bspline_basis()`.                                                                          |
| `lambda_e` (float)    | Weight for encoder L₂ orthonormality penalty.                                                                                  |
| `lambda_d` (float)    | Weight for decoder L₁ sparsity penalty.                                                                                        |
| `lambda_c` (float)    | Weight for latent‐space clustering loss.                                                                                       |
| `t` (array)           | Time grid over which functional data are defined.                                                                              |
| `sim_matrix`          | Precomputed similarity matrix for convex clustering.                                                                           |

#### Training & Inference

```python
# model setup
fae = FunctionalAutoencoder(
    p, layers,
    l=l_basis, m=m_basis,
    basis_smoothing=basis_smoothing,
    basis_input=basis_input,
    lambda_e=λ_e, lambda_d=λ_d, lambda_c=λ_c,
    t=t, sim_matrix=sim_matrix
)
fae.model_summary()

# training
fae.train(
    X_train=coeffs,           # from smoothing_features()
    epochs=E,                 # e.g. 100
    learning_rate=LR,         # e.g. 1e-3
    batch_size=B,             # e.g. 16
    neighbors_dict=neighbors, # from NearestNeighborsOpt
    sim_matrix=sim_matrix
)

# clustering & evaluation
S, pred_labels = fae.predict(coeffs, batch_size=B)
```

---

## 🧩 Modular Pipeline

The modular pipeline/framework for FAEclust comprises the following stages:

![Modular Pipeline Diagram](modular_pipeline.png)

1. **Raw Dataset**: N samples of p-dimensional multivariate functional data.
2. **Smoothing & Basis Construction**: Convert raw curves into basis functions and coefficients via `smoothing_features()`.
3. **Distance Computation**: Compute pairwise distances (e.g., DTW, elastic) using `TimeSeriesDistance`.
4. **Graph Construction**: Build an m-nearest-neighbor graph and similarity matrix via `NearestNeighborsOpt`.
5. **Functional Autoencoder**: Encode coefficients into a latent vector space and decode back to functional space, balancing reconstruction and clustering objectives.
6. **Convex Clustering**: Apply clustering loss in latent space to obtain final cluster assignments.

---

## 🏗️ Model Framework Schematic

The overall model framework integrates the autoencoder network with clustering updates in an joint optimization scheme:

![Model Framework Schematic](framework.png)

* **Finite-dimensional Functional Space**: Representation via basis expansion of input functions.
* **Encoder (ℇ)**: Projects functional input into a low-dimensional latent vector space.
* **Decoder (𝒟)**: Reconstructs functional output from latent vectors, ensuring expressive power.
* **Loss Components**:

  * **Reconstruction Loss**: Measures discrepancy between input and decoder output.
  * **Clustering Loss**: Encourages latent embeddings to form well-separated clusters (convex clustering in latent space).
  * **Regularization Penalties**: L₂ orthonormality for encoder weights and L₁ sparsity for decoder weights.
* **Training Loop**:

  1. **Network Update**: Backpropagate combined loss to update encoder/decoder weights.
  2. **Cluster Update**: Update cluster centroids/assignments based on current latent representations.

---

## 🚀 Usage Example: 

### 1. UCR “Plane” Dataset

[🔗 View the full notebook](test.ipynb)

### 2. Simulated “Pendulum” Dataset

[🔗 View the full notebook](simulation.ipynb)

---
