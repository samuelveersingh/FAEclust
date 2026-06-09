# Example: cluster the real UCR "Plane" time-series dataset with FAEclust.

import numpy as np
import time
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

from FAEclust import (
    smoothing_features, bspline_basis, rescale,
    TimeSeriesDistance, NearestNeighborsOpt, FunctionalAutoencoder,
)

start = time.time()

# 1. Load dataset -----------------------------------------------------------
dataset_name = 'Plane'
from aeon.datasets import load_classification
X, y = load_classification(dataset_name)
X, y = rescale(np.array(X), y, dataset_name)

# 2. Pairwise distance between curves (fast DTW) ----------------------------
n_samples, n_features, n_timesteps = X.shape
D = TimeSeriesDistance(X, metric='fastdtw', n_jobs=-1).compute_distances()

# 3. Nearest-neighbor graph + similarity ------------------------------------
# Pick the smallest number of neighbors that keeps the graph connected,
# capped at n//5 to keep the graph sparse (using too many neighbors hurts
# clustering).
opt = NearestNeighborsOpt(D)
m_con = opt.estimate_optimal_m(method='connectivity', max_m=X.shape[0] - 1)
m = int(max(5, min(m_con, X.shape[0] // 5)))
print(f"nearest-neighbor graph size m={m} (connectivity, capped at n//5)")
neighbors_dict = opt.get_nearest_neighbors(opt_m=m)
sim_matrix = opt.compute_similarity(neighbors_dict)

# 4. Smoothing (B-spline) with functional standardization -------------------
m_basis, dis_p = 50, 300
t = np.linspace(0, 1, dis_p)
coeffs, curves, basis_smoothing = smoothing_features(
    X, m=m_basis, dis_p=dis_p, fit='bspline', standardize=True)
print(f'{dataset_name}: pre-processing Time = {time.time() - start:.2f} s')

# 5. Train ------------------------------------------------------------------
p = n_features
layers = [32, 16, 8, 16, 32, 32, 32]
l_basis = 50
basis_input = bspline_basis(num_basis=l_basis)

model = FunctionalAutoencoder(
    p, layers, l=l_basis, m=m_basis,
    basis_smoothing=basis_smoothing, basis_input=basis_input,
    lambda_e=0.5, lambda_d=0.05, lambda_c=0.5,
    t=t, sim_matrix=sim_matrix, tau=0.9, use_bn=True, manifold=None)
model.model_summary()

start = time.time()
model.train_model(
    coeffs, epochs=100, learning_rate=1e-3, batch_size=16,
    neighbors_dict=neighbors_dict, sim_matrix=sim_matrix, beta=0.9,
    pretrain_epochs=30, criterion='silhouette', verbose=False)
print(f'{dataset_name}: network training Time = {time.time() - start:.2f} s')

# 6. Evaluate ---------------------------------------------------------------
S, labels = model.predict(coeffs, batch_size=16)
print(f'AMI score: {adjusted_mutual_info_score(y, labels):.4f}')
print(f'ARI score: {adjusted_rand_score(y, labels):.4f}')
print(f'Number of clusters predicted: {len(np.unique(labels))}')
