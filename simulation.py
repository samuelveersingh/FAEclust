# Example: cluster a synthetic "pendulum" dataset with FAEclust.
# Shows the full workflow from data generation through to evaluation.

import numpy as np
import time
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

from FAEclust import (
    smoothing_features, bspline_basis, rescale,
    DatasetGenerator, TimeSeriesDistance, NearestNeighborsOpt,
    FunctionalAutoencoder,
)

start = time.time()

# 1. Synthetic dataset (pendulum, 4 clusters) -------------------------------
name, n_samples, n_features, n_steps, n_clusters = 'pendulum', 200, 2, 100, 4
print(f"Generating {name}: ({n_samples}, {n_features}, {n_steps}, {n_clusters} clusters)")
gen = DatasetGenerator(n_samples, n_features, n_steps, n_clusters)
X, y = getattr(gen, f"generate_{name}")()
X, y = rescale(X, y, name)

# 2. Pairwise elastic distance between curves -------------------------------
D = TimeSeriesDistance(X, metric='elastic', n_jobs=-1).compute_distances()

# 3. Nearest-neighbor graph + similarity ------------------------------------
# Pick the smallest number of neighbors that keeps the graph connected,
# capped at n//5 to keep the graph sparse. (Using too many neighbors makes
# an almost fully connected graph that hurts clustering.)
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
print(f'{name}: pre-processing Time = {time.time() - start:.2f} s')

# 5. Train the functional autoencoder ---------------------------------------
p = n_features
layers = [16, 8, 2, 8, 16, 16, 16]
l_basis = 50
basis_input = bspline_basis(num_basis=l_basis)
lambda_e, lambda_d, lambda_c = 0.5, 0.05, 0.5

model = FunctionalAutoencoder(
    p, layers, l=l_basis, m=m_basis,
    basis_smoothing=basis_smoothing, basis_input=basis_input,
    lambda_e=lambda_e, lambda_d=lambda_d, lambda_c=lambda_c,
    t=t, sim_matrix=sim_matrix, tau=0.9, use_bn=True, manifold=None)
model.model_summary()

start = time.time()
model.train_model(
    coeffs, epochs=150, learning_rate=1e-3, batch_size=32,
    neighbors_dict=neighbors_dict, sim_matrix=sim_matrix, beta=0.9,
    pretrain_epochs=50, criterion='silhouette', verbose=False)
print(f'{name}: network training Time = {time.time() - start:.2f} s')

# 6. Evaluate ---------------------------------------------------------------
S, labels = model.predict(coeffs, batch_size=32)
print(f'AMI score: {adjusted_mutual_info_score(y, labels):.4f}')
print(f'ARI score: {adjusted_rand_score(y, labels):.4f}')
print(f'Number of clusters predicted: {len(np.unique(labels))}')
