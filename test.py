# Functional Autoencoder Clustering on real Datasets
# ----------------------------------------------------
# This script loads a time series dataset from UCR repository, 
# computes pairwise distances, builds an m-NN graph,
# and trains a functional autoencoder to cluster the data.
# Evaluation uses AMI and ARI clustering metrics.
import numpy as np
from aeon.datasets import load_classification
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
import time 

# Import custom modules for functional autoencoder clustering
from FAEclust import (
    smoothing_features,
    bspline_basis,
    rescale,
    TimeSeriesDistance,
    NearestNeighborsOpt,
    FunctionalAutoencoder,
)


## ------------------------------------------------------------------------- ##
# Start preprocessing timing
start = time.time()
## --------------------------------------------------------------------- ##
# 1. Load and preprocess dataset
dataset_name = 'Plane'
X, y = load_classification(dataset_name)

# Standardise data for network stability and extract labels as int32 
X, y = rescale(np.array(X), y, dataset_name)
# ground truth labels (y) are only used for evaluating AMI and ARI 

## --------------------------------------------------------------------- ##
# 2. Compute pairwise distance matrix using FastDTW
n_samples, n_features, n_timesteps = X.shape
tsd = TimeSeriesDistance(X, metric='fastdtw', n_jobs=-1)
D = tsd.compute_distances()

## --------------------------------------------------------------------- ##
# 3. Build m-nearest-neighbor graph and similarity matrix
# Estimate optimal m (number of neighbors) by average distance
opt = NearestNeighborsOpt(D)
m = opt.estimate_optimal_m(method='avg_distance', max_m=X.shape[0]-1)
print(f"Optimal m (avg_distance): {m}")

# Retrieve neighbor indices and compute similarity
neighbors_dict = opt.get_nearest_neighbors(opt_m= m)
sim_matrix = opt.compute_similarity(neighbors_dict)

## --------------------------------------------------------------------- ##
# 4. Smooth and extract functional features via B-splines
m_basis = 50                    # Number of smoothing basis functions
dis_p = 300                     # Number of discretization points (grid)
t = np.linspace(0, 1, dis_p)    # Generate time grid
# Compute coefficients and smoothed curves
coeffs, curves, basis_smoothing = smoothing_features(
                                        X, m=m_basis, dis_p=dis_p, fit='bspline'
                                        )

# Report preprocessing time
end = time.time()
print(f'{dataset_name}: pre-processing Time = {end - start:.2f} s')

## --------------------------------------------------------------------- ##
# 5. Initialize and train Functional Autoencoder (FAE)
p = n_features                          # Input dimensionality
# layers [functional, ... MLP ..., functional, functional, functional]
layers = [32, 16, 8, 16, 32, 32, 32]

# Prepare B-spline basis for FAE model functional weights
l_basis = 50                        # Number of functional weight basis functions
basis_input = bspline_basis(num_basis=l_basis)

# Regularization hyperparameters
lambda_e, lambda_d, lambda_c = 0.5, 0.05, 0.5
# Training settings
epochs = 100
learning_rate = 1e-3
batch_size = 16
beta = 0.9

# Create FAE instance with similarity constraints
FAE_model = FunctionalAutoencoder(
                                  p, layers, l=l_basis, m=m_basis,
                                  basis_smoothing=basis_smoothing,
                                  basis_input=basis_input,
                                  lambda_e=lambda_e, lambda_d=lambda_d, lambda_c=lambda_c,
                                  t=t, sim_matrix=sim_matrix
                                  )

# Display model architecture summary
FAE_model.model_summary()

# Train the model and measure training time
start = time.time()
FAE_model.train(
                coeffs,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                neighbors_dict=neighbors_dict,
                sim_matrix=sim_matrix,
                beta=beta
                )
end = time.time()
print(f'{dataset_name}: network training Time = {end - start:.2f} s')

## --------------------------------------------------------------------- ##
# 6. Cluster assignments and evaluate metrics
S, labels = FAE_model.predict(coeffs, batch_size=batch_size)
# S is the latent representation of each sample (number of samples, latent space dimension)
ami = adjusted_mutual_info_score(y, labels)
ari = adjusted_rand_score(y, labels)

print(f'AMI score: {ami:.4f}')
print(f'ARI score: {ari:.4f}')
print(f'Number of clusters predicted: {len(np.unique(labels))}')

## ------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------- ##