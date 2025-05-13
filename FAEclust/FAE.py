import tensorflow as tf
from .fista import ConvexClustering   # Convex clustering algorithm (from custom module)
import numpy as np


## ------------------------------------------------------------------------- ##
class FunctionalAutoencoder:
    """
    A joint functional-autoencoder with clustering in the latent space.

    Parameters
    ----------
    p : int
        Dimensionality of the input (e.g., number of spatial locations/features).
    layers : list of int
        Architecture specification:
            [q1, q2, ..., s, ..., Q2, Q1, Z1, Z2]
        - q1: number of functional weights in encoder layer
        - q2,...,s: sizes of successive dense encoder layers
        - ...,Q2,Q1: sizes of successive dense decoder layers
        - Z1,Z2: number of functional weights in decoder layers
    l : int
        Number of basis functions for encoder expansion.
    m : int
        Number of basis functions for smoothing expansion.
    basis_smoothing : list of callables representing functions
        Each callable maps time grid t to a smoothing basis vector of length T.
    basis_input : list of callables representing functions
        Each callable maps time grid t to an input basis vector of length T.
    lambda_e : float
        Weight of L2 orthonormality penalty on encoder filters.
    lambda_d : float
        Weight of L1 sparsity penalty on decoder filters.
    lambda_c : float
        Weight of clustering loss in latent space.
    t : array_like, shape (T,)
        Time grid over which functional data are defined.
    sim_matrix : array_like, shape (N, N)
        Pairwise similarity matrix for convex clustering.
    """
    def __init__(
        self,
        p,
        layers,
        l,
        m,
        basis_smoothing,
        basis_input,
        lambda_e,
        lambda_d,
        lambda_c,
        t,
        sim_matrix
    ):
        
        # Store basic properties
        self.p = p              # input FD dimensions
        self.layers = layers
        # Extract sizes for first and final functional dims
        self.q1 = layers[0]
        self.Z1 = layers[-2]
        self.Z2 = layers[-1]

        # Extract dense (MLP) layer sizes, excluding first and last three functional dims
        dense_dims = layers[1:-2]  # includes q2,...,s,...,Q2
        # Split into encoder vs. decoder halves
        N = len(dense_dims)
        mid = N // 2
        # encoder dense dims: q2,...,s
        self.encoder_dense_dims = dense_dims[:mid] 
        # decoder dense dims: ...,Q2
        self.decoder_dense_dims = dense_dims[mid:]

        # Other hyperparameters
        self.l = l          # input basis count
        self.m = m          # smoothing basis count
        self.s = self.encoder_dense_dims[-1]  # final encoder latent size
        self.lambda_e = lambda_e
        self.lambda_d = lambda_d
        self.lambda_c = lambda_c
        self.basis_smoothing = basis_smoothing
        self.basis_input = basis_input
        self.t = t          # time grid
        # Similarity matrix tensor for clustering
        self.W = tf.convert_to_tensor(sim_matrix)

        # Precompute basis matrices:
        # Phi_s: [m, T], smoothing bases
        # Phi_i: [l, T], input bases
        self.Phi_s = tf.cast(tf.stack([b(self.t) for b in basis_smoothing], axis=0), tf.float32)  
        self.Phi_i = tf.cast(tf.stack([g(self.t) for g in basis_input], axis=0), tf.float32)
        
        # Gram matrices A and B for functional operations
        T = tf.cast(tf.shape(self.Phi_s)[1], tf.float32)
        self.A = tf.matmul(self.Phi_s, self.Phi_i, transpose_b=True) / T  # [m, l]
        self.B = tf.matmul(self.Phi_i, self.Phi_i, transpose_b=True) / T  # [l, l]

        # --- Functional encoder parameters ---
        self.wfn = tf.Variable(
            tf.random.uniform(
                [self.q1, self.p, self.l],
                minval=-np.sqrt(6 / (self.q1 + self.p)),
                maxval=np.sqrt(6 / (self.q1 + self.p))
            ),
            dtype=tf.float32,
            name="wfn"
        )
        self.bfn = tf.Variable(tf.zeros([self.q1]), name="bfn")

        # --- Dense encoder (MLP) parameters ---
        self.W_dense_enc = []
        self.b_dense_enc = []
        prev_dim = self.q1
        count = 0
        for dim in self.encoder_dense_dims:
            count += 1
            W = self.glorot_uniform_init([prev_dim, dim], name=f"W_enc_{count}")
            b = tf.Variable(tf.zeros([dim]), name=f"b_enc_{count}")
            self.W_dense_enc.append(W)
            self.b_dense_enc.append(b)
            prev_dim = dim

        # --- Dense decoder (MLP) parameters ---
        self.W_dense_dec = []
        self.b_dense_dec = []
        prev_dim = self.encoder_dense_dims[-1]  # s
        for dim in self.decoder_dense_dims:
            W = self.glorot_uniform_init([prev_dim, dim], name=f"W_dec_{count}")
            b = tf.Variable(tf.zeros([dim]), name=f"b_dec_{count}")
            self.W_dense_dec.append(W)
            self.b_dense_dec.append(b)
            prev_dim = dim
            count -= 1
        self.Q1 = prev_dim

        # --- Functional decoder parameters ---
        # Three functional layers with learnable weights and biases
        self.Wfn1 = tf.Variable(
            tf.random.uniform(
                [self.Q1, self.Z1, self.m],
                minval=-np.sqrt(6 / (self.Q1 + self.Z1)),
                maxval=np.sqrt(6 / (self.Q1 + self.Z1))
            ),
            dtype=tf.float32,
            name="Wfn1"
        )
        self.Bfn1 = tf.Variable(tf.zeros([self.Z1, self.m]), name="Bfn1")
        self.Wfn2 = tf.Variable(
            tf.random.uniform(
                [self.Z1, self.Z2, self.m],
                minval=-np.sqrt(6 / (self.Z1 + self.Z2)),
                maxval=np.sqrt(6 / (self.Z1 + self.Z2))
            ),
            dtype=tf.float32,
            name="Wfn2"
        )
        self.Bfn2 = tf.Variable(tf.zeros([self.Z2]), name="Bfn2")
        self.Wfn3 = tf.Variable(
            tf.random.uniform(
                [self.Z2, self.p, self.m],
                minval=-np.sqrt(6 / (self.Z2 + self.p)),
                maxval=np.sqrt(6 / (self.Z2 + self.p))
            ),
            dtype=tf.float32,
            name="Wfn3"
        )

        # Collect trainable variables
        self.encoder_vars = [self.wfn, self.bfn] + self.W_dense_enc + self.b_dense_enc
        self.decoder_vars = self.W_dense_dec + self.b_dense_dec + [
            self.Wfn1, self.Bfn1, self.Wfn2, self.Bfn2, self.Wfn3
        ]

    def glorot_uniform_init(self, shape, name):
        """
        Glorot uniform initializer for a weight matrix.

        Parameters
        ----------
        shape : tuple (fan_in, fan_out)
            Dimensions of the weight matrix.
        name : str
            TensorFlow variable name.

        Returns
        -------
        tf.Variable
            Initialized with uniform(-limit, +limit), where limit = sqrt(6/(fan_in+fan_out)).
        """
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return tf.Variable(
            tf.random.uniform(shape, minval=-limit, maxval=limit),
            name=name,
            dtype=tf.float32
        )

    @tf.function
    def encoder(self, x):
        """
        Encode input FD represented as coefficients into a latent vector.

        Parameters
        ----------
        x : tf.Tensor, shape [batch, p, l]
            Input coefficients in the input basis.

        Returns
        -------
        tf.Tensor, shape [batch, s]
            Latent representation after functional + MLP encoding.
        """
        # Functional filter application and basis contraction:
        #   x1[b,q] = sum_{d,k,l} wfn[q,d,l] * A[k,l] * x[b,d,k]
        x1 = tf.einsum('q d l, k l, b d k -> b q', self.wfn, self.A, x) + self.bfn
        h = tf.nn.relu(x1)
        # Pass through each dense encoder layer
        for W, b in zip(self.W_dense_enc, self.b_dense_enc):
            h = tf.nn.elu(tf.matmul(h, W) + b)
        return h

    @tf.function
    def decoder(self, h):
        """
        Decode latent vector back to functional output.

        Parameters
        ----------
        h : tf.Tensor, shape [batch, s]
            Latent codes from encoder.

        Returns
        -------
        tf.Tensor, shape [batch, p, T]
            Reconstructed functional signals over time grid.
        """
        # Dense decoder MLP 
        x = h
        for W, b in zip(self.W_dense_dec, self.b_dense_dec):
            x = tf.nn.elu(tf.matmul(x, W) + b)
        h_dec = x       # shape [batch, Q1]
        
        # Functional decoding: three-layer tensor contractions
        X1 = tf.einsum('b i, i d k -> b d k', h_dec, self.Wfn1) + self.Bfn1
        Y1 = tf.nn.elu(tf.matmul(X1, self.Phi_s))
        w1 = tf.matmul(self.Wfn2, self.Phi_s)
        Y2_int = tf.einsum('b d k, d j k -> b j k', Y1, w1) + tf.reshape(self.Bfn2, (1, -1, 1))
        Y2_int = tf.nn.elu(Y2_int)
        w2 = tf.matmul(self.Wfn3, self.Phi_s)
        Y2 = tf.einsum('b d k, d j k -> b j k', Y2_int, w2)
        return Y2
    
    @tf.function
    def forward(self, x):
        """One-step encode-decode pass."""
        return self.decoder(self.encoder(x))
    
    @tf.function
    def loss_r(self, y_coeff, reconstruction):
        """
        Reconstruction MSE loss.

        Parameters
        ----------
        y_coeff : tf.Tensor, shape [batch, p, l]
            True input coefficients.
        reconstruction : tf.Tensor, shape [batch, p, T]
            Decoder output over time grid.

        Returns
        -------
        tf.Tensor
            Scalar MSE * dt.
        """
        # y_coeff: [batch, p, m], reconstruction: [batch, p, T]
        y = tf.matmul(y_coeff, self.Phi_s)       # [batch, p, T]
        dt = self.t[1] - self.t[0]
        # sum over all p and T, then multiply dt
        return dt * tf.reduce_sum((y - reconstruction) ** 2)
    
    @tf.function
    def loss_l2(self):
        """
        L2 orthonormality penalty on encoder filters.
        """
        # Compute Gram matrix C[j,k] = sum_{d,r,s} wfn[j,d,r] * B[r,s] * wfn[k,d,s]
        C = tf.einsum('jdr,rs,kds->jk', self.wfn, self.B, self.wfn)  # now shape [q1, q1]
        q1 = tf.cast(self.q1, tf.float32)
        # off‐diagonal penalty
        mask = tf.eye(self.q1)
        off = tf.reduce_sum((C * (1. - mask))**2) / (q1 * (q1 - 1) / 2)
        # diagonal penalty
        diag = tf.linalg.tensor_diag_part(C)
        on  = tf.reduce_sum((diag - 1.)**2) / q1
        return off + on
    
    @tf.function
    def loss_l1(self):
        """
        L1 sparsity penalty on all functional decoder weights & biases.
        """
        L1 = (
            tf.reduce_sum(tf.abs(self.Wfn1)) / (self.Q1 * self.Z1 * self.m)
            + tf.reduce_sum(tf.abs(self.Wfn2)) / (self.Z1 * self.Z2 * self.m)
            + tf.reduce_sum(tf.abs(self.Wfn3)) / (self.Z2 * self.p * self.m)
            + tf.reduce_sum(tf.abs(self.Bfn1)) / (self.Z1 * self.m)
            + tf.reduce_sum(tf.abs(self.Bfn2)) / self.Z2
        )
        return L1
    
    @tf.function
    def clustering_loss(self, X_latent, cluster_labels):
        """
        Compute clustering validation L_c over latent codes.

        L_c = (2*within-cluster SSE - total SSE) / N

        Parameters
        ----------
        X_latent : tf.Tensor, shape [N, latent_dim]
            Latent codes for all samples.
        cluster_labels : tf.Tensor, shape [N], dtype int32
            Discrete cluster assignment per sample.

        Returns
        -------
        tf.Tensor
            Scalar clustering loss.
        """
        # Number of data samples
        N = tf.shape(X_latent)[0]
        N = tf.cast(N, tf.float32)
        # Determine number of clusters (assumes cluster_labels are integer IDs 0,...,K-1)
        num_clusters = tf.reduce_max(cluster_labels) + 1
        
        # Compute cluster means u_k for each cluster k
        cluster_means = tf.math.unsorted_segment_mean(
            X_latent, cluster_labels, num_segments=num_clusters)
        
        # Compute global mean of all latent points
        global_mean = tf.reduce_mean(X_latent, axis=0)
        
        # Compute sum of squared distances of points to their cluster mean (within-cluster dispersion)
        # For each sample i, get its cluster mean u_{c_i}:
        means_per_point = tf.gather(cluster_means, cluster_labels)    # shape [N, latent_dim]
        diff_to_cluster = X_latent - means_per_point                  # differences x_i - u_{c_i}
        sq_dist_to_cluster = tf.reduce_sum(tf.square(diff_to_cluster), axis=1)  # ||x_i - u_ci||^2 for each i
        within_cluster_sse = tf.reduce_sum(sq_dist_to_cluster)        #  sum_k sum_{i in C_k} ||x_i - u_k||^2
        
        # Compute sum of squared distances of points to global mean (total dispersion)
        diff_to_global = X_latent - global_mean                       # difference x_i - x_bar
        sq_dist_to_global = tf.reduce_sum(tf.square(diff_to_global), axis=1)  # ||x_i - x_bar||^2 for each i
        total_sse = tf.reduce_sum(sq_dist_to_global)                  #  sum_i ||x_i - x_bar||^2
        
        # Clustering loss: (2 * within_cluster_SSE - total_SSE) / N
        L_c = (2.0 * within_cluster_sse - total_sse) / N
        return L_c
    
    @tf.function
    def loss_t(self, y_coeff, reconstruction):
        """
        Total autoencoder loss: reconstruction + L2 + L1 penalties.
        """
        return (
            self.loss_r(y_coeff, reconstruction)
            + self.lambda_e * self.loss_l2()
            + self.lambda_d * self.loss_l1()
        )

    def train(self, X_train, epochs, learning_rate, batch_size, neighbors_dict, sim_matrix):
        """
        Train the Functional Autoencoder.

        Alternates mini-batch reconstruction updates
        with a global convex clustering step each epoch.

        Parameters
        ----------
        X_train : array_like or tf.Tensor, shape [N, p, l]
            Training input coefficients.
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for Adam optimizer.
        batch_size : int
            Mini-batch size.
        neighbors_dict : dict
            Adjacency for convex clustering.
        sim_matrix : array_like, shape [N,N]
            Similarity matrix for convex clustering.
        """
        num_samples = X_train.shape[0]
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.epoch_loss = []
        self.neighbors_dict = neighbors_dict
        self.sim_matrix = sim_matrix
        
        for epoch in range(epochs):
            # Shuffle data indices for mini-batch sampling
            indices = np.random.permutation(num_samples)
            # Placeholder to store latent codes in original order
            X_reduced_full = np.empty((num_samples, self.s), dtype=np.float32)
            
            # Mini-batch training loop
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]                       # original indices for this batch
                x_batch = X_train[batch_idx]                         # fetch batch data using shuffled indices

                with tf.GradientTape() as tape:
                    # Forward pass: encode and decode        
                    latent = self.encoder(x_batch)                   # latent representation for batch
                    recon  = self.decoder(latent)                    # reconstruction of x_batch
                    # Total loss for this batch (excluding clustering loss, which is global)
                    total_loss = self.loss_t(x_batch, recon)    
                
                # Backprop reconstruction + regularization losses
                grads = tape.gradient(total_loss, self.encoder_vars + self.decoder_vars)
                optimizer.apply_gradients(zip(grads, self.encoder_vars + self.decoder_vars))
                
                # Store this batch's latent vectors in the full array at original indices
                X_reduced_full[batch_idx] = latent.numpy()
            
            # --- End of mini-batch loop for this epoch ---
            
            # Compute clustering on the full latent space (in original sample order):
            # Use the encoder outputs collected in X_reduced_full (aligned to original indices)
            cluster = ConvexClustering(X_reduced_full, self.neighbors_dict, self.sim_matrix)
            cluster_labels = cluster.fit()
            # Convert cluster labels to a Tensor for loss computation
            cluster_labels_tf = tf.constant(cluster_labels, dtype=tf.int32)
            
            # Compute clustering loss and backprop only through encoder_vars
            with tf.GradientTape() as tape:
                X_latent_full = self.encoder(X_train)
                L_c = self.clustering_loss(X_latent_full, cluster_labels_tf)
                total_cluster_loss = self.lambda_c * L_c
            
            # Backpropagate the clustering loss to update encoder weights
            cluster_grads = tape.gradient(total_cluster_loss, self.encoder_vars)
            optimizer.apply_gradients(zip(cluster_grads, self.encoder_vars))
            
            # Logging the losses for monitoring
            epoch_recon_loss = total_loss.numpy()  # last batch recon loss (or compute average over batches separately)
            # epoch_cluster_loss = L_c.numpy()
            self.epoch_loss.append(epoch_recon_loss)
            # print(f"Epoch {epoch+1}: Loss = {epoch_recon_loss:.4f}")
    
    def predict(self, coeffs, batch_size = None):
        """
        Encode new data, then cluster latent codes.

        Parameters
        ----------
        coeffs : array_like or tf.Tensor, shape [N, p, l]
            Input coefficients to encode.
        batch_size : int or None
            Batch size for encoding. Defaults to all samples at once.

        Returns
        -------
        latents : np.ndarray, shape [N, s]
            Latent representations.
        labels : list of int, length N
            Cluster labels assigned by convex clustering.
        """
        if batch_size == None:
            ds = tf.data.Dataset.from_tensor_slices(coeffs).batch(coeffs.shape[0])
        else:
            ds = tf.data.Dataset.from_tensor_slices(coeffs).batch(batch_size)
            
        latents = []

        for x_batch in ds:
            z = self.encoder(x_batch)  # [batch, s]
            latents.append(z)
        
        # Cluster the latent outputs
        latents = tf.concat(latents, axis=0)
        cluster = ConvexClustering(latents, self.neighbors_dict, self.sim_matrix)
        labels = cluster.fit()
        return latents.numpy(), labels
    
    def model_summary(self):
        """
        Print a summary of all trainable parameters.

        Displays layer names, shapes, and parameter counts.
        """
        print("=" * 70)
        print("Functional Autoencoder:")
        print(f"{'Layer (type)':<25} {'Shape':<30} {'Param #':<10}")
        print("=" * 70)
        total_params = 0
        # functional encoder first layer
        for var in (self.wfn, self.bfn):
            name = var.name.split(':')[0]
            shape = var.shape
            params = np.prod(shape)
            print(f"{name:<25} {str(shape):<30} {int(params):<10}")
            total_params += params
        # encoder MLP layers 
        for W, b in zip(self.W_dense_enc, self.b_dense_enc):
            for var in (W, b):
                name = var.name.split(':')[0]
                shape = var.shape
                params = np.prod(shape)
                print(f"{name:<25} {str(shape):<30} {int(params):<10}")
                total_params += params
        # decoder MLP layers
        for W, b in zip(self.W_dense_dec, self.b_dense_dec):
            for var in (W, b):
                name = var.name.split(':')[0]
                shape = var.shape
                params = np.prod(shape)
                print(f"{name:<25} {str(shape):<30} {int(params):<10}")
                total_params += params
        # functional decoder layers
        for var in (self.Wfn1, self.Bfn1, self.Wfn2, self.Bfn2, self.Wfn3):
            name = var.name.split(':')[0]
            shape = var.shape
            params = np.prod(shape)
            print(f"{name:<25} {str(shape):<30} {int(params):<10}")
            total_params += params
        print("=" * 70)
        print(f"Total trainable parameters: {int(total_params)}")
        print("=" * 70)
   
## ------------------------------------------------------------------------- ##
