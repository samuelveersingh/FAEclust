"""TensorFlow/Keras functional autoencoder with joint convex clustering.

Key design points:

* MLP layers run in the order fully-connected -> batch normalization ->
  ELU activation -> dropout (with keep-probability tau).
* The decoder's functional-weight penalty is a 2nd-order roughness penalty:
  the L1 norm of the 2nd difference of the basis-expansion coefficients. This
  discourages wiggly reconstructed functions.
* The clustering loss is normalised by ``n * s`` (number of samples times the
  latent dimension).
* A single backward step over all parameters minimises the combined loss
  ``L = L_r + lambda_w L_w + lambda_c L_c`` (reconstruction + weight penalty +
  clustering).
* Training first pre-trains on the penalized reconstruction loss, then
  fine-tunes with the clustering term added.
* The optimizer is a manual mini-batch SGD-with-momentum loop inside a
  ``tf.GradientTape`` (not a Keras optimizer / model.fit).
* Optional non-trainable manifold readout rho applied to the decoder output.

Implementation note: this uses Keras layers for the MLP blocks (so batch
normalization and dropout come for free) and raw ``tf.Variable`` +
``tf.einsum`` for the functional layers. Neither ``tf.Module`` nor
``keras.layers.Layer`` collects both kinds of variable automatically, so this
is a plain Python class with an explicit ``trainable_variables`` property.
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers as klayers

from .fista import ConvexClustering
from .manifolds import ManifoldReadout


def pick_device(device=None):
    """Resolve a TF device string: explicit -> /GPU:0 if available -> /CPU:0."""
    if device is not None:
        return device if isinstance(device, str) else str(device)
    gpus = tf.config.list_physical_devices('GPU')
    return '/GPU:0' if gpus else '/CPU:0'


# --------------------------------------------------------------------------- #
class _MLP:
    """Dense stack: fully-connected -> batch norm -> ELU -> dropout per block.

    Built from Keras layers but kept as a plain holder (not a Layer/Model) so
    the parent class can assemble the full list of trainable variables by hand.
    ``__call__`` passes the ``training`` flag into every batch-norm and dropout
    layer, since those behave differently during training and evaluation.
    """

    def __init__(self, dims, tau=1.0, use_bn=True, name='mlp'):
        self.blocks = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            dense = klayers.Dense(b, name=f'{name}_fc{i}')
            # epsilon=1e-5 is the small constant added for numerical stability
            # in batch normalization. We keep Keras' default momentum=0.99 for
            # smooth moving statistics: the eval-mode embedding is what the
            # convex clustering sees, and smooth moving averages keep it stable
            # across cluster refreshes (a faster momentum=0.9 made it collapse
            # to a single cluster).
            bn = (klayers.BatchNormalization(epsilon=1e-5, name=f'{name}_bn{i}')
                  if use_bn else None)
            elu = klayers.ELU(name=f'{name}_elu{i}')
            drop = klayers.Dropout(rate=1.0 - tau, name=f'{name}_do{i}') if tau < 1.0 else None
            self.blocks.append((dense, bn, elu, drop))

    def __call__(self, x, training=False):
        for dense, bn, elu, drop in self.blocks:
            x = dense(x)
            if bn is not None:
                x = bn(x, training=training)
            x = elu(x)
            if drop is not None:
                x = drop(x, training=training)
        return x

    @property
    def trainable_variables(self):
        v = []
        for dense, bn, _, _ in self.blocks:
            v += list(dense.trainable_variables)
            if bn is not None:
                v += list(bn.trainable_variables)
        return v

    @property
    def non_trainable_variables(self):
        v = []
        for _, bn, _, _ in self.blocks:
            if bn is not None:
                v += list(bn.non_trainable_variables)
        return v


class FunctionalAutoencoder:
    """Functional autoencoder with a clustering-aware latent space.

    Main parameters: p, layers, l, m, basis_smoothing, basis_input, lambda_e,
    lambda_d, lambda_c, t, sim_matrix, plus:

    tau : float in (0, 1], dropout keep-probability for MLP layers (1.0 = off).
    use_bn : bool, whether to use batch normalization in the MLP layers.
    manifold : {None,'euclidean','sphere','poincare'}, decoder readout rho.
    """

    def __init__(self, p, layers, l, m, basis_smoothing, basis_input,
                 lambda_e, lambda_d, lambda_c, t, sim_matrix,
                 tau=1.0, use_bn=True, manifold=None, seed=0):
        tf.random.set_seed(seed)
        self.seed = seed
        self.p = p
        self.layers = layers
        self.q1 = layers[0]
        self.Z1 = layers[-2]
        self.Z2 = layers[-1]
        dense_dims = layers[1:-2]
        mid = len(dense_dims) // 2
        self.encoder_dense_dims = dense_dims[:mid]
        self.decoder_dense_dims = dense_dims[mid:]
        self.l = l
        self.m = m
        self.s = self.encoder_dense_dims[-1]
        self.lambda_e = lambda_e
        self.lambda_d = lambda_d
        self.lambda_c = lambda_c

        t = np.asarray(t, dtype=np.float32)
        Phi_s = np.stack([np.asarray(b(t), np.float32) for b in basis_smoothing], 0)  # [m,T]
        Phi_i = np.stack([np.asarray(g(t), np.float32) for g in basis_input], 0)      # [l,T]
        T = Phi_s.shape[1]
        # Non-trainable buffers as tf.constant (never enter trainable_variables).
        self.t = tf.constant(t)
        self.Phi_s = tf.constant(Phi_s)
        self.Phi_i = tf.constant(Phi_i)
        self.A = tf.constant(Phi_s @ Phi_i.T / T)   # [m,l]
        self.B = tf.constant(Phi_i @ Phi_i.T / T)   # [l,l]
        self.W = tf.constant(np.asarray(sim_matrix, np.float32))

        # functional encoder weights (Glorot-uniform initialization)
        lim = np.sqrt(6.0 / (self.q1 + self.p))
        self.wfn = tf.Variable(tf.random.uniform([self.q1, self.p, self.l], -lim, lim),
                               name='wfn')
        self.bfn = tf.Variable(tf.zeros([self.q1]), name='bfn')

        # MLP encoder / decoder
        enc_dims = [self.q1] + list(self.encoder_dense_dims)
        self.enc_mlp = _MLP(enc_dims, tau=tau, use_bn=use_bn, name='enc')
        dec_dims = [self.s] + list(self.decoder_dense_dims)
        self.dec_mlp = _MLP(dec_dims, tau=tau, use_bn=use_bn, name='dec')
        self.Q1 = dec_dims[-1]

        # functional decoder (3 functional layers, with functional biases)
        l1 = np.sqrt(6.0 / (self.Q1 + self.Z1))
        self.Wfn1 = tf.Variable(tf.random.uniform([self.Q1, self.Z1, self.m], -l1, l1),
                                name='Wfn1')
        self.Bfn1 = tf.Variable(tf.zeros([self.Z1, self.m]), name='Bfn1')
        l2 = np.sqrt(6.0 / (self.Z1 + self.Z2))
        self.Wfn2 = tf.Variable(tf.random.uniform([self.Z1, self.Z2, self.m], -l2, l2),
                                name='Wfn2')
        self.Bfn2 = tf.Variable(tf.zeros([self.Z2]), name='Bfn2')
        l3 = np.sqrt(6.0 / (self.Z2 + self.p))
        self.Wfn3 = tf.Variable(tf.random.uniform([self.Z2, self.p, self.m], -l3, l3),
                                name='Wfn3')

        self.readout = ManifoldReadout(manifold)
        self.device = None

        # Build the Keras MLP blocks so their variables exist before the first
        # trainable_variables access (Keras layers are lazily built on call).
        self.enc_mlp(tf.zeros([1, self.q1]), training=False)
        self.dec_mlp(tf.zeros([1, self.s]), training=False)

    # -- variable tracking (load-bearing) -----------------------------------
    @property
    def trainable_variables(self):
        return ([self.wfn, self.bfn]
                + self.enc_mlp.trainable_variables
                + self.dec_mlp.trainable_variables
                + [self.Wfn1, self.Bfn1, self.Wfn2, self.Bfn2, self.Wfn3])

    # -- forward -------------------------------------------------------------
    def encoder(self, x, training=False):
        x1 = tf.einsum('qdl,kl,bdk->bq', self.wfn, self.A, x) + self.bfn
        h = tf.nn.relu(x1)
        return self.enc_mlp(h, training=training)

    def decoder(self, h, training=False):
        z = self.dec_mlp(h, training=training)                 # [b, Q1]
        X1 = tf.einsum('bi,idk->bdk', z, self.Wfn1) + self.Bfn1
        Y1 = tf.nn.elu(tf.matmul(X1, self.Phi_s))              # [b,Z1,T]
        w1 = tf.matmul(self.Wfn2, self.Phi_s)
        Y2 = tf.einsum('bdk,djk->bjk', Y1, w1) + tf.reshape(self.Bfn2, (1, -1, 1))
        Y2 = tf.nn.elu(Y2)
        w2 = tf.matmul(self.Wfn3, self.Phi_s)
        out = tf.einsum('bdk,djk->bjk', Y2, w2)                # [b,p,T]
        return self.readout(out)

    def forward(self, x, training=False):
        return self.decoder(self.encoder(x, training=training), training=training)

    __call__ = forward

    # -- losses --------------------------------------------------------------
    def loss_r(self, y_coeff, recon):
        y = tf.matmul(y_coeff, self.Phi_s)                     # [b,p,T]
        return tf.reduce_mean((y - recon) ** 2)

    def loss_orth(self):
        C = tf.einsum('jdr,rs,kds->jk', self.wfn, self.B, self.wfn)
        q1 = float(self.q1)
        mask = tf.eye(self.q1)
        off = tf.reduce_sum((C * (1.0 - mask)) ** 2) / (q1 * (q1 - 1) / 2)
        diag = tf.linalg.diag_part(C)
        on = tf.reduce_sum((diag - 1.0) ** 2) / q1
        return off + on

    @staticmethod
    def _rough(c):
        """L1 norm of the 2nd difference along the basis-coefficient axis.

        This is a roughness penalty: it is large when neighbouring coefficients
        change abruptly, and small when they vary smoothly.
        """
        d2 = c[..., :-2] - 2.0 * c[..., 1:-1] + c[..., 2:]
        return tf.reduce_mean(tf.abs(d2))

    def loss_rough(self):
        return (self._rough(self.Wfn1) + self._rough(self.Wfn2)
                + self._rough(self.Wfn3) + self._rough(self.Bfn1)
                + tf.reduce_mean(tf.abs(self.Bfn2)))

    def clustering_loss(self, X, labels):
        N = tf.cast(tf.shape(X)[0], tf.float32)
        s = X.shape[1]
        labels = tf.cast(labels, tf.int32)
        K = int(tf.reduce_max(labels).numpy()) + 1
        means = tf.math.unsorted_segment_mean(X, labels, num_segments=K)
        within = tf.reduce_sum((X - tf.gather(means, labels)) ** 2)
        total = tf.reduce_sum((X - tf.reduce_mean(X, axis=0)) ** 2)
        return (2.0 * within - total) / (N * s)                # divide by n*s

    @staticmethod
    def clustering_loss_cached(Xb, labels_b, means, global_mean):
        """Clustering loss on a minibatch.

        Uses the cluster centroids (``means``) and the global mean cached from
        the last full clustering pass. These are treated as constants while
        optimising the network parameters, so they are passed in already
        wrapped in tf.stop_gradient.
        """
        B = tf.cast(tf.shape(Xb)[0], tf.float32)
        s = Xb.shape[1]
        within = tf.reduce_sum((Xb - tf.gather(means, labels_b)) ** 2)
        total = tf.reduce_sum((Xb - global_mean) ** 2)
        return (2.0 * within - total) / (B * s)

    def penalized_recon(self, y_coeff, recon):
        return (self.loss_r(y_coeff, recon)
                + self.lambda_e * self.loss_orth()
                + self.lambda_d * self.loss_rough())

    # -- training ------------------------------------------------------------
    def _make_steps(self, params, mom, lr, beta):
        """Build the per-minibatch SGD-with-momentum update functions.

        Returns two functions: ``pretrain_step`` (reconstruction loss only) and
        ``finetune_step`` (reconstruction plus clustering). They are split so
        each can be traced once by ``tf.function``. ``mom`` is a list of
        non-trainable tf.Variables holding the momentum state, which must
        persist across calls; ``lr`` and ``beta`` are baked in as constants.
        """
        def _update(grads):
            for i in range(len(params)):
                g = grads[i] if grads[i] is not None else tf.zeros_like(params[i])
                mom[i].assign(beta * mom[i] + (1.0 - beta) * g)
                params[i].assign_sub(lr * mom[i])

        def pretrain_step(xb):
            with tf.GradientTape() as tape:
                zb = self.encoder(xb, training=True)
                recon = self.decoder(zb, training=True)
                loss = self.penalized_recon(xb, recon)
            _update(tape.gradient(loss, params))
            return loss

        def finetune_step(xb, lbl_b, means, global_mean):
            with tf.GradientTape() as tape:
                zb = self.encoder(xb, training=True)
                recon = self.decoder(zb, training=True)
                loss = self.penalized_recon(xb, recon) + self.lambda_c * \
                    self.clustering_loss_cached(zb, lbl_b, means, global_mean)
            _update(tape.gradient(loss, params))
            return loss

        return pretrain_step, finetune_step

    def train_model(self, X_train, epochs, learning_rate, batch_size,
                     neighbors_dict, sim_matrix, beta=0.9,
                     pretrain_epochs=0, criterion='silhouette', verbose=False,
                     device=None, cluster_every=5, jit=True):
        """Train the model with manual mini-batch SGD-with-momentum.

        The first ``pretrain_epochs`` epochs minimise the penalized
        reconstruction loss only; the remaining epochs add the clustering term
        and take a single backward step over all parameters.

        The expensive convex-clustering step plus a full-data encode runs once
        every ``cluster_every`` epochs. In between, the clustering loss is
        evaluated per minibatch against the cached, detached cluster centroids
        and global mean. Minibatch steps run with training=True (so batch norm
        uses batch statistics and updates its moving averages, and dropout is
        active); the cluster-refresh encode runs with training=False.

        jit : bool, default True
            Wrap the per-minibatch train step in ``tf.function`` (graph mode).
            This removes the per-op Python overhead of the many small
            einsum/matmul ops and the momentum update, making training
            noticeably faster. A fixed input signature (with the batch and
            cluster-count axes left dynamic) means each step is traced exactly
            once. Set ``jit=False`` to keep a pure-eager loop (useful for
            debugging).
        """
        dev = pick_device(device)
        self.device = dev
        self.neighbors_dict = neighbors_dict
        self.sim_matrix = np.asarray(sim_matrix)
        self.epoch_loss = []
        lr = float(learning_rate)
        beta = float(beta)

        with tf.device(dev):
            Xt = tf.constant(np.asarray(X_train), dtype=tf.float32)
            n = Xt.shape[0]
            params = self.trainable_variables
            # Momentum buffers must be Variables so their state survives across
            # tf.function calls (rebinding a plain tensor would not persist).
            mom = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]

            # Deterministic minibatch shuffle keyed to the model seed, so a run
            # is fully reproducible from `seed` alone (the global np.random state
            # left by earlier cells / data generation no longer affects it).
            rng = np.random.default_rng(self.seed)

            pretrain_step, finetune_step = self._make_steps(params, mom, lr, beta)
            if jit:
                sx = tf.TensorSpec([None, self.p, self.m], tf.float32)
                sl = tf.TensorSpec([None], tf.int32)
                sm = tf.TensorSpec([None, self.s], tf.float32)
                sg = tf.TensorSpec([self.s], tf.float32)
                pretrain_step = tf.function(pretrain_step, input_signature=[sx])
                finetune_step = tf.function(finetune_step,
                                            input_signature=[sx, sl, sm, sg])

            labels_t = means = global_mean = None
            for epoch in range(epochs):
                do_cluster = epoch >= pretrain_epochs and self.lambda_c != 0
                # Refresh cluster assignment + cached centroids periodically.
                if do_cluster and (labels_t is None
                                   or (epoch - pretrain_epochs) % cluster_every == 0):
                    Z = self.encoder(Xt, training=False)
                    Znp = Z.numpy()
                    cc = ConvexClustering(Znp, self.neighbors_dict,
                                          self.sim_matrix, criterion=criterion)
                    labels = np.asarray(cc.fit())
                    labels_t = tf.constant(labels, dtype=tf.int32)
                    K = int(labels.max()) + 1
                    means = tf.stop_gradient(
                        tf.math.unsorted_segment_mean(Z, labels_t, num_segments=K))
                    global_mean = tf.stop_gradient(tf.reduce_mean(Z, axis=0))

                perm = rng.permutation(n)
                running = 0.0
                n_batches = 0
                for s0 in range(0, n, batch_size):
                    idx = perm[s0:s0 + batch_size]
                    xb = tf.gather(Xt, idx)
                    if do_cluster:
                        lbl_b = tf.gather(labels_t, idx)
                        loss = finetune_step(xb, lbl_b, means, global_mean)
                    else:
                        loss = pretrain_step(xb)
                    running += float(loss.numpy())
                    n_batches += 1
                self.epoch_loss.append(running / max(1, n_batches))
                if verbose:
                    tag = "pretrain" if not do_cluster else "finetune"
                    print(f"epoch {epoch+1}/{epochs} [{tag}] "
                          f"loss={self.epoch_loss[-1]:.4f}")

    def predict(self, coeffs, batch_size=None, criterion='silhouette'):
        dev = getattr(self, 'device', None) or pick_device(None)
        with tf.device(dev):
            Xt = tf.constant(np.asarray(coeffs), dtype=tf.float32)
            if batch_size is None:
                Z = self.encoder(Xt, training=False)
            else:
                Z = tf.concat([self.encoder(Xt[i:i + batch_size], training=False)
                               for i in range(0, Xt.shape[0], batch_size)], axis=0)
        Z = Z.numpy()
        cc = ConvexClustering(Z, self.neighbors_dict, self.sim_matrix,
                              criterion=criterion)
        labels = cc.fit()
        return Z, labels

    def model_summary(self):
        params = self.trainable_variables
        tot = int(sum(np.prod(v.shape) for v in params))
        print("=" * 60)
        print("FunctionalAutoencoder (TensorFlow)")
        for v in params:
            name = getattr(v, 'path', None) or v.name
            print(f"{name:<28} {tuple(v.shape)!s:<22} {int(np.prod(v.shape))}")
        print("=" * 60)
        print(f"Total trainable parameters: {tot}")

## ------------------------------------------------------------------------- ##
