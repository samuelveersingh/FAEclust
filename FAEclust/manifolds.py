import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.lines as mlines
import pandas as pd


## ------------------------------------------------------------------------- ##
class DatasetGenerator:
    """
    Generate, save, load, and plot synthetic time-series datasets,
    supporting multiple cluster types (hypersphere, hyperbolic, etc.).

    Parameters
    ----------
    n_samples : int
        Number of time-series samples to generate.
    n_features : int
        Dimensionality of each sample at each time step.
    n_steps : int
        Length of each time-series (number of time steps).
    n_clusters : int, default=2
        Number of distinct clusters or underlying dynamics.
    base_noise : float, default=0.0
        Base noise level for cluster-dependent perturbations.
    omega_delta : float, default=2.0
        Angular velocity separation multiplier for pendulum dynamics.
    """
    def __init__(self, n_samples, n_features, n_steps, 
                 n_clusters=2, base_noise=0.0, omega_delta=2.0):
        # Save initialization parameters
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_steps = n_steps
        self.n_clusters = n_clusters
        self.base_noise = base_noise
        self.omega_delta = omega_delta

    def _cluster_idx(self, i):
        """
        Map sample index i to a cluster label in [0, n_clusters-1].
        Equally partition indices across clusters.
        """
        idx = int(i * self.n_clusters / self.n_samples)
        # Ensure max index does not exceed n_clusters-1
        return min(idx, self.n_clusters - 1)

    def generate_hypersphere(self):
        """
        Generate samples moving on unit hyperspheres with cluster-specific angular speeds.

        Returns
        -------
        X : np.ndarray, shape=(n_samples, n_features, n_steps)
            Generated trajectories on the hypersphere.
        y : np.ndarray, shape=(n_samples,)
            Integer cluster labels.
        """
        X = np.zeros((self.n_samples, self.n_features, self.n_steps))
        y = np.zeros(self.n_samples, dtype=int)
        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            # random orthonormal frame
            a = np.random.randn(self.n_features)
            a /= np.linalg.norm(a)
            temp = np.random.randn(self.n_features)
            b = temp - a * np.dot(a, temp)
            b /= np.linalg.norm(b)
            # cluster-specific angular velocity and noise
            omega = (k + 1) * 2 * np.pi / self.n_steps
            noise = self.base_noise * k
            for t in range(self.n_steps):
                theta = omega * t
                point = np.cos(theta) * a + np.sin(theta) * b
                if noise > 0:
                    point += noise * np.random.randn(self.n_features)
                    point /= np.linalg.norm(point)
                X[i, :, t] = point
        return X, y

    def generate_hyperbolic(self):
        """
        Generate samples moving in the Poincaré ball model of hyperbolic space.

        Returns
        -------
        X, y as in generate_hypersphere.
        """
        X = np.zeros((self.n_samples, self.n_features, self.n_steps))
        y = np.zeros(self.n_samples, dtype=int)
        # linearly spaced time horizon per cluster
        T_vals = np.linspace(1.0, 2.0, self.n_clusters)
        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            T = T_vals[k]
            # random direction
            v = np.random.randn(self.n_features)
            v /= np.linalg.norm(v)
            times = np.linspace(0, T, self.n_steps)
            for t_idx, t_val in enumerate(times):
                cosh_t = np.cosh(t_val)
                sinh_t = np.sinh(t_val)
                X0 = cosh_t
                Xspatial = sinh_t * v
                point = Xspatial / (1.0 + X0)
                # ensure point inside unit ball
                norm_pt = np.linalg.norm(point)
                if norm_pt >= 1.0:
                    point *= 0.999 / norm_pt
                X[i, :, t_idx] = point
        return X, y

    def generate_swiss_roll(self):
        """
        Generate trajectories on a 3D Swiss-roll manifold, varying height by cluster.
        """
        X = np.zeros((self.n_samples, self.n_features, self.n_steps))
        y = np.zeros(self.n_samples, dtype=int)
        # height bins across [0, 10]
        h_bins = np.linspace(0.0, 10.0, self.n_clusters + 1)
        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            h_low, h_high = h_bins[k], h_bins[k + 1]
            h = np.random.uniform(h_low, h_high)
            t0 = np.random.uniform(1.5 * np.pi, 3.0 * np.pi)
            t1 = t0 + 1.5 * np.pi
            t_vals = np.linspace(t0, t1, self.n_steps)
            for t_idx, t in enumerate(t_vals):
                x_coord = t * np.cos(t)
                z_coord = t * np.sin(t)
                y_coord = h
                point = np.array([x_coord, y_coord, z_coord])
                if self.n_features < 3:
                    X[i, :, t_idx] = point[: self.n_features]
                else:
                    padded = np.pad(point, (0, max(0, self.n_features - 3)), constant_values=0)
                    X[i, :, t_idx] = padded[: self.n_features]
        return X, y

    def generate_lorenz(self):
        """
        Generate Lorenz attractor trajectories with cluster-specific rho parameter.
        """
        X = np.zeros((self.n_samples, self.n_features, self.n_steps))
        y = np.zeros(self.n_samples, dtype=int)
        sigma = 10.0
        beta = 8.0 / 3.0
        dt = 0.01
        # linearly spaced rho values
        rho_vals = np.linspace(14.0, 28.0, self.n_clusters)
        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            rho = rho_vals[k]
            state = np.random.uniform(-10, 10, size=3)
            for t in range(self.n_steps):
                # record
                if self.n_features < 3:
                    X[i, :, t] = state[: self.n_features]
                else:
                    padded = np.pad(state, (0, max(0, self.n_features - 3)), constant_values=0)
                    X[i, :, t] = padded[: self.n_features]
                x_s, y_s, z = state
                dx = sigma * (y_s - x_s)
                dy = x_s * (rho - z) - y_s
                dz = x_s * y_s - beta * z
                state = state + dt * np.array([dx, dy, dz])
        return X, y

    def generate_pendulum(self):
        """
        Generate simple pendulum dynamics with cluster-specific angular velocity.
        """     
        g = 9.81
        L = 1.0
        dt = 0.02
        X = np.zeros((self.n_samples, self.n_features, self.n_steps))
        y = np.zeros(self.n_samples, dtype=int)
        # separation in omega per cluster
        omega_sep = 2.0
        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            # same angular displacement range
            theta = np.random.uniform(-0.5, 0.5)
            # cluster-specific angular velocity range
            omega_low = -0.5 + k * omega_sep
            omega_high = 0.5 + k * omega_sep
            omega = np.random.uniform(omega_low, omega_high)
            for t in range(self.n_steps):
                features = np.array([np.cos(theta), np.sin(theta), omega])
                if self.n_features < features.size:
                    X[i, :, t] = features[: self.n_features]
                else:
                    padded = np.concatenate([features, np.zeros(self.n_features - features.size)])
                    X[i, :, t] = padded[: self.n_features]
                # update dynamics
                alpha = -(g / L) * np.sin(theta)
                theta = theta + omega * dt
                omega = omega + alpha * dt
                # wrap angle
                if theta > np.pi:
                    theta -= 2 * np.pi
                elif theta < -np.pi:
                    theta += 2 * np.pi
        return X, y

    def save_dataset(self, X, y, filepath):
        """
        Flatten and save dataset (X,y) to CSV via pandas.
        Columns: f{feat}_t{step}, and 'label'.
        """
        n_samples, n_features, n_steps = X.shape
        # flatten time-series into columns
        flat_X = X.reshape(n_samples, n_features * n_steps)
        # create column names f{feature}_t{step}
        cols = [f"f{feat}_t{step}" for feat in range(n_features) for step in range(n_steps)]
        df = pd.DataFrame(flat_X, columns=cols)
        df['label'] = y
        # write to CSV
        df.to_csv(filepath, index=False)

    @staticmethod
    def load_dataset(filepath, n_features, n_steps):
        """
        Load dataset CSV back into X (3D) and y (1D).
        """
        df = pd.read_csv(filepath)
        y = df['label'].astype(int).values
        flat_X = df.drop(columns=['label']).values
        n_samples = flat_X.shape[0]
        X = flat_X.reshape(n_samples, n_features, n_steps)
        return X, y


    def plot_dataset(self, name):
        """
        Plot example trajectories for each cluster in the named generator.

        Supports 1D (line), 2D (subplot + 3D), and higher dimensions.
        """
        if not hasattr(self, f"generate_{name}"):
            raise ValueError(f"No generator named {name}")
        X, y = getattr(self, f"generate_{name}")()
        labels = np.unique(y)
        n_feats = self.n_features
        
        cmap = plt.get_cmap('tab10')
        handles = [mlines.Line2D([], [], color=cmap(int(lbl) % 10), lw=2) for lbl in labels]
        label_names = [f"{lbl}" for lbl in labels]

        if n_feats == 1:
            plt.figure(figsize=(8, 4))
            for i in range(self.n_samples):
                plt.plot(range(self.n_steps), X[i, 0, :], color=cmap(y[i] % 10), alpha=0.9)
            plt.xlabel("Time step")
            plt.ylabel("Feature value")
            plt.legend(handles, label_names)
            plt.title(f"{name} (1D)")
            plt.show()

        elif n_feats == 2:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            # feature 0
            for i in range(self.n_samples):
                ax1.plot(range(self.n_steps), X[i, 0, :], color=cmap(y[i] % 10), alpha=0.9)
            ax1.set_title(f"{name} - Feature 0 vs Time")
            ax1.set_xlabel("Time step")
            ax1.set_ylabel("Value")
            ax1.legend(handles, label_names)
            # feature 1
            for i in range(self.n_samples):
                ax2.plot(range(self.n_steps), X[i, 1, :], color=cmap(y[i] % 10), alpha=0.9)
            ax2.set_title(f"{name} - Feature 1 vs Time")
            ax2.set_xlabel("Time step")
            ax2.set_ylabel("Value")
            ax2.legend(handles, label_names)
            # 3D trajectories
            for i in range(self.n_samples):
                ax3.plot(X[i, 0, :], X[i, 1, :], np.arange(self.n_steps), color=cmap(y[i] % 10), alpha=0.9)
            ax3.set_title(f"{name} - 3D Trajectories (2 features with time)")
            ax3.set_xlabel("Feature 0")
            ax3.set_ylabel("Feature 1")
            ax3.set_zlabel("Time step")
            ax3.legend(handles, label_names)
            plt.tight_layout()
            plt.show()

        else:
            n_cols = 2
            n_rows = int(np.ceil(n_feats / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            axes = axes.flatten()
            for f in range(n_feats):
                ax = axes[f]
                for i in range(self.n_samples):
                    ax.plot(range(self.n_steps), X[i, f, :], color=cmap(y[i] % 10), alpha=0.9)
                ax.set_title(f"Feature {f}")
                ax.set_xlabel("Time step")
                ax.set_ylabel(f"$f_{f}(t)$")
                ax.legend(handles, label_names)
            for ax in axes[n_feats:]:
                fig.delaxes(ax)
            fig.suptitle(name)
            plt.tight_layout()
            plt.show()

## ------------------------------------------------------------------------- ##
if __name__ == '__main__':
    output_dir = os.path.join(os.getcwd(), 'datasets')
    os.makedirs(output_dir, exist_ok=True)
    # specs: (dataset_name, n_samples, n_features, n_steps, n_clusters)
    specs = [
        ('hypersphere', 100, 3, 100, 2),
        ('hyperbolic', 200, 2, 50, 2),
        ('swiss_roll', 300, 2, 200, 4),
        ('lorenz', 100, 3, 100, 3),
        ('pendulum', 200, 2, 100, 4),
    ]
    for name, n_samples, n_features, n_steps, n_clusters in specs:
        print(f"Generating {name}: ({n_samples}, {n_features}, {n_steps}, {n_clusters} clusters)")
        gen = DatasetGenerator(n_samples, n_features, n_steps, n_clusters)
        X, y = getattr(gen, f"generate_{name}")()
        gen.plot_dataset(name)
        # filepath = os.path.join(output_dir, f"{name}.csv")
        # gen.save_dataset(X, y, filepath)

## ------------------------------------------------------------------------- ##
