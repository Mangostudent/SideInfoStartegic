import numpy as np
from scipy.stats import multivariate_normal

# --- 2. Data Handling ---
class Dataset:
    def __init__(self, n_train, n_test, rng, parameter_mode='random', verbose=True):
        self.n_train = n_train; self.n_test = n_test; self.rng = rng
        self.parameter_mode = parameter_mode
        self.verbose = verbose
        self._generate_data_distribution_parameters()
        self.X_train, self.Y_train, self.Z_train = None, None, None
        self.X_test, self.Y_test, self.Z_test = None, None, None

    def _generate_data_distribution_parameters(self):
        self.yz_pairs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

        if self.parameter_mode == 'manual':
            if self.verbose: print("\n--- Using MANUALLY DEFINED Data Distribution Parameters ---")
            self.yz_probabilities = np.array([0.23, 0.27, 0.22, 0.28])
            self.centers = {
                (-1, -1): np.array([-0.6, -0.6]), (-1, 1):  np.array([-0.65, -0.55]),
                (1, -1):  np.array([-0.5, -0.6]),  (1, 1):   np.array([-0.6, 0.55])
            }
            self.covariances = {
                (-1, -1): np.array([[0.1, 0.04], [0.04, 0.08]]), (-1, 1):  np.array([[0.08, -0.04], [-0.04, 0.1]]),
                (1, -1):  np.array([[0.08, 0.04], [0.04, 0.1]]),  (1, 1):   np.array([[0.1, -0.04], [-0.04, 0.08]])
            }
        elif self.parameter_mode == 'random':
            if self.verbose: print("\n--- Using RANDOMLY GENERATED Data Distribution Parameters ---")
            self.yz_probabilities = self.rng.dirichlet(np.ones(4))
            self.centers = {tuple(p): self.rng.uniform(-0.8, 0.8, size=2) for p in self.yz_pairs}
            self.covariances = {}
            for pair in self.yz_pairs:
                A = self.rng.uniform(-0.5, 0.5, size=(2, 2)); cov = A @ A.T + np.eye(2) * 0.01
                self.covariances[tuple(pair)] = cov * self.rng.uniform(0.05, 0.15)
        else:
            raise ValueError(f"Invalid parameter_mode: {self.parameter_mode}. Must be 'manual' or 'random'.")

    def _sample_from_distribution(self, n_samples):
        yz_idx = self.rng.choice(4, size=n_samples, p=self.yz_probabilities); Y = self.yz_pairs[yz_idx, 0]; Z = self.yz_pairs[yz_idx, 1]
        X = np.zeros((n_samples, 2))
        for i in range(n_samples):
            mean = self.centers[(Y[i], Z[i])]; cov = self.covariances[(Y[i], Z[i])]
            X[i, :] = self.rng.multivariate_normal(mean, cov=cov)
        return X, Y, Z
    def generate_data(self):
        self.X_train, self.Y_train, self.Z_train = self._sample_from_distribution(self.n_train)
        self.X_test, self.Y_test, self.Z_test = self._sample_from_distribution(self.n_test)
        if self.verbose: print(f"Generated {self.n_train} training samples and {self.n_test} test samples.")
