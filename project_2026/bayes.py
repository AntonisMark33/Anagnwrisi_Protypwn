import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class BayesianClassifier:
    """
    Implements P(K=k | x) proportional to pi_k * N(x | mu_k, Sigma_k)[cite: 151].
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)  # Regularization term for stability
        self.mus = None
        self.sigmas = None
        self.classes_ = None
        self.log_priors_ = None

    def fit(self, mus, sigmas, y):
        """
        Stores the MLE parameters and estimates class priors pi_k = N_k / N[cite: 148].
        """
        self.mus = np.asarray(mus, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)

        # Estimate priors from training labels
        self.classes_, counts = np.unique(y, return_counts=True)
        priors = counts / counts.sum()
        self.log_priors_ = np.log(priors + 1e-300)  # Log space for stability

        return self

    def _logpdf_mvn(self, X, mu, Sigma):
        """
        Calculates log Gaussian PDF:
        -0.5 * (d*log(2pi) + log|Sigma| + (x-mu)^T Sigma^-1 (x-mu))
        """
        d = X.shape[1]
        # Add epsilon to diagonal for numerical stability (avoid singular matrix)
        Sigma_reg = Sigma + self.eps * np.eye(d)

        try:
            # Cholesky decomposition L*L.T = Sigma is faster/stable for determinant and inverse
            L = np.linalg.cholesky(Sigma_reg)
        except np.linalg.LinAlgError:
            # Fallback: stronger regularization if matrix is very ill-conditioned
            Sigma_reg = Sigma + (self.eps * 100.0) * np.eye(d)
            L = np.linalg.cholesky(Sigma_reg)

        # Solve L * z = (x - mu)^T
        diff = (X - mu)
        z = np.linalg.solve(L, diff.T)

        # Mahalanobis term: z^T z
        maha = np.sum(z * z, axis=0)

        # Log determinant: 2 * sum(log(diag(L)))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)

    def predict_proba(self, X):
        """
        Returns posterior probabilities pi_hat(k)[cite: 154].
        """
        X = np.asarray(X, dtype=float)
        K = self.mus.shape[0]

        log_posteriors = np.empty((X.shape[0], K), dtype=float)

        # Compute log joint probability: log(P(x|k)) + log(P(k))
        for k in range(K):
            log_posteriors[:, k] = self._logpdf_mvn(X, self.mus[k], self.sigmas[k]) + self.log_priors_[k]

        # Softmax normalization (using log-sum-exp trick for stability)
        # P(k|x) = exp(log_post_k) / sum_j exp(log_post_j)
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_vals = np.exp(log_posteriors - max_log)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        return probs

    def predict(self, X):
        """
        Returns hard classification c_hat = argmax P(K=k|x)[cite: 156].
        """
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

    def plot_decision_regions(self, X_pca, y, title, save_path):
        """
        Visualizes decision boundaries in 2D PCA space[cite: 160].
        NOTE: To do this, we essentially train a 'mini' Bayes classifier
        just on the 2D projected data.
        """
        # Create meshgrid
        h = .02
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Fit temporary 2D Bayes for visualization
        temp_clf = BayesianClassifier(eps=self.eps)
        classes = np.unique(y)
        K = len(classes)
        mus_2d = np.array([np.mean(X_pca[y == k], axis=0) for k in classes])
        sigmas_2d = np.array([np.cov(X_pca[y == k], rowvar=False, bias=True) for k in classes])

        temp_clf.fit(mus_2d, sigmas_2d, y)

        # Predict on mesh
        Z = temp_clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(10, 8))
        cmap_light = ListedColormap(
            ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', '#AAFFFF', '#FFD700', '#C0C0C0'])
        cmap_bold = ListedColormap(
            ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#808080'])

        # Map class labels to 0..K-1 for coloring if needed, assuming classes are ints
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.8)

        for i, k in enumerate(classes):
            plt.scatter(X_pca[y == k, 0], X_pca[y == k, 1], c=[cmap_bold.colors[i % len(cmap_bold.colors)]],
                        label=f'Killer {k}', edgecolors='k', s=20)

        plt.title(title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig(save_path)
        plt.close()