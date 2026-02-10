import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class GaussianMLE:
    def __init__(self):
        self.mus = None
        self.sigmas = None
        self.classes_ = None

    def fit(self, X_cont, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        D = X_cont.shape[1]

        self.mus = np.zeros((K, D))
        self.sigmas = np.zeros((K, D, D))

        for i, k in enumerate(self.classes_):
            Xk = X_cont[y == k]

            # MLE Mean: standard sample mean
            self.mus[i] = np.mean(Xk, axis=0)

            # NOTE: The assignment asks for the Maximum Likelihood Estimator.
            # The MLE for covariance requires dividing by N, not (N-1).
            # Default np.cov uses (N-1) (unbiased). We set bias=True to force division by N.
            self.sigmas[i] = np.cov(Xk, rowvar=False, bias=True)

    def plot_covariances(self):
        for i, k in enumerate(self.classes_):
            sigma = self.sigmas[i]
            plt.figure(figsize=(6, 5))
            plt.imshow(sigma, cmap='coolwarm', interpolation='nearest')
            plt.colorbar(label='Covariance')
            plt.title(f'MLE Covariance Matrix - Killer {k}')
            plt.tight_layout()
            plt.savefig(f'q2_covariance_killer_{k}.png')
            plt.close()

    def plot_ellipses(self, X_cont, y, feat1_idx, feat2_idx, feat_names):
        """
        Draws ellipses defined by (x-mu)^T Sigma^-1 (x-mu) = c_k
        where c_k is the max Mahalanobis distance in the training set.
        """
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.classes_)))

        for i, k in enumerate(self.classes_):
            Xk = X_cont[y == k][:, [feat1_idx, feat2_idx]]
            mu = self.mus[i, [feat1_idx, feat2_idx]]

            # Extract 2D sub-matrix.
            sigma = self.sigmas[i][np.ix_([feat1_idx, feat2_idx], [feat1_idx, feat2_idx])]

            plt.scatter(Xk[:, 0], Xk[:, 1], color=colors[i], alpha=0.3, label=f'Killer {k}', s=15)

            try:
                inv_sigma = np.linalg.inv(sigma)
                diff = Xk - mu
                d2 = np.sum((diff @ inv_sigma) * diff, axis=1)
                ck = np.max(d2)  # Outer boundary of training data

                vals, vecs = np.linalg.eigh(sigma)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * np.sqrt(ck * vals)

                ell = Ellipse(xy=mu, width=w, height=h, angle=theta,
                              edgecolor=colors[i], facecolor='none', lw=2, linestyle='--')
                plt.gca().add_patch(ell)
            except np.linalg.LinAlgError:
                pass

        plt.xlabel(feat_names[0])
        plt.ylabel(feat_names[1])
        plt.title(f'MLE Ellipses (Training Data): {feat_names[0]} vs {feat_names[1]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'q2_ellipses_{feat_names[0]}_{feat_names[1]}.png')
        plt.close()