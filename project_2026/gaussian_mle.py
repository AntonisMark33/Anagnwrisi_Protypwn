import numpy as np
import matplotlib.pyplot as plt

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
            self.mus[i] = np.mean(Xk, axis=0)
            self.sigmas[i] = np.cov(Xk, rowvar=False)

    def plot_covariances(self):
        for i, k in enumerate(self.classes_):
            sigma = self.sigmas[i]
            plt.figure(figsize=(8, 6))
            plt.imshow(sigma, cmap='coolwarm')
            plt.colorbar()
            plt.title(f'Covariance matrix - Killer {k}')
            plt.savefig(f'q2_covariance_killer_{k}.png')
            plt.close()

    def plot_ellipses(self, X_cont, y, feat1_idx, feat2_idx, feat_names):
        from matplotlib.patches import Ellipse
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.classes_)))
        
        for i, k in enumerate(self.classes_):
            Xk = X_cont[y == k][:, [feat1_idx, feat2_idx]]
            mu = self.mus[i, [feat1_idx, feat2_idx]]
            sigma = self.sigmas[i][np.ix_([feat1_idx, feat2_idx], [feat1_idx, feat2_idx])]
            
            # Scatter plot for points
            plt.scatter(Xk[:, 0], Xk[:, 1], color=colors[i], alpha=0.3, label=f'Killer {k}', s=10)
            
            # Calculate Mahalanobis distance D^2 = (x-mu)^T Sigma^-1 (x-mu)
            try:
                inv_sigma = np.linalg.inv(sigma)
                diff = Xk - mu
                d2 = np.sum((diff @ inv_sigma) * diff, axis=1)
                ck = np.max(d2)
                
                # Draw ellipse: (x-mu)^T Sigma^-1 (x-mu) = ck
                # Eigenvalues/vectors for orientation
                vals, vecs = np.linalg.eigh(sigma)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                
                # Width and height are 2 * sqrt(ck * eigenvalue)
                w, h = 2 * np.sqrt(ck * vals)
                ell = Ellipse(xy=mu, width=w, height=h, angle=theta,
                              edgecolor=colors[i], facecolor='none', lw=2)
                plt.gca().add_patch(ell)
            except np.linalg.LinAlgError:
                print(f"Singular covariance for killer {k}, skipping ellipse.")

        plt.xlabel(feat_names[0])
        plt.ylabel(feat_names[1])
        plt.title(f'Ellipses containing all TRAIN points: {feat_names[0]} vs {feat_names[1]}')
        plt.legend()
        plt.savefig(f'q2_ellipses_{feat_names[0]}_{feat_names[1]}.png')
        plt.close()
