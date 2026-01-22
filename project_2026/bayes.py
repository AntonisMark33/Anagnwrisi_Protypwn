import numpy as np


class BayesianClassifier:
    """
    Bayesian classifier that consumes per-class Gaussian parameters:
      - mus:    array-like (K, D)
      - sigmas: array-like (K, D, D) covariance matrices
    and predicts p(class | x) using multivariate normal likelihoods.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mus = None
        self.sigmas = None
        self.classes_ = None
        self.log_priors_ = None

    def fit(self, mus, sigmas, y):
        mus = np.asarray(mus, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        y = np.asarray(y)

        if mus.ndim != 2:
            raise ValueError(f"mus must have shape (K, D). Got {mus.shape}.")
        if sigmas.ndim != 3 or sigmas.shape[0] != mus.shape[0] or sigmas.shape[1] != mus.shape[1] or sigmas.shape[2] != mus.shape[1]:
            raise ValueError(
                f"sigmas must have shape (K, D, D) matching mus (K, D). Got mus={mus.shape}, sigmas={sigmas.shape}."
            )

        self.mus = mus
        self.sigmas = sigmas

        # Classes are inferred from y; assumes mus/sigmas are in the same order as unique(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        priors = counts / counts.sum()
        self.log_priors_ = np.log(priors + 1e-300)  # avoid log(0)

        if len(self.classes_) != mus.shape[0]:
            raise ValueError(
                f"Number of classes in y ({len(self.classes_)}) does not match mus/sigmas K ({mus.shape[0]}). "
                "Ensure your GaussianMLE produces parameters per unique class in y, in the same order."
            )

        return self

    def _logpdf_mvn(self, X, mu, Sigma):
        """
        Compute log N(X | mu, Sigma) for all rows of X.
        Uses Cholesky for stability; adds eps*I regularization.
        """
        X = np.asarray(X, dtype=float)
        mu = np.asarray(mu, dtype=float)
        Sigma = np.asarray(Sigma, dtype=float)

        d = X.shape[1]
        Sigma_reg = Sigma + self.eps * np.eye(d)

        try:
            L = np.linalg.cholesky(Sigma_reg)
        except np.linalg.LinAlgError:
            # Fallback: increase regularization if covariance is near-singular
            Sigma_reg = Sigma + (self.eps * 10.0) * np.eye(d)
            L = np.linalg.cholesky(Sigma_reg)

        diff = (X - mu)

        # Solve L * z = diff.T  => z = L^{-1} diff.T
        z = np.linalg.solve(L, diff.T)  # shape (D, N)
        maha = np.sum(z * z, axis=0)  # length N

        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)

    def predict_proba(self, X):
        if self.mus is None or self.sigmas is None or self.log_priors_ is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")

        X = np.asarray(X, dtype=float)
        K = self.mus.shape[0]

        log_liks = np.empty((X.shape[0], K), dtype=float)
        for k in range(K):
            log_liks[:, k] = self._logpdf_mvn(X, self.mus[k], self.sigmas[k]) + self.log_priors_[k]

        # log-softmax for numerical stability
        m = np.max(log_liks, axis=1, keepdims=True)
        exp_ = np.exp(log_liks - m)
        probs = exp_ / np.sum(exp_, axis=1, keepdims=True)
        return probs

    def plot_decision_regions(self, X_pca, y, title, save_path):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        h = .02  # step size in the mesh
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # We need to project the mesh points back to full space or 
        # just use a model trained on PCA components for visualization
        # The assignment says: "In a two-dimensional projection (for example, the first two principal components from PCA), 
        # visualise the decision regions induced by the Bayes classifier"
        # This implies we should train the Bayes classifier ON THE PCA PROJECTION for this plot.
        
        temp_clf = BayesianClassifier(eps=self.eps)
        # Assuming y is already mapped to 0...K-1 internally or using the same classes_
        # To simplify, we'll just fit it here for the visualization
        classes = np.unique(y)
        K = len(classes)
        D = 2
        mus_pca = np.zeros((K, D))
        sigmas_pca = np.zeros((K, D, D))
        for i, k in enumerate(classes):
            Xk = X_pca[y == k]
            mus_pca[i] = np.mean(Xk, axis=0)
            sigmas_pca[i] = np.cov(Xk, rowvar=False)
        
        temp_clf.fit(mus_pca, sigmas_pca, y)
        Z = temp_clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Convert Z back to indices for plotting if they are killer_ids
        Z_idx = np.zeros_like(Z, dtype=int)
        for i, k in enumerate(classes):
            Z_idx[Z == k] = i
            
        plt.figure(figsize=(10, 8))
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', '#AAFFFF', '#FFD700', '#C0C0C0'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#808080'])
        
        plt.pcolormesh(xx, yy, Z_idx, cmap=cmap_light, alpha=0.8)
        
        for i, k in enumerate(classes):
            plt.scatter(X_pca[y == k, 0], X_pca[y == k, 1], c=[cmap_bold.colors[i]], label=f'Killer {k}', edgecolors='k', s=20)
            
        plt.title(title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
