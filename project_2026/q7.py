import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_q7(data):
    print("Running Q7: PCA...")

    all_features = data.cont_cols + data.cat_encoded_cols
    X_train = data.train[all_features].values

    # Standardize data (zero mean, unit variance) [cite: 201]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # PCA on TRAIN
    pca = PCA()
    pca.fit(X_train_scaled)

    # Scree Plot (Eigenvalues) [cite: 203]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 15), pca.explained_variance_[:14], 'o-')  # Plot first few
    plt.title('Scree Plot: Eigenvalues vs Component Index')
    plt.xlabel('Component Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.savefig('q7_scree_plot.png')
    plt.close()

    # Project VAL for later use
    X_val = data.val[all_features].values
    X_val_scaled = scaler.transform(X_val)
    Z_val = pca.transform(X_val_scaled)

    return pca, scaler, Z_val