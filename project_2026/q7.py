import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_q7(data):
    print("Running Q7: PCA...")
    
    all_features = data.cont_cols + data.cat_encoded_cols
    X_train = data.train[all_features].values
    
    # Standardize continuous features (and one-hot)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Run PCA on TRAIN
    pca = PCA()
    pca.fit(X_train_scaled)
    
    # Plot eigenvalues (explained variance)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, 'o-')
    plt.title('Scree Plot (PCA Eigenvalues)')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Eigenvalue (Explained Variance)')
    plt.grid(True)
    plt.savefig('q7_scree_plot.png')
    plt.close()
    
    # Cumulative variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cum_var) + 1), cum_var, 's-')
    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.grid(True)
    plt.savefig('q7_cumulative_variance.png')
    plt.close()
    
    # Project VAL onto first two components
    X_val = data.val[all_features].values
    X_val_scaled = scaler.transform(X_val)
    Z_val = pca.transform(X_val_scaled)
    
    # We'll use the SVM classifier from Q5 (passed later or just use true labels for now)
    # The requirement says "colouring each point according to the predicted killer label from your SVM classifier in Q5"
    
    return pca, scaler, Z_val
