import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import accuracy_score

def run_q8(data, pca, scaler, m):
    print(f"Running Q8: k-means with m={m} components...")
    
    all_features = data.cont_cols + data.cat_encoded_cols
    
    # Project TRAIN, VAL, TEST onto first m components
    X_train_scaled = scaler.transform(data.train[all_features].values)
    Z_train = pca.transform(X_train_scaled)[:, :m]
    
    X_val_scaled = scaler.transform(data.val[all_features].values)
    Z_val = pca.transform(X_val_scaled)[:, :m]
    
    X_test_scaled = scaler.transform(data.test[all_features].values)
    Z_test = pca.transform(X_test_scaled)[:, :m]
    
    S = data.S
    kmeans = KMeans(n_clusters=S, random_state=42, n_init=10)
    clusters_train = kmeans.fit_predict(Z_train)
    
    # Build mapping from k-means clusters to killer labels using majority vote
    y_train = data.train['killer_id'].values
    mapping = {}
    for q in range(S):
        members = y_train[clusters_train == q]
        if len(members) > 0:
            mapping[q] = Counter(members).most_common(1)[0][0]
        else:
            mapping[q] = -1 # Should not happen with enough data
            
    # Evaluate on VAL
    clusters_val = kmeans.predict(Z_val)
    y_pred_val = np.array([mapping[c] for c in clusters_val])
    y_val = data.val['killer_id'].values
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"k-means (S={S}) VAL Accuracy: {acc_val:.4f}")
    
    # Final predictions for TEST
    clusters_test = kmeans.predict(Z_test)
    y_pred_test = np.array([mapping[c] for c in clusters_test])
    
    # Visualization of TEST on PC1/PC2
    Z_test_2d = pca.transform(X_test_scaled)[:, :2]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z_test_2d[:, 0], Z_test_2d[:, 1], c=y_pred_test, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Predicted Killer")
    plt.title('k-means clustering on TEST (PC1 vs PC2 projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('q8_kmeans_test.png')
    plt.close()
    
    return kmeans, mapping, y_pred_test
