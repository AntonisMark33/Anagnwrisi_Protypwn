import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import accuracy_score


def run_q8(data, pca, scaler, m):
    """
    m: Number of principal components to use [cite: 211]
    """
    print(f"Running Q8: k-means clustering in {m}-dim PCA space...")

    all_features = data.cont_cols + data.cat_encoded_cols

    # Project data to m dimensions [cite: 214]
    Z_train = pca.transform(scaler.transform(data.train[all_features].values))[:, :m]
    Z_val = pca.transform(scaler.transform(data.val[all_features].values))[:, :m]
    Z_test = pca.transform(scaler.transform(data.test[all_features].values))[:, :m]

    # Run k-means with k=S on TRAIN [cite: 216]
    S = data.S
    kmeans = KMeans(n_clusters=S, random_state=42, n_init=10)
    cluster_labels_train = kmeans.fit_predict(Z_train)

    # Map clusters to killers (Majority Vote) [cite: 219]
    # g(q) = argmax_k Count(killer=k in cluster q)
    y_train = data.train['killer_id'].values
    cluster_map = {}

    for q in range(S):
        indices_in_cluster = np.where(cluster_labels_train == q)[0]
        if len(indices_in_cluster) > 0:
            true_labels_in_cluster = y_train[indices_in_cluster]
            most_common_killer = Counter(true_labels_in_cluster).most_common(1)[0][0]
            cluster_map[q] = most_common_killer
        else:
            cluster_map[q] = np.random.choice(data.train['killer_id'].unique())

    # Evaluate on VAL [cite: 224]
    val_cluster_ids = kmeans.predict(Z_val)
    val_killer_pred = np.array([cluster_map[c] for c in val_cluster_ids])
    acc_val = accuracy_score(data.val['killer_id'].values, val_killer_pred)
    print(f"Unsupervised k-means (Mapped) VAL Accuracy: {acc_val:.4f}")

    # Predict on TEST [cite: 229]
    test_cluster_ids = kmeans.predict(Z_test)
    test_killer_pred = np.array([cluster_map[c] for c in test_cluster_ids])

    # Visualize TEST predictions on 2D PCA [cite: 230]
    Z_test_2d = pca.transform(scaler.transform(data.test[all_features].values))[:, :2]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z_test_2d[:, 0], Z_test_2d[:, 1], c=test_killer_pred, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Predicted Killer")
    plt.title(f'k-means Predictions on TEST (Projected to 2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('q8_kmeans_test_predictions.png')
    plt.close()

    return kmeans, cluster_map, test_killer_pred