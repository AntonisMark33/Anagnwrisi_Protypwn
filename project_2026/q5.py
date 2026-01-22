import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def run_q5(data, X_pca_train, X_pca_val):
    print("Running Q5: SVM...")
    
    X_train = data.train[data.cont_cols + data.cat_encoded_cols].values
    y_train = data.train['killer_id'].values
    
    X_val = data.val[data.cont_cols + data.cat_encoded_cols].values
    y_val = data.val['killer_id'].values
    
    # Non-linear SVM with RBF kernel
    # One-vs-rest is default in SVC for multiclass
    clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"SVM VAL Accuracy: {acc_val:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - SVM (VAL)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('q5_confusion_matrix.png')
    plt.close()
    
    # Visualize decision boundaries in 2D PCA projection
    h = .05 # larger step for SVM as it's slower
    x_min, x_max = X_pca_train[:, 0].min() - 1, X_pca_train[:, 0].max() + 1
    y_min, y_max = X_pca_train[:, 1].min() - 1, X_pca_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    temp_clf = SVC(kernel='rbf', C=1.0)
    temp_clf.fit(X_pca_train[:, :2], y_train)
    Z = temp_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    classes = np.unique(y_train)
    Z_idx = np.zeros_like(Z, dtype=int)
    for i, k in enumerate(classes):
        Z_idx[Z == k] = i

    plt.figure(figsize=(10, 8))
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', '#AAFFFF', '#FFD700', '#C0C0C0'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#808080'])
    
    plt.pcolormesh(xx, yy, Z_idx, cmap=cmap_light, alpha=0.8)
    for i, k in enumerate(classes):
        plt.scatter(X_pca_train[y_train == k, 0], X_pca_train[y_train == k, 1], c=[cmap_bold.colors[i]], label=f'Killer {k}', edgecolors='k', s=20)
    
    # Indicate support vectors (on the 2D model)
    sv = temp_clf.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
        
    plt.title('SVM Decision Regions and Support Vectors (PCA 2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig('q5_decision_regions.png')
    plt.close()
    
    return clf
