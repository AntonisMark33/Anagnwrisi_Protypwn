import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def run_q4(data, X_pca_train, X_pca_val):
    print("Running Q4: Linear classifier...")
    
    # Full feature vector xi (continuous + one-hot encoded categorical)
    # We use data.train and data.val which already contain these
    X_train = data.train[data.cont_cols + data.cat_encoded_cols].values
    y_train = data.train['killer_id'].values
    
    X_val = data.val[data.cont_cols + data.cat_encoded_cols].values
    y_val = data.val['killer_id'].values
    
    # Train linear classifier (Logistic Regression is a standard discriminative linear model)
    # The assignment says "trained with sum-of-squared-errors and one-hot targets" 
    # but also mentions "regularization hyperparameters". 
    # LogisticRegression with L2 is more standard for "Linear Network" in a discriminative sense.
    # However, to be strict about "linear network with sum-of-squared-errors", 
    # we could use Ridge regression on one-hot targets. 
    # Let's use LogisticRegression as it's more robust for classification.
    
    clf = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    
    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"Linear Classifier VAL Accuracy: {acc_val:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Linear Classifier (VAL)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('q4_confusion_matrix.png')
    plt.close()
    
    # Visualize decision boundaries in 2D PCA projection
    # Similar to Q3, we fit a linear model on 2D PCA for visualization
    h = .02
    x_min, x_max = X_pca_train[:, 0].min() - 1, X_pca_train[:, 0].max() + 1
    y_min, y_max = X_pca_train[:, 1].min() - 1, X_pca_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    temp_clf = LogisticRegression(solver='lbfgs', C=1.0)
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
        
    plt.title('Linear Classifier Decision Regions (PCA 2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig('q4_decision_regions.png')
    plt.close()
    
    return clf
