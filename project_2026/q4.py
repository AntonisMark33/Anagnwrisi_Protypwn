import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


def run_q4(data, X_pca_train, X_pca_val):
    print("Running Q4: Linear Classifier...")

    X_train = data.train[data.cont_cols + data.cat_encoded_cols].values
    y_train = data.train['killer_id'].values
    X_val = data.val[data.cont_cols + data.cat_encoded_cols].values
    y_val = data.val['killer_id'].values

    # NOTE: The assignment specifically asks for a linear classifier trained with
    # "sum-of-squared-errors" (MSE).
    # Logistic Regression optimizes Log-Loss (Cross-Entropy).
    # RidgeClassifier optimizes the Squared Error (MSE) with L2 regularization,
    # satisfying the strict mathematical requirement of the prompt.
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, y_train)

    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"Linear Classifier (Ridge/MSE) VAL Accuracy: {acc_val:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Linear Classifier (Ridge)')
    plt.savefig('q4_confusion_matrix.png')
    plt.close()

    # Visualization on PCA (Proxy model)
    h = .02
    x_min, x_max = X_pca_train[:, 0].min() - 1, X_pca_train[:, 0].max() + 1
    y_min, y_max = X_pca_train[:, 1].min() - 1, X_pca_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    viz_clf = RidgeClassifier(alpha=1.0)
    viz_clf.fit(X_pca_train[:, :2], y_train)
    Z = viz_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    scatter = plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=y_train, cmap='tab10', edgecolors='k', s=20)
    plt.legend(*scatter.legend_elements(), title="Killer ID")
    plt.title('Linear Decision Boundaries (Ridge/MSE) on PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('q4_decision_regions.png')
    plt.close()

    return clf