import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


def run_q5(data, X_pca_train, X_pca_val):
    print("Running Q5: Support Vector Machines...")

    X_train = data.train[data.cont_cols + data.cat_encoded_cols].values
    y_train = data.train['killer_id'].values
    X_val = data.val[data.cont_cols + data.cat_encoded_cols].values
    y_val = data.val['killer_id'].values

    # NOTE: The assignment explicitly requests a "one-vs-rest" multiclass strategy.
    # The standard sklearn.svm.SVC uses "one-vs-one" by default.
    # We must wrap the SVC in OneVsRestClassifier to comply with the instructions.
    base_svc = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    clf = OneVsRestClassifier(base_svc)
    clf.fit(X_train, y_train)

    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"SVM (One-vs-Rest) VAL Accuracy: {acc_val:.4f}")

    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - SVM (OVR)')
    plt.savefig('q5_confusion_matrix.png')
    plt.close()

    # Visualization (using standard SVC for plotting efficiency on mesh)
    h = .05
    x_min, x_max = X_pca_train[:, 0].min() - 1, X_pca_train[:, 0].max() + 1
    y_min, y_max = X_pca_train[:, 1].min() - 1, X_pca_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    viz_clf = SVC(kernel='rbf', C=1.0)
    viz_clf.fit(X_pca_train[:, :2], y_train)
    Z = viz_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    scatter = plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=y_train, cmap='tab10', edgecolors='k', s=20)

    sv = viz_clf.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=80, facecolors='none', edgecolors='white', linewidths=0.5,
                label='Support Vectors')

    plt.title('SVM Decision Regions (PCA Projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(*scatter.legend_elements(), title="Killer ID")
    plt.savefig('q5_decision_regions.png')
    plt.close()

    return clf