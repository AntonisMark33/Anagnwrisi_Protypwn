import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


def run_q6(data):
    print("Running Q6: Multi-Layer Perceptron...")

    all_features = data.cont_cols + data.cat_encoded_cols
    X_train = data.train[all_features].values
    y_train = data.train['killer_id'].values
    X_val = data.val[all_features].values
    y_val = data.val['killer_id'].values

    # Architecture: Input -> Hidden(64) -> Hidden(32) -> Output(S) [cite: 186]
    # Uses Early Stopping on validation set [cite: 188]
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                        solver='adam', max_iter=1000,
                        early_stopping=True, validation_fraction=0.1,
                        random_state=42)
    clf.fit(X_train, y_train)

    acc_val = clf.score(X_val, y_val)
    print(f"MLP VAL Accuracy: {acc_val:.4f}")

    # Feature Importance via Permutation [cite: 190]
    # Delta A_j = A_base - A_j
    print("Calculating permutation importance...")
    r = permutation_importance(clf, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)

    # Sort and plot top 5
    sorted_idx = r.importances_mean.argsort()[::-1]
    top_5_idx = sorted_idx[:5]
    top_5_names = [all_features[i] for i in top_5_idx]

    plt.figure(figsize=(10, 6))
    plt.bar(range(5), r.importances_mean[top_5_idx], align='center')
    plt.xticks(range(5), top_5_names, rotation=45, ha='right')
    plt.title('Top 5 Features by Permutation Importance (MLP)')
    plt.ylabel('Accuracy Drop')
    plt.tight_layout()
    plt.savefig('q6_feature_importance.png')
    plt.close()

    print(f"Top 5 Important Features: {top_5_names}")

    return clf