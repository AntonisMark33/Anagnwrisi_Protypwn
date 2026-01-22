import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

def run_q6(data):
    print("Running Q6: MLP...")
    
    all_features = data.cont_cols + data.cat_encoded_cols
    X_train = data.train[all_features].values
    y_train = data.train['killer_id'].values
    
    X_val = data.val[all_features].values
    y_val = data.val['killer_id'].values
    
    # MLP with 2 hidden layers (64 and 32 units)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
                        max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
    clf.fit(X_train, y_train)
    
    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"MLP VAL Accuracy: {acc_val:.4f}")
    
    # Permutation feature importance on VAL
    print("Computing permutation importance (this may take a minute)...")
    r = permutation_importance(clf, X_val, y_val, n_repeats=5, random_state=42)
    
    # Sort features by importance
    sorted_idx = r.importances_mean.argsort()[::-1]
    top_5_idx = sorted_idx[:5]
    top_5_features = [all_features[i] for i in top_5_idx]
    top_5_importances = r.importances_mean[top_5_idx]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(5), top_5_importances, align='center')
    plt.xticks(range(5), top_5_features, rotation=45, ha='right')
    plt.title('Top 5 Most Important Features (MLP)')
    plt.ylabel('Drop in Accuracy')
    plt.tight_layout()
    plt.savefig('q6_feature_importance.png')
    plt.close()
    
    print(f"Top features: {top_5_features}")
    
    return clf
