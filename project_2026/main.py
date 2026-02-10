import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from dataset import CrimeDataset
from gaussian_mle import GaussianMLE
from bayes import BayesianClassifier
import q1, q4, q5, q6, q7, q8
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    print("--- Piraeus Vice Pattern Recognition Project ---")

    # 1. Load Data
    data = CrimeDataset('crimes.csv')
    data.preprocess()
    data.split_train_val_test()

    # Q1: Exploratory Analysis
    q1.run_q1(data)

    # Q2: Gaussian MLE (using bias=True for 1/N)
    print("\n--- Q2: Gaussian MLE ---")
    mle = GaussianMLE()
    X_train_cont = data.train[data.cont_cols].values
    y_train = data.train['killer_id'].values
    mle.fit(X_train_cont, y_train)
    mle.plot_covariances()
    mle.plot_ellipses(X_train_cont, y_train, 1, 2, ['latitude', 'longitude'])

    # Q3: Bayesian Classifier
    print("\n--- Q3: Bayesian Classifier ---")
    bayesian_clf = BayesianClassifier()
    bayesian_clf.fit(mle.mus, mle.sigmas, y_train)

    X_val_cont = data.val[data.cont_cols].values
    y_val = data.val['killer_id'].values
    preds_val_bayes = bayesian_clf.predict(X_val_cont)
    print(f"Gaussian Bayes VAL Accuracy: {accuracy_score(y_val, preds_val_bayes):.4f}")

    # Visualization for Q3
    scaler_viz = StandardScaler()
    X_train_full = data.train[data.cont_cols + data.cat_encoded_cols].values
    X_train_scaled = scaler_viz.fit_transform(X_train_full)
    pca_viz = PCA(n_components=2)
    X_pca_2d = pca_viz.fit_transform(X_train_scaled)
    bayesian_clf.plot_decision_regions(X_pca_2d, y_train, "Gaussian Bayes Decision Regions (PCA 2D)",
                                       "q3_decision_regions.png")

    # Q7: PCA Setup (Run early to provide inputs for Q4 and Q5 visualization)
    print("\n--- Q7: PCA Setup ---")
    pca_full, scaler_full, Z_val = q7.run_q7(data)
    X_train_scaled_full = scaler_full.transform(X_train_full)
    Z_train = pca_full.transform(X_train_scaled_full)

    # Q4: Linear Classifier (Ridge/MSE)
    print("\n--- Q4: Linear Classifier ---")
    linear_clf = q4.run_q4(data, Z_train, Z_val)

    # Q5: SVM (One-vs-Rest)
    print("\n--- Q5: SVM ---")
    svm_clf = q5.run_q5(data, Z_train, Z_val)

    # Q7c: Visualization of SVM Predictions on VAL
    # This specifically answers Q7c which asks to color VAL points by SVM labels
    print("Generating Q7c plot (VAL set + SVM predictions)...")
    svm_preds_val = svm_clf.predict(data.val[data.cont_cols + data.cat_encoded_cols].values)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z_val[:, 0], Z_val[:, 1], c=svm_preds_val, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="SVM Pred")
    plt.title('Validation Set on PC1 vs PC2 (Colored by SVM Predictions)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('q7_val_svm_predictions.png')
    plt.close()

    # Q6: MLP
    print("\n--- Q6: MLP ---")
    mlp_clf = q6.run_q6(data)

    # Q8: k-means
    print("\n--- Q8: Unsupervised k-means ---")
    kmeans_model, mapping, y_pred_test_kmeans = q8.run_q8(data, pca_full, scaler_full, m=10)

    # Final Submission Generation
    print("\nGenerating submission.csv...")
    X_test = data.test[data.cont_cols + data.cat_encoded_cols].values

    # We use MLP for final probabilities as it generally handles this complexity best
    probs_test = mlp_clf.predict_proba(X_test)
    preds_test = mlp_clf.predict(X_test)

    submission = pd.DataFrame({
        'incident_id': data.test_raw['incident_id'].values,
        'predicted_killer': preds_test
    })

    for k in range(1, data.S + 1):
        submission[f'p_killer_{k}'] = probs_test[:, k - 1]

    submission.to_csv('submission.csv', index=False)
    print("Done. Saved submission.csv.")


if __name__ == "__main__":
    main()