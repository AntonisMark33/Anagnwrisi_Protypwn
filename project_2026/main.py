import numpy as np
import pandas as pd
from dataset import CrimeDataset
from gaussian_mle import GaussianMLE
from bayes import BayesianClassifier
import q1, q4, q5, q6, q7, q8
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    # Load and preprocess data
    print("Initializing...")
    data = CrimeDataset('crimes.csv')
    data.preprocess()
    data.split_train_val_test()
    
    # Q1: Exploratory distributions
    q1.run_q1(data)
    
    # Q2: Gaussian MLE per killer
    print("Running Q2: Gaussian MLE...")
    mle = GaussianMLE()
    X_train_cont = data.train[data.cont_cols].values
    y_train = data.train['killer_id'].values
    mle.fit(X_train_cont, y_train)
    mle.plot_covariances()
    mle.plot_ellipses(X_train_cont, y_train, 1, 2, ['latitude', 'longitude'])
    mle.plot_ellipses(X_train_cont, y_train, 1, 0, ['latitude', 'hour_float'])
    
    # Q3: Multiclass Gaussian Bayes Classifier
    print("Running Q3: Bayesian Classifier...")
    bayesian_clf = BayesianClassifier()
    bayesian_clf.fit(mle.mus, mle.sigmas, y_train)
    
    X_val_cont = data.val[data.cont_cols].values
    y_val = data.val['killer_id'].values
    preds_val_bayes = bayesian_clf.predict(X_val_cont)
    print(f"Bayesian Classifier VAL Accuracy: {accuracy_score(y_val, preds_val_bayes):.4f}")
    
    # For decision regions, we need PCA
    mini_pca = PCA(n_components=2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(data.train[data.cont_cols + data.cat_encoded_cols].values)
    X_pca_train = mini_pca.fit_transform(X_train_scaled)
    bayesian_clf.plot_decision_regions(X_pca_train, y_train, "Bayesian Classifier Decision Regions (PCA 2D)", "q3_decision_regions.png")
    
    # Q7: PCA (running early to provide X_pca to Q4/Q5)
    pca_full, scaler_full, Z_val = q7.run_q7(data)
    # Z_train for visualization in Q4/Q5
    X_train_full_scaled = scaler_full.transform(data.train[data.cont_cols + data.cat_encoded_cols].values)
    Z_train = pca_full.transform(X_train_full_scaled)
    
    # Q4: Linear Classifier
    linear_clf = q4.run_q4(data, Z_train, Z_val)
    
    # Q5: SVM
    svm_clf = q5.run_q5(data, Z_train, Z_val)
    
    # Q6: MLP
    mlp_clf = q6.run_q6(data)
    
    # Q8: k-means
    # Choose m components (e.g., m=10 based on cumulative variance plot)
    kmeans_model, mapping, y_pred_test_kmeans = q8.run_q8(data, pca_full, scaler_full, m=10)
    
    # Generate final submission.csv
    print("Generating submission.csv...")
    # We'll use the MLP predictions as our "best" model for probabilities
    X_test = data.test[data.cont_cols + data.cat_encoded_cols].values
    probs_test = mlp_clf.predict_proba(X_test)
    preds_test = mlp_clf.predict(X_test)
    
    submission = pd.DataFrame({
        'incident_id': data.test_raw['incident_id'].values,
        'predicted_killer': preds_test
    })
    
    for k in range(1, data.S + 1):
        submission[f'p_killer_{k}'] = probs_test[:, k-1]
        
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved to submission.csv")

if __name__ == "__main__":
    main()