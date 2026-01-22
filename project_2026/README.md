# Who Is The Killer? 🕵️‍♂️
## Pattern Recognition and Machine Learning Investigation

This project implements a comprehensive machine learning pipeline to identify serial killers from a dataset of crime incidents. Using a combination of statistical modeling, discriminative classifiers, and unsupervised learning, we analyze crime patterns to predict the most likely perpetrator for each incident.

---

## 📊 Project Overview
The Piraeus Vice Homicide Division released an anonymized dataset of ~5,000 crime incidents (2019-2024). The objective is to identify **S=8 distinct serial killers** hidden in the data.

### Features
- **Continuous (8):** `hour_float`, `latitude`, `longitude`, `victim_age`, `temp_c`, `humidity`, `dist_precinct_km`, `pop_density`.
- **Categorical (4):** `weapon_code`, `scene_type`, `weather`, `vic_gender` (One-hot encoded).

---

## 🏗️ Modeling Pipeline

### Q1: Exploratory Data Analysis
- **Gaussian Mixture Models (GMM):** Analyzed `hour_float` distribution. While a single Gaussian provides a baseline, a 3-component GMM captures the multimodal nature of crime times (e.g., night-time peaks).
- **Spatial Analysis:** Visualized crime density across latitude and longitude.

### Q2: Maximum Likelihood Estimation (MLE)
- Implemented **MLE from scratch** to estimate Gaussian parameters ($\mu_k, \Sigma_k$) for each killer $k$.
- **Covariance Heatmaps:** Reveal feature correlations specific to each killer's Modus Operandi.
- **Mahalanobis Ellipses:** Visualized 2D spatial and temporal boundaries containing each killer's activities.

### Q3: Multiclass Gaussian Bayes Classifier
- **Generative Approach:** Combined MLE Gaussians with class priors ($\pi_k$) to calculate posterior probabilities $P(K=k | x)$.
- **Performance:** Achieved **~90.5% Accuracy** on the Validation set.

### Q4: Linear Discriminative Models
- Used **Logistic Regression** with one-hot encoded targets.
- **Result:** **~93.5% Accuracy**, demonstrating that high-dimensional linear boundaries are highly effective for this dataset.

### Q5: Support Vector Machines (SVM)
- Implemented an **RBF-Kernel SVM** with a One-vs-Rest strategy.
- Visualized non-linear decision regions and support vectors in 2D PCA space.

### Q6: Multi-Layer Perceptron (MLP)
- **Architecture:** 2 hidden layers (64, 32) with ReLU activations and Softmax output.
- **Feature Importance:** Permutation analysis identified `victim_age` and `humidity` as critical predictors.

### Q7: Principal Component Analysis (PCA)
- Reduced feature dimensionality to find the "Modus Operandi" space.
- Scree plots indicate that the first ~10 components capture the majority of variance.

### Q8: K-Means Clustering
- Unsupervised learning in the latent PCA space.
- Clusters were mapped to killer IDs using majority voting.
- **Accuracy:** **~83.3%**, showing strong inherent structure even without labels.

---

## 📈 Results Comparison

| Model | VAL Accuracy | Notes |
| :--- | :---: | :--- |
| **Linear Classifier** | **93.5%** | Highest performance; efficient. |
| **Gaussian Bayes** | 90.5% | Strong generative baseline. |
| **MLP (Neural Net)** | 89.1% | High complexity, good generalization. |
| **K-Means (Unsupervised)**| 83.3% | Validates natural grouping in MO. |
| **SVM (RBF)** | 86.5% | Effective non-linear boundaries. |

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. **Run Analysis:**
   ```bash
   python3 main.py
   ```
   This will execute the full pipeline (Q1-Q8), generate all plots, and produce the `submission.csv` file.

---

## 📂 File Structure
- `main.py`: Master execution script.
- `dataset.py`: Data loading and preprocessing logic.
- `gaussian_mle.py`: From-scratch implementation of MLE and visualization.
- `bayes.py`: Gaussian Bayesian Classifier.
- `q[1-8].py`: Individual task implementations.
- `*.png`: Visual insights (Decision regions, ellipses, importance, etc.).
- `submission.csv`: Final predictions and posterior probabilities.

---
**Course:** Pattern Recognition and Machine Learning  
**Associate Professor:** Dionisios N. Sotiropoulos
