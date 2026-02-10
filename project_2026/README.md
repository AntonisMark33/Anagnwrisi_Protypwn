# Who Is The Killer? 🕵️‍♂️
## Pattern Recognition and Machine Learning Investigation

This project implements a comprehensive machine learning pipeline to identify serial killers from a dataset of crime incidents. Using a combination of statistical modeling, discriminative classifiers, and unsupervised learning, we analyze crime patterns to predict the most likely perpetrator for each incident.

---

## 📊 Project Overview
[cite_start]The **Piraeus Vice Homicide Division** has released an anonymized dataset of approximately 5,000 crime incidents spanning the years 2019-2024[cite: 5]. [cite_start]The core objective is to uncover the activity of **$S=8$ distinct serial killers** hidden within this data[cite: 6].

### Feature Space
[cite_start]The data is decomposed into continuous and categorical components [cite: 27-35]:
- **Continuous ($d_c=8$):** `hour_float`, `latitude`, `longitude`, `victim_age`, `temp_c`, `humidity`, `dist_precinct_km`, `pop_density`.
- **Categorical ($d_{cat}=17$):** `weapon_code`, `scene_type`, `weather`, `vic_gender` (One-hot encoded).

---

## 🏗️ Modeling Pipeline

### Q1: Exploratory Data Analysis
- **Gaussian Mixture Models (GMM):** Analyzed the `hour_float` distribution. [cite_start]While a single Gaussian provides a baseline, a 3-component GMM was fitted to capture the multimodal nature of crime times (e.g., distinguishing "morning", "evening", and "late-night" modes)[cite: 111].
- **Spatial Analysis:** Visualized crime density across latitude and longitude to detect spatial clustering.

### Q2: Maximum Likelihood Estimation (MLE)
- [cite_start]**Implementation:** Derived and implemented **MLE from scratch** (using the biased estimator $\frac{1}{N}$) to estimate Gaussian parameters ($\mu_k, \Sigma_k$) for each killer $k$[cite: 126].
- **Covariance Heatmaps:** Generated to reveal feature correlations specific to each killer's *Modus Operandi*.
- [cite_start]**Mahalanobis Ellipses:** Visualized 2D spatial boundaries (e.g., Lat/Lon) containing the training points for each killer[cite: 138].

### Q3: Multiclass Gaussian Bayes Classifier
- [cite_start]**Generative Approach:** Combined the MLE-derived Gaussians with class priors ($\pi_k$) to calculate the posterior probability $P(K=k | x)$ for each incident[cite: 151].
- **Decision Boundaries:** Visualized the probabilistic decision regions in the PCA-projected space.

### Q4: Linear Discriminative Model
- **Model:** **Ridge Classifier** (Linear Least Squares).
- [cite_start]**Rationale:** Trained using **sum-of-squared-errors** on one-hot targets to strictly satisfy the assignment's "Linear Network" requirement.
- **Performance:** Demonstrates that high-dimensional linear boundaries are highly effective for identifying specific killer signatures.

### Q5: Support Vector Machines (SVM)
- **Model:** Non-linear SVM with an **RBF Kernel**.
- [cite_start]**Strategy:** Employed a **One-vs-Rest** multiclass strategy[cite: 178].
- **Visualization:** Plotted non-linear decision regions and highlighted support vectors in the 2D PCA latent space.

### Q6: Multi-Layer Perceptron (MLP)
- [cite_start]**Architecture:** Feed-forward neural network with 2 hidden layers (64, 32 units), ReLU activations, and a Softmax output layer[cite: 186].
- [cite_start]**Feature Importance:** Performed permutation feature importance analysis to identify critical predictors (e.g., `victim_age`, `humidity`) by measuring accuracy drops on the validation set[cite: 190].

### Q7: Principal Component Analysis (PCA)
- [cite_start]**Dimensionality Reduction:** Standardized features and computed the eigendecomposition to find the "Modus Operandi" space[cite: 200].
- **Analysis:** Scree plots were generated to determine the optimal number of components ($m$) that explain the majority of the variance.

### Q8: K-Means Clustering
- [cite_start]**Unsupervised Learning:** Applied k-means clustering in the latent PCA space ($m=10$)[cite: 210].
- [cite_start]**Mapping:** Clusters were mapped to killer IDs using a majority voting scheme on the training set to evaluate unsupervised identification capability[cite: 218].

---

## 📈 Performance Summary

| Model | Type | Key Characteristic |
| :--- | :---: | :--- |
| **Gaussian Bayes** | Generative | Probabilistic baseline using MLE parameters. |
| **Linear (Ridge)** | Discriminative | Minimizes squared error; simple & effective. |
| **SVM (RBF)** | Discriminative | Captures non-linear boundaries via kernels. |
| **MLP (Neural Net)** | Non-linear | Learns complex feature interactions. |
| **K-Means** | Unsupervised | Detects structure without labels. |

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn