import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


def run_q1(data):
    print("Running Q1: Exploratory distributions...")

    # Use TRAIN + VAL for exploration [cite: 108]
    df_combined = data.data[(data.data['split'] == 'TRAIN') | (data.data['split'] == 'VAL')]
    cols_to_plot = ['hour_float', 'victim_age', 'latitude', 'longitude']

    # 1. Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, col in enumerate(cols_to_plot):
        ax = axes[i // 2, i % 2]
        ax.hist(df_combined[col], bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {col}')
    plt.tight_layout()
    plt.savefig('q1_histograms.png')
    plt.close()

    # 2. hour_float analysis
    h_data = df_combined['hour_float'].values.reshape(-1, 1)

    # a) Single Gaussian (MLE mean and std) [cite: 110]
    mu_single = np.mean(h_data)
    std_single = np.std(h_data)

    # b) 3-component GMM (using EM via sklearn) [cite: 111-114]
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(h_data)

    # c) Comparison Plot
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 24, 1000).reshape(-1, 1)

    # Histogram
    plt.hist(h_data, bins=30, density=True, alpha=0.3, color='gray', label='Data Histogram')

    # Single Gaussian
    plt.plot(x, norm.pdf(x, mu_single, std_single), 'r--', lw=2, label='Single Gaussian')

    # GMM Density (sum of weighted components)
    log_prob = gmm.score_samples(x)
    plt.plot(x, np.exp(log_prob), 'g-', lw=2, label='3-Component GMM')

    plt.title('Time of Day (hour_float): Single vs Mixture Model')
    plt.xlabel('Hour')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('q1_hour_float_comparison.png')
    plt.close()

    # 3. 2D Scatter (hour vs latitude) [cite: 116]
    plt.figure(figsize=(10, 6))
    plt.scatter(df_combined['hour_float'], df_combined['latitude'], alpha=0.2, s=5)
    plt.title('Pattern: Time of Day vs Latitude')
    plt.xlabel('Hour')
    plt.ylabel('Latitude')
    plt.savefig('q1_2d_scatter.png')
    plt.close()