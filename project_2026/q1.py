import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

def run_q1(data):
    print("Running Q1: Exploratory distributions...")
    
    # 1. Plot histograms for hour_float, victim_age, latitude, longitude (using TRAIN + VAL)
    df_combined = data.data[(data.data['split'] == 'TRAIN') | (data.data['split'] == 'VAL')]
    cols_to_plot = ['hour_float', 'victim_age', 'latitude', 'longitude']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, col in enumerate(cols_to_plot):
        ax = axes[i//2, i%2]
        ax.hist(df_combined[col], bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
    plt.tight_layout()
    plt.savefig('q1_histograms.png')
    plt.close()

    # 2. For hour_float:
    h_data = df_combined['hour_float'].values.reshape(-1, 1)
    
    # a) Fit a single Gaussian
    mu_single = np.mean(h_data)
    std_single = np.std(h_data)
    
    # b) Fit a 3-component 1D Gaussian mixture model
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(h_data)
    
    # c) Plot comparison
    plt.figure(figsize=(10, 6))
    plt.hist(h_data, bins=30, density=True, alpha=0.4, label='Histogram', color='gray')
    
    x = np.linspace(0, 24, 1000).reshape(-1, 1)
    
    # Single Gaussian density
    plt.plot(x, norm.pdf(x, mu_single, std_single), 'r--', lw=2, label=f'Single Gaussian (μ={mu_single:.2f}, σ={std_single:.2f})')
    
    # GMM density
    log_prob = gmm.score_samples(x)
    plt.plot(x, np.exp(log_prob), 'g-', lw=2, label='3-component GMM')
    
    # Plot individual GMM components
    for j in range(3):
        weight = gmm.weights_[j]
        mean = gmm.means_[j, 0]
        var = gmm.covariances_[j, 0, 0]
        plt.plot(x, weight * norm.pdf(x, mean, np.sqrt(var)), 'b:', lw=1)

    plt.title('hour_float: Single Gaussian vs 3-component GMM')
    plt.xlabel('hour_float')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('q1_hour_float_comparison.png')
    plt.close()

    # 3. Two-dimensional plot (hour_float vs latitude)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_combined['hour_float'], df_combined['latitude'], alpha=0.1, s=1)
    plt.title('hour_float vs latitude')
    plt.xlabel('hour_float')
    plt.ylabel('latitude')
    plt.savefig('q1_2d_scatter.png')
    plt.close()
    
    print("Q1 plots saved: q1_histograms.png, q1_hour_float_comparison.png, q1_2d_scatter.png")
