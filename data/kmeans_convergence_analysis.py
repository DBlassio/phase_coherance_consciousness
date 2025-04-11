import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time
from tqdm import tqdm
import matplotlib.gridspec as gridspec

def analyze_kmeans_ninit_stability(data, k_opt=5, n_init_values=None, max_iter=300, 
                                  n_experiments=5, improvement_threshold=0.05):
    """
    Comprehensive analysis of how n_init parameter affects KMeans clustering stability.
    
    This function:
    1. Runs KMeans with various n_init values multiple times to assess stability
    2. Measures metrics like inertia, Manhattan distance, and centroid correlation
    3. Creates visualizations showing the effect of n_init on clustering stability
    4. Identifies the optimal n_init value for the dataset
    5. Shows example clusters for different n_init values
    6. Compares multiple runs side by side to demonstrate stability differences
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input dataset for clustering.
    k_opt : int, default=5
        The fixed number of clusters to use in KMeans.
    n_init_values : list of int, optional
        A list of n_init values to try. If None, defaults to [1, 5, 10, 20, 50, 100, 200].
    max_iter : int, default=300
        Maximum number of iterations for each KMeans run.
    n_experiments : int, default=5
        Number of experiments to run for each n_init value to measure variability.
    improvement_threshold : float, default=0.05
        The threshold (5% by default) used to decide when improvements are minimal.
    
    Returns
    -------
    df_stability : pd.DataFrame
        A DataFrame summarizing stability results for each n_init value.
    optimal_n_init : int
        The suggested optimal n_init value based on the improvement threshold.
    """
    # Helper functions
    def compute_manhattan_distance(data, centroids, labels):
        """Compute sum of Manhattan distances between points and their centroids."""
        distances = cdist(data, centroids, metric='cityblock')
        return np.sum([distances[i, labels[i]] for i in range(data.shape[0])])
    
    def compute_centroid_correlation(centroids1, centroids2):
        """Compute correlation between two sets of centroids."""
        sorted1 = centroids1[np.argsort(centroids1[:, 0])]
        sorted2 = centroids2[np.argsort(centroids2[:, 0])]
        flat1 = sorted1.flatten()
        flat2 = sorted2.flatten()
        corr = np.corrcoef(flat1, flat2)[0, 1]
        return corr

    # Default n_init values if not provided
    if n_init_values is None:
        n_init_values = [1, 5, 10, 20, 50, 100, 200]
    
    print(f"Analyzing KMeans stability with different n_init values: {n_init_values}")
    print(f"Running {n_experiments} experiments for each n_init value...")
    
    # Data structures to store results
    mean_manhattan = []
    std_manhattan = []
    mean_inertia = []
    std_inertia = []
    mean_centroid_stability = []
    exec_times = []
    
    # Store all centroids for later visualization
    all_centroids_by_n_init = {}
    all_labels_by_n_init = {}
    
    # For each n_init value
    for n_init in tqdm(n_init_values, desc="Testing n_init values"):
        manhattan_runs = []
        inertia_runs = []
        centroid_correlations = []
        time_runs = []
        all_centroids = []
        all_labels = []
        
        # Run multiple experiments to measure variability
        for _ in range(n_experiments):
            start = time.time()
            kmeans = KMeans(n_clusters=k_opt, max_iter=max_iter, n_init=n_init, random_state=None)
            kmeans.fit(data)
            end = time.time()
            
            time_runs.append(end - start)
            manhattan_runs.append(compute_manhattan_distance(data, kmeans.cluster_centers_, kmeans.labels_))
            inertia_runs.append(kmeans.inertia_)
            all_centroids.append(kmeans.cluster_centers_)
            all_labels.append(kmeans.labels_)
            
        # Calculate pairwise correlations between centroids from different runs
        if n_experiments > 1:
            for i in range(n_experiments):
                for j in range(i+1, n_experiments):
                    corr = compute_centroid_correlation(all_centroids[i], all_centroids[j])
                    centroid_correlations.append(corr)
        
        # Append mean and std values for this n_init
        mean_manhattan.append(np.mean(manhattan_runs))
        std_manhattan.append(np.std(manhattan_runs))
        mean_inertia.append(np.mean(inertia_runs))
        std_inertia.append(np.std(inertia_runs))
        mean_centroid_stability.append(np.mean(centroid_correlations) if centroid_correlations else 0)
        exec_times.append(np.mean(time_runs))
        
        # Store for later visualization
        all_centroids_by_n_init[n_init] = all_centroids
        all_labels_by_n_init[n_init] = all_labels
    
    # Create DataFrame with all metrics
    df_stability = pd.DataFrame({
        "n_init": n_init_values,
        "mean_manhattan": mean_manhattan,
        "std_manhattan": std_manhattan,
        "mean_inertia": mean_inertia,
        "std_inertia": std_inertia,
        "centroid_stability": mean_centroid_stability,
        "execution_time": exec_times
    })
    
    print("\nStability analysis results:")
    print(df_stability)
    
    # =================================================================
    # SECTION 1: PRIMARY ANALYSIS PLOTS
    # =================================================================
    
    # Create figure for main analysis plots
    fig = plt.figure(figsize=(16, 12))
    grid = gridspec.GridSpec(2, 2, figure=fig)
    
    # ---- Plot 1: Manhattan Distance vs n_init with error bars ----
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.errorbar(df_stability["n_init"], df_stability["mean_manhattan"], 
               yerr=df_stability["std_manhattan"], marker='o', linewidth=2, capsize=5)
    ax1.set_xscale('log')
    ax1.set_xlabel("Number of Initializations (n_init)")
    ax1.set_ylabel("Mean Sum of Manhattan Distances")
    ax1.set_title("Manhattan Distance Stability vs. Number of Initializations")
    ax1.grid(True)

    # ---- Plot 2: Inertia vs n_init with error bars ----
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.errorbar(df_stability["n_init"], df_stability["mean_inertia"], 
               yerr=df_stability["std_inertia"], marker='s', linewidth=2, color='red', capsize=5)
    ax2.set_xscale('log')
    ax2.set_xlabel("Number of Initializations (n_init)")
    ax2.set_ylabel("Mean Inertia")
    ax2.set_title("Inertia Stability vs. Number of Initializations")
    ax2.grid(True)

    # ---- Plot 3: Centroid Stability vs n_init ----
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.plot(df_stability["n_init"], df_stability["centroid_stability"], marker='D', linewidth=2, color='green')
    ax3.set_xscale('log')
    ax3.set_xlabel("Number of Initializations (n_init)")
    ax3.set_ylabel("Centroid Stability (Mean Correlation)")
    ax3.set_title("Centroid Position Stability vs. Number of Initializations")
    ax3.grid(True)

    # ---- Plot 4: Elbow Analysis (Normalized Metrics vs n_init) ----
    # Normalize metrics to [0, 1] scale for comparison
    norm_manhattan_mean = 1 - (df_stability["mean_manhattan"] - df_stability["mean_manhattan"].min()) / \
                     (df_stability["mean_manhattan"].max() - df_stability["mean_manhattan"].min())
    norm_inertia_mean = 1 - (df_stability["mean_inertia"] - df_stability["mean_inertia"].min()) / \
                   (df_stability["mean_inertia"].max() - df_stability["mean_inertia"].min())
    norm_std_manhattan = 1 - (df_stability["std_manhattan"] - df_stability["std_manhattan"].min()) / \
                    (df_stability["std_manhattan"].max() - df_stability["std_manhattan"].min() + 1e-10)
    
    ax4 = fig.add_subplot(grid[1, 1])
    ax4.plot(df_stability["n_init"], norm_manhattan_mean, marker='o', linewidth=2,
           label="Normalized Mean Manhattan Distance")
    ax4.plot(df_stability["n_init"], norm_inertia_mean, marker='s', linewidth=2,
           label="Normalized Mean Inertia")
    ax4.plot(df_stability["n_init"], norm_std_manhattan, marker='^', linewidth=2,
           label="Normalized Manhattan Std (Stability)")
    ax4.plot(df_stability["n_init"], df_stability["centroid_stability"], marker='D', linewidth=2,
           label="Centroid Correlation")
    
    # Identify the optimal n_init where improvement is less than the threshold
    optimal_n_init = df_stability["n_init"].iloc[-1]
    for i in range(1, len(df_stability)):
        improvements = [
            abs(norm_manhattan_mean.iloc[i] - norm_manhattan_mean.iloc[i-1]),
            abs(norm_inertia_mean.iloc[i] - norm_inertia_mean.iloc[i-1]),
            abs(norm_std_manhattan.iloc[i] - norm_std_manhattan.iloc[i-1])
        ]
        if all(imp < improvement_threshold for imp in improvements):
            optimal_n_init = df_stability["n_init"].iloc[i]
            break

    ax4.axhline(y=1 - improvement_threshold, color='r', linestyle='--',
              label=f"Improvement < {int(improvement_threshold*100)}%")
    ax4.axvline(x=optimal_n_init, color='g', linestyle='-', label=f"Optimal n_init ≈ {optimal_n_init}")

    ax4.set_xscale('log')
    ax4.set_xlabel("Number of Initializations (n_init)")
    ax4.set_ylabel("Normalized Metrics")
    ax4.set_title("Elbow Analysis: Stability vs. Number of Initializations")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()
    
    print(f"Based on the elbow analysis, the optimal n_init value is approximately: {optimal_n_init}")
    print(f"At this value, further increases in n_init yield less than {improvement_threshold*100}% improvement in stability metrics.")
    
    # =================================================================
    # SECTION 2: EXAMPLE CLUSTERS VISUALIZATION
    # =================================================================
    
    # Only proceed with visualization if data is 2D
    if data.shape[1] == 2:
        # Select a subset of n_init values to visualize (the first, middle, and last values)
        viz_n_init = [n_init_values[0], n_init_values[len(n_init_values)//2], n_init_values[-1]]
        print(f"\nVisualizing example clusters for n_init values: {viz_n_init}")
        
        # Set up the figure for cluster visualizations
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Example Cluster Formations with Different n_init Values", fontsize=16)
        
        for i, n_init_val in enumerate(viz_n_init):
            # Use the first experiment's results for this n_init
            centroids = all_centroids_by_n_init[n_init_val][0]
            labels = all_labels_by_n_init[n_init_val][0]
            
            # Plot data points colored by cluster, with centroids overlaid
            ax = axs[i]
            ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
            ax.set_title(f"n_init = {n_init_val}")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
        
        plt.tight_layout()
        plt.show()
        
        # =================================================================
        # SECTION 3: MULTIPLE RUNS COMPARISON
        # =================================================================
        
        # Select key n_init values to compare across multiple runs
        key_n_init = [n_init_values[0], n_init_values[-1]]  # Just compare first and last for clarity
        print(f"\nComparing multiple runs for key n_init values: {key_n_init}")
        
        # Limit to showing 4 runs at most to keep visualization clean
        n_runs_to_show = min(4, n_experiments)
        
        for n_init_val in key_n_init:
            fig, axs = plt.subplots(1, n_runs_to_show, figsize=(n_runs_to_show*5, 5))
            fig.suptitle(f"Multiple Runs with n_init={n_init_val}", fontsize=16)
            
            inertia_values = []
            
            for run in range(n_runs_to_show):
                centroids = all_centroids_by_n_init[n_init_val][run]
                labels = all_labels_by_n_init[n_init_val][run]
                inertia = df_stability.loc[df_stability['n_init'] == n_init_val, 'mean_inertia'].values[0]
                inertia_values.append(inertia)
                
                # Plot this run
                ax = axs[run]
                ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
                ax.set_title(f"Run {run+1}")
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
            
            # Add information about stability
            inertia_std = df_stability.loc[df_stability['n_init'] == n_init_val, 'std_inertia'].values[0]
            inertia_mean = df_stability.loc[df_stability['n_init'] == n_init_val, 'mean_inertia'].values[0]
            stability = df_stability.loc[df_stability['n_init'] == n_init_val, 'centroid_stability'].values[0]
            cv = (inertia_std / inertia_mean) * 100 if inertia_mean != 0 else 0
            
            fig.text(0.5, 0.01, 
                     f"Inertia: Mean={inertia_mean:.2f}, STD={inertia_std:.2f}, CV={cv:.2f}%\n" +
                     f"Centroid Stability: {stability:.4f} (higher is more stable)",
                     ha='center', fontsize=12)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.show()
    else:
        print("\nData is not 2D - skipping cluster visualizations.")

    return df_stability, optimal_n_init

