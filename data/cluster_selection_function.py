def cluster_selection(all_pattern2D, rangek=range(3, 8), metrics_det=None):
    """
    Perform cluster analysis on the given data and determine the optimal number of clusters.
    
    Parameters:
    -----------
    all_pattern2D : numpy.ndarray
        The input data array for clustering
    rangek : list or range, optional
        Range of k values to evaluate (default: range(3, 8))
    metrics_det : dict, optional
        Dictionary of metrics to use and their objectives ('Maximize' or 'Minimize')
        If None, default metrics will be used
        
    Returns:
    --------
    optimal_k : int
        The optimal number of clusters determined by analysis
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from scipy.spatial.distance import cityblock
    from scipy.stats import entropy
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed
    from matplotlib.ticker import MaxNLocator 


    # ============================================= Metrics

    # Manhattan Distances
    def compute_sum_distance(data, centroids, labels):
        return sum(cityblock(data[i], centroids[labels[i]]) for i in range(len(data)))

    # Shannon Entropy
    def cluster_shannon_entropy(cluster_data):
        hist, _ = np.histogram(cluster_data.flatten(), bins=10, density=True)
        return entropy(hist, base=2) if len(hist) > 0 else 0

    # IPVC 
    def compute_IPVC(cluster_data):
        n_samples = cluster_data.shape[0]
        if n_samples < 2:
            return 0
        
        corr_matrix = np.corrcoef(cluster_data)
        triu_indices = np.triu_indices_from(corr_matrix, k=1)
        correlations = corr_matrix[triu_indices]
        return np.var(correlations)

    # K-Means
    def evaluate_kmeans(k, data):
        kmeans = KMeans(n_clusters=k, max_iter=200, n_init=10, random_state=0)
        labels = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_
        
        # Metrics
        metrics = {
            "manhattan_distance": compute_sum_distance(data, centroids, labels),
            "silhouette": silhouette_score(data, labels, metric='cityblock'),
            "davies_bouldin": davies_bouldin_score(data, labels),
            "calinski_harabasz": calinski_harabasz_score(data, labels)
        }
        
        ipvc_values = []
        shannon_values = []
        
        # Per cluster
        for i in range(k):
            cluster_data = data[labels == i]
            if len(cluster_data) > 0:
                ipvc_values.append(compute_IPVC(cluster_data))
                shannon_values.append(cluster_shannon_entropy(cluster_data))
        
        metrics["ipvc"] = np.mean(ipvc_values) if ipvc_values else 0
        metrics["shannon_entropy"] = np.mean(shannon_values) if shannon_values else 0
        
        return k, metrics

    # ================================================ Plots

    def visualize_results(df_results, metrics_used):
        n_metrics = len(metrics_used)
        
        # Adjust Metric
        if n_metrics <= 3:
            fig, axes = plt.subplots(nrows=1, ncols=n_metrics, figsize=(5*n_metrics, 4))
            if n_metrics == 1:  
                axes = [axes]
        else:
            nrows = (n_metrics + 1) // 2 
            ncols = 2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows))
            axes = axes.flatten()
        
        for i, (metric, objective) in enumerate(metrics_used.items()):
            ax = axes[i]
            ax.plot(df_results['n_clusters'], df_results[metric], 'o-', linewidth=2)
            ax.set_xlabel('Number of clusters (k)')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} - k ({objective})')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True)
            
            # Marcar el mejor valor
            if objective == "Minimize":
                best_k = df_results.loc[df_results[metric].idxmin(), 'n_clusters']
                best_value = df_results[metric].min()
            else:
                best_k = df_results.loc[df_results[metric].idxmax(), 'n_clusters']
                best_value = df_results[metric].max()
                
            ax.scatter([best_k], [best_value], c='red', s=100, label=f'Best k={best_k}')
            ax.legend()
        
        if n_metrics > 3 and n_metrics % 2 == 1:
            axes[-1].axis('off')
            
        plt.tight_layout()
        plt.show()

    def calculate_optimal_k(df_results, metrics_used):
        """Calcula el k óptimo basado en múltiples métricas"""
        # Normalizar todas las métricas a un rango [0,1]
        normalized_df = df_results.copy()
        
        for metric, objective in metrics_used.items():
            min_val = normalized_df[metric].min()
            max_val = normalized_df[metric].max()
            range_val = max_val - min_val
            
            if range_val == 0: 
                normalized_df[metric] = 0
            else:
                if objective == "Maximize":
                    normalized_df[metric] = (normalized_df[metric] - min_val) / range_val
                else: 
                    normalized_df[metric] = 1 - ((normalized_df[metric] - min_val) / range_val)
        
        normalized_df['combined_score'] = normalized_df[list(metrics_used.keys())].mean(axis=1)
        
        # Over-all Grade Visualization
        plt.figure(figsize=(8, 4))
        plt.plot(normalized_df['n_clusters'], normalized_df['combined_score'], 'o-', linewidth=2)
        
        best_idx = normalized_df['combined_score'].idxmax()
        best_k = normalized_df.loc[best_idx, 'n_clusters']
        best_score = normalized_df.loc[best_idx, 'combined_score']
        
        plt.scatter([best_k], [best_score], c='red', s=100, label=f'Optimum: k={best_k}')
        plt.grid(True)
        plt.xlabel('Number of clusters (k)')
        plt.title('Average grade based on all the normalized metrics')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.show()
        
        return best_k

    # ==================================Configuración y parámetros

    # Pre-default Metrics
    default_metrics = {
        "manhattan_distance": "Minimize",
        "silhouette": "Maximize",
        "davies_bouldin": "Minimize",
        "calinski_harabasz": "Maximize",
        "ipvc": "Minimize",
        "shannon_entropy": "Minimize"
    }
    
    # Use provided metrics or default ones
    metrics_used = metrics_det if metrics_det is not None else default_metrics

    # ============================================ Ejecución del análisis de clustering

    # Mostrar información sobre el proceso
    print(f"Evaluating clusters in the range: {min(rangek)} to {max(rangek)}")
    print(f"Number of data points: {all_pattern2D.shape[0]}")
    print(f"Dimensions: {all_pattern2D.shape[1]}")

    results = Parallel(n_jobs=-1)(delayed(evaluate_kmeans)(k, all_pattern2D) for k in rangek)
    results_dict = {k: metrics for k, metrics in results}

    # DataFrame
    df_results = pd.DataFrame.from_dict(results_dict, orient='index')
    df_results.index.name = 'n_clusters'
    df_results = df_results.reset_index()

    # Results
    print("\nEvaluation of Clusters:")
    print(df_results)

    # ============================== Analyze Results

    best_clusters = {}
    for metric, objective in metrics_used.items():
        if objective == "Minimize":
            best_k = df_results.loc[df_results[metric].idxmin(), 'n_clusters']
            best_value = df_results[metric].min()
        else:  # Maximize
            best_k = df_results.loc[df_results[metric].idxmax(), 'n_clusters']
            best_value = df_results[metric].max()
        best_clusters[metric] = (best_k, best_value)

    print("\nBest number of cluster per metric:")
    for met, (best_k, best_val) in best_clusters.items():
        print(f"- {met}: k={best_k} (value={best_val:.4f})")

    # ================================== Visualizations
    visualize_results(df_results, metrics_used)
    optimal_k = calculate_optimal_k(df_results, metrics_used)
    print(f"\nOptimum value of k based on all of the metrics: {optimal_k}")
    
    return optimal_k