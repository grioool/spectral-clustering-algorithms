import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph


def normalized_spectral_clustering(X, n_clusters, random_state=None, gamma=0.5):
    """
    Uses an RBF kernel to create a fully-connected affinity matrix and performs spectral clustering.
    """
    affinity_matrix = rbf_kernel(X, gamma=gamma)
    sc = SpectralClustering(
        n_clusters=n_clusters,
        random_state=random_state,
        affinity='precomputed'
    )
    return sc.fit_predict(affinity_matrix)


def landmark_spectral_clustering(X, n_clusters, n_landmarks=50, random_state=None):
    if n_landmarks >= X.shape[0]:
        warnings.warn("n_landmarks >= number of data points. Using all data points as landmarks.")
        landmarks = X
    else:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.choice(X.shape[0], n_landmarks, replace=False)
        landmarks = X[indices]

    sigma = 1.0
    sim_all = np.exp(-np.sum((X[:, None, :] - landmarks[None, :, :]) ** 2, axis=2) / (2 * sigma ** 2))
    sim_land = np.exp(-np.sum((landmarks[:, None, :] - landmarks[None, :, :]) ** 2, axis=2) / (2 * sigma ** 2))
    D_land = np.diag(np.sum(sim_land, axis=1))
    L_land = D_land - sim_land
    try:
        D_inv_sqrt = np.linalg.inv(np.sqrt(D_land))
        L_norm = D_inv_sqrt @ L_land @ D_inv_sqrt
    except np.linalg.LinAlgError:
        L_norm = L_land

    e_vals, e_vecs = eigh(L_norm)
    e_vecs = e_vecs[:, :n_clusters]
    embedding = sim_all @ e_vecs
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(embedding)


def mncut_spectral_clustering(X, n_clusters, random_state=None):
    knn_graph = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
    W = knn_graph.toarray()
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    try:
        D_inv = np.linalg.inv(D)
        L_norm = D_inv @ L
    except np.linalg.LinAlgError:
        L_norm = L
    eigenvalues, eigenvectors = np.linalg.eig(L_norm)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    idx = np.argsort(eigenvalues)
    V = eigenvectors[:, idx[:n_clusters]]
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(V)


def kernel_spectral_clustering(X, n_clusters, kernel='rbf', gamma=1.0, random_state=None):
    from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel
    if kernel == 'rbf':
        S = rbf_kernel(X, gamma=gamma)
    elif kernel == 'polynomial':
        S = polynomial_kernel(X, degree=3)
    elif kernel == 'laplacian':
        S = laplacian_kernel(X, gamma=gamma)
    else:
        raise ValueError(f"Kernel '{kernel}' not supported.")

    D = np.diag(np.sum(S, axis=1))
    try:
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        L_norm = D_inv_sqrt @ (D - S) @ D_inv_sqrt
    except np.linalg.LinAlgError:
        L_norm = D - S

    eigenvalues, eigenvectors = eigh(L_norm)
    eigenvectors = eigenvectors[:, :n_clusters]
    for i in range(eigenvectors.shape[0]):
        norm_val = np.linalg.norm(eigenvectors[i])
        if norm_val != 0:
            eigenvectors[i] = eigenvectors[i] / norm_val
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(eigenvectors)


def adjusted_asymmetric_accuracy(labels_true, labels_pred):
    """
    Computes the Adjusted Asymmetric Accuracy (AAA).
    Here, we use adjusted Rand index from scikit-learn as a proxy.
    """
    n = len(labels_true)
    if n != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have the same length.")

    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)

    true_map = {label: i for i, label in enumerate(unique_true)}
    pred_map = {label: j for j, label in enumerate(unique_pred)}

    remapped_true = np.array([true_map[label] for label in labels_true])
    remapped_pred = np.array([pred_map[label] for label in labels_pred])

    return adjusted_rand_score(remapped_true, remapped_pred)


def evaluate_clustering_algorithms_for_dataset(X, y_true, algorithms, n_runs=5, base_random_state=42):
    results = {}
    for algo_name, algo_func in algorithms.items():
        print(f"  Algorithm: {algo_name}")
        algo_results = {'aaa_scores': [], 'times': []}
        for run in range(n_runs):
            current_seed = None if base_random_state is None else base_random_state + run
            start = time.time()
            try:
                n_clusters = len(np.unique(y_true)) if y_true is not None else 3
                y_pred = algo_func(X, n_clusters=n_clusters, random_state=current_seed)
            except Exception as e:
                print(f"    Run {run + 1} failed: {e}")
                y_pred = None
            elapsed = time.time() - start
            if y_pred is not None and y_true is not None:
                if 0 in np.unique(y_true):
                    mask = y_true != 0
                    score = adjusted_asymmetric_accuracy(y_true[mask], y_pred[mask])
                else:
                    score = adjusted_asymmetric_accuracy(y_true, y_pred)
            else:
                score = 0.0
            algo_results['aaa_scores'].append(score)
            algo_results['times'].append(elapsed)
        results[algo_name] = algo_results
    return results


def visualize_dataset(dname, X, y_true, algorithms, random_state=42):
    """
    Visualizes the dataset using a scatter plot. If the dataset is not 2D,
    PCA is used to reduce the dimensionality to 2 for visualization.
    """
    if X.shape[1] != 2:
        pca = PCA(n_components=2, random_state=random_state)
        X_vis = pca.fit_transform(X)
        print(f"Visualizing dataset '{dname}' using PCA for dimensionality reduction.")
    else:
        X_vis = X

    n_subplots = 1 + len(algorithms)
    plt.figure(figsize=(5 * n_subplots, 4))

    # Ground truth subplot
    plt.subplot(1, n_subplots, 1)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_true, cmap='viridis', s=30)
    plt.title(f"{dname}\nGround Truth")

    idx_subplot = 2
    for algo_name, algo_func in algorithms.items():
        plt.subplot(1, n_subplots, idx_subplot)
        idx_subplot += 1
        try:
            n_clusters = len(np.unique(y_true))
            y_pred = algo_func(X, n_clusters=n_clusters, random_state=random_state)
            plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred, cmap='viridis', s=30)
            plt.title(f"{algo_name}")
        except Exception as e:
            print(f"Failed to cluster '{dname}' with '{algo_name}': {e}")
            plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

    plt.tight_layout()
    save_path = f"{dname}_all_algos.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Saved visualization for {dname} to {save_path}")
