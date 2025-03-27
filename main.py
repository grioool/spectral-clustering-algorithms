import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from dataset_loader import BenchmarkDatasetLoader
from plotter import ResultsPlotter
from spectral_clustering import visualize_dataset, evaluate_clustering_algorithms_for_dataset, \
    kernel_spectral_clustering, mncut_spectral_clustering, landmark_spectral_clustering, normalized_spectral_clustering

# Ignore user warnings ("Graph is not fully connected")
warnings.filterwarnings("ignore", category=UserWarning)
# Ignore runtime warnings (overflow, invalid value encountered)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Ignore convergence warnings from scikit-learn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":
    loader = BenchmarkDatasetLoader(
        wut_path="data/wut",
        uci_path="data/uci",
        sipu_path="data/sipu",
        mnist_path="data/mnist"
    )
    dataset_names = [
        "synthetic1",
        "synthetic2",
        "iris",
        "circles",
        "cross",
        "graph",
        "mk1",
        "mk2",
        "mk3",
        "trapped_lovers",
        "twosplashes",
        "smile",

        "ecoli",
        "glass",
        "ionosphere",
        "sonar",
        "wdbc",
        "wine",
        "yeast",

        "a1",
        "a2",
        "aggregation",
        "spiral",
        # "digits",
        # "fashion"
    ]

    overall_results = {}
    algorithms = {
        "Normalized SC": normalized_spectral_clustering,
        "Landmark SC": landmark_spectral_clustering,
        "MNCut SC": mncut_spectral_clustering,
        "Kernel SC": kernel_spectral_clustering,
    }

    for name in dataset_names:
        print(f"\nProcessing dataset: {name}")
        try:
            X, y = loader.load_benchmark_data(name)
        except Exception as e:
            print(f"Failed to load dataset '{name}': {e}")
            continue

        if X is None or y is None:
            print(f"Skipping '{name}' due to missing data or labels.")
            continue

        results = evaluate_clustering_algorithms_for_dataset(X, y, algorithms, n_runs=5, base_random_state=42)
        overall_results[name] = results

        print(f"Results for dataset: {name}")
        for algo_name, res in results.items():
            mean_aaa = np.mean(res['aaa_scores'])
            std_aaa = np.std(res['aaa_scores'])
            mean_time = np.mean(res['times'])
            print(f"  {algo_name}: Mean AAA = {mean_aaa:.4f}, Std AAA = {std_aaa:.4f}, Avg Time = {mean_time:.4f} sec")

        visualize_dataset(name, X, y, algorithms, random_state=42)

    with open("output/clustering_evaluation_results.json", "w") as f:
        json.dump(overall_results, f, indent=4)
    print("\nSaved results to clustering_evaluation_results.json")

    plotter = ResultsPlotter(overall_results, algorithms)
    plotter.plot_all_domains()
