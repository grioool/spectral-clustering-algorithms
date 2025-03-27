import numpy as np
from matplotlib import pyplot as plt

"""
  - Box plots of AAA scores (for all runs and datasets in a domain)
  - Bar charts of maximum average AAA per dataset.
"""


class ResultsPlotter:
    SIPU_DATASETS = {"a1", "a2", "aggregation", "spiral"}
    UCI_DATASETS = {"ecoli", "glass", "ionosphere", "sonar", "wdbc", "wine", "yeast"}
    MNIST_DATASETS = {"digits", "fashion"}
    WUT_DATASETS = {"circles", "cross", "graph", "mk1", "mk2", "mk3", "smile", "trapped_lovers", "twosplashes"}
    ALL_DATASETS = {"synthetic1", "synthetic2", "iris", "circles", "cross", "graph", "mk1", "mk2", "mk3",
                    "trapped_lovers", "twosplashes", "smile", "ecoli", "glass", "ionosphere", "sonar", "wdbc", "wine",
                    "yeast", "a1", "a2", "aggregation", "spiral", "digits", "fashion"}

    def __init__(self, overall_results, algorithms):
        self.overall_results = overall_results
        self.algorithms = algorithms

    def plot_boxplot_for_domain(self, domain_name, dataset_list):
        algo_scores = {algo: [] for algo in self.algorithms.keys()}

        for ds in dataset_list:
            if ds not in self.overall_results:
                continue
            for algo_name, res in self.overall_results[ds].items():
                if algo_name in algo_scores:
                    algo_scores[algo_name].extend(res["aaa_scores"])

        labels = list(algo_scores.keys())
        data_to_plot = [algo_scores[algo] for algo in labels]

        if all(len(scores) == 0 for scores in data_to_plot):
            print(f"No AAA scores found for domain '{domain_name}'. Skipping box plot.")
            return

        plt.figure(figsize=(8, 5))
        plt.boxplot(data_to_plot, tick_labels=labels, showmeans=True)
        plt.title(f"AAA Distribution - {domain_name} Domain")
        plt.ylabel("AAA (Adjusted Rand Index)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_max_aaa_for_domain(self, domain_name, dataset_list):
        datasets = []
        max_aaa_values = []

        for ds in dataset_list:
            if ds not in self.overall_results:
                continue
            best_avg = 0.0
            for algo_name, res in self.overall_results[ds].items():
                aaa_scores = res["aaa_scores"]
                if len(aaa_scores) > 0:
                    avg_aaa = np.mean(aaa_scores)
                    if avg_aaa > best_avg:
                        best_avg = avg_aaa
            datasets.append(ds)
            max_aaa_values.append(best_avg)

        if not datasets:
            print(f"No datasets found in domain '{domain_name}'. Skipping bar chart.")
            return

        plt.figure(figsize=(8, 5))
        plt.bar(datasets, max_aaa_values, color="skyblue")
        plt.title(f"Max Average AAA by Dataset - {domain_name} Domain")
        plt.ylabel("Max Average AAA (Adjusted Rand Index)")
        plt.xticks(rotation=45)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()

    def plot_all_domains(self):
        domains = [
            ("WUT", self.WUT_DATASETS),
            ("SIPU", self.SIPU_DATASETS),
            ("UCI", self.UCI_DATASETS),
            ("MNIST", self.MNIST_DATASETS),
            ("ALL", self.ALL_DATASETS)
        ]
        for domain_name, ds_list in domains:
            print(f"\nPlotting for domain: {domain_name}")
            self.plot_boxplot_for_domain(domain_name, ds_list)
            self.plot_max_aaa_for_domain(domain_name, ds_list)
