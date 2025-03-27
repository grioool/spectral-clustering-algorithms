import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, load_iris


class BenchmarkDatasetLoader:

    def __init__(self,
                 wut_path="data/wut",
                 uci_path="data/uci",
                 sipu_path="data/sipu",
                 mnist_path="data/mnist"):
        self.wut_path = wut_path
        self.uci_path = uci_path
        self.sipu_path = sipu_path
        self.mnist_path = mnist_path

    def generate_synthetic_data(self, n_samples=100, n_clusters=3,
                                centers=None, cluster_std=1.0,
                                random_state=None):
        if centers is not None:
            n_clusters = centers.shape[0]
        X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2,
                          cluster_std=cluster_std, random_state=random_state)
        return X, y

    def load_wut_dataset_csv(self, dataset_name):
        data_file = os.path.join(self.wut_path, f"{dataset_name}.data.csv")
        labels_file = os.path.join(self.wut_path, f"{dataset_name}.labels0.csv")

        try:
            df_data = pd.read_csv(data_file, header=None, sep=r"\s+")
        except Exception as e:
            raise IOError(f"Error loading {data_file}: {e}")
        X = df_data.values.astype(float)

        try:
            df_labels = pd.read_csv(labels_file, header=None, sep=r"\s+")
        except Exception as e:
            raise IOError(f"Error loading {labels_file}: {e}")
        y = df_labels.values.ravel().astype(int)

        return X, y

    def load_uci_dataset(self, dataset_name):
        data_file = os.path.join(self.uci_path, f"{dataset_name}.data.csv")
        labels_file = os.path.join(self.uci_path, f"{dataset_name}.labels0.csv")

        try:
            df_data = pd.read_csv(data_file, header=None, sep=r"\s+")
        except Exception as e:
            raise IOError(f"Error loading {data_file}: {e}")
        X = df_data.values.astype(float)

        try:
            df_labels = pd.read_csv(labels_file, header=None, sep=r"\s+")
        except Exception as e:
            raise IOError(f"Error loading {labels_file}: {e}")
        y = df_labels.values.ravel().astype(int)

        return X, y

    def load_sipu_dataset(self, dataset_name):
        data_file = os.path.join(self.sipu_path, f"{dataset_name}.data.csv")
        labels_file = os.path.join(self.sipu_path, f"{dataset_name}.labels0.csv")

        try:
            df_data = pd.read_csv(data_file, header=None, sep=r"\s+")
        except Exception as e:
            raise IOError(f"Error loading {data_file}: {e}")
        X = df_data.values.astype(float)

        try:
            df_labels = pd.read_csv(labels_file, header=None, sep=r"\s+")
        except Exception as e:
            raise IOError(f"Error loading {labels_file}: {e}")
        y = df_labels.values.ravel().astype(int)

        return X, y

    def load_mnist_dataset(self, dataset_name):
        data_file = os.path.join(self.mnist_path, f"{dataset_name}.data.csv")
        labels_file = os.path.join(self.mnist_path, f"{dataset_name}.labels0.csv")
        if not os.path.exists(data_file) or not os.path.exists(labels_file):
            raise IOError(f"MNIST files not found for {dataset_name} in {self.mnist_path}")

        df_data = pd.read_csv(data_file, header=None, sep=r"\s+")
        X = df_data.values.astype(float)

        df_labels = pd.read_csv(labels_file, header=None, sep=r"\s+")
        y = df_labels.values.ravel().astype(int)

        return X, y

    def load_csv_dataset(self, csv_path):
        df = pd.read_csv(csv_path)
        data = df.values
        if np.issubdtype(data[:, -1].dtype, np.number):
            X = data[:, :-1].astype(float)
            y = data[:, -1].astype(int)
        else:
            X = data[:, :-1].astype(float)
            y = None
        return X, y

    def load_benchmark_data(self, dataset_name):
        if dataset_name == "synthetic1":
            return self.generate_synthetic_data(n_samples=200, n_clusters=3, random_state=0)
        elif dataset_name == "synthetic2":
            centers = np.array([[1, 1], [2, 2], [1.5, 1.5],
                                [5, 5], [6, 6], [5.5, 5.5]])
            return self.generate_synthetic_data(n_samples=300, centers=centers,
                                                cluster_std=0.8, random_state=1)
        elif dataset_name == "iris":
            iris = load_iris()
            X = iris.data[:, :2]
            y = iris.target
            return X, y
        elif dataset_name.endswith(".csv"):
            return self.load_csv_dataset(dataset_name)
        elif dataset_name in [
            "ecoli", "glass", "ionosphere", "sonar", "statlog",
            "wdbc", "yeast", "wine"
        ]:
            return self.load_uci_dataset(dataset_name)
        elif dataset_name in ["s1", "s2", "a1", "a2", "a3", "unbalance", "birch1", "birch2", "aggregation", "worms_2",
                              "worms_64", "spiral"]:
            return self.load_sipu_dataset(dataset_name)
        elif dataset_name in ["digits", "fashion"]:
            return self.load_mnist_dataset(dataset_name)
        else:
            return self.load_wut_dataset_csv(dataset_name)
