# Spectral Clustering Algorithms Implementation

## Normalized Spectral Clustering

- **Affinity Matrix:**  
  Uses an RBF (Gaussian) kernel to compute pairwise similarities.  
  ```python
  affinity_matrix = rbf_kernel(X, gamma=gamma)
  ```

- **Clustering:**  
  Calls scikit-learn’s `SpectralClustering` with a precomputed affinity matrix.  
  ```python
  sc = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity='precomputed')
  return sc.fit_predict(affinity_matrix)
  ```

- **Key Idea:**  
  Leverages a fully-connected similarity graph to capture non-linear relationships.

---

## Landmark Spectral Clustering

- **Landmark Selection:**  
  Randomly picks a subset of points (landmarks) to approximate the full dataset.  
  ```python
  indices = np.random.choice(X.shape[0], n_landmarks, replace=False)
  landmarks = X[indices]
  ```

- **Similarity Computation:**  
  Computes similarities between all points and landmarks, and among landmarks.  
  ```python
  sim_all = np.exp(-np.sum((X[:, None, :] - landmarks[None, :, :])**2, axis=2) / (2 * sigma**2))
  sim_land = np.exp(-np.sum((landmarks[:, None, :] - landmarks[None, :, :])**2, axis=2) / (2 * sigma**2))
  ```

- **Laplacian on Landmarks:**  
  Constructs the Laplacian and performs eigendecomposition.  
  ```python
  D_land = np.diag(np.sum(sim_land, axis=1))
  L_land = D_land - sim_land
  e_vals, e_vecs = eigh(L_land)
  e_vecs = e_vecs[:, :n_clusters]
  ```

- **Embedding & Clustering:**  
  Embeds all data using the landmark eigenvectors and applies k-means.  
  ```python
  embedding = sim_all @ e_vecs
  km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
  return km.fit_predict(embedding)
  ```

- **Key Idea:**  
  Reduces computation by using a compressed representation (landmarks) of the data.

---

## MNCut Spectral Clustering

- **Graph Construction:**  
  Builds a k-nearest neighbors (KNN) graph to form the connectivity matrix.  
  ```python
  knn_graph = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
  W = knn_graph.toarray()
  ```

- **Laplacian Formation:**  
  Computes the unnormalized Laplacian \( L = D - W \) and attempts normalization.  
  ```python
  D = np.diag(np.sum(W, axis=1))
  L = D - W
  D_inv = np.linalg.inv(D)  # if possible, else fallback to L
  L_norm = D_inv @ L
  ```

- **Eigendecomposition & Clustering:**  
  Sorts eigenvalues, selects the eigenvectors for the smallest ones, and applies k-means.  
  ```python
  eigenvalues, eigenvectors = np.linalg.eig(L_norm)
  idx = np.argsort(np.real(eigenvalues))
  V = np.real(eigenvectors)[:, idx[:n_clusters]]
  km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
  return km.fit_predict(V)
  ```

- **Key Idea:**  
  Minimizes the normalized cut by using the spectrum of the graph Laplacian.

---

## Kernel Spectral Clustering

- **Kernel Matrix:**  
  Computes a kernel matrix using a specified kernel (RBF, polynomial, or Laplacian).  
  ```python
  if kernel == 'rbf':
      S = rbf_kernel(X, gamma=gamma)
  elif kernel == 'polynomial':
      S = polynomial_kernel(X, degree=3)
  elif kernel == 'laplacian':
      S = laplacian_kernel(X, gamma=gamma)
  else:
      raise ValueError(f"Kernel '{kernel}' not supported.")
  ```

- **Laplacian & Normalization:**  
  Constructs the degree matrix and computes the normalized Laplacian.  
  ```python
  D = np.diag(np.sum(S, axis=1))
  D_inv_sqrt = np.linalg.inv(np.sqrt(D))
  L_norm = D_inv_sqrt @ (D - S) @ D_inv_sqrt
  ```

- **Eigenvector Computation & Row Normalization:**  
  Uses `eigh` to obtain eigenvectors and normalizes each row.  
  ```python
  eigenvalues, eigenvectors = eigh(L_norm)
  eigenvectors = eigenvectors[:, :n_clusters]
  for i in range(eigenvectors.shape[0]):
      if np.linalg.norm(eigenvectors[i]) != 0:
          eigenvectors[i] /= np.linalg.norm(eigenvectors[i])
  ```

- **Clustering:**  
  Final clustering is performed using k-means on the normalized eigenvector space.  
  ```python
  km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
  return km.fit_predict(eigenvectors)
  ```

- **Key Idea:**  
  Uses kernel functions to transform data into a space where clusters are more easily separable.