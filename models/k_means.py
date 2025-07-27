import numpy as np

class KMeans:
    def __init__(self, k):
        """
        Initializes the KMeans clustering object.

        Parameters:
        - k (int): Number of clusters
        """
        self.k = k
        self.centroids = None

    def fit(self, x, max_iter=100):
        """
        Computes KMeans clustering on the input data.

        Parameters:
        - x (ndarray): Input data of shape (n_samples, n_features)
        - max_iter (int): Maximum number of iterations
        """
        n_samples, n_features = x.shape
        centroids_index = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = x[centroids_index]

        for iteration in range(max_iter):
            labels = np.zeros(n_samples, dtype=int)
            for i, x_i in enumerate(x):
                distances = np.linalg.norm(self.centroids - x_i, axis=1)
                labels[i] = np.argmin(distances)

            # Update centroids
            new_centroids = np.array([
                x[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.k)
            ])

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, x):
        """
        Predicts the closest cluster index for each data point.

        Parameters:
        - x (ndarray): Input data

        Returns:
        - ndarray: Cluster labels for each sample
        """
        labels = np.zeros(x.shape[0], dtype=int)
        for i, x_i in enumerate(x):
            distances = np.linalg.norm(self.centroids - x_i, axis=1)
            labels[i] = np.argmin(distances)
        return labels
