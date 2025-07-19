import numpy as np

class KNearestNeighbor:
    """
    A simple K-Nearest Neighbors (KNN) regressor/classifier.

    Parameters:
    -----------
    k : int
        Number of nearest neighbors to consider.
    """
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.n_samples = 0
        self.n_features = 0

    def fit(self, X, y):
        """
        Store the training data.

        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target labels of shape (n_samples,).
        """
        self.X_train = X
        self.y_train = y
        self.n_samples, self.n_features = self.X_train.shape

    def predict(self, X):
        """
        Predict the output for the input data using KNN.

        Parameters:
        -----------
        X : np.ndarray
            Test data of shape (n_samples, n_features).

        Returns:
        --------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        n_test_samples = X.shape[0]
        y_pred = np.zeros(n_test_samples)

        for i in range(n_test_samples):
            # Compute Euclidean distances to all training samples
            distances = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))
            # Find the indices of the k smallest distances
            k_indices = np.argsort(distances)[:self.k]
            # Retrieve the corresponding labels
            k_neighbor_labels = self.y_train[k_indices]
            # Use majority vote (for classification) or mean (for regression)
            y_pred[i] = np.mean(k_neighbor_labels)  # Change to np.bincount().argmax() for classification

        return y_pred
