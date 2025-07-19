import numpy as np

class Node:
    """
    A class representing a node in the decision tree.

    Attributes:
    -----------
    left : Node or None
        Left child of the current node.
    right : Node or None
        Right child of the current node.
    feature : int or None
        The index of the feature used for splitting at this node.
    trh : float or None
        The threshold value used for splitting.
    gain : float or None
        The information gain from the split.
    value : int or None
        Class label if the node is a leaf.
    """
    def __init__(self, left=None, right=None, feature=None, trh=None, gain=None, value=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.trh = trh
        self.gain = gain
        self.value = value

class DecisionTree:
    """
    A simple decision tree classifier using entropy and information gain.

    Parameters:
    -----------
    max_depth : int
        The maximum depth of the decision tree.
    min_sample_split : int
        The minimum number of samples required to split an internal node.
    """
    def __init__(self, max_depth, min_sample_split):
        self.root = None
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.

        Parameters:
        -----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Training label vector of shape (n_samples,).
        """
        self.root = self._train(X, y, curr_depth=0)

    def _train(self, X, y, curr_depth):
        """
        Recursively build the decision tree.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix for the current node.
        y : np.ndarray
            Label vector for the current node.
        curr_depth : int
            Current depth of the node in the tree.

        Returns:
        --------
        Node
            The root of the (sub)tree built from the data.
        """
        num_samples, num_features = X.shape
        best_gain = -float('inf')
        best_split = {}

        if num_samples >= self.min_sample_split and curr_depth <= self.max_depth:
            for feature_idx in range(num_features):
                feature_values = X[:, feature_idx]
                possible_thresholds = np.unique(feature_values)

                for threshold in possible_thresholds:
                    left_indices = feature_values < threshold
                    right_indices = feature_values > threshold
                    left_y, right_y = y[left_indices], y[right_indices]

                    if len(left_y) > 0 and len(right_y) > 0:
                        gain = self._information_gain(y, left_y, right_y)

                        if gain > best_gain:
                            best_gain = gain
                            best_split = {
                                "feature": feature_idx,
                                "threshold": threshold,
                                "gain": gain,
                                "left_X": X[left_indices],
                                "left_y": left_y,
                                "right_X": X[right_indices],
                                "right_y": right_y
                            }

        if best_gain > 0:
            left_subtree = self._train(best_split["left_X"], best_split["left_y"], curr_depth + 1)
            right_subtree = self._train(best_split["right_X"], best_split["right_y"], curr_depth + 1)
            return Node(
                left=left_subtree,
                right=right_subtree,
                feature=best_split["feature"],
                trh=best_split["threshold"],
                gain=best_split["gain"]
            )
        else:
            # Create a leaf node with the most common class label
            return Node(value=np.bincount(y).argmax())

    def _entropy(self, y):
        """
        Calculate the entropy of a label distribution.

        Parameters:
        -----------
        y : np.ndarray
            Label vector.

        Returns:
        --------
        float
            The entropy value.
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _information_gain(self, parent_y, left_y, right_y):
        """
        Compute the information gain from a proposed split.

        Parameters:
        -----------
        parent_y : np.ndarray
            Label vector before the split.
        left_y : np.ndarray
            Label vector of the left split.
        right_y : np.ndarray
            Label vector of the right split.

        Returns:
        --------
        float
            The information gain achieved by the split.
        """
        weight_left = len(left_y) / len(parent_y)
        weight_right = len(right_y) / len(parent_y)
        return self._entropy(parent_y) - (weight_left * self._entropy(left_y) + weight_right * self._entropy(right_y))

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns:
        --------
        np.ndarray
            Predicted class labels.
        """
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, x, tree):
        """
        Recursively traverse the tree to predict a single sample.

        Parameters:
        -----------
        x : np.ndarray
            Feature vector of a single sample.
        tree : Node
            The current node in the tree.

        Returns:
        --------
        int
            Predicted class label.
        """
        if tree.value is not None:
            return tree.value
        if x[tree.feature] < tree.trh:
            return self._predict_sample(x, tree.left)
        else:
            return self._predict_sample(x, tree.right)
