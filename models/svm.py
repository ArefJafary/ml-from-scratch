import numpy as np

class SVM:
    def __init__(self, lr, lamb, n_iteration):
        """
        lr: Learning rate
        lamb: Regularization parameter (lambda)
        n_iteration: Number of training iterations
        """
        self.lr = lr
        self.lamb = lamb
        self.n_iteration = n_iteration
        self.w = None
        self.b = None

    def train(self, x, y):
        """
        Train the SVM using gradient descent.

        x: Features (numpy array of shape [n_samples, n_features])
        y: Labels (numpy array of shape [n_samples])
        """
        self.w = np.zeros(x.shape[1])
        self.b = 0
        labels = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iteration):
            for i, x_i in enumerate(x):
                condition = labels[i] * (np.dot(x_i, self.w) - self.b)
                if condition >= 1:
                    dw = 2 * self.lamb * self.w
                    db = 0
                else:
                    dw = 2 * self.lamb * self.w - labels[i] * x_i
                    db = -labels[i]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, x):
        """
        Predict labels for input features.

        x: Features (numpy array of shape [n_samples, n_features])
        Returns: Predicted labels (-1 or 1)
        """
        approx = np.dot(x, self.w) - self.b
        return np.sign(approx)
