import numpy as np

class LinearRegression:
    def __init__(self):
        """
        Initializes the LinearRegression object with default parameters.
        """
        self.lr = None       # Learning rate
        self.landa = None    # Regularization strength (lambda)
        self.m = None        # Weight vector
        self.h = None        # Bias term

    def fit(self, x, y, lr=0.0001, landa=0.01, max_iter=1000):
        """
        Trains the linear regression model using gradient descent with L2 regularization.

        Parameters:
        - x (ndarray): Input features of shape (n_samples, n_features)
        - y (ndarray): Target values of shape (n_samples,)
        - lr (float): Learning rate for gradient descent
        - landa (float): Regularization parameter (L2)
        - max_iter (int): Number of iterations for training
        """

        self.lr = lr
        self.landa = landa
        n, n_features = x.shape
        self.m = np.zeros(n_features)
        self.h = 0

        for i in range(max_iter):
            y_pred = self.predict(x)
            error = y_pred - y

            # Gradients
            dm = (2 / n) * (np.dot(x.T, error) + self.landa * self.m)
            dh = (2 / n) * np.sum(error)

            # Update
            self.m -= self.lr * dm
            self.h -= self.lr * dh

            if np.isnan(self.m).any() or np.isinf(self.m).any():
                print("NaN or Inf in parameters. Stopping early.")
                break

            if i % 100 == 0 or i == max_iter - 1:
                loss = self.loss_function(x, y)
                print(f"Iteration {i}: Loss = {loss:.4f}")


    def loss_function(self, x, y):
        """
        Computes the L2-regularized mean squared error loss.

        Parameters:
        - x (ndarray): Input features
        - y (ndarray): True target values

        Returns:
        - float: Regularized loss value
        """
        y_pred = self.predict(x)
        mse = np.mean((y_pred - y) ** 2)
        reg = self.landa * np.sum(self.m ** 2)
        return mse + reg

    def predict(self, x):
        """
        Makes predictions using the trained linear regression model.

        Parameters:
        - x (ndarray): Input features

        Returns:
        - ndarray: Predicted target values
        """
        return np.dot(x, self.m) + self.h
