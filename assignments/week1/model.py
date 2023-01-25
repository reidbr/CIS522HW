import numpy as np


class LinearRegression:
    """
    A linear regression model that uses the closed form solution to  fit the model
    """
    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate the closed form solution for the Linear Regression model

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): Th target

        Returns:
            None

        """
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = np.mean(y) - np.mean(X, axis=0).dot(self.w) 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Calculate the fir for the Linear Regression model using gradient descent

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The target
            lr (float): The learning rate
            epochs (int): The number of loops to run for gradient descent

        Returns:
            None

        """
        self.w = np.zeros(X.shape[1])
        self.b = 0
        n = len(X)
        for _ in range(epochs):
            y_pred = X @ self.w + self.b
            gradient_weight = (-2/n) * X.T @ (y - y_pred)
            gradient_bias = (-2/n) * np.sum(y - y_pred)
            self.w -= lr * gradient_weight
            self.b -= lr * gradient_bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b
