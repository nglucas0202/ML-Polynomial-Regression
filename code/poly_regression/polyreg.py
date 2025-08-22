"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore

        # You can add additional fields
        self.mu_: np.ndarray = None     # feature means for standardization
        self.sigma_: np.ndarray = None  # feature std for standardization

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert X.shape[1] == 1, "polyfeatures expects (n,1) input."

        n = X.shape[0]
        return np.hstack([X ** p for p in range(1, degree + 1)])
    
    def _standardize(self, Phi: np.ndarray, is_fit: bool) -> np.ndarray:
        """Standardize columns of Phi. If is_fit, compute and store mean/std."""
        eps = 1e-12
        if is_fit:
            self.mu_ = Phi.mean(axis=0)
            self.sigma_ = Phi.std(axis=0, ddof=0)
            self.sigma_ = np.where(self.sigma_ < eps, 1.0, self.sigma_)
        return (Phi - self.mu_) / self.sigma_

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # polynomial expansion
        Phi = self.polyfeatures(X, self.degree)
        # standardize
        Phi_std = self._standardize(Phi, is_fit=True)
        # add bias column
        n = Phi_std.shape[0]
        Z = np.hstack([np.ones((n, 1)), Phi_std])

        # closed-form ridge regression
        d_plus_1 = Z.shape[1]
        L = np.eye(d_plus_1)
        L[0, 0] = 0.0  # no penalty on bias

        A = Z.T @ Z + self.reg_lambda * L
        b = Z.T @ y
        self.weight = np.linalg.solve(A, b)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        assert self.weight is not None, "Model not trained yet."

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Phi = self.polyfeatures(X, self.degree)
        Phi_std = self._standardize(Phi, is_fit=False)
        n = Phi_std.shape[0]
        Z = np.hstack([np.ones((n, 1)), Phi_std])

        return Z @ self.weight


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    return float(np.mean((a - b) ** 2))


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    # loop i from 1 to n-1
    for i in range(1, n):
        Xtrain_i = Xtrain[:i+1]
        Ytrain_i = Ytrain[:i+1]

        # obtain the weight using the training set
        model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(Xtrain_i, Ytrain_i)

        # calculate the prediction from training data
        predict_Ytrain_i = model.predict(Xtrain_i)

        # calculate the prediction from testing data
        Xtest_i = Xtest[:i+1]
        Ytest_i = Ytest[:i+1]
        predict_Ytest_i = model.predict(Xtest_i)

        # Fill in errorTrain and errorTest arrays
        errorTrain[i] = mean_squared_error(predict_Ytrain_i, Ytrain_i)
        errorTest[i] = mean_squared_error(predict_Ytest_i, Ytest_i)

    return [errorTrain, errorTest]
