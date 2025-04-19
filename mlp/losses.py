from abc import abstractmethod

import numpy as np


class Loss:
    """
    Class to be inherited by loss functions.
    """

    @abstractmethod
    def loss(self, y, y_pred):
        """
        Returns:
            loss_value
        """
        pass

    @abstractmethod
    def dloss(self, y, y_pred):
        """
        Method to compute the gradient of the loss function.

        Returns:
            loss_gradient
        """
        pass


class MSELoss(Loss):
    def loss(self, y, y_pred):
        return np.mean(np.power(y - y_pred, 2)) / 2

    def dloss(self, y, y_pred):
        return y - y_pred


class BCELoss(Loss):
    """Binary cross entropy loss function following."""

    def __init__(self, eps=1e-15):
        self.eps = eps

    def loss(self, y, y_pred):
        # Ensure predicted is a single probability value
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return np.mean(-(y * np.log(y_pred + self.eps) + (1 - y) * np.log(1 - y_pred + self.eps)))

    def dloss(self, y, y_pred):
        # Gradient of the loss with respect to the predicted value
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return -(y / y_pred) + (1 - y) / (1 - y_pred)


class CrossEntropyLoss(Loss):
    def __init__(self, eps=1e-15):
        self.eps = eps

    def loss(self, y, y_pred):
        # predicted should be a probability distribution (output of softmax)
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return 1 / y_pred.shape[0] * -np.sum(y * np.log(y_pred))

    def dloss(self, y, y_pred):
        # Gradient of the loss with respect to the predicted values
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return y_pred - y
