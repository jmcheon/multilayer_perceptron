from abc import abstractmethod
import numpy as np

class Loss:
    """
    Class to be inherited by loss functions.
    """
    @abstractmethod
    def loss(self, y_pred, y):
        pass

    @abstractmethod
    def dloss(self, y_pred, y):
        pass

class MSELoss(Loss):
    def loss(self, y, y_pred):
        return np.mean(np.power(y - y_pred, 2)) / 2

    def dloss(self, y, y_pred):
        return (y - y_pred)

class CrossEntropyLoss(Loss):
    def loss(self, y_pred, y):
        # predicted should be a probability distribution (output of softmax)
        return -np.mean(y * np.log(y_pred))

    def dloss(self, y_pred, y):
        # Gradient of the loss with respect to the predicted values
        return y_pred - y 

class BCELoss(Loss):
    """Binary cross entropy loss function following."""
    def __init__(self, eps=1e-15):
        self.eps = eps
        
    def loss(self, y_pred, y):
        # Ensure predicted is a single probability value
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)

        return - np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def dloss(self, y_pred, y):
        # Gradient of the loss with respect to the predicted value
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return - (y / y_pred) + (1 - y) / (1 - y_pred)

def mse(y, y_pred):
    return np.mean(np.power(y - y_pred, 2)) / 2

def mse_derivative(y, y_pred):
    return (y - y_pred)

def binary_crossentropy_elem(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

def binary_crossentropy(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return float(loss)

def binary_crossentropy_derivative(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y / y_pred - (1 - y) / (1 - y_pred))
