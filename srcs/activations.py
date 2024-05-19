from abc import abstractmethod
import numpy as np

class Activation:
    """
    Class to be inherited by activation functions.
    """
    @abstractmethod
    def f(self, x):
        """
        Method that implements the function.
        """
        pass

    @abstractmethod
    def df(self, x):
        """
        Derivative of the function with respect to its input.
        """
        pass

class Sigmoid(Activation):
    def f(self, x):
        return 1/(1 + np.exp(-x))

    def df(self, x):
        return self.f(x) * (1 - self.f(x))

class Softmax(Activation):
    def f(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtract np.max(x) for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def df(self, x):
        # df here is the Jacobian matrix of softmax function
        s = self.f(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

class LeakyRelu(Activation):
    """
    Leaky Rectified Linear Unit.
    """
    def __init__(self, leaky_param=0.1):
        self.alpha = leaky_param

    def f(self, x):
        return np.maximum(x, x * self.alpha)

    def df(self, x):
        return np.maximum(x > 0, self.alpha)


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(np.clip(-x, -709, 709)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def sigmoid_derivative(outputs, gradient):
    return gradient * (1 - outputs) * outputs


def relu_derivative(inputs, gradient):
    gradient[inputs <= 0] = 0
    return gradient
