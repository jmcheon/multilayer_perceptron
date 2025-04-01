from abc import abstractmethod

import numpy as np


class Activation:
    """
    Class to be inherited by activation functions.
    """

    @abstractmethod
    def forward(self, x):
        """
        Method that implements the activation function.
        """
        pass

    @abstractmethod
    def backward(self, x, gradients):
        """
        Derivative of the function with respect to its input.
        """
        pass

    def __call__(self, x):
        return self.forward(x)


class Sigmoid(Activation):
    """
    Sigmoid activation.
    """

    def forward(self, x):
        return 1 / (1 + np.exp(np.clip(-x, -709, 709)))

    def backward(self, outputs, gradients):
        return gradients * outputs * (1 - outputs)


class Softmax(Activation):
    """
    Softmax activation.
    """

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, outputs, gradients):
        # backward here is the Jacobian matrix of softmax function
        dinputs = np.empty_like(gradients)
        for index, (single_output, single_grad) in enumerate(zip(outputs, gradients)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            dinputs[index] = np.dot(jacobian_matrix, single_grad)
        return dinputs


class ReLU(Activation):
    """
    Rectified Linear Unit.
    """

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, outputs, gradients):
        gradients[outputs <= 0] = 0
        return gradients


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit.
    """

    def __init__(self, leaky_param=0.1):
        self.alpha = leaky_param

    def forward(self, x):
        return np.maximum(x, x * self.alpha)

    def backward(self, x, gradients):
        gradients[x <= 0] *= self.alpha
        return gradients
