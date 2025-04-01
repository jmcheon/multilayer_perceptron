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

    def __init__(self, name="Sigmoid"):
        self.name = name

    def forward(self, x):
        self.outputs = 1 / (1 + np.exp(np.clip(-x, -709, 709)))
        return self.outputs

    def backward(self, gradients):
        return gradients * self.outputs * (1 - self.outputs)


class Softmax(Activation):
    """
    Softmax activation.
    """

    def __init__(self, name="Softmax"):
        self.name = name

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.outpus = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.outputs

    def backward(self, gradients):
        # backward here is the Jacobian matrix of softmax function
        dinputs = np.empty_like(gradients)
        for index, (single_output, single_grad) in enumerate(zip(self.outputs, gradients)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            dinputs[index] = np.dot(jacobian_matrix, single_grad)
        return dinputs


class ReLU(Activation):
    """
    Rectified Linear Unit.
    """

    def __init__(self, name="ReLU"):
        self.name = name

    def forward(self, x):
        self.inputs = x
        return np.maximum(0, x)

    def backward(self, gradients):
        gradients[self.inputs <= 0] = 0
        return gradients


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit.
    """

    def __init__(self, leaky_param=0.1, name="Leaky ReLU"):
        self.alpha = leaky_param
        self.name = name

    def forward(self, x):
        self.inputs = x
        return np.maximum(x, x * self.alpha)

    def backward(self, x):
        self.inputs[x <= 0] *= self.alpha
        return self.inputs
