import numpy as np


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
