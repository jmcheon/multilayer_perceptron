import numpy as np

from activations import (relu, relu_derivative, sigmoid, sigmoid_derivative,
                         softmax)
from utils import heUniform


class Layer:
    def __init__(self, input_shape, output_shape, activation, weights_initializer):
        self.shape = (input_shape, output_shape)
        #print('shape:', self.shape)
        self.outputs = []
        self.inputs = []

        # Initialize weights and bias
        if weights_initializer == 'heUniform':
            self.weights = heUniform((input_shape, output_shape))
            self.weights_initializer = 'heUniform'
            self.bias = heUniform(output_shape)
        elif weights_initializer == 'random':
            self.weights = np.random.randn(input_shape, output_shape)
            self.weights_initializer = 'random'
            self.bias = np.random.randn(output_shape)
        elif weights_initializer == 'zero':
            self.weights = np.zeros((input_shape, output_shape))
            self.weights_initializer = 'zero'
            self.bias = np.zeros((output_shape))
            #self.bias = np.zeros(self.weights.shape[1])
            


        # Activation function
        if activation.lower() == 'relu':
            self.activation = relu
            self.activation_derivative = lambda gradient :relu_derivative(self.outputs, gradient)
        elif activation.lower() == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = lambda gradient :sigmoid_derivative(self.outputs, gradient)
        elif activation.lower() == 'softmax':
            self.activation = softmax


    def forward(self, x):
        pass

    def backward(self, output_gradient, alpha):
        pass

class DenseLayer(Layer):
    def __init__(self, input_shape, output_shape, activation, weights_initializer='random'):
        super().__init__(input_shape, output_shape, activation, weights_initializer)
        self.deltas = None
        print(self.bias.shape)


    def set_weights(self, weights, bias):
        if weights.shape != self.weights.shape:
            print(weights.shape, self.weights.shape)
            raise ValueError("Incompatible shape of weights.")
        self.weights = weights
        self.bias = bias

    def set_activation_gradient(self, gradient):
        self.deltas = self.activation_derivative(gradient) 

    def forward(self, input_data):
        self.inputs = input_data
        z = np.dot(self.inputs, self.weights) + self.bias
        self.z = z
        self.outputs = self.activation(z)
        return self.outputs

    def backward(self, output_gradient, alpha):
        activation_gradient = (self.activation(self.z) * (1 - self.activation(self.z)))

        weights_gradient = np.dot(output_gradient * activation_gradient, self.input_data.T)
        bias_gradient = output_gradient * activation_gradient

        input_gradient = np.dot(self.weights.T, output_gradient * activation_gradient)
        self.weights -= alpha * weights_gradient
        self.bias -= alpha * output_gradient * activation_gradient

        return input_gradient, weights_gradient, bias_gradient

