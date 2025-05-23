import numpy as np

import multilayer_perceptron.srcs.activations as activations
import multilayer_perceptron.srcs.initializers as initializers


class Layer:
    def __init__(
        self,
        input_shape,
        output_shape,
        activation,
        weights_initializer,
        bias_initializer="zeros",
    ):
        self.shape = (input_shape, output_shape)
        self.outputs = []
        self.inputs = []

        # Initialize weights and bias
        if weights_initializer == "glorot_uniform":
            self.weights = initializers.glorot_uniform(input_shape, output_shape)
            self.weights_initializer = "glorot_uniform"
        elif weights_initializer == "he_uniform":
            self.weights = initializers.he_uniform((input_shape, output_shape))
            self.weights_initializer = "he_uniform"
            self.bias = initializers.he_uniform((1, output_shape))
        elif weights_initializer == "random":
            self.weights = np.random.randn(input_shape, output_shape)
            self.weights_initializer = "random"
            self.bias = np.random.randn(1, output_shape)
        elif weights_initializer == "zeros":
            self.weights = np.zeros((input_shape, output_shape))
            self.weights_initializer = "zeros"

        if bias_initializer == "zeros":
            self.bias = np.zeros((1, output_shape))

        # Activation function
        if activation.lower() == "relu":
            self.activation = activations.ReLU()
        elif activation.lower() == "sigmoid":
            self.activation = activations.Sigmoid()
        elif activation.lower() == "softmax":
            self.activation = activations.Softmax()

    def forward(self, x):
        pass

    def backward(self, output_gradient):
        pass


class Dense(Layer):
    def __init__(
        self,
        input_shape,
        output_shape,
        activation,
        weights_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        super().__init__(
            input_shape, output_shape, activation, weights_initializer, bias_initializer
        )
        self.deltas = None

    def set_parameters(self, weights, bias):
        if weights.shape != self.weights.shape:
            print(weights.shape, self.weights.shape)
            raise ValueError("Incompatible shape of weights.")
        self.weights = weights
        self.bias = bias

    def forward(self, input_data):
        self.inputs = input_data
        z = np.dot(self.inputs, self.weights) + self.bias
        self.z = z
        self.outputs = self.activation.f(z)
        return self.outputs

    def backward(self, gradients):
        self.deltas = self.activation.df(self.outputs, gradients)
        # dL/dinputs
        grads = np.dot(self.deltas, self.weights.T)
        return grads
