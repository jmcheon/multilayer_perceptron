import mlp.activations as activations
import mlp.initializers as initializers
import numpy as np
from mlp.module import Module


class Layer:
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        initializer,
    ):
        self.shape = (in_features, out_features)
        self.outputs = []
        self.inputs = []

        # Initialize weights and biases
        if initializer == "glorot_uniform":
            self.weights = initializers.glorot_uniform((in_features, out_features))
            self.initializer = "glorot_uniform"
            self.biases = np.zeros((1, out_features))
        elif initializer == "he_uniform":
            self.weights = initializers.he_uniform((in_features, out_features))
            self.initializer = "he_uniform"
            self.biases = initializers.he_uniform((1, out_features))
        elif initializer == "random":
            self.weights = np.random.randn(in_features, out_features)
            self.initializer = "random"
            self.biases = np.random.randn(1, out_features)
        elif initializer == "zeros":
            self.weights = np.zeros((in_features, out_features))
            self.initializer = "zeros"
            self.biases = np.zeros((1, out_features))

        # Activation function
        if activation.lower() == "relu":
            self.activation = activations.ReLU()
        elif activation.lower() == "sigmoid":
            self.activation = activations.Sigmoid()
        elif activation.lower() == "softmax":
            self.activation = activations.Softmax()

    def forward(self, x):
        pass

    def backward(self, grad_output):
        pass

    def parameters(self):
        return [self.weights, self.biases]

    def gradients(self):
        return [self.dweights, self.dbias]


class Dense(Module):
    def __init__(
        self, in_features, out_features, activation, initializer=initializers.glorot_uniform
    ):
        self.shape = (in_features, out_features)
        self.initializer = initializer
        self.weights = initializer((in_features, out_features))
        self.biases = np.zeros((1, out_features))
        self.activation = activation
        self.inputs = None
        self.deltas = None
        self.dweights = None
        self.dbiases = None

    def set_parameters(self, weights, biases):
        if weights.shape != self.weights.shape:
            print(weights.shape, self.weights.shape)
            raise ValueError("Incompatible shape of weights.")
        self.weights = weights
        self.biases = biases

    def forward(self, x):
        self.inputs = x
        linear_output = np.dot(self.inputs, self.weights) + self.biases
        self.outputs = self.activation(linear_output)
        return self.outputs

    def backward(self, grad_output):
        self.deltas = self.activation.backward(self.outputs, grad_output)
        # dL/dinputs
        grads = np.dot(self.deltas, self.weights.T)
        return grads

    def parameters(self):
        return [self.weights, self.biases]

    def zero_grad(self):
        self.dweights = None
        self.dbiases = None
