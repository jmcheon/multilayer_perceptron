import numpy as np

import srcs.activations as activations
import srcs.initializers as initializers


class Layer:
    def __init__(self, 
                 input_shape, 
                 output_shape, 
                 activation, 
                 weights_initializer,
                 bias_initializer='zeros',
        ):
        self.shape = (input_shape, output_shape)
        self.outputs = []
        self.inputs = []

        # Initialize weights and bias
        if weights_initializer == 'glorot_uniform':
            self.weights = initializers.glorot_uniform(input_shape, output_shape)
            self.weights_initializer = 'glorot_uniform'
        elif weights_initializer == 'he_uniform':
            self.weights = initializers.he_uniform((input_shape, output_shape))
            self.weights_initializer = 'he_uniform'
            self.bias = initializers.he_uniform(output_shape)
        elif weights_initializer == 'random':
            self.weights = np.random.randn(input_shape, output_shape)
            self.weights_initializer = 'random'
            self.bias = np.random.randn(output_shape)
        elif weights_initializer == 'zeros':
            self.weights = np.zeros((input_shape, output_shape))
            self.weights_initializer = 'zeros'

        if bias_initializer == 'zeros':
            self.bias = np.zeros((output_shape))

        # Activation function
        if activation.lower() == 'relu':
            self.activation = activations.relu
            self.activation_derivative = lambda gradient :activations.relu_derivative(self.outputs, gradient)
        elif activation.lower() == 'sigmoid':
            self.activation = activations.sigmoid
            self.activation_derivative = lambda gradient :activations.sigmoid_derivative(self.outputs, gradient)
        elif activation.lower() == 'softmax':
            self.activation = activations.softmax


    def forward(self, x):
        pass

    def backward(self, output_gradient, alpha):
        pass

class Dense(Layer):
    def __init__(self, 
                 input_shape, 
                 output_shape, 
                 activation, 
                 weights_initializer='glorot_uniform',
                 bias_initializer='zeros',
        ):
        super().__init__(input_shape, output_shape, activation, weights_initializer, bias_initializer)
        self.deltas = None
        #print(self.bias.shape)


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
