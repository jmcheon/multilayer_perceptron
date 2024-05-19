import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))
        self.activation = activation
    
    def forward(self, x):
        self.z = np.dot(self.weights, x) + self.bias
        self.a = self.activation.f(self.z)
        return self.a

class NeuralNet:
    def __init__(self, layers, loss_function, learning_rate):
        self.layers = layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x, y):
        # Forward pass
        output = self.forward(x)
        
        # Calculate loss
        loss = self.loss_function.loss(output, y)
        
        # Backward pass (simplified version)
        gradient = self.loss_function.dloss(output, y)
        for layer in reversed(self.layers):
            gradient = self.update_weights(layer, gradient, x)
        
        return loss

    def update_weights(self, layer, gradient, x):
        # Dummy implementation for weight update
        layer.weights -= self.learning_rate * np.dot(gradient, x.T)
        layer.bias -= self.learning_rate * gradient
        return np.dot(layer.weights.T, gradient)

# Example usage:
layers = [
    Layer(784, 16, Sigmoid()),
    Layer(16, 16, Sigmoid()),
    Layer(16, 10, Sigmoid()),
]
net = NeuralNet(layers, BinaryCrossEntropyLoss(), 0.001)

# Assuming x is a (784, 1) input vector and y is a (10, 1) one-hot encoded target
x = np.random.randn(784, 1)
y = np.zeros((10, 1))
y[3] = 1  # Assuming the true class is the 4th class

loss = net.train(x, y)
print(f"Loss: {loss}")
