import numpy as np

def initialize_weights(outs, ins):
    return np.random.default_rng().normal(loc=0, scale=1/(outs * ins), size=(outs, ins))
    # return np.zeros((outs, ins))

def initialize_bias(outs):
    """create a column vector as a matrix"""
    # return np.zeros((outs, 1))
    return initialize_weights(outs, 1)

class Layer:
    """
    Layer class that represents the connections and the flow of information between a column of neurons and the next.
    It deals with what happens in between two columns of neurons instead of having the layer specifially represent the neurons of each vertical column
    """
    def __init__(self, ins, outs, act_function) -> None:
        self.ins = ins
        self.outs = outs
        self.act_function = act_function

        self._W = initialize_weights(self.outs, self.ins)
        self._b = initialize_bias(self.outs)

    def forward(self, x):
        """
        helper method that computes the forward pass in the layer

        Parameters:
        x: a set of neuron states

        Returns:
        the next set of neuron states
        """
        ret = self.act_function.f(np.dot(self._W, x) + self._b)
        # print("ret:", ret)
        return  ret

class NeuralNet:
    """
    A series of layers connected and compatible.
    """
    def __init__(self, layers, loss_function, lr) -> None:
        self._layers = layers
        self._loss_function = loss_function
        self.lr = lr
        self.check_layer_compatibility()

    def check_layer_compatibility(self):
        for from_, to_ in zip(self._layers[:-1], self._layers[1:]):
            print("from, to:", from_.ins, to_.ins)
            if from_.outs != to_.ins:
                raise ValueError("Layers should have compatible shapes.")

    def forward(self, x):
        # xs = [x]
        # for layer in self._layers:
        #     xs.append(layer.forward(xs[-1]))
        # return xs
        out = x
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def predict(self, x):
        a = self.forward(x)
        # return (a >= 0.5).astype(int)
        return np.argmax(a, axis=0)

    def loss(self, y_pred, y):
        return self._loss_function.loss(y_pred, y)

    def backward(self, a, y):
        dz = a.pop() - y
        m = y.shape[1]
        n_layers = len(self._layers)
        # print("m examples:", m)
        for i, (layer, a) in enumerate(zip(self._layers[::-1], a[::-1])):

            # Compute the derivatives
            # print(i)
            dW = 1 / m * np.dot(dz, a.T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            if i < n_layers: 
                dz = np.dot(layer._W.T, dz) * a * (1 - a)

            # Update parameters
            layer._W -= self.lr * dW
            layer._b -= self.lr * db


    def fit(self, x, y):
        """
        Train the network on input x and expected output y.
        """
        # activations during forward pass
        a = [x]
        for layer in self._layers:
            a.append(layer.forward(a[-1]))

        # backpropagation
        self.backward(a, y)