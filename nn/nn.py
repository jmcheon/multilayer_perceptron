import numpy as np

def initialize_weights(outs, ins):
    return np.zeros((outs, ins))
    return np.random.default_rng().normal(loc=0, scale=1/(nrows * ncols), size=(nrows, ncols))

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

    def loss(self, y_pred, y):
        return self._loss_function.loss(y_pred, y)

    def train(self, x, t):
        """
        Train the network on input x and expected output t.
        """
        # Accumulate intermediate results during forward pass.
        xs = [x]
        for layer in self._layers:
            xs.append(layer.forward(xs[-1]))

        # x = xs.pop()
        # print("x in net train:", x)
        dx = self._loss_function.dloss(xs.pop(), t)
        for layer, x in zip(self._layers[::-1], xs[::-1]):

            # Compute the derivatives
            y = np.dot(layer._W, x) + layer._b
            db = layer.act_function.df(y) * dx
            dx = np.dot(layer._W.T, db)
            dW = np.dot(db, x.T)

            # Update parameters
            layer._W -= self.lr * dW
            layer._b -= self.lr * db