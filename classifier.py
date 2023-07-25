import numpy as np
from utils import data_spliter, load_data, normalization

class Layer:

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        pass

    def backward(self, output_gradient, alpha):
        pass


class Dense(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, x):
        self.x = x 
        return np.dot(self.weights, self.x) + self.bias

    def backward(self, output_gradient, alpha):
        weights_gradient = np.dot(output_gradient, self.x.T)
        self.weights -= alpha * weights_gradient
        self.bias -= alpha * output_gradient
        return np.dot(self.weights.T, output_gradient)

class classificationNet():

    def __init__(self, output_size = 1):
        self.layer1 = Dense(input_size = 31, output_size = 25)
        self.layer2 = Dense(input_size = 25, output_size = output_size)

    def forward(self, x):
        x = self.layer1.forward(x.T)
        x = self.layer2.forward(x)
        return x

    def backward(self, output_gradient, alpha):
        output_gradient = self.layer2.backward(output_gradient, alpha)
        self.layer1.backward(output_gradient, alpha)
        
class Activation(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, x):
        self.x = x
        return self.activation(self.x)

    def backward(self, output_gradient, alpha):
        return np.multiply(output_gradient, self.activation_prime(self.x))

class Tanh(Activation):

    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class Softmax(Layer):

    def forward(self, x):
        temp = np.exp(x)
        self.y = temp / np.sum(temp)
        return self.y

    def backward(self, output_gradient, alpha):
        n = np.size(self.y)
        temp = np.tile(self.y, n)
        return np.dot(temp * (np.identity(n) - np.transpose(temp)), output_gradient)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_true - y_pred) / np.size(y_true)


def xor_test():
    X = np.reshape([[0, 0], [1, 0], [0, 1], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    
    network = [
        Dense(2, 3),
        Tanh(),
        Dense(3, 1),
        Tanh()
    ]
    
    epochs = 10000
    alpha = 0.1
    
    # train
    for i in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            # forward
            output = x
            for layer in network:
                output = layer.forward(output)
    
            # error
            error += mse(y, output)
    
            # backward
            grad = mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, alpha)
    
            error /= len(X)
    
            print('%d/%d, error=%f' % (i + 1, epochs, error))

sigmoid = lambda x: 1 / (1 + np.exp(-x))
tolerance = 1e-6
nll = lambda y_pred, y_true: -(y_true * np.log(y_pred + tolerance) + (1 - y_true) * np.log(1 - y_pred + tolerance))
nll_grad = lambda y_pred, y_true: y_pred - y_true

def train(x, y):
    x_train, x_valid, y_train, y_valid = data_spliter(x, y, 0.8)
    net = classificationNet(1)
    alpha = 1e-2
    epochs = 50

    for i in range(epochs):
        loss = 0
        for x, y in zip(x_train, y_train):
            layer1_x = net.forward(x.reshape(1, -1))
            y_pred = sigmoid(int(net.forward(x.reshape(1, -1))))
            grad = nll_grad(y_pred, y).reshape(1, -1)
            loss += nll(y_pred, y)[0]

            net.backward(grad, alpha)

        if i % 10 == 0:
            print(f"Epoch {i} train loss: {loss / len(x_train)}")
            loss = 0
            for x, y in zip(x_valid, y_valid):
                y_pred = sigmoid(net.forward(x.reshape(1, -1)))
                loss += nll(y_pred, y)[0, 0]
            print(f"Valid loss: {loss / len(x_valid)}")



if __name__ == "__main__":
    mapping = {'B': 0, 'M': 1}

    df, features = load_data('data.csv', header=False)
    print(df)
    data = df.values
    x = df.select_dtypes(include='number').values
    normalized_x, data_min, data_max = normalization(x)

    y = df[1].replace(mapping).values
    print(normalized_x.shape, y.shape)
    #x_train, x_valid, y_train, y_valid = data_spliter(x, y, 0.8)
    #print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    train(normalized_x, y)





