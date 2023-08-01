import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Helper functions
nll_grad = lambda y_pred, y_true: y_pred - y_true


def sigmoid(x):
    return 1 / (1 + np.exp(np.clip(-x, -709, 709)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def heUniform(shape):
    fan_in, _ = shape
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)

class Layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        pass

    def backward(self, output_gradient, alpha):
        pass

class DenseLayer(Layer):
    def __init__(self, input_shape, output_shape, activation, weights_initializer=heUniform):
        self.shape = (input_shape, output_shape)
        self.weights = np.random.randn(output_shape, input_shape)
        self.bias = np.random.randn(output_shape, 1)
        self.activation = activation

    def set_weights(self, weights, bias):
        print(weights.shape, bias.shape, self.weights.shape, self.bias.shape)
        if weights.shape != self.weights.shape:
            raise ValueError("Uncompatiable shape of weights.")
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        self.x = x
        z = np.dot(self.weights, self.x) + self.bias
        return self.activation(z)

    def backward(self, output_gradient, alpha):
        activation_gradient = np.mean(self.activation(self.x) * (1 - self.activation(self.x)))
        weights_gradient = np.dot(output_gradient * activation_gradient, self.x.T)
        self.weights -= alpha * weights_gradient
        self.bias -= alpha * output_gradient * activation_gradient
        return np.dot(self.weights.T, output_gradient)

class classificationNet():
    def __init__(self, output_shape=1):
        self.network = None

    def create_network(self, layers_list):
        network = layers_list
        self.network = network
        return network

    def get_weights(self):
        weights_and_biases = []
        for layer in self.network:
            if isinstance(layer, DenseLayer):
                weights_and_biases.append((layer.weights, layer.bias))
        return weights_and_biases

    def set_weights(self, initial_weights):
        if not isinstance(initial_weights, list):
            raise TypeError("Invalid type of initial_weights, a list of weights required.")
        if not len(initial_weights) == 2 * len(self.network):
            print(len(initial_weights), len(self.network))
            raise ValueError("Invalid input of list: not enought values to set weights and biases.")

        for index, layer in zip(range(0, len(initial_weights), 2), self.network):
            if isinstance(layer, DenseLayer):
                layer.set_weights(initial_weights[index], initial_weights[index + 1].reshape(-1, 1))

    def network_topology(self):
        topology_string = ""
        depth = 0
        for layer in self.network:
            if isinstance(layer, DenseLayer):
                layer_str = f"Dense({layer.shape}, activation={layer.activation.__name__})"
                depth += 1
            else:
                continue
            topology_string += layer_str + " -> "
        # Remove the final " -> "
        topology_string = topology_string[:-depth]
        return topology_string

    def to_json(self, file_path=None):
        model_data = {
            'network_topology': self.network_topology(),
            'input_size': self.network[0].shape[0],
            'output_size': self.network[-1].shape[1],
            'layers': [],
        }

        for layer in self.network:
            if isinstance(layer, DenseLayer):
                layer_data = {
                    'type': 'Dense',
                    'shape': layer.shape,
                    'activation_function': f'{layer.activation.__name__}',
                    'weights': layer.weights.tolist(),
                    'bias': layer.bias.tolist(),
                }
                model_data['layers'].append(layer_data)

        json_data = json.dumps(model_data)

        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_data)
            print(f"Model data saved to '{file_path}'")

        return json_data


    def forward(self, x):
        x = self.network[0].forward(x.T)
        for layer in self.network[1:]:
            x = layer.forward(x)
        return x

    def backward(self, output_gradient, alpha):
        for layer in reversed(self.network):
            output_gradient = layer.backward(output_gradient, alpha)

    def fit(self, network, data_train, data_valid, loss, learning_rate, batch_size, epochs):
        patience=5
        lr_decay_factor=0.1
        lr_decay_patience=3
     
        accuracy_list = []
        loss_list = []
        val_accuracy_list = []
        val_loss_list = []
        epoch_list = []
        best_loss = float('inf')
        counter = 0
        alpha = learning_rate
    
        # Unpack data_train into x_train and y_train
        x_train = data_train[:, :-2]
        y_train = data_train[:, -2:]
        
        # Unpack data_valid into x_val and y_val
        x_val = data_valid[:, :-2]
        y_val = data_valid[:, -2:]
    
        print('x_train shape :', x_train.shape)
        print('x_valid shape :', x_val.shape)
        for epoch in range(epochs):
            binary_predictions = np.zeros_like(y_train)
            total_loss = 0
            for index, (x_i, y_i) in enumerate(zip(x_train, y_train)):
                y_pred = (self.forward(x_i.reshape(1, -1))).reshape(1, -1)
                binary_pred = convert_to_binary_pred(y_pred) 
                binary_predictions[index] = binary_pred
                grad = nll_grad(y_pred, y_i).reshape(-1, 1)#.reshape(1, -1))
                total_loss = loss_(y_i, y_pred)
                self.backward(grad, alpha)
    
            accuracy = accuracy_score(y_train, binary_predictions)
            accuracy_list.append(accuracy)
            loss_list.append(total_loss)
            epoch_list.append(epoch)
    
            # Calculate validation loss and accuracy
            val_loss = 0
            val_binary_predictions = np.zeros_like(y_val)
            for index, (x_val_i, y_val_i) in enumerate(zip(x_val, y_val)):
                y_val_pred = (self.forward(x_val_i.reshape(1, -1)))
                val_loss = loss_(y_val_i, y_val_pred)
                val_binary_pred = convert_to_binary_pred(y_val_pred) 
                val_binary_predictions[index] = val_binary_pred
    
            val_accuracy = accuracy_score(y_val, val_binary_predictions)
            val_accuracy_list.append(val_accuracy)
            val_loss_list.append(val_loss)



            padding_width = len(str(epochs))
            print(f'epoch {epoch + 1:0{padding_width}d}/{epochs} - loss: {total_loss:.4f} - val_loss: {val_loss:.4f}')
    
            # Check if validation loss is decreasing
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
    
            # Learning rate decay
            if counter >= lr_decay_patience:
                alpha *= lr_decay_factor
                print(f"Learning rate decayed to {alpha}.")
                counter = 0
                pass
    
            # Stop early if the validation loss hasn't improved for 'patience' epochs
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    
        plot_(epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list)
        return epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list

    def predict(self, data_test):
        alpha = 0.5
        x_test = data_test[:, :-2]
        y_test = data_test[:, -2:]
        for index, (x_i, y_i) in enumerate(zip(x_test, y_test)):
            y_pred = (self.forward(x_i.reshape(1, -1))).reshape(1, -1)
            print('y_pred:', y_pred)
            grad = ((y_pred - y_i) ** 2 / 2).reshape(-1, 1)#.reshape(1, -1))
            print('Error:', grad)
            self.backward(grad, alpha)

def loss_(y, y_hat, eps=1e-15):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return float(loss)

def convert_to_binary_pred(y_pred, threshold=0.5):
    max_index = np.argmax(y_pred)
    
    binary_pred = np.zeros((1, 2))
    binary_pred[0][max_index] = 1

    return binary_pred

def save(model):
    # Save model configuration as a JSON file
    model_config = model.to_json()
    with open('./saved_model_config.json', 'w') as json_file:
        json.dump(model_config, json_file)
        print("> Saving model configuration to './saved_model_config.json'")
    
    # Save model weights as a .npy file
    model_weights = model.get_weights()
    np.save('./saved_model_weights.npy', model_weights)
    print("> Saving model weights to './saved_model_weights.npy'")

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    data[1] = data[1].map({"M": 1, "B": 0})
    y = data[1].values
    x = data.drop([0, 1], axis=1).values
    # Normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = one_hot_encode_binary_labels(y)
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

def load_data1(filename):
    data = pd.read_csv(filename, header=None)
    data[1] = data[1].map({"M": 1, "B": 0})
    y = data[1].values
    x = data.drop([0, 1], axis=1).values

    # Normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    return x, y 

def one_hot_encode_binary_labels(labels):
    one_hot_encoded_labels = np.zeros((len(labels), 2))
    print(one_hot_encoded_labels.shape)
    for i, label in enumerate(labels):
        one_hot_encoded_labels[i, int(label)] = 1

    return one_hot_encoded_labels


#def plot_(x, y ,net):
def plot_(epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list):
    #epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = train(x, y, net)
    # Plot training and validation accuracy and loss
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(2):
        ax = axes[i]
        if (i == 0):
            ax.plot(epoch_list, loss_list, label='training loss')
            ax.plot(epoch_list, val_loss_list, label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        else:
            ax.plot(epoch_list, accuracy_list, label='training accuracy')
            ax.plot(epoch_list, val_accuracy_list, label='validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()

    plt.show()

def plot_compare(x, y ,net, net1):
    epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = train(x, y, net)
    # Plot training and validation accuracy and loss
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for i in range(2):
        ax = axes[0][i]
        if (i == 0):
            ax.plot(epoch_list, loss_list, label='training loss')
            ax.plot(epoch_list, val_loss_list, label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        else:
            ax.plot(epoch_list, accuracy_list, label='training accuracy')
            ax.plot(epoch_list, val_accuracy_list, label='validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
    epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = train(x, y, net1)
    for i in range(2):
        ax = axes[1][i]
        if (i == 0):
            ax.plot(epoch_list, loss_list, label='training loss')
            ax.plot(epoch_list, val_loss_list, label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        else:
            ax.plot(epoch_list, accuracy_list, label='training accuracy')
            ax.plot(epoch_list, val_accuracy_list, label='validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()

    plt.show()

if __name__ == "__main__":
    np.random.seed(0)

    # Load and split the data
    #x_train, x_test, y_train, y_test = load_data('data.csv')
    x_train, x_val, y_train, y_val= load_data('data.csv')
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

    # Combine x_train and y_train as data_train
    data_train = np.hstack((x_train, y_train))
    # Combine x_val and y_val as data_valid
    data_valid = np.hstack((x_val, y_val))

    model = classificationNet(output_shape=2)
    network = model.create_network([
        DenseLayer(30, 20, activation=sigmoid),
        DenseLayer(20, 10, activation=sigmoid, weights_initializer='heUniform'),
        DenseLayer(10, 2, activation=sigmoid, weights_initializer='heUniform'),
        DenseLayer(2, 2, activation=softmax, weights_initializer='heUniform')
        ])

    model.fit(network, data_train, data_valid, loss=loss_, learning_rate=1e-3, batch_size=8, epochs=70)
    #save(model)
