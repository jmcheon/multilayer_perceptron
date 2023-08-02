import numpy as np
import json
from sklearn.metrics import accuracy_score
from utils import binary_cross_entropy_loss, binary_cross_entropy_derivative, convert_to_binary_pred
from utils import plot_ 
from DenseLayer import DenseLayer, Layer

class NeuralNet():
    def __init__(self):
        self.network = None

    def create_network(self, net):
        network = None
        if isinstance(net, list) and all(isinstance(layer, Layer) for layer in net):
            network = net
            self.network = network
        elif isinstance(net, list) and all(isinstance(layer_data, dict) for layer_data in net):
            print("Creating a neural network...")
            network = []
            for layer_data in net:
                if layer_data['type'] == 'DenseLayer':
                    network.append(DenseLayer(layer_data['input_shape'],
                                            layer_data['output_shape'], 
                                            activation=layer_data['activation'],
                                            weights_initializer=layer_data['weights_initializer']))
            self.network = network
        else:
            raise TypeError("Invalid form of input to create a neural network.")
        return network

    def forward(self, input_data):
        input_data = self.network[0].forward(input_data.T)
        for layer in self.network[1:]:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient, alpha):
        for layer in reversed(self.network):
            output_gradient = layer.backward(output_gradient, alpha)

    def get_weights(self):
        weights_and_biases = []
        for layer in self.network:
            if isinstance(layer, DenseLayer):
                weights_and_biases.append(layer.weights)
                weights_and_biases.append(layer.bias)
        return weights_and_biases

    def set_weights(self, initial_weights):
        if not isinstance(initial_weights, list):
            raise TypeError("Invalid type of initial_weights, a list of weights required.")
        if not len(initial_weights) == 2 * len(self.network):
            #print(len(initial_weights), len(self.network))
            raise ValueError("Invalid input of list: not enought values to set weights and biases.")

        for index, layer in zip(range(0, len(initial_weights), 2), self.network):
            if isinstance(layer, DenseLayer):
                layer.set_weights(initial_weights[index], initial_weights[index + 1].reshape(-1, 1))

    def get_network_topology(self):
        layers = []
        for layer in self.network:
            if isinstance(layer, DenseLayer):
                #layer_str = f"DenseLayer({layer.shape}, activation={layer.activation.__name__})"
                layer_data = {
                    'type': 'DenseLayer',
                    'shape': layer.shape,
                    'input_shape': layer.shape[0],
                    'output_shape': layer.shape[1],
                    'activation': f'{layer.activation.__name__}',
                    'weights_initializer': f'{layer.weights_initializer}',
                    #'weights': layer.weights.tolist(),
                    #'bias': layer.bias.tolist(),
                }
                layers.append(layer_data)
            else:
                # for later layer types
                continue
        return layers 

    def to_json(self, file_path=None):
        model_data = {
            'network_topology': self.get_network_topology(),
            'input_size': self.network[0].shape[0],
            'output_size': self.network[-1].shape[1],
        }
        json_data = json.dumps(model_data)

        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_data)
            print(f"Model data saved to '{file_path}'")

        return json_data

    def create_mini_batches(x, y, batch_size):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, x.shape[0] - batch_size + 1, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            yield x[batch_idx], y[batch_idx]


    def fit(self, network, data_train, data_valid, loss, learning_rate, batch_size, epochs):
        if loss == 'binary_cross_entropy_loss':
            loss = binary_cross_entropy_loss
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
                #print('y_pred:', y_pred, 'x_i:', x_i, 'y_i:', y_i)
                binary_pred = convert_to_binary_pred(y_pred) 
                binary_predictions[index] = binary_pred
                #grad = nll_grad(y_pred, y_i).reshape(-1, 1)#.reshape(1, -1))
                grad = binary_cross_entropy_derivative(y_i, y_pred).reshape(-1, 1)
                total_loss = loss(y_i, y_pred)
                self.backward(grad, alpha)
    
            accuracy = accuracy_score(y_train, binary_predictions)
            accuracy_list.append(accuracy)
            loss_list.append(total_loss)
            epoch_list.append(epoch)
    
            # Calculate validation loss and accuracy
            val_loss = 0
            val_binary_predictions = np.zeros_like(y_val)
            for index, (x_val_i, y_val_i) in enumerate(zip(x_val, y_val)):
                y_val_pred = (self.forward(x_val_i.reshape(1, -1))).reshape(1, -1)
                val_loss = loss(y_val_i, y_val_pred)
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
                #alpha *= lr_decay_factor
                #print(f"Learning rate decayed to {alpha}.")
                counter = 0
                pass
    
            # Stop early if the validation loss hasn't improved for 'patience' epochs
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    
        print('Accuray:', accuracy, val_accuracy)
        plot_(epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list)
        return epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list

    def predict(self, data_test):
        alpha = 0.5
        x_test = data_test[:, :-2]
        y_test = data_test[:, -2:]

        binary_predictions = np.zeros_like(y_test)
        for index, (x_i, y_i) in enumerate(zip(x_test, y_test)):
            y_pred = (self.forward(x_i.reshape(1, -1))).reshape(1, -1)
            #print('y_pred:', y_pred)
            #error = ((y_pred - y_i) ** 2 / 2).reshape(-1, 1)#.reshape(1, -1))
            #error = binary_cross_entropy_loss(y_i, y_pred)
            #print('Error:', error)
            #grad = (y_pred - y_i).reshape(-1, 1)#.reshape(1, -1))
            grad = binary_cross_entropy_derivative(y_i, y_pred).reshape(-1, 1)
            #print('grad:', grad)
            #self.backward(grad, alpha)

            binary_pred = convert_to_binary_pred(y_pred) 
            binary_predictions[index] = binary_pred
    
        accuracy = accuracy_score(y_test, binary_predictions)
        #print(binary_predictions)
        print('Accuray:', accuracy)
        return binary_predictions
