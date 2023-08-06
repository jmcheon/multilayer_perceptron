import numpy as np
import json
from metrics import accuracy_score, precision_score, recall_score, f1_score
from losses import binary_cross_entropy, binary_cross_entropy_elem, binary_cross_entropy_derivative 
from utils import convert_to_binary_pred
from DenseLayer import DenseLayer, Layer

class NeuralNet():
    def __init__(self, optimizer='momentum', momentum=0.9, decay_rate=0.9, nesterov=False):
        self.network = None
        self.metrics_historic = []

        self.optimizer = optimizer
        if self.optimizer != 'momentum':
            self.nesterov = False
        else:
            self.nesterov = nesterov
        self.momentum = momentum

        self.decay_rate = decay_rate
        self.epsilon = 1e-8

        self.params = {}
        self.velocity = {}

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
        print("Setting neural network params...")
        for layer_num, layer in enumerate(self.network):
            if isinstance(layer, DenseLayer):
                self.params[f'W{layer_num + 1}'] = layer.weights
                self.params[f'b{layer_num + 1}'] = layer.bias

        for param_name, param_value in self.params.items():
            self.velocity[param_name] = np.zeros_like(param_value)

        return network

    def save_model(self):
        # Save model configuration as a JSON file
        model_config = self.to_json()
        with open('./saved_model_config.json', 'w') as json_file:
            json.dump(model_config, json_file)
            print("> Saving model configuration to './saved_model_config.json'")
        
        # Save model weights as a .npy file
        model_weights = self.get_weights()
        np.save('./saved_model_weights.npy', model_weights)
        print("> Saving model weights to './saved_model_weights.npy'")


    def forward(self, input_data):
        input_data = self.network[0].forward(input_data.T)
        for layer in self.network[1:]:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient, alpha):
        grads = {}
        total_layers = len(self.network)
        for index, layer in enumerate(reversed(self.network)):
            layer_num = total_layers - index - 1
            #print('layer_num:', layer_num)
            output_gradient, weights_gradient, bias_gradient = layer.backward(output_gradient, alpha)

            layer_grads = {
                f'W{layer_num + 1}': weights_gradient,
                f'b{layer_num + 1}': bias_gradient
            }
            for param_name, grad in layer_grads.items():
                grads[param_name] = grad

        # Return the aggregated gradients
        return grads

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
            raise ValueError("Invalid input of list: not enought values to set weights and biases.")

        for index, layer in zip(range(0, len(initial_weights), 2), self.network):
            if isinstance(layer, DenseLayer):
                layer.set_weights(initial_weights[index], initial_weights[index + 1].reshape(-1, 1))

    def get_network_topology(self):
        layers = []
        for layer in self.network:
            if isinstance(layer, DenseLayer):
                layer_data = {
                    'type': 'DenseLayer',
                    'shape': layer.shape,
                    'input_shape': layer.shape[0],
                    'output_shape': layer.shape[1],
                    'activation': f'{layer.activation.__name__}',
                    'weights_initializer': f'{layer.weights_initializer}',
                }
                layers.append(layer_data)
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

    def print_metrics_historic(self):
        if len(self.metrics_historic) == 0:
            print("It hasn't trained yet.")
            return
        for metric in self.metrics_historic:
            print(f"epoch {metric['epoch']} - loss: {metric['loss']} - val_loss: {metric['val_loss']}")

    def create_mini_batches(self, x, y, batch_size):
        if isinstance(batch_size, str) and batch_size == 'batch':
            yield x, y
        else:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)

            for start_idx in range(0, x.shape[0] - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                yield x[batch_idx], y[batch_idx]

    def evaluate_metrics(self, y, y_pred):
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        return accuracy, precision, recall, f1

    def update_parameters(self, grads, learning_rate):
        if self.optimizer == 'momentum':
            for param_name in self.params:
                if self.nesterov:
                    look_ahead_grad = self.params[param_name] + self.momentum * self.velocity[param_name]
                    self.velocity[param_name] = self.momentum * self.velocity[param_name] - learning_rate * grads[param_name]
                    self.params[param_name] = look_ahead_grad - learning_rate * self.velocity[param_name]
                else:
                    self.velocity[param_name] = self.momentum * self.velocity[param_name] - learning_rate * grads[param_name]
                    self.params[param_name] += self.velocity[param_name]
        
        elif self.optimizer == 'rmsprop':
            for param_name in self.params:
                self.velocity[param_name] = self.decay_rate * self.velocity[param_name] + (1 - self.decay_rate) * grads[param_name]**2
                self.params[param_name] -= learning_rate * grads[param_name] / (np.sqrt(self.velocity[param_name]) + self.epsilon)
        
        else:
            raise ValueError(f"Invalid optimizer '{self.optimizer}', expected 'momentum', 'rmsprop' or 'adam'.")


    def fit(self, data_train, data_valid, loss, learning_rate, batch_size, epochs):
        if loss == 'binary_cross_entropy':
            loss = binary_cross_entropy
            loss_prime = binary_cross_entropy_derivative
        elif loss == 'mse':
            loss = mse
            loss_prime = mse_derivative
        patience=5
        lr_decay_factor=0.1
        lr_decay_patience=3
     
        accuracy_list = []
        loss_list = []

        val_accuracy_list = []
        val_precision_list = []
        val_recall_list = []
        val_f1_list = []
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
            total_loss = 0
            n_batches = 0

            y_train_batch = np.empty((0, y_train.shape[1]))
            binary_predictions = np.empty((0, y_train.shape[1]))
            for x_batch, y_batch in self.create_mini_batches(x_train, y_train, batch_size):
                y_train_batch = np.vstack((y_train_batch, y_batch))
                batch_loss = 0
                for x_i, y_i in zip(x_batch, y_batch):
                    y_pred = (self.forward(x_i.reshape(1, -1))).T
                    binary_predictions = np.vstack((binary_predictions, convert_to_binary_pred(y_pred)))
                    grad = loss_prime(y_i, y_pred).T
                    batch_loss = loss(y_i, y_pred)
                    grads = self.backward(grad, alpha)
                    self.update_parameters(grads, alpha)

                total_loss += batch_loss
                n_batches += 1

            total_loss /= n_batches
    
            accuracy = accuracy_score(y_train_batch, binary_predictions)
            accuracy_list.append(accuracy)
            loss_list.append(total_loss)
            epoch_list.append(epoch)
    
            # Calculate validation loss and accuracy
            val_loss = 0
            n_val_batches = 0

            y_val_batch = np.empty((0, y_val.shape[1]))
            val_binary_predictions = np.empty((0, y_val.shape[1]))
            for x_batch, y_batch in self.create_mini_batches(x_val, y_val, batch_size):
                y_val_batch = np.vstack((y_val_batch, y_batch))
                val_batch_loss = 0
                for x_val_i, y_val_i in zip(x_batch, y_batch):
                    y_val_pred = (self.forward(x_val_i.reshape(1, -1))).T
                    val_batch_loss = loss(y_val_i, y_val_pred)
                    val_binary_predictions = np.vstack((val_binary_predictions, convert_to_binary_pred(y_val_pred)))

                val_loss += val_batch_loss
                n_val_batches += 1

            val_loss /= n_val_batches
    
            val_accuracy, val_precision, val_recall, val_f1 = self.evaluate_metrics(y_val_batch, val_binary_predictions)
            val_accuracy_list.append(val_accuracy)
            val_precision_list.append(val_precision)
            val_recall_list.append(val_recall)
            val_f1_list.append(val_f1)
            val_loss_list.append(val_loss)



            padding_width = len(str(epochs))
            print(f'epoch {epoch + 1:0{padding_width}d}/{epochs} - loss: {total_loss:.4f} - val_loss: {val_loss:.4f}, ', end="")
            print(f"Accuracy: {val_accuracy:.2f}%, Precision: {val_precision:.2f}%, Recall: {val_recall:.2f}%, F1 score: {val_f1:.2f}%")

            self.metrics_historic.append({"epoch": f'{epoch + 1:0{padding_width}d}/{epochs}', "loss": f'{total_loss:.4f}', "val_loss": f'{val_loss:.4f}'})

    
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
                #counter = 0
                pass
    
            # Stop early if the validation loss hasn't improved for 'patience' epochs
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    

        print('Train accuracy:', accuracy, 'Validation accuracy:', val_accuracy)
        return epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list

    def predict(self, data_test):
        x_test = data_test[:, :-2]
        y_test = data_test[:, -2:]

        y_pred = self.forward(x_test).T
        #error = binary_cross_entropy_elem(y_test, y_pred)
        #print('loss:', error[:,0])
        accuracy = accuracy_score(y_test, convert_to_binary_pred(y_pred))
        print('\nAccuracy:', accuracy)
        return y_pred
