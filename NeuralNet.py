import json

import numpy as np

import optimizers
from DenseLayer import DenseLayer, Layer
from losses import (binary_crossentropy, binary_crossentropy_derivative,
                    binary_crossentropy_elem)
from metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import convert_to_binary_pred


class NeuralNet():
    def __init__(self, name="NeuralNet"):
        self.network = None
        self.compiled = False
        self.optimizer = None
        self.history = {}
        self.name = name
        self.metrics = []

    def create_network(self, net, name=None):
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

        if name:
            self.name = name

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
        for layer in self.network:
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

    def backprop(self, y_true, y_pred):
        grad = self.loss_prime(y_true, y_pred)

        self.optimizer.pre_update_params()
        total_layers = len(self.network)
        for index, layer in enumerate(reversed(self.network)):
            layer_num = total_layers - index - 2
            layer.set_activation_gradient(grad)
            grad = np.dot(layer.deltas, layer.weights.T)
            self.optimizer.update_params(layer, self.network[layer_num].outputs.T)
        self.optimizer.post_update_params()

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
                layer.set_weights(initial_weights[index], initial_weights[index + 1])

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

    def print_history(self):
        if len(self.history) == 0:
            print("It hasn't trained yet.")
            return
        for metric in self.history:
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

    def compile(self, 
                loss, 
                metrics=None, 
                optimizer='sgd', 
                loss_weights=None, 
                weighted_metrics=None, 
                run_eagerly=None, 
                steps_per_execution=1
        ):

        if isinstance(optimizer, optimizers.Optimizer):
            self.optimizer = optimizer
        elif optimizer == 'sgd':
            self.optimizer = optimizers.SGD()

        if not loss:
            raise ValueError("No loss found. You may have forgotten to provide a `loss` argument in the `compile()` method.")
        elif loss == 'binary_crossentropy':
            self.loss = binary_crossentropy
            self.loss_prime = binary_crossentropy_derivative
            self.history['loss'] = []
        elif loss == 'mse':
            self.loss = mse
            self.loss_prime = mse_derivative
            self.history['loss'] = []

        if metrics:
            self.metrics = metrics
            for metric in metrics:
                self.history[metric.lower()] = []

        self.compiled = True
        return

    def update_history(self, y_true, y_pred, validation=False):
        accuracy, precision, recall, f1 = self.evaluate_metrics(y_true, y_pred)
        if validation:
            valid = "val_"
        else:
            valid = ""
        if 'accuracy' in self.metrics:
            self.history[valid + 'accuracy'].append(accuracy)
            print(f" - {valid}accuracy: {accuracy:.2f}%", end="")
        if 'Precision' in self.metrics:
            self.history[valid + 'precision'].append(precision)
            print(f" - {valid}precision: {precision:.2f}%", end="")
        if 'Recall' in self.metrics:
            self.history[valid + 'recall'].append(recall)
            print(f" - {valid}recall: {recall:.2f}%", end="")

    def fit(self, 
            x_train, 
            y_train, 
            batch_size, 
            epochs=1, 
            validation_data=None):
        if self.compiled == False:
            raise RuntimeError("You must compile your model before training/testing. Use `model.compile(optimizer, loss)")
        patience=5
        lr_decay_factor=0.1
        lr_decay_patience=3
     
        best_loss = float('inf')
        counter = 0

        print('x_train shape :', x_train.shape)
        print('y_train shape :', y_train.shape)
        if validation_data:
            if not isinstance(validation_data, tuple):
                raise TypeError("tuple validation_data is needed.")
            x_val = validation_data[0]
            y_val = validation_data[1]
            self.history['val_loss'] = []
            for metric in self.metrics:
                self.history[f"val_{metric.lower()}"] = []
            print('x_valid shape :', x_val.shape)
    
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            y_train_batch = np.empty((0, y_train.shape[1]))
            binary_predictions = np.empty((0, y_train.shape[1]))
            for x_batch, y_batch in self.create_mini_batches(x_train, y_train, batch_size):
                y_pred = self.forward(x_batch)

                y_train_batch = np.vstack((y_train_batch, y_batch))
                binary_predictions = np.vstack((binary_predictions, convert_to_binary_pred(y_pred)))

                total_loss += self.loss(y_batch, y_pred)
                self.backprop(y_batch, y_pred)

                n_batches += 1

            total_loss /= n_batches

            padding_width = len(str(epochs))
            print(f'\nEpoch {epoch + 1:0{padding_width}d}/{epochs} - loss: {total_loss:.4f}', end="")
            self.history['loss'].append(total_loss)
    
            self.update_history(y_train_batch, binary_predictions)

            # Calculate validation loss and accuracy
            if validation_data:
                val_loss = 0
                n_val_batches = 0

                y_val_batch = np.empty((0, y_val.shape[1]))
                val_binary_predictions = np.empty((0, y_val.shape[1]))
                for x_batch, y_batch in self.create_mini_batches(x_val, y_val, batch_size):
                    y_val_pred = self.forward(x_batch)

                    y_val_batch = np.vstack((y_val_batch, y_batch))
                    val_binary_predictions = np.vstack((val_binary_predictions, convert_to_binary_pred(y_val_pred)))
                    val_loss += self.loss(y_batch, y_val_pred)
                    n_val_batches += 1

                val_loss /= n_val_batches
                self.history['val_loss'].append(val_loss)
                print(f' - val_loss: {val_loss:.4f}', end="")
    
                self.update_history(y_val_batch, val_binary_predictions, True)

                # Check if validation loss is decreasing
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1
    
    
            '''
            # Stop early if the validation loss hasn't improved for 'patience' epochs
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
            '''

        #print('Train accuracy:', accuracy, 'Validation accuracy:', val_accuracy)
        return self

    def predict(self, x_test, y_test):
        y_pred = self.forward(x_test)
        #error = binary_cross_entropy_elem(y_test, y_pred)
        #print('loss:', error[:,0])
        accuracy = accuracy_score(y_test, convert_to_binary_pred(y_pred))
        print('\nAccuracy:', accuracy)
        return y_pred
