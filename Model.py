import json

import numpy as np

import config
import srcs.optimizers as optimizers
from srcs.layers import Dense, Layer
from srcs.losses import binary_crossentropy, binary_crossentropy_derivative
import srcs.losses as losses
from srcs.metrics import (accuracy_score, f1_score, precision_score,
                          recall_score)
from srcs.utils import convert_to_binary_pred, one_hot_encode_labels


class Model():
    def __init__(self, name="Model"):
        self._is_compiled = False
        self.layers = None
        self.n_layers = 0 
        self.optimizer = None
        self.history = {}
        self.metrics = []
        self.name = name

    def create_network(self, net, name=None):
        layers = None
        if isinstance(net, list) and all(isinstance(layer, Layer) for layer in net):
            layers = net
            self.layers = layers
        elif isinstance(net, list) and all(isinstance(layer_data, dict) for layer_data in net):
            print("Creating a neural network...")
            layers = []
            for layer_data in net:
                if layer_data['type'] == 'Dense':
                    if 'weights_initializer' not in layer_data:
                        layers.append(Dense(layer_data['shape'][0],
                                                layer_data['shape'][1], 
                                                activation=layer_data['activation']))
                    else:
                        layers.append(Dense(layer_data['shape'][0],
                                                layer_data['shape'][1], 
                                                activation=layer_data['activation'],
                                                weights_initializer=layer_data['weights_initializer']))
            self.layers = layers
            self.n_layers = len(layers)
        else:
            raise TypeError("Invalid form of input to create a neural network.")

        if name:
            self.name = name

        return layers

    def save_model(self) -> None:
        # Save model configuration as a JSON file
        model_config = self.get_network_topology()
        with open(config.models_dir + config.config_path, 'w') as json_file:
            json.dump(model_config, json_file)
            print(f"> Saving model configuration to '{config.models_dir + config.config_path}'")
        
        # Save model weights as a .npy file
        model_weights = self.get_weights()
        np.savez(config.weights_dir + config.weights_path, *model_weights)
        print(f"> Saving model weights to '{config.weights_dir + config.weights_path}'")


    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, y_true, y_pred) -> None:
        loss_gradient = self.loss_prime(y_true, y_pred)

        self.optimizer.pre_update_params()
        for l in reversed(range(self.n_layers)):
            # print("dloss:", loss_gradient.shape, l)
            self.layers[l].set_activation_gradient(loss_gradient)
            loss_gradient = np.dot(self.layers[l].deltas, self.layers[l].weights.T)
            if l > 0:
                self.optimizer.update_params(self.layers[l], self.layers[l - 1].outputs.T)
            else:
                self.optimizer.update_params(self.layers[l], self.layers[0].inputs.T)
        self.optimizer.post_update_params()

    def get_weights(self) -> list[np.ndarray]:
        weights_and_bias = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                weights_and_bias.append(layer.weights)
                weights_and_bias.append(layer.bias)
        return weights_and_bias

    def set_weights(self, initial_weights):
        if not isinstance(initial_weights, list):
            raise TypeError("Invalid type of initial_weights, a list of weights required.")
        if not len(initial_weights) == 2 * len(self.layers):
            raise ValueError("Invalid input of list: not enought values to set weights and biases.")

        for index, layer in zip(range(0, len(initial_weights), 2), self.layers):
            if isinstance(layer, Dense):
                layer.set_weights(initial_weights[index], initial_weights[index + 1])

    def get_network_topology(self):
        layers = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer_data = {
                    'type': 'Dense',
                    'shape': layer.shape,
                    #'input_shape': layer.shape[0],
                    #'output_shape': layer.shape[1],
                    'activation': f'{layer.activation.__name__}',
                    'weights_initializer': f'{layer.weights_initializer}',
                }
                layers.append(layer_data)
        return layers 

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
                optimizer='rmsprop', 
                loss=None, 
                metrics=None, 
                loss_weights=None, 
                weighted_metrics=None, 
                run_eagerly=None, 
                steps_per_execution=None
        ):

        if isinstance(optimizer, optimizers.Optimizer):
            self.optimizer = optimizer
        elif optimizer == 'sgd':
            self.optimizer = optimizers.SGD()
        elif optimizer == 'rmsprop':
            self.optimizer = optimizers.RMSprop()

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
        elif isinstance(loss, losses.Loss):
            print("loss is a class")
            self.loss = loss.loss
            self.loss_prime = loss.dloss
            self.history['loss'] = []

        if metrics:
            self.metrics = metrics
            for metric in metrics:
                self.history[metric.lower()] = []

        self._is_compiled = True
        return

    def update_history(self, y_true, y_pred, validation_data=None):
        accuracy, precision, recall, f1 = self.evaluate_metrics(y_true, y_pred)
        if validation_data:
            valid = "val_"
        else:
            valid = ""
        print(f' - {valid}loss: {self.history[f"{valid}loss"][-1]:.4f}', end="")
        if 'accuracy' in self.metrics:
            self.history[valid + 'accuracy'].append(accuracy)
            print(f" - {valid}accuracy: {accuracy:.2f}%", end="")
        if 'Precision' in self.metrics:
            self.history[valid + 'precision'].append(precision)
            print(f" - {valid}precision: {precision:.2f}%", end="")
        if 'Recall' in self.metrics:
            self.history[valid + 'recall'].append(recall)
            print(f" - {valid}recall: {recall:.2f}%", end="")

        return accuracy, precision, recall, f1

    def fit(self, 
            x, 
            y, 
            batch_size, 
            validation_data=None):
        if self._is_compiled == False:
            raise RuntimeError("You must compile your model before training/testing. Use `model.compile(optimizer, loss)")
        best_loss = float('inf')
        counter = 0

        total_loss = 0
        n_batches = 0

        n_classes = len(np.unique(y))
        y_train_batch = np.empty((0, n_classes))
        predictions = np.empty((0, n_classes))
        for x_batch, y_batch in self.create_mini_batches(x, y, batch_size):
            y_batch = one_hot_encode_labels(y_batch, n_classes)
            y_pred = self.forward(x_batch)

            y_train_batch = np.vstack((y_train_batch, y_batch))
            predictions = np.vstack((predictions, convert_to_binary_pred(y_pred)))

            total_loss += self.loss(y_batch, y_pred)
            n_batches += 1
            self.backward(y_batch, y_pred)


        total_loss /= n_batches
        self.history['loss'].append(total_loss)
        accuracy, _, _, _ = self.update_history(y_train_batch, predictions)


        # Calculate validation loss and accuracy
        if validation_data:
            if not isinstance(validation_data, tuple):
                raise TypeError("tuple validation_data is needed.")
            x_val, y_val = validation_data
            if 'val_loss' not in self.history:
                self.history['val_loss'] = []
                for metric in self.metrics:
                    self.history[f"val_{metric.lower()}"] = []
            # print('x_valid shape :', x_val.shape)

            val_y_pred = self.forward(x_val)
            y_val = one_hot_encode_labels(y_val, n_classes)
            val_loss = self.loss(y_val, val_y_pred)
            self.history['val_loss'].append(val_loss)
            val_accuracy, _, _, _ = self.update_history(y_val, convert_to_binary_pred(val_y_pred), validation_data)

        '''
            # Check if validation loss is decreasing
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
            # Stop early if the validation loss hasn't improved for 'patience' epochs
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        if accuracy:
            print('\nTrain accuracy:', accuracy)
        if val_accuracy:
            print('\nValidation accuracy:', val_accuracy)
            '''
        return self

    def predict(self, x, threshold=0.5):
        y_pred = self.forward(x)
        #error = binary_cross_entropy_elem(y_test, y_pred)
        #print('loss:', error[:,0])
        return convert_to_binary_pred(y_pred) 
