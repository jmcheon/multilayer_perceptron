import json

import numpy as np

import config
from srcs.layers import Dense, Layer
from srcs.metrics import accuracy_score, f1_score, precision_score, recall_score


class Model():
    """
    Base Model Class.

    Methods:
        create_network: Initializes the network architecture.
        save_model: Saves the model parameters to a file.
        get_weights: Retrieves the model weights.
        set_weights: Sets the model weights.
        get_network_topology: Retrieves the network topology.

        evaluate_metrics: Evaluates specified metrics on the model.
        create_mini_batch: Creates mini-batches from the training data.

        print_history: Prints the training history.
        update_history: Updates the training history with new metrics.
    """
    def __init__(self, name="Model"):
        self.layers = None
        self.n_layers = 0 
        self.history = {}
        self.metrics = []
        self.name = name

    def create_network(self, net, name=None):
        """
        Initializes the network architecture.

        Args:
            net (list): A list containing model topology.
        Returns:
            list: A list containing model topology.
        """
        layers = None
        if isinstance(net, list) and all(isinstance(layer, Layer) for layer in net):
            layers = net
            self.layers = layers
            self.n_layers = len(layers)
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
        """
        Saves the model parameters to a file.
        """
        # Save model configuration as a JSON file
        model_config = self.get_network_topology()
        with open(config.models_dir + config.config_path, 'w') as json_file:
            json.dump(model_config, json_file)
            print(f"> Saving model configuration to '{config.models_dir + config.config_path}'")
        
        # Save model weights as a .npy file
        model_weights = self.get_weights()
        np.savez(config.weights_dir + config.weights_path, *model_weights)
        print(f"> Saving model weights to '{config.weights_dir + config.weights_path}'")

    def get_weights(self) -> list[np.ndarray]:
        """
        Retrieves the model weights.

        Returns:
            list: A list containing the model weights.
        """
        weights_and_bias = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                weights_and_bias.append(layer.weights)
                weights_and_bias.append(layer.bias)
        return weights_and_bias

    def set_weights(self, initial_weights) -> None:
        """
        Sets the model weights.

        Args:
            weights (list): A list containing the new weights.

        Returns:
            None.
        """
        if not isinstance(initial_weights, list):
            raise TypeError("Invalid type of initial_weights, a list of weights required.")
        if not len(initial_weights) == 2 * len(self.layers):
            raise ValueError("Invalid input of list: not enought values to set weights and biases.")

        for index, layer in zip(range(0, len(initial_weights), 2), self.layers):
            if isinstance(layer, Dense):
                layer.set_weights(initial_weights[index], initial_weights[index + 1])

    def get_network_topology(self) -> list:
        """
        Retrieves the network topology.

        Returns:
            list: A list describing the network topology.
        """
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

    def evaluate_metrics(self, y, y_pred):
        """
        Evaluates specified metrics on the model.

        Args:
            y (ndarray): True labels.
            y_pred (ndarray): Predicted labels.

        Returns:
            accuracy : The accuracy value.
            precision : The precision value.
            recall : The recall value.
            f1: The f1 score value.
        """
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        return accuracy, precision, recall, f1

    def create_mini_batches(self, x, y, batch_size):
        """
        Creates mini-batches from the training data.

        Args:
            x (ndarray): The x training data.
            y (ndarray): The y training data.
            batch_size (int): The size of each mini-batch.
        """

        if isinstance(batch_size, str) and batch_size == 'batch':
            yield x, y
        else:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)

            for start_idx in range(0, x.shape[0] - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                yield x[batch_idx], y[batch_idx]

    def print_history(self) -> None:
        """
        Prints the training history.

        Returns:
            None.
        """
        if len(self.history) == 0:
            print("It hasn't trained yet.")
            return
        for metric in self.history:
            print(f"epoch {metric['epoch']} - loss: {metric['loss']} - val_loss: {metric['val_loss']}")

    def update_history(self, y, y_pred, validation_data=None):
        """
        Updates the training history with new metrics.

        Args:
            y (ndarray): True labels.
            y_pred (ndarray): Predicted labels.
            validation_data (tuple): Validation dataset of x_val, y_val.

        Returns:
            accuracy : The accuracy value.
            precision : The precision value.
            recall : The recall value.
            f1: The f1 score value.
        """
        accuracy, precision, recall, f1 = self.evaluate_metrics(y, y_pred)
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