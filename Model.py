import json

import numpy as np

import config
from srcs.layers import Dense, Layer
from srcs.metrics import accuracy_score, f1_score, precision_score, recall_score


class Model():
    """
    Base Model Class.

    Methods:
        create_network: Initializes the model architecture.
        save_topology: Saves the model topology to a file.
        save_parameters: Saves the model parameters to a file.

        get_parameters: Retrieves the model parameters.
        set_parameters: Sets the model parameters.
        get_topology: Retrieves the model topology.

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

    def save_topology(self, filepath) -> None:
        """
        Saves the model topology to a file.

        Args:
            filepath (str): Path to the file where model topology will be saved.
        """
        topology = self.get_topology()
        # print(f"filepaht: {filepath}, for model topology")
        # Save model topology as a JSON file
        with open(filepath + ".json", 'w') as json_file:
            json.dump(topology, json_file)
            print(f"> Saving model configuration to '{filepath}.json'")

    def save_parameters(self, filepath) -> None:
        """
        Saves the model parameters to a file.

        Args:
            filepath (str): Path to the file where parameters will be saved.
        """
        # print(f"filepaht: {filepath}, for model parameters")
        # Save model parameters as a .npz file
        parameters = self.get_parameters()
        np.savez(filepath, *parameters)
        print(f"> Saving model parameters to '{filepath}.npz'")

    def get_parameters(self) -> list[np.ndarray]:
        """
        Retrieves the model parameters.

        Returns:
            list: A list containing the model parameters.
        """
        parameters = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                parameters.append(layer.weights)
                parameters.append(layer.bias)
        return parameters

    def set_parameters(self, initial_parameters) -> None:
        """
        Sets the model parameters.

        Args:
            weights (list): A list containing the new parameters.

        Returns:
            None.
        """
        if not isinstance(initial_parameters, list):
            raise TypeError("Invalid type of initial_parameters, a list of parameters required.")
        if not len(initial_parameters) == 2 * len(self.layers):
            raise ValueError("Invalid input of list: not enought values to set parameters.")

        for index, layer in zip(range(0, len(initial_parameters), 2), self.layers):
            if isinstance(layer, Dense):
                layer.set_parameters(initial_parameters[index], initial_parameters[index + 1])

    def get_topology(self) -> list:
        """
        Retrieves the model topology.

        Returns:
            list: A list describing the model topology.
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
            print(f" - {metric}: {self.history[f'{metric}'][-1]:.4f}", end="")

    def update_history(self, y, y_pred, validation_data=None) -> None:
        """
        Updates the training history with new metrics.

        Args:
            y (ndarray): True labels.
            y_pred (ndarray): Predicted labels.
            validation_data (tuple): Validation dataset of x_val, y_val.

        Returns:
            None.
        """
        accuracy, precision, recall, f1 = self.evaluate_metrics(y, y_pred)
        if validation_data:
            valid = "val_"
        else:
            valid = ""
        if 'accuracy' in self.metrics:
            self.history[valid + 'accuracy'].append(accuracy)
        if 'Precision' in self.metrics:
            self.history[valid + 'precision'].append(precision)
        if 'Recall' in self.metrics:
            self.history[valid + 'recall'].append(recall)