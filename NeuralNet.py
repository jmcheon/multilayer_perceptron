import numpy as np

import srcs.optimizers as optimizers
import srcs.losses as losses
from srcs.utils import convert_to_binary_pred, one_hot_encode_labels
from Model import Model


class NeuralNet(Model):
    """
    Neural Network Class inherited from Base Model Class.

    Methods:
        forward: Performs a forward propagation through the network.
        backward: Performs a backward propagation through the network.
        compile: Compiles the model with specified loss and optimizer.
        fit: Trains the model on the provided data.
        predict: Makes predictions using the trained model.
    """
    def __init__(self, name="Model"):
        super().__init__(name)
        self._is_compiled = False
        self.optimizer = None

    def forward(self, input_data):
        """
        Performs a forward propagation through the network.

        Args:
            input_data (ndarray): Input data for the network.

        Returns:
            ndarray: Output of the network after the forward pass.
        """
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, y, y_pred) -> None:
        """
        Performs a backward propagation through the network.

        Args:
            y (ndarray): True labels.
            y_pred (ndarray): Predicted labels.

        Returns:
            dict: Gradients of the network parameters.
        """
        gradients = {}
        loss_gradient = self.loss_prime(y, y_pred)

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

        return gradients

    def compile(self, 
                optimizer='rmsprop', 
                loss=None, 
                metrics=None, 
        ):
        """
        Compiles the model with specified loss and optimizer.

        Args:
            optimizer (str, Optimizer): The optimizer to use in string or an instance of an optimizer class.
            loss (str, Loss): The loss function to use in string or an instance of a loss class.
            metrics (list): The metrics to use.

        Returns:
            None
        """

        if isinstance(optimizer, optimizers.Optimizer):
            self.optimizer = optimizer
        elif optimizer == 'sgd':
            self.optimizer = optimizers.SGD()
        elif optimizer == 'rmsprop':
            self.optimizer = optimizers.RMSprop()

        if not loss:
            raise ValueError("No loss found. You may have forgotten to provide a `loss` argument in the `compile()` method.")
        elif loss == 'binary_crossentropy':
            self.loss = losses.binary_crossentropy
            self.loss_prime = losses.binary_crossentropy_derivative
            self.history['loss'] = []
        elif loss == 'mse':
            self.loss = losses.mse
            self.loss_prime = losses.mse_derivative
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

    def fit(self, 
            x, 
            y, 
            batch_size, 
            validation_data=None):
        """
        Trains the model on the provided data.

        Args:
            x (ndarray): Training data inputs.
            y (ndarray): Training data labels.
            batch_size (int): Size of each training batch.
            validation_data (tuple): Validation dataset of x_val, y_val.

        Returns:
            self: returns the instance itself.
        """
        if self._is_compiled == False:
            raise RuntimeError("You must compile your model before training/testing. Use `model.compile(optimizer, loss)")

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

        return self

    def predict(self, x) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Args:
            x (ndarray): Input data for prediction.

        Returns:
            ndarray: Predicted labels.
        """
        y_pred = self.forward(x)
        #error = binary_cross_entropy_elem(y_test, y_pred)
        #print('loss:', error[:,0])
        return convert_to_binary_pred(y_pred) 
