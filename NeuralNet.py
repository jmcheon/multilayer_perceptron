import numpy as np

import srcs.optimizers as optimizers
import srcs.losses as losses
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
        loss_gradient = self.loss.dloss(y, y_pred)

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
            self.loss = losses.BCELoss()
            self.history['loss'] = []
        elif loss == 'mse':
            self.loss = losses.MSELoss()
            self.history['loss'] = []
        elif isinstance(loss, losses.Loss):
            self.loss = loss
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

        y_train_batch = np.empty((0, self.shape[1]))
        y_train_pred = np.empty((0, self.shape[1]))
        for x_batch, y_batch in self.create_mini_batches(x, y, batch_size):
            y_batch = self.one_hot_encode_labels(y_batch)
            y_pred = self.forward(x_batch)
            self.backward(y_batch, y_pred)

            y_train_batch = np.vstack((y_train_batch, y_batch))
            y_train_pred = np.vstack((y_train_pred, y_pred))

        total_loss = self.loss.loss(y_train_batch, y_train_pred)
        self.update_history(y_train_batch, self.predict(y_train_pred, True))
        self.history['loss'].append(total_loss)


        # Calculate validation loss and accuracy
        if validation_data:
            if not isinstance(validation_data, tuple):
                raise TypeError("tuple validation_data is needed.")
            x_val, y_val = validation_data
            if 'val_loss' not in self.history:
                self.history['val_loss'] = []
                for metric in self.metrics:
                    self.history[f"val_{metric.lower()}"] = []

            y_val_pred = self.forward(x_val)
            y_val = self.one_hot_encode_labels(y_val)
            val_loss = self.loss.loss(y_val, y_val_pred)
            self.history['val_loss'].append(val_loss)
            self.update_history(y_val, self.predict(y_val_pred, True), validation_data)
        self.print_history()

        return self

    def predict(self, x, y_pred=False, threshold=0.5) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Args:
            x (ndarray): Input data for prediction.

        Returns:
            ndarray: Predicted labels.
        """
        if y_pred == False:
            y_pred = self.forward(x)
        else:
            y_pred = x

        if self.shape[1] > 1:
            out = np.argmax(y_pred, axis=1)
            out = self.one_hot_encode_labels(out)
            return out
        return (y_pred > threshold).astype(int)