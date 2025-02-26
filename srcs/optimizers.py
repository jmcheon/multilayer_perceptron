import numpy as np


class Optimizer:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        pass

    def post_update_params(self):
        self.iterations += 1


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, decay=0.0, momentum=0.0, name="SGD"):
        super().__init__(learning_rate, decay, momentum)
        self.name = name

    def update_params(self, layer, prev_layer_output):
        dweights = np.dot(prev_layer_output, layer.deltas)
        dbias = np.sum(layer.deltas, axis=0, keepdims=True)
        """
        print('dbias shape:', dbias.shape)
        print('prev_layer_output shape:', prev_layer_output.shape)
        print('layer.deltas shape:', layer.deltas.shape)
        print('dweights shape:', dweights.shape)
        """
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for bias yet either.
                layer.bias_momentums = np.zeros_like(layer.bias)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = (
                self.momentum * layer.weight_momentums - self.current_learning_rate * dweights
            )
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * dbias
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * dweights
            bias_updates = -self.current_learning_rate * dbias

        layer.weights += weight_updates
        layer.bias += bias_updates


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9, name="RMSprop"):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho
        self.name = name

    def update_params(self, layer, prev_layer_output):
        dweights = np.dot(prev_layer_output, layer.deltas)
        dbias = np.sum(layer.deltas, axis=0, keepdims=True)

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * dbias**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += (
            -self.current_learning_rate * dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.bias += (
            -self.current_learning_rate * dbias / (np.sqrt(layer.bias_cache) + self.epsilon)
        )


class Adam(Optimizer):
    def __init__(
        self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999, name="Adam"
    ):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.name = name

    def update_params(self, layer, prev_layer_output):
        dweights = np.dot(prev_layer_output, layer.deltas)
        dbias = np.sum(layer.deltas, axis=0, keepdims=True)
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.bias)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * dbias
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * dbias**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.bias += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )
