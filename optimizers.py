import numpy as np


class Optimizer:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        pass

    def post_update_params(self):
        self.iterations += 1

    def create_mini_batches(self, x, y, batch_size):
        if isinstance(batch_size, str) and batch_size == 'batch':
            yield x, y
        else:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)

            for start_idx in range(0, x.shape[0] - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                yield x[batch_idx], y[batch_idx]

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

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, decay=0., momentum=0., name='SGD'):
        super().__init__(learning_rate, decay, momentum)
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        '''

    def update_params(self, layer, prev_layer_output):
        dweights = np.dot(prev_layer_output, layer.deltas)
        dbiases = np.sum(layer.deltas, axis=0, keepdims=True).reshape(-1)
        '''
        print('dbiases shape:', dbiases.shape)
        print('prev_layer_output shape:', prev_layer_output.shape)
        print('layer.deltas shape:', layer.deltas.shape)
        print('dweights shape:', dweights.shape)
        '''
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.bias)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                    dweights
            bias_updates = -self.current_learning_rate * \
                    dbiases

        layer.weights += weight_updates
        layer.bias += bias_updates
