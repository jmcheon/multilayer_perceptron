class Module:
    def foward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Given upstream gradients, compute local gradients and return them to propagate further
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        return grad_output

    def parameters(self):
        """
        Return a list of layers containing (weights/biases)
        """
        layers = []

        for layer in self.layers:
            if hasattr(layer, "layers"):
                layers.extend(layer.layers)
            else:
                layers.append(layer)
        return layers

    def zero_grad(self):
        pass

    def __call__(self, x):
        return self.forward(x)
