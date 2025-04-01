class Module:
    def foward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Given upstream gradients, compute local gradients and return them to propagate further
        """
        raise NotImplementedError

    def parameters(self):
        """
        Return a list of tensors (weights/biases)
        """
        return []

    def zero_grad(self):
        pass

    def __call__(self, x):
        return self.forward(x)
