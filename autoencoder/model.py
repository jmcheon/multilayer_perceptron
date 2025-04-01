import mlp


class AutoEncoder(mlp.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.encoder = mlp.Linear(in_features, out_features)
        self.decoder = mlp.Linear(out_features, in_features)

        self.relu = mlp.ReLU()
        self.sigmoid = mlp.Sigmoid()

        self.layers = [self.encoder, self.relu, self.decoder, self.sigmoid]

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)

        x = self.decoder(x)
        x = self.sigmoid(x)

        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        return grad_output
