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


class Encoder(mlp.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = [
            mlp.Linear(in_features, 128),
            mlp.ReLU(),
            mlp.Linear(128, 64),
            mlp.ReLU(),
            mlp.Linear(64, out_features),
            mlp.ReLU(),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(mlp.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = [
            mlp.Linear(out_features, 64),
            mlp.ReLU(),
            mlp.Linear(64, 128),
            mlp.ReLU(),
            mlp.Linear(128, in_features),
            mlp.Sigmoid(),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class DeepAutoEncoder(mlp.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.encoder = Encoder(in_features, out_features)
        self.decoder = Decoder(in_features, out_features)

        self.layers = [
            self.encoder,
            self.decoder,
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
