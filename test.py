from mlp.activations import ReLU, Sigmoid
from mlp.layers import Dense, Linear
from mlp.Sequential import Sequential

if __name__ == "__main__":
    input_dim = 10
    output_dim = 1
    model = Sequential(
        [
            Dense(input_dim, 20, activation="relu"),
            Dense(20, 10, activation="relu"),
            Dense(10, 5, activation="relu"),
            Dense(5, output_dim, activation="sigmoid"),
        ]
    )
    print(model)
    print(model.layers[0].weights.shape)
    parameters = model.parameters()
    print(parameters[0].shape)
    model = Sequential(
        [
            Linear(input_dim, 20),
            ReLU(),
            Linear(20, 10),
            ReLU(),
            Linear(10, 5),
            ReLU(),
            Linear(5, output_dim),
            Sigmoid(),
        ]
    )
    print(model)
    print(model.layers[0].weights.shape)
    parameters = model.parameters()
    print(parameters[0].shape)
