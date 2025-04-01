from mlp.layers import Dense
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
    # print(model.layers)
    print(model)
    # print(model.layers[0].weights.shape)
    # parameters = model.parameters()
    # print(parameters[0].shape)
