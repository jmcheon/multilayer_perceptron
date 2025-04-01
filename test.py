import classification.config as config
import mlp.losses as losses
import mlp.optimizers as optimizers
from classification.ModelPlotter import ModelPlotter
from mlp.activations import ReLU, Sigmoid
from mlp.layers import Linear
from mlp.Sequential import Sequential
from mlp.utils import load_split_data
from train import Trainer

if __name__ == "__main__":
    train_path = config.data_dir + config.train_path
    valid_path = config.data_dir + config.valid_path

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    input_dim = 10
    output_dim = 1
    # model = Sequential(
    #     [
    #         Dense(input_dim, 20, activation="relu"),
    #         Dense(20, 10, activation="relu"),
    #         Dense(10, 5, activation="relu"),
    #         Dense(5, output_dim, activation="sigmoid"),
    #     ]
    # )
    # print(model)
    # print(model.layers[0].weights.shape)
    # parameters = model.parameters()
    # print(parameters[0].shape)
    model = Sequential(
        [
            Linear(x_train.shape[1], 20),
            ReLU(),
            Linear(20, 10),
            ReLU(),
            Linear(10, 5),
            ReLU(),
            Linear(5, y_train.shape[1]),
            Sigmoid(),
        ]
    )
    print(model)
    # print(model.layers[0].weights.shape)
    parameters = model.parameters()
    # print(parameters[0].shape)
    for i in range(len(parameters)):
        print(parameters[i].shape)
    optim = optimizers.SGD(learning_rate=1e-3)
    loss_fn = losses.BCELoss()
    trainer = Trainer(model, optim, loss_fn)

    history = trainer.fit(x_train, y_train, epochs=100)
    plotter = ModelPlotter()

    plotter.set_model_histories([history])
    plotter.plot()
