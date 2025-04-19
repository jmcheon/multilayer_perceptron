import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlp.losses as losses
import mlp.optimizers as optimizers
from autoencoder.model import AutoEncoder, DeepAutoEncoder
from autoencoder.train import Trainer
from classification.ModelPlotter import ModelPlotter
from keras.datasets import mnist

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    y_train = y_train.reshape(-1, 1)

    model = DeepAutoEncoder(784, 32)
    optim = optimizers.Adam(model.parameters(), learning_rate=1e-4)
    loss_fn = losses.BCELoss()
    trainer = Trainer(model, optim, loss_fn)

    history = trainer.fit(x_train, x_train, batch_size=128, epochs=10)
    plotter = ModelPlotter()

    plotter.set_model_histories([history])
    plotter.plot()
