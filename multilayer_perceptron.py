import argparse
import json
import sys

import numpy as np

from DenseLayer import DenseLayer
from NeuralNet import NeuralNet
from plots import compare_models, compare_optimizers, plot_learning_curves
from utils import load_split_data, split_dataset_save


def prediction():
    weights_path = 'saved_model_weights.npy'
    config_path = 'saved_model_config.json'
    data_path = 'data_test.csv'

    x, y = load_split_data(data_path)
    data_test = np.hstack((x, y))

    try:
        weights = np.load(weights_path, allow_pickle=True)
    except:
        print(f"Input file errer: {weights_path} doesn't exist.")
        sys.exit()
    try:
        with open(config_path, 'r') as file:
            json_config = json.load(file)
        config = json.loads(json_config)
    except:
        print(f"Input file errer: {config_path} doesn't exist.")
        sys.exit()

    model = NeuralNet()
    model.create_network(config['network_topology'])
    model.set_weights(list(weights))
    model.predict(data_test)

def train_plot_save():
    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model = NeuralNet()
    network = model.create_network([
        DenseLayer(input_shape, 20, activation='sigmoid'),
        DenseLayer(20, 10, activation='sigmoid', weights_initializer='random'),
        DenseLayer(10, 1, activation='sigmoid', weights_initializer='random'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = model.fit(data_train, data_valid, loss='binary_cross_entropy', learning_rate=1e-3, batch_size=2, epochs=30)
    plot_learning_curves(epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list)
    model.save_model()

def multiple_models_test():
    print("Compare multiple models...")

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model1 = NeuralNet()
    model1.create_network([
        DenseLayer(input_shape, 20, activation='sigmoid'),
        DenseLayer(20, 10, activation='sigmoid', weights_initializer='random'),
        DenseLayer(10, 1, activation='sigmoid', weights_initializer='random'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model2 = NeuralNet()
    model2.create_network([
        DenseLayer(input_shape, 15, activation='sigmoid'),
        DenseLayer(15, 1, activation='sigmoid', weights_initializer='random'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model3 = NeuralNet()
    model3.create_network([
        DenseLayer(input_shape, 1, activation='sigmoid'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model_list = [model1, model2, model3]
    compare_models(data_train, data_valid, model_list, loss='binary_cross_entropy', learning_rate=1e-3, batch_size=2, epochs=50)

def optimizer_test():
    print("Compare optimizers...")

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model1 = NeuralNet(nesterov=False)
    model1.create_network([
        DenseLayer(input_shape, 1, activation='sigmoid'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='zero')
        ])

    model2 = NeuralNet(nesterov=True)
    model2.create_network([
        DenseLayer(input_shape, 1, activation='sigmoid'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='zero')
        ])

    model3 = NeuralNet(optimizer='rmsprop')
    model3.create_network([
        DenseLayer(input_shape, 1, activation='sigmoid'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='zero')
        ])

    model_list = [model1, model3]
    compare_optimizers(data_train, data_valid, model_list, loss='binary_cross_entropy', learning_rate=1e-3, batch_size='batch', epochs=30)

def same_model_test():
    print("Compare same models...")

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model1 = NeuralNet()
    model1.create_network([
        DenseLayer(input_shape, 1, activation='sigmoid'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='zero')
        ])

    model2 = NeuralNet()
    model2.create_network([
        DenseLayer(input_shape, 1, activation='sigmoid'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='zero')
        ])

    model3 = NeuralNet()
    model3.create_network([
        DenseLayer(input_shape, 1, activation='sigmoid'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='zero')
        ])

    model_list = [model1, model2, model3]
    compare_models(data_train, data_valid, model_list, loss='binary_cross_entropy', learning_rate=1e-3, batch_size='batch', epochs=30)

def bonus_test(historic=False):
    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model = NeuralNet()
    model.create_network([
        DenseLayer(input_shape, 1, activation='sigmoid'),
        DenseLayer(1, output_shape, activation='sigmoid', weights_initializer='zero')
        ])

    epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = model.fit(data_train, data_valid, loss='binary_cross_entropy', learning_rate=1e-2, batch_size=5, epochs=70)
    plot_learning_curves(epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list)
    if historic == True:
        print("medel's metrics historic:\n", model.metrics_historic)

if __name__ == "__main__":
    train_path = "data_train.csv"
    valid_path = "data_valid.csv"

    input_shape = 30
    output_shape = 2

    try:
        with open('description.txt', 'r') as file:
            description = file.read()
    except:
            description = "multilayer perceptron"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-s", "--split", type=str, default=None,
                        help="Split dataset into train and validation sets.")

    parser.add_argument("-t", "--train", action="store_true", default=False,
                        help="Train with dataset.")

    parser.add_argument("-p", "--predict", action="store_true", default=False,
                        help="Predict using saved model.")

    parser.add_argument("-c", "--compare", type=str, default=None, nargs='?', choices=["models", "optimizers"],
                        help="Compare models by plotting learning curves.")

    args = parser.parse_args()

    if args.split:
        split_dataset_save(args.split, train_path, valid_path, train_size=0.8, random_state=42)
    elif args.train:
        train_plot_save()
    elif args.predict:
        prediction()
    elif args.compare == "models":
        multiple_models_test()
    elif args.compare == "optimizers":
        optimizer_test()
    elif args.compare == "same models":
        same_model_test()
    elif args.compare == "early stopping":
        bonus_test()
    elif args.compare == "historic":
        bonus_test(True)
    else:
        print(f"Usage: python {sys.argv[0]} -h")
