import argparse
import json
import sys

import numpy as np

import optimizers
from DenseLayer import DenseLayer
from NeuralNet import NeuralNet
from plots import plot_learning_curves, plot_models
from utils import load_split_data, split_dataset_save


def load_weights(filename):
    try:
        weights = np.load(filename, allow_pickle=True)
    except:
        print(f"Input file errer: {filename} doesn't exist.")
        sys.exit()
    return weights

def prediction():
    weights_path = 'saved_model_weights.npy'
    config_path = 'saved_model_config.json'
    data_path = 'data_test.csv'

    x, y = load_split_data(data_path)

    weights = load_weights(weights_path)
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
    model.predict(x, y)

def create_model():
    model = NeuralNet()
    network = model.create_network([
        DenseLayer(input_shape, 20, activation='relu'),
        DenseLayer(20, 10, activation='relu', weights_initializer='random'),
        DenseLayer(10, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model.compile(
            #optimizer='sgd', 
            optimizer=optimizers.SGD(learning_rate=1e-3),
            #optimizer=optimizers.RMSprop(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall'],
    )
    weights = load_weights('saved_tensorflow_weights.npy')
    for i in range(len(weights)):
        print('weights shape:', weights[i].shape)
    model.set_weights(list(weights))
    #print(model.get_weights())

    return model

def train_model(plot=True, save=True):
    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)
    print(x_train.shape, y_train.shape)

    model = create_model()

    history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val), 
            batch_size=1, 
            epochs=30
    )
    if plot:
        plot_learning_curves(history)
    if save:
        model.save_model()
    return model

def multiple_models_test():
    print("Compare multiple models...")

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    model1 = NeuralNet()
    model1.create_network([
        DenseLayer(input_shape, 20, activation='relu'),
        DenseLayer(20, 10, activation='relu', weights_initializer='random'),
        DenseLayer(10, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ], name="model1")

    model2 = NeuralNet()
    model2.create_network([
        DenseLayer(input_shape, 15, activation='relu'),
        DenseLayer(15, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ], name="model2")

    model3 = NeuralNet()
    model3.create_network([
        DenseLayer(input_shape, 5, activation='relu'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ], name="model3")

    model_list = [
            (model1, optimizers.SGD(learning_rate=1e-4)), 
            (model2, optimizers.SGD(learning_rate=1e-4)),
            (model3, optimizers.SGD(learning_rate=1e-4)),
    ]
    #model_list = [model1, model2, model3]
    plot_models(
            x_train, y_train, validation_data=(x_val, y_val), 
            #optimizer=optimizers.SGD(learning_rate=1e-3),
            model_list=model_list, 
            loss='binary_crossentropy', 
            metrics=['accuracy', 'Precision', 'Recall'],
            batch_size=1, 
            epochs=50
    )

def optimizer_test():
    print("Compare optimizers...")

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    model1 = NeuralNet()
    model1.create_network([
        DenseLayer(input_shape, 20, activation='relu'),
        DenseLayer(20, 10, activation='relu', weights_initializer='random'),
        DenseLayer(10, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model2 = NeuralNet()
    model2.create_network([
        DenseLayer(input_shape, 20, activation='relu'),
        DenseLayer(20, 10, activation='relu', weights_initializer='random'),
        DenseLayer(10, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model3 = NeuralNet()
    model3.create_network([
        DenseLayer(input_shape, 20, activation='relu'),
        DenseLayer(20, 10, activation='relu', weights_initializer='random'),
        DenseLayer(10, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model_list = [
            (model1, optimizers.SGD(learning_rate=1e-4)), 
            (model2, optimizers.RMSprop(learning_rate=1e-4)),
            (model3, optimizers.Adam(learning_rate=1e-4)),
    ]

    weights = load_weights('saved_tensorflow_weights.npy')
    for i in range(len(weights)):
        print('weights shape:', weights[i].shape)
    for (model, optimizer) in model_list:
        model.set_weights(list(weights))

    plot_models(
            x_train, y_train, validation_data=(x_val, y_val), 
            model_list=model_list, 
            metrics=['accuracy', 'Precision', 'Recall'],
            loss='binary_crossentropy', 
            batch_size=1, 
            epochs=30
    )

def same_model_test():
    print("Compare same models...")

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    model1 = NeuralNet()
    model1.create_network([
        DenseLayer(input_shape, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model2 = NeuralNet()
    model2.create_network([
        DenseLayer(input_shape, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model3 = NeuralNet()
    model3.create_network([
        DenseLayer(input_shape, 5, activation='relu', weights_initializer='random'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model_list = [
            (model1, optimizers.SGD(learning_rate=1e-3)), 
            (model2, optimizers.SGD(learning_rate=1e-3)),
            (model3, optimizers.SGD(learning_rate=1e-3)),
    ]
    plot_models(
            x_train, y_train, validation_data=(x_val, y_val), 
            model_list=model_list, 
            loss='binary_crossentropy', 
            metrics=['accuracy', 'Precision', 'Recall'],
            batch_size='batch', 
            epochs=30
    )

def bonus_test(history=False):
    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    model = NeuralNet()
    model.create_network([
        DenseLayer(input_shape, 5, activation='relu'),
        DenseLayer(5, output_shape, activation='sigmoid', weights_initializer='random')
        ])

    model_history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val), 
            loss='binary_crossentropy',
            learning_rate=1e-2,
            batch_size=5,
            epochs=30
    )
    plot_learning_curves(model_history)
    if history == True:
        print("medel's history:\n", model.history)

if __name__ == "__main__":
    train_path = "data_train.csv"
    valid_path = "data_valid.csv"

    input_shape = 30
    output_shape = 1

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
        train_model()
    elif args.predict:
        prediction()
    elif args.compare == "models":
        multiple_models_test()
    elif args.compare == "optimizers":
        optimizer_test()
    elif args.compare == "same":
        same_model_test()
    elif args.compare == "early stopping":
        bonus_test()
    elif args.compare == "history":
        bonus_test(True)
    else:
        print(f"Usage: python {sys.argv[0]} -h")
