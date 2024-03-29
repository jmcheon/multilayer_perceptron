import argparse
import json
import sys

import config
import srcs.optimizers as optimizers
from Model import Model
from ModelPlotter import ModelPlotter
from ModelTrainer import ModelTrainer
from srcs.utils import load_config, load_split_data, load_weights


def prediction():
    weights_path = config.weights_dir + config.weights_path
    config_path = config.models_dir + config.config_path
    test_path = config.data_dir + config.test_path

    x, y = load_split_data(test_path)
    weights = load_weights(weights_path)
    config_data = load_config(config_path)


    model = Model()
    model.create_network(config_data)
    model.set_weights(list(weights))
    model.predict(x, y)

def set_argparse():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--params", type=str, default=None, required=True,
                        help="Path of model parameters")

    parser.add_argument("-s", "--split", type=str, default=None,
                        help="Split dataset into train and validation sets.")

    parser.add_argument("-t", "--train", action="store_true", default=False,
                        help="Train with dataset.")

    parser.add_argument("-p", "--predict", action="store_true", default=False,
                        help="Predict using saved model.")

    parser.add_argument("-c", "--compare", type=str, default=None, nargs='?', choices=["optimizers"],
                        help="Compare models by plotting learning curves.")

    # for model compiling
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Gradient descent optimizer')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                        help='Loss function')
    
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    return parser

if __name__ == "__main__":
    train_path = config.data_dir + config.train_path 
    valid_path = config.data_dir + config.valid_path

    input_shape = 30
    output_shape = 1

    try:
        with open('description.txt', 'r') as file:
            description = file.read()
    except:
            description = "multilayer perceptron"

    parser = set_argparse()
    args = parser.parse_args()

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    trainer = ModelTrainer(input_shape, output_shape, train_path, valid_path)
    plotter = ModelPlotter()

    histories, model_names = [], []
    params = None

    if args.params:
        params = load_config(args.params)
        #print(params)
    if args.split:
        split_dataset_save(args.split, train_path, valid_path, train_size=0.8, random_state=42)

    elif args.train:
        model = trainer.create(params)
        optimizer_list = []
        print(len(trainer.model_list))
        for _ in range(len(trainer.model_list)):
            optimizer_list.append(optimizers.SGD(learning_rate=1e-3))
        histories, model_names = trainer.train(
                                trainer.model_list,
                                x_train, 
                                y_train, 
                                optimizer_list,
                                loss=args.loss,
                                metrics=['accuracy', 'Precision', 'Recall'],
                                batch_size=args.batch_size, 
                                epochs=args.epochs, 
                                validation_data=(x_val, y_val),
                )
        if isinstance(model, Model):
            model.save_model()

    elif args.predict:
        prediction()
    elif args.compare == "optimizers":
        histories, model_names = trainer.optimizer_test()
    else:
        print(f"Usage: python {sys.argv[0]} -h")

    if histories and model_names:
        plotter.set_model_histories(histories)
        plotter.set_model_names(model_names)
        plotter.plot()
