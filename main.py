import argparse
import sys, os

import config
from srcs.metrics import accuracy_score
import srcs.optimizers as optimizers
from Model import Model
from NeuralNet import NeuralNet
from ModelPlotter import ModelPlotter
from ModelTrainer import ModelTrainer
from srcs.utils import load_topology, load_split_data, load_parameters, split_dataset_save
import srcs.losses as losses


def prediction(filename):
    parameters_path = config.parameters_dir + filename
    topology_path = config.topologies_dir + filename + ".json" 
    test_path = config.data_dir + config.test_path

    x, y = load_split_data(test_path)
    parameters = load_parameters(parameters_path)
    topology = load_topology(topology_path)


    model = NeuralNet()
    # print("Net doc:", model.__doc__)
    model.create_network(topology)
    model.set_parameters(list(parameters))
    y_pred = model.predict(x)
    y = model.one_hot_encode_labels(y)
    accuracy = accuracy_score(y, y_pred)
    print('\nAccuracy:', accuracy)

def set_argparse():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--topology", type=str, default=None, required=True,
                        help="Path of model topology")

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
    
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    return parser

if __name__ == "__main__":
    train_path = config.data_dir + config.train_path 
    valid_path = config.data_dir + config.valid_path

    try:
        with open('description.txt', 'r') as file:
            description = file.read()
    except:
            description = "multilayer perceptron"

    parser = set_argparse()
    args = parser.parse_args()

    x_train, y_train = load_split_data(train_path)
    x_val, y_val = load_split_data(valid_path)

    trainer = ModelTrainer()
    plotter = ModelPlotter()

    histories, model_names = [], []
    topology = None

    if args.topology:
        topology = load_topology(args.topology)
        filename = os.path.basename(args.topology)
        filename = os.path.splitext(filename)[0]
    if args.split:
        split_dataset_save(args.split, train_path, valid_path, train_size=0.8, random_state=42)

    elif args.train:
        model = trainer.create(topology)
        optimizer_list = []
        print(len(trainer.model_list))
        # parameters = load_parameters("./parameters/" + filename)
        # model.set_parameters(list(parameters))
        # print(model.get_parameters())
        for _ in range(len(trainer.model_list)):
            optimizer_list.append(optimizers.SGD(learning_rate=1e-3))
        histories, model_names = trainer.train(
                                trainer.model_list,
                                x_train, 
                                y_train, 
                                optimizer_list,
                                loss=args.loss,
                                # loss=losses.CrossEntropyLoss(),
                                # metrics=['accuracy', 'Precision', 'Recall'],
                                metrics=['accuracy'],
                                batch_size=args.batch_size, 
                                epochs=args.epochs, 
                                validation_data=(x_val, y_val),
                )
        if isinstance(model, Model):
            # model.save_topology(config.topologies_dir + filename)
            model.save_parameters(config.parameters_dir + filename)

    elif args.predict:
        prediction(filename)
    elif args.compare == "optimizers":
        histories, model_names = trainer.optimizer_test()
    else:
        print(f"Usage: python {sys.argv[0]} -h")

    if histories and model_names:
        plotter.set_model_histories(histories)
        plotter.set_model_names(model_names)
        plotter.plot()
