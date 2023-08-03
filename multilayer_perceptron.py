import numpy as np
import json
from sklearn.metrics import accuracy_score
from utils import load_data, split_data, compare_models, compare_optimizers, plot_learning_curves, save
from DenseLayer import DenseLayer
from NeuralNet import NeuralNet

def prediction():
    weights_path = 'saved_model_weights.npy'
    config_path = 'saved_model_config.json'
    data_path = 'data_test.csv'

    x, y = load_data(data_path)
    data_test = np.hstack((x, y))

    weights = np.load(weights_path, allow_pickle=True)
    with open(config_path, 'r') as file:
        json_config = json.load(file)
    config = json.loads(json_config)

    #print(weights, weights.shape, weights[0].shape, type(weights))
    #print(list(weights), type(list(weights)))
    #print(config, type(config))
    #print(config['network_topology'])
    #for key, value in config.items():
    #    print(f"{key}: {value}")

    #print(config['network_topology'])
    model = NeuralNet()
    model.create_network(config['network_topology'])
    model.set_weights(list(weights))
    model.predict(data_test)

def backprop_test():
    np.random.seed(0)

    x_train = np.array([[0.1, 0.2]])
    y_train = np.array([[0.4, 0.6]])
    x_val = np.array([[0.1, 0.2]])
    y_val = np.array([[0.4, 0.6]])

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model = NeuralNet()

    network = model.create_network([
        DenseLayer(2, 2, activation=sigmoid),
        DenseLayer(2, 2, activation=sigmoid, weights_initializer='heUniform')
        ])

    initial_weights = [
        np.array([[0.3, 0.25], [0.4, 0.35]], dtype=np.float32), 
        np.zeros(2, dtype=np.float32),  # Biases
        np.array([[0.45, 0.4], [0.7, 0.6]], dtype=np.float32),  # Weights 
        np.zeros(2, dtype=np.float32),  # Biases
    ]
    model.set_weights(initial_weights)
    #print(model.network[0].weights)
    #print(model.network[0].bias)
    #print(model.network[1].weights)
    #print(model.network[1].bias)

    #model.fit(network, data_train, data_valid, loss=loss_, learning_rate=0.5, batch_size=8, epochs=100)
    model.predict(data_train)
    print("Updated weights and biases.")
    print(model.network[0].weights)
    print(model.network[0].bias)
    print(model.network[1].weights)
    print(model.network[1].bias)

def train_plot_save():
    np.random.seed(0)
    # Load and split the data
    x, y = load_data('data.csv')
    x_train, x_val, y_train, y_val = split_data(x, y)
    #print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model = NeuralNet()
    network = model.create_network([
        DenseLayer(30, 20, activation='sigmoid'),
        DenseLayer(20, 10, activation='sigmoid', weights_initializer='random'),
        DenseLayer(10, 2, activation='sigmoid', weights_initializer='random'),
        DenseLayer(2, 2, activation='softmax', weights_initializer='random')
        ])

    epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = model.fit(None, data_train, data_valid, loss='binary_cross_entropy_loss', learning_rate=1e-2, batch_size=2, epochs=70)
    plot_learning_curves(epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list)
    save(model)

def multiple_models_test():
    np.random.seed(0)
    # Load and split the data
    x, y = load_data('data.csv')
    x_train, x_val, y_train, y_val = split_data(x, y)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model1 = NeuralNet()
    model1.create_network([
        DenseLayer(30, 20, activation='sigmoid'),
        DenseLayer(20, 10, activation='sigmoid', weights_initializer='random'),
        DenseLayer(10, 2, activation='sigmoid', weights_initializer='random'),
        DenseLayer(2, 2, activation='softmax', weights_initializer='random')
        ])

    model2 = NeuralNet()
    model2.create_network([
        DenseLayer(30, 15, activation='sigmoid'),
        DenseLayer(15, 8, activation='sigmoid', weights_initializer='random'),
        DenseLayer(8, 2, activation='softmax', weights_initializer='random')
        ])

    model3 = NeuralNet()
    model3.create_network([
        DenseLayer(30, 15, activation='sigmoid'),
        DenseLayer(15, 2, activation='softmax', weights_initializer='random')
        ])

    model_list = [model1, model2, model3]
    compare_models(data_train, data_valid, model_list, loss='binary_cross_entropy_loss', learning_rate=1e-2, batch_size=2, epochs=50)

def optimizer_test():
    np.random.seed(0)

    x, y = load_data('data.csv')
    x_train, x_val, y_train, y_val = split_data(x, y)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model1 = NeuralNet(nesterov=False)
    model1.create_network([
        DenseLayer(30, 15, activation='sigmoid'),
        DenseLayer(15, 2, activation='softmax', weights_initializer='zero')
        ])

    model2 = NeuralNet(nesterov=True)
    model2.create_network([
        DenseLayer(30, 15, activation='sigmoid'),
        DenseLayer(15, 2, activation='softmax', weights_initializer='zero')
        ])

    model3 = NeuralNet(optimizer='rmsprop')
    model3.create_network([
        DenseLayer(30, 15, activation='sigmoid'),
        DenseLayer(15, 2, activation='softmax', weights_initializer='zero')
        ])

    model_list = [model1, model3]
    compare_optimizers(data_train, data_valid, model_list, loss='binary_cross_entropy_loss', learning_rate=1e-3, batch_size='batch', epochs=30)

def same_model_test():
    np.random.seed(0)

    x, y = load_data('data.csv')
    x_train, x_val, y_train, y_val = split_data(x, y)

    data_train = np.hstack((x_train, y_train))
    data_valid = np.hstack((x_val, y_val))

    model1 = NeuralNet()
    model1.create_network([
        DenseLayer(30, 15, activation='sigmoid'),
        DenseLayer(15, 2, activation='softmax', weights_initializer='zero')
        ])

    model2 = NeuralNet()
    model2.create_network([
        DenseLayer(30, 15, activation='sigmoid'),
        DenseLayer(15, 2, activation='softmax', weights_initializer='zero')
        ])

    model3 = NeuralNet()
    model3.create_network([
        DenseLayer(30, 15, activation='sigmoid'),
        DenseLayer(15, 2, activation='softmax', weights_initializer='zero')
        ])

    model_list = [model1, model2, model3]
    compare_models(data_train, data_valid, model_list, loss='binary_cross_entropy_loss', learning_rate=1e-3, batch_size='batch', epochs=30)

if __name__ == "__main__":
    #backprop_test()
    #prediction()
    train_plot_save()
    #multiple_models_test()
    #optimizer_test()
    #same_model_test()
