import numpy as np
import json
from sklearn.metrics import accuracy_score
from utils import load_data
from DenseLayer import DenseLayer
from NeuralNet import NeuralNet

def prediction(data_test):
    weights_path = 'saved_model_weights.npy'
    config_path = 'saved_model_config.json'
    data_path = 'data.csv'

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

def main_test():
    np.random.seed(0)
    # Load and split the data
    #x_train, x_test, y_train, y_test = load_data('data.csv')
    x_train, x_val, y_train, y_val= load_data('data.csv')
    #print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

    # Combine x_train and y_train as data_train
    data_train = np.hstack((x_train, y_train))
    # Combine x_val and y_val as data_valid
    data_valid = np.hstack((x_val, y_val))

    model = NeuralNet()
    network = model.create_network([
        DenseLayer(30, 20, activation='sigmoid'),
        DenseLayer(20, 10, activation='sigmoid', weights_initializer='random'),
        DenseLayer(10, 2, activation='sigmoid', weights_initializer='random'),
        DenseLayer(2, 2, activation='softmax', weights_initializer='random')
        ])

    #model.fit(network, data_train, data_valid, loss='binary_cross_entropy_loss', learning_rate=1e-2, batch_size=8, epochs=100)
    #save(model)
    prediction(data_valid)
    #model.predict(data_valid)

if __name__ == "__main__":
    #backprop_test()
    main_test()
