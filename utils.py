import numpy as np
import pandas as pd
import json, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Helper functions
def mse(y, y_pred):
    return np.mean(np.power(y - y_pred, 2)) / 2

def mse_derivative(y, y_pred):
    return (y - y_pred) / np.size(y)

def binary_cross_entropy_loss(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return float(loss)

def binary_cross_entropy_derivative(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y / y_pred - (1 - y) / (1 - y_pred))

def convert_to_binary_pred(y_pred, threshold=0.5):
    binary_pred = (y_pred > threshold).astype(int)
    return binary_pred
    binary_pred = np.zeros_like(y_pred)
    #print(y_pred[:5])
    #print('y_pred:', y_pred.shape)
    #print('binary_pred:', binary_pred.shape)
    for i, y_i in enumerate(y_pred):
        max_index = np.argmax(y_i)
        #print(i, y_i, max_index)
    
        #binary_pred = np.zeros((1, 2))
        binary_pred[i][max_index] = 1

    return binary_pred

def sigmoid(x):
    return 1 / (1 + np.exp(np.clip(-x, -709, 709)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def heUniform(shape):
    fan_in, _ = shape
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)

def save(model):
    # Save model configuration as a JSON file
    model_config = model.to_json()
    with open('./saved_model_config.json', 'w') as json_file:
        json.dump(model_config, json_file)
        print("> Saving model configuration to './saved_model_config.json'")
    
    # Save model weights as a .npy file
    model_weights = model.get_weights()
    np.save('./saved_model_weights.npy', model_weights)
    print("> Saving model weights to './saved_model_weights.npy'")

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    data[1] = data[1].map({"M": 1, "B": 0})
    y = data[1].values
    x = data.drop([0, 1], axis=1).values
    # Normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = one_hot_encode_binary_labels(y)
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

def load_data1(filename):
    data = pd.read_csv(filename, header=None)
    data[1] = data[1].map({"M": 1, "B": 0})
    y = data[1].values
    x = data.drop([0, 1], axis=1).values

    # Normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    return x, y 

def one_hot_encode_binary_labels(labels):
    one_hot_encoded_labels = np.zeros((len(labels), 2))
    print(one_hot_encoded_labels.shape)
    for i, label in enumerate(labels):
        one_hot_encoded_labels[i, int(label)] = 1

    return one_hot_encoded_labels


#def plot_(x, y ,net):
def plot_(epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list):
    #epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = train(x, y, net)
    # Plot training and validation accuracy and loss
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(2):
        ax = axes[i]
        if (i == 0):
            ax.plot(epoch_list, loss_list, label='training loss')
            ax.plot(epoch_list, val_loss_list, label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        else:
            ax.plot(epoch_list, accuracy_list, label='training accuracy')
            ax.plot(epoch_list, val_accuracy_list, label='validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()

    plt.show()

def plot_compare(x, y ,net, net1):
    epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = train(x, y, net)
    # Plot training and validation accuracy and loss
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for i in range(2):
        ax = axes[0][i]
        if (i == 0):
            ax.plot(epoch_list, loss_list, label='training loss')
            ax.plot(epoch_list, val_loss_list, label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        else:
            ax.plot(epoch_list, accuracy_list, label='training accuracy')
            ax.plot(epoch_list, val_accuracy_list, label='validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
    epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = train(x, y, net1)
    for i in range(2):
        ax = axes[1][i]
        if (i == 0):
            ax.plot(epoch_list, loss_list, label='training loss')
            ax.plot(epoch_list, val_loss_list, label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        else:
            ax.plot(epoch_list, accuracy_list, label='training accuracy')
            ax.plot(epoch_list, val_accuracy_list, label='validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()

    plt.show()
