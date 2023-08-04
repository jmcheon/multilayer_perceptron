import numpy as np
import pandas as pd
import json, sys

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

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        return self

    def transform(self, X):
        if self.mean is None or self.scale is None:
            raise ValueError("fit method must be called before transform")
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def split_dataset_save(filename, train_file, val_file, train_size=0.8, random_state=None):
    df = pd.read_csv(filename, header=None)
    # Split the dataset into training and validation sets
    train_data_np, val_data_np = data_spliter2(df.values, train_size, random_state)
    
    # Convert NumPy arrays back to Pandas DataFrames
    train_data = pd.DataFrame(train_data_np)
    val_data = pd.DataFrame(val_data_np)
    # Save the training and validation sets as CSV files
    train_data.to_csv(train_file, index=False, header=False)
    val_data.to_csv(val_file, index=False, header=False)

    print(f"Saved train data to {train_file}")
    print(f"Saved validation data to {val_file}")

def data_spliter2(data, proportion, random_state=None):
    for v in [data]:
        if not isinstance(v, np.ndarray):
    	    print(f"Invalid input: argument {v} of ndarray type required")	
    	    return None
    
    if not isinstance(proportion, float):
        print(f"Invalid input: argument proportion of float type required")	
        return None

    if random_state is not None:
        np.random.default_rng(random_state).shuffle(data)
    else:
        np.random.shuffle(data)

    p = int(data.shape[0] * proportion)
    data_train, data_valid = data[:p], data[p:]
    return data_train, data_valid 

def data_spliter(x, y, proportion, random_state=None):
    for v in [x, y]:
        if not isinstance(v, np.ndarray):
    	    print(f"Invalid input: argument {v} of ndarray type required")	
    	    return None
    
    if not isinstance(proportion, float):
        print(f"Invalid input: argument proportion of float type required")	
        return None

    if not x.ndim == 2:
        print(f"Invalid input: wrong shape of x", x.shape)
        return None

    if random_state is not None:
        np.random.default_rng(random_state).shuffle(data)
    else:
        np.random.shuffle(data)

    data = np.hstack((x, y))
    p = int(x.shape[0] * proportion)
    x_train, x_test= data[:p, :-1], data[p:, :-1]
    y_train, y_test = data[:p, -1:], data[p:, -1:] 
    return x_train, x_test, y_train, y_test

def split_data(x, y):
    return data_spliter(x, y, 0.8, 42)

def load_data(filename):
    try:
        df = pd.read_csv(filename, header=None)
    except:
        print(f"Invalid file error: {filename} doesn't exist.")
        sys.exit()
    df[1] = df[1].map({"M": 1, "B": 0})
    y = df[1].values
    x = df.drop([0, 1], axis=1).values

    # Normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = one_hot_encode_binary_labels(y)

    return x, y

def one_hot_encode_binary_labels(labels):
    one_hot_encoded_labels = np.zeros((len(labels), 2))
    print(one_hot_encoded_labels.shape)
    for i, label in enumerate(labels):
        one_hot_encoded_labels[i, int(label)] = 1

    return one_hot_encoded_labels

