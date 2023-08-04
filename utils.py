import numpy as np
import pandas as pd
import sys

def one_hot_encode_binary_labels(labels):
    one_hot_encoded_labels = np.zeros((len(labels), 2))
    for i, label in enumerate(labels):
        one_hot_encoded_labels[i, int(label)] = 1

    return one_hot_encoded_labels

def convert_to_binary_pred(y_pred, threshold=0.5):
    binary_pred = (y_pred > threshold).astype(int)
    return binary_pred

def heUniform(shape):
    fan_in, _ = shape
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)

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
    df = load_df(filename)

    train_data_np, val_data_np = data_spliter(df.values, train_size, random_state)
    
    train_data = pd.DataFrame(train_data_np)
    val_data = pd.DataFrame(val_data_np)

    # Save the training and validation sets as CSV files
    train_data.to_csv(train_file, index=False, header=False)
    val_data.to_csv(val_file, index=False, header=False)

    print(f"Saved train data to {train_file}")
    print(f"Saved validation data to {val_file}")

def data_spliter(data, proportion, random_state=None):
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

def load_df(filename):
    try:
        df = pd.read_csv(filename, header=None)
    except:
        print(f"Invalid file error.")
        sys.exit()
    return df

def load_split_data(filename):
    df = load_df(filename)

    df[1] = df[1].map({"M": 1, "B": 0})
    y = df[1].values
    x = df.drop([0, 1], axis=1).values

    # Normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = one_hot_encode_binary_labels(y)

    return x, y
