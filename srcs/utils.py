import json
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_parameters(filename):
    try:
        loaded = np.load(filename + '.npz', allow_pickle=True)
        parameters = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

    except:
        print(f"Input file errer: {filename} doesn't exist.")
        sys.exit()
    return parameters

def load_topology(filename):
    try:
        with open(filename, 'r') as file:
            config_data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        sys.exit()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit()
    except:
        print(f"Input file errer: can't open {filename}")
        sys.exit()
    return config_data

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
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except:
        print(f"Invalid file error: {filename}")
        sys.exit()
    return df

def load_split_data(filename):
    df = load_df(filename)
    if df is None:
        return None, None

    # df[1] = df[1].map({"M": 1, "B": 0})
    # y = df[1].values
    # x = df.drop([0, 1], axis=1).values
    y = df[0].values
    x = df.drop([0], axis=1).values

    # Normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype('float32')
    #y = one_hot_encode_binary_labels(y)
    y = y.reshape(-1, 1).astype('float32')
    # config.n_classes = len(np.unique(y))

    return x, y
