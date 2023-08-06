import numpy as np

def mse(y, y_pred):
    return np.mean(np.power(y - y_pred, 2)) / 2

def mse_derivative(y, y_pred):
    return (y - y_pred)

def binary_cross_entropy_elem(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

def binary_cross_entropy(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return float(loss)

def binary_cross_entropy_derivative(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y / y_pred - (1 - y) / (1 - y_pred))
