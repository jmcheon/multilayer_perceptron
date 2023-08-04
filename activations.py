import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(np.clip(-x, -709, 709)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)
