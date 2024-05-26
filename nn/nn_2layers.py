import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def initialize_parameters(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2,
    }
    return parameters

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def forward(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1) 
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2) 

    activations = {
        "A1": A1,
        "A2": A2,
    }
    return activations

def predict(X, parameters):
    activations = forward(X, parameters)
    A2 = activations["A2"]
    return (A2 >= 0.5).astype(int)

def log_loss(A, y, eps=1e-15):
    A = np.clip(A, eps, 1 - eps)
    return - 1/len(y) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

def backward(X, y, activations, parameters):
    A1 = activations["A1"]
    A2 = activations["A2"]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    m = y.shape[1]
    dZ2 = A2 - y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1) 
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        "dW1" : dW1,
        "db1" : db1,
        "dW2" : dW2,
        "db2" : db2,
    }
    return gradients 

def update(gradients, parameters, lr):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2,
    }
    return parameters

def nn(x_train, y_train, n1, x_test=None, y_test=None, lr=0.01, epochs=100, plot_graph=False):

    n0 = x_train.shape[0]
    n2 = y_train.shape[0]

    parameters = initialize_parameters(n0, n1, n2)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in tqdm(range(epochs)):

        activations = forward(x_train, parameters)
        gradients = backward(x_train, y_train, activations, parameters)
        parameters = update(gradients, parameters, lr)

        if epoch % 10 == 0:
            # trian loss
            train_loss.append(log_loss(activations["A2"], y_train))
            # accuracy
            y_pred = predict(x_train, parameters)
            train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

            if x_test is not None and y_test is not None:
                # test loss
                test_activations = forward(x_test, parameters)
                test_loss.append(log_loss(test_activations["A2"], y_test))
                # accuracy
                y_pred = predict(x_test, parameters)
                test_acc.append(accuracy_score(y_test.flatten(), y_pred.flatten()))
    if plot_graph:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label="train loss")
        plt.plot(test_loss , label="test loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label="train_acc")
        plt.plot(test_acc, label="test_acc")
        plt.legend()
        plt.show()

    return parameters, train_loss, train_acc

def plot_learning_curves(X, y, parameters, train_loss, train_acc):
    # Generate input data for decision boundary plot
    x1_range = np.linspace(-1.5, 1.5, 100)
    x2_range = np.linspace(-1.5, 1.5, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Calculate output for each input pair
    z_grid = np.array([[predict(np.array([[x1], [x2]]), parameters)[0, 0] for x1 in x1_range] for x2 in x2_range])

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label="train loss")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(train_acc, label="train_acc")
    plt.legend()
    
    # Plotting decision boundary
    plt.subplot(1, 3, 3)
    plt.contourf(x1_grid, x2_grid, z_grid, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar()

    # Plot the dataset points
    plt.scatter(X[0, :], X[1, :], c=y, cmap="summer", edgecolor='k')
    plt.title("Decision Boundary")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()