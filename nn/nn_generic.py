import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def initialize_parameters(dimensions):
    parameters = {}
    n_layers = len(dimensions)

    for l in range(1, n_layers):
        parameters["W" + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1])
        parameters["b" + str(l)] = np.random.randn(dimensions[l], 1)

    return parameters

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def forward(X, parameters):
    activations = {"A0": X}

    n_layers = len(parameters) // 2

    for l in range(1, n_layers + 1):
        Z = np.dot(parameters["W" + str(l)], activations["A" + str(l - 1)]) + parameters["b" + str(l)]
        activations["A" + str(l)] = sigmoid(Z) 

    return activations

def predict(X, parameters):
    activations = forward(X, parameters)
    n_layers = len(parameters) // 2
    Af = activations["A" + str(n_layers)]
    return (Af >= 0.5).astype(int)

def log_loss(A, y, eps=1e-15):
    A = np.clip(A, eps, 1 - eps)
    return - 1/len(y) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

def backward(y, activations, parameters):
    gradients = {}

    m = y.shape[1]
    n_layers = len(parameters) // 2

    dZ = activations["A" + str(n_layers)] - y

    for l in reversed(range(1, n_layers + 1)):
        gradients["dW" + str(l)] = 1 / m * np.dot(dZ, activations["A" + str(l - 1)].T)
        gradients["db" + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dZ = np.dot(parameters["W" + str(l)].T, dZ) * activations["A" + str(l - 1)] * (1 - activations["A" + str(l - 1)])

    return gradients 

def update(gradients, parameters, lr):

    n_layers = len(parameters) // 2

    for l in range(1, n_layers + 1):
        parameters["W" + str(l)] -= lr * gradients["dW" + str(l)]
        parameters["b" + str(l)] -= lr * gradients["db" + str(l)]

    return parameters

def plot_learning_curves(X, y, parameters, train_loss, train_acc):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(train_loss, label="train loss")
    ax[0].legend()
    ax[1].plot(train_acc, label="train_acc")
    ax[1].legend()

    # Generate input data for decision boundary plot
    x1_range = np.linspace(-1.5, 1.5, 100)
    x2_range = np.linspace(-1.5, 1.5, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Calculate output for each input pair
    z_grid = np.array([[predict(np.array([[x1], [x2]]), parameters)[0, 0] for x1 in x1_range] for x2 in x2_range])

    # Plotting decision boundary
    contour = ax[2].contourf(x1_grid, x2_grid, z_grid, levels=50, cmap='viridis', alpha=0.7)

    # Plot the dataset points
    ax[2].scatter(X[0, :], X[1, :], c=y, cmap="summer", edgecolor='k')
    ax[2].set_title("Decision Boundary")
    ax[2].set_xlabel('x1')
    ax[2].set_ylabel('x2')

    plt.colorbar(contour, ax=ax[2])
    plt.show()

def nn(x_train, y_train, hidden_layers = (32, 32, 32), x_test=None, y_test=None, lr=0.01, epochs=100, plot_graph=False):

    np.random.seed(0)
    dimensions = list(hidden_layers)
    dimensions.insert(0, x_train.shape[0])
    dimensions.append(y_train.shape[0])

    parameters = initialize_parameters(dimensions)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in tqdm(range(epochs)):

        activations = forward(x_train, parameters)
        gradients = backward(y_train, activations, parameters)
        parameters = update(gradients, parameters, lr)

        if epoch % 10 == 0:
            n_layers = len(parameters) // 2
            # trian loss
            train_loss.append(log_loss(activations["A" + str(n_layers)], y_train))
            # accuracy
            y_pred = predict(x_train, parameters)
            train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

            if x_test is not None and y_test is not None:
                # test loss
                test_activations = forward(x_test, parameters)
                test_loss.append(log_loss(test_activations["A" + str(n_layers)], y_test))
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