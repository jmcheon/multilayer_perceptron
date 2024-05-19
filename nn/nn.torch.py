import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Define the loss function
criterion = nn.BCELoss()

# Initialize the neural network
net = SimpleNN()

# Define the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001)

# Sample training data (using random data for demonstration)
x = torch.randn(1, 784)  # Single training example with 784 features
y = torch.zeros(1, 10)   # One-hot encoded target vector for multi-class classification
y[0, 3] = 1  # Assuming the true class is the 4th class

# Training loop
for epoch in range(1000):  # Number of epochs
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = net(x)

    # Compute the loss
    loss = criterion(outputs, y)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    if epoch % 100 == 0:  # Print loss every 100 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")
