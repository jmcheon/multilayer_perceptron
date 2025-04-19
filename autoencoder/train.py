import numpy as np


class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, x, y, batch_size=32, epochs=20):
        n_samples = x.shape[0]
        history = {"loss": []}

        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            batch_losses = []

            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                x_batch, y_batch = x[batch_indices], y[batch_indices]
                # Forward pass
                y_pred = self.model(x_batch)

                # Compute loss & backward
                loss_value = self.loss_fn.loss(y_batch, y_pred)
                batch_losses.append(loss_value)

                grad_loss = self.loss_fn.dloss(y_batch, y_pred)

                self.model.backward(grad_loss)
                self.optimizer.step()

            epoch_loss = np.mean(batch_losses)
            history["loss"].append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, loss={epoch_loss:.4f}")

        return history
