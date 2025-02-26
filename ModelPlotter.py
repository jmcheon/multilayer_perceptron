import matplotlib.pyplot as plt


class ModelPlotter:
    def __init__(self, model_histories=[], model_names=[]):
        self.model_histories = model_histories
        self.model_names = model_names

    def set_model_histories(self, histories):
        self.model_histories = histories

    def set_model_names(self, model_names):
        self.model_names = model_names

    def plot(
        self,
        accuracy=False,
        validation_data=False,
    ):
        if len(self.model_histories) == 1:
            self.plot_learning_curves(self.model_histories[0])
        else:
            if any("accuracy" in history for history in self.model_histories):
                accuracy = True
            if any("val" in str(history.keys()) for history in self.model_histories):
                validation_data = True

            if validation_data and accuracy:
                fig, axes = plt.subplots(2, 2, figsize=(20, 15))
                create_subplots = self.create_subplots_2x2
            elif validation_data:
                fig, axes = plt.subplots(2, 1, figsize=(10, 15))
                create_subplots = self.create_subplots_2x1
            elif accuracy:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                create_subplots = self.create_subplots_1x2
            else:
                fig, axes = plt.subplots(1, 1, figsize=(10, 5))
                create_subplots = self.create_subplots_1x1

            for history, name in zip(self.model_histories, self.model_names):
                create_subplots(history, axes, name)
            plt.tight_layout(pad=5)
            plt.show()

    def plot_learning_curves(self, history):
        if "accuracy" in history:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            for i in range(2):
                ax = axes[i]
                if i == 0:
                    ax.plot(history["loss"], label="training loss")
                    if "val_loss" in history:
                        ax.plot(history["val_loss"], label="validation loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("Learning Curves for loss")
                    ax.legend()
                else:
                    ax.plot(history["accuracy"], label="training accuracy")
                    if "val_accuracy" in history:
                        ax.plot(history["val_accuracy"], label="validation accuracy")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Accuracy")
                    ax.set_title("Learning Curves for accuracy")
                    ax.legend()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(history["loss"], label="training loss")
            if "val_loss" in history:
                ax.plot(history["val_loss"], label="validation loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Learning Curves for loss")
            ax.legend()
        plt.tight_layout(pad=5)
        plt.show()

    def set_subplots(self, metrics, validation_data):
        if validation_data and "accuracy" in metrics:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            create_subplots = self.create_subplots_2x2
        elif validation_data:
            fig, axes = plt.subplots(2, 1, figsize=(10, 15))
            create_subplots = self.create_subplots_2x1
        elif "accuracy" in metrics:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            create_subplots = self.create_subplots_1x2
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 5))
            create_subplots = self.create_subplots_1x1
        return fig, axes, create_subplots

    def create_subplots_2x2(self, history, axes, name):
        # Plot training and validation accuracy and loss
        for i in range(2):
            ax = axes[0][i]
            if i == 0:
                ax.plot(history["loss"], label=f"{name} training loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training loss")
                ax.legend()
            else:
                ax.plot(history["accuracy"], label=f"{name} training accuracy")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.set_title("Training accuracy")
                ax.legend()
        for i in range(2):
            ax = axes[1][i]
            if i == 0:
                ax.plot(history["val_loss"], label=f"{name} validation loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Validation loss")
                ax.legend()
            else:
                ax.plot(history["val_accuracy"], label=f"{name} validation accuracy")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.set_title("Validation accuracy")
                ax.legend()

    def create_subplots_2x1(self, history, axes, name):
        # Plot training and validation loss
        for i in range(2):
            ax = axes[i]
            if i == 0:
                ax.plot(history["loss"], label=f"{name} training loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training loss")
                ax.legend()
            elif "val_loss" in history and i == 1:
                ax.plot(history["val_loss"], label=f"{name} validation loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Validation loss")
                ax.legend()

    def create_subplots_1x2(self, history, axes, name):
        # Plot training loss and accuracy
        for i in range(2):
            ax = axes[i]
            if i == 0:
                ax.plot(history["loss"], label=f"{name} training loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training loss")
                ax.legend()
            else:
                ax.plot(history["accuracy"], label=f"{name} training accuracy")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.set_title("Training accuracy")
                ax.legend()

    def create_subplots_1x1(self, history, axes, name):
        # Plot training loss
        ax = axes
        ax.plot(history["loss"], label=f"{name} training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss")
        ax.legend()
