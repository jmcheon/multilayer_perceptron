import matplotlib.pyplot as plt


def plot_learning_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(2):
        ax = axes[i]
        if (i == 0):
            ax.plot(history.history['loss'], label='training loss')
            ax.plot(history.history['val_loss'], label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Learning Curves for loss')
            ax.legend()
        else:
            ax.plot(history.history['accuracy'], label='training accuracy')
            ax.plot(history.history['val_accuracy'], label='validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Learning Curves for accuracy')
            ax.legend()
    plt.tight_layout(pad=5)
    plt.show()

def compare_models(
        x_train, 
        y_train, 
        optimizer,
        model_list, 
        loss, 
        batch_size, 
        epochs, 
        validation_data=None
    ):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for index, model in enumerate(model_list):
        current_model_index = index + 1
        print(f"\nTraining model #{current_model_index}...")
        model.compile(optimizer=optimizer, loss=loss)
        history = model.fit(
                x_train, y_train, validation_data=validation_data, 
                batch_size=batch_size, 
                epochs=epochs
        )
        # Plot training and validation accuracy and loss
        for i in range(2):
            ax = axes[0][i]
            if (i == 0):
                ax.plot(history.history['loss'], label=f'model #{current_model_index} training loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for training loss:{loss}')
                ax.legend()
            else:
                ax.plot(history.history['accuracy'], label=f'model #{current_model_index} training accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for training accuracy')
                ax.legend()
        for i in range(2):
            ax = axes[1][i]
            if (i == 0):
                ax.plot(history.history['val_loss'], label=f'model #{current_model_index} validation loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for validation loss:{loss}')
                ax.legend()
            else:
                ax.plot(history.history['val_accuracy'], label=f'model #{current_model_index} validation accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for validation accuracy')
                ax.legend()
    plt.tight_layout(pad=5)
    plt.show()

def compare_optimizers(x_train, y_train, model_list, loss, learning_rate, batch_size, epochs, validation_data=None):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for index, model in enumerate(model_list):
        optimizer = model.optimizer
        if model.nesterov == True:
            optimizer = 'nesterov' 
        current_model_index = index + 1
        print(f"\nTraining model #{current_model_index}...")
        model.compile(loss=loss)
        history = model.fit(
                x_train, y_train, validation_data=validation_data, 
                learning_rate=learning_rate, 
                batch_size=batch_size, 
                epochs=epochs
        )
        # Plot training and validation accuracy and loss
        for i in range(2):
            ax = axes[0][i]
            if (i == 0):
                ax.plot(history.history['loss'], label=f'{optimizer} training loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for training loss:{loss}')
                ax.legend()
            else:
                ax.plot(history.history['accuracy'], label=f'{optimizer} training accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for training accuracy')
                ax.legend()
        for i in range(2):
            ax = axes[1][i]
            if (i == 0):
                ax.plot(history.history['val_loss'], label=f'{optimizer} validation loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for validation loss:{loss}')
                ax.legend()
            else:
                ax.plot(history.history['val_accuracy'], label=f'{optimizer} validation accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for validation accuracy')
                ax.legend()
    plt.tight_layout(pad=5)
    plt.show()
