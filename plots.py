import matplotlib.pyplot as plt


def plot_learning_curves(history):
    if 'accuracy' in history.history:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for i in range(2):
            ax = axes[i]
            if (i == 0):
                ax.plot(history.history['loss'], label='training loss')
                if 'val_loss' in history.history:
                    ax.plot(history.history['val_loss'], label='validation loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'Learning Curves for loss')
                ax.legend()
            else:
                ax.plot(history.history['accuracy'], label='training accuracy')
                if 'val_accuracy' in history.history:
                    ax.plot(history.history['val_accuracy'], label='validation accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'Learning Curves for accuracy')
                ax.legend()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(history.history['loss'], label='training loss')
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='validation loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Learning Curves for loss')
        ax.legend()
    plt.tight_layout(pad=5)
    plt.show()

def create_subplots_2x2(history, axes, loss, batch_size, name):
    # Plot training and validation accuracy and loss
    for i in range(2):
        ax = axes[0][i]
        if (i == 0):
            ax.plot(history.history['loss'], label=f'{name} training loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'[Batch size:{batch_size}] Training loss: {loss}')
            ax.legend()
        else:
            ax.plot(history.history['accuracy'], label=f'{name} training accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'[Batch size:{batch_size}] Training accuracy')
            ax.legend()
    for i in range(2):
        ax = axes[1][i]
        if (i == 0):
            ax.plot(history.history['val_loss'], label=f'{name} validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'[Batch size:{batch_size}] Validation loss: {loss}')
            ax.legend()
        else:
            ax.plot(history.history['val_accuracy'], label=f'{name} validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'[Batch size:{batch_size}] Validation accuracy')
            ax.legend()

def create_subplots_2x1(history, axes, loss, batch_size, name):
    # Plot training and validation loss
    for i in range(2):
        ax = axes[i]
        if (i == 0):
            ax.plot(history.history['loss'], label=f'{name} training loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'[Batch size:{batch_size}] Training loss: {loss}')
            ax.legend()
        elif 'val_loss' in history.history and i == 1:
            ax.plot(history.history['val_loss'], label=f'{name} validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'[Batch size:{batch_size}] Validation loss: {loss}')
            ax.legend()

def create_subplots_1x2(history, axes, loss, batch_size, name):
    # Plot training loss and accuracy
    for i in range(2):
        ax = axes[i]
        if (i == 0):
            ax.plot(history.history['loss'], label=f'{name} training loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'[Batch size:{batch_size}] Training loss: {loss}')
            ax.legend()
        else:
            ax.plot(history.history['accuracy'], label=f'{name} training accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'[Batch size:{batch_size}] Training accuracy')
            ax.legend()

def create_subplots_1x1(history, axes, loss, batch_size, name):
    # Plot training loss
        ax = axes
        ax.plot(history.history['loss'], label=f'{name} training loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'[Batch size:{batch_size}] Training loss: {loss}')
        ax.legend()

def plot_models(
        x_train, 
        y_train, 
        model_list, 
        loss, 
        metrics,
        batch_size, 
        epochs, 
        validation_data=None
    ):
    if validation_data and 'accuracy' in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        create_subplots = create_subplots_2x2
    elif validation_data:
        fig, axes = plt.subplots(2, 1, figsize=(10, 15))
        create_subplots = create_subplots_2x1
    elif 'accuracy' in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        create_subplots = create_subplots_1x2
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        create_subplots = create_subplots_1x1
    for (model, optimizer) in (model_list):
        print(f"\nTraining {model.name}...")
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        history = model.fit(
                x_train, y_train, validation_data=validation_data, 
                batch_size=batch_size, 
                epochs=epochs
        )
        create_subplots(history, axes, loss, batch_size, f"{model.name} - {optimizer.name}")
    plt.tight_layout(pad=5)
    plt.show()
