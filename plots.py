import matplotlib.pyplot as plt

def plot_learning_curves(epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(2):
        ax = axes[i]
        if (i == 0):
            ax.plot(epoch_list, loss_list, label='training loss')
            ax.plot(epoch_list, val_loss_list, label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Learning Curves for loss')
            ax.legend()
        else:
            ax.plot(epoch_list, accuracy_list, label='training accuracy')
            ax.plot(epoch_list, val_accuracy_list, label='validation accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Learning Curves for accuracy')
            ax.legend()
    plt.tight_layout(pad=5)
    plt.show()

def compare_models(data_train, data_valid, model_list, loss, learning_rate, batch_size, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for index, model in enumerate(model_list):
        current_model_index = index + 1
        print(f"\nTraining model #{current_model_index}...")
        epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = model.fit(data_train, data_valid, loss, learning_rate, batch_size, epochs)
        # Plot training and validation accuracy and loss
        for i in range(2):
            ax = axes[0][i]
            if (i == 0):
                ax.plot(epoch_list, loss_list, label=f'model #{current_model_index} training loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for training loss:{loss}')
                ax.legend()
            else:
                ax.plot(epoch_list, accuracy_list, label=f'model #{current_model_index} training accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for training accuracy')
                ax.legend()
        for i in range(2):
            ax = axes[1][i]
            if (i == 0):
                ax.plot(epoch_list, val_loss_list, label=f'model #{current_model_index} validation loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for validation loss:{loss}')
                ax.legend()
            else:
                ax.plot(epoch_list, val_accuracy_list, label=f'model #{current_model_index} validation accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for validation accuracy')
                ax.legend()
    plt.tight_layout(pad=5)
    plt.show()

def compare_optimizers(data_train, data_valid, model_list, loss, learning_rate, batch_size, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for index, model in enumerate(model_list):
        optimizer = model.optimizer
        if model.nesterov == True:
            optimizer = 'nesterov' 
        current_model_index = index + 1
        print(f"\nTraining model #{current_model_index}...")
        epoch_list, accuracy_list, loss_list, val_accuracy_list, val_loss_list = model.fit(data_train, data_valid, loss, learning_rate, batch_size, epochs)
        # Plot training and validation accuracy and loss
        for i in range(2):
            ax = axes[0][i]
            if (i == 0):
                ax.plot(epoch_list, loss_list, label=f'{optimizer} training loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for training loss:{loss}')
                ax.legend()
            else:
                ax.plot(epoch_list, accuracy_list, label=f'{optimizer} training accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for training accuracy')
                ax.legend()
        for i in range(2):
            ax = axes[1][i]
            if (i == 0):
                ax.plot(epoch_list, val_loss_list, label=f'{optimizer} validation loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for validation loss:{loss}')
                ax.legend()
            else:
                ax.plot(epoch_list, val_accuracy_list, label=f'{optimizer} validation accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'[Batch size:{batch_size}] Learning Curves for validation accuracy')
                ax.legend()
    plt.tight_layout(pad=5)
    plt.show()
