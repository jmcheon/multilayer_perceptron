import sys

import numpy as np

import config
import srcs.optimizers as optimizers
from Model import Model
from srcs.layers import Dense
from srcs.utils import (load_config, load_split_data, load_weights,
                        split_dataset_save)


class ModelTrainer():
    def __init__(self, input_shape, output_shape, train_path, valid_path):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.train_path = train_path
        self.valid_path = valid_path
        self.model_list = []

    def create(self, model_params):
        if isinstance(model_params, list) and \
            all(isinstance(model, list) and \
                all(isinstance(layer_data, dict) for layer_data in model) for model in model_params):
            print('Multiple model topologies found. creating models...')
            self.create_models(model_params)
            return self.model_list
        elif isinstance(model_params, list) and all(isinstance(layer_data, dict) for layer_data in model_params):
            print('One model topology found. creating one model...')
            model = self.create_model(model_params)
            return model
        else:
            raise TypeError("Invalid form of input to create a neural network.")

    def create_default_model(self):
        model = Model()
        network = model.create_network([
            Dense(self.input_shape, 20, activation='relu'),
            Dense(20, 10, activation='relu'),
            Dense(10, 5, activation='relu'),
            Dense(5, self.output_shape, activation='sigmoid')
            ])
        
        '''
        model.compile(
                optimizer=optimizers.SGD(learning_rate=1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall'],
        )
        '''
        weights = load_weights(config.weights_dir + config.tensorflow_weights_npy)
        '''
        for i in range(len(weights)):
            print('weights shape:', weights[i].shape)
        '''
        model.set_weights(list(weights))
        return model
    
    
    def create_model(self, params=None):
        model = Model()
        if params:
            network = model.create_network(params)
        else:
            model = self.create_default_model()
        self.model_list.append(model)
        return model

    def create_models(self, model_params):
        for params in model_params:
            model = self.create_model(params)

    def train(self,
            model_list,
            x_train, 
            y_train, 
            optimizer_list,
            loss, 
            metrics,
            batch_size, 
            epochs=1, 
            validation_data=None,
        ):
        if len(model_list) == 0:
            raise RuntimeError("You must create your model before training. Use `create(model_params)")
        elif len(model_list) != len(optimizer_list):
            raise RuntimeError("The number of Optimizers doesn't match with model's")
        elif len(model_list) == 1:
            histories, model_names = self.train_model(
                    model_list[0],
                    x_train, 
                    y_train, 
                    optimizer_list[0],
                    loss, 
                    metrics,
                    batch_size, 
                    epochs, 
                    validation_data,
                    )
        else:
            histories, model_names = self.train_models(
                    model_list,
                    x_train, 
                    y_train, 
                    optimizer_list,
                    loss, 
                    metrics,
                    batch_size, 
                    epochs, 
                    validation_data,
                    )
        return histories, model_names

    def train_model(
            self,
            model,
            x_train, 
            y_train, 
            optimizer,
            loss, 
            metrics,
            batch_size, 
            epochs=1, 
            validation_data=None,
        ):

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
        histories = []
        model_names = []
        history = model.fit(
                x_train, y_train, validation_data=validation_data, 
                batch_size=batch_size, 
                epochs=epochs
        )
        histories.append(history)
        model_names.append(f"{model.name} + {model.optimizer.name}")
        return histories, model_names

    def train_models(
            self,
            model_list,
            x_train, 
            y_train, 
            optimizer_list,
            loss, 
            metrics,
            batch_size, 
            epochs=1, 
            validation_data=None,
        ):
        histories = []
        model_names = []
        for (model, optimizer) in zip(model_list, optimizer_list):
            print(f"\nTraining {model.name}...")
            history, model_name = self.train_model(
                        model,
                        x_train, 
                        y_train, 
                        optimizer,
                        loss, 
                        metrics,
                        batch_size, 
                        epochs, 
                        validation_data,
            )
            histories.append(history[0])
            model_names.append(model_name[0])
        return histories, model_names

    def compare_models(self, model_params):
        x_train, y_train = load_split_data(self.train_path)
        x_val, y_val = load_split_data(self.valid_path)
    
        self.create(model_params)
        optimizer = optimizers.SGD(learning_rate=1e-3)
    
        optimizer_list = [
                optimizer, 
                optimizer, 
        ]
        print(len(self.model_list), len(optimizer_list))
        histories, model_names = self.train_models(
            self.model_list,
            x_train, 
            y_train, 
            optimizer_list,
            loss='binary_crossentropy', 
            metrics=['accuracy', 'Precision', 'Recall'],
            batch_size=50, 
            epochs=50,
            validation_data=(x_val, y_val),
        )
        return histories, model_names
    
    
    def optimizer_test(self):
        print("Compare optimizers...")
    
        x_train, y_train = load_split_data(self.train_path)
        x_val, y_val = load_split_data(self.valid_path)
    
        model1 = self.create_model()
        model2 = self.create_model()
        model3 = self.create_model()
    
        model_list = [
                model1,
                model2,
                model3,
        ]
        optimizer_list = [
                optimizers.SGD(learning_rate=1e-3), 
                optimizers.RMSprop(learning_rate=1e-3),
                optimizers.Adam(learning_rate=1e-3),
        ]
        histories, model_names = self.train_models(
            model_list,
            x_train, 
            y_train, 
            optimizer_list,
            loss='binary_crossentropy', 
            metrics=['accuracy', 'Precision', 'Recall'],
            batch_size=100, 
            epochs=10,
            validation_data=(x_val, y_val),
        )
    
        return histories, model_names
