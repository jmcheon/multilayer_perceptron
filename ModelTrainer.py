import multilayer_perceptron.config as config
import multilayer_perceptron.srcs.optimizers as optimizers

from multilayer_perceptron.NeuralNet import NeuralNet
from multilayer_perceptron.srcs.layers import Dense
from multilayer_perceptron.srcs.utils import load_split_data, load_parameters
                        


class ModelTrainer():
    def __init__(self):
        self.model_list = []

    def create(self, model_topologies):
        if isinstance(model_topologies, list) and \
            all(isinstance(model, list) and \
                all(isinstance(layer_data, dict) for layer_data in model) for model in model_topologies):
            print('Multiple model topologies found. creating models...')
            self.create_models(model_topologies)
            return self.model_list
        elif isinstance(model_topologies, list) and all(isinstance(layer_data, dict) for layer_data in model_topologies):
            print('One model topology found. creating one model...')
            model = self.create_model(model_topologies)
            return model
        else:
            raise TypeError("Invalid form of input to create a neural network.")

    def create_default_model(self, model: NeuralNet):
        network = model.create_network([
            Dense(self.shape[0], 20, activation='relu'),
            Dense(20, 10, activation='relu'),
            Dense(10, 5, activation='relu'),
            Dense(5, self.shape[1], activation='sigmoid')
            ])
        
        '''
        model.compile(
                optimizer=optimizers.SGD(learning_rate=1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall'],
        )
        '''
        parameters = load_parameters(config.parameters_dir + config.tensorflow_weights_npy)
        model.set_parameters(list(parameters))
        return model
    
    
    def create_model(self, topology=None):
        model = NeuralNet()
        if topology:
            network = model.create_network(topology)
        else:
            model = self.create_default_model(model)
        self.model_list.append(model)
        return model

    def create_models(self, model_topologies):
        for topology in model_topologies:
            model = self.create_model(topology)

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
        """
        """
        if len(model_list) == 0:
            raise RuntimeError("You must create your model before training. Use `create(model_topologies)")
        elif len(model_list) != len(optimizer_list):
            raise RuntimeError("The number of Optimizers doesn't match with model's")
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
        """
        Train one model
        """

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(
                x_train, y_train, validation_data=validation_data, 
                batch_size=batch_size, epochs=epochs)

        model_name = f"{model.name} + {model.optimizer.name}"
        return model.history, model_name

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
        """
        Train multiple models
        """
        histories = []
        model_names = []
        for (model, optimizer) in zip(model_list, optimizer_list):
            print(f"\nTraining {model.name}...")
            _, model_name = self.train_model(
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
            histories.append(model.history)
            model_names.append(model_name)
        return histories, model_names

    def compare_models(self, model_topologies):
        x_train, y_train = load_split_data(self.train_path)
        x_val, y_val = load_split_data(self.valid_path)
    
        self.create(model_topologies)
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
