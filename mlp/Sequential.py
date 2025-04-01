from typing import List

from mlp.activations import Activation
from mlp.layers import Dense, Linear
from mlp.module import Module


class Sequential(Module):
    def __init__(self, layers, name="Sequential"):
        super().__init__()
        self.layers = layers
        self.name = name

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        return grad_output

    def parameters(self):
        parameters = []

        for layer in self.layers:
            if hasattr(layer, "weights"):
                parameters += layer.parameters()

        return parameters

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def get_topology(self) -> List:
        """
        Retrieves the model topology.

        Returns:
            list: A list describing the model topology.
        """
        topology = []
        model_data = {
            "type": "Model",
            # "shape": self.shape,
            "name": self.name,
            "n_layers": len(self.layers),
        }
        topology.append(model_data)
        layers = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer_data = {
                    "type": "Dense",
                    "shape": layer.shape,
                    "activation": f"{type(layer.activation).__name__}",
                    "initializer": f"{layer.initializer}",
                }
                layers.append(layer_data)
            if isinstance(layer, Linear):
                layer_data = {
                    "type": "Linear",
                    "shape": layer.shape,
                }
                layers.append(layer_data)
            if isinstance(layer, Activation):
                layer_data = {"type": "Activation", "name": layer.name}
                layers.append(layer_data)
        topology.extend(layers)
        return topology

    def __repr__(self) -> str:
        topology = self.get_topology()
        res = f"{self.name}(\n"

        for layer in topology:
            if layer["type"] == "Dense":
                res += f"\t{layer['type']}({layer['shape']}, {layer['activation']})\n"
            if layer["type"] == "Linear":
                res += f"\t{layer['type']}({layer['shape']})\n"
            if layer["type"] == "Activation":
                res += f"\t{layer['name']}()\n"
        res += ")"

        return res
