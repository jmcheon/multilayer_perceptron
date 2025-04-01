from typing import List

from mlp.layers import Dense
from mlp.module import Module


class Sequential(Module):
    def __init__(self, layers, name="Sequential"):
        super().__init__()
        self.layers = layers
        self.name = name
        self.shape = (layers[0].shape[0], layers[-1].shape[1])

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
            "shape": self.shape,
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
        topology.extend(layers)
        return topology

    def __repr__(self) -> str:
        topology = self.get_topology()
        res = f"{self.name}(\n"

        for topo in topology:
            if topo["type"] != "Model":
                res += f"\t{topo['type']}({topo['shape']}, activation={topo['activation']})\n"
        res += ")"

        return res
