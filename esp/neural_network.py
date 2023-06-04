from typing import List
import numpy as np
from .layer import Layer
from .neuron import Neuron
from .activations import Sigmoid


class NeuralNetwork(object):
    def __init__(self,
                 hidden_neurons: List[Neuron]):
        input_count = hidden_neurons[0].input_count
        output_count = hidden_neurons[0].output_count
        self.input_weights = np.zeros((len(hidden_neurons), input_count))
        output_weights = np.zeros((len(hidden_neurons), output_count))
        for i in range(len(hidden_neurons)):
            self.input_weights[i] = hidden_neurons[i].input_weights
            output_weights[i] = hidden_neurons[i].output_weights
        self.output_weights = np.zeros((output_count, len(hidden_neurons)))
        for i in range(output_count):
            self.output_weights[i] = output_weights[:, i]
        self.layers = []
        self.layers.append(Layer(
            weights=self.input_weights,
            activation=Sigmoid()))
        self.layers.append(Layer(
            weights=self.output_weights,
            activation=Sigmoid()))

    def forward(self, input_data: np.array) -> np.array:
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
