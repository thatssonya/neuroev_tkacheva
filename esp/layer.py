import numpy as np
from .activations import AbstractActivationFunction


class Layer(object):
    def __init__(self,
                 weights: np.array,
                 activation: AbstractActivationFunction):
        self.weights = weights
        self.activation = activation

    def forward(self,
                input_data: np.array) -> np.array:
        return self.activation.forward(
            input_data=np.dot(self.weights, input_data))
