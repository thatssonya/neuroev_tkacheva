from typing import List
import numpy as np
from statistics import mean
import time
from .neuron_population import NeuronPopulation
from .neural_network import NeuralNetwork
from .neuron import Neuron
from .utils import mse


def forward_train(neural_network: NeuralNetwork,
                  x_train: np.array,
                  y_train: np.array) -> float:
    dataset_size = x_train.shape[0]
    errors = []
    for i in range(dataset_size):
        output = neural_network.forward(input_data=x_train[i])
        error = mse(y_true=y_train[i], y_pred=output)
        errors.append(error)
    return np.array(errors).mean()


def increment_trials(neurons: List[Neuron]):
    for neuron in neurons:
        neuron.trials += 1


class ESPAlgorithm(object):
    def __init__(self,
                 hidden_layer_size: int,
                 population_size: int,
                 input_count: int,
                 output_count: int,
                 last_generations_count: int,
                 trials_per_neuron: int):
        self.population = NeuronPopulation(
            population_size=hidden_layer_size,
            subpopulation_size=population_size,
            input_count=input_count,
            output_count=output_count,
            last_generations_count=last_generations_count,
            trials_per_neuron=trials_per_neuron)
        self.best_nn = None
        self.best_nn_fitness = 0.0

    def init(self, min_value: float, max_value: float):
        self.population.init(
            min_value=min_value,
            max_value=max_value)

    def train(self,
              generations_count: int,
              x_train: np.array,
              y_train: np.array):
        result = []
        for generation in range(generations_count):
            start_time = time.time()
            trials_count = self.check_fitness(
                x_train=x_train,
                y_train=y_train)
            self.population.fit_avg_fitness()
            if self.update_best_nn():
                result.append((generation, self.best_nn_fitness))
            self.population.check_degeneration()
            self.population.crossover()
            self.population.mutation()
            full_time = time.time() - start_time
            print('Поколение {0:>5d}, Количество попыток {1:>3d}, '
                  'Приспособленность лучшего нейрона {2:2.6f}, Время выполнения {3:3.3f} s'
                  .format(generation, trials_count, self.best_nn_fitness, full_time))
            self.population.reset_trials()
        return result

    def update_best_nn(self):
        best_neurons = self.population.get_best_neurons()
        best_nn = NeuralNetwork(
            hidden_neurons=best_neurons)
        best_nn_fitness = mean([neuron.avg_fitness for neuron in best_neurons])
        if self.best_nn is None or best_nn_fitness < self.best_nn_fitness:
            self.best_nn = best_nn
            self.best_nn_fitness = best_nn_fitness
            return True
        return False

    def check_fitness(self,
                      x_train: np.array,
                      y_train: np.array) -> int:
        trials_count = 0
        while not self.population.is_trials_completed():
            selected_neurons = self.population.get_neurons()
            increment_trials(selected_neurons)
            neural_network = NeuralNetwork(
                hidden_neurons=selected_neurons)
            error = forward_train(
                neural_network=neural_network,
                x_train=x_train,
                y_train=y_train)
            for neuron in selected_neurons:
                neuron.cumulative_fitness += error
            trials_count += 1
        return trials_count
