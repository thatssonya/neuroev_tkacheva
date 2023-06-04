import numpy as np
import random


def crossover_weights(parent1: np.array, parent2: np.array):
    weights_count = parent1.shape[0]
    crossover_point = random.randrange(weights_count)
    child1, child2 = np.zeros(weights_count), np.zeros(weights_count)
    child1[:crossover_point] = parent1[:crossover_point]
    child1[crossover_point:] = parent2[crossover_point:]
    child2[:crossover_point] = parent2[:crossover_point]
    child2[crossover_point:] = parent1[crossover_point:]
    return child1, child2


class Neuron(object):
    def __init__(self,
                 input_count: int,
                 output_count: int,
                 neuron_id: int):
        self.input_count = input_count
        self.output_count = output_count
        self.id = neuron_id
        self.input_weights = None
        self.output_weights = None
        self.cumulative_fitness = 0.0
        self.avg_fitness = 0.0
        self.trials = 0

    def init(self, min_value: float, max_value: float):
        self.input_weights = np.random.uniform(
            low=min_value,
            high=max_value,
            size=self.input_count)
        self.output_weights = np.random.uniform(
            low=min_value,
            high=max_value,
            size=self.output_count)

    def fit_avg_fitness(self):
        self.avg_fitness = self.cumulative_fitness / self.trials

    def mutation(self):
        self.input_weights += np.random.standard_cauchy(self.input_count) * 0.05
        self.output_weights += np.random.standard_cauchy(self.output_count) * 0.05

    @staticmethod
    def crossover(parent1, parent2):
        input_count = parent1.input_count
        output_count = parent1.output_count
        child1 = Neuron(
            input_count=input_count,
            output_count=output_count,
            neuron_id=parent1.id)
        child2 = Neuron(
            input_count=input_count,
            output_count=output_count,
            neuron_id=parent2.id)
        child1.input_weights, child2.input_weights = crossover_weights(
            parent1=parent1.input_weights,
            parent2=parent2.input_weights)
        child1.output_weights, child2.output_weights = crossover_weights(
            parent1=parent1.output_weights,
            parent2=parent2.output_weights)
        return child1, child2
