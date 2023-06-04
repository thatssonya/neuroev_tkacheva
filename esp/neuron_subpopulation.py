import random
import numpy as np
from collections import deque
from .neuron import Neuron


class NeuronSubPopulation(object):
    def __init__(self,
                 population_size: int,
                 input_count: int,
                 output_count: int,
                 last_generations_count: int,
                 trials_per_neuron: int,
                 subpopulation_id: int):
        self.population = []
        for i in range(population_size):
            self.population.append(Neuron(
                input_count=input_count,
                output_count=output_count,
                neuron_id=i))
        self.last_generations_count = last_generations_count
        self.trials_per_neuron = trials_per_neuron
        self.id = subpopulation_id
        self.generation = 0
        self.best_neurons = {}
        # self.population_size = population_size

    def init(self, min_value: float, max_value: float):
        for neuron in self.population:
            neuron.init(
                min_value=min_value,
                max_value=max_value)

    def get_neuron(self) -> Neuron:
        return random.choice(self.population)

    def is_trials_completed(self) -> bool:
        trials = [neuron.trials for neuron in self.population]
        return min(trials) >= self.trials_per_neuron

    def reset_trials(self):
        for neuron in self.population:
            neuron.trials = 0

    def fit_avg_fitness(self):
        for neuron in self.population:
            neuron.fit_avg_fitness()

    def crossover(self):
        self.population.sort(key=lambda x: x.avg_fitness)
        selected_neurons_count = int(len(self.population) / 4)
        selected_neurons_count -= selected_neurons_count % 2
        for i in range(0, selected_neurons_count, 2):
            parent1 = self.population[i]
            parent2 = self.population[i + 1]
            child1, child2 = Neuron.crossover(
                parent1=parent1,
                parent2=parent2)
            self.population[-selected_neurons_count + i] = child1
            self.population[-selected_neurons_count + i + 1] = child2

    def mutation(self):
        bottom_half = int(len(self.population) / 2)
        for neuron in self.population[bottom_half:]:
            neuron.mutation()

    def get_best_neuron(self) -> Neuron:
        self.population.sort(key=lambda x: x.avg_fitness)
        return self.population[0]

    def check_degeneration(self):
        best_neuron = self.get_best_neuron()
        if best_neuron.id in self.best_neurons.keys():
            self.best_neurons[best_neuron.id].append(best_neuron.avg_fitness)
        else:
            self.best_neurons[best_neuron.id] = deque(maxlen=self.last_generations_count)
        clear_best_neurons = False
        for neuron_id, fitness_list in self.best_neurons.items():
            if len(fitness_list) == fitness_list.maxlen:
                if self.population[neuron_id].avg_fitness > min(fitness_list):
                    self.burst_mutation(neuron=best_neuron)
                    clear_best_neurons = True
                    break
        if clear_best_neurons:
            self.best_neurons = {}

    def burst_mutation(self, neuron: Neuron):
        print('Взрывная мутация для подпопуляции {0:>3d}. Текущее поколение {1:>3d}'
              .format(self.id, self.generation))
        input_count = neuron.input_count
        output_count = neuron.output_count
        new_population = []
        for i in range(len(self.population)):
            new_neuron = Neuron(
                input_count=input_count,
                output_count=output_count,
                neuron_id=i)
            new_neuron.input_weights = \
                np.random.standard_cauchy(input_count) * 0.05 + neuron.input_weights
            new_neuron.output_weights = \
                np.random.standard_cauchy(output_count) * 0.05 + neuron.output_weights
            new_population.append(new_neuron)
        self.population = new_population
        self.generation += 1
