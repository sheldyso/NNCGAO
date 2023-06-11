import numpy as np
from typing import List
from network_contents import Layer
from math import log
import copy

class NeuralNetworkClassifier():
    """
    # Neural Network Classifier
    This custom neural network was designed around the idea of making it usable with any 
    given classification dataset such as the R.A Fisher iris data set.

    ## Setup
    The input and output layers of the network will automatically scale to the number of features given and number of classes in the data. Each data 
    point must be passed in as a list and of equal length. The class labels must be passed in as a single list with integer values. To customise 
    the size of the network, pass in a desired layout to the hidden_layer_sizes parameter.
    For example, to have a network structure of input layer, 5 neurons, 3 neurons, output layer; pass in hidden_layer_sizes = [5, 3]. By default, it is set to one
    hidden layer of 3 neurons.

    ## Optimising
    This neural network class has been designed to be passed into optimiser class and does not optimise during each epoch. The optimiser will call the train method once
    which will calculate the cross entropy loss, calculate the sum, then divide the sum by 1 to get a score. High loss will tend towards
    0 and low loss toward 1. Once the optimisation criteria has been met, it will return the network in an optimised state. Only then will the predict method yield
    an accurate prediction.

    ## Example
    
    >>> from neural_network import NeuralNetworkClassifier, GeneticAlgorithmOptimiser
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> irisdata = np.array(iris.data)
    >>> iristarget = np.array(iris.target)
    >>> joined_data = list(zip(irisdata, iristarget))
    >>> random.shuffle(joined_data)
    >>> irisdata, iristarget = zip(*joined_data)
    >>> train_size = int(0.8 * len(irisdata))
    >>> train_data, train_labels = irisdata[:train_size], iristarget[:train_size] 
    >>> test_data, test_labels = irisdata[train_size:], iristarget[train_size:]
    >>> model = NeuralNetworkClassifier(train_data, train_labels, test_data, test_labels, hidden_layer_sizes=[5, 3])
    >>> optimiser = GeneticAlgorithmOptimiser(model, 100, 100, 0.02)
    >>> opt_model, datalogger = optimiser.execute()
    
    opt_model.predict can then be called and will return a label prediction.

    """

    def __init__(self, train_data : list, train_labels : list, test_data : list, test_labels : list, 
                 hidden_layer_sizes : List[int] = [3]) -> None:

        self._has_learned = False
        self.__data = train_data
        self.__labels = train_labels
        self.__test_data = test_data
        self.__test_labels = test_labels
        self.__encoded_labels : List[List] = []

        # Input neurons scale to number of features given
        self.__input_layer_size = len(train_data[0])

        # Size of output layer should be the number of unique labels (0 - n) + 1 to include 0 label
        self.__output_layer_size = max(train_labels) + 1 
        
        self.__num_hidden_layers = len(hidden_layer_sizes)
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.__layers : List[Layer] = []
        self.mean_sum_scores : List[float] = []
        self.average_losses : List[float] = []
        self.accuracy = 0.0
        # ================================================
        # --- Setup ---
        self.__setup()
        
    def __setup(self):

        # Number 2 comes from default input and output layer
        layer_count = 2 + self.__num_hidden_layers

        for i in range(0, layer_count):
            if i == 0:
                self.__layers.append(Layer(self.__input_layer_size, self.__hidden_layer_sizes[i], True))

            elif i == layer_count - 2: # One just before output layer
                self.__layers.append(Layer(self.__hidden_layer_sizes[-1], self.__output_layer_size, False))

            elif i == layer_count - 1:
                self.__layers.append(Layer(self.__output_layer_size, 0, False))

            else:
                # i offset by -1 as for when it accesses the hidden layer sizes, i = 0 has already been passed but needs to be used here.
                self.__layers.append(Layer(self.__hidden_layer_sizes[i - 1], self.__hidden_layer_sizes[i]))

        print(f"Number of layers: {len(self.__layers)}")

        self.encode_labels() # one hot encoder

    def __normalise(self, array : List[float]) -> List[float]:
        array_copy = array.copy()
        min_val = min(array)
        max_val = max(array)
        difference = max_val - min_val
        # formula:
        # 
        # normx = (x - array_minimum) / (array_maximum - array_minimum)

        return [((array_copy[i] - min_val) / difference) for i in range(0, len(array_copy))]

    def __cross_entropy(self, actual : List, predicted : List) -> float:
        cross_scores : List[float] = []
        for i in range(0, len(actual)):
            cross_scores.append(actual[i] * log(predicted[i]))

        return -sum(cross_scores)

    def encode_one_label(self, class_position : int) -> list:
        empty = np.zeros(max(self.__labels) + 1)
        empty[class_position] = 1.0
        return empty

    def encode_labels(self):
        empty = np.zeros(max(self.__labels) + 1, dtype=int) # + 1 to Account for zero
        for i in range(0, len(self.__labels)):
            empty_copy = empty.copy()
            # Set the class position to 1 in the zero list e.g class 1 = [0, 1, 0]
            # Used for cross-entropy
            empty_copy[self.__labels[i]] = 1.0
            self.__encoded_labels.append(empty_copy)

    def get_network_weights(self):
        weights = []
        for i in range(0, len(self.__layers) - 1): # - 1 to exclude output layer
            weights.append(self.__layers[i].get_ordered_weights())

        return weights

    def set_network_weights(self, new_weights : List[List[List[float]]]):

        # List[layer_basis[each_neuron_weights[float_weight_values]]]

        for i in range(0, len(self.__layers) - 1): # - 1 to exclude output layer
            self.__layers[i].set_weights(new_weights=new_weights[i])

    def get_hidden_layer_size(self):
        return self.__hidden_layer_sizes

    def get_input_size(self):
        return self.__input_layer_size

    def feed_forward(self, data : List[float]) -> List[float]:
        # Feed the input features into the first layer
        self.__layers[0].set_inputs(self.__normalise(data))

        for layer in range(0, len(self.__layers)):
            if layer == 0: # Skip past input layer
                continue

            elif layer == len(self.__layers): # Break when output layer reached
                break

            else:

                in_weights = self.__layers[layer-1].get_ordered_weights()
                inputs = self.__layers[layer-1].get_outputs()

                for i in range(0, len(in_weights)):
                    self.__layers[layer].neurons[i]._weighted_sum(inputs, in_weights[i])
                    

        return self.__layers[-1].get_classes() # Returns a probability distribution

    def __argmax(self, prob_dist : List[float]) -> List[float]:
        template = np.zeros(len(prob_dist))
        max_value_pos = 0
        max_value = 0
        for i in range(len(prob_dist)):
            if prob_dist[i] > max_value:
                max_value = prob_dist[i]
                max_value_pos = i

        for i in range(len(template)):
            if i == max_value_pos:
                template[i] = 1

            else: template[i] = 0

        return template

    def calc_accuracy(self):
    
            num_labels = len(self.__test_data)
            predicted_correct = 0

            for i in range(len(self.__test_data)):

                data = self.__test_data[i]
                label = self.encode_one_label(self.__test_labels[i])
                pred = self.feed_forward(data)
                pred = self.__argmax(pred)
                
                if np.array_equiv(pred, label):
                    predicted_correct += 1

            score = (predicted_correct / num_labels) * 100
            return round(score, 2)

    def train(self, epochs):
        self.mean_sum_scores.clear()
        self.average_losses.clear()
        for _ in range(0, epochs + 1): # Account for 0 epochs
            entropy_scores = []
            for i in range(0, len(self.__data)):
                classes = self.feed_forward(self.__data[i])
                entropy_scores.append(self.__cross_entropy(self.__encoded_labels[i], classes))

            self.mean_sum_scores.append(1.0 / sum(entropy_scores) + 1e-15)
            self.average_losses.append(sum(entropy_scores) / len(entropy_scores))
            self.accuracy = self.calc_accuracy()

    def __decode_to_label(self, argmax_list : list):
        for i in range(len(argmax_list)):
            if argmax_list[i] == 1.0:
                return i

    def predict(self, data : list):
        if len(data) != len(self.__layers[0].neurons):
            print("Input data not the same length as training data!")
            return None
        
        else:
            norm_data = self.__normalise(data)
            argmax_list = self.__argmax(self.feed_forward(norm_data))
            return self.__decode_to_label(argmax_list)



class GeneticAlgorithmOptimiser():
    """
    # Genetic Algorithm Optimiser
    This optimiser is for use with the NeuralNetworkClassifier.

    ## Description
    This optimiser uses non-deterministic, evolutionary-based techniques to optimise a series of neural network weights. The
    optimiser will extract the non-optimised weight configuration from the neural network and unpack it into a single sequence which
    mimics a chromosome structure, with each weight representing a gene.

    ## Setup
    Pass in the neural network to optimise and specify the desired number of generations, population size and mutation rate. Adjust the mutation rate to control the networks convergence.
    It's recommended to limit the mutation rate between 0.001 and 0.05. After each generation is complete, the optimiser will print the loss. Once the generation count has been reached, it will return
    the optimised model and a dictionary containing the loss and accuracy during training. In case of a bad mutation before it returns the model, the best sequence is tracked and will be set to the network
    before completion.

    ## Example
    
    >>> from neural_network import NeuralNetworkClassifier, GeneticAlgorithmOptimiser
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> irisdata = np.array(iris.data)
    >>> iristarget = np.array(iris.target)
    >>> joined_data = list(zip(irisdata, iristarget))
    >>> random.shuffle(joined_data)
    >>> irisdata, iristarget = zip(*joined_data)
    >>> train_size = int(0.8 * len(irisdata))
    >>> train_data, train_labels = irisdata[:train_size], iristarget[:train_size] 
    >>> test_data, test_labels = irisdata[train_size:], iristarget[train_size:]
    >>> model = NeuralNetworkClassifier(train_data, train_labels, test_data, test_labels, hidden_layer_sizes=[5, 3])
    >>> optimiser = GeneticAlgorithmOptimiser(model, 100, 100, 0.02)
    >>> opt_model, datalogger = optimiser.execute()
    
    opt_model.predict can then be called and will return a label prediction.
    """
    def __init__(self, network : NeuralNetworkClassifier, n_generations : int, population_size : int, mutation_rate : float = 0.05) -> None:
        self.network : NeuralNetworkClassifier = network
        self.weights : List[List[List[float]]] = self.network.get_network_weights()
        self.pop_size = population_size
        if self.pop_size % 2 != 0: self.pop_size += 1
        self.__n_generations = n_generations
        self.__population : List = []
        self.__sequences : List[float] = []
        self.__scores : List = []
        self.__initial_calc : bool = False
        self.datalogger = {"test_loss_per_generation" : [], "test_accuracy_per_generation" : []}
        self.__mutation_rate = mutation_rate
        self.__best_sequence = None
        self.__best_score = 0

    def fitness_function(self, weight_configuration : List[List[List[float]]]):
        self.network.set_network_weights(weight_configuration)
        self.network.train(0) # Calls train(0) so the entire train data is passed through once.
        return self.network.mean_sum_scores[0] 
        # Returns a value between 1.0 and 0 with close to 1 being low loss and 0 being high.

    def __get_random_sequence(self, weight_structure : List[List[List[float]]]) -> List:
        sequence = []
        network_weights = weight_structure.copy()
        for layer in range(0, len(network_weights)): 
            for neuron in range(0, len(network_weights[layer])):
                for weight in range(0, len(network_weights[layer][neuron])):
                    network_weights[layer][neuron][weight] = round(np.random.uniform(0.0, 1.0), 3)
                    sequence.append(network_weights[layer][neuron][weight])

        return sequence

    def sequence_to_weight_config(self, sequence : list, weight_structure : List[List[List[float]]]) -> List[List[List[float]]]:
        network_weights = copy.deepcopy(weight_structure)
        sequence_index = 0
        for layer in range(0, len(network_weights)): 
                for neuron in range(0, len(network_weights[layer])):
                    for weight in range(0, len(network_weights[layer][neuron])):
                        network_weights[layer][neuron][weight] = sequence[sequence_index]
                        sequence_index += 1

        return network_weights

    def create_population(self):
        self.__population.clear()
        self.__scores.clear()
        for _ in range(0, self.pop_size):
            self.__sequences.append(self.__get_random_sequence(weight_structure = self.weights))
        
        self.insert_sequences_to_weights(start_point = 0)
                    
    def calculate_fitness(self):
        # If new population, calculate over entire population
        if self.__initial_calc == False:
            for weight_config in range(0, len(self.__population)):
                self.__scores.append(self.fitness_function(self.__population[weight_config]))

            self.__initial_calc = True

        else:
            for weight_config in range(int(len(self.__population) / 2), len(self.__population)):
                self.__scores.append(self.fitness_function(self.__population[weight_config]))
            

    def selection(self):
        self.__sequences = [x for _, x in sorted(zip(self.__scores, self.__sequences), reverse=True)]
        self.__scores = sorted(self.__scores, reverse=True)
        if self.__scores[0] > self.__best_score:
            self.__best_score = self.__scores[0]
            self.__best_sequence = self.__sequences[0]

        half_point = int(len(self.__sequences) / 2)
        self.__sequences = self.__sequences[:half_point]
        self.__population = self.__population[:half_point]
        # Keep existing scores so recalculation is not required
        self.__scores = self.__scores[:half_point]

    def __crossover(self, sequence_1 : list, sequence_2 : list) -> List:
        # Not an order based sequence so an ordered crossover is not needed
        # Could explore single, and multipoint crossover

        half_point = int(len(sequence_1) / 2)
        child_one = sequence_1[:half_point] + sequence_2[half_point:]
        child_two = sequence_2[:half_point] + sequence_1[half_point:]
        # Two offspring to increase diversity

        return (child_one, child_two)

    def population_crossover(self):
        crossover_children = []
        position = 0
        sequence_length = len(self.__sequences)
        while position + 2 <= sequence_length:
            children = self.__crossover(self.__sequences[position], self.__sequences[position + 1])
            crossover_children.append(children[0])
            crossover_children.append(children[1])

            position += 2

        self.__sequences.extend(crossover_children)


    def __mutation_amount(self, min_value : float, max_value : float, value_flip_chance : float) -> float:
        mutation_amount = np.random.uniform(min_value, max_value)

        # Add a chance to flip the sign of the value
        value_flip = np.random.uniform(0.0, 1.0)
        if value_flip < value_flip_chance:
            return -mutation_amount
        
        else: return mutation_amount

    def mutate(self, sequence : list, mutation_rate : float):
        sequence_copy = sequence.copy()
        for gene in range(0, len(sequence_copy)):
            mutation_chance = np.random.uniform(0.0, 1.0)
            if mutation_chance < mutation_rate:
                sequence_copy[gene] += self.__mutation_amount(-2.0, 2.0, 0.2)
        
        return sequence_copy

    def mutation(self):
        # Only mutate new children
        for i in range(int(len(self.__sequences) / 2), len(self.__sequences)):
            # Only 5% chance to mutate to reduce likelihood of rapid mutation 
            # and losing better sequences
            self.__sequences[i] = self.mutate(self.__sequences[i], self.__mutation_rate)

    def insert_sequences_to_weights(self, start_point : int) -> List:
        for i in range(int(start_point), len(self.__sequences)):
            self.__population.append(self.sequence_to_weight_config(self.__sequences[i], self.weights))

    def execute(self):
        self.create_population()
        for _ in range(0, self.__n_generations):
            self.calculate_fitness()
            self.selection()
            self.population_crossover()
            self.mutation()
            self.insert_sequences_to_weights(start_point=len(self.__sequences) / 2)
            print(f"Generation {_} complete. Loss: {self.network.average_losses[-1]:.3f}")
            self.datalogger["test_loss_per_generation"].append(self.network.average_losses[-1])
            self.datalogger["test_accuracy_per_generation"].append(self.network.accuracy)

        self.network.set_network_weights(self.sequence_to_weight_config(
            self.__best_sequence, self.weights
            ))

        return self.network, self.datalogger

def argmax(prob_dist : List[float]) -> List[float]:
    template = np.zeros(len(prob_dist))
    max_value_pos = 0
    max_value = 0
    for i in range(len(prob_dist)):
        if prob_dist[i] > max_value:
            max_value = prob_dist[i]
            max_value_pos = i

    for i in range(len(template)):
        if i == max_value_pos:
            template[i] = 1

        else: template[i] = 0

    return template

def accuracy_test(test_data : List[list], test_labels : List[list], model : NeuralNetworkClassifier):
    
    if len(test_data) == len(test_labels):

        num_labels = len(test_data)
        predicted_correct = 0

        for i in range(len(test_data)):

            data = test_data[i]
            label = test_labels[i]
            pred = model.feed_forward(data)
            pred = argmax(pred)
            
            if np.array_equiv(pred, label):
                predicted_correct += 1

        score = (predicted_correct / num_labels) * 100
        return round(score, 2)
