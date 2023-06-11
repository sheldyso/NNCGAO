import numpy as np
from typing import List
np.random.seed(42)
import copy

class Neuron():
    def __init__(self, connections : int) -> None:
        self.__value : float = 0.0 
        # Each neuron will have a set of weights like [0.24, 0.13, 0.3, 0.98].
        # Length of the weight list should be the same as the number of neurons in the next layer.
        self.out_weights = []
        self._initialise_weights(num_connections = connections)

    def _weighted_sum(self, inputs : list, weights : list) -> float:
        self.__value = self._relu(np.sum(np.dot(inputs, weights)))
        # y = (weight * input) + bias

    def _relu(self, x):
        # relu( weighted_sum )

        return max(0.0, x)

    def get_weights(self) -> List:
        return self.out_weights

    def _initialise_weights(self, num_connections):
        self.out_weights.clear()
        if num_connections > 0:
            for _ in range(0, num_connections):
                self.out_weights.append(np.round(np.random.uniform(0.0, 1.0), 3))
        
        else: self.out_weights = []

    def set_input(self, value : float): # Only called once each feed forward and sets the input data at start
        self.__value = value

    def get_value(self) -> float:
        return self.__value

class Layer():
    def __init__(self, num_neurons : int, next_layer_size : int = None, input_layer : bool = None) -> None:
        self.neurons : List[Neuron] = []
        self.__ordered_weights : List[List] = []
        if input_layer:
            # Set neuron as input
            self.neurons = [Neuron(connections=next_layer_size) for _ in range(0, num_neurons)]
            
        elif next_layer_size == 0:
            self.neurons = [Neuron(connections=0) for _ in range(0, num_neurons)]
            # Set neuron as output

        else:
            self.neurons = [Neuron(connections=next_layer_size) for _ in range(0, num_neurons)]
            # Don't modify neuron type

        self.__weights = [self.neurons[i].get_weights() for i in range(0, len(self.neurons))]
        self.__order_weights()

    def __order_weights(self):
        weights = copy.deepcopy(self.__weights)

        while len(weights[0]) != 0:
            order = []
            for i in range(0, len(weights)):
                order.append(weights[i].pop(0))

            self.__ordered_weights.append(order)

    def get_ordered_weights(self) -> List[List[float]]:
        return self.__ordered_weights
    
    def set_weights(self, new_weights : List[List[float]]):
        self.__ordered_weights = new_weights
    
    def set_inputs(self, data : List[float]):
        # Set the input to each neuron
        for i in range(0, len(self.neurons)):
            self.neurons[i].set_input(data[i])

    def get_outputs(self):
        return [self.neurons[i].get_value() for i in range(0, len(self.neurons))]

    def __softmax(self, array : list):
        exp_array = np.exp(array)
        return exp_array/exp_array.sum()

    def get_classes(self):
        outputs = self.get_outputs()
        return self.__softmax(outputs)