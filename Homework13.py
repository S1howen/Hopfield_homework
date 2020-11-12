# ANN Homework 1.3 Stochastic Hopfield network
# import of needed packages
import numpy as np
import math
import random

# functions


def calc_order_parameter(system_state, pattern_list):
    n = len(pattern_list)
    sum_param = 0
    v_state = system_state[:, 0]
    # print(pattern_list[0].pattern)
    # print(np.array_equal(system_state, pattern_list[0].pattern))
    """
    for i in range(n):
        pattern_v = pattern_list[i].pattern[:, 0]
        sum_param += np.dot(v_state, pattern_v)
    
     """
    order_parameter = (np.dot(v_state, pattern_list[0].pattern[:, 0]))/200
    # print(order_parameter)
    return order_parameter


def stochastic_update(x, beta):

    output = 1/(1 + math.exp(-2*beta*x))
    rand_number = random.random()
    if rand_number < output:
        output = 1
    else:
        output = -1
    return output


def change_array2vector(array):
    # n_rows = np.shape(array)[0]
    # n_columns = np.shape(array)[1]
    vector = array.ravel()
    vector = vector.reshape(-1, 1)
    return vector


def create_initial_weight_matrix(n_bits):
    w = np.zeros(shape=(n_bits, n_bits))
    return w


def sigmoid_func(x):
    if x >= 0:
        output = 1
    else:
        output = -1
    return output


class Pattern:

    def __init__(self, n_bits, random=True):
        self.n_bits = n_bits
        if random:
            self.pattern = np.random.randint(2, size=(self.n_bits, 1))
            self.pattern[self.pattern == 0] = -1
        else:
            self.pattern = None

    def store_pattern(self, pattern):
        self.pattern = pattern


def calc_hemming_dist(pattern, pattern_to_compare):

    hamming_dist = 0
    for i in range(len(pattern)):
        hamming_dist += pattern[i, 0] - pattern_to_compare[i, 0]

    return hamming_dist

# specialized Hopfield network class for this task


class HopfieldNetwork:

    def __init__(self, n_bits, pattern_list=None, beta=2):
        self.w = create_initial_weight_matrix(n_bits)
        self.store_patterns_in_w(pattern_list)
        self.n_neurons = n_bits
        self.system_state = np.copy(pattern_list[0].pattern)
        self.beta = beta

    def store_patterns_in_w(self, list_patterns):
        print("{} patterns will be stored".format(len(list_patterns)))
        for pattern in list_patterns:
            self.w += np.outer(pattern.pattern, pattern.pattern)/len(pattern.pattern)
            np.fill_diagonal(self.w, 0)

    def compute_neuron_state(self, neuron_index):
        # old_state = self.system_state[neuron_index, 0]
        weight_vector = self.w[neuron_index, :]
        b = np.dot(weight_vector, self.system_state)
        new_state = stochastic_update(b, self.beta)
        # print("old state was: {} now new state is: {}".format(old_state, new_state))
        return new_state

    def update_random_neuron(self):
        neuron_idx = np.random.randint(self.n_neurons)
        # print("index {} was chosen".format(neuron_idx))
        old_state = self.system_state[neuron_idx, 0]
        new_state = self.compute_neuron_state(neuron_idx)
        """
        if old_state != new_state: 
            print('changed')
        """
        self.system_state[neuron_idx, 0] = new_state

# define the parameters

beta = 2
n_neurons = 200
n_patterns = 7
n_updates = 2*10**5
n_rounds = 100

order_param_sum = 0
order_parameter = 0

# start the simulation
for i in range(n_rounds):
    print(i)
    temp_order_parameter = 0
    temp_param = 0
    pattern_list = [Pattern(n_neurons) for x in range(n_patterns)]
    hopfield = HopfieldNetwork(n_neurons, pattern_list)
    for k in range(n_updates):
        hopfield.update_random_neuron()
        current_state = hopfield.system_state[:]
        temp_order_parameter += calc_order_parameter(current_state, pattern_list)

    order_param_sum += temp_order_parameter/n_updates


average_order_param = order_param_sum/n_rounds
print(average_order_param)

# second part of the task
beta = 2
n_neurons = 200
n_patterns = 45
n_updates = 2*10**5
n_rounds = 100

order_param_sum = 0
order_parameter = 0

# start the simulation again
for i in range(n_rounds):
    print(i)
    temp_order_parameter = 0
    temp_param = 0
    pattern_list = [Pattern(n_neurons) for x in range(n_patterns)]
    hopfield = HopfieldNetwork(n_neurons, pattern_list)
    for k in range(n_updates):
        hopfield.update_random_neuron()
        current_state = hopfield.system_state[:]
        temp_order_parameter += calc_order_parameter(current_state, pattern_list)

    order_param_sum += temp_order_parameter/n_updates


average_order_param = order_param_sum/n_rounds
print(average_order_param)
