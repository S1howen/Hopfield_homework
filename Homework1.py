# Homework 1 One step error probability
# import of needed packages
import numpy as np
import random
# task 1: asynchronus deterministic hopfield networks


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


def create_initial_weight_matrix(n_bits):
    w = np.zeros(shape=(n_bits, n_bits))
    return w


def sigmoid_func(x):
    if x < 0:
        output = -1
    else:
        output = 1
    return output


class HopfieldNetwork:

    def __init__(self, n_bits, pattern_list=None, diagonal_weights_zero=True):
        self.w = create_initial_weight_matrix(n_bits)
        self.store_patterns_in_w(pattern_list, diagonal_weights_zero)
        self.n_neurons = n_bits
        self.system_state = np.copy(pattern_list[random.randrange(len(pattern_list))].pattern)
        # self.system_state = np.copy(pattern_list[0].pattern)

    def store_patterns_in_w(self, list_patterns, diagonal_weights_zero=False):
        # print("{} patterns will be stored".format(len(list_patterns)))
        for pattern in list_patterns:
            self.w += np.outer(pattern.pattern, pattern.pattern)/len(pattern.pattern)
            if diagonal_weights_zero:
                np.fill_diagonal(self.w, 0)

    def compute_neuron_state(self, neuron_index):
        old_state = self.system_state[neuron_index, 0]
        weight_vector = self.w[neuron_index, :]
        b = np.dot(weight_vector, self.system_state)
        new_state = sigmoid_func(b)
        # print("old state was: {} now new state is: {}".format(old_state, new_state))
        return new_state

    def update_random_neuron(self):
        neuron_idx = np.random.randint(self.n_neurons)
        old_state = self.system_state[neuron_idx, 0]
        # print("index {} was chosen".format(neuron_idx))
        new_state = self.compute_neuron_state(neuron_idx)
        self.system_state[neuron_idx, 0] = new_state
        return new_state, old_state



# define number of bits per pattern
N = 120

# define the number of patterns:
n_p_list = [12, 24, 48, 70, 100, 120]
# define the list with the random patterns

# define number of trials
n_trials = 10**5

# prepare the loop
error_1 = 0
error_2 = 0
error_3 = 0
error_4 = 0
error_5 = 0
error_6 = 0

for i in range(n_trials):

    p1 = [Pattern(N) for x in range(n_p_list[0])]
    p2 = [Pattern(N) for x in range(n_p_list[1])]
    p3 = [Pattern(N) for x in range(n_p_list[2])]
    p4 = [Pattern(N) for x in range(n_p_list[3])]
    p5 = [Pattern(N) for x in range(n_p_list[4])]
    p6 = [Pattern(N) for x in range(n_p_list[5])]

    hopfield1 = HopfieldNetwork(N, p1)
    new_state, old_state = hopfield1.update_random_neuron()
    if new_state != old_state:
        error_1 += 1

    hopfield2 = HopfieldNetwork(N, p2)
    new_state, old_state = hopfield2.update_random_neuron()
    if new_state != old_state:
        error_2 += 1

    hopfield3 = HopfieldNetwork(N, p3)
    new_state, old_state = hopfield3.update_random_neuron()
    if new_state != old_state:
        error_3 += 1

    hopfield4 = HopfieldNetwork(N, p4)
    new_state, old_state = hopfield4.update_random_neuron()
    if new_state != old_state:
        error_4 += 1

    hopfield5 = HopfieldNetwork(N, p5)
    new_state, old_state = hopfield5.update_random_neuron()
    if new_state != old_state:
        error_5 += 1

    hopfield6 = HopfieldNetwork(N, p6)
    new_state, old_state = hopfield6.update_random_neuron()
    if new_state != old_state:
        error_6 += 1

P_1 = error_1/n_trials
print(P_1)
P_2 = error_2/n_trials
print(P_2)
P_3 = error_3/n_trials
print(P_3)
P_4 = error_4/n_trials
print(P_4)
P_5 = error_5/n_trials
print(P_5)
P_6 = error_6/n_trials
print(P_6)

# prepare the second loop
error_12 = 0
error_22 = 0
error_32 = 0
error_42 = 0
error_52 = 0
error_62 = 0

for i in range(n_trials):

    p1 = [Pattern(N) for x in range(n_p_list[0])]
    p2 = [Pattern(N) for x in range(n_p_list[1])]
    p3 = [Pattern(N) for x in range(n_p_list[2])]
    p4 = [Pattern(N) for x in range(n_p_list[3])]
    p5 = [Pattern(N) for x in range(n_p_list[4])]
    p6 = [Pattern(N) for x in range(n_p_list[5])]

    hopfield1 = HopfieldNetwork(N, p1, diagonal_weights_zero=False)
    new_state, old_state = hopfield1.update_random_neuron()
    if new_state != old_state:
        error_12 += 1

    hopfield2 = HopfieldNetwork(N, p2, diagonal_weights_zero=False)
    new_state, old_state = hopfield2.update_random_neuron()
    if new_state != old_state:
        error_22 += 1

    hopfield3 = HopfieldNetwork(N, p3, diagonal_weights_zero=False)
    new_state, old_state = hopfield3.update_random_neuron()
    if new_state != old_state:
        error_32 += 1

    hopfield4 = HopfieldNetwork(N, p4, diagonal_weights_zero=False)
    new_state, old_state = hopfield4.update_random_neuron()
    if new_state != old_state:
        error_42 += 1

    hopfield5 = HopfieldNetwork(N, p5, diagonal_weights_zero=False)
    new_state, old_state = hopfield5.update_random_neuron()
    if new_state != old_state:
        error_52 += 1

    hopfield6 = HopfieldNetwork(N, p6,  diagonal_weights_zero=False)
    new_state, old_state = hopfield6.update_random_neuron()
    if new_state != old_state:
        error_62 += 1

P_12 = error_12/n_trials
print(P_12)
P_22 = error_22/n_trials
print(P_22)
P_32 = error_32/n_trials
print(P_32)
P_42 = error_42/n_trials
print(P_42)
P_52 = error_52/n_trials
print(P_52)
P_62 = error_62/n_trials
print(P_62)


