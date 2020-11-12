# ANN Homework 1.2 Recognising Digits
# import of needed packages
import numpy as np


def change_array2vector(array):
    # n_rows = np.shape(array)[0]
    # n_columns = np.shape(array)[1]
    vector = array.ravel()
    vector = vector.reshape(-1, 1)
    return vector


def calc_hemming_dist(pattern, pattern_to_compare):

    hamming_dist = 0
    for i in range(len(pattern)):
        hamming_dist += pattern[i, 0] - pattern_to_compare[i, 0]

    return hamming_dist


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
            self.first_state = self.pattern
        else:
            self.pattern = None
            self.first_state = None

    def store_pattern(self, pattern):
        self.first_state = pattern
        self.pattern = pattern


class HopfieldNetwork:

    def __init__(self, n_bits, initial_pattern, pattern_list=None):
        self.w = create_initial_weight_matrix(n_bits)
        self.store_patterns_in_w(pattern_list)
        self.n_neurons = n_bits
        self.system_state = initial_pattern.pattern

    def store_patterns_in_w(self, list_patterns):
        print("{} patterns will be stored".format(len(list_patterns)))
        for pattern in list_patterns:
            self.w += np.outer(pattern.pattern, pattern.pattern)/len(pattern.pattern)
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
        # print("index {} was chosen".format(neuron_idx))
        new_state = self.compute_neuron_state(neuron_idx)
        self.system_state[neuron_idx, 0] = new_state

    def update_in_order_neuron(self):
        for i in range(self.n_neurons):

            new_state = self.compute_neuron_state(i)
            self.system_state[i, 0] = new_state

    def update_until_convergence(self):
        # print("starting iterateive process")
        for i in range(50000):
            self.update_in_order_neuron()

        return self.system_state
        # print("pattern converges")


# given patterns
x1= np.array([ [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
               [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1], [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
               [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
                [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
               [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1], [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
               [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
                 [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
               [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] ])

x2 = np.array([ [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                 [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                 [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1]])

x3= np.array([ [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1], [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
               [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1], [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
               [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1], [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
               [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1], [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
               [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1], [ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
               [ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],  [1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
               [ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1], [ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
               [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1], [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1] ])

x4=np.array([ [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1] ,[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
                [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
              [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1], [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1] ,[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1]
                ,[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1], [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]])

x5=np.array([ [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
        [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
        [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
        [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
         [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
        [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1], [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
        [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1] ])

# paterns to feed to the network
first_pattern = np.array([[1, 1, 1, -1, -1, -1, -1, 1, 1, 1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [1, 1, 1, -1, -1, -1, -1, 1, 1, 1]])


second_pattern = np.array([[-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                           [-1, -1, -1, 1, 1, 1, 1, 1, -1, -1], [-1, -1, -1, 1, -1, 1, 1, 1, -1, -1],
                           [-1, -1, -1, 1, -1, 1, 1, 1, -1, -1], [-1, -1, -1, 1, -1, 1, 1, 1, -1, -1],
                           [-1, -1, -1, 1, -1, 1, 1, 1, -1, -1], [-1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
                           [-1, 1, 1, 1, 1, 1, 1, 1, -1, -1], [-1, 1, 1, 1, -1, -1, 1, -1, -1, -1],
                           [-1, 1, 1, 1, -1, -1, 1, -1, -1, -1], [-1, 1, 1, 1, -1, -1, 1, -1, -1, -1],
                           [-1, 1, 1, 1, -1, -1, 1, -1, -1, -1], [-1, -1, 1, 1, 1, 1, 1, -1, -1, -1],
                           [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1]])

third_pattern = np.array([[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1],
                           [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1],
                           [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1],
                           [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1],
                           [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1],
                           [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1],
                           [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1],
                           [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1]])


test1 = Pattern(160, False)
test1.store_pattern(change_array2vector(first_pattern))
test2 = Pattern(160, False)
test2.store_pattern(change_array2vector(second_pattern))
test3 = Pattern(160, False)
test3.store_pattern(change_array2vector(third_pattern))

# Initialize a hopfield net and store all the patterns

n_neurons = np.shape(x1)[0]*np.shape(x1)[1]

p1 = Pattern(n_neurons, random=False)
p1.store_pattern(change_array2vector(x1))
p2 = Pattern(n_neurons, random=False)
p2.store_pattern(change_array2vector(x2))
p3 = Pattern(n_neurons, random=False)
p3.store_pattern(change_array2vector(x3))
p4 = Pattern(n_neurons, random=False)
p4.store_pattern(change_array2vector(x4))
p5 = Pattern(n_neurons, random=False)
p5.store_pattern(change_array2vector(x5))

# make the udpates in the hopfield network and store the final results in picture_n with n :[1,2,3]
pattern_list = [p1, p2, p3, p4, p5]
hopfield1 = HopfieldNetwork(n_neurons, test1, pattern_list)
final_state_1 = hopfield1.update_until_convergence()
picture_1 = final_state_1.reshape(16, 10)

hopfield2 = HopfieldNetwork(n_neurons, test2,  pattern_list)
final_state_2 = hopfield2.update_until_convergence()
picture_2 = final_state_2.reshape(16, 10)

hopfield3 = HopfieldNetwork(n_neurons, test3, pattern_list)
final_state_3 = hopfield3.update_until_convergence()
picture_3 = final_state_3.reshape(16, 10)

# Find the matching patterns for all 3 subtasks


def check_pattern_for_similarity(pattern, pattern_list):
    similar_pattern = None
    for idx, pattern_to_compare in enumerate(pattern_list):
        if np.array_equal(pattern, pattern_to_compare.pattern):
            similar_pattern = pattern_to_compare.pattern
            # since pattern 0 is pattern number 1 in the list for the homework idx+1
            print("the pattern is the same as pattern number {}".format(idx+1))

    if similar_pattern is None:
        similar_pattern = pattern
        print('The final pattern is not in the list!')


check_pattern_for_similarity(final_state_1, pattern_list)
check_pattern_for_similarity(final_state_2, pattern_list)
check_pattern_for_similarity(final_state_3, pattern_list)

print(picture_1)
print(picture_2)
print(picture_3)
