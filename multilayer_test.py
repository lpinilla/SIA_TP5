import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
_input = [[0, 0], [0,1], [1, 0], [1, 1]]
_expected = [[0], [1], [1], [0]]

learning_rate = 0.5
momentum = 0.8
test_p = 0.25


nn = MultilayerPerceptron(learning_rate, momentum, act_fun="arctan", deriv_fun="arctan_d", split_data=False, test_p=test_p)

nn.entry_layer(2)
nn.add_hidden_layer(5)
nn.output_layer(1)

error = 1
error = nn.train(_input, _expected, epochs=500)

for i in range(0, len(_input)):
    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))
