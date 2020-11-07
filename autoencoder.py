import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
letras = pickle.load(open('letras.pickle', 'rb'))

_input = letras
_expected = letras

learning_rate = 0.1
momentum = 0.8
test_p = 0.1



nn = MultilayerPerceptron(learning_rate, momentum, act_fun="tanh", deriv_fun="tanh_d", split_data=False, test_p=test_p, use_momentum=True)

nn.entry_layer(7)
nn.add_hidden_layer(5)
nn.add_hidden_layer(3)
nn.add_hidden_layer(2)
nn.add_hidden_layer(3)
nn.add_hidden_layer(5)
nn.output_layer(7)

error = 1
error = nn.train(_input, _expected, epochs=500)

#for i in range(0, len(_input)):
#    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))
