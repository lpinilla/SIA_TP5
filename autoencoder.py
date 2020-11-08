import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
letras = pickle.load(open('letras.pickle', 'rb'))

_input = letras
_expected = letras

learning_rate = 0.001
momentum = 0.3
test_p = 0.1



nn = MultilayerPerceptron(learning_rate, momentum, act_fun="logistic", split_data=False, test_p=test_p, use_momentum=True)

nn.entry_layer(35)
nn.add_hidden_layer(3)
nn.add_hidden_layer(15)
nn.add_hidden_layer(13)
nn.add_hidden_layer(5)
nn.add_hidden_layer(2)
nn.add_hidden_layer(5)
nn.add_hidden_layer(13)
nn.add_hidden_layer(15)
nn.add_hidden_layer(3)
nn.output_layer(35)

error = 1
error = nn.train(_input, _expected, epochs=500)

#for i in range(0, len(_input)):
#    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))
