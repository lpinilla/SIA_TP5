import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron
import matplotlib.pyplot as plt

#cargar el dataset
letras = pickle.load(open('../resources/letras.pickle', 'rb'))
#cargar los pesos ya entrenados
trained_weights = pickle.load(open('trained_weights.pickle','rb'))

_input = letras
_expected = letras

nn = MultilayerPerceptron(latente_position=2, optimizer='BFGS', eta=0.1, momentum=0.9, act_fun="tanh", split_data=False, test_p=0.15, use_momentum=False, adaptative_eta=False)

#re-armar estructura de la red
nn.entry_layer(35)
nn.add_hidden_layer(17, beta=0.9)
nn.add_hidden_layer(15, beta=0.7)
nn.add_hidden_layer(3 , beta=0.6)
nn.add_hidden_layer(2)
nn.add_hidden_layer(3 , beta=0.6)
nn.add_hidden_layer(15, beta=0.7)
nn.add_hidden_layer(17, beta=0.9)
nn.output_layer(35)

#cargar los pesos entrenados en la red
nn.rebuild_net(trained_weights)

#red lista para usar
