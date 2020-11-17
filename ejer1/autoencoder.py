import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

## programa que entrena al autoencoder para reconocer los patrones de letras

# input
letras = pickle.load(open('../resources/letras.pickle', 'rb'))

_input = letras
_expected = letras

nn = MultilayerPerceptron(latente_position=2, optimizer='BFGS', eta=0.1, momentum=0.9, act_fun="tanh", split_data=False, test_p=0.15, use_momentum=False, adaptative_eta=False)

#arq = [35, 17, 15, 3, 2, 3, 15, 17, 35]
#arq = [35, 7, 2, 7, 35]
#nn.create_arq(arq)

#1.4547944007023588
nn.entry_layer(35)
nn.add_hidden_layer(17, beta=0.9)
nn.add_hidden_layer(15, beta=0.7)
nn.add_hidden_layer(3 , beta=0.6)
nn.add_hidden_layer(2)
nn.add_hidden_layer(3 , beta=0.6)
nn.add_hidden_layer(15, beta=0.7)
nn.add_hidden_layer(17, beta=0.9)
nn.output_layer(35)


error = nn.train(_input, _expected, epochs=1500)
#error = nn.train_minimizer(_input, _expected)

if error < 2.3:
    f = open('trained_weights.pickle', 'wb')
    pickle.dump(nn.flatten_weights(), f)

#for i in range(0, 2):
#    nn.predict(_input[i])
#nn.print_activation_values()
