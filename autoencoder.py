import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
letras = pickle.load(open('resources/letras.pickle', 'rb'))

_input = letras[1:10]
_expected = letras[1:10]

nn = MultilayerPerceptron(eta=0.1, momentum=0.8, act_fun="tanh", split_data=False, test_p=0.15, use_momentum=True, adaptative_eta=False)

#arq = [35, 3, 15, 13, 5, 2, 5, 13, 15, 3, 35]

#arq = [35, 18, 13, 5, 2, 5, 13, 18, 35] #la mejor

#arq = [35, 15, 3, 2, 3, 15, 35]

#arq = [35, 17, 10, 5, 2, 2, 10, 17, 35] #arq de brian

#arq = [35, 2, 35]

nn.entry_layer(35)
nn.add_hidden_layer(18)
nn.add_hidden_layer(13)
nn.add_hidden_layer(5)
nn.add_hidden_layer(2)
nn.add_hidden_layer(5)
nn.add_hidden_layer(13)
nn.add_hidden_layer(18)
nn.entry_layer(35)

#nn.create_arq(arq)

#nn.print_layers()

error = nn.train(_input, _expected, epochs=2000)

#print(letras[1])
#print(nn.predict(letras[1]))

#for i in range(0, len(_input)):
#    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))

