import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
letras = pickle.load(open('resources/letras.pickle', 'rb'))

_input = letras[2:3]
_expected = letras[2:3]

nn = MultilayerPerceptron(eta=0.1, momentum=0.8, act_fun="logistic", split_data=False, test_p=0.1, use_momentum=False, adaptative_eta=False)

#arq = [35, 3, 15, 13, 5, 2, 5, 13, 15, 3, 35]

#arq = [35, 18, 13, 5, 2, 5, 13, 18, 35]

#arq = [35, 18, 13, 5, 2, 5, 13, 18, 35]

#arq = [35, 15, 3, 2, 3, 15, 35]

#arq = [35, 17, 10, 5, 2, 2, 10, 17, 35]

arq = [35, 35, 35]


nn.create_arq(arq)

nn.print_layers()

error = nn.train(_input, _expected, epochs=1000)

#print(letras[1])
#print(nn.predict(letras[1]))

#for i in range(0, len(_input)):
#    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))

