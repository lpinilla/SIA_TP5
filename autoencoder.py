import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
letras = pickle.load(open('resources/letras.pickle', 'rb'))

_input = letras
_expected = letras

nn = MultilayerPerceptron(eta=0.01, momentum=0.8, act_fun="logistic", split_data=True, test_p=0.1, use_momentum=True, adaptative_eta=False)

arq = [35, 3, 15, 13, 5, 2, 5, 13, 15, 3, 35]

nn.create_arq(arq)

error = nn.train(_input, _expected, epochs=500)

#for i in range(0, len(_input)):
#    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))
