import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

# input
letras = pickle.load(open('resources/letras.pickle', 'rb'))

_input = letras[1:3]
_expected = letras[1:3]
arq = [35, 15, 7, 4, 2, 4, 7, 15, 35]


nn = MultilayerPerceptron(latente_position=4, original_input=_input,optimizer='BFGS', eta=0.1, momentum=0.9, act_fun="tanh", split_data=False, test_p=0.15,
                          use_momentum=True, adaptative_eta=False)
nn.create_arq(arq)

denoising_letters = []
for i in _input:
    S = nn.modify_pattern(i, 7)
    denoising_letters.append(np.copy(S).tolist())

print("input---------> " + str(_input))
print()
print("modify--------> " + str(denoising_letters))

error = nn.train_minimizer_denoising(denoising_letters, _expected, epochs=500)

print(error)

# for i in range(0, 15):
#    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))

for i in range(0, 2):
    nn.predict(_input[i])
nn.print_activation_values()



