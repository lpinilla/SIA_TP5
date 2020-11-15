import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron


def build_pca_weights(arr, n):
    aux = np.array(arr)
    bias = 1 - aux.sum()
    aux = np.append(aux, bias)
    _l = [aux for i in range(n)]
    return _l


# input
letras = pickle.load(open('resources/letras.pickle', 'rb'))

_input = letras[1:5]
_expected = letras[1:5]

arq = [35, 7, 2, 7, 35]

# nn = MultilayerPerceptron(optimizer='Powell',eta=0.001, momentum=0.9, act_fun="logistic", split_data=False, test_p=0.15, use_momentum=True, adaptative_eta=False)
nn = MultilayerPerceptron(latente_position=2, optimizer='BFGS', eta=0.1, momentum=0.9, act_fun="tanh", split_data=False, test_p=0.15,
                          use_momentum=True, adaptative_eta=False)

# arq = [35, 3, 15, 13, 5, 2, 5, 13, 15, 3, 35]

# arq = [35, 18, 13, 5, 2, 5, 13, 18, 35] #la mejor

# arq = [35, 15, 3, 2, 3, 15, 35]

# arq = [35, 17, 10, 5, 2, 5, 10, 17, 35] #arq de brian

# arq = [35, 2, 35]

# arq = [35, 17, 15, 3, 2, 3, 15, 17, 35]


# nn.entry_layer(35)
# nn.add_hidden_layer(17)
# nn.add_hidden_layer(15)
# nn.add_hidden_layer(3 )
# nn.add_hidden_layer(2 )
# nn.add_hidden_layer(3 )
# nn.add_hidden_layer(15)
# nn.add_hidden_layer(17)
# nn.output_layer(35)

nn.create_arq(arq)

# nn.print_layers()

error = nn.train_minimizer(_input, _expected, epochs=500)

print(error)


new_letter = nn.predict(letras[1], modif=True)
print("original ------> " + str(letras[1]))
print("\n")
print("Nueva ------> " + str(new_letter))




# for i in range(0, 10):
#     nn.predict(_input[i])
# nn.print_activation_values()