import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron
from tqdm import tqdm




def parse_output(arr):
    return [0 if abs(arr[i]) <= 0.1 else 1 for i in range(len(arr))]

def print_letter(letter):
    for i in range(7):
        fila = ""
        for j in range(5):
            if letter[i * 5 + j] == 1:
                fila = fila + " *"
            else:
                fila += "  "
        print(fila)


# input
letras = pickle.load(open('../resources/letras.pickle', 'rb'))

_input = letras[1]
_expected = letras[1]



nn = MultilayerPerceptron(latente_position=4, original_input=_input,optimizer='BFGS', eta=0.1, momentum=0.9, act_fun="tanh", split_data=False, test_p=0.15,
                          use_momentum=True, adaptative_eta=False)
nn.entry_layer(35)
nn.add_hidden_layer(17, beta=0.9)
nn.add_hidden_layer(15, beta=0.7)
nn.add_hidden_layer(3 , beta=0.6)
nn.add_hidden_layer(2)
nn.add_hidden_layer(3 , beta=0.6)
nn.add_hidden_layer(15, beta=0.7)
nn.add_hidden_layer(17, beta=0.9)
nn.output_layer(35)

denoising_letters = []
S = nn.modify_pattern(_input, 7)
denoising_letters.append(np.copy(S))
# for i in tqdm(range(0, 30)):
#     S = nn.modify_pattern(_input, 2)
#     aux = np.array(np.copy(S)).flatten().tolist()
#     error = nn.train_denoising(aux, _expected, epochs=10)


print("input")
print(_input)
print_letter(parse_output(_input))
print()
# for elem in denoising_letters:
aux = np.array(denoising_letters).flatten().tolist()
print("modify")
print(aux)
print_letter(parse_output(aux))

trained_weights = pickle.load(open('trained_weights.pickle','rb'))
nn.rebuild_net(trained_weights)



# error = nn.train_denoising(denoising_letters, _expected, epochs=500)
# error = nn.train_minimizer_denoising(aux, _expected)


res = nn.predict(_input)
print("result")
print(res)

print_letter(parse_output(res))
print("##############################")

