import matplotlib.pyplot as plt
import numpy as np
import pickle
import bezier
from MultilayerPerceptron import MultilayerPerceptron

def curves_between_points(p1, p2, n_curves, samples):
    nodes = [np.asfortranarray([[p1[0], (p2[0] + p1[0]) / 2, p2[0]], [p1[1], (p2[1]+p1[1])/2 + i/30, p2[1]]]) for i in range(1,n_curves+1)]
    curves = [bezier.Curve(i, degree=2) for i in nodes]
    #obtener los valores de x e y
    return [i.evaluate_multi(np.linspace(0, 1, samples)) for i in curves]

def graficar_curvas(points):
    #graficar las curvas
    fig, ax = plt.subplots()
    ax.grid(True)
    for i in points:
        ax.scatter(i[0,:], i[1,:])
    plt.show()

letras = pickle.load(open('../resources/letras.pickle', 'rb'))
letras_etiquetas = ["space", "!", "\"", "#", "$", "%", "&", "\'", "(", ")", "*", "+", "´", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?"]
trained_weights = pickle.load(open('trained_weights.pickle','rb'))

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

pesos_index = 4
ampersand_index = 6

p1 = nn.view_latent_coding(letras[pesos_index])
p2 = nn.view_latent_coding(letras[ampersand_index])

points = curves_between_points(p1, p2, 10, 20)
#graficar_curvas(curves_between_points(p1, p2, 1, 20))

results = []
#por cada curva
for i in range(len(points)):
    #por cada punto de la curva
    for j in range(len(points[i][0])):
        point = [points[i][0][j], points[i][1][j]]
        results.append(nn.generate(point))

def parse_output(arr):
    return [0 if abs(arr[i]) <= 0.1 else 1 for i in range(len(arr))]

def print_letter(letter):
    for i in range(7):
        print(letter[5 * i:5 * (i+1)])

#print_letter(letras[pesos_index])

print(results[0])
print("##############################")
print(parse_output(results[0]))
print("##############################")
print_letter(parse_output(results[0]))

