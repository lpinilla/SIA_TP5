import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#cargar el dataset
letras = pickle.load(open('../resources/letras.pickle', 'rb'))
letras_etiquetas = ["space", "!", "\"", "#", "$", "%", "&", "\'", "(", ")", "*", "+", "´", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?"]

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
plot_vals = []

for i in letras:
    plot_vals.append(nn.view_latent_coding(i))

x_plot = [i[0] for i in plot_vals]
y_plot = [i[1] for i in plot_vals]
colors = cm.rainbow(np.linspace(0, 1, len(plot_vals)))

#graficar los puntos
fig, ax = plt.subplots()
ax.scatter(x_plot, y_plot, color=colors)

#agregar las etiquetas
for i in range(len(letras_etiquetas)):
    ax.annotate(letras_etiquetas[i], (x_plot[i], y_plot[i]))

ax.grid(True)

plt.show()

