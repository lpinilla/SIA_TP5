import numpy as np
import pickle
from MultilayerPerceptron import MultilayerPerceptron

samples = pickle.load(open('sin_samples.pickle', 'rb'))

trained_weights = pickle.load(open('trained_weights.pickle', 'rb'))

nn = MultilayerPerceptron(optimizer='L-BFGS-B',eta=0.1, momentum=0.8, act_fun="tanh", split_data=False, test_p=0.1, use_momentum=True, adaptative_eta=False)

#re-armar la arquitectura entrenada
nn.entry_layer(50)
nn.add_hidden_layer(25, beta=0.9)
nn.add_hidden_layer(5, beta=0.5)
nn.add_hidden_layer(25, beta=0.9)
nn.output_layer(50)

#cargar los pesos
nn.rebuild_net(trained_weights)

#tomar muestra del código latente de algún dato
#coding_0 = nn.view_latent_coding(samples[0])
#print(coding_0)


#codificación de sample 0
#[ 0.99264605 -0.88249352  0.96457358  0.9994266  -0.92983281]


#generar algo a partir de la capa latente
latent_try = [1, -0.7, 0.95, 1, -0.9]

generated = nn.generate(latent_try)

#for i in generated:
#    print(i)

