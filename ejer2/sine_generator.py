import numpy as np
import math
import pickle

#links útils

# https://towardsdatascience.com/can-machine-learn-the-concept-of-sine-4047dced3f11

# sen(ßx)
max_interval = 2 * math.pi

#betas = np.array([i/10 for i in range(5, 105, 5)])

betas = np.array([i for i in range(1, 10)])

#cantidad de muestras a tomar de la función
n_of_samples = 50

#valores de x
x = np.linspace(0, max_interval, n_of_samples)

#guardar las muestras
samples = []

for i in betas:
    _x = i * x
    samples.append(np.array([math.sin(j) for j in _x]))

f = open('sin_samples.pickle', 'wb')
pickle.dump(samples, f)
