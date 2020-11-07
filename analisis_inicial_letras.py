import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

#Analizamos los datos de las letras para decidir una arquitectura de red
#En este caso, analizamos la primer componente principal y la matriz de covarianza para encontrar cuantas (y cuales) variables contienen mayor información, esto nos podría dar una idea de cuantos nodos deben tener las capas ocultas

#letras
f = open('letras.pickle', 'rb')

letras = pickle.load(f)

#inicializar
sc = StandardScaler()
pca = PCA()
#estandarizar los datos
scaled_letters = sc.fit_transform(letras)

#calcular pca
pca.fit(scaled_letters)

print('Componente Principal')
print(pca.components_[0])

#calcular la matriz de covarianza
cov_matrix = np.cov(scaled_letters.T)
#calcular los autovalores
v, w = np.linalg.eig(cov_matrix)
#la suma de los autovalores
sum_autovalores = sum(v)
#encontrar la proporción de información que da cada variable
print('Proporción de información de cada variable')
print(v / sum_autovalores)

#Resultados:
#-----------


# Componente Principal
# [ 0.10864372  0.23754153  0.23576804  0.2345148   0.07508291  0.28934155 -0.00432586 -0.23035321  0.03661111  0.23936349  0.21338037 -0.09337132
#  -0.08227105  0.07631265  0.23678025  0.01697909  0.09539734  0.16505308 0.16014383  0.07384988  0.15932537 -0.03889586 -0.09126593 -0.04751424
#   0.25328642  0.25028046 -0.00465419 -0.23001935  0.10512165  0.26486396 0.07442839  0.15394497  0.2338624   0.19158215  0.10212568]

# Proporción de información de cada variable
# [ 2.02429521e-01  1.30045911e-01  1.00886222e-01  7.87321043e-02 7.23312969e-02  5.35427955e-02  5.07683676e-02  4.95596496e-02
#   3.85995192e-02  3.58481435e-02  2.92879865e-02  2.42459060e-02 2.26850180e-02  1.92585459e-02  1.47922275e-02  1.41065292e-02
#   1.37317610e-02  1.11399443e-02  9.49252977e-03  6.69620434e-03 5.48809983e-03  5.04943618e-03  2.86284947e-03  2.51981739e-03
#   1.92150628e-03  1.34877234e-03  1.12237083e-03  6.81876746e-04 4.64535820e-04  2.24726743e-04  1.35826163e-04  1.18684414e-17
#   3.13611618e-19 -7.68701902e-18 -1.88945164e-17]

# Conclusiones
# Las primeras 3 variables son las más relevantes -> última capa de codificación de 3 nodos?
# Hay 4 "niveles" de importancia, -01, -02, 0-3/4, >-10 -> 4 capas ocultas?

# 1era capa: 5 nodos
# 2da capa: 13
# 3era capa: 15 nodos
# 4ta capa: 3 nodos

