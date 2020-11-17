from MultilayerPerceptron import MultilayerPerceptron
import pickle
from sklearn.preprocessing import StandardScaler

#input
samples = pickle.load(open('sin_samples.pickle', 'rb'))

_input    = samples
_expected = samples

nn = MultilayerPerceptron(optimizer='BFGS',eta=0.1, momentum=0.8, act_fun="tanh", split_data=False, test_p=0.1, use_momentum=True, adaptative_eta=False)

arq = [50, 25, 5, 25, 50]

nn.entry_layer(50)
nn.add_hidden_layer(25, beta=0.9)
nn.add_hidden_layer(5, beta=0.5)
nn.add_hidden_layer(25, beta=0.9)
nn.output_layer(50)

error = nn.train(_input, _expected, epochs=1500)

#if error < 4:
#    f = open('trained_weights.pickle', 'wb')
#    pickle.dump(nn.flatten_weights(), f)


#print(samples[0])
#print(nn.predict(samples[0]))

