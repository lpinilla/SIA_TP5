import random
import math
import numpy as np

layers = []
max_steps = 1000

beta = 0.5

def tanh(x):
    return math.tanh(beta * x)

def tanh_deriv(x):
    return beta * (1 - math.tanh(x)**2)

def logistic(x):
    return 1 / (1 + math.exp(-2 * beta * x))

def logistic_deriv(x):
    act = logistic(x)
    return 2 * beta * act * (1 - act)

def arctan(x):
    return math.atan(x)

def arctan_deriv(x):
    y = arctan(x)
    return 1 / (1 + (y ** 2))


functions = {
    "tanh": tanh,
    "tanh_d": tanh_deriv,
    "logistic": logistic,
    "logistic_d": logistic_deriv,
    "arctan": arctan,
    "arctan_d": arctan_deriv
}

deriv_functions = {
    "tanh": tanh_deriv,
    "logistic": logistic_deriv,
    "arctan": arctan_deriv
}


class MultilayerPerceptron:

    def __init__(self, eta=None, momentum=None, act_fun=None, split_data=True, test_p=None, use_momentum=False):
        global layers
        global max_steps
        self.eta = eta
        self.momentum = momentum
        self.act_fun = functions[act_fun]
        self.deriv_fun = deriv_functions[act_fun]
        self.test_p = test_p
        self.split_data = split_data
        self.use_momentum = use_momentum * 1

    def print_layer(self, i):
        l = layers[i]
        print("i: " + str(i))
        print(" w: " + str(l["w"]))
        print(" v: " + str(l["v"]))
        print(" h: " + str(l["h"]))
        print(" b: " + str(l["b"]))
        print(" e: " + str(l["e"]))

    def print_layers(self):
        for i in range(0, len(layers)):
            self.print_layer(i)

    def create_layer(self, n_of_nodes, fn=None):
        layer = {
            # pesos de cada nodo o entradas si es la capa inicial
            "w": np.random.randn(n_of_nodes) if not layers \
                else [np.random.randn(len(layers[-1]["v"]) + 1) for i in range(n_of_nodes)],
            # pesos anteriores, para usar momentum
            "prev_w": np.zeros(n_of_nodes) if not layers \
                else [np.zeros(len(layers[-1]["v"]) + 1) for i in range(n_of_nodes)],
            # valores de activación
            "v": np.ones(n_of_nodes),
            # valores de exitación
            "h": np.ones(n_of_nodes),
            # valores de error
            "e": np.ones(n_of_nodes),
            # valores de bias
            "b": np.ones(n_of_nodes),
            # función de activación
            "fn": functions[fn] if fn != None else self.act_fun,
            # derivada de la función de activación
            "deriv": deriv_functions[fn] if fn != None else self.deriv_fun
        }
        return layer

    def entry_layer(self, n_of_nodes, fn=None, deriv=None):
        l = self.create_layer(n_of_nodes, fn=fn)
        layers.append(l)

    def add_hidden_layer(self, n_of_nodes, fn=None, deriv=None):
        l = self.create_layer(n_of_nodes, fn=fn)
        layers.append(l)

    def output_layer(self, n_of_nodes, fn=None, deriv=None):
        l = self.create_layer(n_of_nodes, fn=fn)
        layers.append(l)

    def setup_entries(self, entries):
        entry = layers[0]
        entry["v"] = entries
        entry["w"] = []
        entry["prev_w"] = []
        entry["h"] = []
        entry["b"] = []
        entry["e"] = []

    # agregar un 1 al valor (para el sesgo) y devolver un numpy array
    def process_input(self, input_arr, expected_arr):
        inputs = []
        for inp in input_arr:
            # sumar el 1 como input para el sesgo
            input_bias = np.copy(inp)
            # agregar el 1 para el sesgo
            # input_bias = np.append(input_bias, 1)
            # convertir en array de numpy
            input_data = np.array(input_bias)
            inputs.append(input_data)
        # si se seteo, partir el dataset en input y test data
        # en base al % introducido
        if self.split_data:
            split_idx = int(len(input_arr) * (1 - self.test_p))
            return np.array(inputs[:split_idx]), \
                   expected_arr[:split_idx], \
                   np.array(inputs[split_idx:]), \
                   expected_arr[split_idx:]
        return np.array(inputs), expected_arr, np.array(inputs), expected_arr

    def predict(self, _input):
        return self.guess(np.array(_input))

    def guess(self, _input):
        self.setup_entries(_input)
        self.feed_forward()
        return layers[len(layers) - 1]["v"]

    # función para propagar secuencialmente los valores
    def feed_forward(self):
        for i in range(1, len(layers)):
            l = layers[i]
            l_1 = layers[i - 1]

            inp = np.copy(l_1["v"])
            inp_bias = (np.append(inp, 1))
            h = [np.dot(l["w"][j], inp_bias) for j in range(len(l["h"]))]
            l["h"] = np.array(h)
            l["v"] = np.array([l["fn"](h[i]) for i in range(len(h))])
            # l["v"] = np.array( [l["fn"](h[i]) + l["b"][i] for i in range(len(l["h"]))])

    # función que propaga regresivamente el valor de error
    def back_propagation(self):
        for i in range(len(layers) - 1, 1, -1):
            l = layers[i]
            l_1 = layers[i - 1]
            errors = []
            # calculamos los nuevos errores en base a los de la capa superior
            for j in range(len(l_1["e"])):
                # agarrar todas las conexiones del nodo j con la capa superior
                w_1 = np.array([l["w"][k][j] for k in range(len(l["w"]))])
                # agregar a la lista de errores el nuevo error del nodo j de
                # la capa i - 1
                errors.append(l_1["deriv"](l_1["h"][j]) * np.dot(w_1, l["e"]))
            l_1["e"] = np.array(errors, dtype='double')

    def update_weights(self):
        for i in range(len(layers) - 1, 0, -1):
            l = layers[i]
            l_1 = layers[i - 1]
            w = l["w"]
            delta_w = 0
            for e in range(0, len(l["e"])):
                for j in range(0, len(w[e]) - 1):
                    delta_w = self.eta * l["e"][e] * l_1["v"][j]
                    # actualizar los pesos
                    l["w"][e][j] += delta_w   + self.use_momentum * self.momentum * l["prev_w"][e][j]
                    l["prev_w"][e][j] = delta_w
                    # actualizar el bias
                    # l["b"][e] += self.eta * l["e"][e]

    # función para calcular el error de la muestra en la última capa
    def calculate_last_layer_error(self, expected):
        l = layers[-1]
        aux = np.sum(expected - l["v"])
        l["e"] = np.array([l["deriv"](l["h"][i]) * aux for i in range(len(l["e"]))])

    def calculate_error(self, test_data, test_exp):
        guesses = [self.guess(i) for i in test_data]
        return np.sum(
            [(np.subtract(test_exp[i], guesses[i]) ** 2).sum() \
             for i in range(0, len(test_exp))]
        ) / len(test_data)

    def adapt_eta(self, i, err_history, error):
        if i < 2:
            err_history.append(error)
            return 0
        bigger = all(error >= i for i in err_history)
        smaller = all(error < i for i in err_history)
        err_history.append(error)
        err_history = err_history[1:]
        if bigger:
            return - 0.5 * self.eta
        if smaller:
            return 0.1
        return 0

    def train(self, inputs, expected, epochs):
        inp_data, inp_exp, test_data, test_exp = \
            self.process_input(inputs, expected)
        error = 1
        curr_step = 0
        error_min = 1
        err_history = []
        idxs = [i for i in range(0, len(inp_data))]
        for i in range(0, epochs):
            order = random.sample(idxs, len(idxs))
            for j in range(0, len(order)):
                # agarrar un índice random para agarrar una muestra
                idx = order[j]
                _in = inp_data[idx]
                _ex = inp_exp[idx]
                # print("entrada " + str(_in))
                self.setup_entries(_in)
                # hacer feed forward
                self.feed_forward()
                # calcular el delta error de la última capa
                self.calculate_last_layer_error(_ex)
                # retropropagar el error hacia las demás capas
                self.back_propagation()
                # ajustar los pesos
                self.update_weights()
                # calcular el error
                error = self.calculate_error(test_data, test_exp)
                if error < error_min:
                    error_min = error
                #self.eta += self.adapt_eta(len(idxs) * i + j, err_history, error)
                # curr_step += 1
            print(error)
        return error

