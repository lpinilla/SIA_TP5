import random
import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

layers = []
max_steps = 1000


def tanh(x, beta):
    return math.tanh(beta * x)


def tanh_deriv(x, beta):
    return beta * (1 - math.tanh(x) ** 2)


def logistic(x, beta):
    return 1 / (1 + math.exp(-2 * beta * x))


def logistic_deriv(x, beta):
    act = logistic(x, beta)
    return 2 * beta * act * (1 - act)


def arctan(x, beta):
    return math.atan(x)


def arctan_deriv(x, beta):
    y = arctan(x, beta)
    return 1 / (1 + (y ** 2))


def relu(x, beta):
    return max(0, x)


def relu_deriv(x, beta):
    return 1 if x > 0 else 0


functions = {
    "tanh": tanh,
    "logistic": logistic,
    "arctan": arctan,
    "relu": relu
}

deriv_functions = {
    "tanh": tanh_deriv,
    "logistic": logistic_deriv,
    "arctan": arctan_deriv,
    "relu": relu_deriv
}


class MultilayerPerceptron:

    def __init__(self, latente_position, optimizer=None, eta=None, momentum=None, act_fun=None, split_data=False, test_p=None,
                 use_momentum=False, adaptative_eta=False):
        global layers
        global max_steps
        self.latente_position = latente_position
        self.activation_values = []
        self.optimizer = optimizer
        self.eta = eta
        self.momentum = momentum
        self.act_fun = functions[act_fun]
        self.deriv_fun = deriv_functions[act_fun]
        self.test_p = test_p
        self.split_data = split_data
        self.use_momentum = use_momentum * 1
        self.use_adapt_eta = adaptative_eta * 1

    def print_layer(self, i):
        l = layers[i]
        print("i: " + str(i))
        print(" w: " + str(l["w"]))
        # print(" v: " + str(l["v"]))
        # print(" h: " + str(l["h"]))
        # print(" e: " + str(l["e"]))
        print("n of nodes: ", str(len(l["v"])))

    def print_layers(self):
        for i in range(len(layers)):
            self.print_layer(i)

    def create_layer(self, n_of_nodes, fn=None, weights=None, beta=0.5):
        if weights == None:
            size = n_of_nodes if not layers else layers[-1]["v"]
            if not layers:
                w = [np.random.uniform(-1, 1, n_of_nodes)]
            else:
                w = [np.random.uniform(-1, 1, len(layers[-1]["v"]) + 1) for i in range(n_of_nodes)]
        else:
            w = weights
        layer = {
            # pesos de cada nodo o entradas si es la capa inicial
            "w": w,
            # pesos anteriores, para usar momentum
            "prev_w": np.zeros(n_of_nodes) if not layers \
                else [np.zeros(len(layers[-1]["v"]) + 1) for i in range(n_of_nodes)],
            # valores de activación
            "v": np.ones(n_of_nodes),
            # valores de exitación
            "h": np.ones(n_of_nodes),
            # valores de error
            "e": np.ones(n_of_nodes),
            # función de activación
            "fn": functions[fn] if fn != None else self.act_fun,
            # derivada de la función de activación
            "deriv": deriv_functions[fn] if fn != None else self.deriv_fun,
            # el valor de beta para cada capa
            "beta": beta
        }
        return layer

    def create_arq(self, arq):
        self.entry_layer(arq[0])
        for i in range(1, len(arq) - 1):
            self.add_hidden_layer(arq[i])
        self.output_layer(arq[-1])

    def entry_layer(self, n_of_nodes, fn=None, beta=0.5):
        l = self.create_layer(n_of_nodes, fn=fn, beta=beta)
        layers.append(l)

    def add_hidden_layer(self, n_of_nodes, fn=None, weights=None, beta=0.5):
        l = self.create_layer(n_of_nodes, fn=fn, weights=weights, beta=beta)
        layers.append(l)

    def output_layer(self, n_of_nodes, fn=None, beta=0.5):
        l = self.create_layer(n_of_nodes, fn=fn, beta=beta)
        layers.append(l)

    def setup_entries(self, entries):
        entry = layers[0]
        entry["v"] = entries
        entry["w"] = [[None] * len(entries)]
        entry["prev_w"] = []
        entry["h"] = []
        entry["e"] = []
        entry["b"] = []

    # agregar un 1 al valor (para el sesgo) y devolver un numpy array
    def process_input(self, input_arr, expected_arr):
        # si se seteo, partir el dataset en input y test data
        # en base al % introducido
        if self.split_data:
            split_idx = int(len(input_arr) * (1 - self.test_p))
            return np.array(input_arr[:split_idx]), \
                   expected_arr[:split_idx], \
                   np.array(input_arr[split_idx:]), \
                   expected_arr[split_idx:]
        return np.array(input_arr), expected_arr, np.array(input_arr), expected_arr

    def predict(self, _input):
        return self.guess(np.array(_input), True)

    def guess(self, _input, predict):
        self.setup_entries(_input)
        self.feed_forward(predict)
        return layers[len(layers) - 1]["v"]

    # función para propagar secuencialmente los valores
    def feed_forward(self, predict):
        aux = []
        for i in range(1, len(layers)):
            l = layers[i]
            l_1 = layers[i - 1]
            inp = np.copy(l_1["v"])
            inp_bias = (np.append(inp, 1))
            h = [np.dot(l["w"][j], inp_bias) for j in range(len(l["h"]))]
            l["h"] = np.array(h)
            l["v"] = np.array([l["fn"](h[i], l["beta"]) for i in range(len(h))])
            if predict and i == self.latente_position:
                self.activation_values.append(l["v"])

    # función que propaga regresivamente el valor de error e de cada capa
    def back_propagation(self):
        for i in range(len(layers) - 1, 1, -1):
            l = layers[i]
            l_1 = layers[i - 1]
            ## calculamos los nuevos errores en base a los de la capa superior
            for j in range(len(l_1["e"])):
                aux = 0
                for k in range(len(l["e"])):
                    aux += l["w"][k][j] * l["e"][k]
                l_1["e"][j] = l_1["deriv"](l_1["h"][j], l_1["beta"]) * aux

    # función que actualiza los pesos de las capas calculando los deltas
    def update_weights(self):
        for i in range(len(layers) - 1, 0, -1):
            l = layers[i]
            l_1 = layers[i - 1]
            w = l["w"]
            delta_w = 0
            for e in range(len(l["e"])):
                for j in range(len(w[e]) - 1):
                    delta_w = self.eta * l["e"][e] * l_1["v"][j]
                    # actualizar los pesos
                    l["w"][e][j] += delta_w + self.use_momentum * self.momentum * l["prev_w"][e][j]
                    l["prev_w"][e][j] = delta_w

    # función para calcular el error de la muestra en la última capa
    def calculate_last_layer_error(self, expected):
        l = layers[-1]
        aux = expected - l["v"]
        l["e"] = np.array([l["deriv"](l["h"][i], l["beta"]) * aux[i] for i in range(len(l["e"]))])

    def calculate_error(self, test_data, test_exp):
        guesses = [self.guess(i, False) for i in test_data]
        return np.sum(
            [(np.subtract(test_exp[i], guesses[i]) ** 2).sum() \
             for i in range(len(test_exp))]
        ) / len(test_data)

    def train(self, inputs, expected, epochs):
        inp_data, inp_exp, test_data, test_exp = self.process_input(inputs, expected)
        error = 1
        error_min = 1
        err_history = []
        idxs = [i for i in range(len(inp_data))]
        for i in range(epochs):
            order = random.sample(idxs, len(idxs))
            for j in range(len(order)):
                # agarrar un índice random para agarrar una muestra
                idx = order[j]
                _in = inp_data[idx]
                _ex = inp_exp[idx]
                # hacer el setup de la entrada
                self.setup_entries(_in)
                # hacer feed forward
                self.feed_forward(False)
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
                # actualizar el eta si se configuró así
                self.eta += self.use_adapt_eta * self.adapt_eta(len(idxs) * i + j, err_history, error)
            print(error)
        return error

    ##################### OPTIMIZACIONES ##########################

    def cost(self, flat_weights, test_data, test_exp):
        self.rebuild_net(flat_weights)
        error = self.calculate_error(test_data, test_exp)
        # print(error)
        return error

    def train_minimizer(self, inputs, expected, epochs):
        inp_data, inp_exp, test_data, test_exp = self.process_input(inputs, expected)
        flattened_weights = self.flatten_weights()
        res = minimize(fun=self.cost, x0=flattened_weights, args=(test_data, test_data), method=self.optimizer)
        error = res.fun
        self.rebuild_net(flattened_weights)
        return error

    def flatten_weights(self):
        all_weights = self.flatten_matrix(layers[1]["w"])
        for i in range(2, len(layers)):
            all_weights = np.append(all_weights, self.flatten_matrix(layers[i]["w"]))
        return all_weights

    def flatten_matrix(self, matrix):
        arr = np.array(matrix[0])
        for i in range(1, len(matrix)):
            arr = np.append(arr, matrix[i])
        return arr

    def rebuild_net(self, flat_weights):
        offset = 0
        for i in range(1, len(layers)):
            l = layers[i]
            l_1 = layers[i - 1]
            n = len(l["v"])
            n_1 = len(l_1["v"])
            aux = n * (n_1 + 1)
            self.rebuild_weight_matrix(i, n, n_1 + 1, flat_weights[offset: offset + aux])
            offset += aux

    def rebuild_weight_matrix(self, layer_n, rows, columns, sub_flat_weights):
        l = layers[layer_n]
        w = []
        r = 0
        for i in range(rows):
            w.append(sub_flat_weights[columns * i: columns * (i + 1)])
        l["w"] = w

    def clear_weights(self):
        for i in range(len(layers)):
            layers[i]["w"] = []

    def unflatten_weights(self, flat_weights):
        aux = np.array([])
        index = 0
        index_base = 0
        for elem in flat_weights:
            if index % (len(flat_weights) / len(layers[index_base]["w"])) == 0:
                layers[index_base]["w"] = aux
                index_base += 1
                aux = np.array([])
            else:
                np.append(aux, elem)
            index += 1

    def adapt_eta(self, i, err_history, error):
        if i < 2:
            err_history.append(error)
            return 0
        bigger = all(error >= j for j in err_history)
        smaller = all(error < j for j in err_history)
        # agregar el error a la lista
        err_history.append(error)
        # "desplazar" la ventana de errores, pisar el anterior
        err_history = err_history[1:]
        if bigger:
            return -0.1 * self.eta
        if smaller:
            return 0.1
        return 0

    def print_activation_values(self):

        x = []
        y = []

        print(self.activation_values)
        for i in self.activation_values:
                x.append(i[0])
                y.append(i[1])

        print(x)
        print(y)
        plt.scatter(x, y)

        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "M", "N", "O", "P", "Q", "R", "S", "T", "U", 
                  "V", "W", "X", "Y", "Z"] 
        index = 0
        for x, y in zip(x, y):
            label = labels[index]

            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            index += 1
        plt.show()
