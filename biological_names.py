import numpy as np
import random
import re

# Data Preprocessing
data = open('species.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
chars = sorted(chars)
char_to_ind = { ch:i for i,ch in enumerate(chars) }
ind_to_char = { i:ch for i,ch in enumerate(chars) }

# Auxillary functions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clip(gradients, maxValue): # for the exploding gradients problem
    dWf, dWi, dWc, dWo, dWy = gradients['dWf'], gradients['dWi'], gradients['dWc'], gradients['dWo'], gradients['dWy']
    dbf, dbi, dbc, dbo, dby = gradients['dbf'], gradients['dbi'], gradients['dbc'], gradients['dbo'], gradients['dby']

    for gradient in [dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby]:
        np.clip(gradient, a_min = -maxValue, a_max = maxValue, out = gradient)

    gradients = {"dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo, "dWy": dWy,"dby": dby}

    return gradients

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length

def print_sample(sample_ind, ind_to_char):
    txt = ''.join(ind_to_char[ind] for ind in sample_ind)
    print(txt)

# Initialize Parameters
def initialize_parameters(n_x, n_a, n_y):

    parameters = dict()

    parameters['Wf'] = np.random.randn(n_a, n_a + n_x) * np.sqrt(6 / (2 * n_a + n_x))
    parameters['Wi'] = np.random.randn(n_a, n_a + n_x) * np.sqrt(6 / (2 * n_a + n_x))
    parameters['Wc'] = np.random.randn(n_a, n_a + n_x) * np.sqrt(6 / (2 * n_a + n_x))
    parameters['Wo'] = np.random.randn(n_a, n_a + n_x) * np.sqrt(6 / (2 * n_a + n_x))
    parameters['Wy'] = np.random.randn(n_y, n_a) * np.sqrt(6 / (n_y + n_a))
    parameters['bf'] = np.zeros((n_a, 1))
    parameters['bi'] = np.zeros((n_a, 1))
    parameters['bc'] = np.zeros((n_a, 1))
    parameters['bo'] = np.zeros((n_a, 1))
    parameters['by'] = np.zeros((n_y, 1))

    return parameters

# LSTM blocks
def lstm_cell_forward(xt, a_prev, c_prev, parameters):

    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x = xt.shape[0]
    n_y, n_a = Wy.shape

    concat = np.concatenate((a_prev, xt), axis = 0)

    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = (ft * c_prev) + (it * cct)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    yt_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

def lstm_forward(X, Y, a0, parameters, vocab_size):
    caches = []

    Wy = parameters['Wy']
    T_x = len(X)
    n_x = vocab_size
    n_y, n_a = Wy.shape

    x = np.zeros((n_x, T_x))
    a = np.zeros((n_a, T_x))
    c = np.zeros((n_a, T_x))
    y_hat = np.zeros((n_y, T_x))

    loss = 0
    a_next = a0
    c_next = np.zeros((n_a, 1))

    for t in range(T_x):
        xt = x[:, t].reshape(-1, 1)
        yt = np.zeros((n_y, 1))
        xt[X[t], 0] = 1
        yt[Y[t], 0] = 1
        a_next, c_next, yt_hat, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:,t] = a_next.reshape(-1,)
        c[:,t]  = c_next.reshape(-1,)
        y_hat[:,t] = yt_hat.reshape(-1,)
        loss += -np.sum(yt * np.log(yt_hat))
        caches.append(cache)

    caches = (caches, x)

    return a, y_hat, c, caches, loss

def lstm_cell_backward(da_next, dc_next, cache, dZt):

    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    n_x, _ = xt.shape
    n_a, _ = a_next.shape

    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next) * (1 - cct ** 2)
    dit = (dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev * da_next) * ft * (1 - ft)

    dWf = np.dot(dft, np.concatenate((a_prev, xt), axis = 0).T)
    dWi = np.dot(dit, np.concatenate((a_prev, xt), axis = 0).T)
    dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis = 0).T)
    dWo = np.dot(dot, np.concatenate((a_prev, xt), axis = 0).T)
    dWy = np.dot(dZt, a_next.T)
    dbf = dft
    dbi = dit
    dbc = dcct
    dbo = dot
    dby = dZt

    da_prev = np.dot(parameters['Wf'][:, : n_a].T, dft) + np.dot(parameters['Wi'][:, : n_a].T, dit) + np.dot(parameters['Wc'][:, : n_a].T, dcct) + np.dot(parameters['Wo'][:, : n_a].T, dot)
    dc_prev = dc_next * ft + dot * (1 - np.tanh(c_next) ** 2) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a : ].T, dft) + np.dot(parameters['Wi'][:, n_a : ].T, dit) + np.dot(parameters['Wc'][:, n_a : ].T, dcct) + np.dot(parameters['Wo'][:, n_a : ].T, dot)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo, "dWy": dWy,"dby": dby}

    return gradients

def lstm_backward(y_hat, Y, a, caches):

    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    n_a, T_x = a.shape
    n_y = y_hat.shape[0]
    n_x, _ = x1.shape
    dZ = np.zeros(y_hat.shape)
    da = np.zeros(a.shape)
    Wy = parameters['Wy']

    dx = np.zeros((n_x, T_x))
    da0 = np.zeros((n_a, 1))
    da_prevt = np.zeros((n_a, 1))
    dc_prevt = np.zeros((n_a, 1))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dWy = np.zeros((n_y, n_a))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    dby = np.zeros((n_y, 1))

    for t in reversed(range(T_x)):

        dZ[:, t] = np.copy(y_hat[:, t])
        dZ[Y[t], t] -= 1
        da[:, t] = np.dot(Wy.T, dZ[:, t].reshape(-1, 1)).reshape(-1,)

        gradients = lstm_cell_backward(da[:, t].reshape(-1, 1) + da_prevt, dc_prevt, caches[t], dZ[:, t].reshape(-1, 1))

        da_prevt = gradients["da_prev"]
        dc_prevt = gradients["dc_prev"]
        dx[:,t] = gradients["dxt"].reshape(-1,)
        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dWy += gradients["dWy"]
        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]
        dby += gradients["dby"]

    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo, "dWy": dWy,"dby": dby}

    return gradients

# Update Parameters
def update_parameters(parameters, gradients, lr):

    parameters['Wf'] += -lr * gradients['dWf']
    parameters['Wi'] += -lr * gradients['dWi']
    parameters['Wc'] += -lr * gradients['dWc']
    parameters['Wo'] += -lr * gradients['dWo']
    parameters['Wy'] += -lr * gradients['dWy']
    parameters['bf'] += -lr * gradients['dbf']
    parameters['bi'] += -lr * gradients['dbi']
    parameters['bc'] += -lr * gradients['dbc']
    parameters['bo'] += -lr * gradients['dbo']
    parameters['by'] += -lr * gradients['dby']

    return parameters

# Sampling
def sample(parameters, char_to_ind):

    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    vocab_size = by.shape[0]
    n_a = Wf.shape[0]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    c_prev = np.zeros((n_a, 1))

    indices = []

    ind = -1

    counter = 0
    newline_character = char_to_ind['\n']

    while (ind != newline_character and counter != 25):

        concat = np.concatenate((a_prev, x), axis = 0)

        ft = sigmoid(np.dot(Wf, concat) + bf)
        it = sigmoid(np.dot(Wi, concat) + bi)
        cct = np.tanh(np.dot(Wc, concat) + bc)
        c_next = (ft * c_prev) + (it * cct)
        ot = sigmoid(np.dot(Wo, concat) + bo)
        a = ot * np.tanh(c_next)

        y = softmax(np.dot(Wy, a) + by)

        ind = np.random.choice(range(vocab_size), p = np.ravel(y))

        indices.append(ind)

        x = np.zeros((vocab_size, 1))
        x[ind] = 1

        a_prev = a

        counter +=1

    if (counter == 50):
        indices.append(char_to_ind['\n'])

    return indices

# Gradient Descent
def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):

    a, y_hat, c, caches, loss = lstm_forward(X, Y, a_prev, parameters, vocab_size)
    gradients = lstm_backward(y_hat, Y, a, caches)
    gradients = clip(gradients, 5)

    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[:, len(X)-1].reshape(-1, 1)

# Model
def model(data, ind_to_char, char_to_ind, vocab_size, num_iterations = 60000, n_a = 50, bio_names = 7):

    n_x, n_y = vocab_size, vocab_size

    parameters = initialize_parameters(n_x, n_a, n_y)

    loss = get_initial_loss(vocab_size, bio_names)

    examples = list()
    with open("species.txt") as f:
        for line in f:
            line = line.rstrip()
            x = re.findall('[0-9]+: N=(\S+ \S+)', line)
            if len(x) > 0:
                example = x[0].lower()
                examples.append(example)

    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))

    for j in range(num_iterations):

        ind = j % len(examples)

        single_example = examples[ind]
        single_example_chars = [c for c in single_example]
        single_example_ind = [char_to_ind[c] for c in single_example_chars]
        X = [None] + single_example_ind

        ind_newline = char_to_ind['\n']
        Y = single_example_ind + [ind_newline]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)

        loss = smooth(loss, curr_loss)

        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            for name in range(bio_names):

                sampled_indices = sample(parameters, char_to_ind)
                print_sample(sampled_indices, ind_to_char)

            print('\n')

    return parameters

parameters = model(data, ind_to_char, char_to_ind, vocab_size)
