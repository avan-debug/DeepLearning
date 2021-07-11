import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_with_zero(dim):
    w = np.zeros(dim, 1)
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    # m : num_px * num_px * 3, 1
    # b:
    # X : num_px * num_px * 3, train_set_num
    # Y : 1, train_set_num

    m = Y.shape[1]
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1 / m) * np.dot(A - Y, X.T)
    db = (1 / m) * np.sum(A - Y)





