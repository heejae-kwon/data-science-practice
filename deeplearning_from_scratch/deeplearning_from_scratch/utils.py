import numpy as np


def step_function(x: np.ndarray):
    return np.array(x > 0, dtype=np.int64)


def sigmoid(x: np.ndarray):
    return 1. / (1.+np.exp(-x))


def relu(x: np.ndarray):
    return np.maximum(0, x)


def identity_funtion(x):
    return x


def sofmax(a: np.ndarray):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def sum_squares_error(y: np.ndarray, t: np.ndarray):
    return 0.5 * np.sum((np.array(y)-np.array(t))**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(np.array(t)*np.log(np.array(y)+delta))
