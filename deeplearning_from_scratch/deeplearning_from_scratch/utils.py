import numpy as np


def step_function(x: np.ndarray):
    return np.array(x > 0, dtype=np.int64)


def sigmoid(x: np.ndarray):
    return 1. / (1.+np.exp(-x))


def relu(x: np.ndarray):
    return np.maximum(0, x)


def identity_funtion(x):
    return x


def softmax(a: np.ndarray):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def sum_squares_error(y: np.ndarray, t: np.ndarray):
    return 0.5 * np.sum((np.array(y)-np.array(t))**2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.array(t)*np.log(np.array(y)+1e-7)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 zero 배열 생성

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        x[idx] = tmp + h
        fx1 = f(x)
        x[idx] = tmp - h
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * h)
        x[idx] = tmp
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad

    return x
