import numpy as np


def step_function(x: np.ndarray):
    return np.array(x > 0, dtype=np.int64)


def sigmoid(x: np.ndarray):
    return 1. / (1.+np.exp(-x))


def relu(x: np.ndarray):
    return np.maximum(0, x)


def identity_funtion(x):
    return x


def softmax(x: np.ndarray):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squares_error(y, t):
    return 0.5 * np.sum((np.array(y)-np.array(t))**2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


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
