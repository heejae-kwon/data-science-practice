import numpy as np
from torchvision import datasets, transforms
import torch


def simple_logistic_regression():
    x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10, 1)
    t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10, 1)

    W = np.random.rand(1, 1)  # (input, 1)
    b = np.random.rand(1)
    print(f'W={W} W.shape={W.shape} b={b} b.shape={b.shape}')

    def sigmoid(x):
        return 1 / (1+np.exp(-x))

    def loss_func(x: np.ndarray, t: np.ndarray):
        delta = 1e-7

        z = np.dot(x, W)+b
        y = sigmoid(z)
        # cross-entropy
        return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))

    def error_val(x: np.ndarray, t: np.ndarray):
        delta = 1e-7

        z = np.dot(x, W)+b
        y = sigmoid(z)
        # cross-entropy
        return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))

    def predict(x):
        z = np.dot(x, W)+b
        y = sigmoid(z)

        if y > 0.5:
            result = 1
        else:
            result = 0

        return y, result

    learning_rate = 1e-2
    def f(x): return loss_func(x_data, t_data)
    print(
        f'Initial error value={error_val(x_data,t_data)} Initial W={W}, b={b}')

    for step in range(10001):
        W -= learning_rate * numerical_derivative(f, W)
        b -= learning_rate*numerical_derivative(f, b)
        if step % 400 == 0:
            print(
                f'step={step}, error_value={error_val(x_data,t_data)}, W={W}, b={b}')

    (real_val, logical_val) = predict(17)
    print(real_val, logical_val)


def simple_regression():
    x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
    t_data = np.array([2, 3, 4, 5, 6]).reshape(5, 1)

    W = np.random.rand(1, 1)  # (input, 1)
    b = np.random.rand(1)
    print(f'W={W} W.shape={W.shape} b={b} b.shape={b.shape}')

    def loss_func(x: np.ndarray, t: np.ndarray):
        y = np.dot(x, W) + b
        return (np.sum((t-y)**2)/len(x))

    def error_val(x: np.ndarray, t: np.ndarray):
        y = np.dot(x, W) + b
        return (np.sum((t-y)**2)/len(x))

    def predict(x: np.ndarray):
        y = np.dot(x, W) + b
        return y

    learning_rate = 1e-2
    def f(x): return loss_func(x_data, t_data)
    print(
        f'Initial error value={error_val(x_data,t_data)} Initial W={W}, b={b}')

    for step in range(8001):
        W -= learning_rate * numerical_derivative(f, W)
        b -= learning_rate*numerical_derivative(f, b)
        if step % 400 == 0:
            print(
                f'step={step}, error_value={error_val(x_data,t_data)}, W={W}, b={b}')

    print(predict(43))

    return


def numerical_derivative(f, x: np.ndarray):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)  # f(x+delta_x)
        x[idx] = tmp_val - delta_x
        fx2 = f(x)  # f(x-delta_x)
        grad[idx] = (fx1-fx2)/(2*delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad


class LogicGate:
    def __init__(self, gate_name, xdata, tdata) -> None:
        self.name = gate_name

        self._xdata = np.array(xdata).reshape(4, 2)
        self._tdata = np.array(tdata).reshape(4, 1)

        self._W2 = np.random.rand(2, 6)
        self._b2 = np.random.rand(6)

        self._W3 = np.random.rand(6, 1)
        self._b3 = np.random.rand(1)

        self._learning_rate = 1e-2

        print(self.name + " object is created")

    def _sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def feed_forward(self):
        delta = 1e-7

        z2 = np.dot(self._xdata, self._W2) + self._b2
        a2 = self._sigmoid(z2)

        z3 = np.dot(a2, self._w3) + self._b3
        y = a3 = self._sigmoid(z3)

        # cross_entropy
        return -np.sum(self._tdata + np.log(y+delta) + (1-self._tdata) + np.log((1-y)+delta))

    def loss_val(self):
        delta = 1e-7

        z2 = np.dot(self._xdata, self._W2) + self._b2
        a2 = self._sigmoid(z2)

        z3 = np.dot(a2, self._w3) + self._b3
        y = a3 = self._sigmoid(z3)

        # cross_entropy
        return -np.sum(self._tdata + np.log(y+delta) + (1-self._tdata) + np.log((1-y)+delta))

    def train(self):
        def f(x): return self.feed_forward()
        print("Initial loss value = ", self.loss_val())

        for step in range(10001):
            self._W2 -= self._learning_rate * numerical_derivative(f, self._W2)
            self._b2 -= self._learning_rate * numerical_derivative(f, self._b2)
            self._W3 -= self._learning_rate * numerical_derivative(f, self._W3)
            self._b3 -= self._learning_rate * numerical_derivative(f, self._b3)

            if (step % 400 == 0):
                print(f"step = {step}, loss value = {self.loss_val()}")

    def predict(self, xdata):
        z2 = np.dot(xdata, self._W2) + self._b2
        a2 = self._sigmoid(z2)

        z3 = np.dot(a2, self._W3) + self._b3
        y = a3 = self._sigmoid(z3)

        result = 1 if y > 0.5 else 0

        return y, result


def run_mnist():

    train_data: torch.Tensor = (datasets.MNIST(root='./data',
                                               train=True,
                                               download=True,
                                               transform=transforms.ToTensor()))
    test_data: torch.Tensor = (datasets.MNIST(root='./data',
                                              train=False,
                                              download=True,
                                              transform=transforms.ToTensor()))

    print(
        f"trainig_data.shape = {train_data.shape}, test_data.shape = {test_data.shape}")

    return


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes) -> None:
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes


def main():
    # simple_regression()
    # simple_logistic_regression()
    run_mnist()
    return


main()
