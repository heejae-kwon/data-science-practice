from datetime import datetime
import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def simple_logistic_regression():
    x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10, 1)
    t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10, 1)

    W = np.random.rand(1, 1)  # (input, 1)
    b = np.random.rand(1)
    print(f'W={W} W.shape={W.shape} b={b} b.shape={b.shape}')

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


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate) -> None:
        #self.target_data = np.zeros()
        #self.input_data = np.zeros()

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.W2 = np.random.randn(
            self.input_nodes, self.hidden_nodes) / np.sqrt(self.input_nodes/2)
        self.b2 = np.random.rand(self.hidden_nodes)

        self.W3 = np.random.randn(
            self.hidden_nodes, self.output_nodes)/np.sqrt(self.hidden_nodes/2)
        self.b3 = np.random.rand(self.output_nodes)

        self.Z3 = np.zeros([1, output_nodes])
        self.A3 = np.zeros([1, output_nodes])

        self.Z2 = np.zeros([1, hidden_nodes])
        self.A2 = np.zeros([1, hidden_nodes])

        self.Z1 = np.zeros([1, input_nodes])
        self.A1 = np.zeros([1, input_nodes])

        self.learning_rate = learning_rate

        self.target_data = np.zeros()

    def feed_forward(self):
        delta = 1e-7

        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)

        # cross_entropy
        return -np.sum(self.target_data * np.log(self.A3+delta) + (1-self.target_data) * np.log((1-self.A3)+delta))

    def loss_val(self):
        return self.feed_forward()

    def train(self, input_data, target_data):
        self.target_data = target_data
        self.input_data = input_data

        loss_val = self.feed_forward()

        loss_3 = (self.A3-self.target_data) * self.A3 * (1-self.A3)

        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)
        self.b3 = self.b3 - self.learning_rate * loss_3

        loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1-self.A2)

        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)
        self.b2 = self.b2 - self.learning_rate*loss_2

    def predict(self, input_data):
        Z2 = np.dot(input_data, self.W2) + self.b2
        A2 = sigmoid(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = sigmoid(Z3)

        predicted_num = np.argmax(A3)

        return predicted_num

    def accuracy(self, test_data):
        matched_list = []
        not_matched_list = []

        for index in range(len(test_data)):
            label = int(test_data[index, 0])  # 정답

            # normalize
            data = (test_data[index, 1:]/255.0*0.99) + 0.01
            predicted_num = self.predict(np.array(data, ndmin=2))

            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)

        print(
            f"Current Accuracy = {100*(len(matched_list)/(len(test_data))) }%")

        return matched_list, not_matched_list


def run_mnist():
    traning_data = np.loadtxt(
        './data/MNIST/mnist_train.csv', delimiter=',', dtype=np.float32)

    test_data = np.loadtxt(
        './data/MNIST/mnist_test.csv', delimiter=',', dtype=np.float32)

    print(
        f'traing_data.shape = {traning_data.shape}, test_data.shape = {test_data.shape}')

    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    epochs = 1

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

    start_time = datetime.now()
    for i in range(epochs):
        for step in range(len(traning_data)):

            target_data = np.zeros(output_nodes)+0.01
            target_data[int(traning_data[step, 0])] = 0.99

            input_data = ((traning_data[step, 1:]/255.0)*0.99) + 0.01

            nn.train(np.array(input_data, ndmin=2),
                     np.array(target_data, ndmin=2))

            if step % 400 == 0:
                print(f"step = {step}, loss_val = {nn.loss_val()}")

    end_time = datetime.now()
    print(f'elapsed time = {end_time-start_time}')

    nn.accuracy(test_data)

    return


def main():
    # simple_regression()
    # simple_logistic_regression()
    run_mnist()
    return


main()
