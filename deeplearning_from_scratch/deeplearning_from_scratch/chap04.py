
import numpy as np
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
from utils import *


class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)  # 가중치 매개변수
# [[ 0.96664203 -0.84718188  0.39608711]
# [ 0.27554852  0.82301459 -0.65741839]]

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
# [ 0.82797889  0.23240401 -0.35402429]
print(np.argmax(p))  # 최댓값의 인덱스
# 0

t = np.array([0, 0, 1])
print(net.loss(x, t))
# 1.8014544459221402


def f(w): return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)
print(grads)

# 미니배치 학습
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

train_loss_list = []

# 하이퍼파라미터
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
