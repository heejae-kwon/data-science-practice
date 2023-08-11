import numpy as np
import matplotlib.pyplot as plt
from common.utils import sigmoid, relu


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x: np.ndarray, train_fig=True):
        if train_fig:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0-self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations: dict[int, np.ndarray] = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    w = (np.random.randn(node_num, node_num) *
         (np.sqrt(2/node_num)))
    a: np.ndarray = np.dot(x, w)
    # z = sigmoid(a)
    z = relu(a)
    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(f"{i+1}-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()
