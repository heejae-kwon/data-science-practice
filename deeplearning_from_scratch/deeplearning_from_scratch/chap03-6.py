import sys
import os
import pickle
import numpy as np

from pathlib import Path
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img: np.ndarray):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data() -> tuple[np.ndarray, np.ndarray]:
    (train_x, train_t), (test_x, test_t) = load_mnist(flatten=True,
                                                      normalize=True,
                                                      one_hot_label=False)
    return x_test, t_test


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


def init_network():
    path = Path('./deeplearning_from_scratch/sample_weight.pkl').absolute()
    with open(path, 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network: dict, x: np.ndarray):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_funtion(a3)

    return y


(train_x, train_t), (test_x, test_t) = load_mnist(flatten=True,
                                                  normalize=False)

x_train: np.ndarray = train_x
t_train: np.ndarray = train_t
x_test: np.ndarray = test_x
t_test: np.ndarray = test_t

img: np.ndarray = x_train[0]
label: np.ndarray = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print(f'Accuracy: {str(float(accuracy_cnt)/len(x))}')


batch_size = 100
accuracy_cnt = 0
# 100 단위로 끊어서
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print(f'Accuracy: {str(float(accuracy_cnt)/len(x))}')
