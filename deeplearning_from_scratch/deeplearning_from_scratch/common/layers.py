import numpy as np
from common.utils import softmax, sigmoid, cross_entropy_error




class Relu:
    def __init__(self) -> None:
        self.mask: np.ndarray = None

    def forward(self, x: np.ndarray):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: np.ndarray):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.out: np.ndarray = None

    def forward(self, x: np.ndarray):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout: np.ndarray):
        dx = dout*(1.0-self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W: np.ndarray = W
        self.b: np.ndarray = b

        self.x: np.ndarray = None
        self.original_x_shape = None
        # 가중치 매개변수 미분
        self.dW: np.ndarray = None
        self.db: np.ndarray = None

    def forward(self, x: np.ndarray):
        # 텐서 입력시
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out: np.ndarray = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx: np.ndarray = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(
            *self.original_x_shape)  # 입력데이터의 형상으로 복구(텐서 표현)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y: np.ndarray = None
        self.t: np.ndarray = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.float64:
        self.t = t
        self.y: np.ndarray = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 원-핫-인코딩
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
