import numpy as np
from step01 import Variable
from step02 import Function, Square
from step03 import Exp


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data-eps)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)


if __name__ == "__main__":
    def ABC(x: Variable):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

    x = Variable(np.array(0.5))
    dy = numerical_diff(ABC, x)
    print(dy)
