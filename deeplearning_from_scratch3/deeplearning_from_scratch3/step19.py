from __future__ import annotations
from dataclasses import dataclass
from heapq import heappop, heappush

import weakref
import numpy as np
import contextlib


def as_array(x):
    if np.isscalar(x):
        return np.array(x)

    return x


@dataclass
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data: np.ndarray, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}(은(는) 지원하지 않습니다.")

        self.name = name
        self.data = data
        self.grad: np.ndarray | None = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+' '*9)
        return f"variable({p})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def cleargrad(self):
        self.grad = None

    def set_creator(self, func: Function):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # heap 버전
        funcs_heap: list[tuple[int, Function]] = []
        seen_set: set[Function] = set()

        def add_func(f: Function):
            if f not in seen_set:
                heappush(funcs_heap, (-f.generation, f))
                seen_set.add(f)

        add_func(self.creator)

        while funcs_heap:
            f = heappop(funcs_heap)[1]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # 전파되는 미분값의 합 구하기
                if x.creator is not None:
                    # funcs.append(x.creator)
                    add_func(x.creator)

            if not retain_grad:  # 중간 변수의 미분값을 저장하지 않는다면
                for y in f.outputs:
                    y().grad = None  # 중간 변수의 미분값을 모두 None으로 재설정


class Function:
    def __call__(self, *inputs: Variable):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def __lt__(self, other: Function):
        self.generation < other.generation

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x*gy
        return gx


def square(x):
    return Square()(x)


if __name__ == "__main__":
    x = Variable(np.array([1, 2, 3]))
    print(x)
