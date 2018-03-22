import numpy as np

from ..base.initializer import Initializer


class One(Initializer):
    @classmethod
    def make(cls, shape, dtype):
        return np.ones(shape, dtype)

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype)


def one():
    return One()
