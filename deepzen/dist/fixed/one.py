import numpy as np

from ..base.distribution import Distribution


class One(Distribution):
    @classmethod
    def make(cls, shape, dtype):
        return np.ones(shape, dtype)

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype)


def one():
    return One()
