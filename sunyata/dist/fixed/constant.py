import numpy as np

from ..base.distribution import Distribution


class Constant(Distribution):
    @classmethod
    def make(cls, value, shape, dtype):
        return np.full(shape, value, dtype)

    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype, meaning=None):
        return self.make(self.value, shape, dtype)


def constant(value):
    return Constant(value)