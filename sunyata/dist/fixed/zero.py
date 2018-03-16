import numpy as np

from ..base.distribution import Distribution


class Zero(Distribution):
    @classmethod
    def make(cls, shape, dtype):
        return np.zeros(shape, dtype)

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype)


def zero():
    return Zero()
