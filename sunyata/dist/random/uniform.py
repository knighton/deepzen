import numpy as np

from ..base.distribution import Distribution


class Uniform(Distribution):
    @classmethod
    def make(self, shape, dtype, min=-0.05, max=0.05):
        return np.random.uniform(min, max, shape).astype(dtype)

    def __init__(self, min=-0.05, max=0.05):
        self.min = min
        self.max = max

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype, self.min, self.max)


def uniform(min=-0.05, max=0.05):
    return Uniform(min, max)
