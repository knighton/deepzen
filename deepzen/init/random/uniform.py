import numpy as np

from ..base.initializer import Initializer
from ..base.registry import register_initializer


@register_initializer
class Uniform(Initializer):
    name = 'uniform'

    @classmethod
    def make(self, shape, dtype, min=-0.05, max=0.05):
        return np.random.uniform(min, max, shape).astype(dtype)

    def __init__(self, min=-0.05, max=0.05):
        self.min = min
        self.max = max

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype, self.min, self.max)
