import numpy as np

from ..base.initializer import Initializer
from ..base.registry import register_initializer


@register_initializer
class Normal(Initializer):
    name = 'normal'

    @classmethod
    def make(cls, shape, dtype, mean=0, std=0.05):
        return np.random.normal(mean, std, shape).astype(dtype)

    def __init__(self, mean=0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype, self.mean, self.std)
