import numpy as np

from ..base.initializer import Initializer
from ..base.registry import register_initializer


@register_initializer
class Zero(Initializer):
    name = 'zero'

    @classmethod
    def make(cls, shape, dtype):
        return np.zeros(shape, dtype)

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype)
