import numpy as np

from ..base.initializer import Initializer
from ..base.registry import register_initializer


@register_initializer
class One(Initializer):
    name = 'one'

    @classmethod
    def make(cls, shape, dtype):
        return np.ones(shape, dtype)

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype)
