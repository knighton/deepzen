import numpy as np

from ..base.initializer import Initializer
from ..base.registry import register_initializer


@register_initializer
class Constant(Initializer):
    name = 'constant'

    @classmethod
    def parse_scalar(cls, x):
        try:
            x = int(x)
        except:
            try:
                x = float(x)
            except:
                x = None
        return x

    @classmethod
    def unpack(cls, x):
        if isinstance(x, (float, int)):
            x = cls(x)
        elif isinstance(x, str):
            x = cls.parse_scalar(x)
            if x is not None:
                x = cls(x)
        else:
            x = None
        return x

    @classmethod
    def make(cls, value, shape, dtype):
        return np.full(shape, value, dtype)

    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype, meaning=None):
        return self.make(self.value, shape, dtype)
