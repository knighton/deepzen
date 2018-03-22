import numpy as np

from ..base.initializer import Initializer
from ..base.registry import register_initializer


@register_initializer
class Eye(Initializer):
    name = 'eye'

    @classmethod
    def make(cls, dim, scale, dtype):
        x = scale * np.eye(dim)
        return x.astype(dtype)

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, shape, dtype, meaning=None):
        assert len(set(shape)) == 1
        dim = shape[0]
        return self.make(dim, self.scale, dtype)
