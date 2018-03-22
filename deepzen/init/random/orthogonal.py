import numpy as np

from ..base.initializer import Initializer


class Orthogonal(Initializer):
    @classmethod
    def make(cls, shape, dtype, scale=1):
        shape_2d = np.prod(shape[:-1]), shape[-1]
        x = np.random.normal(0, 1, shape_2d)
        u, s, v = np.linalg.svd(x, full_matrices=False)
        x = u if u.shape == shape_2d else v
        x = scale * x.reshape(shape)
        return x.astype(dtype)

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype, self.scale)


def orthogonal(scale=1):
    return Orthogonal(scale)
