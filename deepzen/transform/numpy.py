import numpy as np

from .base.transformer import Transformer


class Numpy(Transformer):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x):
        return np.array(x).astype(self.dtype)
