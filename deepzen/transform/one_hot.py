import numpy as np

from .base.transformer import Transformer


def to_one_hot(indices, num_classes, dtype):
    assert indices.ndim == 1
    assert isinstance(num_classes, int)
    assert 0 < num_classes
    x = np.zeros((len(indices), num_classes), dtype)
    x[np.arange(len(indices)), indices] = 1
    return x


class OneHot(Transformer):
    def __init__(self, vocab_size, dtype='float32'):
        self.vocab_size = vocab_size
        self.dtype = dtype

    def fit(self, x):
        if self.vocab_size is None:
            self.vocab_size = int(x.max() + 1)
        else:
            assert x.max() + 1 <= self.vocab_size

    def transform(self, x):
        return to_one_hot(x, self.vocab_size, self.dtype)

    def inverse_transform(self, x):
        assert x.ndim == 2
        return x.argmax(axis=-1)
