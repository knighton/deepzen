import numpy as np


class BaseReshapeAPI(object):
    def reshape_batch(self, x, shape, xsnd=None):
        if xsnd is not None:
            assert self.ndim(x) - 2 == xsnd
        batch_size = self.shape(x)[0]
        return self.reshape(x, (batch_size,) + shape)

    def reshape_batch0d(self, x, shape):
        return self.reshape_batch(x, shape, 0)

    def reshape_batch1d(self, x, shape):
        return self.reshape_batch(x, shape, 1)

    def reshape_batch2d(self, x, shape):
        return self.reshape_batch(x, shape, 2)

    def reshape_batch3d(self, x, shape):
        return self.reshape_batch(x, shape, 3)

    def reshape_batch_signature(self, x_sig, shape):
        prod = 1
        hole = None
        for i, axis in enumerate(shape):
            if axis == -1:
                assert hole is None
                hole = i
            else:
                prod *= axis
        y_sample_shape = list(shape)
        sample_size = x_sig.sample_size()
        if hole is None:
            assert sample_size == prod
        else:
            assert not sample_size % prod
            y_sample_shape[hole] = sample_size // prod
        y_sample_shape = tuple(y_sample_shape)
        return x_sig.as_shape(y_sample_shape)

    def flatten_batch(self, x, xsnd=None):
        if xsnd is not None:
            assert self.ndim(x) - 2 == xsnd
        batch_size = self.shape(x)[0]
        sample_size = int(np.prod(self.shape(x)[1:]))
        return self.reshape(x, (batch_size, sample_size))

    def flatten_batch0d(self, x):
        return self.flatten_batch(x, 0)

    def flatten_batch1d(self, x):
        return self.flatten_batch(x, 1)

    def flatten_batch2d(self, x):
        return self.flatten_batch(x, 2)

    def flatten_batch3d(self, x):
        return self.flatten_batch(x, 3)

    def flatten_batch_signature(self, x_sig):
        shape = int(np.prod(x_sig.sample_shape())),
        return self.reshape_batch_signature(x_sig, shape)
