import numpy as np

from ... import api as Z


class Signature(object):
    """
    Tensor shape/dtype.
    """

    def __init__(self, sample_shape, dtype, has_channels=True):
        assert isinstance(sample_shape, tuple)
        assert dtype
        assert isinstance(dtype, str)
        assert isinstance(has_channels, bool)
        self._sample_shape = sample_shape
        self._dtype = dtype
        self._has_channels = has_channels

    def batch_ndim(self):
        return 1 + len(self._sample_shape)

    def batch_shape(self, batch_size):
        return (batch_size,) + self._sample_shape

    def batch_size(self, batch_size):
        return np.prod((batch_size,) + self._sample_shape)

    def sample_ndim(self):
        return len(self._sample_shape)

    def sample_shape(self):
        return self._sample_shape

    def sample_size(self):
        return np.prod(self._sample_shape)

    def spatial_ndim(self):
        assert self._has_channels
        return -1 + len(self._sample_shape)

    def spatial_ndim_or_none(self):
        if self._has_channels:
            x = -1 + len(self._sample_shape)
        else:
            x = None
        return x

    def spatial_shape(self):
        assert self._has_channels
        return self._sample_shape[1:]

    def spatial_size(self):
        return np.prod(self._sample_shape[1:])

    def channels(self):
        assert self._has_channels
        return self._sample_shape[0]

    def dtype(self):
        return self._dtype

    def has_channels(self):
        return self._has_channels

    def __eq__(self, x):
        if self._sample_shape != x._sample_shape:
            return False

        if self._dtype != x._dtype:
            return False

        if self._has_channels != x._has_channels:
            return False

        return True

    def accepts_batch_tensor(self, x):
        return Z.shape(x)[1:] == self._sample_shape and \
            Z.dtype(x) == self._dtype

    def as_shape(self, new_sample_shape):
        return Signature(new_sample_shape, self._dtype, self._has_channels)
