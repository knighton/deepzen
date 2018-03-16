from ... import api as Z


class Signature(object):
    def __init__(self, sample_shape, dtype, has_channels=True):
        self._sample_shape = sample_shape
        self._dtype = dtype
        self._has_channels = has_channels

    def batch_shape(self, batch_size):
        return (batch_size,) + self._sample_shape

    def batch_ndim(self):
        return 1 + len(self._sample_shape)

    def sample_shape(self):
        return self._sample_shape

    def sample_ndim(self):
        return len(self._sample_shape)

    def spatial_shape(self):
        assert self._has_channels
        return self._sample_shape[1:]

    def spatial_ndim(self):
        assert self._has_channels
        return -1 + len(self._sample_shape)

    def dtype(self):
        return self._dtype

    def has_channels(self):
        return self._has_channels

    def equals(self, x):
        if self._sample_shape != x._sample_shape:
            return False

        if self._dtype != x._dtype:
            return False

        if self._has_channels != x._has_channels:
            return False

        return True

    def check(self, x):
        assert Z.shape(x)[1:] == self._sample_shape
        assert Z.dtype(x) == self._dtype
