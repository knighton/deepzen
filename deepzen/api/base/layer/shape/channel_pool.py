import numpy as np

from .....util.unpack import unpack_shape


class BaseChannelAvgPoolAPI(object):
    def channel_avg_pool(self, x, face, xsnd=None):
        raise NotImplementedError

    def channel_avg_pool0d(self, x, face):
        raise NotImplementedError

    def channel_avg_pool1d(self, x, face):
        raise NotImplementedError

    def channel_avg_pool2d(self, x, face):
        raise NotImplementedError

    def channel_avg_pool3d(self, x, face):
        raise NotImplementedError


class BaseChannelMaxPoolAPI(object):
    def channel_max_pool(self, x, face, xsnd=None):
        raise NotImplementedError

    def channel_max_pool0d(self, x, face):
        raise NotImplementedError

    def channel_max_pool1d(self, x, face):
        raise NotImplementedError

    def channel_max_pool2d(self, x, face):
        raise NotImplementedError

    def channel_max_pool3d(self, x, face):
        raise NotImplementedError


class BaseChannelPoolAPI(BaseChannelAvgPoolAPI, BaseChannelMaxPoolAPI):
    def channel_pool_signature(self, x_sig, face=2):
        assert x_sig.has_channels()
        assert isinstance(face, int)
        assert 1 <= face <= x_sig.channels()
        assert x_sig.channels() % face == 0
        y_sample_shape = list(x_sig.sample_shape())
        y_sample_shape[0] //= face
        y_sample_shape = tuple(y_sample_shape)
        return x_sig.as_shape(y_sample_shape)
