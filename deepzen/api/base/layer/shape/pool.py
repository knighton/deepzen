import numpy as np

from .....util.unpack import unpack_shape


class BaseAvgPoolAPI(object):
    def avg_pool(self, x, face=2, stride=None, padding=0, xsnd=None):
        raise NotImplementedError

    def avg_pool1d(self, x, face=2, stride=None, padding=0):
        raise NotImplementedError

    def avg_pool2d(self, x, face=2, stride=None, padding=0):
        raise NotImplementedError

    def avg_pool3d(self, x, face=2, stride=None, padding=0):
        raise NotImplementedError


class BaseMaxPoolAPI(object):
    def max_pool(self, x, face=2, stride=None, padding=0, xsnd=None):
        raise NotImplementedError

    def max_pool1d(self, x, face=2, stride=None, padding=0):
        raise NotImplementedError

    def max_pool2d(self, x, face=2, stride=None, padding=0):
        raise NotImplementedError

    def max_pool3d(self, x, face=2, stride=None, padding=0):
        raise NotImplementedError


class BasePoolAPI(BaseAvgPoolAPI, BaseMaxPoolAPI):
    def pool_signature(self, x_sig, face=2, stride=None, padding=0):
        assert x_sig.has_channels()
        face = unpack_shape(face, x_sig.spatial_ndim())
        if stride is None:
            stride = face
        else:
            stride = unpack_shape(stride, x_sig.spatial_ndim())
        padding = unpack_shape(padding, x_sig.spatial_ndim())
        y_sample_shape = [x_sig.channels()]
        for i in range(x_sig.spatial_ndim()):
            x_axis = x_sig.spatial_shape()[i]
            y_axis = (x_axis + 2 * padding[i] - face[i]) / stride[i]
            y_axis = int(np.floor(np.maximum(y_axis, 0))) + 1
            y_sample_shape.append(y_axis)
        y_sample_shape = tuple(y_sample_shape)
        return x_sig.as_shape(y_sample_shape)
