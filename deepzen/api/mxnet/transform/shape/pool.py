import mxnet as mx

from .....util.unpack import unpack_shape
from ....base.transform.shape.pool import \
    BaseAvgPoolAPI, BaseMaxPoolAPI, BasePoolAPI


class MXNetAvgPoolAPI(BaseAvgPoolAPI):
    def avg_pool(self, x, face=2, stride=None, padding=0, space=None):
        spatial_ndim = self.spatial_ndim(x)
        if space is not None:
            assert space == spatial_ndim
        face = unpack_shape(face, spatial_ndim)
        if stride is None:
            stride = face
        else:
            stride = unpack_shape(stride, spatial_ndim)
        padding = unpack_shape(padding, spatial_ndim)
        return mx.nd.Pooling(data=x, kernel=face, stride=stride, pad=padding,
                             pool_type='avg')

    def avg_pool1d(self, x, face=2, stride=None, padding=0):
        return self.avg_pool(x, face, stride, padding, 1)

    def avg_pool2d(self, x, face=2, stride=None, padding=0):
        return self.avg_pool(x, face, stride, padding, 2)

    def avg_pool3d(self, x, face=2, stride=None, padding=0):
        return self.avg_pool(x, face, stride, padding, 3)


class MXNetMaxPoolAPI(BaseMaxPoolAPI):
    def max_pool(self, x, face=2, stride=None, padding=0, space=None):
        spatial_ndim = self.spatial_ndim(x)
        if space is not None:
            assert space == spatial_ndim
        face = unpack_shape(face, spatial_ndim)
        if stride is None:
            stride = face
        else:
            stride = unpack_shape(stride, spatial_ndim)
        padding = unpack_shape(padding, spatial_ndim)
        return mx.nd.Pooling(data=x, kernel=face, stride=stride, pad=padding,
                             pool_type='max')

    def max_pool1d(self, x, face=2, stride=None, padding=0):
        return self.max_pool(x, face, stride, padding, 1)

    def max_pool2d(self, x, face=2, stride=None, padding=0):
        return self.max_pool(x, face, stride, padding, 2)

    def max_pool3d(self, x, face=2, stride=None, padding=0):
        return self.max_pool(x, face, stride, padding, 3)


class MXNetPoolAPI(BasePoolAPI, MXNetAvgPoolAPI, MXNetMaxPoolAPI):
    pass
