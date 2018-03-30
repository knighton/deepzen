from torch.nn import functional as F

from .....util.unpack import unpack_dim, unpack_shape
from ....base.transform.shape.pool import \
    BaseAvgPoolAPI, BaseMaxPoolAPI, BasePoolAPI


class PyTorchAvgPoolAPI(BaseAvgPoolAPI):
    def avg_pool(self, x, face=2, stride=None, padding=0, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        if xsnd == 1:
            func = self.avg_pool1d
        elif xsnd == 2:
            func = self.avg_pool2d
        elif xsnd == 3:
            func = self.avg_pool3d
        else:
            assert False
        return func(x, face, stride, padding)

    def avg_pool1d(self, x, face=2, stride=None, padding=0):
        face = unpack_dim(face)
        if stride is None:
            stride = face
        else:
            stride = unpack_dim(stride)
        padding = unpack_dim(padding)
        return F.avg_pool1d(x, face, stride, padding)

    def avg_pool2d(self, x, face=2, stride=None, padding=0):
        face = unpack_shape(face, 2)
        if stride is None:
            stride = face
        else:
            stride = unpack_shape(stride, 2)
        padding = unpack_shape(padding, 2)
        return F.avg_pool2d(x, face, stride, padding)

    def avg_pool3d(self, x, face=2, stride=None, padding=0):
        face = unpack_shape(face, 3)
        if stride is None:
            stride = face
        else:
            stride = unpack_shape(stride, 3)
        padding = unpack_shape(padding, 3)
        return F.avg_pool3d(x, face, stride, padding)


class PyTorchMaxPoolAPI(BaseMaxPoolAPI):
    def max_pool(self, x, face=2, stride=None, padding=0, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        if xsnd == 1:
            func = self.max_pool1d
        elif xsnd == 2:
            func = self.max_pool2d
        elif xsnd == 3:
            func = self.max_pool3d
        else:
            assert False
        return func(x, face, stride, padding)

    def max_pool1d(self, x, face=2, stride=None, padding=0):
        face = unpack_dim(face)
        if stride is None:
            stride = face
        else:
            stride = unpack_dim(stride)
        padding = unpack_dim(padding)
        return F.max_pool1d(x, face, stride, padding)

    def max_pool2d(self, x, face=2, stride=None, padding=0):
        face = unpack_shape(face, 2)
        if stride is None:
            stride = face
        else:
            stride = unpack_shape(stride, 2)
        padding = unpack_shape(padding, 2)
        return F.max_pool2d(x, face, stride, padding)

    def max_pool3d(self, x, face=2, stride=None, padding=0):
        face = unpack_shape(face, 3)
        if stride is None:
            stride = face
        else:
            stride = unpack_shape(stride, 3)
        padding = unpack_shape(padding, 3)
        return F.max_pool3d(x, face, stride, padding)


class PyTorchPoolAPI(BasePoolAPI, PyTorchAvgPoolAPI, PyTorchMaxPoolAPI):
    pass
