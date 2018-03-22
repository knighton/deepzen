import mxnet as mx

from .....util.unpack import unpack_shape
from ....base.transform.dot.conv import BaseConvAPI


class MXNetConvAPI(BaseConvAPI):
    def conv(self, x, kernel, bias, stride, padding, dilation, space=None):
        if space is None:
            space = x.ndim - 2
        else:
            assert space == x.ndim - 2
        face = kernel.shape[2:]
        stride = unpack_shape(stride, space)
        padding = unpack_shape(padding, space)
        dilation = unpack_shape(dilation, space)
        out_dim = kernel.shape[0]
        return mx.nd.Convolution(x, kernel, bias, face, stride, dilation,
                                 padding, out_dim)

    def conv1d(self, x, kernel, bias, stride, padding, dilation):
        return self.conv(x, kernel, bias, stride, padding, dilation, 1)

    def conv2d(self, x, kernel, bias, stride, padding, dilation):
        return self.conv(x, kernel, bias, stride, padding, dilation, 2)

    def conv3d(self, x, kernel, bias, stride, padding, dilation):
        return self.conv(x, kernel, bias, stride, padding, dilation, 3)
