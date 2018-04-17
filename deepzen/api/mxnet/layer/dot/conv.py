import mxnet as mx

from .....util.unpack import unpack_shape
from ....base.layer.dot.conv import BaseConvAPI


class MXNetConvAPI(BaseConvAPI):
    def conv(self, x, kernel, bias, stride, padding, dilation, xsnd=None):
        if xsnd is None:
            xsnd = x.ndim - 2
        else:
            assert xsnd == x.ndim - 2
        face = kernel.shape[2:]
        stride = unpack_shape(stride, xsnd)
        padding = unpack_shape(padding, xsnd)
        dilation = unpack_shape(dilation, xsnd)
        out_dim = kernel.shape[0]
        return mx.nd.Convolution(x, kernel, bias, face, stride, dilation,
                                 padding, out_dim)

    def conv1d(self, x, kernel, bias, stride, padding, dilation):
        return self.conv(x, kernel, bias, stride, padding, dilation, 1)

    def conv2d(self, x, kernel, bias, stride, padding, dilation):
        return self.conv(x, kernel, bias, stride, padding, dilation, 2)

    def conv3d(self, x, kernel, bias, stride, padding, dilation):
        return self.conv(x, kernel, bias, stride, padding, dilation, 3)

    def conv_transpose(self, x, kernel, bias, stride, padding, out_padding,
                       dilation, xsnd=None):
        if xsnd is None:
            xsnd = x.ndim - 2
        else:
            assert xsnd == x.ndim - 2
        face = kernel.shape[2:]
        stride = unpack_shape(stride, xsnd)
        padding = unpack_shape(padding, xsnd)
        out_padding = unpack_shape(out_padding, xsnd)
        dilation = unpack_shape(dilation, xsnd)
        return mx.nd.Deconvolution(x, kernel, bias, face, stride, dilation,
                                   padding, out_padding)

    def conv_transpose1d(self, x, kernel, bias, stride, padding, out_padding,
                         dilation):
        return self.conv_transpose(
            x, kernel, bias, stride, padding, out_padding, dilation, 1)

    def conv_transpose2d(self, x, kernel, bias, stride, padding, out_padding,
                         dilation):
        return self.conv_transpose(
            x, kernel, bias, stride, padding, out_padding, dilation, 2)

    def conv_transpose3d(self, x, kernel, bias, stride, padding, out_padding,
                         dilation):
        return self.conv_transpose(
            x, kernel, bias, stride, padding, out_padding, dilation, 3)
