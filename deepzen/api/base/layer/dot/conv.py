from math import floor

from .....util.unpack import unpack_shape


class BaseConvAPI(object):
    def conv(self, x, kernel, bias, stride, padding, dilation, xsnd=None):
        raise NotImplementedError

    def conv1d(self, x, kernel, bias, stride, padding, dilation):
        raise NotImplementedError

    def conv2d(self, x, kernel, bias, stride, padding, dilation):
        raise NotImplementedError

    def conv3d(self, x, kernel, bias, stride, padding, dilation):
        raise NotImplementedError

    def conv_y_sample_shape(self, x_sample_shape, out_channels, face, stride,
                            padding, dilation):
        x_spatial_ndim = len(x_sample_shape) - 1
        face = unpack_shape(face, x_spatial_ndim)
        stride = unpack_shape(stride, x_spatial_ndim)
        padding = unpack_shape(padding, x_spatial_ndim)
        dilation = unpack_shape(dilation, x_spatial_ndim)
        y_sample_shape = [out_channels] + [0] * x_spatial_ndim
        for i in range(x_spatial_ndim):
            numerator = x_sample_shape[1 + i] + 2 * padding[i] - \
                dilation[i] * (face[i] - 1) - 1
            y_sample_shape[1 + i] = floor(numerator // stride[i] + 1)
        return tuple(y_sample_shape)

    def conv_transpose(self, x, kernel, bias, stride, padding, dilation,
                       xsnd=None):
        raise NotImplementedError

    def conv_transpose1d(self, x, kernel, bias, stride, padding, dilation):
        raise NotImplementedError

    def conv_transpose2d(self, x, kernel, bias, stride, padding, dilation):
        raise NotImplementedError

    def conv_transpose3d(self, x, kernel, bias, stride, padding, dilation):
        raise NotImplementedError

    def conv_transpose_y_sample_shape(self, x_sample_shape, out_channels, face,
                                      stride, padding, dilation):
        x_spatial_ndim = len(x_sample_shape) - 1
        face = unpack_shape(face, x_spatial_ndim)
        stride = unpack_shape(stride, x_spatial_ndim)
        padding = unpack_shape(padding, x_spatial_ndim)
        dilation = unpack_shape(dilation, x_spatial_ndim)
        y_sample_shape = [None] * len(x_sample_shape)
        y_sample_shape[0] = out_channels
        for i in range(x_spatial_ndim):
            y_sample_shape[i + 1] = \
                (x_sample_shape[i + 1] - 1) * stride[i] - 2 * padding[i] + \
                window[i]
        return tuple(y_sample_shape)
