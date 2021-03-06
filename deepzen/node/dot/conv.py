from ... import api as Z
from ...init import get_initializer
from ...util.unpack import unpack_shape
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.signature import Signature
from ..base.spec import XYSpec


class BaseConvLayer(XYLayer):
    def __init__(self, x_sig, y_sig, kernel, bias, stride, padding, dilation):
        XYLayer.__init__(self, x_sig, y_sig)
        self._kernel = self.param(kernel)
        self._bias = self.param(bias)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation


class BaseConvSpec(XYSpec):
    def __init__(self, channels=None, face=3, stride=1, padding=1, dilation=1,
                 kernel_init='glorot_uniform', bias_init='zero', has_bias=True,
                 xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._channels = channels
        self._face = face
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._kernel_init = get_initializer(kernel_init)
        self._bias_init = get_initializer(bias_init)
        self._has_bias = has_bias

    def y_sample_shape(self, x_sample_shape, out_channels, face, stride,
                       padding, dilation):
        raise NotImplementedError

    def make_layer(self, x_sig, y_sig, kernel, bias, stride, padding, dilation):
        raise NotImplementedError

    def build_x_y(self, x_sig):
        assert x_sig.has_channels()
        x_channels = x_sig.channels()
        if self._channels:
            y_channels = self._channels
        else:
            y_channels = x_channels
        face = unpack_shape(self._face, x_sig.spatial_ndim())
        kernel_shape = self.make_kernel_shape(x_channels, y_channels, face)
        kernel = self._kernel_init(kernel_shape, x_sig.dtype(), 'kernel')
        if self._has_bias:
            bias_shape = y_channels,
            bias = self._bias_init(bias_shape, x_sig.dtype(), 'bias')
        else:
            bias = None
        y_sample_shape = self.y_sample_shape(
            x_sig.sample_shape(), y_channels, face)
        y_sig = Signature(y_sample_shape, x_sig.dtype(), True)
        return self.make_layer(x_sig, y_sig, kernel, bias)


class ConvLayer(BaseConvLayer):
    def forward_x_y(self, x, is_training):
        x_sig, = self._x_sigs
        xsnd = x_sig.spatial_ndim()
        return Z.conv(x, self._kernel, self._bias, self._stride, self._padding,
                      self._dilation, xsnd)


class ConvSpec(BaseConvSpec):
    def y_sample_shape(self, x_sample_shape, out_channels, face):
        return Z.conv_y_sample_shape(
            x_sample_shape, out_channels, face, self._stride, self._padding,
            self._dilation)

    def make_kernel_shape(self, x_channels, y_channels, face):
        return (y_channels, x_channels) + face

    def make_layer(self, x_sig, y_sig, kernel, bias):
        return ConvLayer(x_sig, y_sig, kernel, bias, self._stride,
                         self._padding, self._dilation)


Conv, Conv1D, Conv2D, Conv3D = keywordize(ConvSpec, [None, 1, 2, 3])


class ConvTransposeLayer(BaseConvLayer):
    def __init__(self, x_sig, y_sig, kernel, bias, stride, padding, out_padding,
                 dilation):
        BaseConvLayer.__init__(self, x_sig, y_sig, kernel, bias, stride,
                               padding, dilation)
        self._out_padding = out_padding

    def forward_x_y(self, x, is_training):
        x_sig, = self._x_sigs
        xsnd = x_sig.spatial_ndim()
        return Z.conv_transpose(
            x, self._kernel, self._bias, self._stride, self._padding,
            self._out_padding, self._dilation, xsnd)


class ConvTransposeSpec(BaseConvSpec):
    def __init__(self, channels=None, face=3, stride=1, padding=1,
                 out_padding=0, dilation=1, kernel_init='glorot_uniform',
                 bias_init='zero', has_bias=True, xsnd=None):
        BaseConvSpec.__init__(self, channels, face, stride, padding, dilation,
                              kernel_init, bias_init, has_bias, xsnd)
        self._out_padding = out_padding

    def y_sample_shape(self, x_sample_shape, out_channels, face):
        return Z.conv_transpose_y_sample_shape(
            x_sample_shape, out_channels, face, self._stride, self._padding,
            self._out_padding, self._dilation)

    def make_kernel_shape(self, x_channels, y_channels, face):
        return (x_channels, y_channels) + face

    def make_layer(self, x_sig, y_sig, kernel, bias):
        return ConvTransposeLayer(
            x_sig, y_sig, kernel, bias, self._stride, self._padding,
            self._out_padding, self._dilation)


ConvTranspose, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D = \
    keywordize(ConvTransposeSpec, [None, 1, 2, 3])
