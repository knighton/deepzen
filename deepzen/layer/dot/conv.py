from ... import api as Z
from ...init import get_initializer
from ...util.unpack import unpack_shape
from ..base.layer import XYLayer
from ..base.signature import Signature
from ..base.spec import XYSpec


class ConvLayer(XYLayer):
    def __init__(self, x_sig, y_sig, kernel, bias, stride, padding, dilation):
        XYLayer.__init__(self, x_sig, y_sig)
        self._kernel = self.param(kernel)
        self._bias = self.param(bias)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation

    def forward_x_y(self, x, is_training):
        x_sig, = self._x_sigs
        xsnd = x_sig.spatial_ndim()
        return Z.conv(x, self._kernel, self._bias, self._stride, self._padding,
                      self._dilation, xsnd)


class ConvSpec(XYSpec):
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

    def build_x_y(self, x_sig):
        assert x_sig.has_channels()
        x_channels = x_sig.channels()
        if self._channels:
            y_channels = self._channels
        else:
            y_channels = x_channels
        face = unpack_shape(self._face, x_sig.spatial_ndim())
        kernel_shape = (y_channels, x_channels) + face
        kernel = self._kernel_init(kernel_shape, x_sig.dtype(), 'kernel')
        if self._has_bias:
            bias_shape = y_channels,
            bias = self._bias_init(bias_shape, x_sig.dtype(), 'bias')
        else:
            bias = None
        y_sample_shape = Z.conv_y_sample_shape(
            x_sig.sample_shape(), y_channels, face, self._stride,
            self._padding, self._dilation)
        y_sig = Signature(y_sample_shape, x_sig.dtype(), True)
        return ConvLayer(x_sig, y_sig, kernel, bias, self._stride,
                         self._padding, self._dilation)
