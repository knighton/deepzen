from ... import api as Z
from ...dist import unpack_distribution
from ...util.unpack import unpack_shape
from ..base.layer import Layer
from ..base.signature import Signature
from ..base.spec import Spec


class ConvLayer(Layer):
    def __init__(self, x_sig, y_sig, kernel, bias, stride, padding, dilation):
        Layer.__init__(self, x_sig, y_sig)
        self._kernel = self.param(kernel)
        if bias is None:
            self._bias = bias
        else:
            self._bias = self.param(bias)
        self._stride = stride
        self._padding = padding
        self._dilation = dilation

    def forward(self, x, is_training):
        space = self._x_sig.spatial_ndim()
        return Z.conv(x, self._kernel, self._bias, self._stride, self._padding,
                      self._dilation, space)


class ConvSpec(Spec):
    def __init__(self, dim=None, face=3, stride=1, padding=1, dilation=1,
                 kernel_init='glorot_uniform', bias_init='zero', has_bias=True,
                 space=None):
        Spec.__init__(self, space)
        self._dim = dim
        self._face = face
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._kernel_init = unpack_distribution(kernel_init)
        self._bias_init = unpack_distribution(bias_init)
        self._has_bias = has_bias

    def checked_build(self, x_sig):
        assert x_sig.has_channels()
        x_channels = x_sig.channels()
        if self._dim:
            y_channels = self._dim
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
