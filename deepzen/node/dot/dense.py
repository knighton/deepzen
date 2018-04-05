from ... import api as Z
from ...init import get_initializer
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.signature import Signature
from ..base.spec import XYSpec


class DenseLayer(XYLayer):
    def __init__(self, x_sig, y_sig, kernel, bias):
        XYLayer.__init__(self, x_sig, y_sig)
        self._kernel = self.param(kernel)
        self._bias = self.param(bias)

    def forward_x_y(self, x, is_training):
        return Z.dense(x, self._kernel, self._bias)


class DenseSpec(XYSpec):
    def __init__(self, dim=None, has_bias=False, kernel_init='glorot_uniform',
                 bias_init='zero'):
        XYSpec.__init__(self, 0)
        self._dim = dim
        self._has_bias = has_bias
        self._kernel_init = get_initializer(kernel_init)
        self._bias_init = get_initializer(bias_init)

    def build_x_y(self, x_sig):
        assert x_sig.has_channels()
        in_dim, = x_sig.sample_shape()
        if self._dim is None:
            out_dim = in_dim
        else:
            out_dim = self._dim
        kernel = self._kernel_init((out_dim, in_dim), x_sig.dtype(), 'kernel')
        if self._has_bias:
            bias = self._bias_init((out_dim,), x_sig.dtype(), 'bias')
        else:
            bias = None
        out_sample_shape = (out_dim,)
        y_sig = Signature(out_sample_shape, x_sig.dtype(), x_sig.has_channels())
        return DenseLayer(x_sig, y_sig, kernel, bias)


Dense = keywordize(DenseSpec)
