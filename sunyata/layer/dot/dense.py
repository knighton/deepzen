import numpy as np

from ... import api as Z
from ..base.layer import Layer
from ..base.signature import Signature
from ..base.spec import Spec


class DenseLayer(Layer):
    def __init__(self, x_sig, y_sig, kernel, bias):
        Layer.__init__(self, x_sig, y_sig)
        self.kernel = self.param(kernel)
        if bias is None:
            self.bias = None
        else:
            self.bias = self.param(bias)

    def forward(self, x, is_training):
        return Z.dense(x, self.kernel, self.bias)


class DenseSpec(Spec):
    def __init__(self, dim=None, has_bias=False):
        self.dim = dim
        self.has_bias = has_bias

    def build(self, x_sig):
        assert x_sig.has_channels()
        in_dim, = x_sig.sample_shape()
        if self.dim is None:
            out_dim = in_dim
        else:
            out_dim = self.dim
        kernel = np.random.normal(
            0, 0.1, (out_dim, in_dim)).astype('float32')
        if self.has_bias:
            bias = np.zeros(out_dim, 'float32')
        else:
            bias = None
        out_shape = (out_dim,)
        y_sig = Signature(out_shape, x_sig.dtype)
        return DenseLayer(x_sig, y_sig, kernel, bias)
