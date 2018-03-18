from ... import api as Z
from ...dist import unpack_distribution
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
    def __init__(self, dim=None, has_bias=False, kernel_init='glorot_uniform',
                 bias_init='zero'):
        Spec.__init__(self, 0)
        self.dim = dim
        self.has_bias = has_bias
        self.kernel_init = unpack_distribution(kernel_init)
        self.bias_init = unpack_distribution(bias_init)

    def checked_build(self, x_sig):
        assert x_sig.has_channels()
        in_dim, = x_sig.sample_shape()
        if self.dim is None:
            out_dim = in_dim
        else:
            out_dim = self.dim
        kernel = self.kernel_init((out_dim, in_dim), x_sig.dtype(), 'kernel')
        if self.has_bias:
            bias = self.bias_init((out_dim,), x_sig.dtype(), 'bias')
        else:
            bias = None
        out_sample_shape = (out_dim,)
        y_sig = Signature(out_sample_shape, x_sig.dtype(), x_sig.has_channels())
        return DenseLayer(x_sig, y_sig, kernel, bias)
