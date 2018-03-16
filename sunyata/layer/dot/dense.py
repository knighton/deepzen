import numpy as np

from ... import api as Z
from ..base.form import Form
from ..base.layer import Layer
from ..base.spec import Spec


class DenseLayer(Layer):
    def __init__(self, kernel, bias):
        self.kernel = Z.variable(kernel)
        if bias is None:
            self.bias = None
        else:
            self.bias = Z.variable(bias)

    def params(self):
        ret = [self.kernel]
        if self.bias is not None:
            ret.append(self.bias)
        return ret

    def forward(self, x, is_training):
        return Z.dense(x, self.kernel, self.bias)


class DenseSpec(Spec):
    def __init__(self, dim=None, has_bias=False):
        self.out_dim = dim
        self.has_bias = has_bias

    def build(self, form=None):
        in_dim, = form.shape
        if self.out_dim is None:
            out_dim = in_dim
        else:
            out_dim = self.out_dim
        kernel = np.random.normal(
            0, 0.1, (out_dim, in_dim)).astype('float32')
        if self.has_bias:
            bias = np.zeros(out_dim, 'float32')
        else:
            bias = None
        out_shape = (out_dim,)
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)
