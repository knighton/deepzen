import numpy as np

from .. import backend as Z
from .base.form import Form
from .base.layer import Layer
from .base.spec import Spec


class DenseLayer(Layer):
    def __init__(self, kernel, bias):
        self.kernel = Z.variable(Z.tensor(kernel))
        self.bias = Z.variable(Z.tensor(bias))

    def params(self):
        return [self.kernel, self.bias]

    def forward(self, x):
        return Z.matmul(x, self.kernel) + self.bias


class DenseSpec(Spec):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def build(self, form=None):
        in_dim, = form.shape
        kernel = np.random.normal(
            0, 0.1, (in_dim, self.out_dim)).astype('float32')
        bias = np.zeros(self.out_dim, 'float32')
        out_shape = (self.out_dim,)
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)