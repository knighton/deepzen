import numpy as np

from ... import api as Z
from ..base.layer import Layer
from ..base.signature import Signature
from ..base.spec import Spec


class FlattenLayer(Layer):
    def __init__(self, x_sig, y_sig):
        Layer.__init__(self, x_sig, y_sig)

    def forward(self, x, is_training):
        return Z.batch_flatten(x)


class FlattenSpec(Spec):
    def build(self, x_sig=None):
        out_shape = int(np.prod(x_sig.shape)),
        y_sig = Signature(out_shape, x_sig.dtype)
        return FlattenLayer(x_sig, y_sig)
