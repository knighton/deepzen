import numpy as np

from ... import api as Z
from ..base.layer import Layer
from ..base.signature import Signature
from ..base.spec import Spec


class FlattenLayer(Layer):
    def __init__(self):
        Layer.__init__(self)

    def forward(self, x, is_training):
        return Z.batch_flatten(x)


class FlattenSpec(Spec):
    def build(self, sig=None):
        out_shape = int(np.prod(sig.shape)),
        return FlattenLayer(), Signature(out_shape, sig.dtype)
