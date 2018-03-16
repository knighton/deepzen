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
        assert x_sig.has_channels()
        y_sample_shape = int(np.prod(x_sig.sample_shape())),
        y_sig = Signature(y_sample_shape, x_sig.dtype(), True)
        return FlattenLayer(x_sig, y_sig)
