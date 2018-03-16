from ... import api as Z
from ..base.layer import Layer
from ..base.spec import Spec


class ReLULayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig, sig)

    def forward(self, x, is_training):
        return Z.relu(x)


class ReLUSpec(Spec):
    def build(self, x_sig=None):
        return ReLULayer(x_sig)
