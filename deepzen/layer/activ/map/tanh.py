from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class TanhLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig)

    def forward(self, x, is_training):
        return Z.tanh(x)


class TanhSpec(Spec):
    def __init__(self, xsnd=None):
        Spec.__init__(self, xsnd)

    def checked_build(self, x_sig):
        return TanhLayer(x_sig)
