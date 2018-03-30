from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class TanhShrinkLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig)

    def forward(self, x, is_training):
        return Z.tanh_shrink(x)


class TanhShrinkSpec(Spec):
    def __init__(self, xsnd=None):
        Spec.__init__(self, xsnd)

    def checked_build(self, x_sig):
        return TanhShrinkLayer(x_sig)
