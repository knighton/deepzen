from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class HardShrinkLayer(Layer):
    def __init__(self, sig, lam=0.5):
        Layer.__init__(self, sig, sig)
        self._lambda = lam

    def forward(self, x, is_training):
        return Z.hard_shrink(x, self._lambda)


class HardShrinkSpec(Spec):
    def __init__(self, lam=0.5, space=None):
        Spec.__init__(self, space)
        self._lambda = lam

    def checked_build(self, x_sig):
        return HardShrinkLayer(x_sig, self._lambda)
