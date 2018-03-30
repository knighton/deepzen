from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class SwishLayer(Layer):
    def __init__(self, sig, beta=1):
        Layer.__init__(self, sig)
        self._beta = beta

    def forward(self, x, is_training):
        return Z.swish(x, self._beta)


class SwishSpec(Spec):
    def __init__(self, beta=1, space=None):
        Spec.__init__(self, space)
        self._beta = beta

    def checked_build(self, x_sig):
        return SwishLayer(x_sig, self._beta)
