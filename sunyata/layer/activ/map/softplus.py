from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class SoftplusLayer(Layer):
    def __init__(self, sig, beta=1, threshold=20):
        Layer.__init__(self, sig, sig)
        self._beta = beta
        self._threshold = threshold

    def forward(self, x, is_training):
        return Z.softplus(x, self._beta, self._threshold)


class SoftplusSpec(Spec):
    def __init__(self, beta=1, threshold=20, space=None):
        Spec.__init__(self, space)
        self._beta = beta
        self._threshold = threshold

    def checked_build(self, x_sig):
        return SoftplusLayer(x_sig, self._beta, self._threshold)
