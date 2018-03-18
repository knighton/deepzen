from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class SoftExponentialLayer(Layer):
    def __init__(self, sig, alpha=0.25):
        Layer.__init__(self, sig, sig)
        self._alpha = alpha

    def forward(self, x, is_training):
        return Z.soft_exponential(x, self._alpha)


class SoftExponentialSpec(Spec):
    def __init__(self, alpha=0.25, space=None):
        Spec.__init__(self, space)
        self._alpha = alpha

    def checked_build(self, x_sig):
        return SoftExponentialLayer(x_sig, self._alpha)
