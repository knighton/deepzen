from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class LeakyReLULayer(Layer):
    def __init__(self, sig, alpha=0.1):
        Layer.__init__(self, sig)
        self._alpha = alpha

    def forward(self, x, is_training):
        return Z.leaky_relu(x, self._alpha)


class LeakyReLUSpec(Spec):
    def __init__(self, alpha=0.1, space=None):
        Spec.__init__(self, space)
        self._alpha = alpha

    def checked_build(self, x_sig):
        return LeakyReLULayer(x_sig, self._alpha)
