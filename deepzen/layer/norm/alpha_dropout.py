from ... import api as Z
from ..base.layer import Layer
from ..base.spec import Spec


class AlphaDropoutLayer(Layer):
    def __init__(self, sig, rate, axis):
        Layer.__init__(self, sig)
        self._rate = rate
        self._axis = axis

    def forward(self, x, is_training):
        return Z.alpha_dropout(x, is_training, self._rate, self._axis)


class AlphaDropoutSpec(Spec):
    def __init__(self, rate=0.5, axis=None, xsnd=None):
        Spec.__init__(self, xsnd)
        self._rate = rate
        self._axis = axis

    def checked_build(self, x_sig):
        return AlphaDropoutLayer(x_sig, self._rate, self._axis)
