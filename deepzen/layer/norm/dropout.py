from ... import api as Z
from ..base.layer import Layer
from ..base.spec import Spec


class DropoutLayer(Layer):
    def __init__(self, sig, rate, axis):
        Layer.__init__(self, sig)
        self._rate = rate
        self._axis = axis

    def forward(self, x, is_training):
        return Z.dropout(x, is_training, self._rate, self._axis)


class DropoutSpec(Spec):
    def __init__(self, rate=0.5, axis=None, space=None):
        Spec.__init__(self, space)
        self._rate = rate
        self._axis = axis

    def checked_build(self, x_sig):
        return DropoutLayer(x_sig, self._rate, self._axis)
