from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class ReshapeLayer(Layer):
    def __init__(self, x_sig, y_sig, shape):
        Layer.__init__(x_sig, y_sig)
        self._shape = shape

    def forward(self, x, is_training):
        return Z.reshape_batch(x, self._shape)


class ReshapeSpec(Spec):
    def __init__(self, shape, xsnd=None):
        Spec.__init__(self, xsnd)
        self._shape = shape

    def checked_build(self, x_sig):
        y_sig = Z.reshape_batch_signature(x_sig, self._shape)
        return ReshapeLayer(x_sig, y_sig, self._shape)
