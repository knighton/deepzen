from ..base.layer import Layer
from ..base.signature import Signature
from ..base.spec import Spec


class DataLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig, sig)

    def forward(self, x, is_training):
        self._x_sig.check(x)
        return x


class DataSpec(Spec):
    def __init__(self, shape, dtype):
        self._required_sig = Signature(shape, dtype)

    def build(self, x_sig=None):
        if x_sig is not None:
            assert self._required_sig.equals(x_sig)
        return DataLayer(self._required_sig)
