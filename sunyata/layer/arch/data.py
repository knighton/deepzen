from ..base.layer import Layer
from ..base.signature import Signature
from ..base.spec import Spec


class DataLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self)
        self.sig = sig

    def forward(self, x, is_training):
        self.sig.check(x)
        return x


class DataSpec(Spec):
    def __init__(self, shape, dtype):
        self.sig = Signature(shape, dtype)

    def build(self, sig=None):
        if sig is None:
            sig = self.sig
        else:
            assert self.sig.equals(sig)
        return DataLayer(sig), sig
