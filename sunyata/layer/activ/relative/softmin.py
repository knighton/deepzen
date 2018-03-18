from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class SoftminLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig, sig)

    def forward(self, x, is_training):
        return Z.softmin(x)


class SoftminSpec(Spec):
    def __init__(self, space=None):
        Spec.__init__(self, space)

    def checked_build(self, x_sig):
        return SoftminLayer(x_sig)