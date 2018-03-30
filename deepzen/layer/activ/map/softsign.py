from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class SoftsignLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig)

    def forward(self, x, is_training):
        return Z.softsign(x)


class SoftsignSpec(Spec):
    def __init__(self, space=None):
        Spec.__init__(self, space)

    def checked_build(self, x_sig):
        return SoftsignLayer(x_sig)
