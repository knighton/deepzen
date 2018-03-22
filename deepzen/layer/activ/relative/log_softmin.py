from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class LogSoftminLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig, sig)

    def forward(self, x, is_training):
        return Z.log_softmin(x)


class LogSoftminSpec(Spec):
    def __init__(self, space=None):
        Spec.__init__(self, space)

    def checked_build(self, x_sig):
        return LogSoftminLayer(x_sig)
