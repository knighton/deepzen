from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class LogSoftmaxLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig)

    def forward(self, x, is_training):
        return Z.log_softmax(x)


class LogSoftmaxSpec(Spec):
    def __init__(self, xsnd=None):
        Spec.__init__(self, xsnd)

    def checked_build(self, x_sig):
        return LogSoftmaxLayer(x_sig)
