from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class FlattenLayer(Layer):
    def __init__(self, x_sig, y_sig):
        Layer.__init__(self, x_sig, y_sig)

    def forward(self, x, is_training):
        return Z.flatten_batch(x)


class FlattenSpec(Spec):
    def __init__(self, space=None):
        Spec.__init__(self, space)

    def checked_build(self, x_sig):
        y_sig = Z.flatten_batch_signature(x_sig)
        return FlattenLayer(x_sig, y_sig)
