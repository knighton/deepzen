from ... import api as Z
from ..base.layer import Layer
from ..base.spec import Spec


class SoftmaxLayer(Layer):
    def forward(self, x, is_training):
        return Z.softmax(x)


class SoftmaxSpec(Spec):
    def build(self, form=None):
        return SoftmaxLayer(), form
