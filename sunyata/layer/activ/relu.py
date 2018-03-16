from ... import api as Z
from ..base.layer import Layer
from ..base.spec import Spec


class ReLULayer(Layer):
    def forward(self, x, is_training):
        return Z.relu(x)


class ReLUSpec(Spec):
    def build(self, form=None):
        return ReLULayer(), form
