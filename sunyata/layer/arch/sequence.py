from ..base.layer import Layer
from ..base.spec import Spec


class SequenceLayer(Layer):
    def __init__(self, x_sig, y_sig, layers):
        Layer.__init__(self, x_sig, y_sig)
        self.layers = layers

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def forward(self, x, is_training):
        for layer in self.layers:
            x = layer.forward(x, is_training)
        return x


class SequenceSpec(Spec):
    def __init__(self, specs):
        self.specs = specs

    def build(self, x_sig=None):
        layers = []
        sig = x_sig
        for spec in self.specs:
            layer = spec.build(sig)
            layers.append(layer)
            sig = layer.y_sig()
        y_sig = sig
        return SequenceLayer(x_sig, y_sig, layers)
