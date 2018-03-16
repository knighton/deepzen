from ..base.layer import Layer
from ..base.spec import Spec


class SequenceLayer(Layer):
    def __init__(self, layers):
        Layer.__init__(self)
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

    def build(self, sig=None):
        layers = []
        for spec in self.specs:
            layer, sig = spec.build(sig)
            layers.append(layer)
        return SequenceLayer(layers), sig
