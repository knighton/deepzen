from ..base.layer import Layer
from ..base.spec import Spec


class SequenceLayer(Layer):
    def __init__(self, x_sig, y_sig, layers):
        Layer.__init__(self, x_sig, y_sig)
        self._layers = layers

    def params(self):
        params = []
        for layer in self._layers:
            params += layer.params()
        return params

    def forward(self, x, is_training):
        for layer in self._layers:
            x = layer.forward(x, is_training)
        return x


class SequenceSpec(Spec):
    def __init__(self, specs):
        Spec.__init__(self, None)
        self._specs = specs

    def checked_build(self, x_sig=None):
        layers = []
        sig = x_sig
        for spec in self._specs:
            layer = spec.build(sig)
            layers.append(layer)
            sig = layer.y_sig()
        y_sig = sig
        return SequenceLayer(x_sig, y_sig, layers)
