from ..base.layer import XXYLayer
from ..base.spec import XXYSpec


class SequenceLayer(XXYLayer):
    def __init__(self, x_sigs, y_sigs, layers):
        XXYLayer.__init__(self, x_sigs, y_sigs)
        self._layers = layers

    def params(self):
        params = []
        for layer in self._layers:
            params += layer.params()
        return params

    def forward_xx_y(self, xx, is_training):
        for layer in self._layers:
            xx = layer.forward(xx, is_training)
        assert len(xx) == 1
        x, = xx
        return x


class SequenceSpec(XXYSpec):
    def __init__(self, specs):
        XXYSpec.__init__(self, None)
        self._specs = specs

    def build_xx_y(self, x_sigs=None):
        layers = []
        sigs = x_sigs
        for spec in self._specs:
            layer = spec.build(sigs)
            layers.append(layer)
            sigs = layer.y_sigs()
            assert len(sigs) == 1
        y_sigs = sigs
        y_sig, = y_sigs
        return SequenceLayer(x_sigs, y_sig, layers)
