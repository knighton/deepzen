from ... import api as Z
from ..base.layer import Layer
from ..base.spec import Spec


class GaussianNoiseLayer(Layer):
    def __init__(self, sig, std, axis):
        Layer.__init__(self, sig)
        self._std = std
        self._axis = axis

    def forward(self, x, is_training):
        return Z.gaussian_noise(x, is_training, self._std, self._axis)


class GaussianNoiseSpec(Spec):
    def __init__(self, std, axis=None, space=None):
        Spec.__init__(self, space)
        self._std = std
        self._axis = axis

    def checked_build(self, x_sig):
        return GaussianNoiseLayer(x_sig, self._std, self._axis)
