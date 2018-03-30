from ... import api as Z
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class GaussianNoiseLayer(XYLayer):
    def __init__(self, sig, std, axis):
        XYLayer.__init__(self, sig)
        self._std = std
        self._axis = axis

    def forward_x_y(self, x, is_training):
        return Z.gaussian_noise(x, is_training, self._std, self._axis)


class GaussianNoiseSpec(XYSpec):
    def __init__(self, std, axis=None, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._std = std
        self._axis = axis

    def build_x_y(self, x_sig):
        return GaussianNoiseLayer(x_sig, self._std, self._axis)
