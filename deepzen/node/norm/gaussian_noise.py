from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class GaussianNoiseLayer(XYLayer):
    def __init__(self, sig, std, keep_spatial_axis):
        XYLayer.__init__(self, sig)
        self._std = std
        self._keep_spatial_axis = keep_spatial_axis

    def forward_x_y(self, x, is_training):
        return Z.gaussian_noise(x, is_training, self._std,
                                self._keep_spatial_axis)


class GaussianNoiseSpec(XYSpec):
    def __init__(self, std, keep_spatial_axis=None, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._std = std
        self._keep_spatial_axis = keep_spatial_axis

    def build_x_y(self, x_sig):
        return GaussianNoiseLayer(x_sig, self._std, self._keep_spatial_axis)


GaussianNoise, GaussianNoise0D, GaussianNoise1D, GaussianNoise2D, \
    GaussianNoise3D = keywordize(GaussianNoiseSpec, [None, 0, 1, 2, 3])

SpatialGaussianNoise, SpatialGaussianNoise1D, SpatialGaussianNoise2D, \
    SpatialGaussianNoise3D = \
    keywordize(GaussianNoiseSpec, [None, 1, 2, 3], {'keep_spatial_axis': 0})
