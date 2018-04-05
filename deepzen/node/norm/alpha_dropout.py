from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class AlphaDropoutLayer(XYLayer):
    def __init__(self, sig, rate, keep_spatial_axis):
        XYLayer.__init__(self, sig)
        self._rate = rate
        self._keep_spatial_axis = keep_spatial_axis

    def forward_x_y(self, x, is_training):
        return Z.alpha_dropout(x, is_training, self._rate,
                               self._keep_spatial_axis)


class AlphaDropoutSpec(XYSpec):
    def __init__(self, rate=0.5, keep_spatial_axis=None, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._rate = rate
        self._keep_spatial_axis = keep_spatial_axis

    def build_x_y(self, x_sig):
        return AlphaDropoutLayer(x_sig, self._rate, self._keep_spatial_axis)


AlphaDropout, AlphaDropout0D, AlphaDropout1D, AlphaDropout2D, AlphaDropout3D = \
    keywordize(AlphaDropoutSpec, [None, 0, 1, 2, 3])

SpatialAlphaDropout, SpatialAlphaDropout1D, SpatialAlphaDropout2D, \
    SpatialAlphaDropout3D = \
    keywordize(AlphaDropoutSpec, [None, 1, 2, 3], {'keep_spatial_axis': 0})
