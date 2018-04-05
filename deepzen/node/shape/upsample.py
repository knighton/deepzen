from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class UpsampleLayer(XYLayer):
    def __init__(self, x_sig, y_sig, upsample, scale):
        XYLayer.__init__(self, x_sig, y_sig)
        self._upsample = upsample
        self._scale = scale

    def forward_x_y(self, x, is_training):
        return self._upsample(x, self._scale)


class UpsampleSpec(XYSpec):
    def __init__(self, layer_class, scale, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._layer_class = layer_class
        self._scale = scale

    def build_x_y(self, x_sig):
        y_sig = Z.upsample_signature(x_sig, self._scale)
        return self._layer_class(x_sig, y_sig, self._scale)


class LinearUpsampleLayer(UpsampleLayer):
    def __init__(self, x_sig, y_sig, scale):
        UpsampleLayer.__init__(self, x_sig, y_sig, Z.linear_upsample, scale)


class LinearUpsampleSpec(UpsampleSpec):
    def __init__(self, scale, xsnd=None):
        UpsampleSpec.__init__(self, LinearUpsampleLayer, scale, xsnd)


LinearUpsample, LinearUpsample1D, LinearUpsample2D, LinearUpsample3D = \
    keywordize(LinearUpsampleSpec, [None, 1, 2, 3])


class NearestUpsampleLayer(UpsampleLayer):
    def __init__(self, x_sig, y_sig, scale):
        UpsampleLayer.__init__(self, x_sig, y_sig, Z.nearest_upsample, scale)


class NearestUpsampleSpec(UpsampleSpec):
    def __init__(self, scale, xsnd=None):
        UpsampleSpec.__init__(self, NearestUpsampleLayer, scale, xsnd)


NearestUpsample, NearestUpsample1D, NearestUpsample2D, NearestUpsample3D = \
    keywordize(NearestUpsampleSpec, [None, 1, 2, 3])
