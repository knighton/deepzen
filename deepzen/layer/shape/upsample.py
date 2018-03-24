from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class UpsampleLayer(Layer):
    def __init__(self, x_sig, y_sig, upsample, scale):
        Layer.__init__(self, x_sig, y_sig)
        self._upsample = upsample
        self._scale = scale

    def forward(self, x, is_training):
        return self._upsample(x, self._scale)


class UpsampleSpec(Spec):
    def __init__(self, layer_class, scale, space=None):
        Spec.__init__(self, space)
        self._layer_class = layer_class
        self._scale = scale

    def checked_build(self, x_sig):
        y_sig = Z.upsample_signature(x_sig, self._scale)
        return self._layer_class(x_sig, y_sig, self._scale)


class LinearUpsampleLayer(UpsampleLayer):
    def __init__(self, x_sig, y_sig, scale):
        UpsampleLayer.__init__(self, x_sig, y_sig, Z.linear_upsample, scale)


class LinearUpsampleSpec(UpsampleSpec):
    def __init__(self, scale, space=None):
        UpsampleSpec.__init__(self, LinearUpsampleLayer, scale, space)


class NearestUpsampleLayer(UpsampleLayer):
    def __init__(self, x_sig, y_sig, scale):
        UpsampleLayer.__init__(self, x_sig, y_sig, Z.nearest_upsample, scale)


class NearestUpsampleSpec(UpsampleSpec):
    def __init__(self, scale, space=None):
        UpsampleSpec.__init__(self, NearestUpsampleLayer, scale, space)
