from .... import api as Z
from ...base.keyword import keywordize
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class ReshapeLayer(XYLayer):
    def __init__(self, x_sig, y_sig, shape):
        XYLayer.__init__(x_sig, y_sig)
        self._shape = shape

    def forward_x_y(self, x, is_training):
        return Z.reshape_batch(x, self._shape)


class ReshapeSpec(XYSpec):
    def __init__(self, shape, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._shape = shape

    def build_x_y(self, x_sig):
        y_sig = Z.reshape_batch_signature(x_sig, self._shape)
        return ReshapeLayer(x_sig, y_sig, self._shape)


Reshape, Reshape1D, Reshape2D, Reshape3D = \
    keywordize(ReshapeSpec, [None, 1, 2, 3])
