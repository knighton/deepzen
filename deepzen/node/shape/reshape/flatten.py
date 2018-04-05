from .... import api as Z
from ...base.keyword import keywordize
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class FlattenLayer(XYLayer):
    def __init__(self, x_sig, y_sig):
        XYLayer.__init__(self, x_sig, y_sig)

    def forward_x_y(self, x, is_training):
        return Z.flatten_batch(x)


class FlattenSpec(XYSpec):
    def __init__(self, xsnd=None):
        XYSpec.__init__(self, xsnd)

    def build_x_y(self, x_sig):
        y_sig = Z.flatten_batch_signature(x_sig)
        return FlattenLayer(x_sig, y_sig)


Flatten, Flatten1D, Flatten2D, Flatten3D = \
    keywordize(FlattenSpec, [None, 1, 2, 3])
