from .... import api as Z
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class SoftminLayer(XYLayer):
    def __init__(self, sig):
        XYLayer.__init__(self, sig)

    def forward_x_y(self, x, is_training):
        return Z.softmin(x)


class SoftminSpec(XYSpec):
    def __init__(self, xsnd=None):
        XYSpec.__init__(self, xsnd)

    def build_x_y(self, x_sig):
        return SoftminLayer(x_sig)
