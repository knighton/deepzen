from .... import api as Z
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class HardTanhLayer(XYLayer):
    def __init__(self, sig):
        XYLayer.__init__(self, sig)

    def forward_x_y(self, x, is_training):
        return Z.hard_tanh(x)


class HardTanhSpec(XYSpec):
    def __init__(self, xsnd=None):
        XYSpec.__init__(self, xsnd)

    def build_x_y(self, x_sig):
        return HardTanhLayer(x_sig)
