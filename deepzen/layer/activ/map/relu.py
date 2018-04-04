from .... import api as Z
from ...base.keyword import keywordize
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class ReLULayer(XYLayer):
    def __init__(self, sig):
        XYLayer.__init__(self, sig)

    def forward_x_y(self, x, is_training):
        return Z.relu(x)


class ReLUSpec(XYSpec):
    def __init__(self, xsnd=None):
        XYSpec.__init__(self, xsnd)

    def build_x_y(self, x_sig):
        return ReLULayer(x_sig)


ReLU = keywordize(ReLUSpec)
