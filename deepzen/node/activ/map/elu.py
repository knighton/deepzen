from .... import api as Z
from ...base.keyword import keywordize
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class ELULayer(XYLayer):
    def __init__(self, sig):
        XYLayer.__init__(self, sig)

    def forward_x_y(self, x, is_training):
        return Z.elu(x)


class ELUSpec(XYSpec):
    def __init__(self, xsnd=None):
        XYSpec.__init__(self, xsnd)

    def build_x_y(self, x_sig):
        return ELULayer(x_sig)


ELU = keywordize(ELUSpec)
