from .... import api as Z
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class HardShrinkLayer(XYLayer):
    def __init__(self, sig, lam=0.5):
        XYLayer.__init__(self, sig)
        self._lambda = lam

    def forward_x_y(self, x, is_training):
        return Z.hard_shrink(x, self._lambda)


class HardShrinkSpec(XYSpec):
    def __init__(self, lam=0.5, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._lambda = lam

    def build_x_y(self, x_sig):
        return HardShrinkLayer(x_sig, self._lambda)
