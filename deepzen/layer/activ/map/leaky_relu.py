from .... import api as Z
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class LeakyReLULayer(XYLayer):
    def __init__(self, sig, alpha=0.1):
        XYLayer.__init__(self, sig)
        self._alpha = alpha

    def forward_x_y(self, x, is_training):
        return Z.leaky_relu(x, self._alpha)


class LeakyReLUSpec(XYSpec):
    def __init__(self, alpha=0.1, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._alpha = alpha

    def build_x_y(self, x_sig):
        return LeakyReLULayer(x_sig, self._alpha)
