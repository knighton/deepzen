from .... import api as Z
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class SoftplusLayer(XYLayer):
    def __init__(self, sig, beta=1, threshold=20):
        XYLayer.__init__(self, sig)
        self._beta = beta
        self._threshold = threshold

    def forward_x_y(self, x, is_training):
        return Z.softplus(x, self._beta, self._threshold)


class SoftplusSpec(XYSpec):
    def __init__(self, beta=1, threshold=20, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._beta = beta
        self._threshold = threshold

    def build_x_y(self, x_sig):
        return SoftplusLayer(x_sig, self._beta, self._threshold)
