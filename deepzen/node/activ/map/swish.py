from .... import api as Z
from ...base.keyword import keywordize
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class SwishLayer(XYLayer):
    def __init__(self, sig, beta=1):
        XYLayer.__init__(self, sig)
        self._beta = beta

    def forward_x_y(self, x, is_training):
        return Z.swish(x, self._beta)


class SwishSpec(XYSpec):
    def __init__(self, beta=1, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._beta = beta

    def build_x_y(self, x_sig):
        return SwishLayer(x_sig, self._beta)


Swish = keywordize(SwishSpec)
