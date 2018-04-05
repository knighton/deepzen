import numpy as np

from .... import api as Z
from ...base.keyword import keywordize
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class ClipLayer(XYLayer):
    def __init__(self, sig, min=-np.inf, max=np.inf):
        XYLayer.__init__(self, sig)
        self._min = min
        self._max = max

    def forward_x_y(self, x, is_training):
        return Z.clip(x, min, max)


class ClipSpec(XYSpec):
    def __init__(self, min=-np.inf, max=np.inf, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._min = min
        self._max = max

    def build_x_y(self, x_sig):
        return ClipLayer(x_sig, self._min, self._max)


Clip = keywordize(ClipSpec)
