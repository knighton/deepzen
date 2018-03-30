from ... import api as Z
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class AlphaDropoutLayer(XYLayer):
    def __init__(self, sig, rate, axis):
        XYLayer.__init__(self, sig)
        self._rate = rate
        self._axis = axis

    def forward_x_y(self, x, is_training):
        return Z.alpha_dropout(x, is_training, self._rate, self._axis)


class AlphaDropoutSpec(XYSpec):
    def __init__(self, rate=0.5, axis=None, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._rate = rate
        self._axis = axis

    def build_x_y(self, x_sig):
        return AlphaDropoutLayer(x_sig, self._rate, self._axis)
