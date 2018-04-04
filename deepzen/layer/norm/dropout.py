from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class DropoutLayer(XYLayer):
    def __init__(self, sig, rate, axis):
        XYLayer.__init__(self, sig)
        self._rate = rate
        self._axis = axis

    def forward_x_y(self, x, is_training):
        return Z.dropout(x, is_training, self._rate, self._axis)


class DropoutSpec(XYSpec):
    def __init__(self, rate=0.5, axis=None, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._rate = rate
        self._axis = axis

    def build_x_y(self, x_sig):
        return DropoutLayer(x_sig, self._rate, self._axis)


Dropout, Dropout0D, Dropout1D, Dropout2D, Dropout3D = \
    keywordize(DropoutSpec, [None, 0, 1, 2, 3])
