from .... import api as Z
from ...base.keyword import keywordize
from ...base.layer import XYLayer
from ...base.spec import XYSpec


class LogSigmoidLayer(XYLayer):
    def __init__(self, sig):
        XYLayer.__init__(self, sig)

    def forward_x_y(self, x, is_training):
        return Z.log_sigmoid(x)


class LogSigmoidSpec(XYSpec):
    def __init__(self, xsnd=None):
        XYSpec.__init__(self, xsnd)

    def build_x_y(self, x_sig):
        return LogSigmoidLayer(x_sig)


LogSigmoid = keywordize(LogSigmoidSpec)
