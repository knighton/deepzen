from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class ChannelPoolLayer(XYLayer):
    def __init__(self, x_sig, y_sig, channel_pool, face):
        XYLayer.__init__(self, x_sig, y_sig)
        self._channel_pool = channel_pool
        self._face = face

    def forward_x_y(self, x, is_training):
        return self._channel_pool(x, self._face)


class ChannelPoolSpec(XYSpec):
    def __init__(self, layer_class, face=2, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._layer_class = layer_class
        self._face = face
        self._xsnd = xsnd

    def build_x_y(self, x_sig):
        y_sig = Z.channel_pool_signature(x_sig, self._face)
        return self._layer_class(x_sig, y_sig, self._face)


class ChannelAvgPoolLayer(ChannelPoolLayer):
    def __init__(self, x_sig, y_sig, face):
        ChannelPoolLayer.__init__(self, x_sig, y_sig, Z.channel_avg_pool, face)


class ChannelAvgPoolSpec(ChannelPoolSpec):
    def __init__(self, face=2, xsnd=None):
        ChannelPoolSpec.__init__(self, ChannelAvgPoolLayer, face, xsnd)


ChannelAvgPool, ChannelAvgPool1D, ChannelAvgPool2D, ChannelAvgPool3D = \
    keywordize(ChannelAvgPoolSpec, [None, 1, 2, 3])


class ChannelMaxPoolLayer(ChannelPoolLayer):
    def __init__(self, x_sig, y_sig, face):
        ChannelPoolLayer.__init__(self, x_sig, y_sig, Z.channel_max_pool, face)


class ChannelMaxPoolSpec(ChannelPoolSpec):
    def __init__(self, face=2, xsnd=None):
        ChannelPoolSpec.__init__(self, ChannelMaxPoolLayer, face, xsnd)


ChannelMaxPool, ChannelMaxPool1D, ChannelMaxPool2D, ChannelMaxPool3D = \
    keywordize(ChannelMaxPoolSpec, [None, 1, 2, 3])
