from ... import api as Z
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class GlobalPoolLayer(XYLayer):
    def __init__(self, global_pool, x_sig, y_sig):
        XYLayer.__init__(self, x_sig, y_sig)
        self._global_pool = global_pool

    def forward_x_y(self, x, is_training):
        return self._global_pool(x)


class GlobalPoolSpec(XYSpec):
    def __init__(self, layer_class, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._layer_class = layer_class

    def build_x_y(self, x_sig):
        y_sig = Z.global_pool_signature(x_sig)
        return self._layer_class(x_sig, y_sig)


class GlobalAvgPoolLayer(GlobalPoolLayer):
    def __init__(self, x_sig, y_sig):
        GlobalPoolLayer.__init__(self, Z.global_avg_pool, x_sig, y_sig)


class GlobalAvgPoolSpec(GlobalPoolSpec):
    def __init__(self, xsnd=None):
        GlobalPoolSpec.__init__(self, GlobalAvgPoolLayer, xsnd)


class GlobalMaxPoolLayer(GlobalPoolLayer):
    def __init__(self, x_sig, y_sig):
        GlobalPoolLayer.__init__(self, Z.global_max_pool, x_sig, y_sig)


class GlobalMaxPoolSpec(GlobalPoolSpec):
    def __init__(self, xsnd=None):
        GlobalPoolSpec.__init__(self, GlobalMaxPoolLayer, xsnd)
