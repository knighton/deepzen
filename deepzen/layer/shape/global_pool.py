from ... import api as Z
from ..base.layer import Layer
from ..base.spec import Spec


class GlobalPoolLayer(Layer):
    def __init__(self, global_pool, x_sig, y_sig):
        Layer.__init__(self, x_sig, y_sig)
        self._global_pool = global_pool

    def forward(self, x, is_training):
        return self._global_pool(x)


class GlobalPoolSpec(Spec):
    def __init__(self, layer_class, xsnd=None):
        Spec.__init__(self, xsnd)
        self._layer_class = layer_class

    def checked_build(self, x_sig):
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
