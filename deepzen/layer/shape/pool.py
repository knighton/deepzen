from ... import api as Z
from ..base.layer import Layer
from ..base.spec import Spec


class PoolLayer(Layer):
    def __init__(self, x_sig, y_sig, pool, face=2, stride=None, padding=0):
        Layer.__init__(self, x_sig, y_sig)
        self._pool = pool
        self._face = face
        self._stride = stride
        self._padding = padding

    def forward(self, x, is_training):
        return self._pool(x, self._face, self._stride, self._padding)


class PoolSpec(Spec):
    def __init__(self, layer_class, face=2, stride=None, padding=0, xsnd=None):
        Spec.__init__(self, xsnd)
        self._layer_class = layer_class
        self._face = face
        self._stride = stride
        self._padding = padding

    def checked_build(self, x_sig):
        y_sig = Z.pool_signature(x_sig, self._face, self._stride, self._padding)
        return self._layer_class(x_sig, y_sig, self._face, self._stride,
                                 self._padding)


class AvgPoolLayer(PoolLayer):
    def __init__(self, x_sig, y_sig, face=2, stride=None, padding=0):
        PoolLayer.__init__(self, x_sig, y_sig, Z.avg_pool, face, stride,
                           padding)


class AvgPoolSpec(PoolSpec):
    def __init__(self, face=2, stride=None, padding=0, xsnd=None):
        PoolSpec.__init__(self, AvgPoolLayer, face, stride, padding, xsnd)


class MaxPoolLayer(PoolLayer):
    def __init__(self, x_sig, y_sig, face=2, stride=None, padding=0):
        PoolLayer.__init__(self, x_sig, y_sig, Z.max_pool, face, stride,
                           padding)


class MaxPoolSpec(PoolSpec):
    def __init__(self, face=2, stride=None, padding=0, xsnd=None):
        PoolSpec.__init__(self, MaxPoolLayer, face, stride, padding, xsnd)
