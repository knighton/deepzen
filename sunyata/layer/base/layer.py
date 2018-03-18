from ... import api as Z
from .spec import Spec


class Layer(Spec):
    def __init__(self, x_sig, y_sig):
        if x_sig:
            space = x_sig.spatial_ndim_or_none()
        else:
            space = None
        Spec.__init__(self, space)
        self._x_sig = x_sig
        self._y_sig = y_sig
        self._params = []

    def checked_build(self, x_sig):
        assert self._x_sig == x_sig
        return self

    def x_sig(self):
        return self._x_sig

    def y_sig(self):
        return self._y_sig

    def params(self):
        return self._params

    def param(self, x, learned=True):
        if x is None:
            return None
        if learned:
            x = Z.variable(x)
            self._params.append(x)
        else:
            x = Z.constant(x)
        return x

    def forward(self, x, is_training):
        raise NotImplementedError
