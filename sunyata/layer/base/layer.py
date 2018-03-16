from ... import api as Z
from .spec import Spec


class Layer(Spec):
    def __init__(self, x_sig, y_sig):
        self._x_sig = x_sig
        self._y_sig = y_sig
        self._params = []

    def build(self, x_sig=None):
        assert self._x_sig == x_sig
        return self

    def x_sig(self):
        return self._x_sig

    def y_sig(self):
        return self._y_sig

    def params(self):
        return self._params

    def param(self, x, learned=True):
        if learned:
            x = Z.variable(x)
            self._params.append(x)
        else:
            x = Z.constant(x)
        return x

    def forward(self, x, is_training):
        raise NotImplementedError
