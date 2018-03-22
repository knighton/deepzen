import numpy as np

from .... import api as Z
from ...base.layer import Layer
from ...base.spec import Spec


class ClipLayer(Layer):
    def __init__(self, sig, min=-np.inf, max=np.inf):
        Layer.__init__(self, sig, sig)
        self._min = min
        self._max = max

    def forward(self, x, is_training):
        return Z.clip(x, min, max)


class ClipSpec(Spec):
    def __init__(self, min=-np.inf, max=np.inf, space=None):
        Spec.__init__(self, space)
        self._min = min
        self._max = max

    def checked_build(self, x_sig):
        return ClipLayer(x_sig, self._min, self._max)
