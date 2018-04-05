from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.spec import XYSpec


class PadLayer(XYLayer):
    def __init__(self, x_sig, y_sig, padding):
        XYLayer.__init__(self, x_sig, y_sig, padding)
        self._padding = padding


class PadSpec(XYSpec):
    def __init__(self, padding, xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._padding = padding

    def make_layer(self, x_sig, y_sig):
        raise NotImplementedError

    def build_x_y(self, x_sig):
        y_sig = Z.pad_signature(x_sig, self._padding)
        return self.make_layer(x_sig, y_sig)


class ConstantPadLayer(PadLayer):
    def __init__(self, x_sig, y_sig, padding, value):
        PadLayer.__init__(self, x_sig, y_sig, padding)
        self._value = value

    def forward_x_y(self, x, is_training):
        return Z.constant_pad(x, self._padding, self._value)


class ConstantPadSpec(PadSpec):
    def __init__(self, padding, value, xsnd=None):
        PadSpec.__init__(self, padding, xsnd)
        self._value = value

    def make_layer(self, x_sig, y_sig):
        return ConstantPadLayer(x_sig, y_sig, self._padding, self._value)


ConstantPad, ConstantPad1D, ConstantPad2D, ConstantPad3D = \
    keywordize(ConstantPadSpec, [None, 1, 2, 3])


class EdgePadLayer(PadLayer):
    def __init__(self, x_sig, y_sig, padding):
        PadLayer.__init__(self, x_sig, y_sig, padding)

    def forward_x_y(self, x, is_training):
        return Z.edge_pad(x, self._padding)


class EdgePadSpec(PadSpec):
    def __init__(self, padding, xsnd=None):
        PadSpec.__init__(self, padding, xsnd)

    def make_layer(self, x_sig, y_sig):
        return EdgePadLayer(x_sig, y_sig, self._padding)


EdgePad, EdgePad1D, EdgePad2D, EdgePad3D = \
    keywordize(EdgePadSpec, [None, 1, 2, 3])


class ReflectPadLayer(PadLayer):
    def __init__(self, x_sig, y_sig, padding):
        PadLayer.__init__(self, x_sig, y_sig, padding)

    def forward_x_y(self, x, is_training):
        return Z.reflect_pad(x, self._padding)


class ReflectPadSpec(PadSpec):
    def __init__(self, padding, xsnd=None):
        PadSpec.__init__(self, padding, xsnd)

    def make_layer(self, x_sig, y_sig):
        return ReflectPadLayer(x_sig, y_sig, self._padding)


ReflectPad, ReflectPad1D, ReflectPad2D, ReflectPad3D = \
    keywordize(ReflectPadSpec, [None, 1, 2, 3])
