from ... import api as Z


class Layer(object):
    """
    Object that pairs a tensor transformation with the state it uses.
    """

    def __init__(self, x_sigs=None, y_sigs=None):
        if x_sigs is None:
            xsnd = None
        else:
            assert isinstance(x_sigs, list)
            xsnds = []
            for x_sig in x_sigs:
                xsnd = x_sig.spatial_ndim_or_none()
                if xsnd is not None:
                    xsnds.append(xsnd)
            if xsnds:
                assert len(set(xsnds)) == 1
                xsnd = xsnds[0]
            else:
                xsnd = None
        if y_sigs is None:
            y_sigs = x_sigs
        self._x_sigs = x_sigs
        self._y_sigs = y_sigs
        self._params = []

    def x_sigs(self):
        return self._x_sigs

    def y_sigs(self):
        return self._y_sigs

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

    def forward(self, xx, is_training):
        raise NotImplementedError


class XYLayer(Layer):
    """
    Layer that transforms one input into one output.
    """

    def __init__(self, x_sig, y_sig=None):
        x_sigs = [x_sig]
        if y_sig is None:
            y_sigs = None
        else:
            y_sigs = [y_sig]
        Layer.__init__(self, x_sigs, y_sigs)

    def forward_x_y(self, x, is_training):
        raise NotImplementedError

    def forward(self, xx, is_training):
        assert len(xx) == 1
        x, = xx
        y = self.forward_x_y(x, is_training)
        return [y]


class XXYLayer(Layer):
    """
    Layer that transforms multiple inputs into one output.
    """

    def __init__(self, x_sigs, y_sig):
        y_sigs = [y_sig]
        Layer.__init__(self, x_sigs, y_sigs)

    def forward_xx_y(self, x, is_training):
        raise NotImplementedError

    def forward(self, xx, is_training):
        y = self.forward_xx_y(xx, is_training)
        return [y]
