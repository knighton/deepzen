class Spec(object):
    def build(self, x_sigs=None):
        raise NotImplementedError


class XYSpec(Spec):
    def __init__(self, xsnd=None):
        assert xsnd in {None, 0, 1, 2, 3}
        self._req_x_spatial_ndim = xsnd

    def req_x_spatial_ndim(self):
        return self._req_x_spatial_ndim

    def build_x_y(self, x_sig=None):
        raise NotImplementedError

    def build(self, x_sigs=None):
        if x_sigs is None:
            x_sig = None
        else:
            assert len(x_sigs) == 1
            x_sig, = x_sigs
        if self._req_x_spatial_ndim is not None and x_sig is not None:
            assert self._req_x_spatial_ndim == x_sig.spatial_ndim()
        return self.build_x_y(x_sig)


class XXYSpec(Spec):
    def __init__(self, xsnd=None):
        assert xsnd in {None, 0, 1, 2, 3}
        self._req_xx_spatial_ndim = xsnd

    def req_xx_spatial_ndim(self):
        return self._req_xx_spatial_ndim

    def build_xx_y(self, x_sigs=None):
        raise NotImplementedError

    def build(self, x_sigs=None):
        if self._req_xx_spatial_ndim is not None:
            for x_sig in x_sigs:
                if x_sig is None:
                    continue
                assert self._req_spatial_ndim == x_sig.spatial_ndim()
        return self.build_xx_y(x_sigs)
