class Spec(object):
    def __init__(self, xsnd=None):
        assert xsnd in {None, 0, 1, 2, 3}
        self._req_x_spatial_ndim = xsnd

    def req_x_spatial_ndim(self):
        return self._req_x_spatial_ndim

    def checked_build(self, x_sig=None):
        raise NotImplementedError

    def build(self, x_sig=None):
        if x_sig and self._req_x_spatial_ndim is not None:
            assert x_sig.spatial_ndim() == self._req_x_spatial_ndim
        return self.checked_build(x_sig)
