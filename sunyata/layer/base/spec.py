class Spec(object):
    def __init__(self, space=None):
        assert space in {None, 0, 1, 2, 3}
        self._required_x_spatial_ndim = space

    def checked_build(self, x_sig=None):
        raise NotImplementedError

    def build(self, x_sig=None):
        if x_sig and self._required_x_spatial_ndim is not None:
            assert x_sig.spatial_ndim() == self._required_x_spatial_ndim
        return self.checked_build(x_sig)
