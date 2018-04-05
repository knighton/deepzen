from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.signature import Signature
from ..base.spec import XYSpec


class DataLayer(XYLayer):
    def __init__(self, sig):
        XYLayer.__init__(self, sig)

    def forward_x_y(self, x, is_training):
        x_sig, = self._x_sigs
        assert x_sig.accepts_batch_tensor(x)
        return x


class DataSpec(XYSpec):
    def __init__(self, sample_shape, dtype, has_channels=None):
        if has_channels is None:
            dtype = Z.dtype(dtype)
            if dtype.startswith('float'):
                has_channels = True
            else:
                has_channels = False
        required_sig = Signature(sample_shape, dtype, has_channels)
        XYSpec.__init__(self, required_sig.spatial_ndim_or_none())
        self._required_sig = required_sig

    def build_x_y(self, x_sig=None):
        if x_sig is not None:
            assert self._required_sig == x_sig
        return DataLayer(self._required_sig)


Data = keywordize(DataSpec)
