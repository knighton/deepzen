from ... import api as Z
from ..base.layer import Layer
from ..base.signature import Signature
from ..base.spec import Spec


class DataLayer(Layer):
    def __init__(self, sig):
        Layer.__init__(self, sig)

    def forward(self, x, is_training):
        self._x_sig.accepts_batch_tensor(x)
        return x


class DataSpec(Spec):
    def __init__(self, sample_shape, dtype, has_channels=None):
        if has_channels is None:
            dtype = Z.dtype(dtype)
            if dtype.startswith('float'):
                has_channels = True
            else:
                has_channels = False
        required_sig = Signature(sample_shape, dtype, has_channels)
        Spec.__init__(self, required_sig.spatial_ndim_or_none())
        self._required_sig = required_sig

    def checked_build(self, x_sig=None):
        if x_sig is not None:
            assert self._required_sig == x_sig
        return DataLayer(self._required_sig)
