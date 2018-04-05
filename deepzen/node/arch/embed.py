from ... import api as Z
from ...init import get_initializer
from ..base.keyword import keywordize
from ..base.layer import XYLayer
from ..base.signature import Signature
from ..base.spec import XYSpec


class EmbedLayer(XYLayer):
    def __init__(self, x_sig, y_sig, table):
        XYLayer.__init__(self, x_sig, y_sig)
        self._table = self.param(table)

    def forward_x_y(self, x, is_training):
        return Z.embed(x, self._table)


class EmbedSpec(XYSpec):
    def __init__(self, vocab, dim, dtype=None, table_init='uniform', xsnd=None):
        XYSpec.__init__(self, xsnd)
        self._vocab = vocab
        self._dim = dim
        self._dtype = dtype
        self._table_init = get_initializer(table_init)

    def build_x_y(self, x_sig):
        assert not x_sig.has_channels()
        table_shape = self._vocab, self._dim
        y_dtype = Z.dtype(self._dtype)
        table = self._table_init(table_shape, y_dtype)
        y_shape = (self._dim,) + x_sig.sample_shape()
        y_sig = Signature(y_shape, y_dtype, True)
        return EmbedLayer(x_sig, y_sig, table)


Embed = keywordize(EmbedSpec)
