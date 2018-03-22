from ... import api as Z
from ...dist import unpack_distribution
from ..base.layer import Layer
from ..base.signature import Signature
from ..base.spec import Spec


class EmbedLayer(Layer):
    def __init__(self, x_sig, y_sig, table):
        Layer.__init__(self, x_sig, y_sig)
        self._table = self.param(table)

    def forward(self, x, is_training):
        return Z.embed(x, self._table)


class EmbedSpec(Spec):
    def __init__(self, vocab, dim, dtype=None, table_init='uniform',
                 space=None):
        Spec.__init__(self, space)
        self._vocab = vocab
        self._dim = dim
        self._dtype = dtype
        self._table_init = unpack_distribution(table_init)

    def checked_build(self, x_sig):
        assert not x_sig.has_channels()
        table_shape = self._vocab, self._dim
        y_dtype = Z.dtype(self._dtype)
        table = self._table_init(table_shape, y_dtype)
        y_shape = (self._dim,) + x_sig.sample_shape()
        y_sig = Signature(y_shape, y_dtype, True)
        return EmbedLayer(x_sig, y_sig, table)