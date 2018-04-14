from ... import api as Z
from ..base.node import Node
from ..base.signature import Signature


class Input(Node):
    """
    A placeholder for an input tensor in a network.

    If not a source node of a network, it just validates shape/dtype of the
    tensor passing through.
    """

    def __init__(self, sample_shape, dtype, has_channels=None,
                 preds_via_call=None):
        assert isinstance(sample_shape, tuple)
        for dim in sample_shape:
            assert isinstance(dim, int)
            assert 1 <= dim
        assert dtype
        assert isinstance(dtype, str)
        assert has_channels in {None, False, True}

        Node.__init__(self, preds_via_call)

        if has_channels is None:
            dtype = Z.dtype(dtype)
            if dtype.startswith('float'):
                has_channels = True
            else:
                has_channels = False

        self._required_sig = Signature(sample_shape, dtype, has_channels)

    def __call__(self, *preds):
        """
        Inherited from Node.
        """
        assert preds
        sig = self._required_sig
        sample_shape = tuple(sig.sample_shape())
        dtype = sig.dtype()
        has_channels = sig.has_channels()
        return Input(sample_shape, dtype, has_channels, preds)

    def sub_build(self, x_sigs):
        """
        Inherited from Node.
        """
        if x_sigs:
            assert len(x_sigs) == 1
            x_sig, = x_sigs
            assert self._required_sig == x_sig
        return [self._required_sig]

    def sub_params(self, nodes_seen, params_seen, params):
        """
        Inherited from Node.
        """
        pass

    def sub_forward(self, xx, is_training):
        """
        Inherited from Node.
        """
        assert len(xx) == 1
        x, = xx
        assert self._required_sig.accepts_batch_tensor(x)
        return xx
