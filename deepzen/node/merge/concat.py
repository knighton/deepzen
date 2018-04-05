from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XXYLayer
from ..base.spec import XXYSpec


class ConcatLayer(XXYLayer):
    def __init__(self, x_sigs, y_sig, sample_axis):
        XXYLayer.__init__(self, x_sigs, y_sig)
        self._sample_axis = sample_axis

    def forward_xx_y(self, xx, is_training):
        return Z.merge_concat(xx, self._sample_axis)


class ConcatSpec(XXYSpec):
    def __init__(self, sample_axis=0, xsnd=None):
        assert isinstance(sample_axis, int)
        assert 0 <= sample_axis
        XXYSpec.__init__(self, xsnd)
        self._sample_axis = sample_axis

    def build_xx_y(self, x_sigs=None):
        assert x_sigs
        first_x_sig = x_sigs[0]
        match_sample_shape = list(first_x_sig.sample_shape())
        assert self._sample_axis < len(match_sample_shape)
        concat_dim = match_sample_shape[self._sample_axis]
        match_sample_shape[self._sample_axis] = 0
        for x_sig in x_sigs[1:]:
            assert x_sig.dtype() == first_x_sig.dtype()
            sample_shape = list(x_sig.sample_shape())
            concat_dim += sample_shape[self._sample_axis]
            sample_shape[self._sample_axis] = 0
            assert match_sample_shape == sample_shape
        y_sample_shape = match_sample_shape
        y_sample_shape[self._sample_axis] = concat_dim
        y_sample_shape = tuple(y_sample_shape)
        y_sig = first_x_sig.as_shape(y_sample_shape)
        return ConcatLayer(x_sigs, y_sig, self._sample_axis)


Concat = keywordize(ConcatSpec)
