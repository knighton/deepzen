from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XXYLayer
from ..base.spec import XXYSpec


class StackLayer(XXYLayer):
    def __init__(self, x_sigs, y_sig, sample_axis):
        XXYLayer.__init__(self, x_sigs, y_sig)
        self._sample_axis = sample_axis

    def forward_xx_y(self, xx, is_training):
        return Z.merge_stack(xx, self._sample_axis)


class StackSpec(XXYSpec):
    def __init__(self, sample_axis=0, xsnd=None):
        assert isinstance(sample_axis, int)
        assert 0 <= sample_axis
        XXYSpec.__init__(self, xsnd)
        self._sample_axis = sample_axis

    def build_xx_y(self, x_sigs=None):
        assert x_sigs
        first_x_sig = x_sigs[0]
        for x_sig in x_sigs[1:]:
            assert x_sig == first_x_sig
        x_sample_shape = first_x_sig.sample_shape()
        assert 0 <= self._sample_axis <= len(x_sample_shape)
        before = x_sample_shape[:self._sample_axis]
        concat_dim = len(x_sigs)
        after = x_sample_shape[self._sample_axis:]
        y_sample_shape = before + (concat_dim,) + after
        y_sig = first_x_sig.as_shape(y_sample_shape)
        return StackLayer(x_sigs, y_sig, self._sample_axis)


Stack = keywordize(StackSpec)
