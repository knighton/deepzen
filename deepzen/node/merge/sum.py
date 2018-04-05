from ... import api as Z
from ..base.keyword import keywordize
from ..base.layer import XXYLayer
from ..base.spec import XXYSpec


class SumLayer(XXYLayer):
    def __init__(self, x_sigs, y_sig):
        XXYLayer.__init__(self, x_sigs, y_sig)

    def forward_xx_y(self, xx, is_training):
        return Z.merge_sum(xx)


class SumSpec(XXYSpec):
    def __init__(self, xsnd=None):
        XXYSpec.__init__(self, xsnd)

    def build_xx_y(self, x_sigs=None):
        assert x_sigs
        first_x_sig = x_sigs[0]
        for x_sig in x_sigs[1:]:
            assert x_sig == first_x_sig
        y_sig = first_x_sig
        return SumLayer(x_sigs, y_sig)


Sum = keywordize(SumSpec)
