from ...model.base.model import Model
from ..base.node import Node
from ..base.pseudo_node import PseudoNode


class Sequence(Model, Node):
    @classmethod
    def _unpack_steps(cls, steps):
        assert isinstance(steps, tuple)
        rets = []
        for i, step in enumerate(steps):
            assert isinstance(step, PseudoNode)
            ret = step.desugar()
            if i:
                assert not ret._preds
            assert not ret._succs
            rets.append(ret)
        return rets

    def __init__(self, *steps, preds_via_call=None):
        steps = self._unpack_steps(steps)
        Model.__init__(self)
        Node.__init__(self, preds_via_call)
        self._steps = steps

    def __call__(self, *preds):
        assert False  # TODO

    def sub_build(self, x_sigs=None):
        for step in self._steps:
            x_sigs = step.sub_build(x_sigs)
        return x_sigs

    def build(self):
        self.sub_build()

    def sub_params(self, nodes_seen, params_seen, params):
        for step in self._steps:
            step.sub_params(nodes_seen, params_seen, params)

    def params(self):
        nodes_seen = set()
        params_seen = set()
        params = []
        self.sub_params(nodes_seen, params_seen, params)
        return params

    def sub_forward(self, xx, is_training):
        for step in self._steps:
            xx = step.sub_forward(xx, is_training)
        return xx

    def forward(self, xx, is_training):
        return self.sub_forward(xx, is_training)
