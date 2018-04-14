from copy import deepcopy

from ...model.base.model import Model
from ..base.node import Node
from ..base.pseudo_node import PseudoNode


class Sequence(Model, Node):
    @classmethod
    def _unpack_steps(cls, steps):
        assert steps
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
        Model.__init__(self)
        Node.__init__(self, None)
        steps = self._unpack_steps(steps)
        head = steps[0]
        if preds_via_call:
            # If this node was created via __call__ with a list of predecessor
            # nodes, the head node of the sequence must unconnected.
            assert not head._preds
            for pred in preds_via_call:
                assert isinstance(pred, Node)
                self._connect(pred, self)
        else:
            if head._preds:
                # Given a sequence where there are other nodes that are
                # connected forward to the head node, rewire the connections so
                # that they point us (the containing sequence) instead.
                assert 2 <= len(steps)
                for pred, index in zip(head._preds, head._pred_succ_indices):
                    pred._succs[index] = self
                    self._preds.append(pred)
                    self._pred_succ_indices.append(index)
                head._preds = []
                head._pred_succ_indices = []
            else:
                # The simple case: the head node lacks predecessors.
                pass
        self._steps = steps

    def __call__(self, *preds):
        assert preds
        nodes = deepcopy(self._steps)
        return Sequence(*nodes, preds_via_call=preds)

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
