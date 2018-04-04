from copy import deepcopy

from ...layer.base.spec import Spec
from .node import Node


class Atom(Node):
    def __init__(self, spec, preds_via_call=None):
        assert isinstance(spec, Spec)
        Node.__init__(self, preds_via_call)
        self._spec = spec
        self._layer = None

    def __call__(self, *preds):
        assert preds
        spec = deepcopy(self._spec)
        return Atom(spec, preds)

    def sub_build(self, x_sigs):
        self._layer = self._spec.build(x_sigs)
        return self._layer.y_sigs()

    def sub_params(self, nodes_seen, params_seen, params):
        if self in nodes_seen:
            return
        for param in self._layer.params():
            if param in params_seen:
                continue
            params_seen.add(param)
            params.append(param)

    def sub_forward(self, xx, is_training):
        return self._layer.forward(xx, is_training)
