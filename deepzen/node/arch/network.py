from ...model.base.model import Model
from ..base.node import Node
from ..base.signature import Signature


class Network(Model, Node):
    @classmethod
    def _unpack_inputs(cls, x):
        if isinstance(x, Node):
            xx = [x]
        else:
            assert isinstance(x, (list, tuple))
            xx = x
            for x in xx:
                assert isinstance(x, Node)
        return xx

    @classmethod
    def _unpack_outputs(cls, x):
        if isinstance(x, Node):
            xx = [x]
        else:
            assert isinstance(x, (list, tuple))
            xx = x
            for x in xx:
                assert isinstance(x, Node)
        return xx

    def __init__(self, inputs, outputs, preds_via_call=None):
        inputs = self._unpack_inputs(inputs)
        outputs = self._unpack_outputs(outputs)
        Model.__init__(self)
        Node.__init__(self, preds_via_call)
        self._inputs = inputs
        self._outputs = outputs

    def __call__(self, *preds):
        assert False  # TODO

    @classmethod
    def _assign_signatures_to_nodes(cls, x_sigs, num_inputs):
        if x_sigs is None:
            x_sigs_per_input = [None] * num_inputs
        else:
            assert isinstance(x_sigs, list)
            for x_sig in x_sigs:
                assert isinstance(x_sig, Signature)
            if len(x_sigs) == 1:
                x_sigs_per_input = [x_sigs] * num_inputs
            elif num_inputs == 1:
                x_sigs_per_input = [x_sigs]
            else:
                assert num_inputs == len(x_sigs)
                x_sigs_per_input = [[x_sig] for x_sig in x_sigs]
        return x_sigs_per_input

    def sub_build(self, x_sigs=None):
        num_inputs = len(self._inputs)
        x_sigs_per_input = self._assign_signatures_to_nodes(x_sigs, num_inputs)
        for input, x_sigs in zip(self._inputs, x_sigs_per_input):
            input.propagate_build(x_sigs)
        y_sigs = []
        for output in self._outputs:
            assert output._y_sigs
            y_sigs += output._y_sigs
        return y_sigs

    def build(self):
        self.sub_build()

    def sub_params(self, nodes_seen, params_seen, params):
        for input in self._inputs:
            input.propagate_params(nodes_seen, params_seen, params)

    def params(self):
        nodes_seen = set()
        params_seen = set()
        params = []
        self.sub_params(nodes_seen, params_seen, params)
        return params

    @classmethod
    def _assign_tensors_to_nodes(cls, xx, num_inputs):
        assert isinstance(xx, list)
        print('?', type(xx[0]))  # TODO: check type.
        if len(xx) == 1:
            xxx = [xx for i in range(num_inputs)]
        elif num_inputs == 1:
            xxx = [xx]
        else:
            assert num_inputs == len(xx)
            xxx = [[x] for x in xx]
        return xxx

    def sub_forward(self, xx, is_training):
        xxx = self._assign_tensors_to_nodes(xx, len(self._inputs))
        for input, xx in zip(self._inputs, xxx):
            input.propagate_forward(xx, is_training)
        y_sigs = []
        for output in self._outputs:
            assert output._y_sigs
            y_sigs += output._y_sigs
        return y_sigs

    def forward(self, xx, is_training):
        return self.sub_forward(xx, is_training)
