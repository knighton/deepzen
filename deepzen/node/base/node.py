from .pseudo_node import PseudoNode
from .signature import Signature


class Node(PseudoNode):
    """
    A vertex of a static computational graph.

    Three types:
    * Atom (a node that contains a single indivisible transformation)
    * Network (a graph that connects inputs to outputs; is also a Model)
    * Sequence (contains a list of steps executed in order; is also a Model)

    Situates an encapsulated piece of work within a neural network.  Complex
    nodes (Network and Sequence) are recursively composed of other nodes.

    There is only one restriction on what kind of Nodes must go where inside
    Networks and Sequences: the feed nodes of the top-level, end-to-end network
    must all be Inputs (at the bottom of that recursivity).  Input nodes are a
    are a special type of Node that requires an exact tensor shape/dtype.  This
    is used for shape inference so the entire network can be constructed.
    """

    @classmethod
    def _unpack_preds_via_call(cls, x):
        """
        Unpack predecessor nodes connected to us via __call__.
        """
        if x is None:
            xx = []
        else:
            assert isinstance(x, tuple)
            xx = x
            for x in xx:
                assert isinstance(x, Node)
        return xx

    @classmethod
    def _connect(cls, pred, succ):
        """
        Connect one node forward to another node.
        """
        assert isinstance(pred, Node)
        assert isinstance(succ, Node)
        assert pred is not succ

        # Connect forward.
        pred_succ_index = len(pred._succs)
        pred._succs.append(succ)

        # Connect backward.
        succ._preds.append(pred)
        succ._pred_succ_indices.append(pred_succ_index)

    def __init__(self, preds_via_call=None):
        """
        Set up accounting for predecessors, results, and successors.
        """
        preds_via_call = self._unpack_preds_via_call(preds_via_call)

        # Information about inputs:
        # * Connections between predecessors and ourself.
        # * Accounting for propagating build and forward.
        self._preds = []
        self._pred_succ_indices = []
        self._preds_ready_to_build = 0
        self._preds_ready_to_forward = 0

        # Accounting for the outputs of build.
        self._y_sigs = None

        # Accounting for the outputs of forward.
        self._yy = None

        # The list of successor nodes that we output to.
        self._succs = []

        # Connect ourself with each predecessor node.
        for pred in preds_via_call:
            self._connect(pred, self)

    def desugar(self):
        """
        Inherited from PseudoNode.
        """
        return self

    def sub_build(self, x_sigs):
        """
        Build just this node.
        """
        raise NotImplementedError

    def _init_y_sigs(self, y_sigs):
        """
        Initialize the output signatures from sub_build() (only once).
        """
        assert y_sigs
        assert isinstance(y_sigs, list)
        for y_sig in y_sigs:
            assert isinstance(y_sig, Signature)
        assert self._y_sigs is None
        self._y_sigs = y_sigs

    def propagate_build(self):
        """
        Recursively propagate node building across the network.
        """
        self._preds_ready_to_build += 1
        if self._preds_ready_to_build < len(self._preds):
            return

        x_sigs = []
        for pred in self._preds:
            x_sigs += pred._y_sigs

        y_sigs = self.sub_build(x_sigs)
        self._init_y_sigs(y_sigs)

        for succ in self._succs:
            succ.propagate_build()

        self._preds_ready_to_build = None

    def sub_params(self, nodes_seen, params_seen, params):
        """
        Collect the params of just this node.
        """
        raise NotImplementedError

    def propagate_params(self, nodes_seen, params_seen, params):
        """
        Recursively propagate collecting node params across the network.
        """
        if self in nodes_seen:
            return

        self.sub_params(nodes_seen, params_seen, params)
        nodes_seen.add(self)

        for succ in self._succs:
            succ.propagate_params(nodes_seen, params_seen, params)

    def sub_forward(self, xx, is_training):
        """
        Perform a forward pass across just this node.
        """
        raise NotImplementedError

    def _set_yy(self, yy):
        """
        Save the output tensors from sub_forward().
        """
        assert len(self._y_sigs) == len(yy)
        for y_sig, y in zip(self._y_sigs, yy):
            assert y_sig.accepts_batch_tensor(y)
        self._yy = yy

    def propagate_forward(self, is_training):
        """
        Recursively propagate performing forward passes across the network.
        """
        self._preds_ready_to_forward += 1
        if self._preds_ready_to_forward < len(self._preds):
            return

        xx = []
        for pred in self._preds:
            xx += pred._yy

        yy = self.sub_forward(xx, is_training)
        self._set_yy(yy)

        for succ in self._succs:
            succ.propagate_forward(is_training)

        self._preds_ready_to_forward = 0

    def feed(self, x, is_training):
        """
        Feed an input tensor into the network at this node.
        """
        yy = self.sub_forward([x], is_training)
        self._set_yy(yy)

        for succ in self._succs:
            succ.propagate_forward(is_training)
