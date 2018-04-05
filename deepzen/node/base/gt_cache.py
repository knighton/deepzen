from collections import defaultdict


class GTCache(object):
    """
    The cache that is used to implement creating node sequences with ">".

    It works by assigning and propagating forward a sequence ID for each node in
    the ">" chains, and packaging the observed node chains into sequence nodes
    that are returned to the caller.

    Behind the scenes, we create a global singleton of this class, which is
    called from the node base class's __gt__ operator.  So don't use ">"
    syntactic sugar to define networks in different threads at the same time.

    Note: you could probably do this accounting work in the locals and look up
    the call stack for this information dynamically, obviating the need for this
    cache, if that were ever an issue for anyone.
    """

    def __init__(self):
        """
        Create a ">" node chain cache.
        """
        self._next_seq_id = 1
        self._node2seqid = {}
        self._seqid2nodes = defaultdict(list)
        self._prev_right = None

    def connect(self, left, right):
        """
        Evaluate one PseudoNode ">" comparison, returning a new Sequence.

        The returned Sequence will be immediately thrown away unless this is the
        last comparison of the ">" chain.

        This is only called from PseudoNode.__gt__.
        """
        from ..arch.sequence import Sequence
        from .pseudo_node import PseudoNode

        assert isinstance(left, PseudoNode)
        assert isinstance(right, PseudoNode)

        # Save the new previous right node.
        self._prev_right = right

        # Either retrieve or invent the sequence ID of the node on the left.
        seq_id = self._node2seqid.get(left)
        if seq_id is None:
            seq_id = self._next_seq_id
            self._next_seq_id += 1
            self._node2seqid[left] = seq_id
            self._seqid2nodes[seq_id].append(left)

        # Propagate the left node's sequence ID forward to the right node.
        self._node2seqid[right] = seq_id
        self._seqid2nodes[seq_id].append(right)

        # Return a Sequence of the nodes of that sequence ID.
        nodes = self._seqid2nodes[seq_id]
        return Sequence(*nodes)
