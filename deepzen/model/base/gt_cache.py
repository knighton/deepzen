from collections import defaultdict


class GTCache(object):
    def __init__(self):
        self._next_seq_id = 1
        self._node2seqid = {}
        self._seqid2nodes = defaultdict(list)
        self._prev_right = None

    def connect(self, left, right):
        from .pseudo_node import PseudoNode
        from ..sequence import Sequence

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
