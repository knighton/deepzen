from .gt_cache import GTCache


GT_CACHE = GTCache()


class PseudoNode(object):
    """
    A node-like building block for constructing computational graphs.

    Two types:
    * Keyword (fake node class names that return real nodes when called)
    * Node (a static computational graph vertex)

    Static computational graphs are exclusively made of Nodes.  PseudoNodes are
    handles that act like Node class names, but may not be -- for * and >
    sequence construction syntactic sugar to work on these Node "class names",
    they have to be objects.  That is why PseudoNode and Keyword classes exist.
    """

    def __gt__(self, right):
        """
        Construct a Sequence connecting this node to its rightward neighbor.
        """
        return GT_CACHE.connect(self, right)

    def __mul__(self, count):
        """
        Construct a Sequence repeating this node n times.
        """
        from ..arch.sequence import Sequence
        assert isinstance(count, int)
        assert 1 <= count
        steps = [self.copy() for i in range(count)]
        return Sequence(*steps)

    def copy(self):
        """
        Get a deepcopy.
        """
        assert False  # TODO

    def desugar(self):
        """
        Evaluate any Keywords to their Nodes.
        """
        raise NotImplementedError
