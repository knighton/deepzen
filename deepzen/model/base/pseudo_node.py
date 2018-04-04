from .gt_cache import GTCache


GT_CACHE = GTCache()


class PseudoNode(object):
    def __gt__(self, right):
        return GT_CACHE.connect(self, right)

    def __mul__(self, count):
        from ..sequence import Sequence
        assert isinstance(count, int)
        assert 1 <= count
        steps = [self.copy() for i in range(count)]
        return Sequence(*steps)

    def copy(self):
        assert False  # TODO

    def desugar(self):
        raise NotImplementedError
