import numpy as np

from .split import Split


class RamSplit(Split):
    @classmethod
    def normalize(cls, xx):
        if isinstance(xx, np.ndarray):
            xx = [xx]
        else:
            assert isinstance(xx, (list, tuple))
        return xx

    @classmethod
    def check(cls, xx, yy):
        counts = set()
        for x in xx:
            assert isinstance(x, np.ndarray)
            counts.add(len(x))
        for y in yy:
            assert isinstance(y, np.ndarray)
            counts.add(len(y))
        assert len(counts) == 1
        assert counts.pop()

    def __init__(self, xx, yy):
        xx = self.normalize(xx)
        yy = self.normalize(yy)
        self.check(xx, yy)
        self.xx = xx
        self.yy = yy

    def num_samples(self):
        return len(self.xx[0])

    def sample(self, index):
        xx = [x[index] for x in self.xx]
        yy = [y[index] for y in self.yy]
        return xx, yy
