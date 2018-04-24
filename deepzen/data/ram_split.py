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
        samples_per_epoch = len(xx[0])
        x_sample_shapes = [x[0].shape for x in xx]
        x_dtypes = [x[0].dtype.name for x in xx]
        y_sample_shapes = [y[0].shape for y in yy]
        y_dtypes = [y[0].dtype.name for y in yy]
        Split.__init__(self, samples_per_epoch, x_sample_shapes, x_dtypes,
                       y_sample_shapes, y_dtypes)
        self.xx = xx
        self.yy = yy

    def get_batch(self, batch_size, index):
        a = index * batch_size
        z = (index + 1) * batch_size
        batch_xx = [x[a:z] for x in self.xx]
        batch_yy = [y[a:z] for y in self.yy]
        return batch_xx, batch_yy
