import numpy as np


class Split(object):
    def __init__(self, samples_per_epoch, x_sample_shapes, x_dtypes,
                 y_sample_shapes, y_dtypes):
        self.samples_per_epoch = samples_per_epoch

        self.sample_shapes = x_sample_shapes, y_sample_shapes
        self.x_sample_shapes = x_sample_shapes
        self.y_sample_shapes = y_sample_shapes

        self.dtypes = x_dtypes, y_dtypes
        self.x_dtypes = x_dtypes
        self.y_dtypes = y_dtypes

    def batches_per_epoch(self, batch_size):
        return self.samples_per_epoch // batch_size

    def x_batch_shapes(self, batch_size):
        return [(batch_size,) + x for x in self.x_sample_shapes]

    def y_batch_shapes(self, batch_size):
        return [(batch_size,) + y for y in self.y_sample_shapes]

    def batch_shapes(self, batch_size):
        x = self.x_batch_shapes(batch_size),
        y = self.y_batch_shapes(batch_size)
        return x, y

    def get_batch(self, batch_size, index):
        raise NotImplementedError

    def shuffle(self, batch_size):
        batches_per_epoch = self.batches_per_epoch(batch_size)
        x = np.arange(batches_per_epoch)
        np.random.shuffle(x)
        return x
